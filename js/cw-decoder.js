// CW (Morse) decoder: pure logic, no DOM and no audio API, so it runs in Node tests and in any browser.
//
// Pipeline (all driven by CwDecoder.process(samples, sampleRate)):
//  1. Pitch finding: a bank of Goertzel filters (300-1100 Hz, 20 Hz apart) over 40 ms windows; the strongest steady
//     tone that clearly stands out of the rest wins (parabolic interpolation between the bins). Or the pitch is
//     locked to a given frequency (the CW pitch of the radio, see setPitch).
//  2. Tone detection: the audio is mixed down at that pitch (I/Q) and summed per 5 ms block; the envelope is the
//     coherent sum of the last few blocks, about half a dot long (10-50 ms). That is a filter matched to the speed:
//     about 25 Hz wide at 20 WPM instead of 200 Hz, so far less noise gets in (Richard, 4 Nov 2026: weak stations).
//     The phase turning from block to block during a mark tells how far the tone is off: the filter follows it
//     (fine tuning, AFC), also when the pitch is locked (at most FINE_MAX Hz from the lock). The noise level is the 25th percentile of the quiet blocks of the last 2 s, the
//     signal peak follows the marks. The threshold sits between them with hysteresis but never below a few times
//     the noise level, so plain noise hardly ever gives a mark (measured on real band noise of an FT-710). A state change must last a few
//     blocks (debounce, longer for slower speeds) and is back-dated to its first block, so the timing stays exact.
//  3. Timing: the dot length follows the marks (2-means on the log durations of the last marks: dots versus
//     dashes), giving WPM 5-50. Gaps: < 2 dots inside a letter, >= 2 dots letter gap, >= 5 dots word gap.
//  4. Symbols to text: ITU table incl. digits, punctuation and prosigns. Unknown patterns give '*', but the raw
//     pattern is kept: every character is {ch, pattern, confidence} (confidence 0..1, low = a dot/dash length
//     that was hard to tell apart).
//  5. Morse or noise: characters are held back until the signal proves to be Morse (enough elements, marks well
//     above the noise), which normally takes one or two characters. Then the held characters and everything after
//     come out at once; after a few seconds without marks the gate closes again.
//  6. Weak signals (Richard, 5 Nov 2026): steps 2-5 decide block by block and lose a station around 0 dB in 500 Hz,
//     where a trained ear still copies. So the same I/Q blocks also go to a Viterbi decoder (cw-viterbi.js) that
//     finds the most likely text for the whole stretch heard. Its text is the decoder's text (.text, onChar), about
//     a second behind the audio; the fast text of steps 2-5 is shown meanwhile as a guess (onPending), and replaced.
//     On test signals: 0 dB 99% right (fast decoder 0%), -3 dB 82%, -5 dB about half. {viterbi: false} = fast only.

// ---------- Morse tables (cw-decoder-tables.js) ----------

import { MORSE, MORSE_PROSIGNS, decodePattern } from './cw-decoder-tables.js';
const PROSIGNS = MORSE_PROSIGNS;
import { MorseViterbi } from './cw-viterbi.js';
import { buildVocabulary, correctWord, heardReport } from './cw-context.js';

export { MORSE, MORSE_PROSIGNS, decodePattern };

// Text to marks and spaces: [{on, dur}] in seconds at the given speed (PARIS: dot = 1.2 / wpm). Prosigns are
// written as <AR>. Used by the tests and available for a later send function.
export function textToElements(text, wpm = 20) {
  const dot = 1.2 / wpm;
  const out = [];
  const words = String(text).toUpperCase().split(/\s+/).filter(Boolean);
  words.forEach((word, wi) => {
    if (wi) out.push({ on: false, dur: 7 * dot });
    const tokens = word.match(/<[A-Z]+>|./g) || [];
    let first = true;
    for (const tok of tokens) {
      const p = PROSIGNS[tok] || MORSE[tok];
      if (!p) continue;
      if (!first) out.push({ on: false, dur: 3 * dot });
      first = false;
      [...p].forEach((el, i) => {
        if (i) out.push({ on: false, dur: dot });
        out.push({ on: true, dur: (el === '-' ? 3 : 1) * dot });
      });
    }
  });
  return out;
}

// ---------- decoder ----------

const MIN_DOT = 0.022; // 50 WPM
const MAX_DOT = 0.26; // 4.6 WPM
const PRIOR_DOT = 0.08; // 15 WPM, until the first marks have been heard
const SCAN_STEP = 20; // Hz between the pitch filters
const SCAN_WIN = 0.04; // s per pitch window
const PEAK_RATIO = 16; // strongest bin / median bin needed for a pitch candidate
const TONE_WIN = 38; // windows (about 1.5 s) for the tuning indicator
const TONE_NEAR = 25; // Hz: peaks this close count as the same tone
const TONE_MIN_HITS = 6; // at least this many windows ...
const TONE_MIN_SHARE = 0.25; // ... and this share of the last TONE_WIN windows
const EDGE_BIAS = -0.3; // blocks, see #transition
const HISTORY = 16; // marks kept for the dot estimate
const MAX_CHARS = 600; // decoded characters kept in .chars / .text
// Levels below are in units of the noise level q (25th percentile of the quiet envelope). On band noise the
// 15 ms envelope has its median at 1.35 q, 99.9% below 3.7 q and its maximum over 30 s at 4.2-5.2 q.
const SMOOTH = 3; // blocks in the envelope at least
const MAX_WIN = 16; // ... and at most (80 ms)
const WIN_DOTS = 0.7; // envelope window in dots once the speed is known (measured best: 3 dB 0% -> 80% right)
const DEBOUNCE_DOTS = 0.15; // a change must last this many dots
const START_WIN = 4; // blocks while the speed is not known
const FINE_MAX = 60; // Hz: fine tuning around a locked pitch
// Two side filters SIDE_HZ below and above the pitch: a mark only counts when the tone at the pitch is not much
// weaker than there, so a strong station further away (leaking into the pitch filter) is not decoded.
const SIDE_HZ = 250;
const SIDE_RATIO = 1.5; // pitch power / strongest side power needed (a click or broadband noise is about 1)
const NOISE_MEDIAN = 1.35;
const NOISE_BLOCKS = 400; // quiet blocks (2 s) kept for the noise level
const MIN_NOISE = 2e-5; // noise level floor (digital silence)
const ON_Q = 3.6; // a mark starts above this ...
const OFF_Q = 2.5; // ... and ends below this
const VALID_Q = 4.5; // peak needed before any mark is reported
const GATE_STRONG = 5; // mark level that opens the gate after 3 elements
const GATE_WEAK = 3; // mark level that opens it after 2 known characters and 4 elements
const GLITCH = 0.4; // a weak mark shorter than this many dots is a noise spike: left out (also from the speed)
const SURE_Q = 8; // a mark this far above the noise is never a spike
const WEAK_MIN = 0.03; // s: a weak mark (below GATE_STRONG) shorter than this (faster than 40 WPM) is a spike too
const MIN_MARK = 0.025; // s: shorter marks are blips (see #transition) unless the station really is that fast
const LOCK_SAFE_DOT = 0.04; // s (30 WPM): below this the long marks still feed the speed estimate
const HOLD_REL = 0.5; // a mark renews the gate hold only above this share of the recent mark level
const PENDING_HOLD = 1.2; // s of quiet after which held (not yet confirmed as Morse) characters are dropped
const GATE_HOLD = 3; // s without marks before the gate closes (at least 12 dots)

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const median = (a) => {
  const s = a.slice().sort((x, y) => x - y);
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
};

// Splits the mark durations into dots and dashes. Returns {dot, certain}: the dot length and whether two clearly
// separate groups were found.
function estimateDot(marks, prevDot) {
  const n = marks.length;
  const a = marks.slice().sort((x, y) => x - y);
  let best = -1;
  let bestSse = Infinity;
  const logs = a.map(Math.log);
  for (let k = 1; k < n; k++) {
    let sse = 0;
    for (const [from, to] of [[0, k], [k, n]]) {
      let m = 0;
      for (let i = from; i < to; i++) m += logs[i];
      m /= to - from;
      for (let i = from; i < to; i++) sse += (logs[i] - m) ** 2;
    }
    if (sse < bestSse) {
      bestSse = sse;
      best = k;
    }
  }
  if (best > 0) {
    const lo = a.slice(0, best);
    const hi = a.slice(best);
    const m1 = median(lo);
    const m2 = median(hi);
    const ratio = m2 / m1;
    if (ratio >= 1.9) {
      if (ratio > 4.5) {
        // One group is junk. Which one? (Richard, 13 Nov 2026: 52 WPM and only T/M/O for minutes.) Short strong
        // blips from AGC FAST or a ringing APF / narrow filter make the LOWER group the junk: taking it as the dot gave
        // a 22 ms dot, every real dot read as a dash, and it locked there. Lower group much shorter than the previous
        // dot, or the upper group splits into dots and dashes itself: the real marks are the upper group.
        const ref = prevDot || PRIOR_DOT;
        if (hi.length >= 3 && (m1 < 0.5 * ref || estimateDot(hi, null).certain)) return estimateDot(hi, prevDot);
        return { dot: m1, certain: true }; // the upper group is probably junk (a stuck key, QRN)
      }
      return { dot: (m1 * lo.length + (m2 / 3) * hi.length) / n, certain: true };
    }
  }
  // One group: dots or dashes? Decide with the previous estimate.
  const c = median(a);
  const ref = prevDot || PRIOR_DOT;
  return { dot: c < 2 * ref ? c : c / 3, certain: false };
}

const FAST_KEEP_CONF = 0.6; // fast characters at least this sure are kept when the Viterbi decoder stays silent

export class CwDecoder {
  // opts: {sampleRate, onChar(charObj), onPending(text) (the fast guess not yet confirmed), onStatus(status) (about 10
  //        per second of audio), onTone(on), viterbi (default true),
  //        pitchHz (lock), symbols (show + and = instead of <AR> and <BT>), minPitch, maxPitch}
  constructor(opts = {}) {
    this.onChar = opts.onChar || null;
    this.onStatus = opts.onStatus || null;
    this.onTone = opts.onTone || null;
    this.onPending = opts.onPending || null;
    // the fast decoder's own output, for the CW trainer (cw-trainer.js): every element {on, dur (s)} and character
    this.onElement = opts.onElement || null;
    this.onFastChar = opts.onFastChar || null;
    this.useViterbi = opts.viterbi !== false;
    this.vit = this.useViterbi ? new MorseViterbi({ symbols: Boolean(opts.symbols) }) : null;
    this.live = []; // fast characters not yet confirmed: {c, at (Viterbi block)}
    this.fastUntil = -Infinity; // Viterbi block up to which fast characters were kept as text (#keepFast)
    this.wordBuf = []; // Viterbi letters of the word being completed (for the context check)
    this.vocabFn = opts.vocabulary || null;
    this.onReport = opts.onReport || null;
    this.lastReport = null;
    this.pendingText = '';
    this.symbols = Boolean(opts.symbols);
    this.minPitch = opts.minPitch || 300;
    this.maxPitch = opts.maxPitch || 1100;
    this.lock = opts.pitchHz ? Number(opts.pitchHz) : null;
    this.sr = 0;
    this.reset();
    if (opts.sampleRate) this.#init(opts.sampleRate);
  }

  // Forget everything heard (text, speed, noise floor, pitch); the lock stays.
  reset() {
    this.text = '';
    this.chars = [];
    this.#vitReset();
    this.#resetSignal();
    if (this.sr) this.#init(this.sr);
  }

  // Forget the signal state (noise level, peak, held characters, speed) but keep the text: for a new audio stream,
  // whose first blocks are not the same signal.
  resync() {
    this.#vitReset();
    this.#resetSignal();
    if (this.sr) this.#init(this.sr);
  }

  // Locks the pitch to hz (e.g. the CW pitch of the radio); null = find the pitch automatically.
  setPitch(hz) {
    this.lock = hz ? clamp(Number(hz), 200, 1500) : null;
    if (this.lock) this.#tune(this.lock);
  }

  get locked() {
    return this.lock !== null;
  }

  // Current state for a display.
  get status() {
    const noise = this.noise || 0;
    const peak = this.peak || 0;
    return {
      // the speed: the fast decoder's on a clear signal, else the Viterbi decoder's while it reads one
      wpm: !this.gate && this.vit && this.vit.done <= this.vit.sureUntil ? this.vit.wpm : this.dot ? Math.round(1.2 / this.dot) : null,
      pitch: this.tuneHz ? Math.round(this.tuneHz) : null,
      level: peak > 1e-6 ? clamp(this.env / peak, 0, 1) : 0,
      snrDb: noise > 1e-9 && peak > 0 ? Math.max(0, 20 * Math.log10(peak / (NOISE_MEDIAN * noise))) : 0,
      toneOn: this.on,
      // Morse heard: by the fast decoder, or the Viterbi decoder is reading a weak station
      morse: this.gate || Boolean(this.vit && this.vit.done <= this.vit.sureUntil),
      // the strongest steady tone of the last 1.5 s (Hz), for a tuning indicator; null when none stands out
      toneHz: this.toneHz !== null && (this.blk - this.toneBlk) * (this.blockDur || 0.005) < 1.5 ? Math.round(this.toneHz) : null,
      locked: this.lock !== null,
    };
  }

  // Feeds a block of mono samples (any length). sampleRate may be given on every call (a change re-initialises).
  process(samples, sampleRate) {
    if (sampleRate && sampleRate !== this.sr) this.#init(sampleRate);
    if (!this.sr) throw new Error('CwDecoder: sampleRate ontbreekt');
    for (let i = 0; i < samples.length; i++) this.#sample(samples[i]);
  }

  // ----- setup -----

  #resetSignal() {
    this.tuneHz = this.lock;
    this.noise = null; // noise level q
    this.pows = new Float64Array(MAX_WIN);
    this.sidePows = new Float64Array(MAX_WIN);
    this.zRe = new Float64Array(MAX_WIN); // I/Q sum of each of the last blocks
    this.zIm = new Float64Array(MAX_WIN);
    this.powI = 0;
    this.fine = 0; // Hz: fine tuning (AFC) on top of tuneHz
    this.afcRe = 0; // phase turn per block summed over the mark now on
    this.afcIm = 0;
    this.warm = [];
    this.quiet = new Float64Array(NOISE_BLOCKS);
    this.quietN = 0;
    this.quietW = 0;
    this.quietIn = 0;
    this.markSum = 0; // envelope sum and block count of the mark now on
    this.markBlocks = 0;
    this.curLevel = []; // mark levels (in q) of the character being received
    this.marksSinceQuiet = 0; // P4
    this.gate = false; // true once the signal proved to be Morse
    this.pending = []; // characters held back while the gate is closed
    this.lastMarkBlk = -1e9;
    this.markAvg = 0; // recent mark level (in q) of the station being read
    this.peak = 0;
    this.env = 0;
    this.valid = false;
    this.on = false;
    this.cand = 0;
    this.candStart = 0;
    this.blk = 0; // block counter
    this.runStart = 0; // block where the current run (mark or space) began
    this.stage = 0; // gap handling in the current space: 0 nothing, 1 letter flushed, 2 word space sent
    this.dot = null;
    this.history = [];
    this.cur = []; // mark durations (s) of the character being received
    this.statusIn = 0;
    this.pitchCand = null;
    this.pitchCandN = 0;
    this.toneHz = null;
    this.toneBlk = 0;
    this.peaks = []; // the pitch-bank peak of the last TONE_WIN windows (Hz, or null: nothing stood out)
  }

  #init(sr) {
    this.sr = sr;
    this.blockN = Math.max(8, Math.round((sr * 5) / 1000));
    this.blockDur = this.blockN / sr;
    this.bi = 0;
    this.oscRe = 1; // the mixer: e^(-j w n), turned one step per sample
    this.oscIm = 0;
    this.accRe = 0;
    this.accIm = 0;
    this.vOscRe = 1;
    this.vOscIm = 0;
    this.vAccRe = 0;
    this.vAccIm = 0;
    this.vHz = null; // a new sample rate: the Viterbi mixer is set up again
    this.side = new Float64Array(4); // s1, s2 of the two side filters
    this.sideCoef = [0, 0];
    // pitch scan bank
    this.scanN = Math.round(sr * SCAN_WIN);
    this.scanFreqs = [];
    for (let f = 300; f <= 1100; f += SCAN_STEP) if (f >= this.minPitch - 1e-9 && f <= this.maxPitch + 1e-9 && f < sr / 2.2) this.scanFreqs.push(f);
    const nb = this.scanFreqs.length;
    this.scanCoef = Float64Array.from(this.scanFreqs, (f) => 2 * Math.cos((2 * Math.PI * f) / sr));
    this.scanS1 = new Float64Array(nb);
    this.scanS2 = new Float64Array(nb);
    this.scanI = 0;
    // audio kept while no pitch is known, replayed when one is found
    this.preN = Math.round(sr * 0.2);
    this.pre = new Float32Array(this.preN);
    this.preW = 0;
    this.preFill = 0;
    this.#resetSignal();
    if (this.lock) this.#tune(this.lock);
  }

  #tune(hz) {
    this.tuneHz = hz;
    this.fine = 0;
    // the Viterbi mixer only moves when the pitch has gone beyond the offsets it tries itself (a mixer that follows
    // every small step of the pitch finder turns the tone's phase within a dash)
    // (and on its average: the pitch finder jitters on a weak signal)
    // (no mixer before the sample rate is known: the pitch can be locked before the audio starts)
    if (this.sr) {
      this.vAvg = this.vHz == null || this.vAvg == null ? hz : this.vAvg + 0.1 * (hz - this.vAvg);
      if (this.vHz == null || Math.abs(this.vAvg - this.vHz) > 10) this.#vMixer(this.vAvg);
    }
    if (!this.vOscRe && !this.vOscIm) {
      this.vOscRe = 1;
      this.vOscIm = 0;
    }
    this.#detector();
  }

  // The mixer and side filters at the detection frequency: the pitch plus the fine tuning.
  #detector() {
    const hz = this.tuneHz + (this.fine || 0);
    this.detHz = hz;
    const w = (2 * Math.PI * hz) / this.sr;
    this.stepRe = Math.cos(w);
    this.stepIm = -Math.sin(w);
    this.sideCoef = [hz - SIDE_HZ, hz + SIDE_HZ].map((f) => 2 * Math.cos((2 * Math.PI * Math.max(50, f)) / this.sr));
  }

  // Blocks in the envelope: about half a dot (a filter matched to the speed).
  get #win() {
    // speed not known yet: short (20 ms), so even 40 WPM keeps its spaces; the window grows once the speed is known
    if (!this.dot || this.history.length < 6) return START_WIN;
    return clamp(Math.round((WIN_DOTS * this.dot) / this.blockDur), SMOOTH, MAX_WIN);
  }

  // ----- per sample -----

  #sample(x) {
    this.#scanSample(x); // always: it also measures the tone for the tuning indicator
    if (this.lock === null) {
      if (this.tuneHz === null) {
        this.pre[this.preW] = x;
        this.preW = (this.preW + 1) % this.preN;
        if (this.preFill < this.preN) this.preFill++;
      }
    }
    if (this.tuneHz !== null) this.#toneSample(x);
  }

  #toneSample(x) {
    this.accRe += x * this.oscRe;
    this.accIm += x * this.oscIm;
    // the Viterbi decoder has a mixer of its own at the pitch without the fine tuning: the fine tuning follows every
    // mark, also a noise burst, and a tone that wanders would not add up; it tries small offsets itself
    this.vAccRe += x * this.vOscRe;
    this.vAccIm += x * this.vOscIm;
    const vr = this.vOscRe * this.vStepRe - this.vOscIm * this.vStepIm;
    this.vOscIm = this.vOscRe * this.vStepIm + this.vOscIm * this.vStepRe;
    this.vOscRe = vr;
    const r = this.oscRe * this.stepRe - this.oscIm * this.stepIm;
    this.oscIm = this.oscRe * this.stepIm + this.oscIm * this.stepRe;
    this.oscRe = r;
    const sd = this.side;
    const c = this.sideCoef;
    const a0 = x + c[0] * sd[0] - sd[1];
    sd[1] = sd[0];
    sd[0] = a0;
    const b0 = x + c[1] * sd[2] - sd[3];
    sd[3] = sd[2];
    sd[2] = b0;
    if (++this.bi >= this.blockN) {
      const g = 1 / Math.hypot(this.oscRe, this.oscIm); // keep the mixer at length 1
      this.oscRe *= g;
      this.oscIm *= g;
      const zr = this.accRe;
      const zi = this.accIm;
      const vg = 1 / Math.hypot(this.vOscRe, this.vOscIm);
      this.vOscRe *= vg;
      this.vOscIm *= vg;
      const vzr = this.vAccRe;
      const vzi = this.vAccIm;
      this.vAccRe = 0;
      this.vAccIm = 0;
      this.accRe = 0;
      this.accIm = 0;
      const p = zr * zr + zi * zi;
      const pa = sd[0] * sd[0] + sd[1] * sd[1] - c[0] * sd[0] * sd[1];
      const pb = sd[2] * sd[2] + sd[3] * sd[3] - c[1] * sd[2] * sd[3];
      if (this.vit && this.vit.blockDurSet !== this.blockDur) {
        this.vit = new MorseViterbi({ blockDur: this.blockDur, symbols: this.symbols });
        this.vit.blockDurSet = this.blockDur;
      }
      if (this.vit) {
        // a carrier (a tone on for more than 1.5 s) is not Morse: blank for the Viterbi decoder too
        const carrier = this.on && (this.blk - this.runStart) * this.blockDur > 1.5;
        this.vit.push(vzr, vzi, carrier);
        this.#vitTake();
      }
      sd.fill(0);
      this.bi = 0;
      // fine tuning: the phase turn since the previous block, while a mark is on
      const prev = (this.powI - 1 + MAX_WIN) % MAX_WIN;
      if (this.on) {
        this.afcRe += zr * this.zRe[prev] + zi * this.zIm[prev];
        this.afcIm += zi * this.zRe[prev] - zr * this.zIm[prev];
      }
      this.pows[this.powI] = p;
      this.sidePows[this.powI] = Math.max(0, pa, pb);
      this.zRe[this.powI] = zr;
      this.zIm[this.powI] = zi;
      this.powI = (this.powI + 1) % MAX_WIN;
      const n = this.#win;
      let sum = 0;
      let side = 0;
      let cr = 0;
      let ci = 0;
      for (let k = 1; k <= n; k++) {
        const j = (this.powI - k + MAX_WIN) % MAX_WIN;
        sum += this.pows[j];
        side += this.sidePows[j];
        cr += this.zRe[j];
        ci += this.zIm[j];
      }
      // the coherent sum: a tone adds up, noise only partly (the matched filter)
      this.#envelope((2 * Math.hypot(cr, ci)) / (n * this.blockN), sum >= SIDE_RATIO * side);
    }
  }

  #scanSample(x) {
    const c = this.scanCoef;
    const s1 = this.scanS1;
    const s2 = this.scanS2;
    for (let j = 0; j < c.length; j++) {
      const s0 = x + c[j] * s1[j] - s2[j];
      s2[j] = s1[j];
      s1[j] = s0;
    }
    if (++this.scanI >= this.scanN) {
      const pw = new Float64Array(c.length);
      for (let j = 0; j < c.length; j++) {
        pw[j] = s1[j] * s1[j] + s2[j] * s2[j] - c[j] * s1[j] * s2[j];
        s1[j] = 0;
        s2[j] = 0;
      }
      this.scanI = 0;
      this.#scanWindow(pw);
    }
  }

  // Tuning indicator (Richard, 12 Nov 2026: "slingert heen en weer", also after a zero-in on the radio). Behind the
  // radio's narrow CW filter the band noise inside the passband easily stands out against the quiet bins outside it,
  // so the strongest bin of one 40 ms window is often noise. A CW station keeps returning at the same frequency, noise
  // does not: toneHz is set only when at least TONE_MIN_SHARE of the last TONE_WIN windows (about 1.5 s) had their
  // peak within TONE_NEAR Hz of each other; the value is their mean, smoothed a little.
  #notePeak(f) {
    this.peaks.push(f);
    if (this.peaks.length > TONE_WIN) this.peaks.shift();
    if (f === null) return;
    const near = this.peaks.filter((p) => p !== null && Math.abs(p - f) <= TONE_NEAR);
    if (near.length < Math.max(TONE_MIN_HITS, TONE_MIN_SHARE * this.peaks.length)) return;
    const mean = near.reduce((a, b) => a + b, 0) / near.length;
    this.toneHz = this.toneHz !== null && Math.abs(mean - this.toneHz) < 40 ? this.toneHz + 0.3 * (mean - this.toneHz) : mean;
    this.toneBlk = this.blk;
  }

  // Locked pitch, the station a little beside it (24 Nov 2026, measured: a tone exactly 50 Hz off gave 81-85% errors,
  // 35 and 60 Hz were fine). The fast decoder sums a few 5 ms blocks coherently; that filter has nulls at 200/n Hz
  // (50 Hz for 4 blocks, 100 Hz for 2). A tone in such a null gives no marks, and the fine tuning (AFC) only moves on
  // marks, so it never started. The pitch bank has no such null: a steady tone within FINE_MAX of the lock sets the
  // fine tuning (and the Viterbi mixer) at once.
  #lockedFollow(f) {
    const t = this.toneHz;
    if (t === null || this.lock === null || !this.sr) return;
    if ((this.blk - this.toneBlk) * (this.blockDur || 0.005) > 1.5) return; // no steady tone lately
    if (Math.abs(f - t) > 15) return; // this window's peak is not that tone
    const off = t - this.lock;
    if (Math.abs(off) > FINE_MAX) return; // farther away: another station, the lock keeps it out
    if (Math.abs(t - (this.detHz || this.lock)) <= 12) return; // the detector is there already
    this.fine = off;
    this.#detector();
    if (this.vHz && Math.abs(this.detHz - this.vHz) > 8) {
      this.#vMixer(this.detHz);
      this.vit.fOff = 0;
    }
  }

  // One 40 ms window of the pitch bank.
  #scanWindow(pw) {
    const n = pw.length;
    if (n < 3) return;
    let m = 0;
    for (let j = 1; j < n; j++) if (pw[j] > pw[m]) m = j;
    const amp = (2 * Math.sqrt(pw[m])) / this.scanN;
    if (amp < 1e-4) return this.#notePeak(null); // silence
    const sorted = Array.from(pw).sort((a, b) => a - b);
    const med = sorted[n >> 1];
    if (pw[m] < PEAK_RATIO * (med + 1e-20)) return this.#notePeak(null); // nothing stands out
    let f = this.scanFreqs[m];
    if (m > 0 && m < n - 1 && pw[m - 1] > 0 && pw[m + 1] > 0) {
      const a = Math.log(pw[m - 1]);
      const b = Math.log(pw[m]);
      const c = Math.log(pw[m + 1]);
      const den = a - 2 * b + c;
      if (den < 0) f += clamp((0.5 * (a - c)) / den, -1, 1) * SCAN_STEP;
    }
    // tuning indicator (and ZIN, AUTO): only a tone that keeps coming back at the same place
    this.#notePeak(f);
    if (this.lock !== null) return this.#lockedFollow(f); // locked: the filter stays near the radio's pitch
    if (this.tuneHz === null) {
      // first pitch: two windows in a row at the same frequency
      if (this.pitchCand !== null && Math.abs(f - this.pitchCand) < 30) this.pitchCandN++;
      else {
        this.pitchCand = f;
        this.pitchCandN = 1;
      }
      if (this.pitchCandN >= 2) this.#acquire(f);
    } else if (Math.abs(f - this.tuneHz) < 35) {
      this.#tune(this.tuneHz + 0.25 * (f - this.tuneHz)); // follow slow drift
      this.pitchCand = null;
    } else if (this.gate && (this.blk - this.lastMarkBlk) * this.blockDur < GATE_HOLD) {
      // reading Morse: a tone elsewhere (a steady carrier, a birdie heard in the spaces) does not take the filter
      // away; only when this station has been quiet for GATE_HOLD seconds
      this.pitchCand = null;
    } else {
      // a stronger tone elsewhere: only after it has stayed for 5 windows
      if (this.pitchCand !== null && Math.abs(f - this.pitchCand) < 30) this.pitchCandN++;
      else {
        this.pitchCand = f;
        this.pitchCandN = 1;
      }
      if (this.pitchCandN >= 5) {
        this.#vitReset(); // another station
        this.#tune(f);
        this.pitchCand = null;
        this.#clearDetector();
      }
    }
  }

  #clearDetector() {
    this.accRe = 0;
    this.accIm = 0;
    this.vAccRe = 0;
    this.vAccIm = 0;
    this.bi = 0;
    this.side.fill(0);
    this.pows.fill(0);
    this.sidePows.fill(0);
    this.zRe.fill(0);
    this.zIm.fill(0);
  }

  #acquire(f) {
    this.#vitReset();
    this.#tune(f);
    this.pitchCand = null;
    // replay the audio kept so far, so the start of the first element is not lost
    const n = this.preFill;
    const start = (this.preW - n + this.preN) % this.preN;
    this.#clearDetector();
    for (let i = 0; i < n; i++) this.#toneSample(this.pre[(start + i) % this.preN]);
    this.preFill = 0;
  }

  // ----- envelope to marks and spaces -----

  #envelope(env, clean = true) {
    const nb = this.blk;
    this.env = env;
    if (this.noise === null) {
      // Warm-up: the first 12 blocks (60 ms) only set the noise floor (their median), no tone is reported.
      this.warm.push(env);
      this.blk++;
      if (this.warm.length >= 12) {
        // The lowest quarter only, in case a mark was already on.
        const s = this.warm.slice().sort((a, b) => a - b);
        this.noise = Math.max(MIN_NOISE, s[3]);
        this.peak = NOISE_MEDIAN * this.noise;
        this.warm = [];
      }
      return;
    }
    // Noise level: from the quiet blocks (no mark on, no change starting). A tone that stays on for more than 1.5 s
    // is a carrier, not Morse: it counts as background too, so it is not decoded.
    const carrier = this.on && (nb - this.runStart) * this.blockDur > 1.5;
    if ((!this.on && this.cand === 0) || carrier) {
      this.quiet[this.quietW] = env;
      this.quietW = (this.quietW + 1) % NOISE_BLOCKS;
      if (this.quietN < NOISE_BLOCKS) this.quietN++;
      if (++this.quietIn >= 10 && this.quietN >= 20) {
        this.quietIn = 0;
        const s = Array.from(this.quiet.subarray(0, this.quietN)).sort((a, b) => a - b);
        this.noise = Math.max(MIN_NOISE, s[Math.floor(0.25 * (s.length - 1))]);
      }
    }
    const q = this.noise;
    const floor = NOISE_MEDIAN * q;
    // Peak: up quickly, down slowly (to the noise).
    // During a mark the peak follows the mean mark level instead (noise on top of a weak tone would push it up and
    // put the threshold so high that every dip cuts the mark).
    if (this.on && this.history.length >= 6) this.peak += 0.05 * (env - this.peak);
    else if (env > this.peak) this.peak += 0.6 * (env - this.peak);
    else this.peak += 0.002 * (floor - this.peak);
    if (this.peak < floor) this.peak = floor;

    this.valid = this.peak > (this.valid ? 0.8 : 1) * VALID_Q * q;
    const range = this.peak - floor;
    const thr = this.on ? Math.max(floor + 0.4 * range, OFF_Q * q) : Math.max(floor + 0.55 * range, ON_Q * q);
    const raw = this.valid && env > thr && clean;
    if (this.on) {
      this.markSum += env;
      this.markBlocks++;
    }

    // Debounce: a change counts after `need` blocks, back-dated to the first one.
    // (the envelope window is already half a dot or more: together with it a 1-dot space must still show up)
    const need = clamp(Math.round((DEBOUNCE_DOTS * (this.dot || PRIOR_DOT)) / this.blockDur), 2, 6);
    if (raw === this.on) this.cand = 0;
    else {
      if (this.cand === 0) this.candStart = nb;
      if (++this.cand >= need) {
        this.#transition(raw, this.candStart);
        this.cand = 0;
      }
    }
    this.blk++;
    if (!this.on) this.#liveGap(this.cand > 0 ? this.candStart : this.blk);
    if (++this.statusIn >= 20) {
      this.statusIn = 0;
      if (this.onStatus) this.onStatus(this.status);
    }
  }

  #transition(newOn, atBlock) {
    // Edges are found per 5 ms block on the 15 ms envelope, which makes marks come out a bit long and spaces a bit
    // short: EDGE_BIAS (negative) blocks are moved back.
    const raw = (atBlock - this.runStart) * this.blockDur;
    if (this.on) {
      const level = this.markBlocks ? this.markSum / this.markBlocks / Math.max(this.noise, MIN_NOISE) : 0;
      // A short weak mark in noise is a spike, not a dot. Left out, the space before it simply goes on. Without this,
      // spikes look like fast dots, the speed runs up to 50 WPM, the filter gets wide and lets in more spikes.
      // Also a mark shorter than MIN_MARK (faster than about 48 WPM) is a blip unless it is very strong and the speed
      // really is that high (13 Nov 2026: strong blips from AGC/APF ran the speed up to 55 WPM).
      const blip = raw < MIN_MARK && (level < 3 * SURE_Q || (this.dot && this.dot >= LOCK_SAFE_DOT && this.history.length >= 6)) && !(this.dot && this.dot < LOCK_SAFE_DOT && this.history.length >= 6 && median(this.history) < 2 * MIN_MARK);
      const spike = (this.history.length >= 3 && this.dot && raw < GLITCH * this.dot && level < SURE_Q) || (raw < WEAK_MIN && level < GATE_STRONG) || blip;
      if (spike) {
        this.afcRe = 0;
        this.afcIm = 0;
        this.markSum = 0;
        this.markBlocks = 0;
        this.on = false;
        this.runStart = this.spaceStart;
        this.stage = this.spaceStage;
        if (this.onTone) this.onTone(false);
        return;
      }
      this.#afc();
      if (this.onElement) this.onElement({ on: true, dur: raw + EDGE_BIAS * this.blockDur });
      this.curLevel.push(level);
      this.lastLevel = level;
      // Does this mark keep the gate open? A weak mark far below the station we were reading is noise in the passband
      // (24 Nov 2026, "CQ LP OM2VL" at 7004 kHz: after the station stopped, a narrow filter and AGC release made noise
      // marks of 2-3 times the noise level, each one pushing the 3 s gate hold on, so the gate stayed open for 8 s and
      // gave lone E's). Such a mark does not renew the hold.
      // Characters held back before the gate opened are old noise once a long quiet gap has passed (the E's that came out
      // together with the first real letters).
      if (!this.gate && this.pending.length && (this.runStart - this.lastMarkBlk) * this.blockDur > Math.max(PENDING_HOLD, 7.5 * (this.dot || PRIOR_DOT))) this.pending = [];
      if (!this.gate || (level >= GATE_WEAK && level >= HOLD_REL * this.markAvg)) {
        this.lastMarkBlk = atBlock;
        this.markAvg = this.markAvg ? this.markAvg + 0.2 * (level - this.markAvg) : level;
      }
      this.#markEnd(raw + EDGE_BIAS * this.blockDur);
    } else {
      this.spaceStart = this.runStart;
      this.spaceStage = this.stage;
      if (this.onElement && this.history.length) this.onElement({ on: false, dur: Math.max(0, raw - EDGE_BIAS * this.blockDur) });
      this.#spaceEnd(Math.max(0, raw - EDGE_BIAS * this.blockDur));
    }
    this.markSum = 0;
    this.markBlocks = 0;
    this.on = newOn;
    this.runStart = atBlock;
    this.stage = 0;
    if (this.onTone) this.onTone(newOn);
  }

  // After a mark: move the detector to where the tone really is (half of the measured offset, so noise does little).
  #afc() {
    const re = this.afcRe;
    const im = this.afcIm;
    this.afcRe = 0;
    this.afcIm = 0;
    if (!(re * re + im * im > 0)) return;
    const err = Math.atan2(im, re) / (2 * Math.PI * this.blockDur); // Hz, + = the tone is higher
    if (Math.abs(err) > 80) return;
    if (this.lock === null) {
      // free: the pitch itself moves (the pitch scan follows it too)
      this.tuneHz = clamp(this.tuneHz + 0.5 * err, this.minPitch - 50, this.maxPitch + 50);
      this.#detector();
      return;
    }
    const next = clamp((this.fine || 0) + 0.5 * err, -FINE_MAX, FINE_MAX);
    if (Math.abs(next - this.fine) < 0.5) return;
    this.fine = next;
    this.#detector();
  }

  // ----- the Viterbi decoder (step 6) -----

  #vMixer(hz) {
    this.vHz = hz;
    const w = (2 * Math.PI * hz) / this.sr;
    this.vStepRe = Math.cos(w);
    this.vStepIm = -Math.sin(w);
  }

  #vitReset() {
    if (this.vit) this.vit.reset();
    if (this.wordBuf && this.wordBuf.length) this.#wordDone(); // what was heard before stays
    this.vHz = null;
    this.live = [];
    this.fastUntil = -Infinity;
    this.#pending();
  }

  // Letters the Viterbi decoder made final: they are the text; the fast guesses up to there are dropped.
  #vitTake() {
    // the fast decoder's speed is a good hint only on clear marks (in noise its marks are partly spikes)
    // P4: and only after 6 marks of THIS transmission: right after a silence the speed is still that of the previous
    // sender (a speed change between two overs gave a stale hint, so the wrong lane)
    if (this.dot && this.gate && this.history.length >= 6 && this.marksSinceQuiet >= 6 && (this.lastLevel || 0) >= SURE_Q) this.vit.hintDot(this.dot);
    // the tone sits beside the Viterbi mixer (the radio's pitch is locked, the station a bit off): follow it, within
    // the fine tuning range of the fast decoder
    // only a strong signal: on a weak one the fine tuning also follows noise bursts
    const clear = this.gate && this.history.length >= 6 && (this.lastLevel || 0) >= SURE_Q;
    if (clear && this.vHz && this.detHz && Math.abs(this.detHz - this.vHz) > 8) {
      this.#vMixer(this.detHz); // a clear signal: the fast decoder's fine tuning knows where the tone is
      this.vit.fOff = 0;
    } else if (Math.abs(this.vit.fOff) > 6 && this.vHz) {
      const base = this.lock ?? this.tuneHz;
      this.#vMixer(clamp(this.vHz + this.vit.fOff, base - FINE_MAX, base + FINE_MAX));
      this.vit.fOff = 0;
    }
    const got = this.vit.take();
    if (!got.length) {
      // fast guesses the Viterbi decoder did not confirm within a few seconds were noise (or it will not show them)
      if (!this.wordBuf.length && this.live.length && this.vit.abs - this.live[0].at > (2.5 + 3 * (this.dot || PRIOR_DOT)) / this.blockDur) {
        const keep = (2.5 + 3 * (this.dot || PRIOR_DOT)) / this.blockDur;
        const old = this.live.filter((x) => this.vit.abs - x.at > keep);
        this.live = this.live.filter((x) => this.vit.abs - x.at <= keep);
        this.#keepFast(old);
        this.#pending();
      }
      return;
    }
    for (const c of got) {
      // (already given as fast text, see #keepFast: not twice)
      if (typeof c.end === 'number' && c.end <= this.fastUntil) continue;
      if (c.ch === ' ') this.#wordDone();
      else this.wordBuf.push(c);
    }
    // without context the letters go out at once (the word is not needed whole)
    if (!this.vocabFn && this.wordBuf.length) this.#emitLetters(this.wordBuf.splice(0));
    this.#pending();
  }

  // A word from the Viterbi decoder is complete: checked against the context (cw-context.js), then shown.
  #wordDone() {
    const w = this.wordBuf.splice(0);
    if (w.length && this.vocabFn) {
      let fixed = null;
      try {
        if (!this.vocab || Date.now() - this.vocabAt > 2000) {
          this.vocab = buildVocabulary(this.vocabFn() || []);
          this.vocabAt = Date.now();
        }
        fixed = correctWord(w, this.vocab);
      } catch {
        fixed = null;
      }
      if (fixed) {
        const heard = w.map((c) => c.ch).join('');
        const end = Math.max(...w.map((c) => c.end ?? -Infinity));
        const letters = [...fixed].map((ch, i) => ({ ch, pattern: MORSE[ch] || '', confidence: 0.6, fixed: heard, end: i === fixed.length - 1 ? end : undefined }));
        this.#emitLetters(letters);
      } else this.#emitLetters(w);
    } else if (w.length) this.#emitLetters(w);
    if (this.chars.length && this.chars[this.chars.length - 1].ch !== ' ') this.#emit({ ch: ' ', pattern: '', confidence: 1 });
    // the report the other station gives (5NN, 599) for the log line
    const words = this.text.trim().split(/\s+/).slice(-8);
    const rst = heardReport(words);
    if (rst && rst !== this.lastReport) {
      this.lastReport = rst;
      if (this.onReport) this.onReport(rst, this.text.trim().split(/\s+/).slice(-20));
    }
  }

  // Fast characters the Viterbi decoder did not confirm (18 Nov 2026, Richard: over the air the grey guess was right
  // but vanished). With room reverb the gaps are never quiet, and the Viterbi decoder then stays silent for whole
  // stretches. The fast decoder only gives characters once the signal proved to be Morse (the gate), so a clear guess is
  // kept as text instead of dropped; a doubtful one (unknown or unsure characters) is still dropped.
  #keepFast(old) {
    const letters = old.filter((x) => x.c.ch !== ' ');
    const last0 = this.chars[this.chars.length - 1];
    if (!letters.length) {
      // a word space after kept letters
      if (last0 && last0.fast && last0.ch !== ' ') this.#emit({ ch: ' ', pattern: '', confidence: 1, fast: true });
      return;
    }
    // (they expire one by one, so this is mostly one letter: each must be sure itself)
    if (!letters.every((x) => x.c.ch !== '*' && (x.c.confidence ?? 0) >= FAST_KEEP_CONF)) return;
    // a lone E or T far weaker than the station that was being read is a noise mark in the passband (24 Nov 2026: lone
    // E's in the silence after "CQ LP OM2VL"): not kept as text (a real one is confirmed by the Viterbi decoder)
    if (letters.some((x) => x.weak && x.c.pattern.length === 1)) return;
    // a new word after Viterbi text; a letter right after a kept letter belongs to the same word
    const last = this.chars[this.chars.length - 1];
    if (last && last.ch !== ' ' && !last.fast && old[0].c.ch !== ' ') this.#emit({ ch: ' ', pattern: '', confidence: 1 });
    for (const x of old) {
      if (x.c.ch === ' ' && (!this.chars.length || this.chars[this.chars.length - 1].ch === ' ')) continue;
      this.#emit({ ...x.c, fast: true });
    }
    this.fastUntil = Math.max(this.fastUntil, old[old.length - 1].at);
  }

  #emitLetters(letters) {
    let upTo = -Infinity;
    for (const c of letters) {
      if (typeof c.end === 'number') upTo = Math.max(upTo, c.end);
      const out = { ch: c.ch, pattern: c.pattern, confidence: c.confidence };
      if (c.fixed) out.fixed = c.fixed;
      this.#emit(out);
    }
    if (upTo > -Infinity) {
      const margin = (3.5 * (this.dot || PRIOR_DOT)) / this.blockDur;
      this.live = this.live.filter((x) => x.at > upTo + margin);
    }
  }

  // Context for the Viterbi text: fn() -> the calls HamLog knows now ([{call, source}]): spots near the frequency,
  // the call in the log line, your own call. null = no context (the text as heard).
  setVocabulary(fn) {
    this.vocabFn = typeof fn === 'function' ? fn : null;
    this.vocab = null;
  }

  #pending() {
    const t = this.live.map((x) => x.c.ch).join('');
    if (t === this.pendingText) return;
    this.pendingText = t;
    if (this.onPending) this.onPending(t);
  }

  // End of the audio (a recording, the tests): the Viterbi decoder makes the rest final.
  flush() {
    if (!this.vit) return;
    this.vit.flush();
    this.#vitTake();
    if (this.wordBuf.length) this.#wordDone();
    this.live = [];
    this.#pending();
  }

  // the fast guess shown until the Viterbi decoder confirms it
  get guess() {
    return this.pendingText;
  }

  #markEnd(dur) {
    if (dur > 1.2) {
      // a stuck key or carrier, not a Morse element: it breaks the character
      this.cur.push(dur);
      return;
    }
    this.marksSinceQuiet++; // P4
    this.cur.push(dur);
    // (marks far longer than 6 dots stay out of the speed estimate, but not while the dot is implausibly short: then
    // the dashes must be able to pull it back, 13 Nov 2026)
    if (!(this.history.length >= 6 && dur > 6 * (this.dot || PRIOR_DOT) && (this.dot || PRIOR_DOT) >= LOCK_SAFE_DOT)) {
      this.history.push(dur);
      if (this.history.length > HISTORY) this.history.shift();
    }
    // The sender got much faster: the last three marks are all far below the dot length, so forget the old marks.
    const h = this.history;
    // (only for clear marks: weak short ones are more likely noise)
    if (this.dot && h.length > 6 && h.slice(-3).every((d) => d < 0.6 * this.dot) && (this.lastLevel || 0) >= GATE_STRONG) {
      this.history = h.slice(-3);
      this.dot = null;
    }
    const est = estimateDot(this.history, this.dot);
    const next = clamp(est.dot, MIN_DOT, MAX_DOT);
    // Weak signals: the speed changes slowly (at most 15% per mark), so a burst of noise cannot run it away.
    const weak = (this.lastLevel || 0) < GATE_STRONG;
    if (this.dot === null || this.history.length < 8) this.dot = next;
    else if (weak) this.dot = clamp(0.8 * this.dot + 0.2 * next, 0.85 * this.dot, 1.15 * this.dot);
    else this.dot = 0.5 * this.dot + 0.5 * next;
  }

  #spaceEnd(dur) {
    if (this.stage === 0 && this.cur.length) {
      const dot = this.dot || PRIOR_DOT;
      if (dur >= 2 * dot) this.#flush();
      if (dur >= 5 * dot) this.#word();
    }
  }

  // Gap handling while a space is still going on: the letter is complete after 2 dots, the word after 5.
  #liveGap(endBlock) {
    const dot = this.dot || PRIOR_DOT;
    if ((this.gate || this.pending.length) && (endBlock - this.lastMarkBlk) * this.blockDur > Math.max(GATE_HOLD, 12 * dot)) {
      // a long silence: the next signal has to prove itself again; held back characters were noise
      this.gate = false;
      this.pending = [];
      this.markAvg = 0;
      this.marksSinceQuiet = 0; // P4
      if (this.vit) this.vit.forgetHint();
    }
    if (this.stage >= 2) return;
    const elapsed = (endBlock - this.runStart) * this.blockDur;
    if (this.stage === 0 && elapsed >= 2 * dot) {
      this.stage = 1;
      if (this.cur.length) this.#flush();
    }
    if (this.stage === 1 && elapsed >= 5 * dot) {
      this.stage = 2;
      this.#word();
    }
  }

  #flush() {
    const dot = this.dot || PRIOR_DOT;
    const marks = this.cur;
    const levels = this.curLevel;
    this.cur = [];
    this.curLevel = [];
    if (!marks.length) return;
    let pattern = '';
    let conf = 1;
    for (const d of marks) {
      const r = d / dot;
      pattern += r >= 2 ? '-' : '.';
      conf = Math.min(conf, clamp(Math.abs(Math.log(r / 2)) / 0.4, 0, 1)); // 1 dot / 3 dots: sure; near 2 dots: unsure
    }
    if (this.history.length < 3) conf = Math.min(conf, 0.6); // the speed is not known well yet
    let ch = decodePattern(pattern, this.symbols);
    if (ch === '*') conf = Math.min(conf, 0.2);
    const c = { ch, pattern, confidence: Math.round(conf * 100) / 100 };
    const lvl = levels.length ? median(levels) : 0;
    if (this.gate) return this.#out(c, lvl);
    this.pending.push({ c, level: lvl });
    this.#checkGate();
  }

  #word() {
    if (!this.gate) {
      if (this.pending.length && this.pending[this.pending.length - 1].c.ch !== ' ') this.pending.push({ c: { ch: ' ', pattern: '', confidence: 1 }, level: 0 });
      return;
    }
    if (!this.vit && (!this.chars.length || this.chars[this.chars.length - 1].ch === ' ')) return;
    this.#out({ ch: ' ', pattern: '', confidence: 1 });
  }

  // Morse or noise? Opens the gate when the held characters look like Morse: at least 3 elements of a strong
  // signal, or 2 known characters with 4 elements of a weaker one. Keeps at most the last 6 characters.
  #checkGate() {
    const p = this.pending.filter((x) => x.c.ch !== ' ');
    if (p.length > 6) this.pending = this.pending.slice(-6);
    const elements = p.reduce((s, x) => s + x.c.pattern.length, 0);
    const known = p.filter((x) => x.c.ch !== '*').length;
    const level = p.length ? median(p.map((x) => x.level)) : 0;
    // (weak path: not only lone E's and T's; a row of single marks 2-3 times the noise level is a noise burst, and
    // opened the gate for the E's in a quiet band, 24 Nov 2026)
    const multi = p.some((x) => x.c.pattern.length >= 2);
    const open = (elements >= 3 && level >= GATE_STRONG) || (known >= 2 && elements >= 4 && level >= GATE_WEAK && multi);
    if (!open) return;
    this.gate = true;
    // a held character far weaker than the real ones is a noise mark just before them (a lone E in front of the first C)
    const top = Math.max(...p.map((x) => x.level));
    const held = this.pending.filter((x) => x.c.ch === ' ' || x.level >= HOLD_REL * top || x.level >= GATE_STRONG);
    this.markAvg = level;
    this.pending = [];
    for (const x of held) {
      if (x.c.ch === ' ' && (!this.chars.length || this.chars[this.chars.length - 1].ch === ' ')) continue;
      this.#out(x.c, x.level);
    }
  }

  // A character of the fast decoder: the text itself, or (with the Viterbi decoder) a guess shown until confirmed.
  #out(c, level = Infinity) {
    // weak: a mark of the station level far below what has been read lately (see #keepFast)
    const weak = level < SURE_Q && this.markAvg > 0 && level < HOLD_REL * this.markAvg;
    if (this.onFastChar) this.onFastChar(c);
    if (!this.vit) return this.#emit(c);
    if (c.ch === ' ' && (!this.live.length || this.live[this.live.length - 1].c.ch === ' ')) return;
    this.live.push({ c, at: this.vit.abs, weak });
    if (this.live.length > 80) this.live.shift();
    this.#pending();
  }

  #emit(c) {
    this.chars.push(c);
    this.text += c.ch;
    if (this.chars.length > MAX_CHARS) {
      const old = this.chars.shift();
      this.text = this.text.slice(old.ch.length);
    }
    if (this.onChar) this.onChar(c);
  }
}

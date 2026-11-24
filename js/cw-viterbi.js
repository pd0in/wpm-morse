// Viterbi Morse decoder for weak signals (Richard, 5 Nov 2026). Pure logic, no DOM: fed by CwDecoder with the I/Q
// value of every 5 ms block at the pitch, it finds the most likely text for the whole stretch of audio heard so far,
// instead of deciding "tone or no tone" block by block. That is the way CW Skimmer and the Bayesian decoders work
// (VE3NEA; Bell 1977; AG1LE): a soft decision per block, durations judged against the speed, and only letters that
// exist. A dash with a dip in the middle, or a noise burst between two letters, is weighed against the alternatives
// instead of breaking the text.
//
// The model:
//  - a mark from block t1 to t2 scores the log-likelihood ratio "tone of amplitude a" versus "noise only" (Rice
//    versus Rayleigh), on the coherent sum of its blocks: a tone adds up over the whole mark, noise only partly.
//    The sum is taken for a few small frequency offsets (FREQ_HYPS), so a tone a few Hz off still adds up; a
//    block-by-block score (with a penalty) covers a tone that does not add up at all.
//  - durations: dots around 1 dot, dashes around 3, gaps inside a letter around 1, between letters 3 or more
//    (log-normal spread, so a heavy or uneven fist still fits);
//  - only paths through the Morse tree: a letter ends where a pattern exists;
//  - a penalty per mark, so noise does not turn into a stream of E's and T's.
// The speed is not known beforehand (and in the noise the fast decoder cannot measure it), so the search runs in
// LANES at seven speeds side by side (5 to 40 WPM); each lane fits speeds about 20% around its own, and the lane
// with the best score gives the text. The search runs forward block by block (dynamic programming); every POLL
// blocks the best path is traced back and the letters that ended more than LAG ago are final (the path no longer
// changes there). The text is therefore about a second behind the audio; CwDecoder shows its fast guess meanwhile.
import { MORSE, MORSE_PROSIGNS, decodePattern } from './cw-decoder-tables.js';

// ---------- the Morse tree ----------
const NODES = [{ pattern: '', end: false, kids: [-1, -1] }];
{
  const index = new Map([['', 0]]);
  const all = new Set([...Object.values(MORSE), ...Object.values(MORSE_PROSIGNS)].filter((p) => p.length <= 7));
  for (const p of all) {
    for (let i = 1; i <= p.length; i++) {
      const q = p.slice(0, i);
      if (index.has(q)) continue;
      index.set(q, NODES.length);
      NODES.push({ pattern: q, end: false, kids: [-1, -1] });
      NODES[index.get(q.slice(0, -1))].kids[q[i - 1] === '.' ? 0 : 1] = index.get(q);
    }
    NODES[index.get(p)].end = true;
  }
}
const NN = NODES.length;
const KID0 = Int16Array.from(NODES, (n) => n.kids[0]);
const KID1 = Int16Array.from(NODES, (n) => n.kids[1]);
const ENDS = Uint8Array.from(NODES, (n) => (n.end ? 1 : 0));
// Rare characters cost a little extra (log), so a broken letter in QRM does not become ) @ ; or " (Richard's
// recordings of 6 Nov 2026): letters, digits and the usual signs ? / . , <BT> <AR> <SK> <KN> <BK> are free.
const COMMON = new Set([...'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789?/.,'].map((c) => MORSE[c]).concat(['<BT>', '<AR>', '<SK>', '<KN>', '<BK>'].map((c) => MORSE_PROSIGNS[c])));
export const RARE_PEN = 3;
const ENDP = Float64Array.from(NODES, (n) => (n.end && !COMMON.has(n.pattern) ? -RARE_PEN : 0));

// ---------- tuning ----------
const CAP = 1600; // blocks in the search window (8 s at 5 ms)
const POLL = 40; // blocks between trace backs (0.2 s)
const LAG_S = 1.0; // s: letters that ended longer ago than this (or LAG_DOTS dots) are final
const LAG_DOTS = 9;
const START_S = 2.5; // s: the first letters of a transmission wait this long for the speed (unless it is known)
const LANE_WPM = [5, 7, 10, 14, 20, 28, 40]; // the speeds searched side by side
const LANE_SPAN = 1.22; // a lane follows the speed within this factor of its own
const SWITCH = 10; // log: another lane must be this much better before the text follows it
const SIG_MARK = 0.27; // log-normal spread of mark durations
const SIG_SPACE = 0.35; // ... and of gaps
const MARK_PEN = 4; // log penalty per mark (no text from noise)
const FREQ_HYPS = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]; // Hz: offsets tried for the coherent sum
const BEAM = 12; // states this far (log) below the best at their block are dropped
const WORD_MIN = 30; // a word needs this much evidence (summed mark scores) while no Morse has been heard lately
const ELEMENT_MIN = 10; // ... and this much per element (noise that fits gives 6 to 11, a real -5 dB signal 12 or more);
const BLOCK_MIN = 0.35; // ... and per block of tone (long marks in noise gather a score too: a 5 WPM O from crackle)
const AMP_SPREAD = 0.55; // ... and its marks must have about one strength (spread of the log amplitude)
const HOLD_MIN = 7; // ... and this much shortly after a word that had WORD_MIN
const SURE_HOLD_S = 4; // s after a confident word in which weaker words are let through too
const A_CAP = 6; // per-block mark level used in the scores at most: above it the decision is easy anyway, and a
// slightly wrong model (a tone a Hz off, fading inside a dash) must not outweigh the timing
const NONCOH_PEN = 2; // the block-by-block (non-coherent) score of a mark counts with this penalty
const MIN_N0 = 1e-7; // noise power floor per block (digital silence)
const LOOK = 20; // blocks of look-ahead (0.1 s): the scaling of a strong signal is right from its first block on
const FIRST_NOISE = 60; // blocks heard before the noise level is first set (they are searched afterwards)
const NEG = -1e30;
const DIP_MINA = 1.5; // mark level (scaled) from which a dip is judged
const DIP_FRAC = 0.1; // a block below this share of the mark power is quiet
const DIP_RUN = 3; // quiet blocks in a row that make a dip
const DIP_PEN = 3; // log penalty per block of a dip in a mark

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
function logI0(x) {
  if (x < 3.75) {
    const t = (x / 3.75) ** 2;
    return Math.log(1 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813))))));
  }
  return x - 0.5 * Math.log(2 * Math.PI * x) + Math.log(1 + 0.125 / x);
}
const logNormal = (d, mu, s) => {
  const z = Math.log(d / mu) / s;
  return -0.5 * z * z;
};

// One speed: the search states of every block in the window.
class Lane {
  constructor(dot) {
    this.nominal = dot;
    this.D = 0;
    this.setDot(dot);
    const n = (CAP + 1) * NN;
    this.A = new Float32Array(n); // best score with a mark ending at block i in node n
    this.B = new Float32Array(n); // best score with a gap ending at block i (node 0: between letters)
    this.Ab = new Int32Array(n);
    this.An = new Int16Array(n);
    this.Bb = new Int32Array(n);
    this.Bn = new Int16Array(n);
    this.Bk = new Int8Array(n); // 0 gap inside a letter, 1 gap after a letter, 2 idle (one more quiet block), 3 start
    this.bestA = new Float64Array(CAP + 1);
    this.bestB = new Float64Array(CAP + 1);
    // the states within the beam, per block: node lists
    this.actA = new Int16Array(n);
    this.actB = new Int16Array(n);
    this.nA = new Int16Array(CAP + 1);
    this.nB = new Int16Array(CAP + 1);
  }

  // the duration scores at this speed, per length in blocks
  setDot(D) {
    if (Math.abs(D - this.D) < 0.05) return;
    this.D = D;
    this.mMin = Math.max(1, Math.floor(0.35 * D));
    this.mMax = Math.ceil(4.8 * D);
    this.eMax = Math.ceil(2.3 * D);
    this.cMax = Math.ceil(4 * D);
    this.pDot = new Float64Array(this.mMax + 1);
    this.pDash = new Float64Array(this.mMax + 1);
    for (let d = 1; d <= this.mMax; d++) {
      this.pDot[d] = logNormal(d, D, SIG_MARK);
      this.pDash[d] = logNormal(d, 3 * D, SIG_MARK);
    }
    this.pIn = new Float64Array(this.cMax + 1);
    this.pAfter = new Float64Array(this.cMax + 1);
    for (let d = 1; d <= this.cMax; d++) {
      this.pIn[d] = logNormal(d, D, SIG_SPACE);
      this.pAfter[d] = d < 3 * D ? logNormal(d, 3 * D, SIG_SPACE) : 0;
    }
  }

  get score() {
    return this.len ? Math.max(this.bestA[this.len], this.bestB[this.len]) : 0;
  }
}

export class MorseViterbi {
  // blockDur: s per block; symbols: show + and = instead of <AR> and <BT>
  constructor({ blockDur = 0.005, symbols = false } = {}) {
    this.bd = blockDur;
    this.symbols = symbols;
    this.lanes = LANE_WPM.map((w) => new Lane(1.2 / w / blockDur));
    this.maxMark = Math.ceil(4.8 * this.lanes[0].nominal * LANE_SPAN);
    this.markCache = new Float64Array(this.maxMark + 1);
    const H = FREQ_HYPS.length;
    this.PR = Array.from({ length: H }, () => new Float64Array(CAP + 1));
    this.PI = Array.from({ length: H }, () => new Float64Array(CAP + 1));
    this.rot = FREQ_HYPS.map(() => [1, 0]);
    this.yr = new Float64Array(CAP);
    this.yi = new Float64Array(CAP);
    this.av = new Float64Array(CAP); // the mark level a used for each block (scaled)
    this.gain = new Float64Array(CAP); // the scaling of each block (1 = noise level)
    this.PL = new Float64Array(CAP + 1); // prefix sums of the block-by-block mark score
    this.PD = new Float64Array(CAP + 1); // prefix sums of dip blocks (3+ quiet blocks in a row inside a mark)
    this.qRun = 0;
    this.PA = new Float64Array(CAP + 1); // ... of a
    this.PS = new Float64Array(CAP + 1); // ... of sqrt(a)
    this.reset();
  }

  reset() {
    this.s0 = 0; // absolute block number of index 0
    this.len = 0; // blocks in the window
    this.abs = 0; // blocks received
    this.done = 0; // blocks searched (abs minus the queue)
    this.queue = []; // blocks received but not yet searched: [zr, zi, blank, power]
    this.prevP = null;
    this.n0 = 0; // noise power per block
    this.a = 0; // mark power per block, in units of n0
    this.peak = 0;
    this.sust = 0;
    this.marksLearned = 0;
    this.powRing = new Float64Array(400);
    this.kappa = 1;
    this.powN = 0;
    this.powW = 0;
    this.out = []; // final letters not yet taken
    this.lastEnd = -1e9; // absolute block where the last final letter ended
    this.spaceDone = true;
    this.gapMem = []; // P2: [{g (blocks), at (abs block)}] gaps between final letters (letter gaps / word gaps)
    this.sureUntil = -1e9;
    this.pendingWord = []; // final letters of a word still being judged (gate)
    this.pendingScore = 0;
    this.shownInWord = false;
    this.sinceTrace = 0;
    this.lane = 4; // the lane that gives the text (20 WPM until known)
    this.hint = 0;
    this.hintUntil = -1;
    this.fOff = 0; // Hz: where the tone sits from the mixer, measured on final marks (CwDecoder retunes on it)
    for (const L of this.lanes) L.setDot(L.nominal);
    this.#startWindow();
  }

  // the speed of the lane giving the text, in dots of blocks
  get dotBlocks() {
    return this.lanes[this.lane].D;
  }

  get wpm() {
    return Math.round(1.2 / (this.dotBlocks * this.bd));
  }

  // The speed measured by the fast decoder on clear marks (s per dot): on a clear signal that is the speed, and the
  // lane nearest to it gives the text. Without a hint (weak signals) the lanes compete on their scores.
  forgetHint() {
    this.hint = 0; // P4: the gate closed (silence): the speed of the next transmission is not known
    this.hintUntil = -1;
  }

  hintDot(seconds) {
    if (!(seconds > 0)) return;
    this.hint = seconds / this.bd;
    this.hintUntil = this.abs + 2 / this.bd;
    // a clear signal that the fast decoder already reads as Morse: no need to prove it again
    this.sureUntil = Math.max(this.sureUntil, this.done + SURE_HOLD_S / this.bd);
  }

  // One block: the I/Q sum at the pitch (any scale). blank: no signal to judge (a carrier, own transmission).
  push(zr, zi, blank = false) {
    const p = zr * zr + zi * zi;
    if (!blank) {
      this.powRing[this.powW] = p;
      this.powW = (this.powW + 1) % this.powRing.length;
      if (this.powN < this.powRing.length) this.powN++;
    }
    this.abs++;
    this.queue.push([zr, zi, blank, blank ? 0 : p]);
    if (!this.n0) {
      if (this.powN < FIRST_NOISE) return;
      this.#levelsFromRing();
    }
    while (this.queue.length > LOOK) this.#search(this.queue.shift());
  }

  #search([zr, zi, blank, p]) {
    // a click or a crackle: one loud block between quiet ones (a mark lasts several blocks, even at 50 WPM)
    const next = this.queue.length ? this.queue[0][3] : p;
    const prev = this.prevP ?? p;
    this.prevP = p;
    if (!blank && p > 8 * this.n0 && p > 6 * Math.max(prev, next)) blank = true;
    if (this.len >= CAP) this.#rebaseAt(this.len - Math.ceil(this.#lag()));
    // strong signals are scaled as if the noise were higher, so a block never counts for more than A_CAP: then the
    // timing (dots, dashes, gaps) keeps its say, also when the tone is not quite steady. The level comes from this
    // block and the look-ahead, so the first dash of a strong station is scaled like the rest.
    let ahead = p;
    for (const q of this.queue) if (q[3] > ahead) ahead = q[3];
    this.peak = Math.max(ahead / this.n0 - 1, this.peak * 0.995); // decays in about a second
    // a steady strong station after weak junk (the junk set the mark level): follow it at once (3 blocks in a row, so a click does not count)
    this.sust = Math.max(Math.min(prev, p, next) / this.n0 - 1, (this.sust || 0) * 0.995);
    if (!blank && 0.6 * this.sust > Math.max(A_CAP, 2 * this.a)) this.a = 0.6 * this.sust;
    if (this.marksLearned < 4 && 0.6 * this.peak > A_CAP) this.a = Math.max(this.a, 0.6 * this.peak); // strong, nothing learned yet
    const g = Math.max(1, Math.max(this.a || 1, 0.5 * this.peak) / A_CAP);
    const k = blank ? 0 : 1 / Math.sqrt(this.n0 * g);
    this.gain[this.len] = g;
    this.#addBlock(zr * k, zi * k, Math.min(A_CAP, (this.a || 1) / g));
    this.done++;
    if (++this.sinceTrace >= POLL) {
      this.sinceTrace = 0;
      this.#trace(false);
    }
  }

  // Letters that became final since the last call: [{ch, pattern, confidence, end}] (' ' = word space).
  take() {
    const o = this.out;
    this.out = [];
    return o;
  }

  // End of the audio (tests, a recording): everything left becomes final.
  flush() {
    if (!this.n0 && this.powN) this.#levelsFromRing();
    if (this.n0) while (this.queue.length) this.#search(this.queue.shift());
    if (this.len) this.#trace(true);
  }

  // ----- internals -----

  #lag() {
    return Math.max(LAG_S / this.bd, LAG_DOTS * this.dotBlocks);
  }

  // Noise lumpiness (kappa): in the gaps of the decoded path (well away from marks), how much the I/Q sum of 8
  // blocks exceeds the sum of their powers. White noise: 1; noise through a 100-200 Hz radio filter: about 2. Measured
  // on the path, because at weak signals the marks cannot be told from the noise otherwise.
  #measureKappa(marks, until) {
    const W = 8;
    const busy = new Uint8Array(this.len + 1);
    for (const [a, b] of marks) for (let t = Math.max(0, a - 3); t < Math.min(this.len, b + 3); t++) busy[t] = 1;
    const ratios = [];
    for (let s = 0; s + W <= until; s += W) {
      let ok = true;
      let r = 0;
      let im = 0;
      let pw = 0;
      for (let j = s; j < s + W; j++) {
        if (busy[j] || (this.yr[j] === 0 && this.yi[j] === 0)) {
          ok = false;
          break;
        }
        r += this.yr[j];
        im += this.yi[j];
        pw += this.yr[j] ** 2 + this.yi[j] ** 2;
      }
      if (!ok || !(pw > 0)) continue;
      ratios.push((r * r + im * im) / pw);
    }
    const n = ratios.length;
    if (n < 10) return;
    // the median, against that of white noise (0.78 for 8 blocks): a weak mark the path missed hardly moves it
    ratios.sort((a, b) => a - b);
    let k = clamp(ratios[n >> 1] / 0.78, 1, 4);
    if (k < 1.35) k = 1; // white noise (the estimate wanders a little above 1)
    this.kappa += Math.min(1, n / 60) * 0.3 * (k - this.kappa);
  }

  #levelsFromRing() {
    const s = Array.from(this.powRing.subarray(0, this.powN)).sort((x, y) => x - y);
    this.n0 = Math.max(MIN_N0, s[Math.floor(0.2 * (s.length - 1))] / 0.223); // 20th percentile of an exponential
    if (!this.a) this.a = clamp(s[Math.floor(0.9 * (s.length - 1))] / this.n0 - 1, 0.5, 1e12);
  }

  #startWindow() {
    this.len = 0;
    this.PL[0] = 0;
    this.PD[0] = 0;
    this.qRun = 0;
    this.PA[0] = 0;
    this.PS[0] = 0;
    for (let h = 0; h < FREQ_HYPS.length; h++) {
      this.PR[h][0] = 0;
      this.PI[h][0] = 0;
      this.rot[h][0] = 1;
      this.rot[h][1] = 0;
    }
    for (const L of this.lanes) {
      L.len = 0;
      this.#forward(L, 0);
    }
  }

  // a: the mark level (scaled) assumed for this block
  #addBlock(yr, yi, a) {
    const i = this.len;
    this.yr[i] = yr;
    this.yi[i] = yi;
    this.av[i] = a;
    this.PA[i + 1] = this.PA[i] + a;
    this.PS[i + 1] = this.PS[i] + Math.sqrt(a);
    // (one loud block - a click, a crackle - counts at most as a clear mark block, not more)
    {
      // a dip of 3+ blocks with (almost) no tone where a mark is expected: the coherent sum hardly notices it (the
      // gap between the dots of a D from a strong station became a dash: M), so it costs extra
      const quiet = a >= DIP_MINA && yr * yr + yi * yi < DIP_FRAC * a;
      this.qRun = quiet ? this.qRun + 1 : 0;
      this.PD[i + 1] = this.PD[i] + (this.qRun === DIP_RUN ? DIP_RUN : this.qRun > DIP_RUN ? 1 : 0);
    }
    this.PL[i + 1] = this.PL[i] - a + logI0(2 * Math.sqrt(a * Math.min(yr * yr + yi * yi, 3 * (1 + a))));
    for (let h = 0; h < FREQ_HYPS.length; h++) {
      const r = this.rot[h];
      const c = r[0];
      const s = r[1];
      this.PR[h][i + 1] = this.PR[h][i] + yr * c + yi * s;
      this.PI[h][i + 1] = this.PI[h][i] + yi * c - yr * s;
      // turn the reference on by one block
      const w = 2 * Math.PI * FREQ_HYPS[h] * this.bd;
      const cw = Math.cos(w);
      const sw = Math.sin(w);
      r[0] = c * cw - s * sw;
      r[1] = c * sw + s * cw;
    }
    this.len = i + 1;
    // the scores of all marks ending here, once for all lanes
    const t2 = i + 1;
    const maxD = Math.min(this.maxMark, t2);
    for (let d = 1; d <= maxD; d++) this.markCache[d] = this.#markScore(t2 - d, t2);
    for (const L of this.lanes) {
      L.len = t2;
      this.#forward(L, t2);
    }
  }

  // The score of a mark over blocks [t1, t2) of the window (best frequency offset).
  #markScore(t1, t2) {
    const d = t2 - t1;
    const sa = (this.PS[t2] - this.PS[t1]) / d;
    let best = 0;
    for (let h = 0; h < FREQ_HYPS.length; h++) {
      const r = this.PR[h][t2] - this.PR[h][t1];
      const im = this.PI[h][t2] - this.PI[h][t1];
      const y2 = r * r + im * im;
      if (y2 > best) best = y2;
    }
    // kappa: how much more noise adds up over several blocks than white noise would (a narrow radio filter makes the
    // noise come in lumps; without this a lump between two words was read as an E)
    const k = this.kappa;
    const dips = t2 - t1 > 4 ? this.PD[t2 - 1] - this.PD[t1 + 1] : 0;
    const coherent = -(this.PA[t2] - this.PA[t1]) / k + logI0((2 * sa * Math.sqrt(best)) / k) - DIP_PEN * dips;
    const blockwise = (this.PL[t2] - this.PL[t1]) / k - NONCOH_PEN;
    return Math.max(coherent, blockwise) - MARK_PEN;
  }

  // Forward step of one lane: the states at window index i (i blocks done). Uses markCache for the marks ending at i.
  #forward(L, i) {
    const { A, B, Ab, An, Bb, Bn, Bk, actA, actB, nA, nB } = L;
    const row = i * NN;
    A.fill(NEG, row, row + NN);
    B.fill(NEG, row, row + NN);
    if (i === 0) {
      B[0] = 0;
      Bk[0] = 3;
      L.bestA[0] = NEG;
      L.bestB[0] = 0;
      nA[0] = 0;
      nB[0] = 1;
      actB[0] = 0;
      return;
    }
    // idle: one more quiet block between letters / words
    B[row] = B[row - NN];
    Bb[row] = i - 1;
    Bn[row] = 0;
    Bk[row] = 2;
    // marks ending at i
    const mMax = Math.min(this.maxMark, L.mMax);
    for (let d = L.mMin; d <= mMax && d <= i; d++) {
      const t1 = i - d;
      const cnt = nB[t1];
      if (!cnt) continue;
      const e = this.markCache[d];
      const pDot = L.pDot[d];
      const pDash = L.pDash[d];
      const r1 = t1 * NN;
      for (let q = 0; q < cnt; q++) {
        const n = actB[r1 + q];
        const b = B[r1 + n];
        const k0 = KID0[n];
        if (k0 >= 0) {
          const v = b + e + pDot;
          const j = row + k0;
          if (v > A[j]) {
            A[j] = v;
            Ab[j] = t1;
            An[j] = n;
          }
        }
        const k1 = KID1[n];
        if (k1 >= 0) {
          const v = b + e + pDash;
          const j = row + k1;
          if (v > A[j]) {
            A[j] = v;
            Ab[j] = t1;
            An[j] = n;
          }
        }
      }
    }
    // gaps ending at i (after a mark)
    const eMax = L.eMax;
    for (let d = 1; d <= L.cMax && d <= i; d++) {
      const t1 = i - d;
      const cnt = nA[t1];
      if (!cnt) continue;
      const pIn = L.pIn[d];
      const pAfter = L.pAfter[d];
      const r1 = t1 * NN;
      for (let q = 0; q < cnt; q++) {
        const n = actA[r1 + q];
        const s = A[r1 + n];
        if (d <= eMax) {
          const v = s + pIn;
          const j = row + n;
          if (v > B[j]) {
            B[j] = v;
            Bb[j] = t1;
            Bn[j] = n;
            Bk[j] = 0;
          }
        }
        if (ENDS[n]) {
          const v = s + pAfter + ENDP[n];
          if (v > B[row]) {
            B[row] = v;
            Bb[row] = t1;
            Bn[row] = n;
            Bk[row] = 1;
          }
        }
      }
    }
    let ba = NEG;
    let bb = NEG;
    for (let n = 0; n < NN; n++) {
      if (A[row + n] > ba) ba = A[row + n];
      if (B[row + n] > bb) bb = B[row + n];
    }
    L.bestA[i] = ba;
    L.bestB[i] = bb;
    // the states within the beam (of the best of both), for the next blocks
    const floor = Math.max(ba, bb) - BEAM;
    let ca = 0;
    let cb = 0;
    for (let n = 0; n < NN; n++) {
      if (A[row + n] >= floor) actA[row + ca++] = n;
      if (B[row + n] >= floor) actB[row + cb++] = n;
    }
    nA[i] = ca;
    nB[i] = cb;
  }

  // Traces the best path of a lane back from the newest block: letters [{pattern, start, end, score, low, marks}]
  // and all marks [[t1, t2]] (window indexes).
  #path(L) {
    const i = this.len;
    const row = i * NN;
    let bestV = NEG;
    let n = 0;
    let ph = 'B';
    for (let k = 0; k < NN; k++) {
      if (L.A[row + k] > bestV) {
        bestV = L.A[row + k];
        n = k;
        ph = 'A';
      }
      if (L.B[row + k] > bestV) {
        bestV = L.B[row + k];
        n = k;
        ph = 'B';
      }
    }
    const marks = [];
    const letters = [];
    let t = i;
    let guard = 0;
    while (t > 0 && guard++ < 200000) {
      const j = t * NN + n;
      if (ph === 'B') {
        const k = L.Bk[j];
        if (k === 3) break;
        if (k === 2) {
          t -= 1;
          continue;
        }
        if (k === 1) letters.push({ node: L.Bn[j], end: L.Bb[j] });
        const pt = L.Bb[j];
        n = L.Bn[j];
        t = pt;
        ph = 'A';
      } else {
        marks.push([L.Ab[j], t]);
        const pt = L.Ab[j];
        n = L.An[j];
        t = pt;
        ph = 'B';
      }
    }
    marks.reverse();
    letters.reverse();
    const out = [];
    let mi = 0;
    for (const x of letters) {
      const own = [];
      while (mi < marks.length && marks[mi][1] <= x.end) own.push(marks[mi++]);
      if (!own.length) continue;
      let score = 0;
      let low = Infinity;
      for (const [a, b] of own) {
        const s = this.#markScore(a, b) + MARK_PEN;
        score += s;
        low = Math.min(low, s);
      }
      out.push({ pattern: NODES[x.node].pattern, start: own[0][0], end: x.end, score, low, marks: own });
    }
    return { letters: out, marks };
  }

  // Makes the letters that ended more than the lag ago final; final=true: all of them.
  #trace(final) {
    if (this.hint && this.abs <= this.hintUntil) {
      // a clear signal: the speed is about known (the fast decoder's first estimate can be a quarter off): the best of
      // the lanes near it
      let k = -1;
      for (let j = 0; j < this.lanes.length; j++) {
        if (Math.abs(Math.log(this.lanes[j].nominal / this.hint)) > Math.log(1.5)) continue;
        if (k < 0 || this.lanes[j].score > this.lanes[k].score) k = j;
      }
      if (k >= 0) this.lane = k;
    } else {
      // the lane with the best score gives the text (with some hysteresis)
      let best = this.lane;
      for (let k = 0; k < this.lanes.length; k++) if (this.lanes[k].score > this.lanes[best].score) best = k;
      if (this.lanes[best].score > this.lanes[this.lane].score + SWITCH) this.lane = best;
    }
    const L = this.lanes[this.lane];
    const { letters, marks } = this.#path(L);
    let lag = final ? 0 : this.#lag();
    // the start of a transmission, speed not yet known: the lanes need a few seconds to tell the speed apart
    const hinted = this.hint && this.abs <= this.hintUntil;
    if (!final && !hinted && marks.length && this.done - (marks[0][0] + this.s0) < START_S / this.bd && this.lastEnd < marks[0][0] + this.s0) lag = this.len;
    const until = this.len - lag;
    if (!final) this.#measureKappa(marks, Math.floor(until));
    let cut = -1; // window index after the last final letter
    for (const x of letters) {
      if (x.end > until) break;
      cut = x.end;
      if (x.end + this.s0 <= this.lastEnd) continue; // already final
      this.#final(x, L);
    }
    // word space after the last letter once the gap is long enough (and no newer mark started)
    const D = this.dotBlocks;
    const nextMark = marks.find((m) => m[0] + this.s0 > this.lastEnd);
    const gapEnd = nextMark ? nextMark[0] + this.s0 : this.s0 + this.len;
    if (!this.spaceDone && gapEnd - this.lastEnd >= this.#wordGap(D, gapEnd - this.lastEnd)) this.#space();
    if (final) {
      this.#space();
      return;
    }
    // keep the window short: start it again after the final letters when it gets long, or when it is quiet
    const quiet = !marks.some((m) => m[1] > until);
    if (cut >= 0 && (cut > CAP / 2 || this.len >= CAP - 2 * POLL)) this.#rebaseAt(cut);
    else if (quiet && this.len > 3 * lag) this.#rebaseAt(Math.max(cut, Math.floor(until)));
  }

  #final(x, L) {
    const start = x.start + this.s0;
    const end = x.end + this.s0;
    const D = L.D;
    const gap = start - this.lastEnd;
    if (!this.spaceDone && gap >= this.#wordGap(D, gap)) this.#space();
    if (gap >= 2 * D && gap <= 40 * D && this.lastEnd > -1e8) {
      this.gapMem.push({ g: gap / D, at: end, D });
      if (this.gapMem.length > 16) this.gapMem.shift();
    }
    const amps = this.#learn(x, L);
    this.#commit(x, L, start, end, amps);
  }

  // learn the mark level (follows fading) and the speed within the lane from final marks; returns the amplitudes
  #learn(x, L) {
    const D = L.D;
    // learn the mark level (follows fading) and the speed within the lane from final marks
    const amps = [];
    for (const [a, b] of x.marks) {
      const d = b - a;
      let best = 0;
      let bestH = 0;
      for (let h = 0; h < FREQ_HYPS.length; h++) {
        const r = this.PR[h][b] - this.PR[h][a];
        const im = this.PI[h][b] - this.PI[h][a];
        if (r * r + im * im > best) {
          best = r * r + im * im;
          bestH = h;
        }
      }
      // dashes tell the offset best; only clear ones (in the noise the best offset of a dash is half chance)
      if (d >= 2 * D && best / d > 4 * (1 + d * Math.min(A_CAP, this.a))) this.fOff = 0.85 * this.fOff + 0.15 * FREQ_HYPS[bestH];
      let g = 0;
      for (let t = a; t < b; t++) g += this.gain[t] || 1;
      g /= d;
      const amp2 = Math.max(0.05, best / (d * d) - 1 / d) * g; // per-block mark power (noise units), bias taken off
      amps.push(Math.max(0.02, best / (d * d)) * g);
      this.a = clamp(this.a ? 0.75 * this.a + 0.25 * amp2 : amp2, 0.3, 1e12);
      this.marksLearned++;
      const dot = d > 2 * D ? d / 3 : d;
      L.setDot(clamp(0.85 * L.D + 0.15 * dot, L.nominal / LANE_SPAN, L.nominal * LANE_SPAN));
    }
    return amps;
  }

  #commit(x, L, start, end, amps) {
    const conf = clamp(x.low / 8, 0.2, 1);
    const ch = decodePattern(x.pattern, this.symbols);
    this.pendingWord.push({ ch, pattern: x.pattern, confidence: Math.round(conf * 100) / 100, end, low: x.low, top: x.score / x.marks.length, blocks: x.marks.reduce((n, [a, b]) => n + b - a, 0), amps });
    this.pendingScore += x.score;
    this.lastEnd = end;
    this.spaceDone = false;
    this.#judgeWord(false);
  }

  // P2: the gap (in blocks) from which a word space is put. Normally 5 dots. With Farnsworth timing (character speed
  // fast, long gaps: letter gap 10 dots, word gap 25) every letter gap is above 5 dots, so every letter became a word
  // ("P D 0 I N"). The gaps between the last letters tell: gaps are split in a small and a large group; when the small
  // group (the letter gaps) is itself above 4.5 dots, the word space is put at 1.6 times that group (standard timing:
  // small group 3 dots, nothing changes). Without a second group (one word, or all gaps alike) the gaps count as letter
  // gaps when they are at least 8 dots (a standard word gap is 7).
  #wordGap(D, cur) {
    const now = this.abs;
    this.gapMem = this.gapMem.filter((x) => now - x.at < 40 / this.bd && Math.abs(Math.log(x.D / D)) < 0.25); // (not after a speed change: Farnsworth ratio unknown)
    const gs = this.gapMem.map((x) => x.g);
    if (cur / D >= 2 && cur / D <= 40) gs.push(cur / D);
    if (gs.length < 3) return 5 * D;
    gs.sort((a, b) => a - b);
    let at = -1;
    let jump = 1;
    for (let i = 1; i < gs.length; i++) if (gs[i] / gs[i - 1] > jump) { jump = gs[i] / gs[i - 1]; at = i; }
    let low;
    if (jump >= 1.6) low = gs.slice(0, at);
    else if (gs.length >= 4 && gs[0] >= 8) low = gs;
    else return 5 * D;
    const G = low[low.length >> 1];
    // (Farnsworth letter gaps are regular; the gaps between random clicks or static are not: then keep 5 dots,
    // else crackle came out as text. 18 Nov 2026, DX Station test "plain noise gives no text")
    if (low.length < 3 || low[low.length - 1] / low[0] > 1.8) return 5 * D;
    return G >= 4.5 ? Math.max(5, 1.6 * G) * D : 5 * D;
  }

  #space() {
    this.#judgeWord(true);
    this.spaceDone = true;
  }

  // The gate: a word is shown when it carries enough evidence, or Morse has been heard with confidence lately.
  #judgeWord(done) {
    const now = this.lastEnd;
    // noise makes short words of E's and T's: a word that opens the gate needs some elements, and a lone dot or dash
    // shortly after needs a clear score
    const elements = this.pendingWord.reduce((n, c) => n + c.pattern.length, 0);
    const blocks = this.pendingWord.reduce((n, c) => n + (c.blocks || 0), 0);
    // a real station keys at one strength: the marks noise happens to fit vary wildly (log spread: real 0.2-0.45,
    // noise mostly 0.85)
    const logs = this.pendingWord.flatMap((c) => c.amps || []).map(Math.log);
    const mean = logs.reduce((x, y) => x + y, 0) / (logs.length || 1);
    const spread = Math.sqrt(logs.reduce((x, y) => x + (y - mean) ** 2, 0) / (logs.length || 1));
    const strong = this.pendingScore >= WORD_MIN && elements >= 4 && this.pendingScore >= this.#elementMin() * elements && this.pendingScore >= BLOCK_MIN * blocks && spread <= AMP_SPREAD;
    const sure = strong || (now <= this.sureUntil && this.pendingScore >= (elements >= 2 ? HOLD_MIN : 2 * HOLD_MIN) && this.pendingScore >= 0.6 * BLOCK_MIN * blocks);
    if (sure && this.pendingWord.length) {
      if (strong) this.sureUntil = now + SURE_HOLD_S / this.bd;
      // a faint letter beside a strong station is a noise burst that happened to fit (an E before CQ)
      const tops = this.pendingWord.map((c) => c.top).sort((x, y) => x - y);
      const typical = tops[tops.length >> 1];
      for (const c of this.pendingWord) {
        const faint = c.low < 8 && c.low < 0.15 * typical;
        delete c.low;
        delete c.top;
        delete c.blocks;
        delete c.amps;
        if (!faint) this.out.push(c);
      }
      this.pendingWord = [];
      this.pendingScore = 0;
      this.shownInWord = true;
    }
    if (done) {
      if (this.shownInWord) this.out.push({ ch: ' ', pattern: '', confidence: 1, end: this.lastEnd });
      this.pendingWord = [];
      this.pendingScore = 0;
      this.shownInWord = false;
    }
  }

  // Evidence needed per element: fast lanes (short marks, many chances in noise) a quarter more, slow lanes (long
  // marks gather a score from noise too, and a real slow signal has plenty per element) twice as much.
  #elementMin() {
    const D = this.lanes[this.lane].nominal;
    return ELEMENT_MIN * (D < 9 ? 1.25 : D > 20 ? 2 : 1);
  }

  // New window starting at window index c (a point between letters on the best path): the blocks after it are
  // searched again from there, in every lane.
  #rebaseAt(c) {
    c = Math.max(0, Math.min(this.len, Math.floor(c)));
    const keep = this.len - c;
    const yr = this.yr.slice(c, this.len);
    const yi = this.yi.slice(c, this.len);
    const av = this.av.slice(c, this.len);
    const gain = this.gain.slice(c, this.len);
    this.s0 += c;
    this.#startWindow();
    for (let k = 0; k < keep; k++) {
      this.gain[k] = gain[k];
      this.#addBlock(yr[k], yi[k], av[k]);
    }
    // the noise level follows the band (from the blocks heard lately); only quickly downward (marks push it up)
    if (this.powN >= 200) {
      const s = Array.from(this.powRing.subarray(0, this.powN)).sort((x, y) => x - y);
      const n0 = Math.max(MIN_N0, s[Math.floor(0.2 * (s.length - 1))] / 0.223);
      this.n0 = n0 < this.n0 ? n0 : this.n0 + 0.2 * (n0 - this.n0);
    }
  }
}

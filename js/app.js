// WPM Morse Decoder, formerly CW Oor (Richard, 18 Nov 2026): the CW decoder of DX Station on the phone. Like ggmorse it listens through the
// microphone, finds the tone (or the one you tap in the waterfall) and reads the morse.
//
// Audio: microphone (echo cancellation, noise suppression and auto gain OFF: they eat a steady tone) -> AudioWorklet
// that hands mono blocks to the page -> CwDecoder (cw-decoder.js, the same code as DX Station and SatTrack, with
// the Viterbi decoder for weak signals). An AnalyserNode beside it feeds the waterfall.
// ?demo plays a made-up QSO with noise through the same chain instead of the microphone (for testing on a Mac).
import { CwDecoder, textToElements } from './cw-decoder.js';
import { signalNote, wpmDisplay } from './cw-signal.js';
import { createFront } from './cw-front.js'; // de-reverb + level front end for the microphone (proef/ has the measurements)

const VERSION = '2.0 (24 nov 2026)';
const FMIN = 250; // Hz shown in the waterfall; the decoder looks for a tone between 300 and 1100 Hz by itself
const FMAX = 1150;
const KEY = 'cwoor.';
const $ = (id) => document.getElementById(id);

// ---------- settings ----------
function get(k, d) {
  try {
    const v = localStorage.getItem(KEY + k);
    return v === null ? d : JSON.parse(v);
  } catch {
    return d;
  }
}
function put(k, v) {
  try {
    localStorage.setItem(KEY + k, JSON.stringify(v));
  } catch {
    /* private mode: not kept */
  }
}
const cfg = {
  fs: [14, 18, 24].includes(get('fs2', 18)) ? get('fs2', 18) : 18, // (fs2: the font became a normal sans with a smaller range, 18 Nov 2026; the old 'fs' is not used)
  call: get('call', ''),
  wake: get('wake', true),
  fall: get('fall', true),
  symbols: get('symbols', false),
  pitch: get('pitch', 0), // 0 = automatic, else the radio's CW pitch in Hz (19 Nov 2026)
};

// ---------- text view ----------
const textEl = $('text');
const jump = $('jump');
const guess = document.createElement('span');
guess.className = 'guess';
let follow = true;
let touchedAt = 0;
const atBottom = () => textEl.scrollHeight - textEl.scrollTop - textEl.clientHeight < 8;
const toBottom = () => (textEl.scrollTop = textEl.scrollHeight);
for (const ev of ['touchstart', 'wheel', 'pointerdown']) textEl.addEventListener(ev, () => (touchedAt = Date.now()), { passive: true });
textEl.addEventListener('scroll', () => {
  if (Date.now() - touchedAt > 1500) return;
  follow = atBottom();
  jump.hidden = follow;
});
jump.addEventListener('click', () => {
  follow = true;
  jump.hidden = true;
  toBottom();
});
const MAX_SPANS = 3000;
let saved = '';
let saveT = 0;
function addChar(c) {
  const s = document.createElement('span');
  s.textContent = c.ch;
  if (c.ch === '*' || (typeof c.confidence === 'number' && c.confidence < 0.5)) s.className = 'u';
  if (c.ch.length > 1) s.classList.add('pro');
  if (c.fixed) s.classList.add('fx');
  if (guess.parentNode === textEl) guess.before(s);
  else textEl.append(s);
  while (textEl.childNodes.length > MAX_SPANS) textEl.firstChild.remove();
  if (follow) toBottom();
  saved = (saved + c.ch).slice(-4000);
  clearTimeout(saveT);
  saveT = setTimeout(() => put('text', saved), 800);
}
function setGuess(t) {
  if (!t || !t.trim()) {
    guess.remove();
    return;
  }
  guess.textContent = t;
  if (guess.parentNode !== textEl) textEl.append(guess);
  if (follow) toBottom();
}
function plainText() {
  return (guess.parentNode === textEl ? textEl.textContent.slice(0, -guess.textContent.length) : textEl.textContent).trim();
}
// the text of last time
saved = get('text', '');
if (saved) textEl.append(document.createTextNode(saved));

// ---------- decoder ----------
const decoder = new CwDecoder({
  onChar: addChar,
  onPending: setGuess,
  symbols: cfg.symbols,
});
decoder.setVocabulary(() => (cfg.call ? [{ call: cfg.call, source: 'own' }] : []));
if (cfg.pitch) decoder.setPitch(cfg.pitch);

// ---------- audio ----------
let front = null; // cw-front.js; ?nofront in the URL switches it off (to compare)
let ctx = null;
let stream = null;
let nodes = [];
let analyser = null;
let running = false;
let fed = 0; // samples given to the decoder (for checking that none get lost)
let inPeak = 0; // loudest sample since the last display update (0..1)
let startedAt = 0;
let peakDb = -120;
let wakeLock = null;
let demo = new URLSearchParams(location.search).has('demo');

function setState(text, cls = '') {
  const s = $('state');
  s.textContent = text;
  s.className = `state ${cls}`.trim();
}
function setNote(html, cls = '') {
  const n = $('note');
  n.innerHTML = html;
  n.className = `note ${cls}`.trim();
}

// The microphone, first as clean as possible (echo cancellation, noise suppression and auto gain off), and when the
// phone refuses that, step by step with fewer wishes. rawMic tells whether the phone's own processing is off.
let rawMic = true;
async function openMic() {
  const tries = [
    { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    { echoCancellation: false },
    true,
  ];
  let last;
  for (let i = 0; i < tries.length; i++) {
    try {
      const st = await navigator.mediaDevices.getUserMedia({ audio: tries[i] });
      rawMic = i === 0;
      return st;
    } catch (e) {
      last = e;
      if (e && (e.name === 'NotAllowedError' || e.name === 'SecurityError' || e.name === 'NotFoundError')) throw e;
      await new Promise((r) => setTimeout(r, 300));
    }
  }
  throw last;
}

// The last 30 s of microphone sound, to send to the Mac and run through the decoder there (18 Nov 2026: copy over
// the air was poor; measure before changing anything).
const REC_S = 30;
let ring = null;
let ringPos = 0;
let ringFull = false;
function keep(b, rate) {
  if (!ring || ring.rate !== rate) {
    ring = new Float32Array(REC_S * rate);
    ring.rate = rate;
    ringPos = 0;
    ringFull = false;
  }
  for (let i = 0; i < b.length; i++) {
    ring[ringPos++] = b[i];
    if (ringPos === ring.length) {
      ringPos = 0;
      ringFull = true;
    }
  }
}
function wavBlob(samples, sampleRate) {
  let peak = 0;
  for (let i = 0; i < samples.length; i++) peak = Math.max(peak, Math.abs(samples[i]));
  const g = peak > 0 ? 0.9 / peak : 1; // as loud as possible without clipping
  const n = samples.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const str = (o, t) => [...t].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  str(0, 'RIFF');
  v.setUint32(4, 36 + n * 2, true);
  str(8, 'WAVE');
  str(12, 'fmt ');
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  str(36, 'data');
  v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, samples[i] * g)) * 0x7fff, true);
  return new Blob([buf], { type: 'audio/wav' });
}
async function shareRecording() {
  const btn = $('rec');
  if (!ring || (!ringFull && ringPos === 0)) return flash(btn, 'Nog geen geluid gehoord');
  const out = ringFull ? new Float32Array(ring.length) : ring.slice(0, ringPos);
  if (ringFull) {
    out.set(ring.subarray(ringPos), 0);
    out.set(ring.subarray(0, ringPos), ring.length - ringPos);
  }
  const stamp = new Date().toISOString().slice(0, 16).replace(/[-:T]/g, '');
  const file = new File([wavBlob(out, ring.rate)], `cwoor-${stamp}.wav`, { type: 'audio/wav' });
  const diag = `WPM Morse by PD0IN ${VERSION}; ${ring.rate} Hz; microfoon ${rawMic ? 'zuiver' : 'met bewerking'}; pitch ${decoder.status.pitch || '-'} ${decoder.status.locked ? 'vast' : 'auto'}`;
  const text = `${diag}\nGelezen: ${plainText().slice(-300)}`;
  try {
    if (navigator.canShare && navigator.canShare({ files: [file] })) {
      await navigator.share({ files: [file], title: file.name, text });
      return;
    }
  } catch (e) {
    if (e && e.name === 'AbortError') return;
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(file);
  a.download = file.name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
}

async function start() {
  if (running) return;
  if (!demo && !(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
    setState('fout', 'err');
    setNote('Geen microfoon in deze browser');
    return;
  }
  let step = 'geluidsverwerking';
  try {
    const AC = window.AudioContext || window.webkitAudioContext;
    // iPhone (18 Nov 2026): the audio engine is made and started right away, still inside the tap. Started later (after
    // waiting for the microphone) Safari refuses it: "Failed to start the audio device".
    ctx = new AC({ latencyHint: 'interactive' });
    const resumed = ctx.resume();
    let src;
    if (demo) {
      await resumed;
      src = ctx.createBufferSource();
      src.buffer = demoBuffer(ctx.sampleRate);
      src.loop = true;
      src.start();
      // the demo can also be heard
      const out = ctx.createGain();
      out.gain.value = 0.25;
      src.connect(out).connect(ctx.destination);
      nodes.push(out);
    } else {
      step = 'microfoon';
      stream = await openMic();
      step = 'geluidsverwerking';
      await resumed;
      if (ctx.state !== 'running') await ctx.resume();
      src = ctx.createMediaStreamSource(stream);
      for (const t of stream.getAudioTracks()) t.addEventListener('ended', () => stop('Microfoon gestopt: tik op Luister'));
    }
    step = 'decoder';
    await ctx.audioWorklet.addModule('js/cw-worklet.js');
    // Safari only runs audio nodes that lead to the speaker: the tap and the waterfall go to the output through a
    // gain of 0 (silent), else the decoder never gets a sample on the iPhone (18 Nov 2026).
    const tap = new AudioWorkletNode(ctx, 'hamlog-cw-tap', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
    const mute = ctx.createGain();
    mute.gain.value = 0;
    mute.connect(ctx.destination);
    tap.connect(mute);
    nodes.push(mute);
    let skip = Math.round(ctx.sampleRate * 0.15); // the click of the microphone switching on
    const rate = ctx.sampleRate;
    fed = 0;
    inPeak = 0;
    front = demo || new URLSearchParams(location.search).has('nofront') ? null : createFront(rate, { pitchHz: decoder.lock });
    tap.port.onmessage = (e) => {
      if (!running) return;
      const b = e.data;
      keep(b, rate);
      for (let i = 0; i < b.length; i += 4) {
        const v = Math.abs(b[i]);
        if (v > inPeak) inPeak = v;
      }
      if (skip > 0) {
        skip -= b.length;
        return;
      }
      decoder.process(front ? front.process(b) : b, rate); // after keep(): the shared recording stays raw
      fed += b.length;
    };
    analyser = ctx.createAnalyser();
    analyser.fftSize = rate > 30000 ? 8192 : 4096;
    analyser.smoothingTimeConstant = 0;
    src.connect(tap);
    src.connect(analyser);
    analyser.connect(mute);
    nodes.push(src, tap, analyser);
    decoder.resync();
    running = true;
    setState(demo ? 'demo' : 'luistert', 'on');
    if (!demo && !rawMic) setNote('Let op: microfoon met ruisonderdrukking');
    $('go').textContent = 'Stop';
    $('go').classList.add('on');
    if (cfg.wake) keepAwake();
    lastMorseAt = performance.now();
    startedAt = performance.now();
  } catch (e) {
    cleanup();
    setState('fout', 'err');
    const name = e && e.name;
    if (name === 'NotAllowedError' || name === 'SecurityError') {
      setNote('Geen toegang tot de microfoon (zie ⚙︎ › Tips)');
    } else if (name === 'NotFoundError') {
      setNote('Geen microfoon gevonden.');
    } else {
      setNote(`Starten lukte niet (${step}): ${(e && e.message) || e}`);
      $('note').title = 'Sluit andere apps die de microfoon gebruiken en tik opnieuw op Luister.';
    }
  }
}

function cleanup() {
  running = false;
  for (const n of nodes) {
    try {
      n.disconnect();
      if (n.stop) n.stop();
    } catch {
      /* already gone */
    }
  }
  nodes = [];
  analyser = null;
  if (stream) for (const t of stream.getTracks()) t.stop();
  stream = null;
  if (ctx) ctx.close().catch(() => {});
  ctx = null;
  if (wakeLock) wakeLock.release().catch(() => {});
  wakeLock = null;
}

function stop(msg) {
  decoder.flush();
  cleanup();
  setGuess('');
  setState('uit');
  $('go').textContent = 'Luister';
  $('go').classList.remove('on');
  setNote(msg || 'Gestopt');
}

async function keepAwake() {
  try {
    if ('wakeLock' in navigator && !wakeLock) {
      wakeLock = await navigator.wakeLock.request('screen');
      wakeLock.addEventListener('release', () => (wakeLock = null));
    }
  } catch {
    /* not allowed now; tried again when the page is shown */
  }
}
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && running && cfg.wake) keepAwake();
});

// A made-up QSO at 20 WPM, 650 Hz, with band noise and slow fading: for ?demo.
function demoBuffer(sr) {
  const text = 'CQ CQ DE PA3XYZ PA3XYZ K   PA3XYZ DE PD0IN PD0IN K   PD0IN DE PA3XYZ GM OM TNX FER CALL UR RST 579 579 NAME JAN QTH GRONINGEN HW? KN';
  const els = textToElements(text, 20);
  const T = els.reduce((s, e) => s + e.dur, 0) + 2;
  const out = new Float32Array(Math.round(T * sr));
  let t = 1;
  const rise = Math.round(0.004 * sr);
  for (const e of els) {
    if (e.on) {
      const a = Math.round(t * sr);
      const b = Math.round((t + e.dur) * sr);
      for (let i = a; i < b; i++) {
        const env = Math.min(1, (i - a) / rise, (b - i) / rise);
        out[i] = 0.25 * env * Math.sin((2 * Math.PI * 650 * i) / sr);
      }
    }
    t += e.dur;
  }
  // a second, weaker station higher up, so the waterfall shows two signals
  const els2 = textToElements('TEST DE DL1ABC DL1ABC TEST', 26);
  t = 4;
  for (const e of els2) {
    if (e.on) for (let i = Math.round(t * sr); i < Math.round((t + e.dur) * sr) && i < out.length; i++) out[i] += 0.12 * Math.sin((2 * Math.PI * 920 * i) / sr);
    t += e.dur;
  }
  let lp = 0;
  for (let i = 0; i < out.length; i++) {
    lp = 0.7 * lp + 0.3 * (Math.random() * 2 - 1);
    out[i] = out[i] * (0.88 + 0.12 * Math.sin((2 * Math.PI * i) / sr / 7)) + 0.08 * lp;
  }
  const buf = new AudioBuffer({ length: out.length, sampleRate: sr, numberOfChannels: 1 });
  buf.copyToChannel(out, 0);
  return buf;
}

// ---------- waterfall ----------
const fall = $('fall');
const fctx = fall.getContext('2d');
let fbins = null;
let lo = -100;
let hi = -40;
function sizeFall() {
  const r = fall.getBoundingClientRect();
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  fall.width = Math.max(100, Math.round(r.width * dpr));
  fall.height = Math.max(40, Math.round(r.height * dpr));
  fctx.fillStyle = '#05070c';
  fctx.fillRect(0, 0, fall.width, fall.height);
  const axis = $('axis');
  axis.textContent = '';
  for (let f = 300; f <= 1100; f += 200) {
    const s = document.createElement('span');
    s.style.left = `${((f - FMIN) / (FMAX - FMIN)) * 100}%`;
    s.textContent = f;
    axis.append(s);
  }
}
// dark blue -> blue -> amber -> white
function color(v) {
  v = Math.max(0, Math.min(1, v));
  const stops = [
    [0, 5, 7, 12],
    [0.35, 20, 50, 120],
    [0.65, 60, 150, 255],
    [0.85, 255, 181, 71],
    [1, 255, 250, 235],
  ];
  for (let i = 1; i < stops.length; i++) {
    if (v <= stops[i][0]) {
      const a = stops[i - 1];
      const b = stops[i];
      const k = (v - a[0]) / (b[0] - a[0]);
      return [a[1] + k * (b[1] - a[1]), a[2] + k * (b[2] - a[2]), a[3] + k * (b[3] - a[3])];
    }
  }
  return [255, 250, 235];
}
const xOf = (hz) => ((hz - FMIN) / (FMAX - FMIN)) * fall.width;
function drawFall() {
  if (!analyser || !cfg.fall || !fall.width) return;
  const n = analyser.frequencyBinCount;
  if (!fbins || fbins.length !== n) fbins = new Float32Array(n);
  analyser.getFloatFrequencyData(fbins);
  const binHz = ctx.sampleRate / analyser.fftSize;
  const w = fall.width;
  // scroll one row down
  fctx.drawImage(fall, 0, 0, w, fall.height - 1, 0, 1, w, fall.height - 1);
  const row = fctx.createImageData(w, 1);
  const vals = new Float32Array(w);
  for (let x = 0; x < w; x++) {
    const f0 = FMIN + (x / w) * (FMAX - FMIN);
    const f1 = FMIN + ((x + 1) / w) * (FMAX - FMIN);
    let m = -200;
    for (let b = Math.floor(f0 / binHz); b <= Math.ceil(f1 / binHz); b++) if (fbins[b] > m) m = fbins[b];
    vals[x] = m;
  }
  // scale: the noise floor (20th percentile) at the bottom, about 35 dB above it at the top; follows slowly
  const sorted = Float32Array.from(vals).sort();
  const floor = sorted[Math.floor(w * 0.2)];
  const top = Math.max(sorted[w - 1], floor + 25);
  lo += 0.05 * (floor - lo);
  hi += 0.05 * (Math.min(top, floor + 45) - hi);
  for (let x = 0; x < w; x++) {
    const [r, g, b] = color((vals[x] - lo - 3) / Math.max(10, hi - lo));
    row.data.set([r, g, b, 255], x * 4);
  }
  fctx.putImageData(row, 0, 0);
}
function drawMarker() {
  // the pitch the decoder listens to: an amber line over the top rows
  const st = decoder.status;
  if (!st.pitch) return;
  const x = Math.round(xOf(st.pitch));
  fctx.fillStyle = st.locked ? '#ffb547' : 'rgba(255,181,71,.6)';
  fctx.fillRect(x - 1, 0, 2, 6);
}
// the pitch from the settings: fixed (cfg.pitch) or automatic
function applyPitch(hz) {
  decoder.setPitch(hz || null);
  if (front) front.setPitch(hz || null);
  decoder.resync();
}
fall.addEventListener('click', (e) => {
  const r = fall.getBoundingClientRect();
  const hz = Math.round(FMIN + ((e.clientX - r.left) / r.width) * (FMAX - FMIN));
  applyPitch(hz);
  $('auto').textContent = cfg.pitch ? `Terug naar ${cfg.pitch} Hz` : 'Auto';
  $('auto').hidden = false;
  setNote(`Vast op ${hz} Hz`);
});
$('auto').addEventListener('click', () => {
  applyPitch(cfg.pitch);
  $('auto').hidden = true;
  setNote(cfg.pitch ? `Vast op ${cfg.pitch} Hz` : 'Toon: automatisch');
});

// ---------- tuning indicator (as in DX Station: 13 bars of 25 Hz, the middle one green = on your pitch) ----------
const TUNE_STEP = 25;
const TUNE_BARS = 13;
const half = (TUNE_BARS - 1) / 2;
const bars = Array.from({ length: TUNE_BARS }, (_, i) => {
  const b = document.createElement('i');
  b.style.height = `${Math.round(8 + 18 * (1 - Math.abs(i - half) / half))}px`;
  $('tn-bars').append(b);
  return b;
});
let tSmooth = null;
let tHit = false;
let tLast = 0;
let tLit = -1;
function tuneShow(offset, ref) {
  const idx = offset === null ? -1 : Math.max(0, Math.min(TUNE_BARS - 1, half + Math.round(offset / TUNE_STEP)));
  if (idx !== tLit) {
    if (tLit >= 0) bars[tLit].classList.remove('on', 'ok', 'far');
    tLit = idx;
  }
  if (idx >= 0) {
    bars[idx].classList.add('on');
    bars[idx].classList.toggle('ok', idx === half);
    bars[idx].classList.toggle('far', Math.abs(offset) > half * TUNE_STEP + TUNE_STEP / 2);
  }
  $('tune').classList.toggle('hit', idx === half);
  $('tn-lbl').textContent = offset === null ? `${ref} Hz` : idx === half ? `op ${ref} Hz` : `${offset > 0 ? '+' : '−'}${Math.abs(Math.round(offset))} Hz`;
}
function tuneUpdate(s, now) {
  const ref = s.locked ? s.pitch : null;
  $('tune').hidden = !ref;
  if (!ref) return;
  const raw = typeof s.toneHz === 'number' ? s.toneHz - ref : null;
  if (raw !== null) {
    tLast = now;
    if (tSmooth === null) tSmooth = raw;
    else tSmooth += (Math.abs(raw - tSmooth) > 120 ? 0.08 : 0.3) * (raw - tSmooth);
    tHit = tHit ? Math.abs(tSmooth) < 30 : Math.abs(tSmooth) < 15;
  } else if (now - tLast > 1500) {
    tSmooth = null;
    tHit = false;
  }
  tuneShow(tSmooth, ref);
}

// ---------- display loop ----------
let lastWpm = null;
let lastMorseAt = 0;
let lastStatus = 0;
function frame(now) {
  requestAnimationFrame(frame);
  if (!running) return;
  drawFall();
  drawMarker();
  if (now - lastStatus < 150) return;
  lastStatus = now;
  const s = decoder.status;
  const wd = wpmDisplay(s, lastWpm);
  lastWpm = wd.last;
  $('wpm').textContent = wd.text;
  $('wpm').classList.toggle('stale', wd.stale);
  $('pitch').textContent = s.pitch || s.toneHz || '---';
  $('pitch-l').textContent = s.locked ? 'Hz vast' : 'Hz auto';
  $('snr').textContent = s.snrDb ? Math.round(s.snrDb) : '--';
  $('level').style.width = `${Math.round(s.level * 100)}%`;
  $('tone').classList.toggle('on', Boolean(s.toneOn));
  tuneUpdate(s, now);
  if (s.morse) lastMorseAt = now;
  const note = signalNote(s, { quietSecs: (now - lastMorseAt) / 1000 });
  peakDb = inPeak > 0 ? Math.max(-120, 20 * Math.log10(inPeak)) : -120;
  inPeak = 0;
  // no sound coming in at all: say so instead of "alleen ruis"
  if (!demo && now - startedAt > 2500) {
    if (fed === 0) {
      setNote('Geen geluid van de microfoon: tik Stop en Luister');
      return;
    }
    if (peakDb < -85) {
      setNote('Microfoon stil: andere app of AirPods?');
      return;
    }
  }
  if (now < Number($('note').dataset.hold || 0)) return;
  // one short line, no explanations, so the text below never moves (Richard, 19 Nov 2026)
  if (note.kind === 'morse') setNote('Morse gehoord', 'morse');
  else setNote('Geen morse, alleen ruis');
}
requestAnimationFrame(frame);

// keep a user message (tap, auto) on screen for a few seconds before the signal note takes over again
function noteHold(ms) {
  $('note').dataset.hold = String(performance.now() + ms);
}
for (const id of ['fall', 'auto']) $(id).addEventListener('click', () => noteHold(3000));

// ---------- buttons ----------
$('go').addEventListener('click', () => (running ? stop() : start()));
$('rec').addEventListener('click', shareRecording);
$('clear').addEventListener('click', () => {
  textEl.textContent = '';
  saved = '';
  put('text', '');
  follow = true;
  jump.hidden = true;
});
function flash(btn, t) {
  const old = btn.textContent;
  btn.textContent = t;
  setTimeout(() => (btn.textContent = old), 1200);
}
$('copy').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(plainText());
    flash($('copy'), 'Gekopieerd');
  } catch {
    flash($('copy'), 'Lukt niet');
  }
});
$('share').addEventListener('click', async () => {
  const text = plainText();
  if (!text) return flash($('share'), 'Leeg');
  if (navigator.share) {
    try {
      await navigator.share({ title: 'WPM Morse by PD0IN', text });
    } catch {
      /* cancelled */
    }
  } else {
    await navigator.clipboard.writeText(text).catch(() => {});
    flash($('share'), 'Gekopieerd');
  }
});

// ---------- settings ----------
const dlg = $('settings');
function applyCfg() {
  document.documentElement.style.setProperty('--fs', `${cfg.fs}px`);
  if ($('fs')) $('fs').querySelectorAll('[data-fs]').forEach((b) => b.classList.toggle('on', Number(b.dataset.fs) === cfg.fs));
  $('fall-box').hidden = !cfg.fall;
  if (cfg.fall) requestAnimationFrame(sizeFall);
}
$('cog').addEventListener('click', () => {
  showPitchCfg();
  $('call').value = cfg.call;
  $('wake').checked = cfg.wake;
  $('showfall').checked = cfg.fall;
  $('symbols').checked = cfg.symbols;
  const diag = running ? ` Audio: ${ctx.state}, ${ctx.sampleRate} Hz, ${Math.round(fed / ctx.sampleRate)} s gelezen, piek ${Math.round(peakDb)} dBFS, microfoon ${rawMic ? 'zuiver' : 'met bewerking'}.` : '';
  $('ver').textContent = `WPM Morse by PD0IN ${VERSION}. Decoder uit DX Station V3.${diag}`;
  dlg.showModal();
});
// Small / Medium / Large (Richard, 18 Nov 2026)
$('fs').querySelectorAll('[data-fs]').forEach((b) =>
  b.addEventListener('click', () => {
    cfg.fs = Number(b.dataset.fs);
    put('fs2', cfg.fs);
    applyCfg();
  }),
);
$('call').addEventListener('input', () => {
  cfg.call = $('call').value.trim().toUpperCase();
  put('call', cfg.call);
  decoder.setVocabulary(() => (cfg.call ? [{ call: cfg.call, source: 'own' }] : []));
});
$('wake').addEventListener('change', () => {
  cfg.wake = $('wake').checked;
  put('wake', cfg.wake);
  if (cfg.wake && running) keepAwake();
  else if (wakeLock) wakeLock.release().catch(() => {});
});
$('showfall').addEventListener('change', () => {
  cfg.fall = $('showfall').checked;
  put('fall', cfg.fall);
  applyCfg();
});
$('symbols').addEventListener('change', () => {
  cfg.symbols = $('symbols').checked;
  put('symbols', cfg.symbols);
  decoder.symbols = cfg.symbols;
  if (decoder.vit) decoder.vit.symbols = cfg.symbols;
});

// for checking in the browser console: is all audio reaching the decoder?
window.cwOor = { decoder, health: () => ({ fed, seconds: ctx ? ctx.currentTime : 0, rate: ctx ? ctx.sampleRate : 0, peakDb }) };

// Toonhoogte: Automatisch / Vast op ... Hz
function showPitchCfg() {
  $('pmode').querySelectorAll('[data-p]').forEach((b) => b.classList.toggle('on', (b.dataset.p === 'vast') === Boolean(cfg.pitch)));
  $('phz-row').hidden = !cfg.pitch;
  if (cfg.pitch) $('phz').value = cfg.pitch;
}
function setCfgPitch(hz) {
  cfg.pitch = hz ? Math.max(300, Math.min(1100, Math.round(hz))) : 0;
  put('pitch', cfg.pitch);
  applyPitch(cfg.pitch);
  $('auto').hidden = true;
  showPitchCfg();
}
$('pmode').querySelectorAll('[data-p]').forEach((b) =>
  b.addEventListener('click', () => setCfgPitch(b.dataset.p === 'vast' ? Number($('phz').value) || cfg.pitch || 700 : 0)),
);
$('phz').addEventListener('change', () => {
  if (Number($('phz').value) >= 300) setCfgPitch(Number($('phz').value));
});

window.addEventListener('resize', () => cfg.fall && sizeFall());
applyCfg();
sizeFall();
if (demo) setNote('Demo: tik op Luister');

if ('serviceWorker' in navigator && location.protocol === 'https:') navigator.serviceWorker.register('sw.js').catch(() => {});

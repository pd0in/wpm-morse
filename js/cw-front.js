// CW Oor front end: pre-processing of the microphone signal before CwDecoder.process (mobile app only).
// The decoder (cw-decoder.js, shared with DX Station) is tuned for clean USB audio. A phone microphone hearing the
// radio's loudspeaker adds things it handles badly (measured in proef/bench.mjs):
//   1. room reverb: the tail of every mark fills the gap behind it (dots merge into dashes, letters smear),
//   2. level changes (phone AGC pumping, hand moving) that confuse the adaptive threshold,
//   3. other tones/voices in the room and on the band.
// Per sample, in order:
//   - pitch tracking (Goertzel bank, smoothed, hysteresis) unless a pitch is given (setPitch),
//   - band-pass around the tone (two equal biquads),
//   - de-reverberation in the envelope domain: tone power P (I/Q mixed at the pitch, 1 kHz rate) = direct power S + late
//     reverb T, with T a smoothed copy of S:  T[n+1] = d*T[n] + (1-d)*rho*S[n],  S = max(P-T, 0).
//     The gain sqrt(S/P) (>= gmin) multiplies the audio, so phase and noise of the signal are untouched.
//   - slow level normalisation (peak follower on the envelope).
// Cost: a few multiplications per sample plus the Goertzel bank (tens of flops per sample).

const TWO_PI = 2 * Math.PI;

export const FRONT_DEFAULTS = {
  bp: false, // off by default: measured no gain, and it adds noise-only artefacts (see proef/)
  bpWidth: 160, // Hz, -3 dB width of each of the two stages
  derev: true,
  envHz: 120, // Hz, bandwidth of the I/Q envelope used for the de-reverb decision
  gexp: 1, // gain = ((S+N)/P)^gexp; 0.5 = amplitude of the direct part, larger = harder gating
  rt60: 0.4, // s, assumed decay time (60 dB)
  rhoGain: 1, // factor on the measured rho (1.2 = a bit harder)
  rho: 0.5, // late-reverb power / direct power in steady state (0.3 mild room ... 1 bad); 'auto' = measured from the tails
  // behind long marks, EXPERIMENTAL: too few measurements and biased low (see proef/rho.mjs), not used
  gmin: 0.05,
  norm: true,
  normRelease: 1.5, // s
  normAttack: 0.05, // s
  target: 0.3, // output level of the tone peak
  gainMax: 3, // level normalisation: a fading signal is boosted by at most this factor (amplitude)
  normRise: 0.002, // per ms, how fast the gain rises (about 0.5 s)
  normFall: 0.01, // per ms, how fast the gain drops (about 100 ms)
  holdMs: 800, // keep acting this long after the last strong signal
  minPitch: 300,
  maxPitch: 1100,
  pitchHz: null, // lock
};

export function createFront(sr, options = {}) {
  const o = { ...FRONT_DEFAULTS, ...options };
  // ---------- pitch tracker ----------
  const STEP = 20;
  const freqs = [];
  for (let f = o.minPitch; f <= o.maxPitch; f += STEP) freqs.push(f);
  const nb = freqs.length;
  const coef = Float64Array.from(freqs, (f) => 2 * Math.cos((TWO_PI * f) / sr));
  const g1 = new Float64Array(nb), g2 = new Float64Array(nb), smooth = new Float64Array(nb);
  const WIN = Math.round(0.064 * sr);
  const winDur = WIN / sr;
  const SM = Math.exp(-winDur / 1.2);
  let wet = 0; // 0..1 crossfade from raw to processed after the tone was found (no click)
  let wn = 0, locked = o.pitchHz || null, pitch = 0, cand = 0, candSince = 0, tNow = 0;

  function pick() {
    let mx = 0, mi = 0;
    for (let k = 0; k < nb; k++) {
      const p = g1[k] * g1[k] + g2[k] * g2[k] - coef[k] * g1[k] * g2[k];
      smooth[k] = SM * smooth[k] + (1 - SM) * p;
      g1[k] = g2[k] = 0;
      if (smooth[k] > mx) { mx = smooth[k]; mi = k; }
    }
    tNow += winDur;
    if (locked) return;
    const med = Array.from(smooth).sort((a, b) => a - b)[nb >> 1] || 1e-12;
    if (mx < 12 * med) return;
    let f = freqs[mi];
    if (mi > 0 && mi < nb - 1) {
      const a = Math.log(smooth[mi - 1] + 1e-20), b = Math.log(smooth[mi] + 1e-20), c = Math.log(smooth[mi + 1] + 1e-20);
      const den = a - 2 * b + c;
      if (den < 0) f += (STEP * 0.5 * (a - c)) / den;
    }
    if (!pitch) { pitch = f; return; }
    if (Math.abs(f - pitch) < 25) { pitch += 0.1 * (f - pitch); cand = 0; return; }
    const cur = smooth[Math.max(0, Math.min(nb - 1, Math.round((pitch - o.minPitch) / STEP)))];
    if (mx > 2.5 * cur) {
      if (!cand || Math.abs(f - cand) > 25) { cand = f; candSince = tNow; }
      else if (tNow - candSince > 1.0) { pitch = f; cand = 0; }
    } else cand = 0;
  }

  // ---------- band-pass (two stages) ----------
  let b0 = 0, b2 = 0, a1 = 0, a2 = 0, bpFor = 0;
  const z = [0, 0, 0, 0, 0, 0, 0, 0]; // per stage: x1 x2 y1 y2
  function bpSet(f) {
    const q = f / o.bpWidth, w0 = (TWO_PI * f) / sr, al = Math.sin(w0) / (2 * q), a0 = 1 + al;
    b0 = al / a0; b2 = -al / a0; a1 = (-2 * Math.cos(w0)) / a0; a2 = (1 - al) / a0;
    bpFor = f;
  }

  // ---------- envelope / de-reverb / level (1 kHz) ----------
  const BL = Math.max(8, Math.round(sr / 1000));
  const bdt = BL / sr;
  const lpa = 1 - Math.exp(-TWO_PI * o.envHz * bdt);
  const dPow = Math.exp((-13.8 * bdt) / o.rt60);
  const kAtt = Math.exp(-bdt / o.normAttack), kRel = Math.exp(-bdt / o.normRelease);
  let pc = 1, ps = 0, dc = 1, ds = 0, phaseF = 0, bi = 0, bq = 0, bn = 0;
  const refDecay = Math.exp(-bdt / 15);
  let lastSig = -1e9, ref = 0, noise = 0, l1i = 0, l1q = 0, l2i = 0, l2q = 0, T = 0, gEnd = 1, gStart = 1, peak = 0, nEnd = 1, nStart = 1, nSlow = 1, bpos = 0;

  // ----- reverb measurement -----
  // Behind a long mark (a dash) the power falls from (1+rho)*S to the tail rho*S within a few ms and then decays slowly.
  // So 12 ms after the fall: P/Ppre = rho/(1+rho) * exp(-13.8*0.012/rt60), which gives rho. Clean audio gives ~0, a small
  // room 0.2-0.5, a bad one 1-2. The median of the last 9 measurements is used (none before 3 measurements: rho = 0).
  const RING = 128;
  const hist = new Float64Array(RING);
  let ms = 0, probeAt = -1, probePre = 0, probeP6 = 0, rhoAuto = 0;
  const estimates = [];
  const tail12 = Math.exp((-13.8 * 0.012) / o.rt60);
  const at = (back) => hist[(ms - back) & (RING - 1)];
  function tailProbe(P, N) {
    hist[ms & (RING - 1)] = P;
    ms++;
    if (probeAt < 0) {
      if (ms < 80) return;
      let pre = 0;
      for (let k = 2; k <= 9; k++) pre = Math.max(pre, at(k));
      if (P > 0.6 * pre || pre < 20 * N) return;
      // a long, steady mark before the fall: the 50 ms before it all above 60 % of its level
      let mn = Infinity;
      for (let k = 10; k <= 60; k++) mn = Math.min(mn, at(k));
      if (mn < 0.6 * pre) return;
      probeAt = ms + 12; probePre = pre; probeP6 = P;
      return;
    }
    if (ms < probeAt) return;
    probeAt = -1;
    const r = P / probePre;
    if (r > 0.5 || P > 1.1 * probeP6) return; // no real fall, or a new mark started
    if (o.debug) o.debug.push(r);
    const x = Math.min(0.7, Math.max(0, r / tail12));
    estimates.push(x / (1 - x));
    if (estimates.length > 9) estimates.shift();
    if (estimates.length >= 3) rhoAuto = estimates.slice().sort((p, q) => p - q)[estimates.length >> 1];
    ms += 40; // do not measure the same fall twice
  }
  function rhoNow() { return o.rho === 'auto' ? Math.min(2, rhoAuto * o.rhoGain) : o.rho; }

  function block() {
    const nr = Math.hypot(pc, ps); pc /= nr; ps /= nr; // keep the phasor at length 1
    const I = bi / bn, Q = bq / bn;
    bi = bq = bn = 0;
    l1i += lpa * (I - l1i); l1q += lpa * (Q - l1q);
    l2i += lpa * (l1i - l2i); l2q += lpa * (l1q - l2q);
    const P = l2i * l2i + l2q * l2q + 1e-14;
    // noise floor of the envelope power: quantile tracker (25 %), scaled to the mean of an exponential distribution
    if (!noise) noise = P;
    noise *= 1 + 0.0006 * (0.25 - (P < noise ? 1 : 0));
    const N = noise / 0.288;
    tailProbe(P, N);
    // only act while a station is there: noise-only stretches go through untouched (gain modulation of pure noise makes
    // the decoder see false marks)
    if (P > 8 * N) lastSig = ms;
    const active = ms - lastSig < o.holdMs;
    let g = 1;
    if (o.derev && active) {
      // reverb is modelled on the power above the noise floor, so noise-only blocks are left alone
      const S = Math.max(P - N - T, 0);
      T = dPow * T + (1 - dPow) * rhoNow() * S;
      g = Math.min(1, Math.max(o.gmin, Math.pow((S + N) / P, o.gexp)));
    }
    if (o.trace) o.trace.push([P, N, T, g]);
    gStart = gEnd;
    gEnd += (g > gEnd ? 0.7 : 0.35) * (g - gEnd);
    nStart = nEnd;
    if (o.norm) {
      const e = Math.sqrt(P) * gEnd * 2; // amplitude of the tone
      peak = e > peak ? kAtt * peak + (1 - kAtt) * e : kRel * peak + (1 - kRel) * e;
      // reference level: the highest recent peak (decays in ~15 s); the gain only boosts a fading signal back to it, so a
      // new loud station never meets an old high gain
      ref = Math.max(peak, ref * refDecay);
      const want = active && peak > 1e-9 ? Math.min(Math.max(ref / peak, 1), o.gainMax) : 1;
      nSlow += (want < nSlow ? o.normFall : o.normRise) * (want - nSlow);
      nEnd = nSlow;
    }
    bpos = 0;
  }

  function process(x) {
    for (let i = 0; i < x.length; i++) {
      const v = x[i];
      if (!locked) { // the bank is only needed to find the pitch
        for (let k = 0; k < nb; k++) {
          const s = v + coef[k] * g1[k] - g2[k];
          g2[k] = g1[k]; g1[k] = s;
        }
        if (++wn >= WIN) { wn = 0; pick(); }
      }
      const f = locked || pitch;
      if (!f) continue; // no tone yet: pass through
      if (Math.abs(f - bpFor) > 0.5) bpSet(f);
      if (f !== phaseF) { phaseF = f; dc = Math.cos((TWO_PI * f) / sr); ds = Math.sin((TWO_PI * f) / sr); }
      const nc = pc * dc - ps * ds;
      ps = pc * ds + ps * dc; pc = nc; // rotating phasor: no trig per sample
      bi += v * pc; bq += v * ps;
      if (++bn >= BL) block();
      let y = v;
      if (o.bp) {
        let t = b0 * y + b2 * z[1] - a1 * z[2] - a2 * z[3];
        z[1] = z[0]; z[0] = y; z[3] = z[2]; z[2] = t; y = t;
        t = b0 * y + b2 * z[5] - a1 * z[6] - a2 * z[7];
        z[5] = z[4]; z[4] = y; z[7] = z[6]; z[6] = t; y = t;
      }
      const fr = bn / BL;
      if (o.derev) y *= gStart + (gEnd - gStart) * fr;
      if (o.norm) y *= nStart + (nEnd - nStart) * fr;
      wet = Math.min(1, wet + 1 / (0.15 * sr));
      x[i] = v + (y - v) * wet;
    }
    return x;
  }
  return {
    process,
    setPitch(hz) { locked = hz || null; if (locked) pitch = 0; },
    get rho() { return o.rho === 'auto' ? rhoAuto : o.rho; },
    get pitch() { return locked || pitch || null; },
  };
}

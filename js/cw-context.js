// Context for weak CW (Richard, 5 Nov 2026): what a trained ear does. A word the decoder heard with one element
// wrong ("TNO", "FIR", "CEALZ") is compared with the words that come up in nearly every CW QSO and with the calls
// HamLog knows (spots on the frequency, the call in the log line, your own call). When one of them is a single
// element away (a dot read as a dash, one element missed or one too many, a letter gap in the wrong place) and no
// other word is as close, the known word is shown instead, marked so you can see it was corrected (the tooltip says
// what was heard). Short words and words that are not close to anything are left as they were: the context may
// help the text, never invent it. Also: the report the other station gives you (5NN, 599, 579) for the log line.
// Pure, for the tests.
import { MORSE } from './cw-decoder-tables.js';

// Words that come up in nearly every CW QSO (and RST reports).
export const QSO_WORDS = [
  'CQ', 'DE', 'TU', 'TNX', 'TKS', 'FER', 'CALL', 'RST', 'UR', 'NAME', 'QTH', 'HW', 'BK', 'GM', 'GA', 'GE', 'GN', 'FB', 'ES', 'OM', 'YL', 'OP',
  'RIG', 'ANT', 'PWR', 'WX', 'TEMP', 'PSE', 'AGN', 'QRZ', 'QSL', 'QRS', 'QRQ', 'QSB', 'QRM', 'QRN', 'QSY', 'POTA', 'SOTA', 'WWFF', 'TEST',
  'RPRT', 'CFM', 'CUL', 'GL', 'GUD', 'VY', 'DR', 'HR', 'HI', 'SRI', 'COPY', 'ABT', 'WATTS', 'BEST', 'MNI', 'AGE', 'EL', 'DIPOLE', 'VERTICAL',
  'BEAM', 'SUNNY', 'CLOUDY', 'RAIN', 'CLDY', 'FINE', 'NICE', 'QSO', 'BCNU', 'SIG', 'LOC', 'GRID', 'REF', 'PARK', 'NR', 'ALL', 'AND', 'THE',
  'FOR', 'ON', 'IS', 'MY', 'YOUR', 'HERE', 'THERE', 'SO', 'OK', 'TRX', 'KW', 'EFHW', 'DX', 'CONDX',
  // C2: words and prosigns of POTA/SOTA/contest overs, so they are never 'corrected' into a neighbour (UP became UR)
  'UP', 'DN', 'QRP', 'QRO', 'QRL', 'QRT', 'CPY', 'LOOP', 'R', 'K', 'KN', 'AR', 'SK', 'ENN',
  '5NN', '599', '589', '579', '569', '559', '549', '539', '529', '519', '55N', '57N', '58N', '59N', '73', '72', '88',
];

// A word as its elements with '|' between the letters ("DE" -> "-..|."); null when a character has no code.
export function wordPattern(word) {
  const parts = [];
  for (const ch of String(word || '').toUpperCase()) {
    const p = MORSE[ch];
    if (!p) return null;
    parts.push(p);
  }
  return parts.join('|');
}

// Edit distance between two element strings (a '|' counts like an element: a letter gap read in the wrong place).
export function elementDistance(a, b, cap = 9) {
  const n = a.length;
  const m = b.length;
  if (Math.abs(n - m) > cap) return cap + 1;
  let prev = Array.from({ length: m + 1 }, (_, j) => j);
  for (let i = 1; i <= n; i++) {
    const cur = [i];
    let low = i;
    for (let j = 1; j <= m; j++) {
      const v = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
      cur.push(v);
      if (v < low) low = v;
    }
    if (low > cap) return cap + 1;
    prev = cur;
  }
  return prev[m];
}

// The vocabulary for one moment: the QSO words plus the calls given ([{call, source}] or strings).
export function buildVocabulary(calls = []) {
  const out = new Map();
  for (const w of QSO_WORDS) {
    const p = wordPattern(w);
    if (p) out.set(w, { word: w, pattern: p, call: false });
  }
  for (const c of calls) {
    const w = String((c && c.call) || c || '').toUpperCase();
    if (w.length < 3 || out.has(w)) continue;
    const p = wordPattern(w);
    if (p) out.set(w, { word: w, pattern: p, call: true, spot: Boolean(c && c.source === 'spot') });
  }
  return out;
}

// A heard word (its characters [{ch, pattern, confidence}]) against the vocabulary. Returns the known word to show
// instead, or null. One element off is allowed for words of at least 7 elements (two for calls on the frequency of
// at least 14), and only when no other word is equally close.
export function correctWord(chars, vocab) {
  const text = chars.map((c) => c.ch).join('');
  if (!text || vocab.has(text) || chars.some((c) => !c.pattern || c.ch.length > 1)) return null; // (prosigns stay)
  // (C1, skipping confident words, was measured and left out for CW Oor, 18 Nov 2026: over the air, with room
  // reverb, wrong words are confident too, and the own call was found less often.)
  // C3: a number of digits only (449, 579, 123) is a report or a serial number, never 'corrected'
  if (/^[0-9]+$/.test(text)) return null;
  const heard = chars.map((c) => c.pattern).join('|');
  if (heard.length < 7) return null; // E, T, RM, PT...: too short to tell
  let best = null;
  let bestD = Infinity;
  let second = Infinity;
  for (const v of vocab.values()) {
    if (Math.abs(v.pattern.length - heard.length) > 2) continue;
    const allow = v.call && v.spot && v.pattern.length >= 14 ? 2 : 1;
    const d = elementDistance(heard, v.pattern, 3);
    if (d > allow) continue;
    if (d < bestD) {
      second = bestD;
      bestD = d;
      best = v;
    } else if (d < second) second = d;
  }
  if (!best || bestD === 0 || second <= bestD) return null;
  if (best.pattern.length < 6) return null;
  return best.word;
}

// The report the other station gives: "UR RST 579", "UR 5NN", "RST 559 559", "TU 599". words: the last words heard
// (strings). Returns '599' style (N = 9) or null.
export function heardReport(words) {
  for (let i = words.length - 1; i >= 0; i--) {
    const w = String(words[i] || '').toUpperCase();
    if (!/^5[1-9N][1-9N]$/.test(w)) continue;
    const before = words.slice(Math.max(0, i - 3), i).map((x) => String(x).toUpperCase());
    // after UR / RST / TU, or given twice ("559 559")
    if (before.some((b) => b === 'UR' || b === 'RST' || b === 'TU' || b === 'RPRT') || before[before.length - 1] === w || words[i + 1] === w) {
      return w.replace(/N/g, '9');
    }
  }
  return null;
}

// The report YOU send (Richard, 14 Nov 2026: "als ik hem mijn signaalrapport stuur, neem jij die waarde niet over"):
// in your own sent text "UR RST 579 579", "TU 5NN", "UR 449", "RST ENN". Readability 1-5 (E = cut 5), N = 9.
// Returns '579' style or null. The last report in the text wins.
export function sentReport(text) {
  const words = String(text || '').toUpperCase().split(/[\s=]+/).filter(Boolean);
  for (let i = words.length - 1; i >= 0; i--) {
    const w = words[i];
    if (!/^[1-5E][1-9N][1-9N]$/.test(w)) continue;
    const before = words.slice(Math.max(0, i - 3), i);
    if (before.some((b) => b === 'UR' || b === 'RST' || b === 'TU' || b === 'RPRT') || before[before.length - 1] === w || words[i + 1] === w) {
      return w.replace(/^E/, '5').replace(/N/g, '9');
    }
  }
  return null;
}

// What the CW pane tells about the signal it hears (Richard, 6 Nov 2026): morse, a tone without morse, noise, or a radio
// that is not in CW at all (SSB: a voice is not morse). Pure, for the tests; the pane (cw-pane.js) shows it.
//
// Honest limit: the decoder cannot tell a voice from other noise. "Geluid, maar geen CW" is said when there is sound
// with a clear level but no morse for a while; whether it is a voice (SSB), a carrier or interference is for you to hear.

export const NO_CW_AFTER_S = 8; // seconds of sound without morse before it is called "geen CW"

// s: the decoder's status {morse, level (0-1), snrDb}. ctx: {quietSecs: seconds since morse was last heard,
// radio: {cw: bool, name: 'USB'} (optional: where the radio is)}. Returns {kind, text}; text '' = nothing to say.
export function signalNote(s, ctx = {}) {
  const radio = ctx.radio;
  if (radio && radio.cw === false) {
    return { kind: 'mode', text: `De radio staat op ${radio.name || 'een andere stand'}, niet op CW. De decoder leest alleen CW, geen stem (SSB).` };
  }
  if (s && s.morse) return { kind: 'morse', text: '' };
  const quiet = Number(ctx.quietSecs) || 0;
  const sound = Boolean(s) && s.level > 0.05 && s.snrDb > 9;
  if (sound && quiet >= NO_CW_AFTER_S) return { kind: 'nocw', text: 'Geluid, maar geen CW-signaal. Is dit een stem (SSB) of storing? Stem af op een CW-station.' };
  if (sound) return { kind: 'tone', text: 'Toon gehoord, nog geen morse' };
  return { kind: 'noise', text: 'Geen morse, alleen ruis' };
}

// The speed as shown: the measured WPM while there is morse; without morse the last measured one, dimmed (stale).
// Returns {text, stale, last}: pass `last` back in next time.
export function wpmDisplay(s, last = null) {
  const w = s && typeof s.wpm === 'number' && s.wpm > 0 ? Math.round(s.wpm) : null;
  if (s && s.morse && w) return { text: String(w), stale: false, last: w };
  return { text: last ? String(last) : '--', stale: Boolean(last), last };
}

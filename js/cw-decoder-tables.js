// The Morse tables and decodePattern, shared by cw-decoder.js and cw-viterbi.js.

const LETTERS = {
  A: '.-', B: '-...', C: '-.-.', D: '-..', E: '.', F: '..-.', G: '--.', H: '....', I: '..', J: '.---', K: '-.-', L: '.-..', M: '--',
  N: '-.', O: '---', P: '.--.', Q: '--.-', R: '.-.', S: '...', T: '-', U: '..-', V: '...-', W: '.--', X: '-..-', Y: '-.--', Z: '--..',
};
const DIGITS = { 0: '-----', 1: '.----', 2: '..---', 3: '...--', 4: '....-', 5: '.....', 6: '-....', 7: '--...', 8: '---..', 9: '----.' };
const PUNCT = {
  '.': '.-.-.-', ',': '--..--', '?': '..--..', '/': '-..-.', '=': '-...-', '+': '.-.-.', '-': '-....-', '@': '.--.-.', "'": '.----.',
  ':': '---...', ')': '-.--.-', '"': '.-..-.', '!': '-.-.--', ';': '-.-.-.', _: '..--.-', $: '...-..-',
};
const PROSIGNS = {
  '<AR>': '.-.-.', '<SK>': '...-.-', '<BT>': '-...-', '<KN>': '-.--.', '<BK>': '-...-.-', '<AS>': '.-...', '<CT>': '-.-.-', '<HH>': '........',
};
// Characters that share their pattern with a prosign: '+' = <AR>, '=' = <BT>. They are shown as the prosign by
// default (the usual way on the air); {symbols: true} shows them as the punctuation mark.
const SHARED = { '.-.-.': '+', '-...-': '=' };

export const MORSE = { ...LETTERS, ...DIGITS, ...PUNCT };
export const MORSE_PROSIGNS = PROSIGNS;

const DECODE = new Map();
for (const [ch, p] of Object.entries({ ...LETTERS, ...DIGITS, ...PUNCT })) if (!SHARED[p]) DECODE.set(p, ch);
const DECODE_PROSIGN = new Map(Object.entries(PROSIGNS).map(([name, p]) => [p, name]));

// One pattern of '.' and '-' to its text; '*' when it is not in the table.
export function decodePattern(pattern, symbols = false) {
  if (symbols && SHARED[pattern]) return SHARED[pattern];
  return DECODE_PROSIGN.get(pattern) || DECODE.get(pattern) || '*';
}


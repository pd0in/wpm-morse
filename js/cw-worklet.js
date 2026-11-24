// AudioWorklet for the CW decoder (see cw-audio.js): hands the microphone / sound card samples to the main thread
// in blocks of 1024 mono samples. No processing here; the decoder (cw-decoder.js) runs in the page, and works with
// sample counts, not with clock time, so a busy page does not disturb the timing.
class HamlogCwTap extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(1024);
    this.n = 0;
  }

  process(inputs) {
    const chans = inputs[0];
    if (!chans || !chans.length) return true;
    const first = chans[0];
    for (let i = 0; i < first.length; i++) {
      let v = first[i];
      for (let c = 1; c < chans.length; c++) v += chans[c][i];
      this.buf[this.n++] = chans.length > 1 ? v / chans.length : v; // mix down to mono
      if (this.n === this.buf.length) {
        this.port.postMessage(this.buf, [this.buf.buffer]);
        this.buf = new Float32Array(1024);
        this.n = 0;
      }
    }
    return true;
  }
}

registerProcessor('hamlog-cw-tap', HamlogCwTap);

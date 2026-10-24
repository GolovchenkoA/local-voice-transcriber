// Resamples microphone audio to 16 kHz mono and posts ~80 ms blocks of 16-bit PCM.
class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.step = sampleRate / 16000; // input samples per output sample
    this.pos = 0;                   // read position relative to the next block's first sample
    this.prev = 0;                  // last sample of the previous block
    this.out = new Int16Array(1280);
    this.n = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch || ch.length === 0) return true;
    const len = ch.length;
    let p = this.pos;
    while (true) {
      const i0 = Math.floor(p);
      if (i0 + 1 > len - 1) break;
      const a = i0 < 0 ? this.prev : ch[i0];
      const v = a + (ch[i0 + 1] - a) * (p - i0);
      this.out[this.n++] = Math.max(-1, Math.min(1, v)) * 32767;
      if (this.n === this.out.length) {
        this.port.postMessage(this.out.buffer, [this.out.buffer]);
        this.out = new Int16Array(1280);
        this.n = 0;
      }
      p += this.step;
    }
    this.pos = p - len;
    this.prev = ch[len - 1];
    return true;
  }
}

registerProcessor("pcm-processor", PCMProcessor);

"""Test the engine without a browser: transcribe a 16-bit mono WAV file in streaming mode."""

import argparse
import os
import queue
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from app.engine import SAMPLE_RATE, Engine  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", help="16 kHz, mono, 16-bit WAV file")
    parser.add_argument("--model-dir", default=str(ROOT / "models" / "nemotron-speech-streaming-en-0.6b"))
    parser.add_argument("--latency", type=int, default=1120, choices=[80, 160, 320, 560, 1120])
    parser.add_argument("--language", default=None, help="multilingual model only, e.g. pl-PL, uk-UA, ru-RU, auto")
    args = parser.parse_args()

    with wave.open(args.wav, "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            sys.exit("Need a 16 kHz, mono, 16-bit WAV file.")
        pcm = w.readframes(w.getnframes())

    model_dir = Path(args.model_dir)
    engine = Engine(str(model_dir.parent), model_dir.name, latency_ms=args.latency)
    language = engine.resolve_language(args.language)
    print(f"Device: {engine.gpu_name}, model: {engine.model_id} ({engine.kind})")

    q: queue.Queue = queue.Queue()
    block = SAMPLE_RATE * 2 // 12  # about 80 ms
    for i in range(0, len(pcm), block):
        q.put(pcm[i : i + block])
    q.put(None)

    start = time.perf_counter()
    engine.transcribe(q, lambda t: print(t, end="", flush=True), language)
    took = time.perf_counter() - start
    seconds = len(pcm) / 2 / SAMPLE_RATE
    print(f"\n\nAudio {seconds:.1f}s, processed in {took:.1f}s (real-time factor {took / seconds:.2f})")


if __name__ == "__main__":
    main()

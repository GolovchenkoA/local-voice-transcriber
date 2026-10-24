"""Speech-to-text engine: loads, switches and runs streaming and offline ASR models on CUDA."""

import gc
import queue
import threading
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForRNNT, AutoModelForTDT, AutoProcessor, TextIteratorStreamer

SAMPLE_RATE = 16000
# Offline (non-streaming) models keep the whole recording in memory and transcribe it on Stop,
# so the recording length is limited.
OFFLINE_MAX_SECONDS = 300
ALLOWED_LATENCIES_MS = (80, 160, 320, 560, 1120)

# Locales of the multilingual model that NVIDIA lists as "transcription-ready" or "broad coverage".
# The first entries are the ones used most here.
_LANGS = [
    ("auto", "Auto-detect"),
    ("en-US", "English (US)"), ("pl-PL", "Polish"), ("uk-UA", "Ukrainian"), ("ru-RU", "Russian"),
    ("en-GB", "English (UK)"), ("ar-AR", "Arabic"), ("bg-BG", "Bulgarian"), ("zh-CN", "Chinese"),
    ("hr-HR", "Croatian"), ("cs-CZ", "Czech"), ("da-DK", "Danish"), ("nl-NL", "Dutch"),
    ("et-EE", "Estonian"), ("fi-FI", "Finnish"), ("fr-FR", "French"), ("fr-CA", "French (Canada)"),
    ("de-DE", "German"), ("hi-IN", "Hindi"), ("hu-HU", "Hungarian"), ("it-IT", "Italian"),
    ("ja-JP", "Japanese"), ("ko-KR", "Korean"), ("nb-NO", "Norwegian"), ("pt-BR", "Portuguese (Brazil)"),
    ("pt-PT", "Portuguese (Portugal)"), ("ro-RO", "Romanian"), ("sk-SK", "Slovak"),
    ("es-ES", "Spanish (Spain)"), ("es-US", "Spanish (US)"), ("sv-SE", "Swedish"),
    ("tr-TR", "Turkish"), ("vi-VN", "Vietnamese"),
]
MULTILINGUAL = [{"code": c, "label": l} for c, l in _LANGS]

# Models this engine knows about. A model is offered in the page only if its folder exists in models/.
#   kind "streaming": text appears while you speak.   kind "offline": text appears when you press Stop.
#   languages: choices for the language dropdown. Empty means the model has no language setting;
#   language_note then explains why.
KNOWN_MODELS = {
    "nemotron-speech-streaming-en-0.6b": {
        "repo": "nvidia/nemotron-speech-streaming-en-0.6b",
        "label": "Nemotron Streaming 0.6B (English, live)",
        "kind": "streaming",
        "languages": [],
        "language_note": "English only",
    },
    "nemotron-3.5-asr-streaming-0.6b": {
        "repo": "nvidia/nemotron-3.5-asr-streaming-0.6b",
        "label": "Nemotron 3.5 Streaming 0.6B (multilingual, live)",
        "kind": "streaming",
        "languages": MULTILINGUAL,
        "language_note": "",
    },
    "parakeet-tdt-0.6b-v3": {
        "repo": "nvidia/parakeet-tdt-0.6b-v3",
        "label": "Parakeet TDT 0.6B v3 (25 languages, on Stop)",
        "kind": "offline",
        "languages": [],
        "language_note": "Detected automatically",
    },
}
DEFAULT_MODEL = "nemotron-speech-streaming-en-0.6b"
DEFAULT_LANGUAGE = "en-US"


def pcm16_to_float(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


class Engine:
    def __init__(self, models_dir: str, model_id: str = DEFAULT_MODEL, latency_ms: int = 1120,
                 device: str = "auto"):
        if latency_ms not in ALLOWED_LATENCIES_MS:
            raise ValueError(f"latency must be one of {list(ALLOWED_LATENCIES_MS)}")
        cuda_ok = torch.cuda.is_available()
        if device == "auto":
            device = "cuda" if cuda_ok else "cpu"
        if device == "cuda" and not cuda_ok:
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        if device == "cpu":
            print("WARNING: running on CPU. Live transcription will probably be too slow.")

        self.models_dir = Path(models_dir)
        self.device = torch.device(device)
        self.latency_ms = latency_ms
        self.lock = threading.Lock()  # one transcription at a time
        self.processor = None
        self.model = None
        self.model_id = None
        self.kind = "streaming"
        self.languages: list[dict] = []
        self.language_note = ""
        self.load(model_id)

    # ---------- models ----------
    def available_models(self) -> list[dict]:
        """Known models (marked downloaded or not) plus any other folder in models/ with a config.json."""
        out = []
        for mid, meta in KNOWN_MODELS.items():
            out.append({"id": mid, "label": meta["label"], "repo": meta["repo"],
                        "downloaded": (self.models_dir / mid / "config.json").is_file(),
                        "streaming": meta["kind"] == "streaming"})
        if self.models_dir.is_dir():
            for d in sorted(self.models_dir.iterdir()):
                if d.name not in KNOWN_MODELS and (d / "config.json").is_file():
                    out.append({"id": d.name, "label": d.name, "repo": None,
                                "downloaded": True, "streaming": True})
        return out

    def load(self, model_id: str) -> None:
        """Load a model from models/<model_id>, freeing the previous one first."""
        path = self.models_dir / model_id
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"Model folder not found: {path}")
        with self.lock:
            self.processor = None
            self.model = None
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            meta = KNOWN_MODELS.get(model_id, {})
            kind = meta.get("kind", "streaming")
            model_class = AutoModelForTDT if kind == "offline" else AutoModelForRNNT
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
            processor = AutoProcessor.from_pretrained(path)
            model = model_class.from_pretrained(path, dtype=dtype).to(self.device).eval()
            if kind == "streaming":
                self._set_latency(processor)
            self.processor, self.model = processor, model
            self.model_id = model_id
            self.kind = kind
            self.languages = meta.get("languages", [])
            self.language_note = meta.get("language_note", "")
        self.warm_up()

    def _set_latency(self, processor) -> None:
        """Select the lookahead for self.latency_ms, or the nearest one this model supports."""
        supported = getattr(processor, "supported_streaming_latencies_ms", None)  # {lookahead: ms}
        if supported and self.latency_ms not in supported.values():
            nearest = min(supported.values(), key=lambda ms: abs(ms - self.latency_ms))
            print(f"WARNING: this model has no {self.latency_ms} ms mode; using {nearest} ms.")
            self.latency_ms = nearest
        processor.set_num_lookahead_tokens(self.latency_ms // 80 - 1)

    # ---------- info ----------
    @property
    def gpu_name(self) -> str:
        if self.device.type == "cuda":
            return torch.cuda.get_device_name(self.device)
        return "CPU"

    def info(self) -> dict:
        return {
            "device": self.gpu_name,
            "cuda": self.device.type == "cuda",
            "streaming": self.kind == "streaming",
            "latency_ms": self.latency_ms if self.kind == "streaming" else None,
            "max_seconds": OFFLINE_MAX_SECONDS if self.kind == "offline" else None,
            "model": self.model_id,
            "models": self.available_models(),
            "languages": self.languages,
            "language_note": self.language_note,
            "default_language": DEFAULT_LANGUAGE,
        }

    def resolve_language(self, code: str | None) -> str | None:
        """Language code for the loaded model, or None if the model has no language setting."""
        if not self.languages:
            return None
        codes = {lang["code"] for lang in self.languages}
        if code is None or code == "":
            return DEFAULT_LANGUAGE
        if code not in codes:
            raise ValueError(f"Language {code!r} is not supported by this model.")
        return code

    # ---------- transcription ----------
    def warm_up(self) -> None:
        """Run one second of silence through the real streaming path."""
        q: queue.Queue = queue.Queue()
        q.put(np.zeros(SAMPLE_RATE, dtype="<i2").tobytes())
        q.put(None)
        self.transcribe(q, lambda _text: None, self.resolve_language(None))

    def transcribe(self, pcm_queue: "queue.Queue[bytes | None]", emit, language: str | None = None) -> None:
        """Read 16 kHz int16 PCM blocks from pcm_queue until None, calling emit(text) with the text.

        Streaming models emit text while audio arrives. Offline models emit everything once, after the
        stream has ended. Blocks until the last text was emitted.
        """
        if self.kind == "offline":
            self._transcribe_offline(pcm_queue, emit)
        else:
            self._transcribe_streaming(pcm_queue, emit, language)

    def _transcribe_offline(self, pcm_queue, emit) -> None:
        chunks = []
        while (item := pcm_queue.get()) is not None:
            chunks.append(pcm16_to_float(item))
        if not chunks:
            return
        audio = np.concatenate(chunks)[: OFFLINE_MAX_SECONDS * SAMPLE_RATE]
        with self.lock:
            proc, model = self.processor, self.model
            inputs = proc(audio, sampling_rate=SAMPLE_RATE).to(model.device, dtype=model.dtype)
            with torch.inference_mode():
                output = model.generate(**inputs, return_dict_in_generate=True)
            text = proc.decode(output.sequences, skip_special_tokens=True)
        if isinstance(text, (list, tuple)):
            text = " ".join(text)
        text = text.strip()
        if text:
            emit(text)

    def _transcribe_streaming(self, pcm_queue, emit, language: str | None) -> None:
        proc = self.processor
        model = self.model
        lang_kw = {"language": language} if language else {}
        first_n = proc.num_samples_first_audio_chunk
        chunk_n = proc.num_samples_per_audio_chunk
        first_mel = proc.num_mel_frames_first_audio_chunk
        mel_per_chunk = proc.num_mel_frames_per_audio_chunk
        hop = proc.feature_extractor.hop_length
        n_fft = proc.feature_extractor.n_fft

        state = {"buf": np.zeros(0, dtype=np.float32), "base": 0, "ended": False}

        def real_total() -> int:
            return state["base"] + len(state["buf"])

        def fill(upto: int) -> None:
            """Block until absolute sample index `upto` is buffered or the stream ended."""
            while real_total() < upto and not state["ended"]:
                item = pcm_queue.get()
                if item is None:
                    state["ended"] = True
                else:
                    state["buf"] = np.concatenate([state["buf"], pcm16_to_float(item)])

        def take(start: int, end: int) -> np.ndarray:
            """Samples [start, end) padded with silence past the end of the stream."""
            base, buf = state["base"], state["buf"]
            piece = buf[start - base : end - base]
            if len(piece) < end - start:
                piece = np.concatenate([piece, np.zeros(end - start - len(piece), dtype=np.float32)])
            return piece

        def trim(keep_from: int) -> None:
            if keep_from > state["base"]:
                state["buf"] = state["buf"][keep_from - state["base"] :]
                state["base"] = keep_from

        fill(first_n)
        if real_total() == 0:
            return  # nothing was said
        first_audio = take(0, first_n)
        first_inputs = proc(
            first_audio,
            sampling_rate=SAMPLE_RATE,
            is_streaming=True,
            is_first_audio_chunk=True,
            return_tensors="pt",
            **lang_kw,
        ).to(model.device, dtype=model.dtype)

        def features():
            yield first_inputs.input_features[:, :first_mel, :]
            mel_idx = first_mel
            start = mel_idx * hop - n_fft // 2
            while True:
                trim(start)
                end = start + chunk_n
                fill(end)
                silent = start >= real_total()
                inputs = proc(
                    take(start, end),
                    sampling_rate=SAMPLE_RATE,
                    is_streaming=True,
                    is_first_audio_chunk=False,
                    return_tensors="pt",
                    **lang_kw,
                ).to(model.device, dtype=model.dtype)
                yield inputs.input_features
                if state["ended"] and silent:
                    return  # one fully silent chunk after the speech flushes the lookahead
                mel_idx += mel_per_chunk
                start = mel_idx * hop - n_fft // 2

        streamer = TextIteratorStreamer(proc.tokenizer, skip_special_tokens=True)
        failure: list[BaseException] = []

        def run():
            try:
                with torch.inference_mode():
                    kwargs = {**first_inputs, "input_features": features(), "streamer": streamer}
                    model.generate(**kwargs)
            except BaseException as exc:  # unblock the reader below
                failure.append(exc)
                streamer.end()

        with self.lock:
            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            for text in streamer:
                if text:
                    emit(text)
            worker.join()
        if failure:
            raise failure[0]

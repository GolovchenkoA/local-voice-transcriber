"""Speech-to-text engine: loads, switches and runs ASR and diarization models on CUDA.

Modes
  single  one transcript of plain text.
  multi   a dialogue: a list of turns {speaker, start, end, text}. Needs either a model that finds
          the speakers itself (VibeVoice) or an offline model with word timestamps (Parakeet TDT)
          combined with a separate diarization model.
"""

import gc
import importlib.util
import json
import os
import queue
import re
import tempfile
import threading
import time
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForAudioFrameClassification,
    AutoModelForRNNT,
    AutoModelForTDT,
    AutoProcessor,
    TextIteratorStreamer,
)

SAMPLE_RATE = 16000
ALLOWED_LATENCIES_MS = (80, 160, 320, 560, 1120)
# Offline models keep the whole recording in memory and work on it after Stop, so the length is limited.
DEFAULT_MAX_SECONDS = 300

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

# Models this engine knows about. A model is usable only if its folder exists in models/.
#   kind          "streaming": text appears while you speak.
#                 "offline":   Parakeet TDT, text appears after Stop, word timestamps available.
#                 "vibevoice": offline, finds the speakers itself.
#   languages     choices for the language menu; empty means the model has no language setting.
#   modes         "single" and/or "multi" (dialogue).
#   diarization   "external" needs a diarization model, "builtin" finds speakers itself, None = no multi mode.
#   needs         Python package that must be installed for the model.
KNOWN_MODELS = {
    "nemotron-speech-streaming-en-0.6b": {
        "repo": "nvidia/nemotron-speech-streaming-en-0.6b",
        "label": "Nemotron Streaming 0.6B (English, live)",
        "kind": "streaming",
        "languages": [],
        "language_note": "English only",
        "modes": ["single"],
        "diarization": None,
    },
    "nemotron-3.5-asr-streaming-0.6b": {
        "repo": "nvidia/nemotron-3.5-asr-streaming-0.6b",
        "label": "Nemotron 3.5 Streaming 0.6B (multilingual, live)",
        "kind": "streaming",
        "languages": MULTILINGUAL,
        "language_note": "",
        "modes": ["single"],
        "diarization": None,
    },
    "parakeet-tdt-0.6b-v3": {
        "repo": "nvidia/parakeet-tdt-0.6b-v3",
        "label": "Parakeet TDT 0.6B v3 (25 languages, on Stop)",
        "kind": "offline",
        "languages": [],
        "language_note": "Detected automatically",
        "modes": ["single", "multi"],
        "diarization": "external",
    },
    "VibeVoice-ASR-HF-NF4": {
        "repo": "Dubedo/VibeVoice-ASR-HF-NF4",
        "label": "VibeVoice-ASR 4-bit (51 languages, speakers built in, on Stop)",
        "kind": "vibevoice",
        "languages": [],
        "language_note": "Detected automatically",
        "modes": ["single", "multi"],
        "diarization": "builtin",
        "needs": "bitsandbytes",
    },
}
DEFAULT_MODEL = "nemotron-speech-streaming-en-0.6b"
DEFAULT_LANGUAGE = "en-US"

# Diarization models ("who spoke when") used with offline models that give word timestamps.
#   marker   file that shows the model folder is complete
KNOWN_DIARIZERS = {
    "Nemotron-3-Diarization": {
        "repo": "nvidia/Nemotron-3-Diarization",
        "label": "Nemotron-3-Diarization (up to 8 speakers)",
        "backend": "nemotron3",
        "marker": "config.json",
    },
    "speaker-diarization-community-1": {
        "repo": "pyannote/speaker-diarization-community-1",
        "label": "pyannote community-1",
        "backend": "pyannote",
        "marker": "config.yaml",
        "needs": "pyannote.audio",
    },
}


def pcm16_to_float(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def restore_plain_linears(model: torch.nn.Module) -> int:
    """Turn 4-bit layers that hold ordinary (not packed) weights back into plain Linear layers.

    The 4-bit VibeVoice checkpoint keeps its audio tokenizers and projector in full precision, but it
    names them differently from the installed transformers, so the loader wraps them as 4-bit layers
    anyway and they then fail with an assertion in bitsandbytes. Packed 4-bit weights are uint8;
    anything else is an ordinary weight. Returns the number of layers restored.
    """
    import bitsandbytes as bnb

    count = 0
    for name, mod in list(model.named_modules()):
        if isinstance(mod, bnb.nn.Linear4bit) and mod.weight.dtype != torch.uint8:
            plain = torch.nn.Linear(mod.in_features, mod.out_features, bias=mod.bias is not None,
                                    device="meta", dtype=mod.weight.dtype)
            plain.weight = torch.nn.Parameter(mod.weight.data, requires_grad=False)
            if mod.bias is not None:
                plain.bias = torch.nn.Parameter(mod.bias.data, requires_grad=False)
            parent_name, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent_name) if parent_name else model, child, plain)
            count += 1
    return count


def read_settings(path: Path | None) -> dict:
    """The remembered choices (last model and diarization model), or {} if there are none."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    except (OSError, ValueError):
        return {}


def package_available(name: str | None) -> bool:
    if not name:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# ---------- turning words and speaker segments into a dialogue (pure functions) ----------

def words_from_timestamps(text: str, tokens: list[dict]) -> list[dict]:
    """Words with start and end times from Parakeet TDT's decoded text and its token timestamps.

    `tokens` is a list of {"token", "start", "end"}. Each token is placed into the word of `text`
    that contains it; a word spans from its first token's start to its last token's end.
    """
    spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    starts = [s for s, _ in spans]
    words = [{"word": text[s:e], "start": None, "end": None} for s, e in spans]
    pos = 0
    for tok in tokens:
        piece = str(tok.get("token", "")).replace("▁", "")
        if not piece:
            continue
        idx = text.find(piece, pos)
        if idx < 0:
            continue
        pos = idx + len(piece)
        w = bisect_right(starts, idx) - 1
        if w < 0:
            continue
        t0, t1 = float(tok["start"]), float(tok["end"])
        word = words[w]
        word["start"] = t0 if word["start"] is None else min(word["start"], t0)
        word["end"] = t1 if word["end"] is None else max(word["end"], t1)

    # a word that no token landed in (rare): place it between its neighbours
    last_end = 0.0
    for i, word in enumerate(words):
        if word["start"] is None:
            nxt = next((w["start"] for w in words[i + 1:] if w["start"] is not None), last_end)
            word["start"], word["end"] = last_end, max(last_end, nxt)
        last_end = word["end"]
    return words


def assign_speakers(words: list[dict], segments: list[dict]) -> list[dict]:
    """Give each word the speaker who is talking when the word starts (or the nearest speaker in time).

    Parakeet's word end times stretch across the silence after a word, so only the first
    0.4 s of a word is compared with the speaker segments.
    """
    out = []
    for w in words:
        t0 = w["start"]
        t1 = min(w["end"], t0 + 0.4)
        overlap: dict[int, float] = {}
        for s in segments:
            o = min(t1, s["end"]) - max(t0, s["start"])
            if o > 0:
                overlap[s["speaker"]] = overlap.get(s["speaker"], 0.0) + o
        if overlap:
            speaker = max(overlap, key=overlap.get)
        elif segments:
            nearest = min(segments, key=lambda s: 0 if s["start"] <= t0 <= s["end"]
                          else min(abs(t0 - s["start"]), abs(t0 - s["end"])))
            speaker = nearest["speaker"]
        else:
            speaker = 0
        out.append({**w, "speaker": speaker})
    return out


def merge_turns(items: list[dict]) -> list[dict]:
    """Join consecutive items of the same speaker into turns and number speakers by first appearance.

    `items` need "speaker", "start", "end" and either "word" or "text".
    """
    numbering: dict[int, int] = {}
    turns: list[dict] = []
    for it in items:
        piece = (it["word"] if "word" in it else it.get("text") or "").strip()
        if not piece:
            continue
        spk = numbering.setdefault(it["speaker"], len(numbering))
        if turns and turns[-1]["speaker"] == spk:
            turns[-1]["text"] += " " + piece
            turns[-1]["end"] = max(turns[-1]["end"], float(it["end"]))
        else:
            turns.append({"speaker": spk, "start": float(it["start"]),
                          "end": float(it["end"]), "text": piece})
    for t in turns:
        t["start"] = round(t["start"], 2)
        t["end"] = round(t["end"], 2)
    return turns


# ---------- diarization models ----------

class Nemotron3Diarizer:
    """NVIDIA Nemotron-3-Diarization through transformers (offline pass over the whole recording)."""

    def __init__(self, path: Path, device: torch.device):
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = AutoModelForAudioFrameClassification.from_pretrained(path, dtype=dtype).to(device).eval()

    def diarize(self, audio: np.ndarray) -> list[dict]:
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE).to(self.model.device, dtype=self.model.dtype)
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        found = self.processor.extract_speaker_dict(logits.float(), inputs["attention_mask"])[0]
        return [{"start": s["Start"], "end": s["End"], "speaker": s["Speaker"]} for s in found]


class PyannoteDiarizer:
    """pyannote.audio pipeline (community-1 or 3.1), loaded from a local folder."""

    def __init__(self, path: Path, device: torch.device):
        from pyannote.audio import Pipeline  # installed separately: see requirements-multispeaker.txt

        self.pipeline = Pipeline.from_pretrained(str(path))
        self.pipeline.to(device)

    def diarize(self, audio: np.ndarray) -> list[dict]:
        waveform = torch.from_numpy(audio).unsqueeze(0)
        out = self.pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        annotation = getattr(out, "exclusive_speaker_diarization", None)
        if annotation is None:
            annotation = getattr(out, "speaker_diarization", out)
        order: dict[str, int] = {}
        segments = []
        for turn, _, label in annotation.itertracks(yield_label=True):
            segments.append({"start": float(turn.start), "end": float(turn.end),
                             "speaker": order.setdefault(label, len(order))})
        return segments


DIARIZER_BACKENDS = {"nemotron3": Nemotron3Diarizer, "pyannote": PyannoteDiarizer}


class Engine:
    # Models used a moment ago stay in memory besides the active one, so switching back is instant.
    # (The 4-bit VibeVoice model needs nearly all of the GPU memory, so it is never kept this way.)
    MAX_CACHED_MODELS = 2

    def __init__(self, models_dir: str, model_id: str = DEFAULT_MODEL, latency_ms: int = 1120,
                 device: str = "auto", max_seconds: int = DEFAULT_MAX_SECONDS,
                 settings_path: Path | None = None):
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
        self.max_seconds = max_seconds
        self.lock = threading.Lock()  # one transcription at a time
        self.processor = None
        self.model = None
        self.model_id = None
        self.kind = "streaming"
        self.meta: dict = {}
        self.languages: list[dict] = []
        self.language_note = ""
        self.diarizer = None
        self.diarizer_id: str | None = None
        self.settings_path = Path(settings_path) if settings_path else None
        self._cache: dict[str, tuple] = {}  # model id -> (processor, model), oldest first
        self.load(model_id)

    def _save_settings(self, **changes) -> None:
        """Remember the last model / diarization model for the next start."""
        if not self.settings_path:
            return
        data = read_settings(self.settings_path)
        data.update(changes)
        try:
            self.settings_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass  # not being able to remember is not an error

    # ---------- models ----------
    def available_models(self) -> list[dict]:
        """Known models (marked downloaded or not) plus any other folder in models/ with a config.json."""
        out = []
        for mid, meta in KNOWN_MODELS.items():
            out.append({"id": mid, "label": meta["label"], "repo": meta["repo"],
                        "downloaded": (self.models_dir / mid / "config.json").is_file(),
                        "available": package_available(meta.get("needs")),
                        "needs": meta.get("needs"),
                        "streaming": meta["kind"] == "streaming",
                        "modes": meta["modes"], "diarization": meta["diarization"]})
        if self.models_dir.is_dir():
            known_dirs = set(KNOWN_MODELS) | set(KNOWN_DIARIZERS)
            for d in sorted(self.models_dir.iterdir()):
                if d.name not in known_dirs and (d / "config.json").is_file():
                    out.append({"id": d.name, "label": d.name, "repo": None, "downloaded": True,
                                "available": True, "needs": None, "streaming": True,
                                "modes": ["single"], "diarization": None})
        return out

    def available_diarizers(self) -> list[dict]:
        return [{"id": did, "label": meta["label"], "repo": meta["repo"],
                 "downloaded": (self.models_dir / did / meta["marker"]).is_file(),
                 "available": package_available(meta.get("needs")),
                 "needs": meta.get("needs")}
                for did, meta in KNOWN_DIARIZERS.items()]

    def _activate(self, model_id: str, processor, model) -> None:
        meta = KNOWN_MODELS.get(model_id, {})
        self.processor, self.model = processor, model
        self.model_id = model_id
        self.kind = meta.get("kind", "streaming")
        self.meta = meta
        self.languages = meta.get("languages", [])
        self.language_note = meta.get("language_note", "")
        if self.kind == "streaming":
            self.latency_ms = getattr(processor, "streaming_latency_ms", self.latency_ms)

    def load(self, model_id: str) -> None:
        """Make models/<model_id> the active model.

        Already loaded models are not loaded again: the active one is kept as it is, and the last
        few used ones stay in memory, so switching back to them is instant.
        """
        path = self.models_dir / model_id
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"Model folder not found: {path}")
        meta = KNOWN_MODELS.get(model_id, {})
        if not package_available(meta.get("needs")):
            raise RuntimeError(f"This model needs the Python package {meta['needs']!r}. "
                               "See requirements.txt.")
        if model_id == self.model_id and self.model is not None:
            return
        kind = meta.get("kind", "streaming")
        started = time.perf_counter()
        with self.lock:
            prev_id, prev_kind = self.model_id, self.kind
            # keep the current model ready in memory (never the 4-bit VibeVoice model)
            keep = (self.processor, self.model) if self.model is not None and prev_kind != "vibevoice" else None
            cached = self._cache.pop(model_id, None)
            if keep:
                self._cache[prev_id] = keep
            keep = None  # no other reference to the old model may stay alive, memory is tight
            if kind == "vibevoice":
                self._cache.clear()  # the 4-bit VibeVoice model needs nearly all of the GPU memory
                self._drop_diarizer()
            while len(self._cache) > self.MAX_CACHED_MODELS:
                self._cache.pop(next(iter(self._cache)))  # drop the oldest
            self.processor = self.model = None
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            try:
                if cached:
                    processor, model = cached
                else:
                    print(f"Loading {model_id} ...", flush=True)
                    processor, model = self._load_from_disk(path, kind)
            except BaseException:
                # Something went wrong (for example not enough GPU memory): bring the old model back.
                if prev_id in self._cache:
                    self._activate(prev_id, *self._cache.pop(prev_id))
                elif prev_id is not None:
                    print(f"Could not load {model_id}; loading {prev_id} again ...", flush=True)
                    self._activate(prev_id, *self._load_from_disk(self.models_dir / prev_id, prev_kind))
                raise
            self._activate(model_id, processor, model)
        if not cached and kind != "vibevoice":
            self.warm_up()
        where = "from memory" if cached else f"loaded in {time.perf_counter() - started:.1f} s"
        print(f"{model_id} is ready ({where}).", flush=True)
        self._save_settings(model=model_id)

    def _load_from_disk(self, path: Path, kind: str):
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        if kind == "vibevoice":
            from transformers import VibeVoiceAsrForConditionalGeneration

            processor = AutoProcessor.from_pretrained(path)
            # Everything on the GPU: a 4-bit model cannot be split between GPU and CPU, and
            # "auto" would put parts on the CPU on an 8 GB card.
            device_map = {"": self.device.index or 0} if self.device.type == "cuda" else "cpu"
            model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
                path, dtype=dtype, device_map=device_map).eval()
            if "quantization_config" in json.loads((path / "config.json").read_text(encoding="utf-8")):
                restore_plain_linears(model)
            return processor, model
        model_class = AutoModelForTDT if kind == "offline" else AutoModelForRNNT
        processor = AutoProcessor.from_pretrained(path)
        model = model_class.from_pretrained(path, dtype=dtype).to(self.device).eval()
        if kind == "streaming":
            self._set_latency(processor)
        return processor, model

    def _set_latency(self, processor) -> None:
        """Select the lookahead for self.latency_ms, or the nearest one this model supports."""
        supported = getattr(processor, "supported_streaming_latencies_ms", None)  # {lookahead: ms}
        if supported and self.latency_ms not in supported.values():
            nearest = min(supported.values(), key=lambda ms: abs(ms - self.latency_ms))
            print(f"WARNING: this model has no {self.latency_ms} ms mode; using {nearest} ms.")
            self.latency_ms = nearest
        processor.set_num_lookahead_tokens(self.latency_ms // 80 - 1)

    # ---------- diarization models ----------
    def load_diarizer(self, diarizer_id: str) -> None:
        meta = KNOWN_DIARIZERS.get(diarizer_id)
        if meta is None:
            raise ValueError(f"Unknown diarization model {diarizer_id!r}.")
        path = self.models_dir / diarizer_id
        if not (path / meta["marker"]).is_file():
            raise FileNotFoundError(f"Diarization model not found: {path}")
        if not package_available(meta.get("needs")):
            raise RuntimeError(f"This model needs the Python package {meta['needs']!r}. "
                               "See requirements-multispeaker.txt.")
        if diarizer_id == self.diarizer_id and self.diarizer is not None:
            return
        started = time.perf_counter()
        with self.lock:
            self._drop_diarizer()
            print(f"Loading {diarizer_id} ...", flush=True)
            self.diarizer = DIARIZER_BACKENDS[meta["backend"]](path, self.device)
            self.diarizer_id = diarizer_id
        print(f"{diarizer_id} is ready (loaded in {time.perf_counter() - started:.1f} s).", flush=True)
        self._save_settings(diarizer=diarizer_id)

    def unload_diarizer(self) -> None:
        with self.lock:
            self._drop_diarizer()
        self._save_settings(diarizer=None)

    def _drop_diarizer(self) -> None:
        self.diarizer = None
        self.diarizer_id = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

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
            "max_seconds": self.max_seconds if self.kind != "streaming" else None,
            "model": self.model_id,
            "model_label": self.meta.get("label", self.model_id),
            "modes": self.meta.get("modes", ["single"]),
            "diarization": self.meta.get("diarization"),
            "models": self.available_models(),
            "languages": self.languages,
            "language_note": self.language_note,
            "default_language": DEFAULT_LANGUAGE,
            "diarizer": self.diarizer_id,
            "diarizers": self.available_diarizers(),
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

    def check_mode(self, mode: str) -> None:
        """Raise ValueError (with a message for the user) if the loaded setup cannot run `mode`."""
        if mode not in ("single", "multi"):
            raise ValueError(f"Unknown mode {mode!r}.")
        if mode not in self.meta.get("modes", ["single"]):
            raise ValueError("The chosen model cannot separate speakers. Choose another model for multiple speakers.")
        if mode == "multi" and self.meta.get("diarization") == "external" and self.diarizer is None:
            raise ValueError("Choose a diarization model first.")

    # ---------- transcription ----------
    def warm_up(self) -> None:
        """Run one second of silence through the real path (live and Parakeet models)."""
        q: queue.Queue = queue.Queue()
        q.put(np.zeros(SAMPLE_RATE, dtype="<i2").tobytes())
        q.put(None)
        self.transcribe(q, lambda _text: None, self.resolve_language(None))

    def transcribe(self, pcm_queue: "queue.Queue[bytes | None]", emit, language: str | None = None,
                   mode: str = "single", emit_turns=None) -> None:
        """Read 16 kHz int16 PCM blocks from pcm_queue until None.

        Single mode calls emit(text). Live models emit while audio arrives; offline models emit once,
        after the stream has ended. Multi mode (offline models only) calls emit_turns(list of turns).
        Blocks until the last result was emitted.
        """
        self.check_mode(mode)
        if self.kind == "streaming":
            self._transcribe_streaming(pcm_queue, emit, language)
        else:
            self._transcribe_offline(pcm_queue, emit, mode, emit_turns)

    @staticmethod
    def _collect(pcm_queue, max_samples: int) -> np.ndarray | None:
        chunks = []
        while (item := pcm_queue.get()) is not None:
            chunks.append(pcm16_to_float(item))
        if not chunks:
            return None
        return np.concatenate(chunks)[:max_samples]

    def _transcribe_offline(self, pcm_queue, emit, mode: str, emit_turns) -> None:
        audio = self._collect(pcm_queue, self.max_seconds * SAMPLE_RATE)
        if audio is None:
            return
        with self.lock:
            if self.kind == "vibevoice":
                turns = merge_turns(self._vibevoice(audio))
                if mode == "multi":
                    emit_turns(turns)
                else:
                    emit(" ".join(t["text"] for t in turns))
                return
            if mode == "single":
                text = self._tdt_text(audio)
                if text:
                    emit(text)
                return
            words = self._tdt_words(audio)
            segments = self.diarizer.diarize(audio)
        emit_turns(merge_turns(assign_speakers(words, segments)))

    def _tdt_generate(self, audio: np.ndarray, with_durations: bool):
        proc, model = self.processor, self.model
        inputs = proc(audio, sampling_rate=SAMPLE_RATE).to(model.device, dtype=model.dtype)
        with torch.inference_mode():
            output = model.generate(**inputs, return_dict_in_generate=True)
        if with_durations:
            return proc.decode(output.sequences, durations=output.durations, skip_special_tokens=True)
        return proc.decode(output.sequences, skip_special_tokens=True)

    def _tdt_text(self, audio: np.ndarray) -> str:
        text = self._tdt_generate(audio, with_durations=False)
        if isinstance(text, (list, tuple)):
            text = " ".join(text)
        return text.strip()

    def _tdt_words(self, audio: np.ndarray) -> list[dict]:
        decoded, stamps = self._tdt_generate(audio, with_durations=True)
        text = decoded[0] if isinstance(decoded, (list, tuple)) else decoded
        tokens = stamps[0] if stamps and isinstance(stamps[0], list) else stamps
        return words_from_timestamps(text, tokens)

    def _vibevoice(self, audio: np.ndarray) -> list[dict]:
        """Turns {speaker, start, end, text} straight from VibeVoice-ASR."""
        import soundfile as sf

        proc, model = self.processor, self.model
        # a file path is the documented way to hand audio to the processor
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "recording.wav")
            sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
            inputs = proc.apply_transcription_request(audio=path).to(model.device, model.dtype)
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=int(len(audio) / SAMPLE_RATE * 15) + 512)
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        parsed = proc.decode(generated, return_format="parsed")[0]
        if not isinstance(parsed, list):  # the model did not return the expected structure
            return [{"speaker": 0, "start": 0.0, "end": len(audio) / SAMPLE_RATE, "text": str(parsed)}]
        turns = [{"speaker": int(d.get("Speaker", 0)), "start": float(d.get("Start", 0)),
                  "end": float(d.get("End", 0)), "text": str(d.get("Content", ""))} for d in parsed]
        # drop pure sound tags such as "[Silence]" or "[Music]"
        return [t for t in turns if not re.fullmatch(r"\s*\[[^\]]*\]\s*", t["text"])]

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

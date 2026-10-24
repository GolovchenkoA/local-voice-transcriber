# Speaker-aware speech models: investigation

Goal: find models that let the transcript show **who said what** (a dialog, row by row) in the Local Voice Transcriber, running locally on an NVIDIA RTX 5060 Laptop GPU (8 GB VRAM) on Windows 11, with no internet after setup.

Research date: October 2026. Nothing in this report was run or tested on this computer. Every number comes from the model cards, papers or documentation named in the Sources section. Where a fact could not be found, it says "not stated".

## 1. Short answer

- The three models the app uses today (Nemotron Streaming English, Nemotron 3.5 Streaming multilingual, Parakeet TDT v3) **cannot tell speakers apart**. They output words only.
- To get a dialog you need one of three things:
  1. **A separate diarization model** ("who spoke when") next to the speech model, then match words to speakers by time. Most options below are this kind.
  2. **A speaker-attributed speech model** that still needs a diarizer as input (NVIDIA's `multitalker-parakeet-streaming-0.6b-v1`).
  3. **A single model that does both** (Microsoft's VibeVoice-ASR). This is the only verified case where one model recognizes different voices by itself. See section 6.
- **Best fit for this app today:** Parakeet TDT v3 (offline, text on Stop, supports Polish, Ukrainian and Russian, can give word timestamps) plus a diarizer such as Nemotron-3-Diarization, pyannote community-1 or DiariZen. This gives a dialog after you press Stop. Live dialog is possible but weaker (section 7).
- **Not verified for Polish, Ukrainian or Russian:** none of the diarization models published accuracy numbers for these three languages. The independent benchmark covers Chinese, English, German, Japanese and Spanish only.

## 2. Terms used

| Term | Meaning |
|------|---------|
| Diarization | Finding **who spoke when**. Output: time segments labelled "Speaker 1", "Speaker 2". It does not read the words. |
| Speaker-attributed ASR | Speech recognition whose output is already split by speaker. |
| Speaker identification | Recognizing a **known** person by name from a stored voice sample (enrollment). Diarization alone gives anonymous labels. |
| DER | Diarization error rate, in percent. Lower is better. It counts missed speech, false alarms and speaker confusion. |
| cpWER | Word error rate that also penalizes words given to the wrong speaker. |
| Overlap | Two people talking at the same time. The hardest case for every model here. |

## 3. How the top 10 were chosen

The ranking is my judgment, based on: (1) can run locally and offline, (2) usable on an 8 GB GPU or on the CPU, (3) open license, (4) evidence on official cards or in independent papers, (5) fit with this app (live use, Windows, `transformers` or ONNX instead of Linux-only tools). Cloud services (for example pyannoteAI's hosted model, which led the independent benchmark at 11.2 % DER) were left out because the app must work offline.

## 4. Summary table

| # | Model | Type | Live | Max speakers | License | Main framework |
|---|-------|------|------|--------------|---------|----------------|
| 1 | NVIDIA Nemotron-3-Diarization | Diarizer | Yes (and offline) | 8 | OpenMDW 1.1 | NeMo; `transformers` (from source) |
| 2 | NVIDIA Streaming Sortformer v2.1 | Diarizer | Yes | 4 | NVIDIA Open Model License | NeMo |
| 3 | NVIDIA multitalker-parakeet-streaming-0.6b-v1 | Speaker-attributed ASR (needs a diarizer) | Yes | Not stated (tested on 2 to 4) | NVIDIA Open Model License | NeMo |
| 4 | Microsoft VibeVoice-ASR (and VibeVoice-ASR-Streaming) | ASR + diarization in one model | Offline model: no. Streaming variant: yes | Not stated for ASR | MIT (repository) | `transformers` (VibeVoice-ASR-HF) |
| 5 | pyannote speaker-diarization-community-1 (successor of 3.1) | Diarizer | No | No fixed limit | CC-BY-4.0 (gated) | pyannote.audio |
| 6 | DiariZen | Diarizer | No | Not stated | Code MIT, **weights CC BY-NC 4.0** | Own toolkit on pyannote |
| 7 | diart | Streaming diarization framework | Yes | Set in config | MIT (models gated) | Python, pyannote models |
| 8 | NeMo cascaded diarizer (VAD + speaker embeddings + clustering + MSDD) | Diarizer | Mainly offline | Fewer limits than Sortformer | NeMo checkpoints (see card) | NeMo |
| 9 | sherpa-onnx speaker diarization | Diarizer (pyannote segmentation + 3D-Speaker or NeMo embeddings) | Examples are offline | Set count or threshold | Mixed per model | ONNX, many languages (Python included) |
| 10 | SpeechBrain ECAPA-TDNN | Speaker embeddings (voice fingerprints) | Not a diarizer by itself | n/a | Apache-2.0 | SpeechBrain |

## 5. The ten models in detail

### 1. NVIDIA Nemotron-3-Diarization

- **What it is:** an open-weight speaker diarizer from NVIDIA with streaming and offline modes. About 99 M parameters, a 31-layer transformer encoder, and an Arrival-Order Speaker Cache for streaming. Speaker outputs are ordered by who spoke first.
- **Latency modes (input buffer only, excluding compute):** 1.04 s (default), 0.64 s, 0.32 s. Offline uses a 30.4 s buffer. With chunked inference the audio length is not limited.
- **Accuracy (NVIDIA's own evaluation, full-file DER at the 30.4 s setting):** DIHARD III 12.73 against 19.09 for Sortformer v2.1; AMI SDM 11.14 against 21.42; NOTSOFAR1 single-channel 11.00 against 30.49. It also leads at lower latency (DIHARD III at 1.04 s: 13.18 against 19.60).
- **Pros**
  - Up to 8 speakers, double Sortformer.
  - Best published numbers of the NVIDIA models, and small (about 99 M parameters).
  - Streaming and offline in one checkpoint.
  - Training data includes English, Mandarin, Hindi, Kannada, Telugu, Bengali and other languages, so it was not trained on English alone.
  - Permissive license that the card describes as ready for commercial and non-commercial use.
  - Also available through `transformers`, which this app already uses.
- **Cons**
  - Very new. No independent benchmark was found.
  - The card lists only Linux as supported. Windows is not stated.
  - `transformers` support needs an install from source. The v5.20.2 documentation says the model is not in that release, while another source says it arrived in 5.18.0, so check this before relying on it.
  - No per-language accuracy and no supported-language list for inference. Polish, Ukrainian and Russian are not named.
  - The numbers are NVIDIA's own and use forced-alignment reference labels that the card says are not directly comparable with other labels.
  - Behavior above 8 speakers is not stated.

### 2. NVIDIA Streaming Sortformer v2.1 (`diar_streaming_sortformer_4spk-v2.1`)

- **What it is:** a streaming diarizer, a FastConformer encoder with a transformer on top, 117 M parameters. It outputs 4 speaker-activity values per 80 ms frame, which handles overlap naturally.
- **Latency:** "low latency" mode 1.04 s with real-time factor 0.093; "very high latency" mode 30.4 s with real-time factor 0.002 (measured on an RTX 6000 Ada).
- **Accuracy (card):** DIHARD III 19.49 (30.4 s) / 20.21 (1.04 s); CALLHOME 10.10 / 11.19; AMI IHM 15.90 / 20.57.
- **Independent benchmark (v2-streaming):** fast (about 210 times faster than real time on average), strong on Chinese and Japanese, weaker on English and Spanish, and error grows with more speakers because of the 4-speaker limit. Speaker confusion is its main error type.
- **Pros:** streaming, fast, small, handles overlap, widely compared in papers.
- **Cons**
  - Maximum 4 speakers; quality drops with 5 or more.
  - Trained mainly on English; the card warns performance may drop on other languages.
  - Documented only for NeMo, with Linux-style install commands. Windows is not stated.
  - NVIDIA's own Riva notes say speaker tags can be wrong during overlapping speech in low-latency mode.
  - Superseded by Nemotron-3-Diarization (entry 1).

### 3. NVIDIA multitalker-parakeet-streaming-0.6b-v1

- **What it is:** a streaming speech model built on **Nemotron-Speech-Streaming** (the same English base as the app's default model). It adds "speaker kernels" to the encoder. It needs **speaker activity from a diarizer** (the examples use Streaming Sortformer v2 or v2.1) and runs **one model instance per speaker**, each receiving the same mixed audio and producing that speaker's text.
- **Latency:** the same four choices as the base model: 0.08, 0.16, 0.56 and 1.12 s.
- **Accuracy (card):** multitalker cpWER at 1.12 s: AMI IHM 21.26, AMI SDM 37.44, CH109 15.81, Mixer 6 23.81. In single-speaker mode it is slightly worse on average than the base model (7.44 against 7.16 average WER).
- **Pros:** handles overlapping speech well according to NVIDIA, no voice enrollment needed, streaming, same family as the model the app already uses.
- **Cons**
  - **Compute grows with the number of speakers** (one model instance each). Three speakers means three copies of a 0.6 B model plus the diarizer.
  - Depends on the diarizer; the diarization error rate rises sharply above 4 speakers (DIHARD III 42.56 for 5 to 9 speakers).
  - Documented only for NeMo. Hardware, OS, VRAM and speaker limits are not stated.
  - Languages are not stated; the base model is English only.
  - A third-party GGUF port supports only the single-speaker path and gives a flat transcript.

### 4. Microsoft VibeVoice-ASR (and VibeVoice-ASR-Streaming)

- **What it is:** one model that does recognition, diarization and timestamping together. Its output interleaves **who, when and what**. Offline model: 9 B parameters (BF16), up to 60 minutes of audio in a single pass (64 K tokens), more than 50 languages with code-switching, optional hotwords. A `transformers` version (VibeVoice-ASR-HF) exists, plus vLLM support.
- **Streaming variant:** the repository announces VibeVoice-ASR-Streaming (2026-09-03) for "who said what" as speech arrives, with 10 languages. Parameter counts and VRAM for it are inconsistent in the third-party sources found, so treat them as unknown.
- **Pros**
  - **No separate diarization model needed.** This is the all-in-one answer.
  - MIT license, long recordings, many languages, hotwords for names and terms.
  - Part of the `transformers` ecosystem.
- **Cons**
  - **Too big for 8 GB VRAM at full precision.** 9 B parameters in BF16 is about 18 GB for the weights alone (my estimate; the card gives no VRAM figure). The only smaller variant on the page is a BitNet version (1.58 GB, runs on CPU with a real-time factor below 1 on 3 or more threads), whose accuracy was not found.
  - The offline model needs the whole recording, so it gives a dialog only after Stop.
  - Speaker limit, Windows support and exact accuracy numbers (charts only) are not stated.
  - The card says outputs may be unexpected, biased or inaccurate, and the project is aimed at research and development.
  - Per-language quality is not published in the text of the card. (Correction, from the Hugging Face language tags: the offline model lists 51 languages **including Polish `pl`, Ukrainian `uk` and Russian `ru`**. The streaming variant lists only 10 languages: Chinese, English, French, German, Italian, Japanese, Korean, Portuguese, Russian and Spanish, so **no Polish or Ukrainian**.)

### 5. pyannote speaker-diarization-community-1 (and 3.1)

- **What it is:** the most widely used open diarization pipeline. Community-1 combines a segmentation model, WeSpeaker embeddings and VBx clustering. It improves on 3.1 mainly in counting and assigning speakers, and adds an "exclusive" output that is easier to align with transcripts.
- **Accuracy (pyannote's own table, DER without collar):** AMI IHM 18.8 to 17.0; DIHARD 3 21.4 to 20.2; AliMeeting channel 1 24.5 to 20.3; REPERE got worse (7.9 to 8.9) and VoxConverse stayed at 11.2.
- **Independent benchmark (3.1):** speed about 45 times faster than real time; weak on Chinese (19.8), Japanese (28.8) and Spanish (27.3) compared with DiariZen, and good on 5 or more speakers (10.6).
- **Pros:** mature, large community, no fixed speaker limit (you can set a min and max), runs on CPU or CUDA, works offline after download, CC-BY-4.0.
- **Cons**
  - **Gated download:** you must accept conditions on Hugging Face and use an access token for the first download (offline afterwards by cloning the repository).
  - Offline only; no streaming.
  - Weaker on non-English languages in the independent benchmark.
  - Community-1's gains over 3.1 are moderate and come from pyannote's own tables.
  - Windows is not stated on the card.

### 6. DiariZen

- **What it is:** a toolkit combining a WavLM encoder with pyannote 3.1 components (the independent paper calls it a hybrid of end-to-end local diarization, WavLM embeddings and pyannote clustering). Several checkpoints (Base, Large, Large v2).
- **Accuracy:** the best open-source result in the independent benchmark (13.3 % average DER, against 11.2 % for the commercial pyannoteAI). Card numbers for Large-s80-v2: AMI-SDM 13.9, AliMeeting far 10.8, DIHARD3 14.5, VoxConverse 9.1 (pyannote 3.1: 22.4, 24.4, 21.7, 11.3). Strong with many speakers.
- **Pros:** most accurate open model in the independent comparison, good on many languages and on 5 or more speakers.
- **Cons**
  - **Model weights are CC BY-NC 4.0: non-commercial only.** The code is MIT.
  - Slower than the NVIDIA models (about 20 times faster than real time, against about 210 for Sortformer).
  - Offline only as documented; speaker limit and Windows support not stated.
  - Install instructions pin old versions (Python 3.10, PyTorch 2.1.1 with CUDA 12.1 and a bundled pyannote fork). That does not match the PyTorch 2.11 (CUDA 12.8) used for the RTX 5060, so a separate environment would probably be needed.

### 7. diart

- **What it is:** a Python framework for real-time diarization built on pyannote models, with microphone input. Default step 0.5 s over a 5 s window; latency tunable from about 0.5 to 5 s. Other embedding models (WeSpeaker, SpeechBrain, NVIDIA TitaNet) can be plugged in.
- **Pros:** the established open **live** diarization option, MIT license, flexible models.
- **Cons**
  - Default models are gated (Hugging Face conditions and token).
  - Windows support not stated; dependencies are Linux-style (ffmpeg below 4.4, PortAudio 19.6.x).
  - Reproducing published results is advised with pyannote.audio older than 3.1, which conflicts with newer pyannote versions.
  - The page shows no DER numbers in text, no recent-release date, and "transcription" is still listed as coming soon.

### 8. NeMo cascaded diarizer (voice activity detection, TitaNet speaker embeddings, clustering, MSDD)

- **What it is:** NVIDIA's classic modular pipeline: detect speech, extract voice vectors, cluster them, then refine with a neural diarizer (MSDD, a target-speaker voice-activity model).
- **Pros:** according to NVIDIA's documentation it has fewer limits on speaker count and session length than end-to-end Sortformer; parts can be swapped; also usable for speaker identification.
- **Cons:** harder to tune and deploy as a whole (NVIDIA's own wording); mostly offline; NeMo-based and Linux-first; no accuracy figures were found in this investigation; newer end-to-end models (entries 1 and 2) are the NVIDIA focus now.

### 9. sherpa-onnx speaker diarization

- **What it is:** a lightweight ONNX runtime from the k2-fsa project. Diarization uses a pyannote segmentation 3.0 model (5.7 MB, 1.5 MB in int8) or Rev's Reverb segmentation, plus a speaker embedding model (3D-Speaker or NeMo TitaNet small), then clustering. Many language bindings, including Python.
- **Speed (CPU, 57 s test file):** real-time factor 0.11 to 0.45 depending on model combination (fastest: pyannote segmentation with NeMo embeddings, int8, 0.110).
- **Pros:** tiny models, runs on CPU with no GPU memory use, no PyTorch needed, easy to bundle, set the exact speaker count or a threshold.
- **Cons**
  - The documented examples are **offline** only (a command-line tool); no streaming example was found.
  - Accuracy numbers not found in the sources read.
  - Licenses differ per model: the Reverb model is non-commercial; the page does not state one for the pyannote model.
  - Windows wheels are not mentioned on the pages read.
  - The embedding model in the example is trained for Chinese speech (3D-Speaker zh-cn); the NeMo one is English.

### 10. SpeechBrain ECAPA-TDNN (`spkrec-ecapa-voxceleb`)

- **What it is:** a speaker-embedding (voice fingerprint) model trained on VoxCeleb1 and 2. Equal error rate 0.80 % on the cleaned VoxCeleb1 test set. Scores two recordings with cosine distance.
- **Why it is on the list:** it is the building block for **speaker identification**: store a sample of each known person once, then label segments with real names instead of "Speaker 1".
- **Pros:** Apache-2.0, small, well known, flexible.
- **Cons:** not a diarizer on its own (needs segments from another model or a voice-activity detector); the card does not describe clustering or enrollment workflows, offline loading or Windows; trained on celebrity speech, performance on other data is not guaranteed.

### Others looked at, not in the top 10

| Model | Why it was left out |
|-------|---------------------|
| pyannoteAI (cloud) | Best in the independent benchmark (11.2 % DER) but closed and online. |
| Reverb diarization (Rev) | Non-commercial license; the sherpa-onnx page lists it. |
| SpeakerLM | A single model for speaker-attributed transcripts (paper, August 2025). Whether weights are released was not confirmed. |
| NVIDIA Sortformer v1 and v2 | Superseded by v2.1 and Nemotron-3-Diarization. |
| WhisperX | A pipeline that combines Whisper with pyannote. It uses Whisper, not the models of this app, so it was not researched. |

## 6. Can a model recognize different voices by itself?

**Yes, but only in a few cases. None of the three models in this app can.**

| Case | Does it work alone? | Notes |
|------|---------------------|-------|
| **VibeVoice-ASR (offline, about 9 B)** | **Yes.** Output already says who spoke, when, and what. Languages include Polish, Ukrainian and Russian (Hugging Face tags). | Verified on the model card. The full model is about 17 GB (the 4-bit community build lists 17.3 GB as its original size), so it does not fit this GPU unmodified. A 4-bit community build exists (section 11). Gives a dialog after the recording only. |
| **VibeVoice-ASR-Streaming** (named 1.5B and 7B) | **Yes, live.** The Hugging Face card says it transcribes "who said what" as speech arrives. | See the addendum (section 11). The 1.5B checkpoint is listed as about 3 B parameters, so it should fit 8 GB (my estimate; the card gives no VRAM). 10 languages, no Polish or Ukrainian. |
| **VibeVoice-ASR-BitNet** | Yes (same output), on CPU. | 1.58 GB. The page gives no accuracy. |
| SpeakerLM | Yes, per its paper. | Research model; weights not confirmed. |
| multitalker-parakeet-streaming-0.6b-v1 | **No.** | It needs speaker activity from Sortformer as input. |
| Nemotron / Parakeet TDT (the app's models) | **No.** | Words only. |
| Any diarizer (entries 1, 2, 5 to 9) | No. | They give speaker segments but no words. |

## 7. How each model works with the app's three speech models

The three models the app already uses:

- **A:** Nemotron Streaming 0.6B (English, live)
- **B:** Nemotron 3.5 Streaming 0.6B (multilingual, live)
- **C:** Parakeet TDT 0.6B v3 (25 languages, text appears on Stop)

Key technical facts behind the table:

- A diarizer does not read words. To build a dialog, the app must **match each word to a speaker by time**.
- **C** can give word-level timestamps. The Transformers documentation shows `ParakeetForTDT` returning durations that become token start and end times. The app does not use this yet, but it is documented.
- **A and B** give text in chunks (0.08 to 1.12 s) and the app does not read timestamps from them today. A live dialog would use the arrival time of each chunk, which is coarse: if the speaker changes inside a chunk, the words can be given to the wrong person.
- Any diarizer that works on files can also run on the recording after Stop, whichever speech model is chosen.

Legend: **Good** = fits well; **Partial** = works with limits (described); **No** = does not fit; **?** = no evidence found.

| # | Diarizer or speaker model | A: Nemotron EN (live) | B: Nemotron 3.5 multilingual (live) | C: Parakeet TDT v3 (on Stop) |
|---|---------------------------|-----------------------|--------------------------------------|------------------------------|
| 1 | Nemotron-3-Diarization | **Partial.** Live diarizer, words matched by chunk arrival time (coarse). | **Partial.** Same. Diarizer accuracy for Polish, Ukrainian, Russian not published. | **Good.** Run on the recording after Stop and match to word timestamps. Best accuracy of the NVIDIA options. |
| 2 | Streaming Sortformer v2.1 | **Partial.** Live, coarse matching, max 4 speakers. | **Partial.** Same; trained mainly on English. | **Good with limits.** Max 4 speakers; English-trained. |
| 3 | multitalker-parakeet-streaming | **Replaces A.** Built on the same English base, so it would be a new, fifth model (plus a Sortformer diarizer), not an add-on. NeMo only. | **No evidence.** English base; no multilingual version found. | **No.** Different base model. |
| 4 | VibeVoice-ASR | **Replaces all three.** It is its own speech model, not an add-on. | Same. | Same. Possible as an additional model in the menu if memory allows. |
| 5 | pyannote community-1 | **Partial.** Offline only: runs after Stop, words matched by coarse arrival time. | **Partial.** Same. Weaker than DiariZen on non-English in the independent benchmark (for 3.1). | **Good.** After Stop with word timestamps. Gated download. |
| 6 | DiariZen | **Partial.** After Stop only, coarse matching. | **Partial.** Same. | **Good.** Most accurate open option; **non-commercial weights**; separate environment likely. |
| 7 | diart | **Partial.** Live diarization with 0.5 s steps, coarse matching. | **Partial.** Same. | **Partial.** Not live-oriented here; works on a recording but offers no advantage over entries 5 and 6. |
| 8 | NeMo cascaded diarizer | **Partial.** Mostly offline; Linux/NeMo. | **Partial.** Same. | **Good in principle.** Heavy to set up on Windows. |
| 9 | sherpa-onnx diarization | **Partial.** After Stop; very light (CPU). | **Partial.** Same. | **Good.** Simple, CPU only, but accuracy figures were not found. |
| 10 | SpeechBrain ECAPA-TDNN | **Partial.** Needs a segmenter plus stored voice samples; names known people. | **Partial.** Same. | **Good** combined with timestamps and a diarizer. |

## 8. Hardware and platform notes for this computer

| Item | Finding |
|------|---------|
| VRAM (8 GB) | The diarizers are small: about 99 M (Nemotron-3-Diarization) and 117 M parameters (Sortformer v2.1). They fit next to a 0.6 B speech model. VibeVoice-ASR at 9 B does not fit unquantized. |
| Windows | NVIDIA model cards list Linux (Debian/Ubuntu-style commands) and give no Windows statement. The app's current route (`transformers`) avoids NeMo. Nemotron-3-Diarization is reported to work in `transformers` (from source). Sortformer v2.1 is documented for NeMo only. WSL2 on Windows would be the fallback for NeMo. |
| One model at a time | The app loads one speech model at a time. Adding a diarizer means two models in memory (still small for entries 1 and 2). |
| Languages | Polish, Ukrainian and Russian are not covered by any published diarization benchmark found. Diarizers mostly use voice characteristics, not language, but the independent benchmark shows large differences by language (for example pyannote 3.1 on Japanese 28.8 against DiariZen 15.6), so test on your own recordings. |
| Licenses | Most restrictive: DiariZen weights (non-commercial) and Reverb (non-commercial). pyannote is gated (token for the download). NVIDIA models use NVIDIA Open Model License or OpenMDW. VibeVoice and diart are MIT. |

## 9. Recommended path for this app

1. **First step (lowest risk):** add "dialog mode" for the **offline** model C. After Stop: transcribe with word timestamps, run **Nemotron-3-Diarization** (or pyannote community-1 if the NVIDIA model cannot be installed on Windows), match words to speakers, and show rows "Speaker 1 / Speaker 2". This covers Polish, Ukrainian and Russian through model C.
2. **Second step:** try a **live** dialog with a streaming diarizer next to model A or B. Expect label mistakes at speaker changes (chunks are coarse) and a 4 speaker limit with Sortformer (8 with Nemotron-3-Diarization).
3. **Optional:** named speakers through stored voice samples (entry 10).
4. **All-in-one test:** try VibeVoice-ASR only if a quantized or CPU variant proves usable on this hardware; otherwise it needs a bigger GPU.

Before building, test the shortlist on a short recording of your own conversation in Polish, Ukrainian or Russian, because no published number covers these languages.

## 10. Confidence and gaps

| Confidence | Items |
|------------|-------|
| High (read on the primary model card) | Nemotron-3-Diarization, Sortformer v2.1, multitalker-parakeet, VibeVoice-ASR (HF card and GitHub page), pyannote community-1 and 3.1, DiariZen (GitHub), diart (GitHub), SpeechBrain ECAPA, NeMo diarization overview, independent benchmark paper |
| Medium | sherpa-onnx (documentation pages read, but not the full docs), VibeVoice-ASR-Streaming (news line on the GitHub page, details from third-party pages that disagree on size) |
| Low or missing | Windows support for most models; VRAM figures; diarization accuracy for Polish, Ukrainian and Russian; SpeakerLM weights; whether `transformers` 5.20.2 already includes Nemotron-3-Diarization (sources conflict) |
| Added later (Hugging Face connector, tags and model cards) | Section 11 |

## 11. Addendum: second check with the Hugging Face connector

Question asked: is anything better than the top 10? Searches covered diarization, speaker-attributed speech recognition and VibeVoice by downloads, likes, trending and creation date. Short answer: **no new diarizer beats entries 1 to 6 on the evidence found, but four findings change the picture.** Nothing here was run.

### Corrections to this report

| Earlier statement | Corrected |
|-------------------|-----------|
| VibeVoice-ASR languages: "which of the 50 languages ... was not found" | The Hugging Face language tags of `microsoft/VibeVoice-ASR` list 51 languages, including **Polish (`pl`), Ukrainian (`uk`) and Russian (`ru`)**. Quality per language is still unpublished. |
| VibeVoice-ASR-Streaming sizes unknown | Hugging Face lists `VibeVoice-ASR-Streaming-1.5B` as about 2.8 B parameters (the card says "3B"), and a 7B checkpoint also exists. Both are MIT, in `transformers`, 10 languages (en, zh, es, pt, de, ja, ko, fr, ru, it). No Polish or Ukrainian. |

### New finds

| Model | What it is | Pros | Cons |
|-------|------------|------|------|
| [microsoft/VibeVoice-ASR-Streaming-1.5B](https://hf.co/microsoft/VibeVoice-ASR-Streaming-1.5B) | Live "who said what" in one model, with hotwords. Released 2 Sep 2026. | One model for words and speakers, live, about 3 B parameters so it should fit 8 GB, MIT license, `transformers` class `VibeVoiceForASRStreamingTraining`. | No Polish or Ukrainian. No accuracy numbers in the card text (technical report arXiv 2609.02812 exists). Speaker limit, VRAM, latency and Windows support not stated. Brand new. |
| [Dubedo/VibeVoice-ASR-HF-NF4](https://hf.co/Dubedo/VibeVoice-ASR-HF-NF4) | Community 4-bit (bitsandbytes NF4) build of the full VibeVoice-ASR, 5.5 GB file instead of 17.3 GB. Only the language-model part is quantized, so diarization is kept. | Brings the **offline all-in-one model with Polish, Ukrainian and Russian** close to this GPU. | The card says **7 to 8 GB of VRAM**, which is the whole 8 GB of the RTX 5060 and leaves no room for anything else. No benchmark numbers (the card says naive 4-bit breaks diarization). Needs `transformers` 5.3 or newer and `bitsandbytes` 0.48.1 or newer; Windows not stated. A community upload, not from Microsoft. |
| [diarizers-community/DiarizationLM-Gemma-4-E4B-v1](https://hf.co/diarizers-community/DiarizationLM-Gemma-4-E4B-v1) | A language model that **fixes speaker labels** in an existing transcript (speech recognition plus diarizer output). Based on Gemma 4 E4B, Apache-2.0. | Cuts the word-level speaker error in the card's tests (for example Fisher 5.32 % to 2.99 %, CALLHOME 7.74 % to 4.92 %). Works with any diarizer. A 5.3 GB 4-bit GGUF for llama.cpp, with a Windows install route (`winget install llama.cpp`). | English only (trained on four English corpora). Gains are small on meeting data (AMI 15.68 % to 14.89 %). Adds a 5 to 8 GB model, so it would not run alongside the speech model on this GPU without care. The card is inconsistent about size (8 B in metadata, "4 B dense" in text). Works on finished text, not live. |
| [christopherthompson81/sortformer_parakeet_onnx](https://hf.co/christopherthompson81/sortformer_parakeet_onnx) (the Vernacula app bundle) | ONNX versions of Parakeet TDT v3, Streaming Sortformer and Silero VAD, packaged to run together without NeMo. | Closest ready-made match to this app's idea (Parakeet plus Sortformer) and runs on `onnxruntime`. INT8 builds for CPU. | Tagged English only. 4 speakers. Windows not mentioned. Third-party packaging with mixed licenses (Parakeet weights CC-BY-4.0, Sortformer NVIDIA Open Model License, Silero MIT). |
| ONNX ports of Nemotron-3-Diarization, for example [GoodnotesLtd/nemotron-3-diarization-windows-x64-intel-onnx](https://hf.co/GoodnotesLtd/nemotron-3-diarization-windows-x64-intel-onnx), [DimQ1/nemotron-3-diarization-onnx](https://hf.co/DimQ1/nemotron-3-diarization-onnx), [christopherthompson81/nemotron3_diarization_onnx](https://hf.co/christopherthompson81/nemotron3_diarization_onnx) | Community conversions of entry 1 to ONNX; one is built for Windows x64. | This is the most promising route to run entry 1 **on Windows without NeMo**. OpenMDW license on most. | All were uploaded in October 2026 with almost no downloads and no accuracy checks. The Windows build targets Intel hardware (OpenVINO), not NVIDIA CUDA. Not verified here. |

### Seen but not recommended

| Model | Why |
|-------|-----|
| `pyannote/speaker-diarization-3.1` (6.9 M downloads) and `speaker-diarization-community-1` (5.0 M) | Already entry 5. They are by far the most used, which is a point in favor of ecosystem support. |
| Many mirrors, Core ML, LiteRT, GGUF and Axera ports of entries 1, 2 and 5 | Same models for Apple, Android or other chips. They add nothing for this Windows and NVIDIA setup. |
| `Scicom-intl/Who-Spoke-When-Over-the-Phone-Nemotron-3-Diarization` | Fine-tune for phone calls (Malay, English, Chinese). License is CC BY-NC 4.0. |
| `CIIL/Anuvani-Diarization` | DiariZen-based model for Indic languages. |
| `nvidia/diar_sortformer_4spk-v1` and `diar_streaming_sortformer_4spk-v2` | Older versions of entry 2. |

### What this changes

1. **Better than the top 10?** For diarization alone, no: the leaders are still Nemotron-3-Diarization (live, 8 speakers), then DiariZen and pyannote community-1 for offline use. The independent benchmark in section 5 is the only cross-model comparison found, and nothing newer was in it.
2. **VibeVoice moves up for your languages.** The offline model lists Polish, Ukrainian and Russian, and a 4-bit build gets it near this GPU. It is the only all-in-one option with all three of your languages. It is also unproven on this hardware (7 to 8 GB VRAM claimed) and has no published per-language numbers.
3. **Live all-in-one is real but narrower.** VibeVoice-ASR-Streaming gives live speaker labels in one model, but only for 10 languages and without Polish or Ukrainian.
4. **Windows route for entry 1:** community ONNX ports exist, which may avoid the NeMo and Linux problem. They need checking before use.

## Sources

- Hugging Face connector searches (October 2026) for "speaker diarization", "diarization", "sortformer" and "VibeVoice-ASR", and the repository pages linked in section 11.
- NVIDIA Nemotron-3-Diarization: https://huggingface.co/nvidia/Nemotron-3-Diarization
- Streaming Sortformer v2.1: https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1
- multitalker-parakeet-streaming-0.6b-v1: https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1
- VibeVoice (repository): https://github.com/microsoft/VibeVoice
- VibeVoice-ASR: https://huggingface.co/microsoft/VibeVoice-ASR
- VibeVoice-ASR paper: https://www.arxiv.org/pdf/2601.18184
- pyannote community-1: https://huggingface.co/pyannote/speaker-diarization-community-1
- pyannote 3.1: https://huggingface.co/pyannote/speaker-diarization-3.1
- DiariZen: https://github.com/BUTSpeechFIT/DiariZen
- diart: https://github.com/juanmc2005/diart
- NeMo diarization overview: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/intro.html
- sherpa-onnx diarization models: https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/models.html
- SpeechBrain ECAPA-TDNN: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
- Independent benchmark, "Benchmarking Diarization Models": https://arxiv.org/html/2509.26177v1
- Transformers docs for Nemotron ASR Streaming and Parakeet: https://huggingface.co/docs/transformers/en/model_doc/parakeet
- Third-party coverage used only for hints: https://www.unite.ai/nvidia-releases-nemotron-3-diarization-open-weight-speaker-model/ and https://www.orcarouter.ai/blog/vibevoice-asr-streaming-1-5b-explained

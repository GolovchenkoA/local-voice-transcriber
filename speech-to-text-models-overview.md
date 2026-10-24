# Speech-to-text models overview

Which speech-to-text models this app can use, and what languages they support. All three current models are from NVIDIA's Parakeet and Nemotron ASR families. Information comes from the Hugging Face model cards and the Transformers documentation, read in October 2026. Anything marked "not tested" has not been run in this project.

## Language codes used below

| Code | Meaning |
|------|---------|
| `pl` | Polish (`pl-PL` = Polish in Poland) |
| `uk` | **Ukrainian** language (`uk-UA` = Ukrainian in Ukraine). `uk` is the language code. It is not the United Kingdom. |
| `ru` | Russian (`ru-RU`) |
| `en` | English (`en-US` American, `en-GB` British; `GB` is the country code of the United Kingdom) |

## The models

| Model | Languages | Streaming (live) | Used in this app |
|-------|-----------|------------------|------------------|
| `nvidia/nemotron-speech-streaming-en-0.6b` | English only | Yes, cache-aware, chunks of 80, 160, 560 or 1120 ms | **Yes, current default.** Tested on the RTX 5060. |
| `nvidia/nemotron-3.5-asr-streaming-0.6b` | 40 language-locales, including `pl`, `uk` and `ru` | Yes, chunks of 80, 160, 320, 560 or 1120 ms | No. Not tested. Best candidate for adding Polish, Ukrainian and Russian. |
| `nvidia/parakeet-tdt-0.6b-v3` | 25 European languages, including `pl`, `uk` and `ru` | Mainly built for recorded audio. A streaming mode exists through a NeMo script. | No. Not tested. Best for transcribing finished recordings. |

All three have about 600 million parameters, so they fit easily in the 8 GB of the RTX 5060.

### `nemotron-speech-streaming-en-0.6b` (current)

- English only, with punctuation and capitalization.
- Cache-aware FastConformer RNN-T: it processes only each new audio chunk and reuses cached context, instead of re-reading overlapping windows.
- Works with Hugging Face `transformers` 5.13 or newer (class `AutoModelForRNNT`), so no NeMo install is needed on Windows.
- License: NVIDIA Open Model License.

### `nemotron-3.5-asr-streaming-0.6b` (multilingual streaming)

Support is split into three tiers by NVIDIA:

| Tier | Locales |
|------|---------|
| Transcription-ready (19) | en-US, en-GB, es-US, es-ES, fr-FR, fr-CA, it-IT, pt-BR, pt-PT, nl-NL, de-DE, tr-TR, **ru-RU**, ar-AR, hi-IN, ja-JP, ko-KR, vi-VN, **uk-UA** |
| Broad coverage (13) | **pl-PL**, sv-SE, cs-CZ, nb-NO, da-DK, bg-BG, fi-FI, hr-HR, sk-SK, zh-CN, hu-HU, ro-RO, et-EE |
| Adaptation-ready (8) | el-GR, lt-LT, lv-LV, mt-MT, sl-SI, he-IL, th-TH, nn-NO. NVIDIA says these need fine-tuning on your own data for full transcription. |

- Russian and Ukrainian are "transcription-ready". Polish is only "broad coverage".
- The language is chosen through the processor's `language` argument: a locale such as `pl-PL`, a bare code such as `pl`, or `auto`.
- Uses `AutoModelForRNNT` and `transformers` 5.13 or newer, like the English model.

### `parakeet-tdt-0.6b-v3` (offline-first multilingual)

- Supports 25 European languages, including Polish, Ukrainian and Russian.
- Designed for transcribing recordings, including long audio. It is not the best choice for live typing with this app's design.

## Accuracy

Numbers are word error rates (WER): the percentage of words that are wrong. Lower is better. They come from NVIDIA's model cards and use **different test setups**, so do not compare the two tables with each other.

### `nemotron-3.5-asr-streaming-0.6b`: WER by chunk size

Test: FLEURS with the language set explicitly.

| Language | 80 ms | 160 ms | 320 ms | 560 ms | 1120 ms |
|----------|-------|--------|--------|--------|---------|
| Polish (`pl`) | 19.88 | 18.92 | 17.48 | 16.61 | 15.15 |
| Ukrainian (`uk`) | 15.70 | 15.21 | 14.55 | 13.67 | 13.07 |
| Russian (`ru`) | 10.84 | 10.73 | 9.87 | 9.60 | 9.17 |

With automatic language detection (`auto`), errors at 1120 ms are higher: 16.55 (Polish), 14.59 (Ukrainian), 10.03 (Russian). NVIDIA notes that text normalization in the test is imperfect, so real quality may be somewhat better.

### `parakeet-tdt-0.6b-v3`: WER

Test: greedy decoding, no external language model, punctuation and capitalization removed before scoring.

| Language | FLEURS | MLS | CoVoST |
|----------|--------|-----|--------|
| Polish (`pl`) | 7.31 | 7.28 | not reported |
| Ukrainian (`uk`) | 6.79 | not reported | 5.10 |
| Russian (`ru`) | 5.51 | not reported | 3.00 |

## What the numbers tell you

- **Longer chunk, fewer mistakes.** For the streaming models, every step up in chunk size lowers the error rate. This is why the app defaults to `--latency 1120`. Choose 160 ms when you need text to appear almost instantly.
- **Language quality order:** Russian is the most accurate, then Ukrainian, then Polish.
- **Polish is the weakest** of the three in the streaming model, with roughly 15% errors at the best latency setting.
- The recorded-audio model (`parakeet-tdt-0.6b-v3`) reports much lower errors, but it is not measured on the same test and does not stream as naturally.

## Recommendation for this project

| Need | Model |
|------|-------|
| English, live | `nemotron-speech-streaming-en-0.6b` (already in the app) |
| Polish, Ukrainian, Russian or other languages, live | `nemotron-3.5-asr-streaming-0.6b` |
| Finished recordings in many languages | `parakeet-tdt-0.6b-v3` |

## Open items

- Add a `--model-dir` switch for the multilingual model, a `--language` parameter, and a language selector on the page.
- Download the multilingual model (a few GB) and confirm that the app's streaming code works with its processor and language setting. Not done yet.
- Test Polish accuracy with real speech before relying on it.

## Sources

- https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b
- https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
- https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- https://huggingface.co/docs/transformers/en/model_doc/nemotron_asr_streaming

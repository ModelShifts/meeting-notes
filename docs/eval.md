# Evaluation

Figures in the README and on the case page come from this table. They were measured on 12 November 2026 (Asia/Singapore) on an 8-core CPU (15 GB RAM, no GPU).

## Headline run — API backend, 7B notes model

- **Backend:** `api`, against a local llama.cpp server
- **Transcription:** faster-whisper `large-v3-turbo`
- **Notes:** Qwen2.5-7B-Instruct, Q4_K_M GGUF
- **Speaker labels:** sherpa-onnx (pyannote-segmentation-3.0 + CAM++)
- **Speech context:** a general list of meeting terms in `MS_STT_HOTWORDS` (see the raw log in [`eval-api.md`](eval-api.md))
- **Samples:** three synthetic meetings in `samples/`, two speakers each, made with Kokoro TTS. All content is fictional.
- **Command:** `python scripts/evaluate.py`
- **Raw table:** [`eval-api.md`](eval-api.md), [`eval-api.json`](eval-api.json)

An action item counts as found when the notes list a task that matches the scripted item (quote aligned to a transcript line). “Owner right” and “Due right” are scored only on items that were found. Precision is found items over items listed (extras lower it). Transcript error is WER for English, CER for Chinese, and a mixed rate over Chinese characters and English words for the mixed meeting.

| Sample | Lang | Transcript error | Speaker time | Speakers | Action items | Owner | Due | Precision | Decisions (listed/script) | Time (s) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| design-review-zh.wav | zh | CER 2.0% | 99% | 2/2 | 3/3 | 3/3 | 3/3 | 3/3 | 1/2 | 93 | — |
| sprint-planning-mixed.wav | mixed | MER 1.1% | 97% | 2/2 | 3/3 | 3/3 | 3/3 | 3/3 | 1/1 | 101 | — |
| weekly-sync-en.wav | en | WER 0.6% | 96% | 2/2 | 4/4 | 4/4 | 3/4 | 4/4 | 1/1 | 191 | due None for “Ask the design team for the new pictures for the app store” (want today) |
| **All** | | | | | **10/10 (100% recall)** | **10/10** | **9/10** | **10/10** | | **93–191** | |

The two models together used about 14 GB of RAM.

## Smaller models — Hugging Face backend

- **Transcription:** faster-whisper `small`
- **Notes:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Speaker labels:** auto (sherpa-onnx)
- **Raw table:** [`eval-hf.md`](eval-hf.md), [`eval-hf.json`](eval-hf.json)

| Sample | Lang | Transcript error | Speaker time | Speakers | Action items | Owner | Due | Precision | Time (s) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| design-review-zh.wav | zh | CER 6.3% | 98% | 2/2 | 2/3 | 1/2 | 1/2 | 2/4 | 36 | missed: 把新的交互稿发给大家; owner and due wrong on 整理需要调整的颜色清单 |
| sprint-planning-mixed.wav | mixed | MER 6.3% | 97% | 2/2 | 3/3 | 2/3 | 2/3 | 3/5 | 32 | owner and due wrong on “更新 sprint board” |
| weekly-sync-en.wav | en | WER 0.6% | 95% | 2/2 | 4/4 | 3/4 | 3/4 | 4/5 | 31 | owner and due wrong on “Ask the design team…” |
| **All** | | | | | **9/10 (90% recall)** | **6/9** | **6/9** | **9/14** | | |

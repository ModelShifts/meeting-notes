<p align="center">
  <img src="assets/banner.svg" alt="meeting-notes" width="100%">
</p>

<p align="center"><em>Turn a meeting recording in English, Chinese or both into a transcript, summary and action items.</em></p>

<p align="center">
  <img alt="Status: in development" src="https://img.shields.io/badge/status-in%20development-8a3a24?style=flat-square">
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-1c1a17?style=flat-square">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-1c1a17?style=flat-square">
  <img alt="Backends: API or Hugging Face" src="https://img.shields.io/badge/backends-API%20%7C%20Hugging%20Face-1c1a17?style=flat-square">
</p>

---

## Features

- **Mixed-language transcription**: English, Chinese, or both in the same sentence.
- **Speaker labels**: who said what, with timestamps.
- **Summary**: the decisions and key points in a few lines.
- **Action items**: each with an owner and a due date when one was mentioned.
- **Export**: Markdown, JSON and subtitles (SRT).

## Architecture

```mermaid
flowchart LR
  A["Recording<br/>mp3 · m4a · wav"]:::ink --> N["Normalize audio"]:::ink
  N --> T["Transcribe<br/>EN · ZH · mixed"]:::ink
  N --> SP["Speaker labels"]:::ink
  T --> M["Merge<br/>timestamped transcript"]:::ink
  SP --> M
  M --> S["Summarize +<br/>extract action items"]:::ink
  S --> O["Markdown · JSON · SRT"]:::ink
  subgraph BK["Backend: pick one, same interface"]
    direction TB
    A["(a) Bring your own API key<br/>hosted OpenAI-compatible speech + LLM APIs"]:::rust
    H["(b) Open-source on Hugging Face<br/>whisper-large-v3 · pyannote 3.1 · Qwen2.5-7B-Instruct"]:::rust
  end
  BK -.-> T
  BK -.-> S

  classDef ink fill:#f3efe6,stroke:#1c1a17,color:#1c1a17,stroke-width:1px;
  classDef rust fill:#f3efe6,stroke:#8a3a24,color:#8a3a24,stroke-width:1.5px;
```

### Two interchangeable backends

Pick one with `MS_BACKEND` in `.env`. The rest of the pipeline does not change.

| | (a) Bring your own API key | (b) Open-source on Hugging Face |
|---|---|---|
| **What** | Any OpenAI-compatible endpoint | Open models, run locally or via Hugging Face Inference |
| **Set** | `MS_BACKEND=api`, `OPENAI_BASE_URL`, `OPENAI_API_KEY` | `MS_BACKEND=hf`, `HF_MODE=local` or `inference` |
| **Good for** | Fastest start, strongest hosted models | Data stays on your machines, no per-call cost |

**Default open-source models**

| Model | Used for | License |
|---|---|---|
| [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) | transcription | Apache-2.0 |
| [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) | faster transcription | MIT |
| [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1) | speaker labels (gated: accept terms on Hugging Face) | MIT |
| [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | summary and action items | Apache-2.0 |

## Quickstart

> **Coming soon.** The commands below show the planned interface.

```bash
git clone https://github.com/ModelShifts/meeting-notes
cd meeting-notes
cp .env.example .env   # choose a backend
pip install -e .
meeting-notes run weekly-sync.m4a --out notes/
```

## Try it

> **Coming soon.** A hosted demo will be linked here and on [modelshifts.com/work](https://modelshifts.com/work).

## Roadmap

- [x] Repository scaffold and README
- [ ] Transcription with either backend
- [ ] Speaker labels
- [ ] Summary and action items
- [ ] Markdown, JSON and SRT export
- [ ] Web demo

## License

[MIT](LICENSE) © Modelshifts Technology Pte. Ltd.

<p align="center"><sub>Part of the <a href="https://modelshifts.com/work">Modelshifts</a> showcase.</sub></p>

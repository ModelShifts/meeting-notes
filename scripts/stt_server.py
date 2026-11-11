"""A small local stand-in for an OpenAI-compatible speech-to-text endpoint, for trying the api backend.

    pip install -e ".[demo]"          # FastAPI and uvicorn come with Gradio
    python scripts/stt_server.py --model small --port 8002
    # then: MS_BACKEND=api MS_API_BASE=http://127.0.0.1:8002/v1 MS_STT_MODEL=whisper-small ...

It serves ``POST /v1/audio/transcriptions`` (multipart ``file``, ``model``, ``language``,
``response_format`` = ``json`` | ``text`` | ``verbose_json``) with faster-whisper and returns the
same shape as the OpenAI API, including segment and word times for ``verbose_json``. It exists so
the api code path can be exercised without a paid key; it is not meant for production use.
"""


import argparse
import subprocess

import numpy as np


def decode(data: bytes) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1",
                          "-ar", "16000", "-"], input=data, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def build_app(model_size: str, threads: int):
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import PlainTextResponse
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=threads)
    app = FastAPI(title="local speech-to-text")

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": f"whisper-{model_size}", "object": "model"}]}

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(file: UploadFile = File(...), model_name: str = Form("whisper", alias="model"),
                             language: str | None = Form(None), response_format: str = Form("json"),
                             prompt: str | None = Form(None)):
        audio = decode(await file.read())
        segments, info = model.transcribe(audio, language=language or None, vad_filter=True, beam_size=5,
                                          condition_on_previous_text=False, word_timestamps=True,
                                          initial_prompt=prompt or None)
        segments = list(segments)
        text = "".join(s.text for s in segments).strip()
        if response_format == "text":
            return PlainTextResponse(text)
        if response_format != "verbose_json":
            return {"text": text}
        return {
            "task": "transcribe", "language": info.language, "duration": round(len(audio) / 16000, 3), "text": text,
            "segments": [{"id": i, "start": s.start, "end": s.end, "text": s.text, "avg_logprob": s.avg_logprob,
                          "no_speech_prob": s.no_speech_prob} for i, s in enumerate(segments)],
            "words": [{"word": w.word, "start": w.start, "end": w.end} for s in segments for w in (s.words or [])],
        }

    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small", help="faster-whisper model size or path (default: small)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    uvicorn.run(build_app(args.model, args.threads), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

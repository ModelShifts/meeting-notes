"""Read any audio or video file into 16 kHz mono float32 with ffmpeg, and normalise its level."""

from __future__ import annotations

import io
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def load_audio(path: str | Path) -> np.ndarray:
    """Decode with ffmpeg (mp3, m4a, wav, mp4, ...) to 16 kHz mono float32 in [-1, 1]."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is needed to read audio; install it (e.g. apt install ffmpeg, brew install ffmpeg)")
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path), "-f", "f32le", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise ValueError(f"could not decode {path.name}: {proc.stderr.decode(errors='replace').strip()[:300]}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise ValueError(f"{path.name} has no audio")
    return normalize(audio)


def normalize(audio: np.ndarray, peak: float = 0.9) -> np.ndarray:
    """Remove DC offset and scale so the loudest sample sits at ``peak`` (quiet recordings get louder)."""
    audio = audio - float(np.mean(audio))
    top = float(np.max(np.abs(audio))) if audio.size else 0.0
    if top > 1e-4:
        audio = audio * (peak / top)
    return audio.astype(np.float32)


def to_wav_bytes(audio: np.ndarray) -> bytes:
    """16-bit PCM WAV, used to send audio to a speech API."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()

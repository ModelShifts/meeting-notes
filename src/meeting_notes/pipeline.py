"""High-level helpers used by the CLI and the web demo."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

from .asr import describe_language, fix_chinese_script, get_asr
from .audio import SAMPLE_RATE, load_audio
from .config import Settings
from .diarize import assign_speakers, get_diarizer
from .llm import get_llm
from .models import Notes, Transcript
from .summarize import summarize

_DIARIZERS: dict[tuple, object] = {}
# One recording at a time: the models are shared and not safe to call from two threads at once.
_LOCK = threading.Lock()


def file_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _diarizer(settings: Settings):
    key = (settings.diarizer, settings.hf_token != "", settings.cluster_threshold, settings.model_dir)
    if key not in _DIARIZERS:
        _DIARIZERS[key] = get_diarizer(settings)
    return _DIARIZERS[key]


def transcribe(path: str | Path, settings: Settings, num_speakers: int | None = None,
               timings: dict[str, float] | None = None) -> Transcript:
    """Audio file -> speaker-labelled, timestamped transcript (no LLM involved)."""
    with _LOCK:
        return _transcribe(path, settings, num_speakers, timings if timings is not None else {})


def _transcribe(path, settings: Settings, num_speakers: int | None, timings: dict[str, float]) -> Transcript:
    t0 = time.time()
    audio = load_audio(path)
    asr = get_asr(settings)
    segments, language = asr.transcribe(audio, settings.language or None)
    segments = fix_chinese_script(segments, settings.zh_script)
    timings["transcribe"] = round(time.time() - t0, 1)
    t0 = time.time()
    diarizer = _diarizer(settings)
    n = settings.num_speakers if num_speakers is None else num_speakers
    if diarizer is not None and segments:
        segments = assign_speakers(segments, diarizer(audio, n))
        diarizer_name = diarizer.name
    else:
        segments = assign_speakers(segments, [])
        diarizer_name = "none"
    timings["speakers"] = round(time.time() - t0, 1)
    speakers = sorted({s.speaker for s in segments}, key=lambda x: int(x[1:]))
    text = " ".join(s.text for s in segments)
    return Transcript(file=Path(path).name, language=describe_language(language, text),
                      duration=round(len(audio) / SAMPLE_RATE, 2), segments=segments, speakers=speakers,
                      asr_model=asr.name, diarizer=diarizer_name)


def make_notes(transcript: Transcript, settings: Settings, timings: dict[str, float] | None = None) -> Notes:
    """Transcript -> summary, decisions and action items."""
    timings = dict(timings or {})
    t0 = time.time()
    with _LOCK:
        llm = get_llm(settings)
        summary, decisions, items = summarize(transcript, llm)
    timings["notes"] = round(time.time() - t0, 1)
    return Notes(transcript=transcript, summary=summary, decisions=decisions, action_items=items,
                 backend=settings.backend, model=llm.name, timings=timings)


def run(path: str | Path, settings: Settings, num_speakers: int | None = None) -> Notes:
    timings: dict[str, float] = {}
    transcript = transcribe(path, settings, num_speakers, timings)
    return make_notes(transcript, settings, timings)


class ResultCache:
    """Small JSON cache keyed by file content, speaker count and models. Used by the web demo."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, path: str | Path, key: str, settings: Settings) -> Path:
        tag = hashlib.sha256(
            f"{key}|{settings.backend}|{settings.model_name}|{settings.asr_name}|{settings.diarizer}".encode()
        ).hexdigest()[:16]
        return self.root / f"{file_digest(path)}-notes-{tag}.json"

    def get(self, path, key: str, settings: Settings) -> dict | None:
        p = self._path(path, key, settings)
        return json.loads(p.read_text()) if p.exists() else None

    def put(self, path, key: str, settings: Settings, data: dict) -> None:
        p = self._path(path, key, settings)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False))

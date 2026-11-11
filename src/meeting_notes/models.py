"""Data models shared by the pipeline, the exports and the web demo."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Word(BaseModel):
    start: float
    end: float
    text: str


class Segment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str = "S1"
    words: list[Word] = Field(default_factory=list, exclude=True)


class Transcript(BaseModel):
    file: str
    language: str
    duration: float
    segments: list[Segment]
    speakers: list[str]
    asr_model: str
    diarizer: str

    def lines(self) -> list[str]:
        return [f"[{clock(s.start)}] {s.speaker}: {s.text}" for s in self.segments]


class ActionItem(BaseModel):
    task: str
    owner: str | None = None
    due: str | None = None
    at: float | None = None  # seconds into the meeting where it was agreed


class Notes(BaseModel):
    transcript: Transcript
    summary: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    backend: str = ""
    model: str = ""
    timings: dict[str, float] = Field(default_factory=dict)


def clock(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


def truth(name: str) -> dict:
    return json.loads((SAMPLES / f"{name}.truth.json").read_text())


class FakeLLM:
    """Stands in for a chat model. Returns canned replies in turn and records prompts."""

    name = "fake-llm"

    def __init__(self, replies):
        self.replies = list(replies) if isinstance(replies, (list, tuple)) else [replies]
        self.prompts = []

    def complete(self, messages, max_new_tokens=None):
        self.prompts.append(messages)
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


class FakeASR:
    """Returns the script of a sample as if it had been recognised, with word times spread over each turn."""

    def __init__(self, name: str, merge_turns: bool = False):
        self.name = "fake-asr"
        self.t = truth(name)
        self.merge = merge_turns

    def transcribe(self, audio, language=None):
        from meeting_notes.models import Segment, Word

        segs = []
        for turn in self.t["turns"]:
            parts = turn["text"].split(" ") if self.t["language"] == "en" else list(turn["text"])
            step = (turn["end"] - turn["start"]) / max(1, len(parts))
            words = [Word(start=turn["start"] + i * step, end=turn["start"] + (i + 1) * step,
                          text=(" " + p) if self.t["language"] == "en" else p) for i, p in enumerate(parts)]
            segs.append(Segment(start=turn["start"], end=turn["end"], text=turn["text"], words=words))
        if self.merge:  # one long segment spanning every turn, so speaker changes must split it
            words = [w for s in segs for w in s.words]
            segs = [Segment(start=segs[0].start, end=segs[-1].end, text="".join(w.text for w in words).strip(),
                            words=words)]
        return segs, "zh" if self.t["language"] != "en" else "en"


class TruthDiarizer:
    """Returns the script's turns as diarization output."""

    name = "truth-diarizer"

    def __init__(self, name: str):
        self.t = truth(name)

    def __call__(self, audio, num_speakers=0):
        from meeting_notes.diarize import Turn

        ids = {s: i for i, s in enumerate(self.t["speakers"])}
        return [Turn(u["start"], u["end"], ids[u["speaker"]]) for u in self.t["turns"]]


@pytest.fixture
def samples() -> Path:
    return SAMPLES

"""Markdown, JSON and SRT exports."""

from __future__ import annotations

import json
from pathlib import Path

from .models import Notes, Transcript, clock

FORMATS = ("md", "json", "srt")

_LABELS = {
    "en": {"summary": "Summary", "decisions": "Decisions", "actions": "Action items", "transcript": "Transcript",
           "task": "Task", "owner": "Owner", "due": "Due", "at": "At", "none": "None recorded."},
    "zh": {"summary": "摘要", "decisions": "决定", "actions": "待办事项", "transcript": "逐字稿",
           "task": "事项", "owner": "负责人", "due": "截止", "at": "时间", "none": "无。"},
}


def _labels(language: str) -> dict[str, str]:
    return _LABELS["zh"] if language.startswith("zh") else _LABELS["en"]


def _cell(text) -> str:
    return "" if text is None else str(text).replace("|", "\\|").replace("\n", " ")


def to_markdown(notes: Notes) -> str:
    t = notes.transcript
    L = _labels(t.language)
    out = [f"# {Path(t.file).stem}", "",
           f"<sub>{clock(t.duration)} · language: {t.language} · speakers: {', '.join(t.speakers)} · "
           f"transcription: {t.asr_model} · speaker labels: {t.diarizer} · notes: {notes.model}</sub>", ""]
    out += [f"## {L['summary']}", ""] + ([f"- {p}" for p in notes.summary] or [L["none"]]) + [""]
    out += [f"## {L['decisions']}", ""] + ([f"- {d}" for d in notes.decisions] or [L["none"]]) + [""]
    out += [f"## {L['actions']}", ""]
    if notes.action_items:
        out += [f"| # | {L['task']} | {L['owner']} | {L['due']} | {L['at']} |", "|---|---|---|---|---|"]
        for i, a in enumerate(notes.action_items, start=1):
            at = clock(a.at) if a.at is not None else ""
            out.append(f"| {i} | {_cell(a.task)} | {_cell(a.owner) or '—'} | {_cell(a.due) or '—'} | {at} |")
    else:
        out.append(L["none"])
    out += ["", f"## {L['transcript']}", ""]
    out += [f"**{s.speaker}** `{clock(s.start)}` {s.text}  " for s in t.segments]
    return "\n".join(out).rstrip() + "\n"


def to_json(notes: Notes) -> str:
    return json.dumps(notes.model_dump(), indent=2, ensure_ascii=False) + "\n"


def _srt_time(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(transcript: Transcript) -> str:
    blocks = []
    for i, s in enumerate(transcript.segments, start=1):
        blocks.append(f"{i}\n{_srt_time(s.start)} --> {_srt_time(max(s.end, s.start + 0.2))}\n{s.speaker}: {s.text}\n")
    return "\n".join(blocks)


def write_all(notes: Notes, out_dir: str | Path, formats=FORMATS) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(notes.transcript.file).stem
    written = []
    for fmt in formats:
        if fmt not in FORMATS:
            raise ValueError(f"unknown format {fmt!r}; choose from {', '.join(FORMATS)}")
        path = out_dir / f"{stem}.{fmt}"
        text = {"md": to_markdown, "json": to_json}.get(fmt, lambda n: to_srt(n.transcript))(notes)
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written

"""Command-line interface.

    meeting-notes run weekly-sync.m4a --out notes/
    meeting-notes transcribe weekly-sync.m4a --srt weekly-sync.srt
    meeting-notes summarize notes/weekly-sync.json --out notes/
    meeting-notes serve
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import Settings
from .export import FORMATS, to_srt, write_all
from .models import Notes, Transcript, clock


def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    if getattr(args, "language", None):
        settings.language = args.language
    settings.validate()
    return settings


def _formats(value: str) -> list[str]:
    fmts = [f.strip().lower() for f in value.split(",") if f.strip()]
    bad = [f for f in fmts if f not in FORMATS]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown format(s) {bad}; choose from {', '.join(FORMATS)}")
    return fmts


def _print_notes(notes: Notes) -> None:
    for p in notes.summary:
        print(f"- {p}")
    if notes.decisions:
        print("\nDecisions:")
        for d in notes.decisions:
            print(f"- {d}")
    print("\nAction items:")
    for i, a in enumerate(notes.action_items, start=1):
        extra = ", ".join(x for x in (a.owner, a.due and f"due {a.due}", a.at is not None and clock(a.at)) if x)
        print(f"{i}. {a.task}" + (f" ({extra})" if extra else ""))
    if not notes.action_items:
        print("(none)")


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import run

    notes = run(args.file, _settings(args), args.speakers)
    for path in write_all(notes, args.out, args.formats):
        print(f"wrote {path}", file=sys.stderr)
    _print_notes(notes)
    t = notes.timings
    print(f"\n{clock(notes.transcript.duration)} of audio, {len(notes.transcript.segments)} segments, "
          f"speakers {', '.join(notes.transcript.speakers)}; transcribe {t.get('transcribe')}s, "
          f"speakers {t.get('speakers')}s, notes {t.get('notes')}s", file=sys.stderr)
    return 0


def cmd_transcribe(args: argparse.Namespace) -> int:
    from .pipeline import transcribe

    transcript = transcribe(args.file, _settings(args), args.speakers)
    if args.srt:
        Path(args.srt).write_text(to_srt(transcript), encoding="utf-8")
        print(f"wrote {args.srt}", file=sys.stderr)
    if args.json:
        print(json.dumps(transcript.model_dump(), indent=2, ensure_ascii=False))
    else:
        print("\n".join(transcript.lines()))
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    from .pipeline import make_notes

    data = json.loads(Path(args.file).read_text(encoding="utf-8"))
    transcript = Transcript.model_validate(data.get("transcript", data))
    notes = make_notes(transcript, _settings(args))
    for path in write_all(notes, args.out, args.formats):
        print(f"wrote {path}", file=sys.stderr)
    _print_notes(notes)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .webapp import launch

    launch(host=args.host, port=args.port, share=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="meeting-notes", description="Turn a meeting recording in English, Chinese "
                                "or both into a transcript, summary and action items.")
    p.add_argument("--version", action="version", version=f"meeting-notes {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("file")
        sp.add_argument("--speakers", type=int, default=None, help="number of speakers if known (default: estimate)")
        sp.add_argument("--language", help="force a language code, e.g. en or zh (default: detect)")

    r = sub.add_parser("run", help="transcribe, label speakers, summarise; write Markdown, JSON and SRT")
    common(r)
    r.add_argument("--out", default="notes", help="output folder (default: notes/)")
    r.add_argument("--formats", type=_formats, default=list(FORMATS), help="comma list of md,json,srt")
    r.set_defaults(func=cmd_run)

    t = sub.add_parser("transcribe", help="speaker-labelled transcript only (no LLM)")
    common(t)
    t.add_argument("--srt", help="also write subtitles here")
    t.add_argument("--json", action="store_true", help="print the transcript as JSON")
    t.set_defaults(func=cmd_transcribe)

    s = sub.add_parser("summarize", help="redo the summary and action items from a saved JSON transcript")
    s.add_argument("file", help="a JSON file written by 'run' or 'transcribe --json'")
    s.add_argument("--out", default="notes")
    s.add_argument("--formats", type=_formats, default=list(FORMATS))
    s.set_defaults(func=cmd_summarize)

    v = sub.add_parser("serve", help="start the local web demo (Gradio)")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=7860)
    v.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

"""Compare meeting-notes output with the ground truth in ``samples/*.truth.json``.

    pip install -e ".[eval]"
    python scripts/evaluate.py                  # run the configured backend on every sample
    python scripts/evaluate.py --out eval.json  # also save the raw results

On a machine without room for the speech model and the LLM at the same time, run it in two steps:

    python scripts/evaluate.py --transcribe-only --transcripts runs/   # speech model only
    python scripts/evaluate.py --transcripts runs/                     # LLM only, reads the saved transcripts

Metrics per meeting:
- Error rate of the transcript: WER for English, CER for Chinese, and for mixed meetings a
  mixed error rate (MER) over Chinese characters plus English words. Text is lower-cased,
  punctuation is removed and traditional characters are mapped to simplified first.
- Speaker accuracy: share of speech time whose predicted speaker matches the script, after
  mapping predicted labels to script labels in the best way.
- Action items: recall (script items found), owner and due-date accuracy on the found ones,
  and precision (predicted items that match a script item). An item counts as found when a
  predicted task mentions one of the item's keywords.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time
from pathlib import Path

import jiwer
from opencc import OpenCC

from meeting_notes.config import Settings
from meeting_notes.models import Notes, Transcript
from meeting_notes.pipeline import make_notes, transcribe

ROOT = Path(__file__).resolve().parent.parent
_T2S = OpenCC("t2s")
TOKEN = re.compile(r"[\u3400-\u9fff]|[a-z0-9]+(?:'[a-z]+)?")


def tokens(text: str) -> list[str]:
    return TOKEN.findall(_T2S.convert(text).lower())


def error_rate(ref: str, hyp: str) -> float:
    r, h = tokens(ref), tokens(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return jiwer.wer(" ".join(r), " ".join(h) or "<empty>")


def metric_name(language: str) -> str:
    return {"en": "WER", "zh": "CER"}.get(language, "MER")


def speaker_mapping(segments: list[dict], turns: list[dict]) -> tuple[dict[str, str], float]:
    """Best map predicted label -> script label, and the share of speech time it gets right."""
    pred = sorted({s["speaker"] for s in segments})
    truth = sorted({t["speaker"] for t in turns})

    def overlap(p, t):
        return sum(max(0.0, min(s["end"], u["end"]) - max(s["start"], u["start"]))
                   for s in segments if s["speaker"] == p for u in turns if u["speaker"] == t)

    table = {(p, t): overlap(p, t) for p in pred for t in truth}
    total = sum(u["end"] - u["start"] for u in turns)
    best, best_score = {}, -1.0
    slots = truth + [None] * max(0, len(pred) - len(truth))
    for perm in itertools.permutations(slots, len(pred)):
        score = sum(table[(p, t)] for p, t in zip(pred, perm) if t)
        if score > best_score:
            best, best_score = {p: t for p, t in zip(pred, perm) if t}, score
    return best, (best_score / total if total else 0.0)


def _norm(text: str | None) -> str:
    return re.sub(r"[\s\W_]+", "", _T2S.convert(text or "").lower())


def match_items(pred: list[dict], truth: list[dict], mapping: dict[str, str]) -> dict:
    used, found, owner_ok, due_ok, misses = set(), 0, 0, 0, []
    for item in truth:
        keys = [k.lower() for k in item["keywords"]]
        hit = next((i for i, p in enumerate(pred) if i not in used
                    and any(k in _T2S.convert(p["task"]).lower() for k in keys)), None)
        if hit is None:
            misses.append(f"missed: {item['task']}")
            continue
        used.add(hit)
        found += 1
        p = pred[hit]
        owner = mapping.get(p.get("owner") or "", p.get("owner") or "")
        if _norm(owner) == _norm(item["owner"]):
            owner_ok += 1
        else:
            misses.append(f"owner {p.get('owner')!r} for '{item['task']}' (want {item['owner']})")
        want, got = _norm(item.get("due")), _norm(p.get("due"))
        if want and got and (want in got or got in want):
            due_ok += 1
        else:
            misses.append(f"due {p.get('due')!r} for '{item['task']}' (want {item.get('due')})")
    return {"found": found, "total": len(truth), "owner_ok": owner_ok, "due_ok": due_ok,
            "predicted": len(pred), "precise": len(used), "notes": misses}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=str(ROOT / "samples"))
    ap.add_argument("--out")
    ap.add_argument("--transcripts", help="folder for transcripts: reused when present, written when missing")
    ap.add_argument("--transcribe-only", action="store_true", help="stop after writing the transcripts")
    args = ap.parse_args()
    settings = Settings.from_env()
    settings.validate()
    rows, raw = [], []
    sums = {"found": 0, "total": 0, "owner_ok": 0, "due_ok": 0, "predicted": 0, "precise": 0}
    for truth_file in sorted(Path(args.samples).glob("*.truth.json")):
        truth = json.loads(truth_file.read_text())
        audio = truth_file.with_name(truth["file"])
        t0 = time.time()
        saved = Path(args.transcripts) / f"{audio.stem}.transcript.json" if args.transcripts else None
        try:
            timings: dict[str, float] = {}
            if saved and saved.exists():
                data = json.loads(saved.read_text())
                transcript, timings = Transcript.model_validate(data["transcript"]), data["timings"]
            else:
                transcript = transcribe(audio, settings, 0, timings)
                if saved:
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    saved.write_text(json.dumps({"transcript": transcript.model_dump(), "timings": timings},
                                                indent=2, ensure_ascii=False) + "\n")
            if args.transcribe_only:
                print(f"{audio.name}: transcript saved to {saved}", file=sys.stderr)
                continue
            notes: Notes = make_notes(transcript, settings, timings)
        except Exception as exc:  # report and keep going
            rows.append((audio.name, truth["language"], "-", "-", "-", "-", "-", "-", "-", "-", "-", f"error: {exc}"))
            continue
        dt = sum(notes.timings.values())
        data = notes.model_dump()
        raw.append(data)
        segs = data["transcript"]["segments"]
        ref = " ".join(t["text"] for t in truth["turns"])
        hyp = " ".join(s["text"] for s in segs)
        er = error_rate(ref, hyp)
        mapping, spk_acc = speaker_mapping(segs, truth["turns"])
        m = match_items(data["action_items"], truth["action_items"], mapping)
        for k in sums:
            sums[k] += m[k]
        rows.append((audio.name, truth["language"], f"{metric_name(truth['language'])} {100 * er:.1f}%",
                     f"{100 * spk_acc:.0f}%", f"{len(data['transcript']['speakers'])}/{len(truth['speakers'])}",
                     f"{m['found']}/{m['total']}", f"{m['owner_ok']}/{m['found']}", f"{m['due_ok']}/{m['found']}",
                     f"{m['precise']}/{m['predicted']}", f"{len(data['decisions'])}/{len(truth['decisions'])}", f"{dt:.0f}", "; ".join(m["notes"]) or "-"))
        print(f"{audio.name}: {rows[-1][2]}, speakers {rows[-1][3]}, items {rows[-1][5]}, {dt:.0f}s", file=sys.stderr)

    if args.transcribe_only:
        return 0
    asr = raw[0]["transcript"]["asr_model"] if raw else settings.asr_name
    print(f"\nBackend: {settings.backend} · transcription: {asr} · notes: {settings.model_name} · "
          f"speaker labels: {raw[0]['transcript']['diarizer'] if raw else settings.diarizer}")
    if settings.stt_prompt or settings.stt_hotwords:
        print(f"\nSpeech context: prompt={settings.stt_prompt!r} hotwords={settings.stt_hotwords!r}")
    print()
    print("| Sample | Lang | Transcript error | Speaker time correct | Speakers found | Action items found | "
          "Owner right | Due right | Precision | Decisions (listed/script) | Time (s) | Notes |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")
    if sums["total"]:
        print(f"| **All** | | | | | **{sums['found']}/{sums['total']} ({100 * sums['found'] / sums['total']:.0f}% recall)** "
              f"| {sums['owner_ok']}/{sums['found']} | {sums['due_ok']}/{sums['found']} | "
              f"{sums['precise']}/{sums['predicted']} | | | |")
    if args.out:
        Path(args.out).write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

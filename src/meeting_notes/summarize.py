"""Summary, decisions and action items from a speaker-labelled transcript, with one LLM.

Two short prompts per part of the transcript (notes, then action items) work better
with small open models than one long prompt that asks for everything at once.
"""

from __future__ import annotations

import json
import re

from .llm import LLM, parse_json_object
from .models import ActionItem, Transcript

MAX_CHARS = 9000  # longer transcripts are read in parts, then the part summaries are merged

SYSTEM = (
    "You write meeting notes from a transcript. Speakers are labelled S1, S2, ... and each line starts with "
    "[mm:ss]. Use only what is said in the transcript. Reply with one JSON object and nothing else."
)

NOTES_PROMPT = """Read the meeting transcript and return JSON with two keys:
{{"summary": ["..."], "decisions": ["..."]}}

- "summary": 2 to 4 short points on what was discussed.
- "decisions": each thing the meeting clearly agreed or decided. Use [] if nothing was decided.
- Write in {language}.

Transcript:
{transcript}"""

ACTIONS_PROMPT = """List the action items agreed in this meeting transcript. Return JSON:
{{"action_items": [{{"task": "...", "owner": "...", "due": "...", "at": "mm:ss"}}]}}

How to fill each item:
- One item per task that someone agreed to do. Skip ideas that were only discussed, and work that was
  postponed to later.
- "owner": the label of the speaker who will do the task. "I will ..." from S2 means owner "S2". If S1 asks
  S2 to do something and S2 agrees, the owner is "S2". If the task is for a named team or group, the owner
  is that team, e.g. "test team". null if unclear.
- "due": the deadline as said, e.g. "Friday" or "下周二"; null if no deadline was said.
- "at": the [mm:ss] of the line where the task was agreed.
- Write "task" in {language}.

Example. Transcript:
[00:05] S1: Can you send the budget sheet before Monday?
[00:08] S2: Yes, I will send it.
[00:12] S1: And the support team should check the open tickets.
Answer:
{{"action_items": [{{"task": "Send the budget sheet", "owner": "S2", "due": "Monday", "at": "00:08"}},
 {{"task": "Check the open tickets", "owner": "support team", "due": null, "at": "00:12"}}]}}

Transcript:
{transcript}"""

MERGE = """Below are notes from consecutive parts of one meeting. Merge them into one JSON object with keys
"summary" (2 to 5 short points for the whole meeting) and "decisions" (all decisions, without duplicates).
Write in {language}. Reply with JSON only.

{parts}"""

LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Simplified Chinese",
    "zh+en": "Simplified Chinese, keeping English technical terms as spoken",
}


def output_language(language: str) -> str:
    return LANGUAGE_NAMES.get(language, "the language of the meeting")


def chunk_lines(lines: list[str], max_chars: int = MAX_CHARS) -> list[list[str]]:
    chunks, cur, size = [], [], 0
    for line in lines:
        if cur and size + len(line) > max_chars:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        chunks.append(cur)
    return chunks


def parse_clock(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"(?:(\d+):)?(\d{1,2}):(\d{2})", str(value))
    if not m:
        return None
    h, mi, s = int(m.group(1) or 0), int(m.group(2)), int(m.group(3))
    return float(h * 3600 + mi * 60 + s)


def _clean_str(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "n/a", "unknown", "无", "未提及", "未知"}:
        return None
    return text


def normalize_owner(owner, speakers: list[str]) -> str | None:
    owner = _clean_str(owner)
    if owner is None:
        return None
    m = re.fullmatch(r"(?:s|speaker\s*|说话人\s*)(\d+)", owner.strip(), flags=re.I)
    if m:
        label = f"S{int(m.group(1))}"
        return label if not speakers or label in speakers else owner
    return owner


def _as_list(value) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def parse_items(raw, speakers: list[str]) -> list[ActionItem]:
    items = []
    for it in raw if isinstance(raw, list) else []:
        if isinstance(it, str):
            it = {"task": it}
        if not isinstance(it, dict) or not _clean_str(it.get("task")):
            continue
        items.append(ActionItem(task=_clean_str(it.get("task")), owner=normalize_owner(it.get("owner"), speakers),
                                due=_clean_str(it.get("due")), at=parse_clock(it.get("at"))))
    return items


def _dedupe(texts: list[str]) -> list[str]:
    seen, out = set(), []
    for t in texts:
        k = re.sub(r"\W+", "", t.lower())
        if k and k not in seen:
            seen.add(k)
            out.append(t)
    return out


def _ask(llm: LLM, prompt: str) -> dict:
    reply = llm.complete([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
    try:
        return parse_json_object(reply)
    except (ValueError, json.JSONDecodeError):
        return {}


def summarize(transcript: Transcript, llm: LLM, max_chars: int = MAX_CHARS) -> tuple[list[str], list[str], list[ActionItem]]:
    """Return (summary points, decisions, action items)."""
    language = output_language(transcript.language)
    parts = []
    items: list[ActionItem] = []
    for lines in chunk_lines(transcript.lines(), max_chars):
        text = "\n".join(lines)
        data = _ask(llm, NOTES_PROMPT.format(language=language, transcript=text))
        parts.append({"summary": _as_list(data.get("summary")), "decisions": _as_list(data.get("decisions"))})
        data = _ask(llm, ACTIONS_PROMPT.format(language=language, transcript=text))
        items += parse_items(data.get("action_items"), transcript.speakers)
    if len(parts) == 1:
        summary, decisions = parts[0]["summary"], parts[0]["decisions"]
    else:
        data = _ask(llm, MERGE.format(language=language, parts=json.dumps(parts, ensure_ascii=False)))
        summary = _as_list(data.get("summary")) or [p for part in parts for p in part["summary"]]
        decisions = _as_list(data.get("decisions")) or [d for part in parts for d in part["decisions"]]
    seen, unique = set(), []
    for it in items:
        k = (re.sub(r"\W+", "", it.task.lower()), it.owner)
        if k not in seen:
            seen.add(k)
            unique.append(it)
    return _dedupe(summary), _dedupe(decisions), unique

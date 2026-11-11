"""Summary, decisions and action items from a speaker-labelled transcript, with one LLM.

Two short prompts per part of the transcript (notes, then action items) work better
with open models than one long prompt that asks for everything at once. Each action item
and decision must quote the transcript line it comes from; items whose quote is not found
in the transcript are dropped, and the quoted line gives the item its time.
"""

from __future__ import annotations

import json
import re

from .llm import LLM, parse_json_object
from .models import ActionItem, Transcript

MAX_CHARS = 9000  # longer transcripts are read in parts, then the part summaries are merged

SYSTEM = (
    "You write meeting notes from a transcript. Speakers are labelled S1, S2, ... and each line starts with "
    "[mm:ss]. Report only what the transcript says. Never add tasks, decisions, people, dates or details that "
    "are not in it. Reply with one JSON object and nothing else."
)

NOTES_PROMPT = """Read the meeting transcript and return JSON:
{{"summary": ["..."], "decisions": [{{"text": "...", "quote": "..."}}]}}

- "summary": 2 to 4 short points on what was discussed, using only facts stated in the transcript.
- "decisions": each thing the participants settled in this meeting: a date, a priority, a scope, a change of
  plan or an approach that a speaker stated as settled ("we will ...", "let's ...", "我们定在...", "放到下一个...")
  and nobody objected to, or that someone confirmed ("agreed", "that is decided", "好的，就这么定了").
  A confirmation at the end can settle several earlier proposals; list each of them. List each decision once.
  Do not list problems, opinions, open questions, proposals that were rejected or left open, or tasks that one
  person will do (those are action items).
- "text": the decision as one short complete sentence, in your own words but with only facts that were said.
  Do not copy filler words such as "that is decided" or "那".
- "quote": copy the exact words from the transcript line that states the decision.
- Use [] if nothing was decided.
- Write "summary" and "text" in {language}.

Example. Transcript:
[00:02] S1: The venue is too small, let's book the larger hall instead.
[00:06] S2: Fine. And I will call the caterer.
[00:09] S1: We could also add a second day.
[00:11] S2: Maybe next year.
Answer:
{{"summary": ["The venue is too small; a second day was suggested but not planned."],
 "decisions": [{{"text": "Book the larger hall instead of the current venue.",
   "quote": "let's book the larger hall instead"}}]}}

Transcript:
{transcript}"""

ACTIONS_PROMPT = """List the action items in this meeting transcript. Return JSON:
{{"action_items": [{{"task": "...", "owner": "...", "due": "...", "quote": "..."}}]}}

An action item is a concrete piece of work that a specific person or team will do after the meeting.
Include an item only when:
- a speaker commits to it ("I will ...", "I'll ...", "我来...", "我会...", "我...之前把..."), or
- a speaker asks someone to do it and that person agrees ("Sure", "Yes", "可以", "好的"), or
- the meeting assigns it to a named team or group.
Do not include: decisions about dates, priorities or scope (for example moving a release or postponing a
feature), problems that were only described, ideas or suggestions nobody committed to, or work that was
put off to a later time. A line that agrees with a plan and also takes on work ("OK, agreed, I will ...")
does give an action item. Check every line, including the last ones.

List each piece of work once. When a request is accepted, the item belongs to the person who agreed:
use their speaker label as owner and quote their reply, not the request.

Fields:
- "task": the work, in a few words, as said. Do not add details that were not said.
- "owner": the speaker label of the person who will do it (the one who said "I will", or the one who agreed
  to a request), or the team name in the words of the transcript (e.g. "test team", "测试团队"). null if
  nobody took it on.
- "due": the deadline for this task in the words of the transcript, not translated ("Wednesday", "下周二",
  "今天下班之前"). It can be
  said in the request or in the reply. null if no deadline was said for this task. Never reuse a deadline
  that belongs to another task.
- "quote": copy the exact words from the transcript line where the task was committed to, agreed or assigned.
- Write "task" in {language}.

Example. Transcript:
[00:05] S1: Can you send the budget sheet?
[00:08] S2: Yes, I will send it before Monday.
[00:10] S1: We could also redo the logo at some point.
[00:12] S1: And the support team should check the open tickets.
[00:15] S2: Then the launch moves to May.
[00:18] S1: Agreed. I'll book the room for Thursday.
Answer:
{{"action_items": [
 {{"task": "Send the budget sheet", "owner": "S2", "due": "Monday", "quote": "Yes, I will send it before Monday."}},
 {{"task": "Check the open tickets", "owner": "support team", "due": null,
   "quote": "the support team should check the open tickets"}},
 {{"task": "Book the room", "owner": "S1", "due": "Thursday", "quote": "Agreed. I'll book the room for Thursday."}}]}}

Transcript:
{transcript}"""

MERGE = """Below are notes from consecutive parts of one meeting. Merge them into one JSON object with the key
"summary" (2 to 5 short points for the whole meeting). Use only facts in the notes.
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


_TOKEN = re.compile(r"[\u3400-\u9fff]|[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    """Chinese characters and lower-cased Latin words; traditional characters folded to simplified."""
    from opencc import OpenCC

    global _T2S
    if _T2S is None:
        _T2S = OpenCC("t2s")
    return _TOKEN.findall(_T2S.convert(str(text)).lower())


_T2S = None


def _grams(tokens: list[str]) -> set[tuple[str, ...]]:
    if len(tokens) < 2:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + 2]) for i in range(len(tokens) - 1)}


def find_quote(quote, transcript: Transcript, min_score: float = 0.6):
    """The segment that contains ``quote`` (allowing small differences), or ``None``.

    The score is the share of the quote's token pairs (Chinese characters, English words) that occur
    in one segment (or a segment and the next), so a quote that was paraphrased or invented scores low.
    """
    grams = _grams(_tokens(quote or ""))
    if not grams:
        return None
    segs = transcript.segments

    def score(text: str) -> float:
        return len(grams & _grams(_tokens(text))) / len(grams)

    singles = [score(seg.text) for seg in segs]
    best_score = max(singles, default=0.0)
    best = segs[singles.index(best_score)] if segs else None
    if best_score < min_score:
        # a quote can run over into the next segment when the recogniser split a sentence
        for i in range(len(segs) - 1):
            sc = score(segs[i].text + " " + segs[i + 1].text)
            if sc > best_score:
                best, best_score = segs[i], sc
    return best if best_score >= min_score else None


def grounded_items(raw, transcript: Transcript) -> list[ActionItem]:
    """Parse action items and keep only those whose quote is found in the transcript."""
    items = []
    for it in raw if isinstance(raw, list) else []:
        if not isinstance(it, dict) or not _clean_str(it.get("task")):
            continue
        seg = find_quote(it.get("quote"), transcript)
        if seg is None:
            continue
        items.append(ActionItem(task=_clean_str(it.get("task")),
                                owner=normalize_owner(it.get("owner"), transcript.speakers),
                                due=_clean_str(it.get("due")), at=seg.start))
    return items


def _overlap(a: str, b: str) -> float:
    ta, tb = set(_tokens(a)), set(_tokens(b))
    return len(ta & tb) / min(len(ta), len(tb)) if ta and tb else 0.0


def drop_requests(items: list[ActionItem], transcript: Transcript) -> list[ActionItem]:
    """Drop an item taken from a request when the next speaker's reply that accepts it is also an item."""
    index = {seg.start: i for i, seg in enumerate(transcript.segments)}
    segs = transcript.segments
    out = []
    for it in items:
        i = index.get(it.at)
        later = [o for o in items if o is not it and o.at is not None and i is not None
                 and index.get(o.at) is not None and 0 < index[o.at] - i <= 2
                 and segs[index[o.at]].speaker != segs[i].speaker and _overlap(it.task, o.task) >= 0.5]
        if not later:
            out.append(it)
    return out


def _task_lines(items: list[ActionItem], transcript: Transcript) -> set[float]:
    segs = transcript.segments
    index = {seg.start: i for i, seg in enumerate(segs)}
    lines = set()
    for it in items:
        lines.add(it.at)
        i = index.get(it.at)
        if i and segs[i - 1].speaker != segs[i].speaker and _overlap(it.task, segs[i - 1].text) >= 0.5:
            lines.add(segs[i - 1].start)
    return lines


def grounded_decisions(raw, transcript: Transcript) -> list[tuple[str, float | None]]:
    """(decision text, time of the quoted line) for decisions whose quote is found in the transcript.

    Plain strings are kept as they are, with no time.
    """
    out = []
    for d in raw if isinstance(raw, list) else [raw]:
        if isinstance(d, str):
            if d.strip():
                out.append((d.strip(), None))
        elif isinstance(d, dict) and _clean_str(d.get("text")):
            seg = find_quote(d.get("quote"), transcript)
            if seg is not None:
                out.append((_clean_str(d.get("text")), seg.start))
    return out


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
        parts.append({"summary": _as_list(data.get("summary")),
                      "decisions": grounded_decisions(data.get("decisions") or [], transcript)})
        data = _ask(llm, ACTIONS_PROMPT.format(language=language, transcript=text))
        items += grounded_items(data.get("action_items"), transcript)
    if len(parts) == 1:
        summary, decisions = parts[0]["summary"], parts[0]["decisions"]
    else:
        data = _ask(llm, MERGE.format(language=language, parts=json.dumps(parts, ensure_ascii=False)))
        summary = _as_list(data.get("summary")) or [p for part in parts for p in part["summary"]]
        # the merged decisions may only restate decisions that were found in the parts
        decisions = [d for part in parts for d in part["decisions"]]
    # an item needs an owner: work that nobody took on is not an action item
    items = drop_requests([it for it in items if it.owner], transcript)
    seen, unique = set(), []
    for it in items:
        k = (re.sub(r"\W+", "", it.task.lower()), it.owner)
        if k not in seen:
            seen.add(k)
            unique.append(it)
    # a "decision" quoted from an action item's line, or from the request that the item accepts,
    # is that task, not a separate decision
    task_lines = _task_lines(unique, transcript)
    decisions = [text for text, at in decisions if at is None or at not in task_lines]
    unique.sort(key=lambda it: it.at if it.at is not None else float("inf"))  # in meeting order
    return _dedupe(summary), _dedupe(decisions), unique

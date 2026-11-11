import json

from meeting_notes.diarize import assign_speakers
from meeting_notes.export import to_markdown, to_srt, write_all
from meeting_notes.llm import parse_json_object
from meeting_notes.models import Notes, Transcript
from meeting_notes.summarize import ACTIONS_PROMPT, chunk_lines, find_quote, normalize_owner, parse_clock, summarize

from conftest import FakeASR, FakeLLM, TruthDiarizer

NOTES_REPLY = json.dumps({"summary": ["The release moves."], "decisions": [
    {"text": "Release on Tuesday.", "quote": "we move the release from Friday to next Tuesday"},
    {"text": "Invented decision.", "quote": "we will hire three more people"},
]})
ITEMS_REPLY = json.dumps({"action_items": [
    {"task": "Fix the offline sync bug", "owner": "Speaker 2", "due": "Wednesday",
     "quote": "I will fix the offline sync bug and push a new build by Wednesday."},
    {"task": "Run the regression suite", "owner": "test team", "due": "null",
     "quote": "I would like the test team to run the full regression suite"},
    {"task": "Book a team dinner", "owner": "S1", "due": "Friday", "quote": "Let's book a team dinner on Friday."},
    {"task": "", "owner": "S1", "quote": "Sure, I will ask them today."},
]})


def transcript(name="weekly-sync-en") -> Transcript:
    asr = FakeASR(name)
    segs, lang = asr.transcribe(None)
    segs = assign_speakers(segs, TruthDiarizer(name)(None))
    return Transcript(file=asr.t["file"], language="en" if lang == "en" else "zh+en", duration=asr.t["duration"],
                      segments=segs, speakers=["S1", "S2"], asr_model="fake-asr", diarizer="truth")


def test_summarize_keeps_only_grounded_items_and_decisions():
    llm = FakeLLM([NOTES_REPLY, ITEMS_REPLY])
    t = transcript()
    summary, decisions, items = summarize(t, llm)
    assert summary == ["The release moves."]
    assert decisions == ["Release on Tuesday."]  # the decision whose quote is not in the transcript is dropped
    assert [(i.task, i.owner, i.due) for i in items] == [("Fix the offline sync bug", "S2", "Wednesday"),
                                                         ("Run the regression suite", "test team", None)]
    # the time comes from the quoted line, not from the model
    starts = {s.text[:20]: s.start for s in t.segments}
    assert items[0].at == starts["About two days. I wi"] and items[1].at == starts["One more thing. The "]
    prompt = llm.prompts[1][1]["content"]
    assert "[00:14] S2: About two days." in prompt and "Write \"task\" in English" in prompt


def test_summarize_asks_for_chinese_on_mixed_meetings():
    llm = FakeLLM([NOTES_REPLY, ITEMS_REPLY])
    summarize(transcript("sprint-planning-mixed"), llm)
    assert "Simplified Chinese, keeping English technical terms" in llm.prompts[0][1]["content"]


def test_long_transcripts_are_read_in_parts_and_merged():
    t = transcript()
    replies = []
    parts = chunk_lines(t.lines(), max_chars=300)
    assert len(parts) > 1
    for _ in parts:
        replies += [NOTES_REPLY, ITEMS_REPLY]
    replies.append(json.dumps({"summary": ["Merged."], "decisions": ["Release on Tuesday."]}))
    llm = FakeLLM(replies)
    summary, decisions, items = summarize(t, llm, max_chars=300)
    assert summary == ["Merged."] and len(llm.prompts) == 2 * len(parts) + 1
    assert len(items) <= 2  # duplicates across parts are removed, quotes are checked per transcript


def test_bad_json_is_repaired_or_ignored():
    broken = '```json\n{"summary": ["a", "b",\n "decisions": ["c"]}\n```'
    assert parse_json_object(broken)["summary"][:2] == ["a", "b"]
    llm = FakeLLM(["no json here", "still nothing"])
    assert summarize(transcript(), llm) == ([], [], [])


def test_find_quote_tolerates_small_differences_but_not_invention():
    t = transcript()
    assert find_quote("i will fix the offline sync bug, and push a new build by wednesday", t).speaker == "S2"
    assert find_quote("We agreed to cancel the release entirely.", t) is None
    assert find_quote("", t) is None
    zh = transcript("design-review-zh")
    assert find_quote("可以，我周四之前整理好颜色清单", zh).speaker == "S1"


def test_helpers():
    assert parse_clock("[01:02:03]") == 3723.0 and parse_clock("bad") is None and parse_clock(5) == 5.0
    assert normalize_owner("说话人2", ["S1", "S2"]) == "S2" and normalize_owner("unknown", []) is None
    assert "{" in ACTIONS_PROMPT.format(language="English", transcript="x")


def test_exports(tmp_path):
    t = transcript()
    summary, decisions, items = summarize(t, FakeLLM([NOTES_REPLY, ITEMS_REPLY]))
    notes = Notes(transcript=t, summary=summary, decisions=decisions, action_items=items, backend="api", model="fake")
    md = to_markdown(notes)
    assert "## Action items" in md and "| 1 | Fix the offline sync bug | S2 | Wednesday | 00:14 |" in md
    srt = to_srt(t)
    assert srt.startswith("1\n00:00:00,500 --> ") and "S2: Right." in srt
    paths = write_all(notes, tmp_path)
    assert sorted(p.suffix for p in paths) == [".json", ".md", ".srt"]
    back = Notes.model_validate(json.loads((tmp_path / "weekly-sync-en.json").read_text()))
    assert back.action_items[0].owner == "S2"


def test_chinese_markdown_headings():
    t = transcript("design-review-zh")
    t.language = "zh"
    md = to_markdown(Notes(transcript=t))
    assert "## 待办事项" in md and "## 逐字稿" in md


def test_requests_answered_by_an_accepted_item_are_not_listed_twice():
    t = transcript()
    lines = {s.text.split()[0]: s.text for s in t.segments}
    items = json.dumps({"action_items": [
        {"task": "Ask the design team for the new pictures", "owner": "S2", "quote": lines["Could"]},
        {"task": "Ask the design team for pictures", "owner": "S1", "due": "today", "quote": lines["Sure,"]},
        {"task": "Move the release", "owner": None, "quote": lines["Okay."]},
    ]})
    notes = json.dumps({"summary": ["s"], "decisions": [
        {"text": "Design team sends pictures.", "quote": lines["Could"]},
        {"text": "Release on Tuesday.", "quote": "we move the release from Friday to next Tuesday"}]})
    summary, decisions, got = summarize(t, FakeLLM([notes, items]))
    # the request line is dropped in favour of the reply, and an item that nobody took on is dropped
    assert [(i.task, i.owner, i.due) for i in got] == [("Ask the design team for pictures", "S1", "today")]
    # a "decision" quoted from the request that a task accepts is that task, not a decision
    assert decisions == ["Release on Tuesday."]

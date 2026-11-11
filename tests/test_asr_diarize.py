from meeting_notes.asr import build_prompt, describe_language, fix_chinese_script, is_hallucination, parse_verbose_json
from meeting_notes.diarize import Turn, assign_speakers
from meeting_notes.models import Segment, Word

from conftest import FakeASR, TruthDiarizer


def test_parse_verbose_json_attaches_words_and_drops_credits():
    data = {
        "language": "english",
        "segments": [{"start": 0.0, "end": 2.0, "text": " Hello there."},
                     {"start": 2.5, "end": 4.0, "text": " Thanks for watching!"}],
        "words": [{"word": " Hello", "start": 0.1, "end": 0.5}, {"word": " there.", "start": 0.6, "end": 1.0}],
    }
    segs, lang = parse_verbose_json(data, 4.0)
    assert lang == "en" and [s.text for s in segs] == ["Hello there."]
    assert [w.text for w in segs[0].words] == [" Hello", " there."]


def test_parse_verbose_json_text_only():
    segs, lang = parse_verbose_json({"text": "只有文字", "language": "zh"}, 3.0)
    assert len(segs) == 1 and segs[0].end == 3.0 and lang == "zh"


def test_hallucination_filter():
    assert is_hallucination("字幕by某某")
    assert is_hallucination("Thank you for watching.")
    assert not is_hallucination("这个字幕的颜色要改一下")
    assert is_hallucination("whatever", no_speech_prob=0.9, avg_logprob=-1.5)


def test_language_description():
    assert describe_language("zh", "我们开始这个 sprint 的 planning 吧，先看 backlog 里的 ticket") == "zh+en"
    assert describe_language("zh", "大家好，今天我们评审一下新版设置页面的设计稿") == "zh"
    assert describe_language("en", "Let's move the release to Friday") == "en"


def test_traditional_to_simplified():
    segs = [Segment(start=0, end=1, text="這個我來修", words=[Word(start=0, end=1, text="這個")])]
    fix_chinese_script(segs, "simplified")
    assert segs[0].text == "这个我来修" and segs[0].words[0].text == "这个"


def test_assign_speakers_labels_in_order_of_appearance():
    asr = FakeASR("weekly-sync-en")
    segs, _ = asr.transcribe(None)
    out = assign_speakers(segs, TruthDiarizer("weekly-sync-en")(None))
    assert [s.speaker for s in out] == [u["speaker"] for u in asr.t["turns"]]


def test_assign_speakers_splits_a_segment_at_speaker_changes():
    asr = FakeASR("design-review-zh", merge_turns=True)
    segs, _ = asr.transcribe(None)
    assert len(segs) == 1
    out = assign_speakers(segs, TruthDiarizer("design-review-zh")(None))
    assert [s.speaker for s in out] == [u["speaker"] for u in asr.t["turns"]]
    assert out[3].text == asr.t["turns"][3]["text"]


def test_assign_speakers_absorbs_tiny_fragments():
    words = [Word(start=i * 0.3, end=i * 0.3 + 0.25, text=f" w{i}") for i in range(6)]
    seg = Segment(start=0, end=1.8, text="w0 w1 w2 w3 w4 w5", words=words)
    turns = [Turn(0.0, 1.45, 0), Turn(1.45, 1.6, 1), Turn(1.6, 1.8, 0)]
    out = assign_speakers([seg], turns)
    assert len(out) == 1 and out[0].speaker == "S1"


def test_no_diarizer_gives_one_speaker():
    segs, _ = FakeASR("weekly-sync-en").transcribe(None)
    assert {s.speaker for s in assign_speakers(segs, [])} == {"S1"}


def test_speech_prompt_combines_chinese_hint_and_user_context():
    assert build_prompt("en") is None
    assert build_prompt("en", "Weekly sync of the payments team.") == "Weekly sync of the payments team."
    assert build_prompt("zh", "术语：sprint、cache").startswith("以下是会议记录") and "cache" in build_prompt("zh", "术语：sprint、cache")


def test_api_transcriber_sends_prompt_and_terms(monkeypatch):
    from meeting_notes import asr as asr_mod
    from meeting_notes.config import Settings
    import numpy as np

    sent = {}

    class FakeTranscriptions:
        def create(self, **kw):
            sent.update(kw)
            return {"language": "zh", "segments": [{"start": 0, "end": 1, "text": "好的"}]}

    class FakeClient:
        def __init__(self, **kw):
            self.audio = type("A", (), {"transcriptions": FakeTranscriptions()})()

    import openai
    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    s = Settings(backend="api", api_base="http://x/v1", model="m", stt_model="stt", stt_api_base="http://y/v1",
                 stt_hotwords="sprint, cache")
    t = asr_mod.APITranscriber(s)
    segs, lang = t.transcribe(np.zeros(1600, np.float32), language="zh")
    assert lang == "zh" and segs[0].text == "好的"
    assert sent["model"] == "stt" and sent["language"] == "zh" and "Terms: sprint, cache." in sent["prompt"]


def test_a_fragment_that_opens_a_segment_joins_the_rest_of_it():
    # word timestamps run early, so the first word of S2's sentence falls inside S1's turn
    words = [Word(start=0.0, end=0.4, text="Then"), Word(start=0.5, end=0.9, text=" I"),
             Word(start=1.0, end=1.5, text=" will"), Word(start=1.6, end=2.0, text=" go.")]
    seg = Segment(start=0.0, end=2.0, text="Then I will go.", words=words)
    first = Segment(start=-2.0, end=-0.5, text="Hello.", words=[Word(start=-2.0, end=-0.5, text="Hello.")])
    out = assign_speakers([first, seg], [Turn(-2.0, 0.45, 0), Turn(0.45, 2.0, 1)])
    assert [(s.speaker, s.text) for s in out] == [("S1", "Hello."), ("S2", "Then I will go.")]

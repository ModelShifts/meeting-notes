from meeting_notes.asr import describe_language, fix_chinese_script, is_hallucination, parse_verbose_json
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

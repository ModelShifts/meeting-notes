import json

import pytest

from meeting_notes import cli, pipeline

from conftest import FakeASR, FakeLLM, TruthDiarizer, truth

NOTES_REPLY = json.dumps({"summary": ["Release planning."], "decisions": ["Move the release to Tuesday."]})
ITEMS_REPLY = json.dumps({"action_items": [{"task": "Fix the offline sync bug", "owner": "S2", "due": "Wednesday",
                                            "at": "00:14"}]})


@pytest.fixture
def fake_models(monkeypatch):
    def patch(name):
        monkeypatch.setattr(pipeline, "get_asr", lambda s: FakeASR(name))
        monkeypatch.setattr(pipeline, "_diarizer", lambda s: TruthDiarizer(name))
        monkeypatch.setattr(pipeline, "get_llm", lambda s: FakeLLM([NOTES_REPLY, ITEMS_REPLY]))
        monkeypatch.setenv("MS_BACKEND", "api")
        monkeypatch.setenv("MS_API_BASE", "http://localhost:0/v1")
        monkeypatch.setenv("MS_MODEL", "fake")
    return patch


def test_run_writes_all_formats(fake_models, samples, tmp_path, capsys):
    fake_models("weekly-sync-en")
    assert cli.main(["run", str(samples / "weekly-sync-en.wav"), "--out", str(tmp_path)]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["weekly-sync-en.json", "weekly-sync-en.md",
                                                         "weekly-sync-en.srt"]
    out = capsys.readouterr().out
    assert "1. Fix the offline sync bug (S2, due Wednesday, 00:14)" in out
    data = json.loads((tmp_path / "weekly-sync-en.json").read_text())
    assert data["transcript"]["speakers"] == ["S1", "S2"] and data["transcript"]["diarizer"] == "truth-diarizer"


def test_transcribe_and_summarize_roundtrip(fake_models, samples, tmp_path, capsys):
    fake_models("design-review-zh")
    assert cli.main(["transcribe", str(samples / "design-review-zh.wav"), "--json", "--srt",
                     str(tmp_path / "t.srt")]) == 0
    printed = capsys.readouterr().out
    (tmp_path / "t.json").write_text(printed)
    t = json.loads(printed)
    assert t["language"] == "zh" and len(t["segments"]) == len(truth("design-review-zh")["turns"])
    assert (tmp_path / "t.srt").read_text().count("-->") == len(t["segments"])
    assert cli.main(["summarize", str(tmp_path / "t.json"), "--out", str(tmp_path / "n"), "--formats", "md"]) == 0
    assert (tmp_path / "n" / "design-review-zh.md").exists()


def test_cli_errors(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MS_BACKEND", "api")
    monkeypatch.delenv("MS_API_BASE", raising=False)
    monkeypatch.setenv("MS_API_BASE", "")
    assert cli.main(["run", str(tmp_path / "x.wav")]) == 2
    assert "MS_BACKEND=api needs MS_API_BASE" in capsys.readouterr().err
    monkeypatch.setenv("MS_BACKEND", "nope")
    assert cli.main(["transcribe", str(tmp_path / "x.wav")]) == 2
    with pytest.raises(SystemExit):
        cli.main(["run", "x.wav", "--formats", "pdf"])


def test_web_demo_builds_and_runs(fake_models, samples, tmp_path):
    gr = pytest.importorskip("gradio")
    fake_models("sprint-planning-mixed")
    from meeting_notes import webapp
    from meeting_notes.config import Settings

    settings = Settings.from_env()
    settings.cache_dir = str(tmp_path / "cache")
    demo = webapp.build_app(settings)
    assert isinstance(demo, gr.Blocks)
    notes = pipeline.run(samples / "sprint-planning-mixed.wav", settings)
    html = webapp.transcript_html(notes)
    assert "ms-timeline" in html and html.count("class='ms-line") == len(notes.transcript.segments)
    assert "A1" in html and webapp.item_rows(notes)[0][:3] == ["A1", "Fix the offline sync bug", "S2"]
    files = webapp.export_files(notes)
    assert [f.rsplit(".", 1)[1] for f in files] == ["md", "json", "srt"]
    assert "Sprint planning" in webapp.sample_title(samples / "sprint-planning-mixed.wav")

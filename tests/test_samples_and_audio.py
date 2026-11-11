import json
import wave

import numpy as np
import pytest
import yaml

from meeting_notes.audio import SAMPLE_RATE, load_audio, normalize, to_wav_bytes

from conftest import ROOT, SAMPLES, truth

NAMES = ["weekly-sync-en", "design-review-zh", "sprint-planning-mixed"]


@pytest.mark.parametrize("name", NAMES)
def test_truth_matches_script(name):
    script = yaml.safe_load((SAMPLES / "scripts" / f"{name}.yaml").read_text())
    t = truth(name)
    assert [u["text"] for u in t["turns"]] == [text for _, text in script["turns"]]
    assert [u["speaker"] for u in t["turns"]] == [spk for spk, _ in script["turns"]]
    assert t["action_items"] == script["action_items"]
    assert len(set(script["voices"].values())) == 2, "two distinct voices"
    times = [(u["start"], u["end"]) for u in t["turns"]]
    assert all(a < b for a, b in times) and all(times[i][1] < times[i + 1][0] for i in range(len(times) - 1))


@pytest.mark.parametrize("name", NAMES)
def test_sample_audio_is_16k_mono_and_matches_duration(name):
    t = truth(name)
    with wave.open(str(SAMPLES / t["file"])) as w:
        assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1
        assert abs(w.getnframes() / SAMPLE_RATE - t["duration"]) < 0.05


def test_load_audio_decodes_and_normalises():
    audio = load_audio(SAMPLES / "weekly-sync-en.wav")
    assert audio.dtype == np.float32 and abs(len(audio) / SAMPLE_RATE - truth("weekly-sync-en")["duration"]) < 0.05
    assert abs(float(np.abs(audio).max()) - 0.9) < 1e-3


def test_normalize_lifts_quiet_audio_and_removes_dc():
    quiet = (0.01 * np.sin(np.linspace(0, 100, 16000)) + 0.2).astype(np.float32)
    out = normalize(quiet)
    assert abs(float(out.mean())) < 1e-3 and abs(float(np.abs(out).max()) - 0.9) < 1e-3


def test_wav_bytes_roundtrip(tmp_path):
    audio = np.zeros(1600, np.float32)
    p = tmp_path / "x.wav"
    p.write_bytes(to_wav_bytes(audio))
    assert len(load_audio(p)) == 1600


def test_load_audio_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_audio(tmp_path / "missing.wav")
    bad = tmp_path / "bad.wav"
    bad.write_text("not audio")
    with pytest.raises(ValueError):
        load_audio(bad)


def test_samples_are_marked_synthetic():
    for name in NAMES:
        assert truth(name)["synthetic"] is True
    assert (ROOT / "scripts" / "make_samples.py").exists()

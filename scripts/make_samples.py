"""Generate the synthetic sample meetings in ``samples/`` from the scripts in ``samples/scripts``.

    pip install -e ".[samples]"
    python scripts/make_samples.py              # all meetings
    python scripts/make_samples.py --only weekly-sync-en
    python scripts/make_samples.py --truth-only   # refresh the .truth.json files from the scripts, keep the audio

Speech is synthesised with Kokoro-82M v1.1-zh (Apache-2.0) through sherpa-onnx
(Apache-2.0). The model (about 150 MB) is downloaded on first run. Each script
has two speakers with distinct voices. The output per meeting is:

    samples/<name>.wav          16 kHz mono speech with a short pause between turns
    samples/<name>.truth.json   the script with exact turn times, action items and decisions

All text in the scripts is fictional.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "samples" / "scripts"
OUT = ROOT / "samples"
TTS_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-int8-multi-lang-v1_1.tar.bz2"
TTS_DIR = Path.home() / ".cache" / "meeting-notes" / "kokoro-int8-multi-lang-v1_1"
SR = 16000


def ensure_tts() -> Path:
    if (TTS_DIR / "model.int8.onnx").exists():
        return TTS_DIR
    TTS_DIR.parent.mkdir(parents=True, exist_ok=True)
    archive = TTS_DIR.parent / "kokoro.tar.bz2"
    print(f"downloading {TTS_URL}", file=sys.stderr)
    urllib.request.urlretrieve(TTS_URL, archive)
    with tarfile.open(archive, "r:bz2") as tar:
        tar.extractall(TTS_DIR.parent, filter="data")
    archive.unlink()
    return TTS_DIR


def load_tts(d: Path):
    import sherpa_onnx

    kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig(
        model=str(d / "model.int8.onnx"), voices=str(d / "voices.bin"), tokens=str(d / "tokens.txt"),
        data_dir=str(d / "espeak-ng-data"), dict_dir=str(d / "dict"),
        lexicon=f"{d / 'lexicon-us-en.txt'},{d / 'lexicon-zh.txt'}",
    )
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(kokoro=kokoro, num_threads=4),
        rule_fsts=f"{d / 'date-zh.fst'},{d / 'number-zh.fst'}",
    )
    return sherpa_onnx.OfflineTts(cfg)


def check_words(scripts: list[dict], d: Path) -> None:
    """Fail early if an English word is missing from the TTS lexicon (it would be skipped silently)."""
    lexicon = {line.split(" ", 1)[0].lower() for line in (d / "lexicon-us-en.txt").read_text().splitlines() if line}
    missing = set()
    for s in scripts:
        for _, text in s["turns"]:
            missing |= {w for w in re.findall(r"[A-Za-z']+", text) if w.lower() not in lexicon}
    if missing:
        raise SystemExit(f"words not in the TTS lexicon (rephrase them): {sorted(missing)}")


def resample(samples: np.ndarray, rate: int) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "f32le", "-ar", str(rate), "-ac", "1", "-i", "-",
         "-f", "f32le", "-ar", str(SR), "-"],
        input=np.asarray(samples, dtype=np.float32).tobytes(), capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def build(name: str, script: dict, tts) -> None:
    rng = np.random.default_rng(sum(map(ord, name)))  # deterministic pauses and room tone
    pieces = [np.zeros(int(0.5 * SR), np.float32)]
    t = 0.5
    turns = []
    for i, (speaker, text) in enumerate(script["turns"]):
        audio = tts.generate(text, sid=int(script["voices"][speaker]), speed=1.0)
        speech = resample(np.asarray(audio.samples), audio.sample_rate)
        speech = 0.6 * speech / max(1e-6, float(np.abs(speech).max()))
        # trim the leading/trailing near-silence the TTS adds, so turn times are tight
        voiced = np.flatnonzero(np.abs(speech) > 0.01)
        speech = speech[max(0, voiced[0] - 800): voiced[-1] + 800]
        dur = len(speech) / SR
        turns.append({"speaker": speaker, "start": round(t, 2), "end": round(t + dur, 2), "text": text})
        gap = float(rng.uniform(0.6, 1.0))
        pieces += [speech, np.zeros(int(gap * SR), np.float32)]
        t += dur + gap
    audio = np.concatenate(pieces)
    audio = audio + rng.normal(0, 0.002, len(audio)).astype(np.float32)  # faint room tone
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    wav = OUT / f"{name}.wav"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "-",
         "-fflags", "+bitexact", "-flags:a", "+bitexact", "-map_metadata", "-1", str(wav)],
        input=pcm.tobytes(), check=True,
    )
    truth = {
        "file": wav.name,
        "title": script["title"],
        "language": script["language"],
        "duration": round(len(audio) / SR, 2),
        "speakers": sorted(script["voices"]),
        "synthetic": True,
        "tts": "Kokoro-82M v1.1-zh via sherpa-onnx, voices " + ", ".join(f"{k}=sid {v}" for k, v in script["voices"].items()),
        "turns": turns,
        "action_items": script["action_items"],
        "decisions": script["decisions"],
    }
    (OUT / f"{name}.truth.json").write_text(json.dumps(truth, indent=2, ensure_ascii=False) + "\n")
    print(f"{wav.name}: {truth['duration']:.1f}s, {len(turns)} turns", file=sys.stderr)


def refresh_truth(name: str, script: dict) -> None:
    """Rewrite the parts of ``<name>.truth.json`` that come from the script; keep the measured turn times."""
    path = OUT / f"{name}.truth.json"
    truth = json.loads(path.read_text())
    if [t["text"] for t in truth["turns"]] != [text for _, text in script["turns"]]:
        raise SystemExit(f"{name}: the turns changed, so the audio must be rebuilt (run without --truth-only)")
    truth.update(title=script["title"], language=script["language"], action_items=script["action_items"],
                 decisions=script["decisions"])
    path.write_text(json.dumps(truth, indent=2, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="build one meeting by name")
    ap.add_argument("--truth-only", action="store_true", help="refresh truth files only (no TTS, no audio)")
    args = ap.parse_args()
    files = sorted(SCRIPTS.glob("*.yaml"))
    if args.only:
        files = [f for f in files if f.stem == args.only]
    scripts = {f.stem: yaml.safe_load(f.read_text()) for f in files}
    if args.truth_only:
        for name, script in scripts.items():
            refresh_truth(name, script)
        return 0
    d = ensure_tts()
    check_words(list(scripts.values()), d)
    tts = load_tts(d)
    for name, script in scripts.items():
        build(name, script, tts)
    return 0


if __name__ == "__main__":
    sys.exit(main())

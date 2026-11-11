"""Speaker labels ("who spoke when") and merging them with the transcript.

``sherpa`` (default, no account or token needed): sherpa-onnx offline diarization with
  the pyannote segmentation-3.0 model (MIT, ONNX copy published by the sherpa-onnx
  project) and the 3D-Speaker CAM++ zh/en speaker-embedding model (Apache-2.0),
  followed by agglomerative clustering. Models download from GitHub on first use (35 MB).
``pyannote`` (optional): pyannote/speaker-diarization-3.1. The model is gated on
  Hugging Face: accept its terms, set HF_TOKEN and ``pip install -e '.[pyannote]'``.
``none``: every segment gets the same speaker.
"""

from __future__ import annotations

import sys
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE
from .config import Settings
from .models import Segment, Word

RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION_URL = f"{RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
EMBEDDING_URL = f"{RELEASES}/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"


@dataclass
class Turn:
    start: float
    end: float
    speaker: int


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}", file=sys.stderr)
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def sherpa_models(model_dir: str | Path) -> tuple[Path, Path]:
    root = Path(model_dir)
    seg = root / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
    emb = root / Path(EMBEDDING_URL).name
    if not seg.exists():
        archive = root / "segmentation.tar.bz2"
        _download(SEGMENTATION_URL, archive)
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(root, filter="data")
        archive.unlink()
    if not emb.exists():
        _download(EMBEDDING_URL, emb)
    return seg, emb


class SherpaDiarizer:
    name = "sherpa-onnx (pyannote-segmentation-3.0 + CAM++)"

    def __init__(self, settings: Settings):
        try:
            import sherpa_onnx
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("speaker labels need sherpa-onnx: pip install -e .") from exc
        self._sherpa = sherpa_onnx
        self.settings = settings
        self.seg, self.emb = sherpa_models(settings.model_dir)

    def __call__(self, audio: np.ndarray, num_speakers: int = 0) -> list[Turn]:
        s = self._sherpa
        threads = self.settings.threads
        config = s.OfflineSpeakerDiarizationConfig(
            segmentation=s.OfflineSpeakerSegmentationModelConfig(
                pyannote=s.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(self.seg)), num_threads=threads),
            embedding=s.SpeakerEmbeddingExtractorConfig(model=str(self.emb), num_threads=threads),
            clustering=s.FastClusteringConfig(num_clusters=num_speakers if num_speakers > 0 else -1,
                                              threshold=self.settings.cluster_threshold),
            min_duration_on=0.3,
            min_duration_off=0.5,
        )
        if not config.validate():  # pragma: no cover
            raise RuntimeError("sherpa-onnx diarization config is invalid (model files missing?)")
        sd = s.OfflineSpeakerDiarization(config)
        result = sd.process(np.ascontiguousarray(audio, dtype=np.float32)).sort_by_start_time()
        return [Turn(r.start, r.end, r.speaker) for r in result]


class PyannoteDiarizer:
    name = "pyannote/speaker-diarization-3.1"

    def __init__(self, settings: Settings):
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise RuntimeError("MS_DIARIZER=pyannote needs: pip install -e '.[pyannote]'") from exc
        if not settings.hf_token:
            raise RuntimeError("MS_DIARIZER=pyannote needs HF_TOKEN (the model is gated; accept its terms first)")
        self._torch = torch
        try:
            self.pipeline = Pipeline.from_pretrained(self.name, token=settings.hf_token)
        except TypeError:  # older pyannote.audio
            self.pipeline = Pipeline.from_pretrained(self.name, use_auth_token=settings.hf_token)

    def __call__(self, audio: np.ndarray, num_speakers: int = 0) -> list[Turn]:
        kwargs = {"num_speakers": num_speakers} if num_speakers > 0 else {}
        waveform = self._torch.from_numpy(audio).unsqueeze(0)
        out = self.pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE}, **kwargs)
        annotation = getattr(out, "speaker_diarization", out)
        ids: dict[str, int] = {}
        turns = []
        for seg, _, label in annotation.itertracks(yield_label=True):
            turns.append(Turn(seg.start, seg.end, ids.setdefault(label, len(ids))))
        return turns


def get_diarizer(settings: Settings):
    """Return a diarizer callable, or ``None`` when speaker labels are switched off."""
    choice = settings.diarizer
    if choice == "none":
        return None
    if choice == "pyannote" or (choice == "auto" and settings.hf_token and _has_pyannote()):
        return PyannoteDiarizer(settings)
    return SherpaDiarizer(settings)


def _has_pyannote() -> bool:
    try:
        import pyannote.audio  # noqa: F401
    except ImportError:
        return False
    return True


def _speaker_at(start: float, end: float, turns: list[Turn]) -> int | None:
    """The speaker with most overlap with [start, end]; the nearest turn if none overlaps."""
    best, best_overlap = None, 0.0
    for t in turns:
        overlap = min(end, t.end) - max(start, t.start)
        if overlap > best_overlap:
            best, best_overlap = t.speaker, overlap
    if best is None and turns:
        mid = (start + end) / 2
        best = min(turns, key=lambda t: min(abs(mid - t.start), abs(mid - t.end))).speaker
    return best


def _join(words: list[Word]) -> str:
    return "".join(w.text for w in words).strip()


def assign_speakers(segments: list[Segment], turns: list[Turn]) -> list[Segment]:
    """Label each segment with a speaker; split a segment where its words change speaker.

    Labels are ``S1``, ``S2``, ... in order of first appearance.
    """
    if not turns:
        for s in segments:
            s.speaker = "S1"
        return segments
    pieces: list[tuple[int | None, Segment]] = []
    for seg in segments:
        if not seg.words:
            pieces.append((_speaker_at(seg.start, seg.end, turns), seg))
            continue
        run: list[Word] = []
        run_spk = None
        for w in seg.words:
            spk = _speaker_at(w.start, w.end, turns)
            if run and spk != run_spk:
                pieces.append((run_spk, Segment(start=run[0].start, end=run[-1].end, text=_join(run), words=run)))
                run = []
            run_spk = spk
            run.append(w)
        if run:
            if len(run) == len(seg.words):
                pieces.append((run_spk, seg))
            else:
                pieces.append((run_spk, Segment(start=run[0].start, end=run[-1].end, text=_join(run), words=run)))
    # absorb one- or two-word fragments into their neighbour: they are usually boundary noise
    cleaned: list[tuple[int | None, Segment]] = []
    for spk, seg in pieces:
        if cleaned and spk != cleaned[-1][0] and 0 < len(seg.words) <= 2 and seg.start - cleaned[-1][1].end < 0.3:
            prev_spk, prev = cleaned[-1]
            merged = prev.words + seg.words
            cleaned[-1] = (prev_spk, Segment(start=prev.start, end=seg.end, text=_join(merged), words=merged))
            continue
        cleaned.append((spk, seg))
    labels: dict[int | None, str] = {}
    out = []
    for spk, seg in cleaned:
        seg.speaker = labels.setdefault(spk, f"S{len(labels) + 1}")
        out.append(seg)
    return out

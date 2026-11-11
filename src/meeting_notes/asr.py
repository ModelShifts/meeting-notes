"""Speech recognition.

``LocalWhisper``: faster-whisper (CTranslate2 build of Whisper), runs on CPU or GPU.
``APITranscriber``: an OpenAI-compatible ``/v1/audio/transcriptions`` endpoint.

Both return ``(segments, language)`` where each segment has start/end times, text
and, when available, word times (used to split a segment where the speaker changes).
"""

from __future__ import annotations

import re
from typing import Protocol

import numpy as np

from .audio import SAMPLE_RATE, to_wav_bytes
from .config import Settings
from .models import Segment, Word

# A short simplified-Chinese prompt keeps Whisper from switching to traditional characters mid-meeting.
ZH_PROMPT = "以下是会议记录，使用简体中文。"
CJK = re.compile(r"[\u3400-\u9fff]")
LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
# Lines Whisper is known to invent in silence (it learnt them from subtitle credits in its training data).
HALLUCINATION = re.compile(
    r"(字幕\s*(by|由|提供|制作)|subtitles?\s*by|thanks? (you )?for watching|请不吝点赞|订阅.*频道|谢谢观看|amara\.org)", re.I
)


def is_hallucination(text: str, no_speech_prob: float = 0.0, avg_logprob: float = 0.0) -> bool:
    if HALLUCINATION.search(text):
        return True
    return no_speech_prob > 0.6 and avg_logprob < -1.0


def build_prompt(language: str | None, extra: str = "") -> str | None:
    """Whisper's initial prompt: the simplified-Chinese hint for Chinese audio plus the user's own context."""
    parts = [ZH_PROMPT] if language == "zh" else []
    if extra.strip():
        parts.append(extra.strip())
    return " ".join(parts) or None


class ASR(Protocol):
    name: str

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> tuple[list[Segment], str]: ...


class LocalWhisper:
    def __init__(self, settings: Settings):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("local transcription needs faster-whisper: pip install -e .") from exc
        try:
            import ctranslate2

            cuda = ctranslate2.get_cuda_device_count() > 0
        except Exception:  # pragma: no cover
            cuda = False
        device = "cuda" if cuda else "cpu"
        compute = settings.whisper_compute
        if compute == "auto":
            # int8 is smaller and faster, but in our tests it made large-v3 garble mixed Chinese and English
            compute = "float16" if cuda else "float32"
        self.name = f"faster-whisper {settings.whisper_model}"
        self.prompt = settings.stt_prompt
        self.hotwords = settings.stt_hotwords
        self.model = WhisperModel(settings.whisper_model, device=device, compute_type=compute,
                                  cpu_threads=settings.threads)

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> tuple[list[Segment], str]:
        if not language:
            language, _, _ = self.model.detect_language(audio, vad_filter=True)
        segments, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=5,
            vad_filter=True,  # Silero VAD: skips silence, which avoids repeated or dropped lines
            condition_on_previous_text=False,
            word_timestamps=True,
            initial_prompt=build_prompt(language, self.prompt),
            hotwords=self.hotwords or None,
        )
        out = []
        for s in segments:
            if is_hallucination(s.text, s.no_speech_prob, s.avg_logprob):
                continue
            words = [Word(start=w.start, end=w.end, text=w.word) for w in (s.words or [])]
            out.append(Segment(start=s.start, end=s.end, text=s.text.strip(), words=words))
        return out, info.language


class APITranscriber:
    def __init__(self, settings: Settings):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("MS_BACKEND=api needs the 'api' extra: pip install -e '.[api]'") from exc
        self.client = OpenAI(base_url=settings.stt_api_base or settings.api_base,
                             api_key=settings.stt_api_key or settings.api_key or "not-needed")
        self.name = settings.stt_model
        # speech APIs take one prompt, so the terms list is appended to it
        terms = settings.stt_hotwords.strip()
        self.prompt = " ".join(x for x in (settings.stt_prompt.strip(), f"Terms: {terms}." if terms else "") if x)

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> tuple[list[Segment], str]:
        kwargs = {"language": language} if language else {}
        prompt = build_prompt(language, self.prompt)
        if prompt:
            kwargs["prompt"] = prompt
        resp = self.client.audio.transcriptions.create(
            model=self.name,
            file=("meeting.wav", to_wav_bytes(audio), "audio/wav"),
            response_format="verbose_json",
            timestamp_granularities=["segment", "word"],
            **kwargs,
        )
        data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
        return parse_verbose_json(data, len(audio) / SAMPLE_RATE)


def parse_verbose_json(data: dict, duration: float) -> tuple[list[Segment], str]:
    """Turn an OpenAI ``verbose_json`` transcription into segments (words attached when present)."""
    words = [Word(start=float(w["start"]), end=float(w["end"]), text=str(w.get("word", "")))
             for w in (data.get("words") or [])]
    segments = []
    for s in data.get("segments") or []:
        text = str(s.get("text", "")).strip()
        if not text or is_hallucination(text, float(s.get("no_speech_prob") or 0), float(s.get("avg_logprob") or 0)):
            continue
        start, end = float(s["start"]), float(s["end"])
        mine = [w for w in words if w.start >= start - 0.01 and w.end <= end + 0.01]
        segments.append(Segment(start=start, end=end, text=text, words=mine))
    if not segments and str(data.get("text", "")).strip():
        segments = [Segment(start=0.0, end=duration, text=str(data["text"]).strip(), words=words)]
    language = normalize_language(str(data.get("language") or ""))
    return segments, language


_LANG_NAMES = {"english": "en", "chinese": "zh", "mandarin": "zh"}


def normalize_language(name: str) -> str:
    name = name.strip().lower()
    return _LANG_NAMES.get(name, name[:2] if name else "")


def describe_language(language: str, text: str) -> str:
    """``en``, ``zh``, or ``zh+en`` when a Chinese meeting also has a fair share of English words."""
    cjk = len(CJK.findall(text))
    latin = len(LATIN_WORD.findall(text))
    if cjk and latin and min(cjk, latin * 2) / max(cjk, latin * 2) > 0.08:
        return "zh+en"
    return language or ("zh" if cjk > latin else "en")


def fix_chinese_script(segments: list[Segment], mode: str) -> list[Segment]:
    """Whisper sometimes mixes traditional and simplified characters; convert them to one script."""
    if mode == "keep" or not any(CJK.search(s.text) for s in segments):
        return segments
    from opencc import OpenCC

    cc = OpenCC("t2s" if mode == "simplified" else "s2t")
    for s in segments:
        s.text = cc.convert(s.text)
        for w in s.words:
            w.text = cc.convert(w.text)
    return segments


_ASR: dict[tuple, ASR] = {}


def get_asr(settings: Settings) -> ASR:
    key = (settings.backend, settings.stt_model, settings.whisper_model, settings.api_base, settings.stt_api_base,
           settings.stt_prompt, settings.stt_hotwords)
    if key not in _ASR:
        if settings.backend == "api" and settings.stt_model:
            _ASR[key] = APITranscriber(settings)
        else:
            _ASR[key] = LocalWhisper(settings)
    return _ASR[key]

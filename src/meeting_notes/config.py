"""Settings read from environment variables (and a local ``.env`` file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_WHISPER_MODEL = "large-v3"


def load_dotenv_if_present(path: str | os.PathLike | None = None) -> None:
    """Load ``.env`` from the working directory without overriding real env vars."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is a core dependency
        return
    load_dotenv(dotenv_path=path or Path.cwd() / ".env", override=False)


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name, default)
    return value.strip() if isinstance(value, str) else default


@dataclass
class Settings:
    backend: str = "hf"
    # api backend
    api_base: str = ""
    api_key: str = ""
    model: str = ""
    stt_model: str = ""  # speech-to-text model; empty = local faster-whisper
    stt_api_base: str = ""  # defaults to api_base
    stt_api_key: str = ""  # defaults to api_key
    # hf backend
    hf_model: str = DEFAULT_HF_MODEL
    hf_dtype: str = "auto"  # auto (bfloat16 on GPU, float32 on CPU) | float32 | bfloat16
    # local speech recognition (used by hf, and by api when MS_STT_MODEL is empty)
    whisper_model: str = DEFAULT_WHISPER_MODEL
    whisper_compute: str = "auto"  # auto | int8 | float16 | float32
    language: str = ""  # empty = detect; "en", "zh", ...
    stt_prompt: str = ""  # optional context for speech recognition, written like the speech (Whisper initial prompt)
    stt_hotwords: str = ""  # optional comma-separated terms that come up in your meetings (product names, jargon)
    zh_script: str = "simplified"  # simplified | traditional | keep
    # speaker labels
    diarizer: str = "auto"  # auto | sherpa | pyannote | none
    num_speakers: int = 0  # 0 = estimate
    cluster_threshold: float = 0.5
    hf_token: str = ""
    # shared
    max_new_tokens: int = 1024
    threads: int = 4
    cache_dir: str = ".cache/results"
    model_dir: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv_if_present()
        return cls(
            backend=_env("MS_BACKEND", "hf").lower(),
            api_base=_env("MS_API_BASE"),
            api_key=_env("MS_API_KEY"),
            model=_env("MS_MODEL"),
            stt_model=_env("MS_STT_MODEL"),
            stt_api_base=_env("MS_STT_API_BASE"),
            stt_api_key=_env("MS_STT_API_KEY"),
            hf_model=_env("MS_HF_MODEL", DEFAULT_HF_MODEL),
            hf_dtype=_env("MS_HF_DTYPE", "auto").lower(),
            whisper_model=_env("MS_WHISPER_MODEL", DEFAULT_WHISPER_MODEL),
            whisper_compute=_env("MS_WHISPER_COMPUTE", "auto").lower(),
            language=_env("MS_LANGUAGE").lower(),
            stt_prompt=_env("MS_STT_PROMPT"),
            stt_hotwords=_env("MS_STT_HOTWORDS"),
            zh_script=_env("MS_ZH_SCRIPT", "simplified").lower(),
            diarizer=_env("MS_DIARIZER", "auto").lower(),
            num_speakers=int(_env("MS_NUM_SPEAKERS", "0") or 0),
            cluster_threshold=float(_env("MS_CLUSTER_THRESHOLD", "0.5") or 0.5),
            hf_token=_env("HF_TOKEN"),
            max_new_tokens=int(_env("MS_MAX_NEW_TOKENS", "1024") or 1024),
            threads=int(_env("MS_THREADS", "4") or 4),
            cache_dir=_env("MS_CACHE_DIR", ".cache/results"),
            model_dir=str(Path(_env("MS_MODEL_DIR") or Path.home() / ".cache" / "meeting-notes").expanduser()),
        )

    @property
    def model_name(self) -> str:
        return self.model if self.backend == "api" else self.hf_model

    @property
    def asr_name(self) -> str:
        if self.backend == "api" and self.stt_model:
            return self.stt_model
        return f"faster-whisper {self.whisper_model}"

    def validate(self) -> None:
        if self.backend not in {"api", "hf"}:
            raise ValueError(f"MS_BACKEND must be 'api' or 'hf', got {self.backend!r}")
        if self.backend == "api":
            missing = [n for n, v in (("MS_API_BASE", self.api_base), ("MS_MODEL", self.model)) if not v]
            if missing:
                raise ValueError(f"MS_BACKEND=api needs {', '.join(missing)} (see .env.example)")
        if self.diarizer not in {"auto", "sherpa", "pyannote", "none"}:
            raise ValueError(f"MS_DIARIZER must be auto, sherpa, pyannote or none, got {self.diarizer!r}")
        if self.zh_script not in {"simplified", "traditional", "keep"}:
            raise ValueError(f"MS_ZH_SCRIPT must be simplified, traditional or keep, got {self.zh_script!r}")

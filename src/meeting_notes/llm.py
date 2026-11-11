"""Chat model backends.

``api``: any OpenAI-compatible ``/v1/chat/completions`` endpoint.
``hf``:  a local ``transformers`` causal LM (default Qwen/Qwen2.5-7B-Instruct;
       set MS_HF_MODEL=Qwen/Qwen2.5-1.5B-Instruct on a CPU).

Both expose ``complete(messages) -> str``. Decoding is greedy (temperature 0).
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from .config import Settings

Message = dict[str, str]


class LLM(Protocol):
    name: str

    def complete(self, messages: list[Message], max_new_tokens: int | None = None) -> str: ...


class APIChat:
    def __init__(self, settings: Settings):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("MS_BACKEND=api needs the 'api' extra: pip install -e '.[api]'") from exc
        self.client = OpenAI(base_url=settings.api_base, api_key=settings.api_key or "not-needed")
        self.name = settings.model
        self.max_new_tokens = settings.max_new_tokens

    def complete(self, messages: list[Message], max_new_tokens: int | None = None) -> str:
        response = self.client.chat.completions.create(
            model=self.name,
            messages=messages,
            temperature=0,
            max_tokens=max_new_tokens or self.max_new_tokens,
        )
        return response.choices[0].message.content or ""


class HFChat:
    def __init__(self, settings: Settings):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("MS_BACKEND=hf needs the 'hf' extra: pip install -e '.[hf]'") from exc
        self._torch = torch
        if settings.threads > 0:
            torch.set_num_threads(settings.threads)
        self.name = settings.hf_model
        self.max_new_tokens = settings.max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if settings.hf_dtype in {"bfloat16", "float32", "float16"}:
            dtype = getattr(torch, settings.hf_dtype)
        else:
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForCausalLM.from_pretrained(self.name, dtype=dtype).to(self.device).eval()

    def complete(self, messages: list[Message], max_new_tokens: int | None = None) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


_CACHE: dict[tuple, Any] = {}


def get_llm(settings: Settings) -> LLM:
    """Build (and memoise) the chat model for the configured backend."""
    settings.validate()
    key = ("llm", settings.backend, settings.model_name, settings.api_base)
    if key not in _CACHE:
        _CACHE[key] = APIChat(settings) if settings.backend == "api" else HFChat(settings)
    return _CACHE[key]



def parse_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply (tolerates code fences, chatter and small syntax slips)."""
    try:
        return _strict_json_object(text)
    except (ValueError, json.JSONDecodeError):
        pass
    from json_repair import repair_json  # small models sometimes forget a bracket or a comma

    body = re.sub(r"```(?:json)?", "", text)
    repaired = repair_json(body[max(body.find("{"), 0):], return_objects=True)
    if isinstance(repaired, list):
        repaired = next((x for x in repaired if isinstance(x, dict)), None)
    if not isinstance(repaired, dict) or not repaired:
        raise ValueError(f"model reply has no JSON object: {text[:200]!r}")
    return repaired


def _strict_json_object(text: str) -> dict[str, Any]:
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        raise ValueError(f"model reply has no JSON object: {text[:200]!r}")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError(f"model reply has an unterminated JSON object: {text[:200]!r}")

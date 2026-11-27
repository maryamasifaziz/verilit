"""The LLM used by the Planner, Claim Generator and Synthesizer.

Works with any provider that offers an OpenAI-compatible API. Two free-tier
presets are built in; choose one with the VERILIT_LLM environment variable:

    gemini (default)  key in GEMINI_API_KEY   model gemini-3.5-flash-lite
    groq              key in GROQ_API_KEY     model llama-3.1-8b-instant

VERILIT_LLM_MODEL overrides the model name. Requests are spaced out to stay
under the free tier's per-minute limit, retried with backoff when the API says
"too many requests", and cached on disk (see verilit/cache.py).
"""

from __future__ import annotations

import json
import os
import re
import time

import openai
from openai import OpenAI

from verilit.cache import MISSING, DiskCache

PRESETS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-3.5-flash-lite",
        "key_env": "GEMINI_API_KEY",
        "min_interval": 4.5,  # seconds between requests: about 13 per minute (free tier allows ~15)
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.1-8b-instant",
        "key_env": "GROQ_API_KEY",
        "min_interval": 2.5,
    },
}


class LLMError(RuntimeError):
    pass


def parse_json(text: str) -> dict:
    """Parse a JSON object from a reply, tolerating ```json fences or text around it."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


class LLM:
    def __init__(self, provider: str | None = None, model: str | None = None, use_cache: bool = True):
        self.provider = provider or os.environ.get("VERILIT_LLM", "gemini")
        if self.provider not in PRESETS:
            raise LLMError(f"Unknown VERILIT_LLM={self.provider!r}; use one of {list(PRESETS)}")
        preset = PRESETS[self.provider]
        self.model = model or os.environ.get("VERILIT_LLM_MODEL") or preset["model"]
        api_key = os.environ.get(preset["key_env"])
        if not api_key:
            raise LLMError(f"Set the {preset['key_env']} environment variable to your {self.provider} API key.")
        self.client = OpenAI(base_url=preset["base_url"], api_key=api_key, max_retries=0, timeout=120)
        self.min_interval = float(os.environ.get("VERILIT_LLM_MIN_INTERVAL", preset["min_interval"]))
        self.cache = DiskCache("llm") if use_cache else None
        self.requests = 0  # API requests actually sent (what counts against your quota)
        self.cache_hits = 0
        self._last_request = 0.0

    def _wait_turn(self) -> None:
        wait = self._last_request + self.min_interval - time.time()
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.time()

    def json(self, system: str, user: str, max_attempts: int = 6) -> dict:
        """Send one request and return the reply parsed as a JSON object."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        key = {"provider": self.provider, "model": self.model, "messages": messages}
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not MISSING:
                self.cache_hits += 1
                return cached

        last_error: Exception | None = None
        for attempt in range(max_attempts):
            self._wait_turn()
            self.requests += 1
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0,
                )
                value = parse_json(response.choices[0].message.content or "")
                if not isinstance(value, dict):
                    raise ValueError("reply is not a JSON object")
            except openai.RateLimitError as e:
                last_error = e
                delay = min(60, 5 * 2**attempt)
                print(f"  [llm] rate limited by {self.provider}; waiting {delay}s (attempt {attempt + 1}/{max_attempts})")
                time.sleep(delay)
                continue
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as e:
                last_error = e
                time.sleep(min(30, 3 * 2**attempt))
                continue
            except (ValueError, json.JSONDecodeError) as e:  # malformed JSON: ask again
                last_error = e
                continue
            if self.cache is not None:
                self.cache.set(key, value)
            return value

        hint = ""
        if isinstance(last_error, openai.RateLimitError):
            other = "groq" if self.provider == "gemini" else "gemini"
            hint = f" The daily quota may be used up; try again later or set VERILIT_LLM={other}."
        raise LLMError(f"{self.provider} request failed after {max_attempts} attempts: {last_error}.{hint}")

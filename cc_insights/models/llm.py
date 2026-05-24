"""Minimal OpenAI-compatible chat client with on-disk LRU cache and retries.

Works with any provider that implements OpenAI's /chat/completions schema
(OpenAI itself, OpenRouter, vLLM, Ollama, etc.). Cache key = (model, prompt_hash).
Use force=True to bypass.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


class LLMError(RuntimeError):
    pass


class _Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT NOT NULL)"
        )
        self._conn.commit()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT v FROM cache WHERE k=?", (key,)).fetchone()
        return row[0] if row else None

    def set(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (k, v) VALUES (?, ?)", (key, value)
            )
            self._conn.commit()


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        cache_path: Path,
        timeout_s: float = 60.0,
    ):
        if not api_key:
            raise LLMError("LLM_API_KEY is empty")
        self._api_key = api_key
        self._base = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout_s)
        self._cache = _Cache(cache_path)

    def _cache_key(self, model: str, payload: dict) -> str:
        h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return f"{model}::{h}"

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20))
    def _post(self, payload: dict) -> dict:
        r = self._client.post(
            f"{self._base}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if r.status_code >= 400:
            raise LLMError(f"LLM provider {r.status_code}: {r.text[:500]}")
        return r.json()

    def chat(
        self,
        model: str,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 800,
        force: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        key = self._cache_key(model, payload)
        if not force:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
        data = self._post(payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"Malformed LLM response: {data}") from e
        self._cache.set(key, content)
        return content

    def chat_json(self, model: str, system: str, user: str, **kw) -> dict:
        raw = self.chat(model, system, user, json_mode=True, **kw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise LLMError(f"LLM returned non-JSON: {raw[:300]}") from e

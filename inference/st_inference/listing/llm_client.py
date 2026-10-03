"""Client for a local OpenAI-compatible server (llama.cpp ``llama-server``, or vLLM).

Deliberately protocol-only: no vendor SDK, nothing hosted. Swapping the model means pointing
ST_LLM_BASE_URL somewhere else.
"""

from __future__ import annotations

import asyncio
import json

import httpx


class LLMUnavailable(Exception):
    pass


class LocalLLM:
    def __init__(self, base_url: str, model: str = "local", *, timeout_s: float = 180.0, retries: int = 2,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.model = model
        self.retries = retries
        self.http = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout_s, transport=transport)

    async def aclose(self) -> None:
        await self.http.aclose()

    async def healthy(self) -> bool:
        try:
            r = await self.http.get("/models", timeout=5)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def chat(self, messages: list[dict], *, schema: dict | None = None, temperature: float = 0.3,
                   max_tokens: int = 1500) -> str:
        # Persian costs ~2 tokens per character in this tokenizer: a 400-character listing is
        # ~900 completion tokens. A 700-token cap truncates mid-JSON (measured, not guessed).
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Small models repeat whole sentences in Persian descriptions without this.
            "frequency_penalty": 0.3,
            "stream": False,
        }
        if schema is not None:
            # llama-server understands both; json_schema constrains sampling with a grammar.
            payload["response_format"] = {"type": "json_object"}
            payload["json_schema"] = schema
        delay = 1.0
        last: Exception | None = None
        for _ in range(self.retries + 1):
            try:
                r = await self.http.post("/chat/completions", json=payload)
                if r.status_code >= 500:
                    last = LLMUnavailable(f"{r.status_code} from LLM server")
                else:
                    r.raise_for_status()
                    data = r.json()
                    return data["choices"][0]["message"]["content"] or ""
            except (httpx.HTTPError, KeyError, json.JSONDecodeError) as e:
                last = e
            await asyncio.sleep(delay)
            delay *= 2
        raise LLMUnavailable(str(last))

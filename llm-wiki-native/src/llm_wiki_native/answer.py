"""Answer generation protocol for native query responses."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Any, Mapping, Protocol
import urllib.request


class AnswerGenerator(Protocol):
    def generate(self, query: str, context: dict[str, Any], *, mode: str) -> str | dict[str, Any]:
        """Generate an answer from assembled native context."""


@dataclass(frozen=True)
class NativeAnswerConfig:
    base_url: str
    model: str
    api_key: str = field(repr=False)
    timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "NativeAnswerConfig":
        values = os.environ if env is None else env
        base_url = values.get("LLM_WIKI_NATIVE_ANSWER_BASE_URL", "").strip()
        model = values.get("LLM_WIKI_NATIVE_ANSWER_MODEL", "").strip()
        api_key = values.get("LLM_WIKI_NATIVE_ANSWER_API_KEY", "").strip()
        if not base_url:
            raise ValueError("LLM_WIKI_NATIVE_ANSWER_BASE_URL is required")
        if not model:
            raise ValueError("LLM_WIKI_NATIVE_ANSWER_MODEL is required")
        if not api_key:
            raise ValueError("LLM_WIKI_NATIVE_ANSWER_API_KEY is required")
        timeout = float(values.get("LLM_WIKI_NATIVE_ANSWER_TIMEOUT_SECONDS", "120"))
        return cls(base_url=base_url, model=model, api_key=api_key, timeout_seconds=timeout)


class NativeAnswerGenerator:
    def __init__(self, config: NativeAnswerConfig) -> None:
        self.config = config

    def generate(self, query: str, context: dict[str, Any], *, mode: str) -> dict[str, Any]:
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Answer using only the provided native retrieval context. If the context is insufficient, say so.",
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "query": query,
                            "mode": mode,
                            "context_blocks": context.get("context_blocks", []),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        request = urllib.request.Request(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        return {
            "response": _message_content(response_payload),
            "references": list(context.get("source_paths", [])),
        }


def _message_content(payload: object) -> str:
    if not isinstance(payload, dict):
        raise ValueError("chat response must be a JSON object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("chat response must include choices[0].message.content")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("chat response must include choices[0].message.content")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("chat response must include choices[0].message.content")
    content = message.get("content")
    if not isinstance(content, str) or not content:
        raise ValueError("chat response must include choices[0].message.content")
    return content

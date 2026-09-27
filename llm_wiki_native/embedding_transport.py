"""Shared OpenAI-compatible embedding request and response handling."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import math
import struct
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class EmbeddingTransportOptions:
    send_dimensions: bool = False
    use_base64: bool = False
    token_limit: int | None = None
    truncation_side: str = "right"
    asymmetric: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "EmbeddingTransportOptions":
        token_limit = _optional_positive_int(env.get("EMBEDDING_TOKEN_LIMIT"), "EMBEDDING_TOKEN_LIMIT")
        truncation_side = (env.get("EMBEDDING_TRUNCATION_SIDE") or "right").strip().lower()
        if truncation_side not in {"left", "right"}:
            raise ValueError("EMBEDDING_TRUNCATION_SIDE must be 'left' or 'right'")
        return cls(
            send_dimensions=_env_bool(env.get("EMBEDDING_SEND_DIM"), default=False),
            use_base64=_env_bool(env.get("EMBEDDING_USE_BASE64"), default=False),
            token_limit=token_limit,
            truncation_side=truncation_side,
            asymmetric=_env_bool(env.get("EMBEDDING_ASYMMETRIC"), default=False),
        )


def _env_bool(raw: str | None, *, default: bool) -> bool:
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected boolean value, got {raw!r}")


def _optional_positive_int(raw: str | None, name: str) -> int | None:
    if raw is None or not raw.strip():
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _prefix_text(text: str, *, input_type: str, asymmetric: bool) -> str:
    if not asymmetric:
        return text
    if input_type not in {"query", "passage"}:
        raise ValueError("input_type must be 'query' or 'passage'")
    prefix = f"{input_type}:"
    if text.lstrip().lower().startswith(prefix):
        return text
    return f"{prefix} {text}"


def build_embedding_payload(
    *,
    model: str,
    inputs: str | Sequence[str],
    expected_dim: int | None,
    options: EmbeddingTransportOptions,
    input_type: str,
) -> dict[str, Any]:
    if isinstance(inputs, str):
        prepared: str | list[str] = _prefix_text(inputs, input_type=input_type, asymmetric=options.asymmetric)
    else:
        prepared = [_prefix_text(str(text), input_type=input_type, asymmetric=options.asymmetric) for text in inputs]
    payload: dict[str, Any] = {"model": model, "input": prepared}
    if options.send_dimensions:
        if expected_dim is None or expected_dim <= 0:
            raise ValueError("a positive embedding dimension is required when EMBEDDING_SEND_DIM is enabled")
        payload["dimensions"] = expected_dim
    if options.use_base64:
        payload["encoding_format"] = "base64"
    if options.token_limit is not None:
        payload["truncate_prompt_tokens"] = options.token_limit
        payload["truncation_side"] = options.truncation_side
    return payload


def _decode_embedding(value: object, *, expected_dim: int | None) -> list[float]:
    if isinstance(value, str):
        try:
            raw = base64.b64decode(value, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("embedding response contains invalid Base64") from exc
        if not raw or len(raw) % 4:
            raise ValueError("Base64 embedding payload must contain float32 bytes")
        count = len(raw) // 4
        if expected_dim is not None and count != expected_dim:
            raise ValueError(f"embedding response dimension mismatch: expected {expected_dim}, got {count}")
        vector = list(struct.unpack(f"<{count}f", raw))
    elif isinstance(value, list) and value:
        vector = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError("embedding response must contain finite numbers")
            vector.append(float(item))
    else:
        raise ValueError("embedding response must include a non-empty embedding value")
    if expected_dim is not None and len(vector) != expected_dim:
        raise ValueError(f"embedding response dimension mismatch: expected {expected_dim}, got {len(vector)}")
    if any(not math.isfinite(number) for number in vector):
        raise ValueError("embedding response must contain finite numbers")
    return vector


def embedding_vectors_from_payload(
    payload: object,
    *,
    expected_count: int,
    expected_dim: int | None,
) -> list[list[float]]:
    if not isinstance(payload, dict):
        raise ValueError("embedding response must be a JSON object")
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("embedding response must include data")
    ordered = sorted(data, key=lambda row: row.get("index", 0) if isinstance(row, dict) else 0)
    if len(ordered) != expected_count:
        raise ValueError(f"embedding response count mismatch: expected {expected_count}, got {len(ordered)}")
    vectors: list[list[float]] = []
    for row in ordered:
        if not isinstance(row, dict):
            raise ValueError("embedding response data rows must be objects")
        vectors.append(_decode_embedding(row.get("embedding"), expected_dim=expected_dim))
    return vectors

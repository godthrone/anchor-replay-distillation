from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ChatAPIConfig:
    api_base: str
    model_name: str
    api_key: str

    @property
    def chat_completions_url(self) -> str:
        return self.api_base.rstrip("/") + "/chat/completions"


@dataclass(slots=True)
class EmbeddingAPIConfig:
    api_base: str
    model_name: str
    api_key: str

    @property
    def embeddings_url(self) -> str:
        return self.api_base.rstrip("/") + "/embeddings"


@dataclass(slots=True)
class ChatCompletionResult:
    text: str
    reasoning_status: str = "absent"


def load_api_env_file(path: str | Path | None) -> dict[str, str]:
    values: dict[str, str] = {}
    if path is None:
        return values
    env_path = Path(path)
    if not env_path.exists():
        raise FileNotFoundError(f"API env file does not exist: {env_path}")
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip().lstrip("\ufeff")
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().lstrip("\ufeff")] = value.strip().strip('"').strip("'")
    return values


def chat_api_config_from_env(
    env_file: str | Path | None = None,
    model_name: str | None = None,
    env_prefix: str = "ARD_TARGET",
) -> ChatAPIConfig:
    file_values = load_api_env_file(env_file)
    api_base_key = f"{env_prefix}_API_BASE"
    model_name_key = f"{env_prefix}_MODEL_NAME"
    api_key_key = f"{env_prefix}_API_KEY"
    api_base = file_values.get(api_base_key) or os.environ.get(api_base_key)
    env_model = file_values.get(model_name_key) or os.environ.get(model_name_key)
    api_key = file_values.get(api_key_key) or os.environ.get(api_key_key)
    resolved_model = model_name or env_model
    missing = [
        name
        for name, value in {
            api_base_key: api_base,
            model_name_key: resolved_model,
            api_key_key: api_key,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing API configuration: {', '.join(missing)}")
    return ChatAPIConfig(
        api_base=str(api_base),
        model_name=str(resolved_model),
        api_key=str(api_key),
    )


def embedding_api_config_from_env(
    env_file: str | Path | None = None,
    model_name: str | None = None,
    env_prefix: str = "ARD_EMBEDDING",
) -> EmbeddingAPIConfig:
    file_values = load_api_env_file(env_file)
    api_base_key = f"{env_prefix}_API_BASE"
    model_name_key = f"{env_prefix}_MODEL_NAME"
    api_key_key = f"{env_prefix}_API_KEY"
    api_base = file_values.get(api_base_key) or os.environ.get(api_base_key)
    env_model = file_values.get(model_name_key) or os.environ.get(model_name_key)
    api_key = file_values.get(api_key_key) or os.environ.get(api_key_key)
    resolved_model = model_name or env_model
    missing = [
        name
        for name, value in {
            api_base_key: api_base,
            model_name_key: resolved_model,
            api_key_key: api_key,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing embedding API configuration: {', '.join(missing)}")
    return EmbeddingAPIConfig(
        api_base=str(api_base),
        model_name=str(resolved_model),
        api_key=str(api_key),
    )


def _post_chat_completion(
    config: ChatAPIConfig,
    messages: list[dict[str, str]],
    max_tokens: int | None = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    timeout: float = 60,
) -> dict[str, Any]:
    payload = {
        "model": config.model_name,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    request = urllib.request.Request(
        config.chat_completions_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"chat completions request failed with HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"chat completions request failed: {exc.reason}") from exc
    return response_payload


def _first_chat_message(response_payload: dict[str, Any]) -> dict[str, Any]:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("chat completions response did not contain choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise RuntimeError("chat completions response did not contain message content")
    return message


def chat_completion(
    config: ChatAPIConfig,
    messages: list[dict[str, str]],
    max_tokens: int | None = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    timeout: float = 60,
) -> str:
    message = _first_chat_message(
        _post_chat_completion(
            config=config,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
        )
    )
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("chat completions response did not contain message content")
    return content.strip()


def _contains_think_tag(text: str) -> bool:
    return "<think" in text.lower()


def format_target_answer(
    content: str | None,
    reasoning_content: str | None,
) -> ChatCompletionResult:
    final = content.strip() if isinstance(content, str) else ""
    reasoning = reasoning_content.strip() if isinstance(reasoning_content, str) else ""
    if final and _contains_think_tag(final):
        return ChatCompletionResult(text=final, reasoning_status="inline")
    if reasoning and final:
        return ChatCompletionResult(
            text=f"<think>\n{reasoning}\n</think>\n\n{final}",
            reasoning_status="separate",
        )
    if reasoning:
        return ChatCompletionResult(
            text=f"<think>\n{reasoning}\n</think>",
            reasoning_status="only",
        )
    if final:
        return ChatCompletionResult(text=final, reasoning_status="absent")
    raise RuntimeError("chat completions response did not contain message content")


def target_chat_completion(
    config: ChatAPIConfig,
    messages: list[dict[str, str]],
    max_tokens: int | None = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    timeout: float = 60,
) -> ChatCompletionResult:
    message = _first_chat_message(
        _post_chat_completion(
            config=config,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
        )
    )
    content = message.get("content")
    reasoning_content = message.get("reasoning_content")
    return format_target_answer(
        content if isinstance(content, str) else None,
        reasoning_content if isinstance(reasoning_content, str) else None,
    )


def create_embeddings(
    config: EmbeddingAPIConfig,
    texts: list[str],
    timeout: float = 60,
) -> list[list[float]]:
    payload = {
        "model": config.model_name,
        "input": texts,
    }
    request = urllib.request.Request(
        config.embeddings_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"embeddings request failed with HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"embeddings request failed: {exc.reason}") from exc

    data = response_payload.get("data")
    if not isinstance(data, list) or len(data) != len(texts):
        raise RuntimeError("embeddings response did not contain one vector per input")
    vectors: list[list[float]] = []
    for item in sorted(data, key=lambda record: int(record.get("index", 0))):
        embedding = item.get("embedding") if isinstance(item, dict) else None
        if not isinstance(embedding, list):
            raise RuntimeError("embeddings response item did not contain an embedding list")
        vectors.append([float(value) for value in embedding])
    return vectors

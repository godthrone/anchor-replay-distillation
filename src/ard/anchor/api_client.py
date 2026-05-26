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


def chat_completion(
    config: ChatAPIConfig,
    messages: list[dict[str, str]],
    max_tokens: int | None = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    timeout: float = 60,
) -> str:
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

    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("chat completions response did not contain choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if (not isinstance(content, str) or not content.strip()) and isinstance(message, dict):
        content = message.get("reasoning_content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("chat completions response did not contain message content")
    return content.strip()

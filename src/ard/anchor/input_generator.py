from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

from ard.anchor.api_client import ChatAPIConfig, chat_completion
from ard.anchor.bank import AnchorPrompt, GeneratedInputAnchor, read_anchor_prompts, write_jsonl

ChatFn = Callable[[ChatAPIConfig, list[dict[str, str]], int | None, float, float, float], str]
LogFn = Callable[[str], None]


@dataclass(slots=True)
class InputGenerationStats:
    input_count: int = 0
    kept_count: int = 0
    failed_count: int = 0
    parse_failed_count: int = 0
    api_failed_count: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _is_multi_turn(conversation_type: str) -> bool:
    return conversation_type != "single_turn"


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _extract_json_payload(text: str) -> str:
    stripped = _strip_json_fence(text)
    if stripped.startswith("[") or stripped.startswith("{"):
        return stripped
    array_start = stripped.find("[")
    array_end = stripped.rfind("]")
    if array_start != -1 and array_end > array_start:
        return stripped[array_start : array_end + 1]
    object_start = stripped.find("{")
    object_end = stripped.rfind("}")
    if object_start != -1 and object_end > object_start:
        return stripped[object_start : object_end + 1]
    return stripped


def _normalize_message(record: object) -> dict[str, str]:
    if not isinstance(record, dict):
        raise ValueError("message must be an object")
    role = str(record.get("role", "")).strip()
    content = str(record.get("content", "")).strip()
    if role not in {"user", "assistant"}:
        raise ValueError(f"invalid message role: {role}")
    if not content:
        raise ValueError("message content must not be empty")
    return {"role": role, "content": content}


def parse_input_generator_messages(text: str, conversation_type: str) -> list[dict[str, str]]:
    content = text.strip()
    if not content:
        raise ValueError("input_generator output is empty")
    if not _is_multi_turn(conversation_type):
        return [{"role": "user", "content": content}]

    payload = json.loads(_extract_json_payload(content))
    if isinstance(payload, dict) and "messages" in payload:
        payload = payload["messages"]
    if not isinstance(payload, list):
        raise ValueError("multi-turn input_generator output must be a JSON message array")

    messages = [_normalize_message(item) for item in payload]
    if len(messages) < 2:
        raise ValueError("multi-turn input_generator output must contain at least two messages")
    if messages[0]["role"] != "user":
        raise ValueError("multi-turn input_generator output must start with user")
    if messages[-1]["role"] != "user":
        raise ValueError("multi-turn input_generator output must end with user")
    for prev, current in zip(messages, messages[1:]):
        if prev["role"] == current["role"]:
            raise ValueError("multi-turn messages must alternate roles")
    return messages


def _generate_one_input(
    item: AnchorPrompt,
    api_config: ChatAPIConfig,
    max_tokens: int | None,
    temperature: float,
    top_p: float,
    timeout: float,
    max_retries: int,
    chat_fn: ChatFn,
) -> GeneratedInputAnchor:
    conversation_type = str(item.anchor_meta.get("conversation_type", "single_turn"))
    last_error: Exception | None = None
    for _attempt in range(max_retries + 1):
        try:
            raw = chat_fn(api_config, item.messages, max_tokens, temperature, top_p, timeout)
            messages = parse_input_generator_messages(raw, conversation_type)
            meta = dict(item.anchor_meta)
            meta["input_generator_model"] = api_config.model_name
            return GeneratedInputAnchor(
                id=item.id,
                messages=messages,
                input_generator_model=api_config.model_name,
                anchor_meta=meta,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
    raise RuntimeError(f"input generation failed after max retries: {last_error}") from last_error


def generate_anchor_inputs(
    api_config: ChatAPIConfig,
    input_path: str | Path,
    output_path: str | Path,
    max_tokens: int | None = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    timeout: float = 60,
    max_retries: int = 2,
    limit: int | None = None,
    stats_output: str | Path | None = None,
    chat_fn: ChatFn = chat_completion,
    logger: LogFn | None = None,
) -> tuple[list[GeneratedInputAnchor], InputGenerationStats]:
    prompts = read_anchor_prompts(input_path)
    if limit is not None:
        prompts = prompts[:limit]

    stats = InputGenerationStats(input_count=len(prompts))
    generated_inputs: list[GeneratedInputAnchor] = []
    start = perf_counter()
    for idx, item in enumerate(prompts, start=1):
        item_start = perf_counter()
        try:
            generated_inputs_item = _generate_one_input(
                item=item,
                api_config=api_config,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                timeout=timeout,
                max_retries=max_retries,
                chat_fn=chat_fn,
            )
            generated_inputs.append(generated_inputs_item)
            if logger:
                logger(
                    "stage=input_generation "
                    f"idx={idx}/{len(prompts)} status=kept "
                    f"conversation={generated_inputs_item.anchor_meta.get('conversation_type', 'unknown')} "
                    f"kept={len(generated_inputs)} failed={stats.failed_count} "
                    f"elapsed={perf_counter() - item_start:.1f}s total={perf_counter() - start:.1f}s"
                )
        except Exception as exc:  # noqa: BLE001
            stats.failed_count += 1
            if "request failed" in str(exc):
                stats.api_failed_count += 1
            else:
                stats.parse_failed_count += 1
            if logger:
                logger(
                    "stage=input_generation "
                    f"idx={idx}/{len(prompts)} status=failed "
                    f"kept={len(generated_inputs)} failed={stats.failed_count} "
                    f"api_failed={stats.api_failed_count} parse_failed={stats.parse_failed_count} "
                    f"error={type(exc).__name__} elapsed={perf_counter() - item_start:.1f}s "
                    f"total={perf_counter() - start:.1f}s"
                )

    stats.kept_count = len(generated_inputs)
    write_jsonl(generated_inputs, output_path)
    if stats_output:
        output = Path(stats_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(stats.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return generated_inputs, stats

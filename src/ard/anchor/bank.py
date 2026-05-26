from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).lower()


def stable_hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def anchor_id(prompt: str, salt: str = "") -> str:
    payload = f"{salt}\n{prompt}"
    return f"anchor_{stable_hash(payload)}"


@dataclass(slots=True)
class AnchorPrompt:
    id: str
    messages: list[dict[str, str]]
    anchor_meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_prompt(cls, prompt: str, meta: dict[str, Any], salt: str = "") -> "AnchorPrompt":
        return cls(
            id=anchor_id(prompt, salt),
            messages=[{"role": "user", "content": prompt}],
            anchor_meta=meta,
        )

    def prompt_text(self) -> str:
        return "\n\n".join(msg["content"] for msg in self.messages if msg.get("role") == "user")

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass(slots=True)
class TargetAnswerAnchor:
    id: str
    messages: list[dict[str, str]]
    target_answer: str
    target_model: str
    anchor_meta: dict[str, Any] = field(default_factory=dict)
    input_generator_model: str = ""

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "TargetAnswerAnchor":
        return cls(
            id=str(record["id"]),
            messages=list(record["messages"]),
            target_answer=str(record["target_answer"]),
            target_model=str(record["target_model"]),
            anchor_meta=dict(record.get("anchor_meta", {})),
            input_generator_model=str(
                record.get(
                    "input_generator_model",
                    record.get("anchor_meta", {}).get("input_generator_model", ""),
                )
            ),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass(slots=True)
class GeneratedInputAnchor:
    id: str
    messages: list[dict[str, str]]
    input_generator_model: str
    anchor_meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "GeneratedInputAnchor":
        return cls(
            id=str(record["id"]),
            messages=list(record["messages"]),
            input_generator_model=str(
                record.get(
                    "input_generator_model",
                    record.get("anchor_meta", {}).get("input_generator_model", ""),
                )
            ),
            anchor_meta=dict(record.get("anchor_meta", {})),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass(slots=True)
class FilterStats:
    input_count: int = 0
    kept_count: int = 0
    dropped_empty: int = 0
    dropped_failure: int = 0
    dropped_too_short: int = 0
    dropped_too_long: int = 0
    dropped_duplicate_prompt: int = 0
    dropped_duplicate_answer: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def read_anchor_prompts(path: str | Path) -> list[AnchorPrompt]:
    items: list[AnchorPrompt] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            items.append(
                AnchorPrompt(
                    id=str(record["id"]),
                    messages=list(record["messages"]),
                    anchor_meta=dict(record.get("anchor_meta", {})),
                )
            )
    return items


def read_target_answer_anchors(path: str | Path) -> list[TargetAnswerAnchor]:
    items: list[TargetAnswerAnchor] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            items.append(TargetAnswerAnchor.from_record(json.loads(line)))
    return items


def read_generated_input_anchors(path: str | Path) -> list[GeneratedInputAnchor]:
    items: list[GeneratedInputAnchor] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            items.append(GeneratedInputAnchor.from_record(json.loads(line)))
    return items


def write_jsonl(items: list[Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(item.to_json() + "\n")


def filter_target_answer_anchors(
    anchors: list[TargetAnswerAnchor],
    min_answer_chars: int = 8,
    max_answer_chars: int | None = None,
    failure_markers: tuple[str, ...] = ("generation failed", "failed after max retries"),
    existing_normalized_prompts: set[str] | None = None,
    existing_normalized_answers: set[str] | None = None,
) -> tuple[list[TargetAnswerAnchor], FilterStats]:
    stats = FilterStats(input_count=len(anchors))
    kept: list[TargetAnswerAnchor] = []
    seen_prompts: set[str] = set(existing_normalized_prompts or ())
    seen_answers: set[str] = set(existing_normalized_answers or ())

    for anchor in anchors:
        prompt = "\n\n".join(msg.get("content", "") for msg in anchor.messages)
        answer = anchor.target_answer.strip()
        norm_prompt = normalize_text(prompt)
        norm_answer = normalize_text(answer)

        if not answer:
            stats.dropped_empty += 1
            continue
        if any(marker in norm_answer for marker in failure_markers):
            stats.dropped_failure += 1
            continue
        if len(answer) < min_answer_chars:
            stats.dropped_too_short += 1
            continue
        if max_answer_chars is not None and len(answer) > max_answer_chars:
            stats.dropped_too_long += 1
            continue
        if norm_prompt in seen_prompts:
            stats.dropped_duplicate_prompt += 1
            continue
        if norm_answer in seen_answers:
            stats.dropped_duplicate_answer += 1
            continue

        seen_prompts.add(norm_prompt)
        seen_answers.add(norm_answer)
        kept.append(anchor)

    stats.kept_count = len(kept)
    return kept, stats


def split_target_answer_anchors(
    anchors: list[TargetAnswerAnchor],
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[TargetAnswerAnchor], list[TargetAnswerAnchor]]:
    if not 0 <= eval_ratio < 1:
        raise ValueError("eval_ratio must be in [0, 1)")
    shuffled = anchors[:]
    random.Random(seed).shuffle(shuffled)
    eval_count = int(round(len(shuffled) * eval_ratio))
    eval_items = shuffled[:eval_count]
    train_items = shuffled[eval_count:]
    return train_items, eval_items

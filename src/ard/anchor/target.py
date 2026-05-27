from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import torch

from ard.anchor.api_client import ChatAPIConfig, chat_completion
from ard.anchor.bank import (
    TargetAnswerAnchor,
    read_anchor_prompts,
    read_generated_input_anchors,
    write_jsonl,
)

LogFn = Callable[[str], None]


@dataclass(slots=True)
class TargetAnswerStats:
    input_count: int = 0
    kept_count: int = 0
    failed_count: int = 0
    api_failed_count: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def ensure_tokenizer_ready(tokenizer: Any) -> None:
    """Ensure a causal-LM tokenizer has a pad token for batching/generation."""
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is not None:
            tokenizer.pad_token = eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})


def render_messages(
    tokenizer: Any,
    messages: list[dict[str, str]],
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **(chat_template_kwargs or {}),
        )
    return "\n\n".join(message.get("content", "") for message in messages)


@torch.no_grad()
def answer_anchor_prompts(
    model_path: str,
    input_path: str | Path,
    output_path: str | Path,
    trust_remote_code: bool = True,
    torch_dtype: str = "bfloat16",
    max_new_tokens: int = 512,
    max_prompt_length: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.95,
    chat_template_kwargs: dict[str, Any] | None = None,
    limit: int | None = None,
) -> list[TargetAnswerAnchor]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    ensure_tokenizer_ready(tokenizer)
    dtype = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[torch_dtype.lower()]
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()

    prompts = read_anchor_prompts(input_path)
    if limit is not None:
        prompts = prompts[:limit]

    answered: list[TargetAnswerAnchor] = []
    for item in prompts:
        rendered = render_messages(tokenizer, item.messages, chat_template_kwargs)
        inputs = tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_length,
        )
        input_device = next(model.parameters()).device
        inputs = {key: value.to(input_device) for key, value in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        answer = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True).strip()
        answered.append(
            TargetAnswerAnchor(
                id=item.id,
                messages=item.messages,
                target_answer=answer,
                target_model=model_path,
                anchor_meta=item.anchor_meta,
                input_generator_model=item.anchor_meta.get("input_generator_model", ""),
            )
        )

    write_jsonl(answered, output_path)
    return answered


def answer_one_generated_input_api(
    item: Any,
    api_config: ChatAPIConfig,
    max_tokens: int | None = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    timeout: float = 60,
    chat_fn: Any = chat_completion,
) -> TargetAnswerAnchor:
    answer = chat_fn(
        config=api_config,
        messages=item.messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout,
    )
    meta = dict(item.anchor_meta)
    meta["target_model"] = api_config.model_name
    return TargetAnswerAnchor(
        id=item.id,
        messages=item.messages,
        target_answer=answer,
        target_model=api_config.model_name,
        anchor_meta=meta,
        input_generator_model=item.input_generator_model,
    )


def answer_generated_inputs_api(
    api_config: ChatAPIConfig,
    input_path: str | Path,
    output_path: str | Path,
    max_tokens: int | None = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    timeout: float = 60,
    limit: int | None = None,
    stats_output: str | Path | None = None,
    chat_fn: Any = chat_completion,
    logger: LogFn | None = None,
) -> list[TargetAnswerAnchor]:
    generated_inputs = read_generated_input_anchors(input_path)
    if limit is not None:
        generated_inputs = generated_inputs[:limit]

    stats = TargetAnswerStats(input_count=len(generated_inputs))
    answered: list[TargetAnswerAnchor] = []
    start = perf_counter()
    for idx, item in enumerate(generated_inputs, start=1):
        item_start = perf_counter()
        try:
            answered_item = answer_one_generated_input_api(
                item=item,
                api_config=api_config,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                timeout=timeout,
                chat_fn=chat_fn,
            )
        except Exception:  # noqa: BLE001
            stats.failed_count += 1
            stats.api_failed_count += 1
            if logger:
                logger(
                    "stage=target_answer "
                    f"idx={idx}/{len(generated_inputs)} status=failed "
                    f"kept={len(answered)} failed={stats.failed_count} "
                    f"error=api elapsed={perf_counter() - item_start:.1f}s "
                    f"total={perf_counter() - start:.1f}s"
                )
            continue
        answered.append(answered_item)
        if logger:
            logger(
                "stage=target_answer "
                f"idx={idx}/{len(generated_inputs)} status=kept "
                f"kept={len(answered)} failed={stats.failed_count} "
                f"answer_chars={len(answered_item.target_answer)} "
                f"elapsed={perf_counter() - item_start:.1f}s "
                f"total={perf_counter() - start:.1f}s"
            )

    stats.kept_count = len(answered)
    write_jsonl(answered, output_path)
    if stats_output:
        output = Path(stats_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(stats.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return answered

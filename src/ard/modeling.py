from __future__ import annotations

from pathlib import Path
from typing import Any


def ensure_tokenizer_ready(tokenizer: Any) -> None:
    """Ensure a causal-LM tokenizer has a pad token for batching/generation."""
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is not None:
            tokenizer.pad_token = eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})


def _infer_lora_targets(model: Any) -> list[str]:
    candidates = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "W_pack",
        "c_attn",
        "c_proj",
    }
    found: set[str] = set()
    for name, module in model.named_modules():
        if module.__class__.__name__.lower() != "linear":
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in candidates:
            found.add(leaf)
    return sorted(found) or ["q_proj", "v_proj"]


def build_peft_config(config: Any, model: Any):
    from peft import LoraConfig

    target_modules = config.lora.target_modules
    if target_modules is None and config.lora.auto_target_modules:
        target_modules = _infer_lora_targets(model)
    return LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=target_modules,
        bias=config.lora.bias,
        task_type=config.lora.task_type,
    )


def save_lora_adapter(model: Any, tokenizer: Any, path: str | Path) -> None:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)

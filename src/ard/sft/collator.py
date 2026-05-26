from __future__ import annotations

from typing import Any, cast

import torch

from ard.sft.data import SFTSample


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
    return "\n\n".join(item["content"] for item in messages)


class ARDDataCollator:
    def __init__(
        self, tokenizer: Any, max_length: int, chat_template_kwargs: dict[str, Any] | None = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chat_template_kwargs = chat_template_kwargs or {}

    def __call__(self, samples: list[SFTSample]) -> dict[str, torch.Tensor]:
        input_ids_list: list[torch.Tensor] = []
        attention_masks: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        weights: list[float] = []
        is_anchor: list[bool] = []

        for sample in samples:
            prompt_text = render_messages(
                self.tokenizer, sample.messages, self.chat_template_kwargs
            )
            target_text = sample.target + (self.tokenizer.eos_token or "")
            prompt_ids = self.tokenizer(
                prompt_text,
                add_special_tokens=False,
            )["input_ids"]
            target_ids = self.tokenizer(
                target_text,
                add_special_tokens=False,
            )["input_ids"]
            prompt_ids = cast(list[int], prompt_ids)
            target_ids = cast(list[int], target_ids)
            full_ids = (prompt_ids + target_ids)[: self.max_length]
            labels = [-100] * len(full_ids)
            for idx in range(min(len(prompt_ids), len(full_ids)), len(full_ids)):
                labels[idx] = full_ids[idx]
            input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
            attention_masks.append(torch.ones(len(full_ids), dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))
            is_anchor.append(sample.sample_type == "anchor")
            weights.append(1.0)

        max_len = max(len(ids) for ids in input_ids_list)
        pad_id = self.tokenizer.pad_token_id
        batch_input_ids: list[torch.Tensor] = []
        batch_attention: list[torch.Tensor] = []
        batch_labels: list[torch.Tensor] = []
        for input_tensor, attention_tensor, label_tensor in zip(
            input_ids_list, attention_masks, labels_list
        ):
            pad_len = max_len - len(input_tensor)
            batch_input_ids.append(
                torch.nn.functional.pad(input_tensor, (0, pad_len), value=pad_id)
            )
            batch_attention.append(torch.nn.functional.pad(attention_tensor, (0, pad_len), value=0))
            batch_labels.append(torch.nn.functional.pad(label_tensor, (0, pad_len), value=-100))

        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention),
            "labels": torch.stack(batch_labels),
            "is_anchor": torch.tensor(is_anchor, dtype=torch.bool),
            "sample_weights": torch.tensor(weights, dtype=torch.float),
        }

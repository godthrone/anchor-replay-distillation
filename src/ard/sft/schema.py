from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ModelConfig:
    model_path: str = "${MODEL_PATH}"
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: str | None = None
    gradient_checkpointing: bool = True
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] | None = None
    auto_target_modules: bool = True
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass(slots=True)
class ARDDataConfig:
    hard_train_path: str = "${HARD_DATA_PATH}"
    anchor_train_path: str = "${ANCHOR_DATA_PATH}"
    max_length: int = 4096


@dataclass(slots=True)
class KLConfig:
    enabled: bool = False
    weight: float = 0.05
    temperature: float = 1.0


@dataclass(slots=True)
class ARDTrainingConfig:
    output_dir: str = "${OUTPUT_DIR}"
    seed: int = 42
    total_epochs: int = 1
    max_steps: int = -1
    per_device_batch_size: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    anchor_ce_weight: float = 0.2
    save_steps: int = 100
    logging_steps: int = 10
    dataloader_num_workers: int = 0


@dataclass(slots=True)
class ARDSFTConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: ARDDataConfig = field(default_factory=ARDDataConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: ARDTrainingConfig = field(default_factory=ARDTrainingConfig)
    kl_distillation: KLConfig = field(default_factory=KLConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ARDSFTConfig":
        import yaml

        text = Path(path).read_text(encoding="utf-8")
        text = os.path.expandvars(text)
        return cls.from_dict(yaml.safe_load(text))

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ARDSFTConfig":
        data = data or {}
        return cls(
            model=ModelConfig(**data.get("model", {})),
            data=ARDDataConfig(**data.get("data", {})),
            lora=LoRAConfig(**data.get("lora", {})),
            training=ARDTrainingConfig(**data.get("training", {})),
            kl_distillation=KLConfig(**data.get("kl_distillation", {})),
        )

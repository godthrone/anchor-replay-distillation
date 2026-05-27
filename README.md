# Anchor Replay Distillation

中文 | [English](#english)

Anchor Replay Distillation（ARD）是一个 two-hop 数据生成工具，用于构建 replay anchors，在领域微调时帮助保护基座模型的通用能力。仓库默认提供一键数据生成入口：创建 `.venv`、安装依赖、调用 OpenAI-compatible API，并输出 SFT-ready `anchor_bank.jsonl`。

## 快速开始

1. 安装 uv：

```bash
pipx install uv
```

也可以参考 uv 官方安装方式：https://docs.astral.sh/uv/

2. 配置 API：

```bash
cp .env-example .env
```

打开 `.env`，填入：

```env
ARD_INPUT_GENERATOR_API_BASE=https://example.com/v1
ARD_INPUT_GENERATOR_MODEL_NAME=deepseek-v4-flash
ARD_INPUT_GENERATOR_API_KEY=replace-me

ARD_TARGET_API_BASE=https://example.com/v1
ARD_TARGET_MODEL_NAME=deepseek-v4-flash
ARD_TARGET_API_KEY=replace-me
```

`ARD_INPUT_GENERATOR_*` 是更强的输入生成模型，用来把 anchor spec 写成真实用户消息。`ARD_TARGET_*` 是你要训练、评估或保持能力的目标模型，它的回答会成为 SFT 监督目标。

`.env` 包含密钥，不要提交；`.env-example` 是可提交的配置模板。

3. 一键生成默认数据：

```bash
uv run python scripts/generate_anchors.py
```

脚本会自动运行 `uv sync --extra dev`，然后默认尝试生成 10 个候选 anchors 到：

```text
outputs/ard_anchor_dataset_default/
```

主要输出：

- `anchor_bank.jsonl`：最终 SFT-ready 数据
- `manifest.json`：数据分布、seed、模型元数据
- `input_generation_stats.json` / `target_answer_stats.json`：生成统计
- `batch_*/`：中间 prompts/generated-inputs/target-answers，便于排查

默认是“尝试数量语义”：`ARD_TARGET_COUNT=10` 表示尝试 10 个候选，过滤后 `anchor_bank.jsonl` 可能少于 10 条。过滤只筛明显坏样本；如果你必须拿到精确数量，把 `ARD_REQUIRE_EXACT_COUNT=true`。

## 修改默认参数

所有常用参数都在 `.env` 中修改：

```env
ARD_TARGET_COUNT=10
ARD_BATCH_SIZE=10
ARD_MAX_BATCHES=3
ARD_REQUIRE_EXACT_COUNT=false
ARD_SEED=42
ARD_OUTPUT_DIR=outputs/ard_anchor_dataset_default
ARD_CONFIG_PATH=configs/anchor_generation.yaml
ARD_LANGUAGES=
ARD_TASK_TYPES=
ARD_TEMPERATURE=0.7
ARD_TOP_P=0.95
ARD_TIMEOUT=60
ARD_MAX_RETRIES=2
ARD_INPUT_GENERATOR_CONCURRENCY=4
ARD_TARGET_CONCURRENCY=4
ARD_MIN_TARGET_ANSWER_CHARS=8
ARD_MAX_TARGET_ANSWER_CHARS=
ARD_INPUT_GENERATOR_MAX_TOKENS=
ARD_TARGET_MAX_TOKENS=
```

`ARD_INPUT_GENERATOR_CONCURRENCY` 和 `ARD_TARGET_CONCURRENCY` 分别控制两个 API 阶段的并发度。端到端构建器采用流水线执行：一条 input 生成完成后，可以立刻开始 target answer，不需要等全部 input 都生成完。

`ARD_LANGUAGES` 和 `ARD_TASK_TYPES` 使用逗号分隔，例如：

```env
ARD_LANGUAGES=English,简体中文,bilingual_zh_en
ARD_TASK_TYPES=qa,explanation,reasoning,coding,debugging
```

`ARD_MAX_TARGET_ANSWER_CHARS` 留空表示不过滤长答案。严格要求：默认不要设置 `ARD_INPUT_GENERATOR_MAX_TOKENS` 或 `ARD_TARGET_MAX_TOKENS`。留空时，ARD 不会向大模型 API 发送 `max_tokens` 或等价 token cap，让模型/API 使用默认输出预算。只有你明确要覆盖默认预算时才填写。

## 开发检查

```bash
uv run ruff format .
uv run ruff check --fix .
uv run mypy
uv run pytest -q
```

## English

Anchor Replay Distillation (ARD) is a two-hop data generation utility for building replay anchors that help preserve a base model's general abilities during domain fine-tuning. The repository provides one default entry point that creates `.venv`, installs dependencies, calls an OpenAI-compatible API, and writes an SFT-ready `anchor_bank.jsonl`.

## Quick Start

1. Install uv:

```bash
pipx install uv
```

You can also use the official uv install guide: https://docs.astral.sh/uv/

2. Configure the API:

```bash
cp .env-example .env
```

Open `.env` and fill in:

```env
ARD_INPUT_GENERATOR_API_BASE=https://example.com/v1
ARD_INPUT_GENERATOR_MODEL_NAME=deepseek-v4-flash
ARD_INPUT_GENERATOR_API_KEY=replace-me

ARD_TARGET_API_BASE=https://example.com/v1
ARD_TARGET_MODEL_NAME=deepseek-v4-flash
ARD_TARGET_API_KEY=replace-me
```

`ARD_INPUT_GENERATOR_*` is the stronger input generator model that turns anchor specs into realistic user messages. `ARD_TARGET_*` is the model you are training, evaluating, or preserving; its answers become the SFT target.

`.env` contains secrets and must not be committed. `.env-example` is the committed template.

3. Generate the default dataset:

```bash
uv run python scripts/generate_anchors.py
```

The script runs `uv sync --extra dev`, then attempts 10 candidate anchors by default under:

```text
outputs/ard_anchor_dataset_default/
```

Main outputs:

- `anchor_bank.jsonl`: final SFT-ready data
- `manifest.json`: distribution, seed, and model metadata
- `input_generation_stats.json` / `target_answer_stats.json`: generation stats
- `batch_*/`: intermediate prompts/generated-inputs/target-answers for debugging

The default uses attempt-count semantics: `ARD_TARGET_COUNT=10` means ARD attempts 10 candidates, so `anchor_bank.jsonl` may contain fewer than 10 records after filtering. Filtering is intended to remove only clearly bad samples; set `ARD_REQUIRE_EXACT_COUNT=true` only when you need an exact final count.

## Changing Defaults

Edit `.env` to customize common parameters:

```env
ARD_TARGET_COUNT=10
ARD_BATCH_SIZE=10
ARD_MAX_BATCHES=3
ARD_REQUIRE_EXACT_COUNT=false
ARD_SEED=42
ARD_OUTPUT_DIR=outputs/ard_anchor_dataset_default
ARD_CONFIG_PATH=configs/anchor_generation.yaml
ARD_LANGUAGES=
ARD_TASK_TYPES=
ARD_TEMPERATURE=0.7
ARD_TOP_P=0.95
ARD_TIMEOUT=60
ARD_MAX_RETRIES=2
ARD_INPUT_GENERATOR_CONCURRENCY=4
ARD_TARGET_CONCURRENCY=4
ARD_MIN_TARGET_ANSWER_CHARS=8
ARD_MAX_TARGET_ANSWER_CHARS=
ARD_INPUT_GENERATOR_MAX_TOKENS=
ARD_TARGET_MAX_TOKENS=
```

`ARD_INPUT_GENERATOR_CONCURRENCY` and `ARD_TARGET_CONCURRENCY` control the two API stages independently. The end-to-end builder is pipelined: once one input is generated, its target answer can start immediately without waiting for all inputs to finish.

`ARD_LANGUAGES` and `ARD_TASK_TYPES` are comma-separated, for example:

```env
ARD_LANGUAGES=English,简体中文,bilingual_zh_en
ARD_TASK_TYPES=qa,explanation,reasoning,coding,debugging
```

Leave `ARD_MAX_TARGET_ANSWER_CHARS` blank to keep long target answers. Strict requirement: do not set `ARD_INPUT_GENERATOR_MAX_TOKENS` or `ARD_TARGET_MAX_TOKENS` by default. When they are blank, ARD does not send `max_tokens` or any equivalent token cap to large-model API calls, letting the provider/model use its default output budget. Fill them only when you intentionally want to override that default.

## Development Checks

```bash
uv run ruff format .
uv run ruff check --fix .
uv run mypy
uv run pytest -q
```

# Anchor Replay Distillation

English | [简体中文](README_zh.md)

Anchor Replay Distillation (ARD) helps you generate ready-to-use chat data for
supervised fine-tuning. You configure two OpenAI-compatible models:
`ARD_INPUT_GENERATOR_*` writes realistic user requests, and `ARD_TARGET_*`
answers those requests as the model you want to train, evaluate, or preserve.
The result is an SFT-ready `anchor_bank.jsonl` that you can mix into a
fine-tuning dataset to reduce forgetting of general skills such as question
answering, reasoning, translation, coding, and safety behavior.

The default workflow is intentionally simple: one command creates `.venv`,
installs dependencies, calls the APIs, and writes the generated data.

## Quick Start

1. Install uv and verify the toolchain.

This project requires Python 3.11 or newer, but you do not need to install Python or
`pipx` first when using uv's standalone installer. uv can install and manage the
project Python for you.

Windows PowerShell:

If PowerShell reports that the execution policy blocks the installer, enable
user-level signed scripts first. This does not require administrator privileges:

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then run the uv installer:

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

If you prefer a temporary policy change for this terminal session only, use
`Set-ExecutionPolicy Bypass -Scope Process -Force` before the installer instead.

macOS/Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close and reopen the terminal if `uv` is not found immediately, then verify:

```bash
uv --version
uv run --no-project --python 3.11 python --version
```

The Python check should report Python 3.11 or newer without installing this
project's dependencies. If no compatible Python is installed yet, uv may download
one automatically. You can also install the project Python explicitly:

```bash
uv python install 3.11
```

If you already have Python and `pipx`, `pipx install uv` is also fine. See the
official uv installation guide for more options: https://docs.astral.sh/uv/

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

For the expected JSONL shape without any real API output, see
`examples/anchor_bank.sample.jsonl`.

During generation, logs include timestamps, progress percentage, throughput, and
ETA. Input generation and target answer logs are interleaved because both model
stages run as a pipeline.

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

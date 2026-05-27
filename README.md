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

All user-facing generation settings live in `.env`. The default seed is
`ARD_SEED=42`; change it when you want a different reproducible sample mix.

3. Generate the default dataset:

```bash
uv run python scripts/generate_anchors.py
```

The script runs `uv sync --extra dev`, then attempts 10 candidate anchors by default under a timestamped run directory:

```text
outputs/ard_anchor_dataset_YYYYMMDD_HHMMSS_n10_seed42/
```

Main outputs:

- `anchor_bank.jsonl`: final SFT-ready data
- `manifest.json`: distribution, seed, and model metadata
- `input_generation_stats.json` / `target_answer_stats.json`: generation stats
- `batch_*/`: intermediate prompts/generated-inputs/target-answers for debugging

For the expected JSONL shape without any real API output, see
[examples/anchor_bank.sample.jsonl](examples/anchor_bank.sample.jsonl).

During generation, logs include timestamps, progress percentage, throughput, and
ETA. Input generation and target answer logs are interleaved because both model
stages run as a pipeline.

The default uses attempt-count semantics: `ARD_TARGET_COUNT=10` means ARD attempts 10 candidates, so `anchor_bank.jsonl` may contain fewer than 10 records after filtering. Filtering is intended to remove only clearly bad samples; set `ARD_EXACT_FINAL_COUNT_ENABLED=true` only when you need the final kept row count to reach `ARD_TARGET_COUNT`.

## Output Format

`anchor_bank.jsonl` is newline-delimited JSON. Each line is one SFT-ready chat
sample. Check `messages` to understand whether the sample is single-turn or
multi-turn, and check `anchor_meta` to understand its language, capability,
task type, safety boundary, and sampling bucket.

| Field | Meaning |
| --- | --- |
| `id` | Stable anchor sample ID. |
| `messages` | Chat input context. Single-turn samples contain one user message; multi-turn samples contain prior user/assistant turns plus the final user request. |
| `target_answer` | The target model answer used as the SFT supervision text. |
| `target_model` | Model that produced `target_answer`. |
| `input_generator_model` | Strong model that generated the realistic user input. |
| `anchor_meta` | Sampling metadata and capability labels. |

Key `anchor_meta` fields:

| Field | Meaning |
| --- | --- |
| `language` | Output/input language bucket, such as `English`, `简体中文`, or `bilingual_zh_en`. |
| `knowledge_domain` | Knowledge or work domain sampled for the anchor. |
| `capability` | Capability being preserved, such as explanation, comparison, reasoning, tool choice, or uncertainty handling. |
| `task_type` | Task label used for filtering or balancing. |
| `conversation_type` | Conversation pattern, such as `single_turn`, `troubleshooting_3_turn`, or `constraint_update_4_turn`. |
| `is_multi_turn` | Whether `messages` includes multiple conversation turns. |
| `safety_boundary` | Expected safety/authority behavior, such as standard answer, clarification, refusal, or safe redirect. |
| `seed` | Seed used for reproducible sampling. |

## Changing Defaults

For runtime settings, `.env` is the main configuration surface. `.env-example`
lists every supported `ARD_*` environment variable, including the default seed
`ARD_SEED=42`. All sampling choices for generated questions live in one JSON
file: `data/anchor_seed/anchor_ontology.json`. Copy and edit that file when you
want ARD to generate data for your own domains, languages, capabilities,
conversation types, safety boundaries, or language features.

Edit `.env` to customize common parameters:

```env
ARD_TARGET_COUNT=10
ARD_EXACT_FINAL_COUNT_ENABLED=false
ARD_EXACT_FINAL_COUNT_BATCH_SIZE=10
ARD_EXACT_FINAL_COUNT_MAX_BATCHES=3
ARD_SEED=42
ARD_OUTPUT_DIR=
ARD_OVERWRITE_OUTPUT=false
ARD_CONFIG_PATH=configs/anchor_generation.yaml
ARD_ONTOLOGY_PATH=data/anchor_seed/anchor_ontology.json
ARD_LANGUAGES=
ARD_TASK_TYPES=
ARD_TEMPERATURE=0.7
ARD_TOP_P=0.95
ARD_TIMEOUT=60
ARD_MAX_RETRIES=2
ARD_INPUT_GENERATOR_CONCURRENCY=100
ARD_TARGET_CONCURRENCY=100
ARD_MIN_TARGET_ANSWER_CHARS=8
ARD_MAX_TARGET_ANSWER_CHARS=
ARD_INPUT_GENERATOR_MAX_TOKENS=
ARD_TARGET_MAX_TOKENS=
```

`ARD_INPUT_GENERATOR_CONCURRENCY` and `ARD_TARGET_CONCURRENCY` control the two API stages independently. The defaults assume a provider/model tier that allows high request concurrency; lower them if your API provider rate-limits or times out. The end-to-end builder is pipelined: once one input is generated, its target answer can start immediately without waiting for all inputs to finish.

Leave `ARD_OUTPUT_DIR` blank to avoid overwriting previous runs. If you set a fixed
`ARD_OUTPUT_DIR`, ARD refuses to write into a non-empty directory unless
`ARD_OVERWRITE_OUTPUT=true`.

`ARD_LANGUAGES` and `ARD_TASK_TYPES` are comma-separated runtime filters. Leave
them blank to use the full ontology defaults, or narrow a run like this:

```env
ARD_LANGUAGES=English,简体中文,bilingual_zh_en
ARD_TASK_TYPES=qa,explanation,reasoning,coding,debugging
```

Leave `ARD_MAX_TARGET_ANSWER_CHARS` blank to keep long target answers. Strict requirement: do not set `ARD_INPUT_GENERATOR_MAX_TOKENS` or `ARD_TARGET_MAX_TOKENS` by default. When they are blank, ARD does not send `max_tokens` or any equivalent token cap to large-model API calls, letting the provider/model use its default output budget. Fill them only when you intentionally want to override that default.

## Sampling Ontology

ARD samples from a single JSON file: `data/anchor_seed/anchor_ontology.json`. The
file is plain nested JSON. Dictionaries create paths, lists contain leaf values,
and each leaf becomes a possible sampling choice.

Top-level sections:

| Section | Meaning | Default top-level values |
| --- | --- | --- |
| `languages` | Output language buckets. | `English`, `简体中文`, `bilingual_zh_en` |
| `knowledge_domains` | Knowledge/work domains. | `software_engineering`, `systems_devops`, `data_ai_ml`, `math_logic`, `science_engineering`, `business_operations`, `finance_economics`, `law_policy_safety`, `medicine_health_safety`, `humanities_world_knowledge`, `language_writing_translation`, `agent_tool_use` |
| `capabilities` | Capability/task labels. | `knowledge_response`, `reasoning`, `coding_and_data`, `language_work`, `agentic_behavior` |
| `conversation_types` | Single-turn and multi-turn shapes. | `single_turn`, `clarification`, `troubleshooting`, `revision`, `tool_and_safety` |
| `safety_boundaries` | Safety and authority-boundary behavior. | `normal`, `regulated_domain`, `boundary` |
| `language_features` | Style, format, difficulty, context length, noise, and answer expectation. | `style`, `format`, `difficulty`, `context_length`, `noise`, `answer_expectation` |

To customize the sampling space, copy `data/anchor_seed/anchor_ontology.json`,
edit the copy, then set `ARD_ONTOLOGY_PATH` to that file. To only narrow one run,
use `ARD_LANGUAGES` or `ARD_TASK_TYPES` without editing the ontology.

## Development Checks

```bash
uv run ruff format .
uv run ruff check --fix .
uv run mypy
uv run pytest -q
```

## Environment Variables

All runtime configuration lives in `.env`. `.env-example` is the short template;
this section is the complete reference.

| Variable | Default | Meaning / when to change |
| --- | --- | --- |
| `ARD_INPUT_GENERATOR_API_BASE` | required | OpenAI-compatible base URL for the strong input generator model. |
| `ARD_INPUT_GENERATOR_MODEL_NAME` | required | Strong model used to create realistic user requests. |
| `ARD_INPUT_GENERATOR_API_KEY` | required | API key for the input generator model. Never commit `.env`. |
| `ARD_TARGET_API_BASE` | required | OpenAI-compatible base URL for the target model. |
| `ARD_TARGET_MODEL_NAME` | required | Model you are training, evaluating, or preserving; its answers become SFT targets. |
| `ARD_TARGET_API_KEY` | required | API key for the target model. Never commit `.env`. |
| `ARD_TARGET_COUNT` | `10` | Number of candidate inputs to attempt. This is not an exact final row guarantee because filtering may drop bad samples. |
| `ARD_EXACT_FINAL_COUNT_ENABLED` | `false` | Set `true` only when you require the final kept row count to reach `ARD_TARGET_COUNT`. Default `false` keeps API calls predictable. |
| `ARD_EXACT_FINAL_COUNT_BATCH_SIZE` | `10` | Extra candidate batch size used only when exact-final-count mode is enabled. |
| `ARD_EXACT_FINAL_COUNT_MAX_BATCHES` | `3` | Maximum batches to try only when exact-final-count mode is enabled. |
| `ARD_SEED` | `42` | Sampling seed. Change it to get a different but reproducible sample mix. |
| `ARD_OUTPUT_DIR` | blank | Leave blank to create a timestamped directory under `outputs/`. Set a fixed path when you want a stable output location. |
| `ARD_OVERWRITE_OUTPUT` | `false` | Set `true` only when you intentionally want to write into a non-empty output directory. |
| `ARD_CONFIG_PATH` | `configs/anchor_generation.yaml` | Base generation config. Most users should keep the default. |
| `ARD_ONTOLOGY_PATH` | `data/anchor_seed/anchor_ontology.json` | Single ontology JSON containing languages, domains, capabilities, conversation types, safety boundaries, and language features. |
| `ARD_LANGUAGES` | blank | Optional comma-separated language filter, for example `English,简体中文`. Blank uses the ontology defaults. |
| `ARD_TASK_TYPES` | blank | Optional comma-separated task filter, for example `qa,explanation,reasoning`. Blank uses the ontology defaults. |
| `ARD_TEMPERATURE` | `0.7` | Sampling temperature for both model calls. |
| `ARD_TOP_P` | `0.95` | Top-p sampling value for both model calls. |
| `ARD_TIMEOUT` | `60` | Per-request timeout in seconds. Increase this if the provider often returns long answers slowly. |
| `ARD_MAX_RETRIES` | `2` | API retry count for failed calls. |
| `ARD_INPUT_GENERATOR_CONCURRENCY` | `100` | Parallelism for input generation requests. Lower this if your provider rate-limits or times out. |
| `ARD_TARGET_CONCURRENCY` | `100` | Parallelism for target answer requests. Lower this if your provider rate-limits or times out. |
| `ARD_MIN_TARGET_ANSWER_CHARS` | `8` | Drop answers shorter than this many characters. |
| `ARD_MAX_TARGET_ANSWER_CHARS` | blank | Optional max answer length filter. Blank means long answers are kept. |
| `ARD_INPUT_GENERATOR_MAX_TOKENS` | blank | Optional explicit token cap for input generation. Blank means ARD does not send `max_tokens`. |
| `ARD_TARGET_MAX_TOKENS` | blank | Optional explicit token cap for target answers. Blank means ARD does not send `max_tokens`. |

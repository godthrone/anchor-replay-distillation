# Anchor Replay Distillation

English | [简体中文](README_zh.md)

Anchor Replay Distillation (ARD) generates research-neutral chat data for
supervised fine-tuning. You configure two OpenAI-compatible models:
`ARD_INPUT_GENERATOR_*` writes realistic user-side requests from sampled anchor
specs, and `ARD_TARGET_*` answers those requests as the model you want to train,
evaluate, or preserve.

The result is an SFT-ready `anchor_bank.jsonl` that can be mixed into another
fine-tuning dataset to reduce forgetting of broad capabilities such as question
answering, reasoning, translation, coding, writing, tool use, and general world
knowledge. ARD is a data-generation utility; it does not include a training
runner.

## Quick Start

1. Install uv and verify Python 3.11+:

```bash
uv --version
uv run --no-project --python 3.11 python --version
```

If uv is not installed, see https://docs.astral.sh/uv/. On Windows PowerShell,
you may need `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` before
running uv's installer.

2. Configure the APIs:

```bash
cp .env-example .env
```

Fill in the required values:

```env
ARD_INPUT_GENERATOR_API_BASE=https://example.com/v1
ARD_INPUT_GENERATOR_MODEL_NAME=deepseek-v4-flash
ARD_INPUT_GENERATOR_API_KEY=replace-me

ARD_TARGET_API_BASE=https://example.com/v1
ARD_TARGET_MODEL_NAME=deepseek-v4-flash
ARD_TARGET_API_KEY=replace-me
```

`.env` contains secrets and must not be committed. `.env-example` is the
committed template.

3. Generate the default dataset:

```bash
uv run python scripts/generate_anchors.py
```

The script runs `uv sync --extra dev`, then attempts 10 candidate anchors by
default under a timestamped `outputs/ard_anchor_dataset_*` directory.

Main outputs:

- `anchor_bank.jsonl`: final SFT-ready data
- `manifest.json`: distribution, seed, model, and sampling metadata
- `input_generation_stats.json` / `target_answer_stats.json`: generation stats
- `batch_*/`: intermediate prompts, generated inputs, and target answers

For the expected JSONL shape without real API output, see
[examples/anchor_bank.sample.jsonl](examples/anchor_bank.sample.jsonl).

All repository text artifacts and generated JSON/JSONL outputs are written as
UTF-8. Non-ASCII language content is kept directly readable rather than escaped
as `\uXXXX`, except for JSON-required escaping such as newlines and backslashes.

## Output Format

`anchor_bank.jsonl` is newline-delimited JSON. Each line is one SFT-ready chat
sample. `messages` shows the single-turn or multi-turn chat context, and
`anchor_meta` records the research-neutral sampling dimensions.

| Field | Meaning |
| --- | --- |
| `id` | Stable anchor sample ID. |
| `messages` | Chat input context. |
| `target_answer` | Target model answer used as SFT supervision text. |
| `target_model` | Model that produced `target_answer`. |
| `input_generator_model` | Model that generated realistic user-side input. |
| `anchor_meta` | Sampling metadata and capability labels. |

For reasoning models, `target_answer` preserves the target model's training
output format. When an API returns separate `reasoning_content` and `content`,
ARD stores them as:

```text
<think>
reasoning_content
</think>

content
```

If `content` already contains an inline `<think>...</think>` block, ARD keeps it
unchanged. If a model returns no reasoning field, the final answer is saved as
usual.

Key `anchor_meta` fields:

| Field | Meaning |
| --- | --- |
| `language` | Language bucket, such as `English`, `简体中文`, `Español`, or `日本語`. |
| `knowledge_domain` | Knowledge or research domain sampled for the anchor. |
| `capability` | Capability label, such as explanation, comparison, reasoning, tool choice, or writing. |
| `task_type` | Task label used for filtering or balancing. |
| `conversation_type` | Conversation pattern, such as `single_turn` or `constraint_update_4_turn`. |
| `is_multi_turn` | Whether `messages` includes multiple conversation turns. |
| `sampling_strategy` | `farthest`, `balanced`, or `random`. |
| `ontology_sha256` | Hash used to validate ontology embedding sidecars when applicable. |
| `target_reasoning_status` | `separate`, `inline`, `only`, or `absent` for target think coverage. |
| `seed` | Seed used for reproducible sampling. |

## Changing Defaults

Runtime settings live in `.env` and CLI flags. The ontology lives in
`configs/anchor_ontology.json`; the default precomputed embedding sidecar lives
in `configs/anchor_ontology_embeddings.json`.

```env
ARD_TARGET_COUNT=10
ARD_EXACT_FINAL_COUNT_ENABLED=false
ARD_EXACT_FINAL_COUNT_BATCH_SIZE=10
ARD_EXACT_FINAL_COUNT_MAX_BATCHES=3
ARD_SEED=42
ARD_OUTPUT_DIR=
ARD_OVERWRITE_OUTPUT=false
ARD_ONTOLOGY_PATH=configs/anchor_ontology.json
ARD_ONTOLOGY_EMBEDDINGS_PATH=configs/anchor_ontology_embeddings.json
ARD_SAMPLING_STRATEGY=farthest
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

`ARD_LANGUAGES` and `ARD_TASK_TYPES` are comma-separated runtime filters. Leave
them blank to use ontology defaults, or narrow a run like this:

```env
ARD_LANGUAGES=English,简体中文,Español,日本語
ARD_TASK_TYPES=qa,explanation,reasoning,coding,debugging
```

Leave `ARD_MAX_TARGET_ANSWER_CHARS` blank to keep long target answers.

Strict API requirement: do not set `ARD_INPUT_GENERATOR_MAX_TOKENS` or
`ARD_TARGET_MAX_TOKENS` by default. When they are blank, ARD does not send
`max_tokens`, letting the provider/model use its default output budget.

## Sampling Ontology

ARD samples from `configs/anchor_ontology.json`. Dictionaries create paths,
lists contain leaf values, and each leaf becomes a possible sampling choice.

Top-level sections:

| Section | Meaning |
| --- | --- |
| `languages` | Output language buckets. |
| `knowledge_domains` | Broad research and work domains, including science exploration, art, philosophy, religion/folklore, esoterica as cultural phenomena, society, history, culture, unusual phenomena, future speculation, technical work, finance, law, medicine, writing, and tool use. |
| `capabilities` | Capability/task labels. |
| `conversation_types` | Single-turn and multi-turn shapes. |
| `language_features` | Style, format, difficulty, context length, noise, and answer expectation. |

The default `farthest` strategy reads `configs/anchor_ontology_embeddings.json`
and validates its `ontology_sha256` against `configs/anchor_ontology.json`. It
then uses cosine farthest point sampling over `knowledge_domains`, and balances
language, capability, conversation type, and language features. If you change
ontology nodes, regenerate the sidecar with `ard ontology-embed` or use
`--sampling-strategy balanced` until a matching sidecar exists.

To regenerate an embedding sidecar for a custom ontology:

```bash
uv run --extra embed ard ontology-embed \
  --ontology configs/anchor_ontology.json \
  --output configs/anchor_ontology_embeddings.json \
  --backend local \
  --model Qwen/Qwen3-Embedding-0.6B
```

An OpenAI-compatible embedding API backend is also available:

```bash
uv run ard ontology-embed \
  --ontology configs/custom_ontology.json \
  --output configs/custom_ontology_embeddings.json \
  --backend api \
  --api-env-file .env
```

The committed sidecar was generated with `Qwen/Qwen3-Embedding-0.6B`, contains
1024-dimensional embeddings, and is ready for normal users without downloading
an embedding model. Regenerate it with Qwen or an API backend after changing
ontology nodes.

## Validation Snapshot

The current checked-in ontology and Qwen sidecar have been validated with an
end-to-end run using `ARD_TARGET_COUNT=1000` and the default `farthest`
strategy. The run attempted 1000 anchors, kept 1000 generated inputs, produced
986 final target-answer rows, and recorded 14 target API timeout/failure cases.
The full generated `outputs/` directory remains ignored and is not committed.

## Development Checks

```bash
uv run ruff format .
uv run ruff check --fix .
uv run mypy
uv run pytest -q
```

## Environment Variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `ARD_INPUT_GENERATOR_API_BASE` | required | OpenAI-compatible base URL for the input generator model. |
| `ARD_INPUT_GENERATOR_MODEL_NAME` | required | Model used to create realistic user requests. |
| `ARD_INPUT_GENERATOR_API_KEY` | required | API key for the input generator model. |
| `ARD_TARGET_API_BASE` | required | OpenAI-compatible base URL for the target model. |
| `ARD_TARGET_MODEL_NAME` | required | Model whose answers become SFT targets. |
| `ARD_TARGET_API_KEY` | required | API key for the target model. |
| `ARD_TARGET_COUNT` | `10` | Candidate inputs to attempt. Filtering may leave fewer rows. |
| `ARD_EXACT_FINAL_COUNT_ENABLED` | `false` | Try extra batches until final kept rows reach `ARD_TARGET_COUNT`. |
| `ARD_SEED` | `42` | Sampling seed. |
| `ARD_OUTPUT_DIR` | blank | Blank creates a timestamped directory under `outputs/`. |
| `ARD_OVERWRITE_OUTPUT` | `false` | Allow writing into a non-empty output directory. |
| `ARD_ONTOLOGY_PATH` | `configs/anchor_ontology.json` | Ontology JSON. |
| `ARD_ONTOLOGY_EMBEDDINGS_PATH` | `configs/anchor_ontology_embeddings.json` | Embedding sidecar used by `farthest`. |
| `ARD_SAMPLING_STRATEGY` | `farthest` | `farthest`, `balanced`, or `random`. |
| `ARD_LANGUAGES` | blank | Optional comma-separated language filter. |
| `ARD_TASK_TYPES` | blank | Optional comma-separated task filter. |
| `ARD_TEMPERATURE` | `0.7` | Sampling temperature for model calls. |
| `ARD_TOP_P` | `0.95` | Top-p sampling value for model calls. |
| `ARD_TIMEOUT` | `60` | Per-request timeout in seconds. |
| `ARD_MAX_RETRIES` | `2` | API retry count. |
| `ARD_INPUT_GENERATOR_CONCURRENCY` | `100` | Input generation request parallelism. |
| `ARD_TARGET_CONCURRENCY` | `100` | Target answer request parallelism. |
| `ARD_MIN_TARGET_ANSWER_CHARS` | `8` | Drop answers shorter than this many characters. |
| `ARD_MAX_TARGET_ANSWER_CHARS` | blank | Optional max answer length filter. |
| `ARD_INPUT_GENERATOR_MAX_TOKENS` | blank | Optional explicit token cap for input generation. |
| `ARD_TARGET_MAX_TOKENS` | blank | Optional explicit token cap for target answers. |

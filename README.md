# ARD — Anchor Replay Distillation

[中文文档](README.zh-CN.md)

> Anchor dataset generation for continual LLM fine-tuning — ontology-driven
> sampling, teacher-model distillation with token-level log-probabilities,
> and native Graspo/OPD integration.

**ARD** generates high-quality anchor datasets that prevent catastrophic
forgetting during continual fine-tuning. It samples diverse prompts from
a structured knowledge ontology, calls a teacher model to generate answers
with full token-level log-probabilities, and exports in a format directly
compatible with On-Policy Distillation (OPD) and Graspo reinforcement
learning pipelines.

**Key features:**
- 🧠 **Ontology-driven sampling** — Four-dimensional ontology (knowledge
  domains, languages, capabilities, task types) with embedding-based
  hierarchical farthest-point sampling ensures broad, diverse coverage.
- 🎯 **Teacher distillation with logprobs** — Full token-level
  log-probabilities from a strong teacher model provide high-quality
  replay signals for OPD training.
- 🖼️ **Multi-modal support** — Text and text+image anchors in a single
  unified pipeline and output format.
- 📦 **Graspo-compatible** — Output format is directly consumable by
  Graspo anchor bank, no conversion needed.
- ⚡ **One command** — Single CLI, TOML-driven config, secrets separated,
  Docker one-click launch.

## Quick Start

### Prerequisites

- Docker
- Embedding data: `data/anchor_ontology_embeddings.json` (49 pre-computed
  embeddings, 1024-dim, used by the hierarchical FPS sampler)

### 1. Build the Docker image

```bash
git clone https://github.com/your-org/anchor-replay-distillation.git
cd anchor-replay-distillation
bash docker/build.sh
```

### 2. Create your override config

```bash
mkdir -p .local
cp configs/config.override.sample.toml .local/config.override.toml
# Edit .local/config.override.toml: fill in api_base, model_name, api_key
```

Edit `.local/config.override.toml` and fill in your API credentials:

```toml
[input_generator]
api_base = "https://your-api.example.com/v1"
model_name = "your-model-name"
api_key = "your-api-key"

[target_model]
api_base = "https://your-api.example.com/v1"
model_name = "your-model-name"
api_key = "your-api-key"
```

### 3. Generate anchors

```bash
# Text-only anchors
bash run.sh --config configs/config.toml --override .local/config.override.toml

# Multimodal anchors (with sample images)
bash run.sh --config configs/config.toml --override .local/config.override.toml \
    --image-dir examples/images
```

### 4. Check the output

```bash
ls outputs/
```

Find the `ard_dataset_*` directory (e.g. `ard_dataset_20240101_120000`).
See `examples/anchor_bank.sample.jsonl` for the expected format.

## Configuration

ARD uses a **layered TOML configuration** model. There are two config files:

| File | Purpose | Git-tracked? |
|------|---------|-------------|
| `configs/config.toml` | Base config — all fields with defaults | ✅ Yes |
| `.local/config.override.toml` | Override — deployment-specific secrets | ❌ No (`.gitignore`d) |

**`configs/config.toml`** is the single source of truth for the configuration
schema. It defines every field with sensible defaults. Non-secret fields
(e.g., `temperature`, `max_tokens`, `seed`) are ready to use out of the box.
Secret fields (`api_base`, `model_name`, `api_key`) are left empty and must
be filled via the override file.

**`.local/config.override.toml`** contains only the fields you need to
override — typically `api_base`, `model_name`, and `api_key` for both
`[input_generator]` and `[target_model]`. It lives in `.local/` so it is
never committed to git. You cannot add new fields that don't exist in the
base config.

At startup, the override file is deep-merged into the base config.
The merged result is a single config dict used throughout the program.

| Section | Purpose |
|---------|---------|
| `[input_generator]` | VLM/LLM that generates user questions |
| `[target_model]` | Teacher model that generates answers with log-probs |
| `[generation]` | Target count, seed, concurrency, max turns, system persona, language and task type filters |
| `[ontology]` | Ontology file path |
| `[output]` | Output directory settings |

## Configuration Reference

All parameters are defined in `configs/config.toml`. Secret fields (`api_base`,
`model_name`, `api_key`) are left empty and filled in `.local/config.override.toml`.

### `[input_generator]` — Question Generator

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_base` | string | `""` | API endpoint URL (OpenAI-compatible) |
| `model_name` | string | `""` | Model name |
| `api_key` | string | `""` | API key (secret — fill in override) |
| `temperature` | float | `0.8` | Sampling temperature, higher = more random |
| `max_tokens` | int | `4096` | Maximum tokens per response |
| `timeout` | float | `180.0` | Request timeout in seconds |
| `max_retries` | int | `3` | Retries on failure |

### `[target_model]` — Teacher Model

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_base` | string | `""` | API endpoint URL |
| `model_name` | string | `""` | Teacher model name |
| `api_key` | string | `""` | API key (secret) |
| `temperature` | float | `0.0` | Sampling temperature, 0.0 = deterministic |
| `max_tokens` | int | `4096` | Maximum tokens per response |
| `timeout` | float | `180.0` | Request timeout in seconds |
| `max_retries` | int | `3` | Retries on failure |
| `enable_thinking` | bool | `false` | Enable reasoning mode (Qwen3, DeepSeek-R1, etc.). When enabled, the model outputs `...` reasoning before the answer; both `content` and `logprobs` include reasoning tokens. **Only enable when distilling to a reasoning model** |

### `[generation]` — Generation Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_count` | int | `100` | Target number of anchors. Recommend ≥200 for full capability coverage |
| `seed` | int | `42` | Random seed |
| `concurrency` | int | `4` | Concurrent requests; too high may trigger rate limits |
| `languages` | list | `[]` | Language filter (empty = all). Options: `zh-CN`, `en`, `ja`, `ko` |
| `task_types` | list | `[]` | Task type filter (empty = all) |
| `max_turns` | int | `1` | Max conversation turns (1 = single-turn, 2-10 = multi-turn) |
| `system_persona` | string | `"none"` | System persona mode: `none` / `one_sentence` / `appropriate` / `detailed` |
| `max_turns_with_image` | int | `1` | Max turns with image (≤ `max_turns`) |
| `embeddings_path` | string | `"data/anchor_ontology_embeddings.json"` | Pre-computed ontology embedding file |

### `[ontology]` — Ontology

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | string | `"data/anchor_ontology.json"` | Ontology JSON file path |

### `[output]` — Output

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `directory` | string | `""` | Output directory (empty = auto-generated timestamped dir) |
| `overwrite` | bool | `false` | Overwrite existing output directory |

## CLI

```
ard --config <path> [--override <path>] [--image-dir <path>]
```

A single command handles everything:

- `--config` — Path to base config TOML (required)
- `--override` — Path to override config TOML (optional; auto-detects `config.override.toml` alongside `--config` if not provided)
- `--image-dir` — Image directory for multimodal anchors (optional)
  - ⚠️ Both `input_generator` and `target_model` must support multimodal inputs. If either model does not support multimodal, the API will return an error.

## Output

```
outputs/<dataset_name>/
├── anchor_bank.jsonl          # Unified anchor data (graspo-compatible)
├── images/                    # Multimodal images (if any)
└── manifest.json              # Summary statistics
```

### Data Format

All anchors are written to a single `anchor_bank.jsonl` file in a format
compatible with graspo:

```json
{
  "id": "anchor_a1b2c3d4",
  "source": "ard",
  "messages": [
    {"role": "user", "content": "Explain entropy..."},
    {"role": "assistant", "content": "Entropy is a measure of disorder..."},
    {"role": "user", "content": "Can you give an example?"},
    {"role": "assistant", "content": "Sure! Melting ice..."}
  ],
  "targets": [{
    "id": "primary",
    "output": {
      "content": "Sure! Melting ice...",
      "logprobs": {
        "token_ids": [1, 2, 3],
        "log_probs": [-0.1, -0.2, -0.3]
      }
    }
  }],
  "anchor_meta": {"language": "English", "knowledge_domain": "science"},
  "teacher_id": "Qwen3.8-27B"
}
```

## Examples

The `examples/` directory contains sample inputs and outputs to help you
understand the project without running it:

```
examples/
├── images/                    # Sample images for multimodal mode
│   ├── sample_01.jpg
│   └── ...
└── anchor_bank.sample.jsonl   # Sample output (3 anchors)
```

You can browse `examples/` directly on GitHub to see the input/output format.

## Docker

The Docker image is built with `docker/build.sh`, which tags the image with the
current git version. The image uses a two-layer build (dependencies + source)
with BuildKit cache mounts for fast rebuilds.

### Custom image tag

```bash
IMAGE_NAME=ard:latest bash docker/build.sh
ARD_IMAGE=ard:latest bash run.sh --config configs/config.toml --override .local/config.override.toml
```

### Without git

```bash
VERSION=1.0.0 bash docker/build.sh
```

## Development

```bash
# Install dependencies
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Run type checking
uv run mypy src/ard/

# Run linting
uv run ruff check src/ tests/
```

## FAQ

### How do I configure the API key?

Create `.local/config.override.toml` from the sample template and fill in
your API keys:

```bash
mkdir -p .local
cp configs/config.override.sample.toml .local/config.override.toml
```

Then edit the file:

```toml
[input_generator]
api_base = "https://your-api.example.com/v1"
model_name = "your-model-name"
api_key = "sk-..."

[target_model]
api_base = "https://your-api.example.com/v1"
model_name = "your-model-name"
api_key = "sk-..."
```

The override file is deep-merged with `configs/config.toml` at startup.
Only secrets go here; all other configuration stays in `configs/config.toml`.

### How do I add a custom ontology?

Place your ontology JSON file in the `data/` directory (or any path), then
set `path` in `configs/config.toml`:

```toml
[ontology]
path = "data/my_ontology.json"
```

The ontology must follow the expected schema with `knowledge_domains`,
`capabilities`, and `languages` fields.

### What is the output format?

All anchors are written to a single `anchor_bank.jsonl` file in JSONL format
(one JSON object per line). Each record includes `id`, `source`, `messages`,
`targets` (with `content` and token-level `logprobs`), `anchor_meta`, and
`teacher_id`. The format is compatible with graspo's anchor bank format. See
the [Data Format](#data-format) section for the full schema.

### How do I generate multimodal anchors?

Pass the `--image-dir` flag pointing to a directory of images:

```bash
bash run.sh --config configs/config.toml --override .local/config.override.toml \
    --image-dir examples/images
```

The pipeline will sample images from the directory, generate VLM-based
questions about them, and produce multimodal anchors alongside text anchors.

### What models are required for multimodal anchor generation?

Both `input_generator` and `target_model` must support multimodal (vision-language)
inputs. If either model does not support multimodal, the API will return an error
directly — there is no separate config flag to maintain.

### What is the difference between text and multimodal anchors?

Text anchors are conversations (single-turn or multi-turn) generated from
the ontology without images. Multimodal anchors include an image in the
conversation messages and are generated when `--image-dir` is provided.
Both types share the same output format and are written to the same
`anchor_bank.jsonl` file. The number of turns is controlled by `max_turns` in the config.

### How do I resume an interrupted generation?

ARD automatically resumes from the last committed anchor. Just re-run the
same command — the pipeline detects existing anchors in `anchor_bank.jsonl`
and only generates the remaining ones up to `target_count`.

### Where does multimodal anchor diversity come from?

Multimodal anchor diversity comes from three independent sources:

1. **Ontology diversity** (FPS guarantee): The FPS sampler selects optimal
   anchor specs from 10,080 ontology combinations. Metadata like language,
   knowledge domain, capability, and conversation type is passed through the
   VLM prompt, ensuring prompt diversity even with a single image pool.

2. **Image pool diversity**: Images are randomly sampled from the user-specified
   directory. The image content itself provides visual input diversity.

3. **VLM randomness** (temperature=0.8): The Input Generator's sampling
   temperature defaults to 0.8, so the VLM produces different phrasings
   even with the same metadata and image.

### What format are the generated token_ids?

The `token_ids` returned by the vLLM API use integer token IDs
(`logprobs.content[].token_id`) with a fallback to string tokens
(`logprobs.content[].token`). When teacher and student share the same
tokenizer, integer IDs can be used directly for OPD training. With
different tokenizers, string tokens are more portable — downstream
training can re-encode with its own tokenizer.

### How to control the ratio of multimodal to text anchors?

This is currently a **binary switch**: providing `--image-dir` generates
multimodal anchors; omitting it generates pure text anchors. For mixed
ratios (e.g. 30% images + 70% text), run twice and merge manually: once
without `--image-dir` for text anchors, once with `--image-dir` for
multimodal anchors, then combine both `anchor_bank.jsonl` files.

### What is the recommended target count?

At `target_count = 100`, knowledge domain and language coverage can reach
100%, but **capability coverage is ~75-90%** — the weak point (20 capabilities
are hard to fully cover in 100 samples). Recommend `target_count ≥ 200`
for full capability coverage. Total combinations: 18 domains × 20 capabilities
× 4 languages × 7 conversation types = 10,080.

## License

MIT
# ARD — Anchor Replay Distillation

[中文文档](README.zh-CN.md)

> Anchor dataset generation for continual LLM fine-tuning — ontology-driven
> sampling, teacher-model distillation with the reasoning trace kept as a
> separate field, and native Graspo/OPD integration.

**ARD** generates high-quality anchor datasets that prevent catastrophic
forgetting during continual fine-tuning. It samples diverse prompts from
a structured knowledge ontology, calls a teacher model to generate answers
(with the teacher's reasoning trace stored separately from the answer), and
exports in a format directly compatible with On-Policy Distillation (OPD) and
Graspo reinforcement learning pipelines.

**Key features:**
- 🧠 **Ontology-driven sampling** — Five-dimensional ontology (knowledge
  domains, languages, capabilities, conversation types, system prompt modes)
  with embedding-based hierarchical farthest-point sampling ensures broad,
  diverse coverage.
- 🎯 **Teacher distillation with reasoning** — A strong teacher model supplies
  the answer plus its reasoning trace as a separate `reasoning` field, giving
  students two supervision targets instead of one.
- 🖼️ **Multi-modal support** — Text and text+image anchors in a single
  unified pipeline and output format.
- 📦 **Graspo-compatible** — Output format is directly consumable by
  Graspo anchor bank, no conversion needed.
- ⚡ **One command** — Single CLI, TOML-driven config, secrets separated,
  Docker one-click launch.

## Quick Start

### Prerequisites

- Docker
- Embedding data: `ontology/anchor_ontology_embeddings.json` — **tracked in this
  repo**, so a clone already has it (55 pre-computed embeddings, 1024-dim, used
  by the hierarchical FPS sampler). It is a read-only pipeline input, not
  generated output.
- **Optional:** `rawpy` (included in `pyproject.toml` dependencies) for RAW
  image format support (19 formats: CR2, NEF, ARW, DNG, etc.). The library ships as a
  manylinux wheel with bundled `libraw.so` — no system packages required.
  If RAW formats are not needed, rawpy's import is lazy and won't be triggered.

### 1. Build the Docker image

```bash
git clone https://github.com/godthrone/anchor-replay-distillation.git
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
The merged result is a single config dict used throughout the program —
there are never two configuration sources inside the program.

| Section | Purpose |
|---------|---------|
| `[input_generator]` | VLM/LLM that generates user questions |
| `[target_model]` | Target answering model (= teacher) that generates answers (plus the reasoning trace) |
| `[generation]` | Target count, seed, concurrency, max turns, system persona, language and task type filters, backpressure thresholds |
| `[ontology]` | Ontology file path |
| `[output]` | Output directory settings |

## Configuration Reference

All parameters are defined in `configs/config.toml`. Secret fields (`api_base`,
`model_name`, `api_key`) are left empty and filled in `.local/config.override.toml`.

### Model terminology — ARD vs. downstream SFT/OPD

One table, one rule, so the two roles in ARD can never be confused with the
roles downstream frameworks talk about:

| ARD name | Downstream SFT/OPD term | Responsibility |
|----------|-------------------------|----------------|
| `input_generator` (question generator, i.e. the prompting side) | no counterpart — SFT/OPD does not "set questions" | generates the `user` turn (the questioning end) |
| `target_model` (target answering model) | **teacher model** | generates the target answer `targets[0].output` (the teaching signal) |
| — | student model | the learner — **not part of ARD** (owned by graspo/SFT) |

> **One-line rule: whose output is the target answer is the teacher.** In ARD
> the answer has exactly one source — `target_model`; the input generator only
> asks questions and never produces supervision. ARD never runs the student
> model.

### `[input_generator]` — Question Generator

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_base` | string | `""` | API endpoint URL (OpenAI-compatible) |
| `model_name` | string | `""` | Model name |
| `api_key` | string | `""` | API key (secret — fill in override) |
| `temperature` | float | `0.8` | Sampling temperature, higher = more random |
| `max_tokens` | int \| null | `null` (commented out) | Maximum tokens per response. **Not set** in the base config — the line is commented out, so the API provider decides. Uncomment to enforce a limit |
| `connect_timeout` | float | `10.0` | TCP connection + TLS handshake timeout in seconds |
| `first_token_timeout` | float | `300.0` | Maximum wait for first token (prefill + queue), in seconds |
| `inter_token_timeout` | float | `15.0` | Maximum wait between tokens after first, in seconds |
| `retry_on_timeout` | bool | `false` | Whether to retry on timeout errors (requires `max_retries > 0`) |
| `max_retries` | int | `3` | Retries on failure |

> **Note:** the question generator never runs in reasoning mode. It is always
> called with `enable_thinking = false` sent explicitly to the server, and this
> is **not** configurable — `[input_generator]` has no `enable_thinking` field,
> and `[target_model].enable_thinking` affects the teacher model only. The
> generated user turn is stored verbatim as the anchor's question, so reasoning
> tokens would spend `max_tokens` on text that must not end up in the question.

### `[target_model]` — Target Answering Model (= teacher)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_base` | string | `""` | API endpoint URL |
| `model_name` | string | `""` | Teacher model name |
| `api_key` | string | `""` | API key (secret) |
| `temperature` | float | `0.1` | Sampling temperature for the teacher's answer. Default 0.1 (R12 ruling, was 0.0): near-greedy but lets the teacher vary phrasing between anchors — enough for answer diversity without drifting into inconsistent answers |
| `max_tokens` | int \| null | `null` (commented out) | Maximum tokens per response. **Not set** in the base config — the line is commented out, so the API provider decides. Uncomment to enforce a limit |
| `connect_timeout` | float | `10.0` | TCP connection + TLS handshake timeout in seconds |
| `first_token_timeout` | float | `300.0` | Maximum wait for first token (prefill + queue), in seconds |
| `inter_token_timeout` | float | `15.0` | Maximum wait between tokens after first, in seconds |
| `retry_on_timeout` | bool | `false` | Whether to retry on timeout errors (requires `max_retries > 0`) |
| `max_retries` | int | `3` | Retries on failure |
| `enable_thinking` | bool | `false` | Enable reasoning mode (Qwen3, DeepSeek-R1, etc.). When enabled, the model's reasoning is streamed into the separate `reasoning` field of `targets[0].output` and is **never** merged into `content`; when disabled, `reasoning` is `null`. **Only enable when distilling to a reasoning model**. **This value is always sent to the server** — `false` explicitly disables reasoning, `true` explicitly enables it |

> **⚠️ Behavior change (v0.3+):** Prior versions did **not** send `enable_thinking` to the server
> when set to `false` — the server's chat template treated "undefined" as "reasoning ON",
> so `false` was silently ignored. As of this version, `enable_thinking` is **always** sent
> explicitly. If you relied on the old behavior (reasoning *on* with `enable_thinking = false`),
> set it to `true`.

### `[generation]` — Generation Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_count` | int | `100` | Target number of anchors. The capability dimension is the coverage weak point: a single run covers roughly 30% of the capabilities, and it grows slowly with `target_count` (see the [FAQ](#what-is-the-recommended-target-count)) |
| `seed` | int | *unset → random* | Sampling seed. Unset = a fresh seed each run; set an integer to pin that run's sampling order |
| `concurrency` | int | `4` | Concurrent requests; too high may trigger rate limits |
| `languages` | list | `[]` | Language filter (empty = all). Options: `zh-CN`, `en`, `ja`, `ko` |
| `task_types` | list | `[]` | Task type filter (empty = all) |
| `max_turns` | int | `1` | Max conversation turns (1 = single-turn, 2-10 = multi-turn) |
| — | — | — | The system prompt is no longer a config switch: the ontology samples it (`system_prompt_presence` / `system_prompt_style`), and each anchor that has one gets its text generated at run time and stored in `messages[0]` |
| `max_turns_with_image` | int | `1` | Max turns with image (≤ `max_turns`) |
| `embeddings_path` | string | `"ontology/anchor_ontology_embeddings.json"` | Pre-computed ontology embedding file |
| `backpressure_threshold` | int | `3` | Consecutive server-side failures that trigger a cooldown |
| `backpressure_cooldown` | float | `60.0` | Cooldown pause in seconds once the threshold is reached |

### `[ontology]` — Ontology

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | string | `"ontology/anchor_ontology.json"` | Ontology JSON file path |

### `[output]` — Output

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `directory` | string | `""` | Output directory (empty = auto-generated timestamped dir) |
| `overwrite` | bool | `false` | Overwrite existing output directory |

### Setting a field to "empty" / using API defaults

TOML does not have a `null` value. To let the API provider decide a value
(e.g. `max_tokens`), **comment out or delete the line** in `config.toml`:

```toml
[input_generator]
# max_tokens = 4096   ← commented out → API uses its own default
```

The pydantic config model uses `None` as the default for optional fields.
When a field is `None`, it is omitted from the API request entirely.

## CLI

```
ard --config <path> [--override <path>] [--image-dir <path>] [--no-convert]
```

A single command handles everything:

- `--config` — Path to base config TOML (required)
- `--override` — Path to override config TOML (optional; auto-detects `config.override.toml` alongside `--config` if not provided)
- `--image-dir` — Image directory for multimodal anchors (optional)
  - ⚠️ Both `input_generator` and `target_model` must support multimodal inputs. If either model does not support multimodal, the API will return an error.
- `--no-convert` — Disable automatic image format conversion (optional)
  - By default, the pipeline auto-converts all images to JPG/PNG (PNG → PNG copy, RAW/BMP/TIFF/GIF/WebP → JPG quality=95).
  - With this flag only PNG/JPEG/GIF/WebP are accepted; **files in any other format are silently ignored by the directory scan** — they are neither converted nor reported as errors. If that leaves no usable images, the pipeline logs a WARNING (`No images found in <dir>. All anchors will be pure text.`) and still produces **text-only anchors**. Check the startup log line if you expect multimodal output.

## Output

```
outputs/<dataset_name>/
├── anchor_bank.jsonl          # Unified anchor data (graspo-compatible)
├── config.json                # Merged-config snapshot (reproducibility)
├── images/                    # Multimodal images (if any)
├── logs/
│   ├── ard.log                # Human-readable pipeline log (INFO+)
│   ├── ard_debug.log          # Machine-parseable debug log (DEBUG+)
│   └── ard_error.log          # Error log (ERROR+)
└── manifest.json              # Summary statistics + generation health counters
```

### `manifest.json`

Besides the anchor-bank summary (`total_anchors` / `domains` / `languages` /
`capabilities` / `output_dir`), the manifest reports how healthy the run was:

```json
"generation": {
  "counters": {
    "requested": 120,
    "succeeded": 101,
    "abandoned_total": 19,
    "abandoned_by_reason": {"empty_content": 4, "timeout": 12},
    "written": 100,
    "rejected_invalid_shape": 0,
    "duplicate_ids": 1,
    "backpressure_events": 2
  },
  "failures": {
    "reasoning_only_responses": 4,
    "truncated_empty": 4
  }
}
```

* `counters` — where every requested anchor ended up: produced, written, dropped
  (with its machine-readable reason), rejected by the message-shape gate,
  de-duplicated by id, and how many backpressure cooldowns were triggered.
  `abandoned_by_reason` is keyed by the tag the code actually records — e.g.
  `timeout` / `transport_error` for server instability, `empty_content`,
  `answer_too_short` / `answer_too_long`, `invalid_shape`, `role_mismatch`,
  `unknown_role`, `history_not_advanced`, `no_final_turn` for model-output
  failures.
* `failures` — process-level counters over the *responses* the teacher model
  returned: `responses` / `reasoning_responses` / `reasoning_chars` /
  `reasoning_only_responses` / `empty_content` / `truncated_empty`. They reveal
  "thinking ate the token budget" (`truncated_empty` = empty answer **and**
  `finish_reason == "length"`; `empty_content` is a superset of
  `reasoning_only_responses`).

Both sub-objects are written **only when non-empty**, and zero-valued entries are
dropped, so a healthy run gains no noise and an unhealthy one cannot look healthy.

The `generation` sub-objects are keyed by the counters the code actually
records; the examples above are drawn from that set (`generation.counters`
fields per `AnchorGenerationStats.to_manifest_dict`, `generation.failures`
fields per `reasoning_stats()`).

**Backward compatibility:** `generation` is optional. Manifests written before it
existed stay readable — treat a missing field as "no generation statistics were
recorded for that run", not as all-zero (test with `manifest.get("generation") is None`).
The six pre-existing fields (`total_anchors` / `domains` / `languages` /
`capabilities` / `output_dir` / `config`, the last being the merged-config
snapshot) keep their meaning and shape. A bank written by v3 also carries
`system_prompt_modes` and `data_sources` at the top level.

### Data Format

All anchors are written to a single `anchor_bank.jsonl` file in a format
compatible with graspo. Multi-turn anchors always start **and** end with a
`user` message with strictly alternating roles, so the message shape is either
`U` or `UAU` (this is enforced before writing — violating anchors are rejected
and counted under `rejected_invalid_shape`). See
`examples/anchor_bank.sample.jsonl` for a complete real record.

**One run, one `data_source`.** Every record in a bank carries the same
`data_source`: `ard_text` when `--image-dir` is omitted, `ard_multi` when it is
supplied (a directory with no usable images degrades the whole run to text).
Text and multimodal anchors are never mixed in one bank — run the two modes
separately and you get one `anchor_bank.jsonl` each. Anchor ids are hashed from
the five-dimensional metadata, so they are unique **within a run** but not
across runs: when merging banks, do not expect an id to identify a record
globally.

`seed` is optional and unset by default — each run then draws a fresh seed from
the system random source. Set it to an integer to pin the sampling order for that
run.

```json
{
  "id": "anchor_a1b2c3d4",
  "source": "ard",
  "data_source": "ard_text",
  "schema_version": "3.0.0",
  "messages": [
    {"role": "system", "content": "You are a precise science tutor."},
    {"role": "user", "content": "Explain entropy..."},
    {"role": "assistant", "content": "Entropy is a measure of disorder..."},
    {"role": "user", "content": "Can you give an example?"}
  ],
  "targets": [{
    "id": "primary",
    "output": {
      "content": "Sure! Melting ice...",
      "reasoning": "The user asks for a concrete example, so pick one phase change..."
    }
  }],
  "anchor_meta": {"language": "English", "knowledge_domain": "science_exploration"},
  "input_generator_model": "your-question-model",
  "teacher_id": "your-model-name"
}
```

The leading `system` message is optional (at most one, only at position 0) and is
present only when the ontology sampled a system prompt for that anchor — the
`messages` array is the single source of truth for it, there is no separate
top-level field. `reasoning` is `null` (never `""`) when the teacher did not
think, i.e. when `enable_thinking = false`.

There is **no** `logprobs` / `token_ids` / `log_probs` key anywhere in a v3
record: ARD does not request, collect or persist token-level log-probabilities
(see `docs/architecture.md` §5 and §9.3 for why). The teacher's logprob belongs
to OPD training and is produced there by the original LLM teacher at train time.

Multimodal anchors carry the image inside the `user` message `content`, which
becomes a list of parts instead of a plain string:

```json
{"role": "user", "content": [
  {"type": "image", "image": "images/sample_01.jpg"},
  {"type": "text", "text": "What natural formation is shown in this image?"}
]}
```

## Examples

The `examples/` directory contains real inputs and outputs so you can
understand the project without running it. Everything here was produced by a
real run of the pipeline — nothing is hand-written:

```
examples/
├── README.md                    # What's in here and how to read it
├── images/                      # Sample images for multimodal mode
│   ├── sample_01.jpg
│   └── ...                      # 10 small JPEGs (400×267), safe for smoke runs
├── anchor_bank.sample.jsonl     # Sample output: 6 real anchors
│                                #   (3 single-turn `U` + 3 three-turn `UAU`,
│                                #    4 languages, 1–2 images each)
└── manifest.sample.json         # The run manifest a real run writes alongside it
```

Start with `examples/README.md` — it explains the record schema field by field
and shows why `messages` always starts and ends with a `user` turn.

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

When git metadata is unavailable, `build.sh` falls back to version `1.0.0`, so the
built image is `ard:1.0.0`. `VERSION` only overrides the *package* version recorded
**inside** the image — it does **not** rename the image; use `IMAGE_NAME` for that.

```bash
VERSION=1.0.0 IMAGE_NAME=ard:latest bash docker/build.sh
```

The script prints the resulting image name on its last line (`Built: <image>`).

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
api_key = "REPLACE_WITH_YOUR_API_KEY"

[target_model]
api_base = "https://your-api.example.com/v1"
model_name = "your-model-name"
api_key = "REPLACE_WITH_YOUR_API_KEY"
```

The override file is deep-merged with `configs/config.toml` at startup.
Only secrets go here; all other configuration stays in `configs/config.toml`.

### How do I add a custom ontology?

Place your ontology JSON file in the `ontology/` directory (or any path), then
set `path` in `configs/config.toml`:

```toml
[ontology]
path = "ontology/my_ontology.json"
```

The ontology must follow the expected schema with `knowledge_domains`,
`capabilities`, and `languages` fields.

### What is the output format?

All anchors are written to a single `anchor_bank.jsonl` file in JSONL format
(one JSON object per line). Each record includes `id`, `source`, `data_source`,
`schema_version`, `messages`, `targets` (whose `output` holds `content` and
`reasoning`), `anchor_meta`, `input_generator_model`, and `teacher_id`. The
format is compatible with graspo's anchor bank format. See
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
Both types use the same record format, but one run produces only one of them
(see **Data Format**): each mode has its own run and its own
`anchor_bank.jsonl`. The number of turns is controlled by `max_turns` in the config.

### How do I resume an interrupted generation?

ARD automatically resumes from the last committed anchor. Just re-run the
same command — the pipeline detects existing anchors in `anchor_bank.jsonl`
and only generates the remaining ones up to `target_count`.

### Where does multimodal anchor diversity come from?

Multimodal anchor diversity comes from three independent sources:

1. **Ontology diversity** (FPS guarantee): The FPS sampler selects optimal
   anchor specs from 50,400 ontology combinations. The metadata — language,
   knowledge domain, capability, conversation type and system prompt mode —
   shapes which anchors are sampled, and language, capability and conversation
   type are written into the VLM prompt, so they promote prompt diversity: even
   with a single image pool, anchors with a different language, capability or
   conversation type hand the VLM a different instruction. Note this is
   promotion, not a guarantee: two anchors with the same language, capability
   and conversation type hand the VLM exactly the same instruction text (which
   is why source 3 below matters), and the system prompt's own text takes no
   part in question generation — it is written separately at run time and
   stored in `messages[0]`.

2. **Image pool diversity**: Images are randomly sampled from the user-specified
   directory. The image content itself provides visual input diversity.

3. **VLM randomness** (`[input_generator].temperature`, default `0.8`):
   The Input Generator samples at its **configured** temperature — the value
   in `configs/config.toml` is what is actually sent to the API — so the VLM
   produces different phrasings even with the same metadata and image.

### What format are the generated token_ids?

There are none: **v3 records carry no `token_ids` field.** ARD does not request
`logprobs`/`top_logprobs` from the API and does not persist token-level data, so
`targets[0].output` holds only `content` (the answer) and `reasoning` (the
teacher's thinking trace, `null` when thinking is off).

Historically (v2, before the logprob chain was removed) the collected token list
preferred **string** tokens (`logprobs.content[].token`) and fell back to
integer IDs (`logprobs.content[].token_id`) when the string form was absent —
string tokens are portable across tokenizers, so downstream training could
re-encode with its own tokenizer. That machinery, and the field itself, is gone
in v3 (see `docs/architecture.md` §5.2); if you need token-level supervision,
it is produced at OPD training time by the original LLM teacher.

### How to control the ratio of multimodal to text anchors?

This is currently a **binary switch**: providing `--image-dir` generates
multimodal anchors; omitting it generates pure text anchors. For mixed
ratios (e.g. 30% images + 70% text), run twice and merge manually: once
without `--image-dir` for text anchors, once with `--image-dir` for
multimodal anchors, then combine both `anchor_bank.jsonl` files.

### What is the recommended target count?

**Capability is the coverage weak point.** Measured on a single run
(`k = 100`, `seed = 42`), the five dimensions cover:

| Dimension | Total | Covered at `k = 100` | Note |
|-----------|-------|----------------------|------|
| Knowledge domain | 18 | **18 (100%)** | **Structural**: the layer-1 FPS quota is per domain, so 18/18 is guaranteed by construction, not an achievement of the sampler |
| System prompt | 5 | **5 (100%)** | Also structural: every sampled anchor carries one of the 5 modes |
| Conversation type | 7 | 6 | |
| Language | 4 | 3 | |
| **Capability** | 20 | **6 (30%)** | The weak point; the distribution is very uneven — the hottest capability (`translation`) gets 37 anchors while `uncertainty_handling` gets 1, a **37:1** ratio |

Capability coverage rises only slowly with `target_count`, and **monotonicity is
not guaranteed** — the per-domain quota is an integer division
(`max(1, target_count // 18)`), so the sampling chain changes with `k`. Measured:

| `target_count` | 200 | 600 | 2800 | 5040 |
|----------------|-----|-----|------|------|
| Capabilities covered | 7/20 | 10/20 | 15/20 | 18/20 |

> These are **single-seed measurements** and therefore fluctuate with the seed.
> There is no `target_count` short of the full combination space at which
> capability coverage is known to complete — raising `target_count` helps slowly
> and without a monotonicity guarantee.

Total combination space: 18 knowledge domains × 20 capabilities × 4 languages ×
7 conversation types × 5 system prompt modes = **50,400**; counting both output
shapes (text and multimodal) = **100,800**. If your training needs full
capability coverage, raise `target_count` substantially or add a
per-dimension quota — the current sampler does not provide one.

## License

MIT
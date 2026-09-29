# anchor-replay-distillation

[English](README.md) | [简体中文](README.zh-CN.md)

## Introduction

**Anchor Replay Distillation (ARD) turns an ontology-defined coordinate space into a
reproducible supervised-finetuning corpus.** For every anchor coordinate it asks a question
generator for the user turn and a target "teacher" model for the answer, then keeps the
teacher's answer — and, when enabled, its reasoning — beside the question that produced it.
ARD is the *data-generation* half of a distillation pipeline: it does not train, distil or
evaluate a student model.

The plan is built from the v4 ontology (`ontology/anchor_ontology.v4.json`) by one rule —
**cycle-shuffle** — and the unit count `U` is enumerated from the ontology at run time, never
hand-copied into the ontology or the code. The rule itself, and the command that recounts `U`,
are in [docs/algorithm.md](docs/algorithm.md) §1–§2.

**How many anchors a run produces is yours to decide**, in exactly one place — the
`[generation] count` field of `configs/config.toml`. Leaving it unset means one full round
(`U`), and there is **no upper bound**: a larger `N` simply rolls into round 1, 2, … See
[Choosing how many anchors](#choosing-how-many-anchors-n).

Want the plain-language picture first? Read [docs/walkthrough.md](docs/walkthrough.md) — what the
machine does after you press Enter, step by step.

The words this document leans on:

| Term | Meaning |
|---|---|
| anchor | one generated example — the question and the teacher's answer for one coordinate; one line of `anchor_bank.jsonl` |
| ontology | `ontology/anchor_ontology.v4.json`: the axes, their values and which combinations are legal |
| coordinate | the value tuple that labels an anchor: one value per axis, plus the sample field `modality` |
| unit / round | one `(modality, legal restricted block)` pair; a round walks every unit exactly once |
| plan | the ordered list of coordinates a run generates, fixed before the first endpoint call |
| N | how many anchors a run produces: the `[generation] count` field; unset = one round = `U` |
| manifest | `manifest.json`, the run's authoritative declaration of what it built |

## Quick Start

Docker is the default way to run ARD. A fresh clone reaches a real artifact in **two commands**.
There is **no published ARD image**: the first `./run.sh` builds `ard:<latest git tag>` from
`docker/Dockerfile` on your machine (once, a few minutes). The build needs access to PyPI and to
the `python:3.11.15-slim` base image — Docker fetches that base image from its registry unless
your machine already has it cached, so a restricted or offline machine must provide it locally.

### 1. Provide endpoint credentials

```bash
mkdir -p .local && cp configs/config.override.sample.toml .local/config.override.toml
```

Then fill in `api_base`, `model_name` and `api_key` for `[input_generator]` and
`[target_model]`.

The `mkdir -p` is not decoration: `.local/` is gitignored, so a fresh clone does not have it and
`cp` would fail with `No such file or directory`. If `.local/config.override.toml` **already
exists, edit it instead of copying over it** — the copy would silently overwrite credentials that
are already deployed there.

The override is merged over `configs/config.toml` by a fixed three-tier priority, first hit wins:

1. an explicit `--override PATH` (a missing file is an error);
2. `<project_root>/.local/config.override.toml` — the recommended, gitignored location;
3. `<directory of --config>/config.override.toml` — backwards compatibility.

The choice is never silent: every run logs at INFO which override file it used, or that it used
the base configuration only. Field names are validated with `extra="forbid"`, so a typo is an
error rather than a silently ignored setting.

With no credentials the run stops **before creating any output directory**, with a field-level
message and exit code 1. Its first line (the message continues with a fix example) is:

```
ERROR: [input_generator] is missing `api_base`, `model_name`.
```

### 2. Run the smoke test

```bash
./run.sh --config configs/config.toml --smoke --image-dir examples/images
```

The first call builds the image `ard:<git describe --tags --abbrev=0>` (the current tag; override
the name with `ARD_IMAGE=<name>`), then runs `python -m ard` inside it as
`docker run --rm --network=host`. `configs/`, `ontology/`, `examples/` and `.local/` are mounted
read-only, `outputs/` read-write, so the artifact lands in your checkout. `--network=host` is what
lets the container reach an endpoint running on the host's `localhost`; on Docker Desktop, host
networking must be enabled in Docker's settings.

Want to build the image alone first, and watch the build log?

```bash
bash docker/build.sh
```

It builds the same tag (`IMAGE_NAME=<name>` overrides it) and stops there. If Docker is missing,
`run.sh` exits 127 with a message; if the build fails, it prints the rebuild command.

**What `--smoke` means.** It materialises the *same* cycle-shuffle rule at a reduced scale:
**8 anchors — 4 text-only + 4 image-capable**, taken from round 0's shuffle order (the first four
units of each modality). Its point is that a fresh checkout can see a complete artifact in
minutes, and it is a real endpoint run, not an offline demo. The artifact is deliberately small
and says so in three independent places: the run directory name gets the `_smoke` suffix, the log
carries a WARNING, and `manifest.json` sets `smoke: true` plus a `smoke_plan` block. `--smoke` is
a CLI run-boundary flag, not a configuration field, and changes nothing about a run without it.

A smoke plan is **not a prefix of the full plan**: the full plan takes the first `N` units of the
single round-0 order, while smoke takes the first four units *of each modality* — so a shared id
can point at a different coordinate. That is why a smoke directory and a full directory must not
be mixed.

The artifact lands in `outputs/<run_name>/` (default `ard_dataset_<YYYYmmdd_HHMMSS>_smoke`; set
`[output] directory` in the config to pin the name). After a few minutes you should have:

- `anchor_bank.jsonl` — the 8 anchors, `schema_version` `5.0.0`;
- `results/coverage.json` + `results/coverage.md` — the zero-model structure readout
  (`report_schema "ard-acceptance-4"`): plan counts against the construction rule, with
  `within_rule` and the diversity counts. No model call, no endpoint, always written. The report
  never pretends 8 anchors are a full-round delivery;
- `manifest.json` — composition, `plan_identity` (v2), the `plan` and `images` sections;
- `config.toml` and `logs/`. While the run is still going there is also
  `plan_identity.in_progress.json`, the intermediate record that binds the directory to its plan;
  a finished run removes it and keeps only `manifest.json` as its declaration.

**Full run.** One full round is `U` anchors (the runtime-enumerated unit count; set
`[generation] count` for any other `N`), one endpoint call per conversational turn. Supply your own image
directory laid out as `<image_dir>/<visual_domain>/<image file>`:

```bash
./run.sh --config configs/config.toml --image-dir /path/to/images
```

A directory outside the repository is mounted read-only at `/data/images` in the container; one
inside the repository (like `examples/images`) is simply used through its relative path. Pictures
rotate, and a domain whose own directory holds no image **reuses** one from the whole image tree
instead of losing its anchors — so a single picture covers every domain and every round (each
reuse is counted and named in `manifest.json`). Only a tree with no usable image at all stops the
run. A full run costs hours of endpoint time; run the smoke test first to validate the setup.

### Without Docker (optional)

The same entry point runs from a local environment. You need [uv](https://docs.astral.sh/uv/) and
Python 3.11 (pinned in `.python-version`; `uv.lock` pins every dependency):

```bash
uv sync
uv run python -m ard --config configs/config.toml --smoke --image-dir examples/images
```

**RAW support is not installed by default — and the shipped image does not add it.** The image
and a plain `uv sync` both install the default closure only, which contains no `rawpy`; RAW camera
formats (`.cr2`, `.nef`, `.arw`, `.dng`, …) therefore need the optional `raw` extra in a local
install:

```bash
uv sync --extra raw
```

Without it a RAW input is reported and skipped (a WARNING naming the file) rather than crashing
the run. Why RAW is opt-in, and the dependency licences in full, are in
[`CONTRIBUTING.md`](CONTRIBUTING.md#third-party-licences).

## Choosing how many anchors (N)

`N` has exactly one source — the `[generation] count` field in `configs/config.toml`:

```toml
[generation]
# Omit (the default) = one full round = U, the runtime-enumerated unit count.
# Set any integer >= 1 to produce exactly that many anchors; N rolls on across
# rounds (finishing a round starts the next one).
count = 5000
```

- **No CLI flag, no `run.sh` passthrough, no environment variable.** `./run.sh --count 10` hands
  `--count 10` straight to `python -m ard`, which rejects it as an unrecognised argument. A CLI
  flag and a config field for the same value would be two sources of truth, so the value lives
  only in the config.
- **No upper bound.** A large `N` is not refused; the run logs how many full rounds it spans and
  how many anchors fall in the last, partial round.
- **What is refused**: `count < 1`, and non-integer types (`0`, negatives, `true`, `"1826"`,
  `1.5`). The error names the field and tells you to remove the line to get one full round.
- **Raising `N` on an existing directory appends.** Anchor ids are plan-position serial numbers
  that do not contain `N`, so running the same `output.directory` with a larger `count` keeps every
  existing record and appends the new ones.
- **Coverage and density are read separately.** *Coverage* counts coordinates and saturates at
  `1.0` once every unit has been visited (`min(distinct, U) / U`); *density* counts entries and
  keeps growing with `N` (`N / U`). Revisiting a coordinate in a later round is a legitimate new
  sample — the generator is stochastic, so the same labels yield a different question. Definitions:
  [docs/algorithm.md](docs/algorithm.md) §7.

## Data Format

`anchor_bank.jsonl` holds one JSON object per line:

```jsonc
{
  "id": "1a2b3c4d-c00000p00137",    // plan-position serial number: run_key + round + position
  "source": "ard",
  "data_source": "ard_text",        // controlled vocabulary: ard_text | ard_multi
  "schema_version": "5.0.0",
  "messages": [ /* the conversation */ ],
  "targets": [
    { "id": "primary",
      "output": { "content": "…the teacher's answer…",
                  "reasoning": "…the teacher's reasoning, or null…" } }
  ],
  "anchor_meta": { /* the coordinate, one key per axis */ },
  "input_generator_model": "…",
  "teacher_id": "…"                 // the model whose output is the target
}
```

`id` is a **plan position**, not a coordinate fingerprint: `run_key` is
`H(ontology hash, seed, sampling algorithm)[:8]`, `c00000` is the round and `p00137` the position
inside it. It deliberately does **not** contain `N`, which is what makes "raise `count` and re-run
the same directory" a clean append. Two records may therefore carry the same `anchor_meta` with
different ids; both are kept.

`anchor_meta` carries the sample field `modality` and every axis value: `language`,
`knowledge_domain`, `capability`, `system_prompt_mode`, `conversation_type`,
`response_style`, `output_format`, `difficulty`, `context_length`, `input_condition`,
`answer_mode` — plus `visual_domain` for image-modality anchors. A text-modality coordinate
omits `visual_domain` entirely (no `null` placeholder) and carries `has_image: false` /
`image_count: 0` instead; an image anchor carries `visual_domain` with `has_image: true` /
`image_count: 1`. `data_source` is per record: `ard_multi` marks a conversation that carries at
least one image, and downstream routing uses that key.

`reasoning` is `str | null` and is `null` — never `""` — when the teacher ran without
thinking (`target_model.enable_thinking = false`). An anchor whose configured reasoning
cannot be obtained is discarded and counted in `manifest.json`, not written with an empty
value.

Images are referenced by **relative path** inside a message's content list, never inlined as
base64:

```jsonc
{ "role": "user",
  "content": [ { "type": "image", "image": "images/<visual_domain>/sample_XX.jpg" },
               { "type": "text",  "text": "…the generated user question…" } ] }
```

The picture comes from `<image_dir>/<visual_domain>/<file>` — a picture filed under the anchor's
own `visual_domain` coordinate is the one whose content the user vouched for, so it is always
preferred. When that directory holds no image the anchor is **not dropped**: the run reuses a
picture from the whole image tree (every usable file under `--image-dir`, in a deterministic
order) and rotates through it. Each reuse is recorded — `fallback_visual_domains` /
`fallback_anchor_count` in `manifest.json`, plus a `fallback` flag on each `resolved_images` row.
Paths are relative to the run directory, so `outputs/<run_name>/images/<visual_domain>/…`
resolves directly; the run copies and (unless `[images] convert = false`) converts the selected
image there. A small, real example of these records is checked in under `examples/` — see
[examples/README.md](examples/README.md).

## Configuration reference

Every field lives in `configs/config.toml` (English comments, all fields defined) and the
override supplies values only. **All artifacts are decided by the config** — the CLI passes
input locations and run boundaries only: `--config` (required), `--override`, `--image-dir`,
`--smoke`.

| Section | What it controls |
|---|---|
| `[input_generator]` | The question-generator endpoint (`api_base`, `model_name`, `api_key`), sampling temperature, timeouts, retries |
| `[target_model]` | The teacher endpoint — its answer is the supervision target. `enable_thinking` switches the reasoning channel on (`targets[0].output.reasoning`) |
| `[generation]` | `count` (the anchor count `N`: omit = one full round = `U`; any integer `>= 1`, **no upper bound**), `concurrency`, the optional `seed` (omit it and every run draws a fresh seed; the seed actually used is recorded in the run's `config.toml`), backpressure thresholds. Turn counts are **not** configurable — they come from the ontology |
| `[ontology]` | Path to the v4 ontology (`ontology/anchor_ontology.v4.json`) |
| `[output]` | `directory` (empty = `outputs/ard_dataset_<timestamp>`), `overwrite` (default `false`: an existing bank is resumed, not replaced) |
| `[images]` | `convert` (default **`true`**): accept RAW/BMP/TIFF/GIF/WebP and normalise every selected picture to JPEG (RAW additionally needs the optional `raw` extra); `false` accepts only the already-web formats and copies them verbatim. `skip_missing_images` (default **`false`**): a domain whose own directory has no picture reuses one from the whole image tree; if the whole tree has no usable picture at all, the run is refused. Set it to `true` to drop those anchors instead — each drop is a WARNING, and the count and the affected domains are declared in `manifest.json` |
| `[coverage]` | `enabled` (default `true`): publish the zero-model structure readout to `results/`. Set it to `false` to skip the phase entirely |

`configs/config.override.sample.toml` is the commented template step 1 copies; it is where
deployment values (endpoints, model names, credentials) go, never `configs/config.toml` itself.

### Image transcoding is a config field

`[images] convert` decides what lands in `outputs/<run>/images`: `true` normalises every selected
picture to JPEG, `false` copies the already-web formats verbatim. There is no CLI flag for that
decision — one config must describe one artifact, so a run archived as `config.toml` reproduces
the image bytes as well as the anchors.

The archive is itself a valid `--config`: `python -m ard --config outputs/<run>/config.toml
--override config.override.toml` re-runs the same configuration with your credentials supplied
by the override.

## Output

```text
outputs/<run_name>/          # default ard_dataset_<YYYYmmdd_HHMMSS>; --smoke appends _smoke
├── anchor_bank.jsonl        # one record per line, schema_version 5.0.0
├── config.toml              # merged config snapshot, credentials redacted
├── plan_identity.in_progress.json  # only while a run is unfinished: its plan-identity record
├── images/                  # image-modality anchors' pictures: images/<visual_domain>/<file> (transcoded or copied, one placed copy per source file); created only when this run has image anchors
├── logs/                    # ard.log / ard_debug.log / ard_error.log
├── results/
│   ├── coverage.json        # machine-readable structure readout (report_schema ard-acceptance-4)
│   └── coverage.md          # the same readout, human-readable
└── manifest.json            # composition, run health, config, plan_identity v2, plan, images, acceptance pointer (the authoritative one)
```

`manifest.json` is the **authoritative** declaration of a finished run: its `status`, the bank's
composition, the run-health counters, and the plan's identity and shape (the sampling rule, the
ontology hash, `seed` and `count`, `U`, cycles, planned vs. written anchors, coverage, density).
While a run is unfinished, `plan_identity.in_progress.json` binds the directory to its plan
instead; the run removes it once the manifest is written, so a finished directory holds exactly
one declaration.

Where to read what: the records actually on disk are in `anchor_bank.jsonl`; the declaration is
the finished `manifest.json`; a `--smoke` artifact declares itself in its directory name, its log
and `manifest.smoke`. Field reference, counter semantics and the resume guard:
[docs/architecture.md](docs/architecture.md) §4.

`results/coverage.{json,md}` is the **structure readout**, not training data
(`report_schema "ard-acceptance-4"`): plan counts against this run's own sampling space and the
records actually on disk — `coverage`, `density`, the round decomposition, `within_rule` and the
two diversity counts. It needs **no model call and no endpoint** and is produced whenever
`[coverage] enabled = true`. When the bank is missing a planned coordinate the readout says so —
`within_rule: false` plus the missing coordinates named in `warnings` — never a silently clean
report. Definitions: [docs/algorithm.md](docs/algorithm.md) §5–§6.

## Adding a knowledge domain

**Adding one `knowledge_domain` leaf means adding it to `knowledge_domain_tree` in
`ontology/anchor_ontology.v4.json` — nothing else.** There are no hand-written counts to update
and no constant to bump: the coverage units, the leaf rotations and the acceptance expectations
are all enumerated from the ontology at runtime, so a new leaf is picked up on the next run
without touching `configs/`, `src/` or the code. (Adding a leaf changes the ontology hash, so it
is a *different plan* — a directory produced before the edit cannot be resumed; use a new
`output.directory`.)

## Development

```bash
uv sync --extra dev     # test, type-check and lint tools
uv run pytest
uv run ruff check src/ tests/
uv run mypy src/ard/
```

Optional extras are declared in `pyproject.toml`: `dev` (the tools above) and `raw` (RAW camera
decoding). The package version is derived from git tags by
`setuptools_scm`; a source archive that carries no `.git` (GitHub "Download ZIP", `git archive`)
falls back to `1.0.0`, the same value `run.sh` and `docker/build.sh` use when there is no tag to
describe.

The same workflow is described in [CONTRIBUTING.md](CONTRIBUTING.md).

**`--smoke` vs. a full run:** the construction rule, the sampling and the per-anchor
generation path are identical; only the plan scale differs (8 anchors vs. one full round). A smoke
artifact is deliberately incomplete and must not be delivered as a dataset — the `_smoke`
suffix, the log WARNING and `manifest.json: smoke` exist so it cannot be mistaken for one.
Both paths call the configured endpoints; neither is an offline demo.

## FAQ

**I set N somewhere else and nothing happened.**
`N` has exactly one source: the `[generation] count` field in the config. There is no
`--count` flag, no `run.sh` option and no environment variable — `./run.sh --count 10` is passed
to `python -m ard`, which rejects it as an unrecognised argument. See
[Choosing how many anchors](#choosing-how-many-anchors-n).

**Re-running refuses my output directory.**
The resume guard compares each existing record against the coordinate its plan position now
points at, so a run whose plan changed — a different `seed`, an edited ontology, or switching
between `--smoke` and a full run — is refused before anything is written. Raise `count` on the
same directory and the new anchors are appended; otherwise use a new `output.directory`, or set
`[output] overwrite = true` to clear the existing bank and start fresh.

**An endpoint is missing or unreachable.**
A missing `api_base` / `model_name` is refused at config load, with the field names in the message
and before any output directory is created. Failing endpoint calls are retried (`max_retries`,
`retry_on_timeout`), and past the backpressure threshold the run pauses for the configured
cooldown.

**An image domain is missing — what happens?**
Images are reused rather than skipped. A picture is always taken from
`<image_dir>/<visual_domain>/` when that directory holds one; when it does not, the run falls
back to the whole image tree and rotates through it, so **if you supplied one picture, every
domain and every round uses that picture** — the run is complete, and whether the repetition
hurts the dataset is your call. Each reuse is visible: `manifest.json.images` carries
`fallback_visual_domains`, `fallback_anchor_count`, `pool_candidate_count` and a per-row
`fallback` flag, and every reuse is logged.

Only a tree with **no usable image at all** is refused, **before the output directory is
created**, with an error that lists every domain left without a picture. Either supply at
least one image, or set `[images] skip_missing_images = true`: those anchors are then dropped,
each with its own WARNING, and the skipped count and domains are declared in `manifest.json` — a
skip is never silent. A run that omits `--image-dir` altogether attaches no pictures at all:
image-modality anchors are still generated, as text conversations, while keeping their
`visual_domain` coordinate.

**Why do RAW camera files need the `raw` extra — and does Docker have it?**
RAW decoding goes through `rawpy`, whose wheels bundle the LibRaw decoder (LGPL-2.1 /
CDDL-1.0), so it is an opt-in extra rather than part of the default set: `uv sync --extra raw`
(see [Without Docker](#without-docker-optional)). The shipped Docker image installs the default
closure only and therefore contains no `rawpy` either — under Docker a RAW input is skipped with
a WARNING naming the file, and the rest of the run proceeds. The default closure is not uniformly
MIT/Apache: `certifi` is MPL-2.0, `tqdm` is `MPL-2.0 AND MIT`, `typing-extensions` is PSF-2.0,
and the rest are MIT / BSD-3-Clause / MIT-CMU. [`CONTRIBUTING.md`](CONTRIBUTING.md#third-party-licences) carries the
full list and the command that reads it out of an installed environment; this repository states
those facts only and leaves any compliance judgement to you.

**What does `MULTI_TURN_DEFAULT = 4` mean?**
The ontology declares `turns: "multi"` for two `conversation_type` values without a numeric upper
bound; the implementation reads "multi" as the largest turn count the ontology itself declares
(`constraint_update` = 4) and errors out if the ontology ever declares more — it never silently
clamps. Turns are not a config field. See [docs/algorithm.md](docs/algorithm.md) §4.

## License and contributing

MIT — see [LICENSE](LICENSE). Contributions are welcome; the development setup, quality
commands and commit conventions are in [CONTRIBUTING.md](CONTRIBUTING.md).

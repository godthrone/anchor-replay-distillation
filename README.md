# anchor-replay-distillation

**Anchor Replay Distillation (ARD) turns an ontology-defined coordinate space into a
reproducible supervised-finetuning corpus.** For every anchor coordinate it asks a question
generator for the user turn and a target "teacher" model for the answer, then keeps the
teacher's answer — and, when enabled, its reasoning — beside the question that produced it.
ARD is the *data-generation* half of a distillation pipeline: it does not train, distil or
evaluate a student model.

The plan comes from the v4 ontology (`ontology/anchor_ontology.v4.json`) and is derived, not
configured: **935 legal text-modality restricted blocks + 891 legal image-modality restricted
blocks = 1,826 anchors** — one anchor per block *within each modality group*, with the 209
`knowledge_domain` leaves (and, for image coordinates, the 21 `visual_domain` leaves) rotated
across them. The 891 image-capable blocks are a subset of the 935, so those coordinates appear
twice, once per modality; the two are told apart by `modality`, which is part of the identity a
duplicate check uses. That fixed
target set is what makes a run auditable: every accepted coordinate is either covered exactly
once or the run refuses to proceed.

## Quick start

Three steps, ending in a real artifact. They call real endpoints — `--smoke` only makes the
first artifact small.

### 1. Install dependencies

```bash
uv sync
```

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 (pinned in `.python-version`);
`uv.lock` pins every dependency. The package version is derived from git tags by
`setuptools_scm`: a `git clone` reports the exact tag-derived version, while a source archive
that carries no `.git` (GitHub "Download ZIP", `git archive`) falls back to the fixed version
`1.0.0` — the same value `run.sh` and `docker/build.sh` use when there is no tag to describe.

**RAW support is not installed by default.** A plain `uv sync` installs no `rawpy` — it lives in
the `raw` extra — so the default install carries no LibRaw. The default dependency closure is
mostly permissive (MIT / BSD / MIT-CMU / PSF-2.0), but it is **not** MIT/Apache only: `certifi`
is **MPL-2.0** (pulled in transitively by `httpx`), `tqdm` is **`MPL-2.0 AND MIT`** (a dual
licence, so the MIT option can be taken) and `typing-extensions` is **PSF-2.0**. RAW camera
formats (`.cr2`, `.nef`, `.arw`, `.dng`, …) need `rawpy`, whose wheels bundle LibRaw
(LGPL-2.1 / CDDL-1.0) — which is why RAW support is an opt-in extra. Install it explicitly when
you feed RAW files:

```bash
uv sync --extra raw
```

Without it a RAW input is reported and skipped (a WARNING naming the file) rather than crashing
the run — the FAQ entry below has the detail and the licence reason. The complete
package-by-package list, and how to reproduce it from your own install, is in
[`docs/licenses.md`](docs/licenses.md); it is stated as fact, not as a compliance verdict.

### 2. Provide endpoint credentials

```bash
mkdir -p .local && cp configs/config.override.sample.toml .local/config.override.toml
```

The `mkdir -p` is not decoration: `.local/` is gitignored, so a fresh clone does not have it
and `cp` would fail with `No such file or directory`. Both READMEs run this as one line.

Then fill in `api_base`, `model_name` and `api_key` for `[input_generator]` and
`[target_model]` (and, only if you want the `q95` readout, `[coverage.embedding]`).

If `.local/config.override.toml` **already exists, edit it instead of copying over it** — the
copy would silently overwrite credentials that are already deployed there.

The override is merged over `configs/config.toml` by a fixed three-tier priority, first hit
wins:

1. an explicit `--override PATH` (a missing file is an error);
2. `<project_root>/.local/config.override.toml` — the recommended, gitignored location;
3. `<directory of --config>/config.override.toml` — backwards compatibility.

The choice is never silent: every run logs at INFO which override file it used, or that it
used the base configuration only. Field names are validated with `extra="forbid"`, so a typo
is an error rather than a silently ignored setting.

With no credentials the run stops **before creating any output directory**, with a
field-level message and exit code 1:

```
ERROR: [input_generator] is missing `api_base`, `model_name`.
```

### 3. Run the smoke test

```bash
./run.sh --config configs/config.toml --smoke --image-dir examples/images
```

It builds the Docker image on first use (needs PyPI access once) and runs
`python -m ard` inside it, with `configs/`, `ontology/`, `examples/` and `.local/` mounted
read-only and `outputs/` writable. Without Docker, the environment from step 1 runs the same
entry point:

```bash
uv run python -m ard --config configs/config.toml --smoke --image-dir examples/images
```

**What `--smoke` means.** It materialises the *same* construction rule at a reduced scale —
8 restricted blocks (4 text blocks + 4 image blocks, evenly spaced over the enumeration, first
and last included). Those are 8 plan slots but only **6 distinct `RestrictedBlock`s**: two blocks
appear once per modality, the same 891⊂935 subset identity behind the 1,826 count above. Those
blocks are a subset of the blocks the full plan enumerates (8/8
block-level coverage), but their coordinates are re-rotated at smoke scale, so the 8 anchors
are **not** rows of the 1,826-anchor plan (only 1/8 coincide coordinate-wise). Its point is
that a fresh checkout can see a complete artifact in minutes. The
artifact is deliberately incomplete and says so in three independent places: the run
directory name gets the `_smoke` suffix, the log carries a WARNING naming the scale, and
`manifest.json` sets `smoke: true` plus a `smoke_plan` block. `--smoke` is a CLI run-boundary
flag, not a configuration field, and changes nothing about a run without it.

The artifact lands in `outputs/<run_name>/` (default `ard_dataset_<YYYYmmdd_HHMMSS>_smoke`;
set `[output] directory` in config to pin the name). After a few minutes you should have:

- `anchor_bank.jsonl` — the 8 anchors, `schema_version 4.0.0`;
- `results/coverage.json` + `results/coverage.md` — the acceptance readout. The **structure
  readout** (plan counts vs. the construction rule) needs no model call and is always
  written; with no target set configured it reports `within_rule: false` against the full
  plan and `metrics: null`, which is exactly what a smoke run should say — 8 anchors are not
  the 1,826-anchor plan, and the report does not pretend otherwise;
- `manifest.json` — composition plus the smoke declaration;
- `config.toml` and `logs/`. While the run is still going there is also
  `plan_identity.in_progress.json`, an intermediate record that binds the directory to the
  plan; a finished run keeps only `manifest.json` as its declaration.

**Full run** — all 1,826 anchors, one endpoint call per turn. Supply your own image directory
laid out as `<image_dir>/<visual_domain>/<image file>` with all 21 visual domains present:

```bash
./run.sh --config configs/config.toml --image-dir /path/to/images
```

A full run costs hours of endpoint time; run the smoke test first to validate the setup.

## Data format

`anchor_bank.jsonl` holds one JSON object per line:

```jsonc
{
  "id": "anchor_…",                 // sha256 over the sampled axes
  "source": "ard",
  "data_source": "ard_text",        // controlled vocabulary: ard_text | ard_multi
  "schema_version": "4.0.0",
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

`anchor_meta` carries the sample field `modality` and every axis value: `language`,
`knowledge_domain`, `capability`, `system_prompt_mode`, `conversation_type`,
`response_style`, `output_format`, `difficulty`, `context_length`, `input_condition`,
`answer_mode` — plus `visual_domain` for image-modality anchors. A text-modality coordinate
omits `visual_domain` entirely (no `null` placeholder) and carries `has_image: false` /
`image_count: 0` instead; an image anchor carries `visual_domain` with `has_image: true` /
`image_count: 1`. `data_source` is per record: `ard_multi` marks a conversation that carries
at least one image, and downstream routing uses that key.

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

The picture comes from `<image_dir>/<visual_domain>/<file>` — a picture belongs to the
anchor's own `visual_domain` coordinate, never to a flat pool, because a flat pool cannot
guarantee that the image matches the label it is attached to. Paths are relative to the run
directory, so `outputs/<run_name>/images/<visual_domain>/…` resolves directly; the run copies
and (unless `[images] convert = false`) converts the selected image there. A worked, small example of
these records is checked in under `examples/` — see [examples/README.md](examples/README.md).

## Configuration reference

Every field lives in `configs/config.toml` (English comments, all fields defined) and the
override supplies values only. **All artifacts are decided by the config** — the CLI passes
input locations and run boundaries only: `--config`, `--override`, `--image-dir`,
`--smoke`.

| Section | What it controls |
|---|---|
| `[input_generator]` | The question-generator endpoint (`api_base`, `model_name`, `api_key`), sampling temperature, timeouts, retries |
| `[target_model]` | The teacher endpoint — its answer is the supervision target. `enable_thinking` switches the reasoning channel on (`targets[0].output.reasoning`) |
| `[generation]` | `concurrency`, the optional `seed` (omit it and every run draws a fresh seed; the seed actually used is recorded in the run's `config.toml`), backpressure thresholds. The anchor count and the turn counts are **not** configurable — they come from the ontology |
| `[ontology]` | Path to the v4 ontology (`ontology/anchor_ontology.v4.json`) |
| `[output]` | `directory` (empty = `outputs/ard_dataset_<timestamp>`), `overwrite` (default `false`: an existing bank is resumed, not replaced) |
| `[images]` | `convert` (default **`true`**): accept RAW/BMP/TIFF/GIF/WebP and normalise every selected picture to JPEG (RAW additionally needs the optional `raw` extra); `false` accepts only the already-web formats and copies them verbatim. `skip_missing_images` (default **`false`**): a missing `visual_domain` directory is refused, not skipped |
| `[coverage]` | `enabled` (default `true`) and `target_set_path` — the target set for the metric readout; empty means structure readout only |
| `[coverage.embedding]` | The OpenAI-compatible `/embeddings` endpoint, model, expected `dimension`, batch size and timeouts used by the metric readout |

`configs/config.override.sample.toml` mirrors every field as a commented template; it is the
file step 2 copies.

### Interface change: image conversion moved into the config

`[images] convert` is new, and the `--no-convert` CLI flag has been removed. The flag decided
what landed in `outputs/<run>/images`, so the project's engineering rule that **one config
describes one artifact** puts that decision in the config: a run archived as `config.toml` must
reproduce the image bytes as well as the anchors. The flag is deliberately **not** kept as a
compatibility alias — a CLI flag and a config field for the same decision would be two sources
of truth. The behaviour the old flag asked for is `convert = false` today.

The archive is itself a valid `--config`: `python -m ard --config outputs/<run>/config.toml
--override config.override.toml` re-runs the same configuration with your credentials supplied
by the override.

## Output

```text
outputs/<run_name>/          # default ard_dataset_<YYYYmmdd_HHMMSS>; --smoke appends _smoke
├── anchor_bank.jsonl        # one record per line, schema_version 4.0.0
├── config.toml              # merged config snapshot, credentials redacted
├── plan_identity.in_progress.json  # only while a run is unfinished: its plan-identity record
├── logs/                    # ard.log / ard_debug.log / ard_error.log
├── results/
│   ├── coverage.json        # machine-readable acceptance readout
│   └── coverage.md          # the same readout, human-readable
└── manifest.json            # composition, run health, config, acceptance pointer (the authoritative one)
```

`manifest.json` is the **authoritative** declaration: `status: "complete"`, the bank's composition, the
run-health counters and the `acceptance` pointer. `plan_identity.in_progress.json` is an intermediate
record, written once the plan exists and before the first endpoint call, and deleted when the run
finishes. It carries `status: "in_progress"` and the run's `plan_identity` — **not** run health, because
none exists yet. It is what makes an interrupted run auditable: the 1,826-anchor path is routinely
stopped and resumed, and this record binds such a directory to the plan that produced it. Its `counters`
are a snapshot, so it must never be read as a completion or acceptance claim; the plan identity in it can
be verified by recomputing `PlanIdentity.of(plan)` and comparing digests. See
[docs/architecture.md](docs/architecture.md) section 4.

`results/coverage.{json,md}` is the acceptance readout, not training data:

- the **structure readout** (plan counts vs. the construction rule) is always produced and
  costs nothing — no model call;
- the **metric readout** (`q95`, `Extent(ε)` with the ε±5% band, paired bootstrap CI, noise
  band) requires both `coverage.target_set_path` and `[coverage.embedding]`.

Without them the run writes the structure readout only and announces the gap in a WARNING
(`metric readout not measured …`), with `metric_readout: false` and `q95: null` in the
manifest — never a silently clean report. The definitions of the ruler and its resolution
limits are in [docs/measurement.md](docs/measurement.md).

The metric space holds **text only**: an image-modality anchor takes part through the text
parts of its final user turn, and its image pixels never enter the space, so the readout
declares the anchor field as `messages[last].content(text parts only)`.

### Re-run the metric readout yourself (three steps)

The metric readout is config-driven — there is no CLI flag for it:

1. Put an embeddings endpoint in `.local/config.override.toml` (the gitignored override).
   Field names and placeholders only — never commit a real endpoint, model name or key:

   ```toml
   [coverage.embedding]
   api_base = "<OpenAI-compatible base URL, including /v1>"
   model = "<embedding model name>"
   dimension = <vector length>
   # api_key = "<only if the server needs one>"
   # normalize = true        # must stay true: the ruler needs unit-norm rows
   ```
2. Point `coverage.target_set_path` at a target set. The small, deterministic sample
   `examples/target_set.sample.jsonl` (32 entries; the construction rule is declared in its
   file header and in [docs/measurement.md](docs/measurement.md)) works as is:

   ```toml
   [coverage]
   target_set_path = "examples/target_set.sample.jsonl"
   ```
3. Run `--smoke` (8 anchors, minutes) or a full run. The readout lands in
   `<output_dir>/results/coverage.{json,md}`.

The three configuration combinations are a contract, not a suggestion: **neither** set →
structure readout + a WARNING; `target_set_path` set but `[coverage.embedding]` incomplete →
the run is refused at config load with the missing field(s) named, before any output
directory exists; **both** set → the metric readout.

## Development

```bash
uv sync --extra dev     # test, type-check and lint tools
uv run pytest
uv run ruff check src/ tests/
uv run mypy src/ard/
```

The same workflow is described in [CONTRIBUTING.md](CONTRIBUTING.md).

**`--smoke` vs. a full run:** the construction rule, the sampling and the per-anchor
generation path are identical; only the plan scale differs (8 anchors vs. 1,826). A smoke
artifact is deliberately incomplete and must not be delivered as a dataset — the `_smoke`
suffix, the log WARNING and `manifest.json: smoke` exist so it cannot be mistaken for one.
Both paths call the configured endpoints; neither is an offline demo.

## FAQ

**An image domain is missing — what happens?**
By default the run is refused **before the output directory is created**, with an error that
lists every missing visual domain, its expected path (`<image_dir>/<visual_domain>/`) and how
many anchors it affects. Either supply the images, or set `[images] skip_missing_images =
true`: those anchors are then dropped, each with its own WARNING, and the skipped count and
domains are declared in `manifest.json` — a skip is never silent. A run that omits
`--image-dir` altogether attaches no pictures at all: image-modality anchors are still
generated, as text conversations, while keeping their `visual_domain` coordinate. Pass
`--image-dir` when you want the images.

**Why do RAW camera files need `.[raw]`?**
RAW decoding goes through `rawpy`, whose wheels bundle the LibRaw decoder (LGPL-2.1 /
CDDL-1.0). A plain `uv sync` installs no LibRaw at all — `rawpy` is the only package the `raw`
extra adds — so the RAW decoder is an opt-in extra rather than part of the default set: `uv sync
--extra raw` (see the install step above). The default dependency closure is not uniformly
MIT/Apache: `certifi` is MPL-2.0, `tqdm` is `MPL-2.0 AND MIT`, `typing-extensions` is PSF-2.0,
and the rest are MIT / BSD-3-Clause / MIT-CMU. [`docs/licenses.md`](docs/licenses.md) carries the
full list together with the command that reads it out of an installed environment; this
repository states those facts only and leaves any compliance judgement to you. A default install
does not break on a RAW input — it logs a WARNING naming the file ("`rawpy not installed, cannot
convert RAW image: …`") and drops that picture while the rest of the run proceeds.

**Why is `q95` the main ruler while coverage is only a reference?**
`q95` is the 95th percentile (Hyndman–Fan type 7) of each target point's distance to its
nearest anchor: a tail statistic that reads "how far is the worst-served 5% of the target
set". `Extent(ε)`, the fraction of target points within ε, is kept as a secondary readout
only, because it moves with the choice of ε — so it is published together with its ε±5% band
rather than as a pass/fail number. Definitions, and the sensitivity that motivates this
split, are in [docs/measurement.md](docs/measurement.md).

**What is the noise band?**
Regenerating the same coordinate does not reproduce the same text, so two answers for one
coordinate sit some distance apart. The noise band is the distribution of those
same-coordinate pairwise distances, `[q50, max]`. A `q95` difference inside it cannot be
distinguished from generation randomness; the reading is therefore used for self-consistency
and regression, not to claim small algorithm improvements, and a reading inside the band is
reported as "indistinguishable" rather than "better". Detail:
[docs/measurement.md](docs/measurement.md).

**What does `MULTI_TURN_DEFAULT = 4` mean?**
The ontology declares `turns: "multi"` for two `conversation_type` values without stating a
numeric upper bound. The implementation reads "multi" as the largest turn count the ontology
itself declares (`constraint_update` = 4), keeps that number in a single constant, and raises
an error if the ontology ever declares more — it never silently clamps. Turns are not a
config field: to change them, change the ontology (or that constant), not
`configs/config.toml`. See [docs/algorithm.md](docs/algorithm.md) §3.

**The ontology md5 changed — do earlier readings need recomputing?**
No. The v4 ontology's md5 changed once, over a one-line `supersedes` provenance edit. The
four counts, the unreachable-pair group and the 1,826-line plan are unchanged, so existing
`q95` readings, counts and plans do not need recomputing. The hashes and the item-by-item
check are in [docs/algorithm.md](docs/algorithm.md) §5.

## License and contributing

MIT — see [LICENSE](LICENSE). Contributions are welcome; the development setup, quality
commands and commit conventions are in [CONTRIBUTING.md](CONTRIBUTING.md).

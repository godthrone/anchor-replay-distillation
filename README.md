# anchor-replay-distillation

**Anchor Replay Distillation (ARD) turns an ontology-defined coordinate space into a
reproducible supervised-finetuning corpus.** For every anchor coordinate it asks a question
generator for the user turn and a target "teacher" model for the answer, then keeps the
teacher's answer — and, when enabled, its reasoning — beside the question that produced it.
ARD is the *data-generation* half of a distillation pipeline: it does not train, distil or
evaluate a student model.

The plan is built from the v4 ontology (`ontology/anchor_ontology.v4.json`) by one rule —
**cycle-shuffle**: every *coverage unit* is a `(modality, legal restricted block)` pair; each
round shuffles the whole unit set, walks it without replacement, then reshuffles for the next
round. The unit count `U` is enumerated at runtime, never written into the ontology or the code
(one unit for every legal text block, plus one for every image-capable block; the image-capable
blocks are a subset of the legal blocks, so those coordinates appear once per modality, told apart
by `modality`).
The recompute command is in [docs/algorithm.md](docs/algorithm.md) §1.

**How many anchors a run produces is yours to decide.** `N` is set in exactly one place —
`[generation] count` in `configs/config.toml`; leaving it unset means one full round (`U`), and
there is **no upper bound**: a larger `N` simply rolls into round 1, 2, … Each anchor's `id` is a
*plan-position serial number* (`run_key` + round + position), so raising `N` leaves every
already-generated id byte-identical and a resumed run appends cleanly. See
[Choosing how many anchors](#choosing-how-many-anchors-n).

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

**What `--smoke` means.** It materialises the *same* cycle-shuffle rule at a reduced scale:
**8 anchors — 4 text-only + 4 image-capable**, taken from round 0's shuffle order (the first four
units of each modality). Its point is that a fresh checkout can see a complete artifact in
minutes, and it is a real endpoint run, not an offline demo. The artifact is deliberately small
and says so in three independent places: the run directory name gets the `_smoke` suffix, the log
carries a WARNING naming the scale, and `manifest.json` sets `smoke: true` plus a `smoke_plan`
block. `--smoke` is a CLI run-boundary flag, not a configuration field, and changes nothing about
a run without it.

A smoke plan is **not a prefix of the full plan**: the full plan takes the first `N` units of the
single round-0 order, while smoke takes the first four units *of each modality* — so a shared id
can point at a different coordinate. That is exactly why the resume guard compares coordinates
position by position instead of comparing id sets, and why a smoke directory and a full directory
must not be mixed.

The artifact lands in `outputs/<run_name>/` (default `ard_dataset_<YYYYmmdd_HHMMSS>_smoke`;
set `[output] directory` in config to pin the name). After a few minutes you should have:

- `anchor_bank.jsonl` — the 8 anchors, `schema_version 5.0.0`;
- `results/coverage.json` + `results/coverage.md` — the acceptance readout, `report_schema
  "ard-acceptance-3"`. The **structure readout** (plan counts vs. this run's own `N` and round
  decomposition) needs no model call and is always written. A smoke plan is judged as what it is —
  an 8-entry plan — so `within_rule` is `true` and the coverage line reads the ratio of those 8
  entries to this run's own unit total `U` (a small number);
  with no target set configured, `metrics: null` and a WARNING say the metric readout was not
  measured. The report never pretends 8 anchors are a full-round delivery;
- `manifest.json` — composition plus `plan_identity` (v2) and the `plan` section;
- `config.toml` and `logs/`. While the run is still going there is also
  `plan_identity.in_progress.json`, an intermediate record that binds the directory to the
  plan; a finished run keeps only `manifest.json` as its declaration.

**Full run** — one full round is `U` anchors (the runtime-enumerated unit count; set
`[generation] count` for any other `N`), one endpoint call per turn. Supply your own image
directory laid out as `<image_dir>/<visual_domain>/<image file>`:

```bash
./run.sh --config configs/config.toml --image-dir /path/to/images
```

Pictures rotate, and few pictures are reused rather than skipped: a domain whose directory
holds no image is served from the whole image tree instead, so a single picture covers every
domain and every round (the reuse is counted and named in `manifest.json`). Only a tree with no
usable image at all stops the run. A full run costs hours of endpoint time; run the smoke test
first to validate the setup.

## Choosing how many anchors (N)

`N` has exactly one source — the `[generation] count` field in `configs/config.toml`:

```toml
[generation]
# Omit (the default) = one full round = U, the runtime-enumerated unit count.
# Set any integer >= 1 to produce exactly that many anchors; N rolls on across
# rounds (finishing a round starts the next one).
count = 5000
```

- **No CLI flag, no `run.sh` passthrough, no environment variable.** A CLI flag and a config field
  for the same value would be two sources of truth, so the value lives only in the config.
- **No upper bound.** A large `N` is not refused; the run logs how many full rounds it spans and
  how many anchors fall in the last, partial round.
- **What is refused**: `count < 1`, and non-integer types (`0`, negatives, `true`, `"1826"`,
  `1.5`). The error names the field and tells you to remove the line to get one full round.
- **Raising `N` on an existing directory appends.** Anchor ids are plan-position serial numbers
  that do not contain `N`, so running the same `output.directory` with a larger `count` keeps every
  existing record and appends the new ones. Changing the seed, the ontology, or the smoke/full
  shape is refused with a message telling you to use a new directory.
- **Coverage and density are read separately.** *Coverage* counts coordinates and saturates at
  `1.0` once every unit has been visited (`min(distinct, U) / U`); *density* counts entries and
  keeps growing with `N` (`N / U`). Revisiting a coordinate in a later round is a legitimate new
  sample — the generator is stochastic, so the same labels yield a different question — and is
  never dropped. Definitions: [docs/measurement.md](docs/measurement.md) §6.

## Data format

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

The picture comes from `<image_dir>/<visual_domain>/<file>` — a picture filed under the anchor's
own `visual_domain` coordinate is the one whose content the user vouched for, so it is always
preferred. When that directory holds no image, the run does **not** drop the anchor: it reuses a
picture from the whole image tree (every usable file under `--image-dir`, in a deterministic
order) and rotates through it, so a user who supplied only one picture still gets a complete
run. Every reuse is recorded — `fallback_visual_domains` / `fallback_anchor_count` in
`manifest.json`, plus a `fallback` flag on each `resolved_images` row — because whether a reused
picture is good enough for the dataset is the user's judgement, and it needs to be visible.
Paths are
relative to the run directory, so `outputs/<run_name>/images/<visual_domain>/…` resolves directly;
the run copies and (unless `[images] convert = false`) converts the selected image there. A
worked, small example of these records is checked in under `examples/` — see
[examples/README.md](examples/README.md).

## Configuration reference

Every field lives in `configs/config.toml` (English comments, all fields defined) and the
override supplies values only. **All artifacts are decided by the config** — the CLI passes
input locations and run boundaries only: `--config`, `--override`, `--image-dir`,
`--smoke`.

| Section | What it controls |
|---|---|
| `[input_generator]` | The question-generator endpoint (`api_base`, `model_name`, `api_key`), sampling temperature, timeouts, retries |
| `[target_model]` | The teacher endpoint — its answer is the supervision target. `enable_thinking` switches the reasoning channel on (`targets[0].output.reasoning`) |
| `[generation]` | `count` (the anchor count `N`: omit = one full round = `U`; any integer `>= 1`, **no upper bound**), `concurrency`, the optional `seed` (omit it and every run draws a fresh seed; the seed actually used is recorded in the run's `config.toml`), backpressure thresholds. Turn counts are **not** configurable — they come from the ontology |
| `[ontology]` | Path to the v4 ontology (`ontology/anchor_ontology.v4.json`) |
| `[output]` | `directory` (empty = `outputs/ard_dataset_<timestamp>`), `overwrite` (default `false`: an existing bank is resumed, not replaced) |
| `[images]` | `convert` (default **`true`**): accept RAW/BMP/TIFF/GIF/WebP and normalise every selected picture to JPEG (RAW additionally needs the optional `raw` extra); `false` accepts only the already-web formats and copies them verbatim. `skip_missing_images` (default **`false`**): only when `--image-dir` holds no usable image at all is a domain left without a picture — that is refused, not skipped |
| `[coverage]` | `enabled` (default `true`) and `target_set_path` — the target set for the metric readout; empty means structure readout only |
| `[coverage.embedding]` | The OpenAI-compatible `/embeddings` endpoint, model, expected `dimension`, batch size and timeouts used by the metric readout |

`configs/config.override.sample.toml` is the commented template step 2 copies; it is where
deployment values (endpoints, model names, credentials) go, never `configs/config.toml` itself.

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
├── anchor_bank.jsonl        # one record per line, schema_version 5.0.0
├── config.toml              # merged config snapshot, credentials redacted
├── plan_identity.in_progress.json  # only while a run is unfinished: its plan-identity record
├── images/                  # where image-modality anchors' pictures land: images/<visual_domain>/<file> (transcoded or copied, one placed copy per source file); created only when this run has image anchors
├── logs/                    # ard.log / ard_debug.log / ard_error.log
├── results/
│   ├── coverage.json        # machine-readable acceptance readout (report_schema ard-acceptance-3)
│   └── coverage.md          # the same readout, human-readable
└── manifest.json            # composition, run health, config, plan_identity v2, plan, images, acceptance pointer (the authoritative one)
```

`manifest.json` is the **authoritative** declaration: `status: "complete"`, the bank's composition, the
run-health counters and the `acceptance` pointer. Three sections describe the plan itself:

- `plan_identity` (**v2**) — the plan's name: `algorithm`, `version`, `sampling`, `ontology_sha256`,
  `seed`, `count` (your `N` exactly as requested; `null` = one full round), `unit_total` (`U`),
  `plan_size` and `digest` (sha256 over the ordered coordinate list and those inputs);
- `plan` — shape and readouts: `unit_total`, `full_cycles`, `last_cycle_size`, `planned_anchors`,
  `written_anchors`, `distinct_coordinates`, `coverage_ratio`, `density`, `smoke`;
- `images` — `resolved_images` (one `{cycle, visual_domain, image, fallback}` row per resolved
  picture), `domain_candidate_counts` (how many usable files each domain offered itself),
  `pool_candidate_count` (how many the tree-wide fallback pool held), and
  `fallback_visual_domains` / `fallback_anchor_count` (which domains, and how many anchors, were
  shown a reused picture).

`plan_identity.in_progress.json` is an intermediate
record, written once the plan exists and before the first endpoint call, and deleted when the run
finishes. It carries `status: "in_progress"` and the run's `plan_identity` — **not** run health, because
none exists yet. It is what makes an interrupted run auditable: a large-`N` path is routinely
stopped and resumed, and this record binds such a directory to the plan that produced it. Its `counters`
are fixed before the first endpoint call and describe the *plan*, not progress: `existing` is what the
bank already held, `new` and `written` are the number of anchors this invocation set out to generate and
write (the same number). They must never be read as a completion, progress or acceptance claim — the
count of records actually persisted is `manifest.json`'s `total_anchors` — the full bank, records that
were already on disk included; `generation.counters.written` counts only what this invocation wrote,
so the two differ on a resumed run. The plan identity
in it can be verified by recomputing `PlanIdentity.of(...)` and comparing digests. See
[docs/architecture.md](docs/architecture.md) section 4.

`results/coverage.{json,md}` is the acceptance readout, not training data
(`report_schema "ard-acceptance-3"`):

- the **structure readout** (plan counts vs. this run's own `N` and round decomposition,
  including `coverage` / `density` / `full_rounds`) is always produced and costs nothing — no model
  call. A smoke run is judged against its own 8-entry plan, so it reports `within_rule: true` and a
  coverage of those 8 entries over this run's own unit total `U` (a small ratio);
- the **metric readout** (`q95`, `Extent(ε)` with the ε±5% band, paired bootstrap CI, noise
  band) requires both `coverage.target_set_path` and `[coverage.embedding]`.

Without them the run writes the structure readout only and announces the gap in a WARNING
(`metric readout not measured …`), with `metric_readout: false` and `q95: null` in the
manifest — never a silently clean report. The definitions of the ruler, of coverage vs. density,
and of its resolution limits are in [docs/measurement.md](docs/measurement.md).

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

## Adding a knowledge domain

**Adding one `knowledge_domain` leaf means editing the leaf list in the ontology and nothing
else.** There are no hand-written counts to update and no constant to bump: the coverage units,
the leaf rotations and the acceptance expectations are all enumerated from the ontology at
runtime, so a new leaf is picked up on the next run without touching `configs/`, `src/` or the
code. (Adding a leaf changes the ontology hash, so it is a *different plan* — a directory
produced before the edit cannot be resumed; use a new `output.directory`.)

## Development

```bash
uv sync --extra dev     # test, type-check and lint tools
uv run pytest
uv run ruff check src/ tests/
uv run mypy src/ard/
```

The same workflow is described in [CONTRIBUTING.md](CONTRIBUTING.md).

**`--smoke` vs. a full run:** the construction rule, the sampling and the per-anchor
generation path are identical; only the plan scale differs (8 anchors vs. one full round). A smoke
artifact is deliberately incomplete and must not be delivered as a dataset — the `_smoke`
suffix, the log WARNING and `manifest.json: smoke` exist so it cannot be mistaken for one.
Both paths call the configured endpoints; neither is an offline demo.

## FAQ

**An image domain is missing — what happens?**
Images are reused rather than skipped. A picture is always taken from
`<image_dir>/<visual_domain>/` when that directory holds one; when it does not, the run falls
back to the whole image tree and rotates through it, so **if you supplied one picture, every
domain and every round uses that picture** — the run is complete, and whether the repetition
hurts the dataset is your call. That call needs the facts, so each reuse is visible:
`manifest.json.images` carries `fallback_visual_domains`, `fallback_anchor_count`,
`pool_candidate_count` and a per-row `fallback` flag, and every reuse is logged.

Only a tree with **no usable image at all** is refused, **before the output directory is
created**, with an error that lists every domain left without a picture, the directory it would
be read from (`<image_dir>/<visual_domain>/`) and how many anchors it affects. Either supply at
least one image, or set `[images] skip_missing_images = true`: those anchors are then dropped,
each with its own WARNING, and the skipped count and domains are declared in `manifest.json` — a
skip is never silent. A run that omits `--image-dir` altogether attaches no pictures at all:
image-modality anchors are still generated, as text conversations, while keeping their
`visual_domain` coordinate. Pass `--image-dir` when you want the images.

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
`configs/config.toml`. See [docs/algorithm.md](docs/algorithm.md) §4.

**Can I make a very large dataset — is there a ceiling on `N`?**
There is no ceiling. `N` comes from `[generation] count` and anything `>= 1` is accepted; a value
larger than one round simply rolls on into the next round. Coverage stops growing once every unit
has been visited (`min(distinct, U) / U`, saturating at `1.0`) while density (`N / U`) keeps
growing, because a coordinate revisited in a later round is a new, legitimate sample — the
question generator is stochastic, so the same coordinate yields a different question. Nothing is
dropped as a "duplicate". The one thing to know is that a very large `N` is a very large amount of
endpoint time, not a limit of the tool.

**Why do anchor ids look like `1a2b3c4d-c00000p00137`?**
Because an id is a **plan position**, not a fingerprint of the coordinate. `run_key` mixes the
ontology hash, the seed and the sampling algorithm; `c00000` is the round and `p00137` the position
inside it. Since `N` is not part of it, enlarging `count` leaves existing ids unchanged and lets a
run append to its own bank. The coordinate is a separate, fully recorded field (`anchor_meta`), and
two records may share one coordinate with two different ids — both are kept on purpose.

## License and contributing

MIT — see [LICENSE](LICENSE). Contributions are welcome; the development setup, quality
commands and commit conventions are in [CONTRIBUTING.md](CONTRIBUTING.md).

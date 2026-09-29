# Examples

Everything in this directory is **real pipeline output**, checked in so you can see what ARD
produces without spending endpoint time. `anchor_bank.sample.jsonl` is the complete anchor bank of
one real `--smoke` run — every line is that run's own output, copied byte for byte, nothing
trimmed, reordered or re-worded. `manifest.sample.json` is the manifest that same run wrote, with a
handful of environment-specific fields normalised (listed below).

| Path | What it is |
|------|------------|
| `anchor_bank.sample.jsonl` | The anchors one real `--smoke` run produced, one JSON object per line (`schema_version 5.0.0`) |
| `manifest.sample.json` | The manifest that run wrote next to its bank (`plan_identity` v2, the `plan` section and `status` included) |
| `images/` | 10 small placeholder JPEGs, one subdirectory per `visual_domain` |
| `images/README.md` | What those pictures are (placeholders) and the addressing convention |

## What the run was

```bash
# from the repository root
./run.sh --config configs/config.toml --smoke --image-dir examples/images
```

The run pinned `[generation] seed = 7360` in the gitignored local override of the checkout it ran
in so the *plan* is reproducible. The repository's own `.local/config.override.toml` leaves `seed`
unset, so to reproduce this exact plan add `seed = 7360` under `[generation]` there (or point
`--override` at a file that sets it). The seed is kept verbatim in the sample because it names the
plan, not the deployment. `[generation] count` was left unset, as a smoke run always does.

**Why eight anchors.** `--smoke` is the standard cycle-shuffle rule at reduced scale: it takes the
**first four units of each modality from round 0's shuffle order** — **4 text-only + 4
image-capable = 8 planned anchors**, under the same ids and the same coordinate rule a full run
uses. A smoke plan is *not* a prefix of the full plan: the full plan takes the first `N` units of
the single round-0 order, while smoke takes the first four *of each modality*, so an id the two
share can point at a different coordinate. All 8 planned anchors are present in the bank.

**The bank was assembled over several invocations of the same command.** Anchor generation calls a
live endpoint, and an anchor is only written when the whole conversation is obtained. The earlier
invocations wrote most of the plan and abandoned the rest — a transient transport error, and a
teacher that returned only reasoning when the configuration asked for both — and re-running the
same command against the same `output.directory` resumed each time: the resume guard verified that
every existing record still sat at its own plan position, then asked the endpoint only for the
anchors still missing. The finishing invocation asked for the **2** anchors still absent. This is
why the sample's `generation.counters` reads `requested 2 / succeeded 2 / written 2` while
`total_anchors` and `plan.planned_anchors` are **8**: the counters describe the invocation that
finished the run, the totals describe the whole bank (see `manifest.sample.json` below).

**Re-running the command does not promise the same number of records.** Generation is stochastic
and an anchor whose conversation fails — a transport error, empty generated content, or a teacher
that returns only reasoning when the configuration asked for both — is **discarded for that
invocation and counted** in `generation.counters.abandoned_by_reason`; it is never written with a
placeholder value. A smoke run may therefore write fewer than 8 records on any given attempt, and
the count is a property of *that run*, not of the smoke scale. The manifest is where the
difference is declared. Re-running an incomplete directory appends the missing records rather than
rewriting the ones already there.

## Provenance — which build produced these files

Both sample files are the artifacts of one `--smoke` run of this repository's v5 working tree
(cycle-shuffle sampling, position-serial anchor ids, `schema_version 5.0.0`), made on 2026-09-28
against a real deployment endpoint (credentials taken from the gitignored local override). The
acceptance phase produced its zero-model **structure readout** only (plan counts vs. the
construction rule); no embedding endpoint was involved.

The run's manifest records the identity of the plan it drew:

```json
"plan_identity": {
  "algorithm": "sha256",
  "version": 2,
  "sampling": "cycle-shuffle/v1",
  "ontology_sha256": "807cd068076f0b2bea7eb5d3eccdecac927b3521fa7034e599456bb3553ad36f",
  "seed": 7360,
  "count": 8,
  "unit_total": 1826,
  "plan_size": 8,
  "digest": "281d213334e7006c89807b18f0a345e7e353bb5c791e21ae3ab484f7e8c9e36a"
}
```

`plan_identity` describes the *plan*, not the endpoint or the deployment, so it is kept in the
sample on purpose: it is what lets a later run prove it is appending to the same plan rather than
to a different one. Version 2 adds the ontology fingerprint, the requested `count`, `U` and the
sampling algorithm to the digest inputs.

## The record schema

Each line of `anchor_bank.sample.jsonl` is one anchor:

```jsonc
{
  "id": "f384795e-c00000p00000",   // plan-position serial number: run_key + round + position
  "source": "ard",                 // dataset tag of the ARD anchor-bank format
  "data_source": "ard_text",       // controlled vocabulary: ard_text | ard_multi
  "schema_version": "5.0.0",       // the record format version — same for every record
  "messages": [ /* the conversation, see below */ ],
  "targets": [
    {
      "id": "primary",
      "output": {
        "content": "…the teacher's answer…",
        "reasoning": "…the teacher's reasoning (or null)…"
      }
    }
  ],
  "anchor_meta": { /* the coordinate: one key per axis */ },
  "input_generator_model": "…",    // the model that produced the user turns
  "teacher_id": "…"                // the model whose output is the supervision target
}
```

Both model fields name the **deployed** models in a real run. In this sample they are the
placeholder `your-model-name` — see the normalised-fields table below.

### `id` is a plan position, not a coordinate fingerprint

An id is `` `run_key` + "-c" + round + "p" + position `` — here
`f384795e-c00000p00000` through `f384795e-c00000p00007`. `run_key` is
`H(ontology hash, seed, sampling algorithm)[:8]`; `c00000` is the round and `p00000` the position
inside it. The id deliberately does **not** contain `N`, so raising `[generation] count` later
leaves every id already on disk byte-identical and a resumed run can append cleanly. Because the
id is a position and not a fingerprint of the coordinate, **two records may carry the same
`anchor_meta` with different ids** — the generator is stochastic, so resampling a coordinate
produces a new, legitimate sample — and both are kept.

The eight ids in this bank are the eight positions of the smoke plan. The bank's line order is
write order (the anchors appended by the later invocations come last), not plan order; sort by
`id`, or by the `p…` field, to see the plan order.

`anchor_meta` carries the sample field `modality` (`text_only` | `image`) and every axis
value: `language`, `knowledge_domain`, `capability`, `system_prompt_mode`,
`conversation_type`, `response_style`, `output_format`, `difficulty`, `context_length`,
`input_condition`, `answer_mode`. An image-modality coordinate adds `visual_domain` and
carries `has_image: true` / `image_count: 1`; a text-only coordinate omits `visual_domain`
entirely — no `null` placeholder — and carries `has_image: false` / `image_count: 0`.

`data_source` is **per record**, not per run: an `ard_multi` record is one whose conversation
carries at least one image. Downstream routes on this key, so it is a closed vocabulary.

### `messages` — the shape invariant

Once an optional leading `system` message is stripped, `messages` must **start with a `user`
turn**, **end with a `user` turn**, and **alternate roles strictly**:

```
U   UAU   UAUAU   …          plus the system-prefixed form:  SU   SUAU   SUAUAU   …
```

Only odd turn counts can satisfy that, so `UAUAU` is *five* turns, not three. Anything else
is rejected before it is written, so a malformed record never reaches the bank. All eight
records here pass that gate. The `system` turn is the `system_prompt_mode` persona rendered
from `configs/prompts/system_prompt/<mode>.md`; `mode = none` means there is no such turn.

### `targets[0].output` — `content` and `reasoning`

This is the point of the dataset: the teacher's **reasoning** kept beside its own answer, in
two separate keys on purpose. `content` is the answer to the final user turn; `reasoning` is
the thinking chain (`str | null`). Thinking is not the answer, so it is never merged into
`content`. If a run cannot obtain the reasoning an anchor's configuration asked for, the
anchor is discarded and counted in the run's `manifest.json` — it is never written with an
empty value. All eight records here ran with thinking enabled, so every `reasoning` is a
non-empty string; a record whose teacher ran without thinking carries `"reasoning": null`.

### Images are addressed by `visual_domain`

A multimodal user turn looks like this:

```jsonc
{
  "role": "user",
  "content": [
    { "type": "image", "image": "images/urban_scenes/sample_10.jpg" },
    { "type": "text",  "text": "…the generated user question…" }
  ]
}
```

The path is relative to the run directory, so `outputs/<run_name>/images/<visual_domain>/…`
resolves directly; the run copies (and, unless `[images] convert = false`, converts) the picture
there. A picture filed under the anchor's own `visual_domain` coordinate is preferred, because it
is the one whose content the user vouched for. **This sample did not have one to prefer:** the
smoke plan's four image coordinates are `clothing`, `indoor_scenes`, `outdoor_scenes` and
`urban_scenes`, and the checked-in `examples/images` tree carries a subdirectory for none of them,
so all four were served by the **tree-wide fallback** — a reusable picture picked from the whole
image tree in a deterministic order. That substitution is declared, not hidden: the manifest reads
`fallback_visual_domains: [clothing, indoor_scenes, outdoor_scenes, urban_scenes]` and
`fallback_anchor_count: 4` against a `pool_candidate_count` of 10, and every `resolved_images` row
carries `fallback: true`. Re-running the documented command against `examples/images` reproduces
exactly that fallback.

## `manifest.sample.json`

A run also writes a manifest next to its bank. Read it to answer "did this run actually
produce healthy data?":

- `total_anchors`, `domains`, `languages`, `capabilities`, `system_prompt_modes`,
  `data_sources` — the produced mix of the **whole bank**;
- `status` — `"complete"` once the run finished; an unfinished run has no `manifest.json` yet
  and keeps only the intermediate `plan_identity.in_progress.json` next to its bank;
- `plan_identity` (**v2**) — the identity of the plan the run drew (`algorithm` / `version` /
  `sampling` / `ontology_sha256` / `seed` / `count` / `unit_total` / `plan_size` / `digest`); a
  resumed run compares it — and, more strictly, the coordinate at each existing id — before
  appending to an existing bank, so a record of it is what makes a bank auditable against a plan;
- `plan` — the plan's shape and readouts: `count` (the requested `N`, or `null` for one full
  round), `unit_total`, `full_cycles`, `last_cycle_size`, `planned_anchors`, `written_anchors`,
  `distinct_coordinates`, `coverage_ratio`, `density`, `smoke`;
- `generation.counters` — what happened to every anchor *this invocation* requested (`requested` /
  `succeeded` / `written`, plus `abandoned_by_reason` when non-zero). It is an invocation-level
  counter, not a bank total: here it reads `requested 2 / written 2` because the finishing
  invocation only had to fill the last gaps, while `total_anchors` is 8;
- `acceptance` — pointers to the run's structure readout, `results/coverage.json` and
  `results/coverage.md` (the acceptance report schema is `ard-acceptance-4`);
- `config` — the merged configuration the run actually used, with credential fields already
  redacted (`api_key = "***REDACTED***"`); `config.images.convert` is part of it;
- `images` — the addressing convention, the domains the bank references, and which of them were
  served from the tree-wide fallback (`resolved_visual_domains`, `domain_candidate_counts`,
  `pool_candidate_count`, `fallback_visual_domains`, `fallback_anchor_count`, plus a `fallback` flag
  on each `resolved_images` row). It is written by the invocation that finished the run but
  describes the **whole bank**, so here it lists all four `visual_domain` directories the bank
  references (see the record table below) — every one of them a fallback;
- `smoke` / `smoke_plan` — the self-declaration of a `--smoke` run.

### The eight sample records

| # | Record id | `data_source` | `modality` | `visual_domain` | Image | Messages (user turns) | Language | Knowledge domain | Capability | `system_prompt_mode` | `conversation_type` |
|---|-----------|---------------|-----------|-----------------|-------|:---:|----------|------------------|------------|----------------------|---------------------|
| 1 | `f384795e-c00000p00000` | `ard_text` | `text_only` | – | – | 5 (3) | 日本語 | data cleaning | debugging | `none` | troubleshooting |
| 2 | `f384795e-c00000p00001` | `ard_text` | `text_only` | – | – | 7 (4) | Español | beauty and ugliness | decision_analysis | `none` | source_review |
| 3 | `f384795e-c00000p00002` | `ard_text` | `text_only` | – | – | 7 (4) | 日本語 | refactoring plan | translation | `none` | constraint_update |
| 4 | `f384795e-c00000p00003` | `ard_text` | `text_only` | – | – | 7 (4) | 简体中文 | hypothesis formation | extraction | `none` | constraint_update |
| 5 | `f384795e-c00000p00004` | `ard_multi` | `image` | indoor_scenes | `images/indoor_scenes/sample_05.jpg` | 2 (1) | Español | mysticism in literature | debugging | `domain_style` | single_turn |
| 6 | `f384795e-c00000p00005` | `ard_multi` | `image` | urban_scenes | `images/urban_scenes/sample_10.jpg` | 4 (2) | English | schema mapping | qa | `minimal_persona` | clarification |
| 7 | `f384795e-c00000p00006` | `ard_multi` | `image` | clothing | `images/clothing/sample_06.jpg` | 5 (3) | 简体中文 | dependency management | data_analysis | `none` | troubleshooting |
| 8 | `f384795e-c00000p00007` | `ard_multi` | `image` | outdoor_scenes | `images/outdoor_scenes/sample_01.jpg` | 1 (1) | 简体中文 | workplace culture | decision_analysis | `none` | single_turn |

This mixture is simply what the smoke plan drew; nothing was steered. The language / domain /
capability mix of a full run is much broader.

**Normalised fields.** Four literal strings that identify the machine, the endpoint, the deployed
model or the run location were replaced so the sample carries no deployment details; everything
else is verbatim. The endpoint, the credentials and the model names are **placeholders**:

| What was replaced | Sample value | Why |
|-------|--------------|-----|
| `config.input_generator.api_base`, `config.target_model.api_base` | `https://your-endpoint.example/v1` | The real endpoint is deployment-specific |
| `config.input_generator.model_name`, `config.target_model.model_name`, and the per-record `input_generator_model` / `teacher_id` | `your-model-name` | The served generator/teacher model's name is deployment-specific |
| `images.image_dir` (an absolute path on the machine that ran it) | `examples/images` | The run location |
| `output_dir`, `config.output.directory`, `smoke_plan.run_name` (the pinned run directory) | `outputs/ard_dataset_<timestamp>` / `outputs/ard_dataset_<timestamp>_smoke` / `ard_dataset_<timestamp>_smoke` | The real value pinned the run directory on the machine that ran it |

Otherwise the sample is the run artifact **byte for byte**: the normalisation is a literal string
substitution, and reversing the substitutions above reproduces the run's own `anchor_bank.jsonl`
and `manifest.json` exactly. The two model placeholders are distinct on purpose, so the reversal
stays unambiguous.

`plan_identity` and `config.generation.seed` are kept verbatim: they identify the plan (and the
sampling order it was drawn in), not the deployment.

`config.*.api_key` already reads `***REDACTED***`: the pipeline redacts credentials before
writing any manifest or `config.toml`, so no key has ever been written to an output
directory. There is no separate sample config file in this directory on purpose — the
configuration template is `configs/config.override.sample.toml`, and duplicating it here
would create a second source of truth that could drift.

## Using these files

- Read `anchor_bank.sample.jsonl` to see the record schema without running anything.
- Validate your own tooling against a real `5.0.0` record.
- Run the smoke command above with `--image-dir examples/images` to see the same shape
  produced locally.

`images/` is the image tree the documented command points at. It ships four `visual_domain`
directories (`animals`, `everyday_objects`, `plants`, `vehicles`) as placeholders; the sample's own
smoke plan drew four *other* domains, so each of its image anchors was served from the tree-wide
fallback described above and the run still completed. Which domains a smoke run needs depends on
its seed, because it takes round 0's shuffled order; **a full run needs all 21 visual domains** —
supply your own image directory and pass it with `--image-dir`.

> The ten JPEGs are **placeholders**, not representatives of their domain, and their
> provenance is not documented in this repository. See
> [images/README.md](images/README.md) before using or redistributing them.

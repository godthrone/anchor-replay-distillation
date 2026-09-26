# Examples

Everything in this directory is **real pipeline output**, checked in so you can see what ARD
produces without spending endpoint time. `anchor_bank.sample.jsonl` is the complete anchor
bank of one real `--smoke` run — every line is that run's own output, copied byte for byte,
nothing trimmed, reordered or re-worded. `manifest.sample.json` is the manifest that same run
wrote, with a handful of environment-specific fields normalised (listed below).

| Path | What it is |
|------|------------|
| `anchor_bank.sample.jsonl` | The 8 anchors one real `--smoke` run produced, one JSON object per line (`schema_version 4.0.0`) |
| `manifest.sample.json` | The manifest that run wrote next to its bank (smoke declaration, `plan_identity` and `status` included) |
| `images/` | 10 small placeholder JPEGs, one subdirectory per `visual_domain` |
| `images/README.md` | What those pictures are (placeholders) and the addressing convention |

## What the run was

```bash
# from the repository root
./run.sh --config configs/config.toml --smoke --image-dir examples/images
```

`--smoke` is the standard construction rule at reduced scale: **8 restricted blocks** (4 text
blocks + 4 image blocks, evenly spaced over the block enumeration, first and last included).
That is 8 plan slots but only **6 distinct `RestrictedBlock`s** — two blocks appear once per
modality, the same 891⊂935 subset identity behind the 1,826-anchor count in the main README.
Those blocks are a subset of the blocks the full plan enumerates (**8/8 block-level coverage**),
but their coordinates are re-rotated at smoke scale, so the 8 slots are **not rows of the
1,826-anchor plan** (only 1/8 coincide coordinate-wise): a smoke artifact is **not a subset** of
the full plan's rows. The run directory name carried the `_smoke` suffix, the log carried a
WARNING, and the manifest below declares `smoke: true`. Generation is stochastic, so re-running
that command reproduces the same *shape* — 8 planned anchors, 4 per modality,
`schema_version 4.0.0`, the same addressing — but neither the same questions and answers nor,
if the endpoint hiccups the same way, the same number of written records.

**Provenance — which build produced these files.** Both sample files are the artifacts of one
run of this repository at commit `13a13a2e5fcc633007ba4a12cfe6ff8bb5ebf426`, made on 2026-09-27
against a real deployment endpoint (credentials taken from the gitignored
`.local/config.override.toml`; the run held the metric readout wired to a real target set). The
run's `manifest.json` records the identity of the plan it drew:

```json
"plan_identity": {
  "algorithm": "sha256",
  "version": 1,
  "plan_size": 8,
  "digest": "3f5424395d78d8a16bb87a41163043d387569880bbf3032755fcde74beb50d06"
}
```

`plan_identity` describes the *plan*, not the endpoint or the deployment, so it is kept in the
sample on purpose: it is what lets a later run prove it is appending to the same plan rather
than to a different one.

**Why eight records?** The run planned **8** anchors and wrote **all 8** — no anchor was
abandoned: `generation.counters` is `requested 8 / succeeded 8 / written 8`, with no
`abandoned_by_reason` key at all. The bank therefore holds 4 text-only + 4 image = **8**
records. (An earlier sample of this directory came from the run at commit `12690b3`, which
lost one `animals` image anchor to a transient `transport_error` and so held 7 records with
`abandoned_by_reason` = `{ "transport_error": 1 }`; this refresh met no such failure. The
record count is a property of *that one run*, not of the smoke scale — a smoke run may write
between 0 and 8 records, and the manifest is where the difference is declared.)

| # | Record id | `data_source` | `modality` | `visual_domain` | Image | Messages (user turns) | Language | Knowledge domain | Capability | `system_prompt_mode` | `conversation_type` |
|---|-----------|---------------|-----------|-----------------|-------|:---:|----------|------------------|------------|----------------------|---------------------|
| 1 | `anchor_3a5bb496d89d3b1c` | `ard_text` | `text_only` | – | – | 1 (1) | Español | origin of the universe | qa | `none` | single_turn |
| 2 | `anchor_a2017869ecd6d9ce` | `ard_multi` | `image` | everyday_objects | `images/everyday_objects/sample_03.jpg` | 1 (1) | 简体中文 | deep ocean unknowns | qa | `none` | single_turn |
| 3 | `anchor_15ec1474fbc89ee0` | `ard_multi` | `image` | animals | `images/animals/sample_06.jpg` | 2 (1) | 日本語 | planetary habitability | reasoning | `detailed_persona` | single_turn |
| 4 | `anchor_6e94bf911182bfa3` | `ard_multi` | `image` | plants | `images/plants/sample_08.jpg` | 2 (1) | 日本語 | complex systems emergence | debugging | `domain_style` | single_turn |
| 5 | `anchor_9291b177c120deba` | `ard_text` | `text_only` | – | – | 8 (4) | 简体中文 | dark matter hypotheses | structured_response | `task_constraint` | tool_assisted |
| 6 | `anchor_07d1410cfb3491f5` | `ard_text` | `text_only` | – | – | 6 (3) | Español | origin of life | reasoning | `detailed_persona` | iterative_revision |
| 7 | `anchor_94c11400295c9edc` | `ard_multi` | `image` | vehicles | `images/vehicles/sample_10.jpg` | 8 (4) | 日本語 | consciousness research | structured_response | `task_constraint` | tool_assisted |
| 8 | `anchor_07715d0d58a6eea7` | `ard_text` | `text_only` | – | – | 6 (3) | 日本語 | limits of scientific measurement | debugging | `domain_style` | iterative_revision |

"Messages" counts every entry of `messages`, including an optional leading `system` turn, so
it is `2n − 1` (or `2n`) for `n` user turns — see the shape invariant below. This mixture is
simply what the smoke plan drew; nothing was steered. The language / domain / capability mix
of a full run is much broader.

All eight records ran with `enable_thinking = true`, so every `reasoning` here is a non-empty
string. A record whose teacher ran without thinking carries `"reasoning": null` — never `""`.

## The record schema

Each line of `anchor_bank.sample.jsonl` is one anchor:

```jsonc
{
  "id": "anchor_3a5bb496d89d3b1c", // sha256 over the sampled axes
  "source": "ard",                 // dataset tag of the ARD anchor-bank format
  "data_source": "ard_text",       // controlled vocabulary: ard_text | ard_multi
  "schema_version": "4.0.0",       // the record format version — same for every record
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
empty value.

### Images are addressed by `visual_domain`

A multimodal user turn looks like this:

```jsonc
{
  "role": "user",
  "content": [
    { "type": "image", "image": "images/everyday_objects/sample_03.jpg" },
    { "type": "text",  "text": "…the generated user question…" }
  ]
}
```

The path is relative to the run directory, so `outputs/<run_name>/images/<visual_domain>/…`
resolves directly; the run copies (and, unless `[images] convert = false`, converts) the picture there.
It was drawn from `<image_dir>/<visual_domain>/<file>` — the anchor's own `visual_domain`
coordinate, never a flat pool, because a flat pool cannot guarantee that the image matches
the label attached to it. The four files the sample bank references exist in this directory at
exactly those paths, so pointing your own run at `--image-dir examples/images` resolves them.

## `manifest.sample.json`

A run also writes a manifest next to its bank. Read it to answer "did this run actually
produce healthy data?":

- `total_anchors`, `domains`, `languages`, `capabilities`, `system_prompt_modes`,
  `data_sources` — the produced mix;
- `generation.counters` — what happened to every requested anchor (`requested` / `succeeded`
  / `written`, plus abandonment and rejection counters when non-zero);
- `acceptance` — pointers to `results/coverage.{json,md}`, plus `metric_readout` and `q95`;
- `plan_identity` — the identity of the plan the run drew (`algorithm` / `version` /
  `plan_size` / `digest`); a resumed run compares it before appending to an existing bank, so a
  record of it is what makes a bank auditable against a plan;
- `config` — the merged configuration the run actually used, with credential fields already
  redacted (`api_key = "***REDACTED***"`); `config.images.convert` is part of it;
- `images` — the addressing convention, the domains it resolved and any it skipped;
- `smoke` / `smoke_plan` — the self-declaration of a `--smoke` run;
- `status` — `"complete"` once the run finished; an unfinished run has no `manifest.json` yet
  and keeps only the intermediate `plan_identity.in_progress.json` next to its bank.

**Normalised fields.** Thirteen fields that identify the machine, the endpoint, the deployed
model or the run location were replaced so the sample carries no deployment details; everything
else is verbatim. The endpoint, the credential and the model name are **placeholders**
throughout this directory:

| Field | Sample value | Why |
|-------|--------------|-----|
| `config.input_generator.api_base` | `https://your-endpoint.example/v1` | The real endpoint is deployment-specific |
| `config.target_model.api_base` | `https://your-endpoint.example/v1` | Same |
| `config.coverage.embedding.api_base` | `https://your-endpoint.example/v1` | Same |
| `config.input_generator.model_name` | `your-model-name` | The served model's name is deployment-specific |
| `config.target_model.model_name` | `your-model-name` | Same |
| `config.coverage.embedding.model` | `your-model-name` | Same (the embedder that measured `q95`) |
| `input_generator_model` (all 8 records) | `your-model-name` | Same, per record |
| `teacher_id` (all 8 records) | `your-model-name` | Same, per record |
| `output_dir` | `outputs/ard_dataset_<timestamp>_smoke` | Run location |
| `config.output.directory` | `outputs/ard_dataset_<timestamp>` | The real value pinned the run directory on the machine that ran it |
| `smoke_plan.run_name` | `ard_dataset_<timestamp>_smoke` | The real value was the run name on the machine that ran it |
| `images.image_dir` | `examples/images` | The real value is an absolute path on the machine that ran it |
| `config.coverage.target_set_path` | `/path/to/your/target_set.jsonl` | The real value is an absolute path on the machine that ran it; a placeholder keeps the wiring visible |

Otherwise the sample is the run artifact **byte for byte**: the normalisation is a literal
string substitution, and reversing the substitutions above reproduces the run's own
`anchor_bank.jsonl` / `manifest.json` exactly.

`plan_identity` and `config.generation.seed` are kept verbatim: they identify the plan (and the
sampling order it was drawn in), not the deployment. This run had the metric readout wired —
hence `config.coverage.enabled = true` with a target set and a 4096-dimension embedder — so its
`acceptance.metric_readout` is `true` and `q95` is a real number, not `null`. A run with no
target set configured reports `metric_readout: false` / `q95: null` and says so in a WARNING.

`config.*.api_key` already reads `***REDACTED***`: the pipeline redacts credentials before
writing any manifest or `config.toml`, so no key has ever been written to an output
directory. There is no separate sample config file in this directory on purpose — the
configuration template is `configs/config.override.sample.toml`, and duplicating it here
would create a second source of truth that could drift.

## Using these files

- Read `anchor_bank.sample.jsonl` to see the record schema without running anything.
- Validate your own tooling against a real `4.0.0` record.
- Run the smoke command above with `--image-dir examples/images` to see the same shape
  produced locally.

`images/` covers exactly the four `visual_domain` directories a `--smoke` run requires
(`everyday_objects`, `animals`, `plants`, `vehicles`). **A full run needs all 21 visual
domains** — supply your own image directory and pass it with `--image-dir`.

> The ten JPEGs are **placeholders**, not representatives of their domain, and their
> provenance is not documented in this repository. See
> [images/README.md](images/README.md) before using or redistributing them.

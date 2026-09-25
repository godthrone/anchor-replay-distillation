# Examples

Everything in this directory is **real pipeline output**, checked in so you can see what ARD
produces without spending endpoint time. `anchor_bank.sample.jsonl` is the complete anchor
bank of one real `--smoke` run — every line is that run's own output, copied byte for byte,
nothing trimmed, reordered or re-worded. `manifest.sample.json` is the manifest that same run
wrote, with a handful of environment-specific fields normalised (listed below).

| Path | What it is |
|------|------------|
| `anchor_bank.sample.jsonl` | The 8 anchors of one real `--smoke` run, one JSON object per line (`schema_version 4.0.0`) |
| `manifest.sample.json` | The manifest that run wrote next to its bank (smoke declaration included) |
| `images/` | 10 small placeholder JPEGs, one subdirectory per `visual_domain` |
| `images/README.md` | What those pictures are (placeholders) and the addressing convention |

## What the run was

```bash
# from the repository root
bash run.sh --config configs/config.toml --smoke --image-dir examples/images
```

`--smoke` is the standard construction rule at reduced scale: **8 of the 1,826 anchors**
(4 text-only + 4 image, evenly spaced over the block enumeration and including both ends).
The run directory name carried the `_smoke` suffix, the log carried a WARNING, and the
manifest below declares `smoke: true`. Generation is stochastic, so re-running that command
reproduces the same *shape* — 8 anchors, 4 per modality, `schema_version 4.0.0`, the same
addressing — but not the same questions or answers.

| # | Record id | `data_source` | `modality` | `visual_domain` | Image | Messages (user turns) | Language | Knowledge domain | Capability | `system_prompt_mode` | `conversation_type` |
|---|-----------|---------------|-----------|-----------------|-------|:---:|----------|------------------|------------|----------------------|---------------------|
| 1 | `anchor_255e07584d699720` | `ard_text` | `text_only` | – | – | 1 (1) | 简体中文 | origin of the universe | qa | `none` | single_turn |
| 2 | `anchor_13df63aeb8db8470` | `ard_text` | `text_only` | – | – | 8 (4) | English | dark matter hypotheses | structured_response | `task_constraint` | tool_assisted |
| 3 | `anchor_0dc58134943a6cb7` | `ard_multi` | `image` | everyday_objects | `images/everyday_objects/sample_02.jpg` | 1 (1) | English | deep ocean unknowns | qa | `none` | single_turn |
| 4 | `anchor_57b9053f9981fddc` | `ard_multi` | `image` | animals | `images/animals/sample_04.jpg` | 2 (1) | English | planetary habitability | reasoning | `detailed_persona` | single_turn |
| 5 | `anchor_6e94bf911182bfa3` | `ard_multi` | `image` | plants | `images/plants/sample_08.jpg` | 2 (1) | 日本語 | complex systems emergence | debugging | `domain_style` | single_turn |
| 6 | `anchor_94c11400295c9edc` | `ard_multi` | `image` | vehicles | `images/vehicles/sample_09.jpg` | 8 (4) | 日本語 | consciousness research | structured_response | `task_constraint` | tool_assisted |
| 7 | `anchor_9ad04614f65c1e22` | `ard_text` | `text_only` | – | – | 6 (3) | 日本語 | origin of life | reasoning | `detailed_persona` | iterative_revision |
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
  "id": "anchor_255e07584d699720", // sha256 over the sampled axes
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
    { "type": "image", "image": "images/everyday_objects/sample_02.jpg" },
    { "type": "text",  "text": "…the generated user question…" }
  ]
}
```

The path is relative to the run directory, so `outputs/<run_name>/images/<visual_domain>/…`
resolves directly; the run copies (and, unless `--no-convert`, converts) the picture there.
It was drawn from `<image_dir>/<visual_domain>/<file>` — the anchor's own `visual_domain`
coordinate, never a flat pool, because a flat pool cannot guarantee that the image matches
the label attached to it. The four referenced files exist in this directory at exactly those
paths, so pointing your own run at `--image-dir examples/images` resolves them.

## `manifest.sample.json`

A run also writes a manifest next to its bank. Read it to answer "did this run actually
produce healthy data?":

- `total_anchors`, `domains`, `languages`, `capabilities`, `system_prompt_modes`,
  `data_sources` — the produced mix;
- `generation.counters` — what happened to every requested anchor (`requested` / `succeeded`
  / `written`, plus abandonment and rejection counters when non-zero);
- `acceptance` — pointers to `results/coverage.{json,md}`, plus `metric_readout` and `q95`;
- `images` — the addressing convention, the domains it resolved and any it skipped;
- `smoke` / `smoke_plan` — the self-declaration of a `--smoke` run.

**Normalised fields.** Five fields that identify the machine or the run were replaced so the
sample carries no deployment details; everything else is verbatim:

| Field | Sample value | Why |
|-------|--------------|-----|
| `config.input_generator.api_base` | `https://your-endpoint.example/v1` | The real endpoint is deployment-specific |
| `config.target_model.api_base` | `https://your-endpoint.example/v1` | Same |
| `config.output.directory` | `""` | Run location; the empty default means "auto-generate under `outputs/`" |
| `output_dir` | `outputs/ard_dataset_<timestamp>_smoke` | Run location |
| `images.image_dir` | `examples/images` | The real value is an absolute path on the machine that ran it |

`config.*.api_key` already reads `***REDACTED***`: the pipeline redacts credentials before
writing any manifest or `config.json`, so no key has ever been written to an output
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

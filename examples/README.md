# Examples

Everything in this directory is **real pipeline output** — nothing here is
hand-written or re-worded. `anchor_bank.sample.jsonl` is a representative subset
of the records written by two real v3.0.0 runs (every line is copied from those
banks byte for byte), and `manifest.sample.json` is the manifest format each run
writes next to its own bank. They are checked in so you can understand what ARD
produces without spending GPU time or needing an API endpoint.

## Contents

| Path | What it is |
|------|------------|
| `anchor_bank.sample.jsonl` | 6 real anchors, one JSON object per line |
| `manifest.sample.json` | The manifest a real run writes next to the anchor bank |
| `images/` | 10 small JPEGs (400 px wide); the sample records reference `sample_04` / `sample_05` / `sample_10` |

The six records are **verbatim lines** taken from **two real v3.0.0 runs**: the
first run produced 60 text anchors, the second produced 40 multimodal anchors;
this sample keeps three records from each bank (a selection made for this
directory, not something a single run performs), picked to cover both
`data_source` buckets, all four ontology languages and both legal `reasoning`
forms:

| # | Record id | `data_source` | Language | Shape | Image | `reasoning` | Knowledge domain | Capability | System prompt |
|---|-----------|---------------|----------|-------|:---:|-------------|------------------|------------|---------------|
| 1 | `anchor_a99c98644ad956bd` | `ard_text` | 日本語 | `U` | – | `str` | art_aesthetics | translation | `none` |
| 2 | `anchor_d72eae87804663f4` | `ard_text` | 简体中文 | `SU` | – | `str` | esoterica_belief_systems | uncertainty_handling | `task_constraint` |
| 3 | `anchor_776f2287038b7d0b` | `ard_text` | English | `SU` | – | `null` | medicine_health | decision_analysis | `domain_style` |
| 4 | `anchor_110374c8f9fd3138` | `ard_multi` | 日本語 | `U` | 1 | `str` | future_speculation | translation | `none` |
| 5 | `anchor_da3b5e8e8c0a8e9e` | `ard_multi` | Español | `SU` | 1 | `str` | software_engineering | translation | `task_constraint` |
| 6 | `anchor_bffd29da463224ed` | `ard_multi` | English | `SU` | 1 | `str` | agent_tool_use | decision_analysis | `domain_style` |

`U` = one user turn; `SU` = a `system` persona turn followed by the user turn.
This language / domain mix is simply **what those two runs happened to draw** —
it was not steered, and the two banks were generated separately (one without
`--image-dir`, one with it). The ontology holds far more combinations than six
records can show (see [How the sample was drawn](#how-the-sample-was-drawn)).
The three `ard_text` records carry **no** `has_image` key in `anchor_meta` at
all, while the three `ard_multi` records carry `has_image` and `image_count` —
a second, independent signal that these are two runs, not one bank.

## The record schema

Each line of `anchor_bank.sample.jsonl` is one anchor:

```jsonc
{
  "id": "anchor_a99c98644ad956bd",   // sha256 over the 5 sampled dimensions (see below)
  "source": "ard",                   // dataset tag, matches the ARD anchor-bank format
  "data_source": "ard_text",         // controlled vocabulary: ard_text | ard_multi
  "schema_version": "3.0.0",         // the record format version — same for every record
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
  "anchor_meta": {
    "language": "日本語",
    "knowledge_domain": "art_aesthetics",
    "capability": "translation",
    "conversation_type": "constraint_update_4_turn",  // a style label, not a turn count
    "system_prompt_presence": "none",                 // none | present
    "system_prompt_style": "none",                    // none + 4 persona styles
    "system_prompt_mode": "none",                     // the 5th sampled dimension
    "has_image": true,                                // multimodal records only
    "image_count": 1                                  // multimodal records only
  },
  "input_generator_model": "…",       // the generator that produced the user messages
  "teacher_id": "…"                   // the teacher (target) model that answered
}
```

`data_source` is **per record**, not per run: an `ard_multi` record is one whose
conversation carries at least one image. (A single run writes only one value,
but this sample draws from two runs, and a resumed run appends to an existing
bank — so a file, and this sample, can hold both kinds side by side.) Downstream
routes on this key, so it is a closed vocabulary rather than a free-form string.

### `messages` — the shape invariant

Once an optional leading `system` message is stripped, `messages` must **start
with a `user` turn**, **end with a `user` turn**, and **alternate roles
strictly**. Because only odd turn counts can satisfy that, the legal conversation
shapes are:

```
U   UAU   UAUAU   UAUAUAU   …
```

plus the `system`-prefixed form (at most one `system`, and only at position 0 —
the array is the single source of truth for the persona prompt, which is what
the `SU` entries in the table above are):

```
SU   SUAU   SUAUAU   …
```

Anything else (`UAUAU` is *not* three turns — it is five, ending on a `user` turn
with no answer) is rejected before it is written: the writer validates the shape
and refuses to persist a malformed record, logging it instead. All six records
here pass that gate.

A multimodal user turn looks like this:

```jsonc
{
  "role": "user",
  "content": [
    { "type": "image", "image": "images/sample_04.jpg" },   // relative path
    { "type": "text",  "text": "…the generated user question…" }
  ]
}
```

Note that images are referenced by **relative path**, not as inline base64, so
the JSONL stays small. The paths are relative to the output directory, and the
files are copied there during the run. The `images/` in this directory are the
same files, so the paths above resolve if you point your own run at
`--image-dir examples/images`.

### `targets[0].output` — `content` and `reasoning`

This is the point of the dataset: the teacher's **reasoning text**, kept beside
its own answer. The two live in separate keys on purpose:

- `content` — the teacher's answer to the final user turn.
- `reasoning` — the teacher's thinking chain, a plain `str | null`. It is `null`
  (never `""`) when the teacher did not think, i.e. when the target model ran
  with `enable_thinking = false`; record #3 above is such a case, records #1, #2
  and #4–#6 carry a non-empty string. Thinking is not the answer, so it is never
  merged into `content`.

If a run cannot obtain `reasoning` for an anchor whose configuration asked for
it, **it does not silently write an empty value** — the anchor is discarded and
counted (a real run's `<output_dir>/manifest.json` reports it under
`generation.failures`; the sample manifest here omits that block, see below).

## `manifest.sample.json`

A real run also writes a manifest next to the anchor bank. Read it to answer
"did this run actually produce healthy data?":

- `total_anchors`, `domains`, `languages`, `capabilities`,
  `system_prompt_modes`, `data_sources` — the produced mix, tallied with the
  same breakdown function the run itself uses.
- `generation.counters` — what happened to every requested anchor:
  `requested` / `succeeded` / `abandoned_total` / `abandoned_by_reason` /
  `written` (plus `rejected_invalid_shape`, `rejected_invalid_data_source`,
  `duplicate_ids` and `backpressure_events` when non-zero).
- `generation.failures` — process-level failure counters, e.g. reasoning /
  empty-content failures by reason (`responses`, `empty_content`,
  `reasoning_only_responses`, `truncated_empty`) — the teacher spent its whole
  token budget on reasoning and produced no answer.

The last two are documented here because they are what a reader of a **real**
manifest should look for — they are **not present** in this sample (see below).

> **Zero-valued counters are dropped**, so a perfectly healthy run may omit
> `generation` entirely. **An empty `generation: {}` is the anomaly, not the
> absence of the key.**

The sample's field set is a real manifest's, with **two deliberate omissions**:
this file lists only the reproducible composition of the six records above and
drops every deployment-specific or run-specific part.

1. The merged runtime `config` (endpoints and model names) is left out so the
   sample carries no deployment details.
2. The `generation` block is **absent** — it is a *per-run execution account*,
   and the six records here come from two different runs, so no single run's
   counters describe this file. The sample is a hand-picked subset of two banks,
   not the verbatim manifest of one run: do not read the absence as "the run was
   unhealthy". A real manifest for a single run does carry `generation` whenever
   any counter is non-zero.

Two numbers are scaled to this 6-record file, everything else is verbatim: the
composition breakdown at the top (`total_anchors` and the five tallies) is
recomputed over the six records here, and `output_dir` is the placeholder a real
run replaces with its own output directory. The `generation` counters of the two
underlying runs are intentionally **not** carried over — they account for runs
that produced 100 records in total, not for this 6-record subset.


## How the sample was drawn

```bash
# from the repository root
bash run.sh --config configs/config.toml \
    --override .local/config.override.toml \
    --image-dir examples/images
```

### The combination space

An anchor is sampled on **5 dimensions** from `ontology/anchor_ontology.json`:
`language` × `knowledge_domain` × `capability` × `conversation_type` ×
`system_prompt_mode` (the system prompt became a sampling dimension in v3.0.0).
For the ontology shipped in this release that is:

**4 × 18 × 20 × 7 × 5 = 50,400 combinations** — and **100,800** when the
multimodal form is counted separately for every combination (`ard_multi` /
`ard_text`, i.e. whether the anchor carries an image).

The sampler spreads `target_count` as far as it can across that space, so a run
asking for 100 anchors and a run asking for 6 anchors draw **different**
subsets: the budget decides where the sample lands, not just how much of it you
see.

### `seed` — what it does and does not pin

`seed` is optional in `[generation]`:

- **Omitted (the default) → every run draws a fresh seed** from the system
  random source, so two runs of the same config sample independently.
- **Set to an integer → that run's sampling order is pinned.** The value that
  was actually used is recorded in `<output_dir>/config.json`, so a run can
  always be traced back to the draw it used.

An anchor **id**, by contrast, is a deterministic function of the sampled
dimensions — it carries no seed and no run identity:

```
id = "anchor_" + sha256("<language>|<knowledge_domain>|<capability>|<conversation_type>|<system_prompt_mode>")[:16]
```

So the same combination always hashes to the same id, whenever it is drawn, and
two runs can legitimately contain records with the same id but different
conversations (the teacher's answers differ from run to run). The ids in the
table above were re-derived from their own `anchor_meta` while writing this
README, so they are a working example of that formula.

### ⚠️ `conversation_type` is an ontology label, not the actual turn count

Do not read the number in `anchor_meta.conversation_type` as the number of
turns. It is a **sampling dimension** drawn from the ontology to describe a
conversational *style* (e.g. `clarification_2_turn`, `constraint_update_4_turn`).
The actual conversation length is decided separately by the turn-count
distribution and is visible only in `messages` — for example record #1 above
carries `conversation_type = "constraint_update_4_turn"` but its `messages`
shape is `U` (a single turn).

If you need the real turn count, count the `user` turns in `messages`.



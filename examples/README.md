# Examples

Everything in this directory is **real pipeline output** — nothing here is
hand-written or trimmed. It is checked in so you can understand what ARD
produces without spending GPU time or needing an API endpoint.

## Contents

| Path | What it is |
|------|------------|
| `anchor_bank.sample.jsonl` | 6 real anchors, one JSON object per line |
| `manifest.sample.json` | The manifest a real run writes next to the anchor bank |
| `images/` | 10 small JPEGs (400×267) — the exact images used to produce the sample records |

The sample covers both conversation shapes and four languages:

| # | Shape | Language | Images | Knowledge domain |
|---|-------|----------|:---:|------------------|
| 1 | `U` | 日本語 | 1 | art_aesthetics |
| 2 | `U` | 日本語 | 1 | medicine_health |
| 3 | `U` | Español | 1 | software_engineering |
| 4 | `UAU` | 日本語 | 2 | religion_myth_folklore |
| 5 | `UAU` | Español | 2 | finance_economics |
| 6 | `UAU` | 简体中文 | 2 | esoterica_belief_systems |

`U` = single-turn anchor (one user question, one teacher answer).
`UAU` = three-turn anchor (user → assistant → user, with the teacher answering
the final user turn).

## The record schema

Each line of `anchor_bank.sample.jsonl` is one anchor:

```jsonc
{
  "id": "anchor_63df9a95c23c7e30",   // deterministic: sha256 of the 4 sampled dimensions
  "source": "ard",                   // dataset tag, matches the graspo anchor-bank format
  "messages": [ /* the conversation, see below */ ],
  "targets": [
    {
      "id": "primary",
      "output": {
        "content": "…the teacher's answer…",
        "logprobs": {
          "token_ids": ["ユーザー", "は", "画像", "に", "写", …],
          "log_probs": [-0.00811, -0.00927, -0.00010, …]
        }
      }
    }
  ],
  "anchor_meta": {
    "language": "日本語",
    "knowledge_domain": "art_aesthetics",
    "capability": "translation",
    "conversation_type": "…",
    "has_image": true,
    "image_count": 1
  },
  "teacher_id": "latest"             // the teacher model name you configured
}
```

### `messages` — the shape invariant

`messages` obeys a hard invariant: it **starts with a `user` turn**, **ends
with a `user` turn**, and roles **strictly alternate**. Because only odd turn
counts can satisfy that, the legal shapes are:

```
U   UAU   UAUAU   UAUAUAU   …
```

Anything else (`UAUAU` is *not* three turns — it is five, ending on a `user` turn
with no answer) is rejected before it is written: the writer validates the shape
and refuses to persist a malformed record, logging it instead.

A multimodal user turn looks like this:

```jsonc
{
  "role": "user",
  "content": [
    { "type": "image", "image": "images/sample_02.jpg" },   // relative path
    { "type": "text",  "text": "…the generated user question…" }
  ]
}
```

Note that images are referenced by **relative path**, not as inline base64, so
the JSONL stays small. The paths are relative to the output directory, and the
files are copied there during the run. The `images/` in this directory are the
same files, so the paths above resolve if you point your own run at
`--image-dir examples/images`.

### `targets[0].output.logprobs`

This is the point of the dataset: the teacher's **token-level** log-probabilities
for its own answer, which downstream on-policy distillation uses as the
supervision signal.

- `token_ids` and `log_probs` are **parallel arrays of equal length**.
- `token_ids` holds **string tokens** (e.g. `"ユーザー"`), not integer IDs — that
  is what the OpenAI-compatible API returns. Integer IDs are not available from
  every server, and string tokens are portable across tokenizers.
- `log_probs` are negative floats, one per token of `content`.

If a run cannot obtain log-probs for an anchor, it **does not silently write an
empty array** — it raises and the anchor is discarded and counted (see
`generation.failures` in the manifest).

## `manifest.sample.json`

A real run also writes a manifest next to the anchor bank. Read it to answer
"did this run actually produce healthy data?":

- `total_anchors`, `domains`, `languages`, `capabilities` — the produced mix.
- `generation.counters` — what happened to every requested anchor:
  `requested` / `succeeded` / `abandoned_total` / `abandoned_by_reason` /
  `written` (plus `rejected_invalid_shape` and `duplicate_ids` when non-zero).
- `generation.failures` — process-level failure counters, e.g. log-prob
  failures by reason, and `empty_content` / `truncated_empty` (the teacher spent
  its whole token budget on reasoning and produced no answer).

> **Zero-valued counters are dropped**, so a perfectly healthy run may omit
> `generation` entirely. **An empty `generation: {}` is the anomaly, not the
> absence of the key.** The sample file was scaled down to the 6 records here;
> the numbers come from a real 100-anchor run.

The file's field set is identical to a real manifest's, with **one deliberate
omission**: a real manifest also embeds the fully merged runtime config under
`config`, which contains your endpoint and model names. That section is left out
here so the sample carries no deployment details. Everything else — including
the `generation` health counters — is verbatim.


## Reproducing these samples

```bash
# from the repository root
bash run.sh --config configs/config.toml \
    --override .local/config.override.toml \
    --image-dir examples/images
```

**Sampling is deterministic, but the selection depends on how many anchors you
ask for.** The farthest-point sampler spreads the requested budget across the
whole ontology, so a run with `target_count = 100` picks a different set of
combinations than a run with `target_count = 6` — and both are reproducible.

Concretely: the six records here came from a 100-anchor run
(`seed = 42`, `max_turns = 3`, `max_turns_with_image = 3`, `concurrency = 4`).
To get the *same* six back you must reproduce that run and then look at the
ids `anchor_63df9a95c23c7e30`, `anchor_7529dd0bdb69bed4`,
`anchor_707a3ca861b71d36`, `anchor_30093dde3d79e005`,
`anchor_05dcfa350411fd72`, `anchor_93600c7ec174bfb8`
(they are at positions 1, 43, 19, 83, 69, 98 of that run).

If you only want a fast look at the format, run a small `target_count` with the
same `seed` — you will get valid anchors of the same shape, just a different
subset of the ontology.

The one thing that is stable for a fixed `(seed, target_count, max_turns)` tuple
is the anchor **ids**, because an id is a hash of the sampled dimensions:

```
id = "anchor_" + sha256("<language>|<knowledge_domain>|<capability>|<conversation_type>")[:16]
```

### ⚠️ `conversation_type` is an ontology label, not the actual turn count

Do not read the number in `anchor_meta.conversation_type` as the number of
turns. It is a **sampling dimension** drawn from the ontology to describe a
conversational *style* (e.g. `clarification_2_turn`, `constraint_update_4_turn`).
The actual conversation length is decided separately by the turn-count
distribution and is visible only in `messages` — for example record #1 above
carries `conversation_type = "constraint_update_4_turn"` but its `messages`
shape is `U` (a single turn).

If you need the real turn count, count the `user` turns in `messages`.



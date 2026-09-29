# Prompt wording files

Generator-facing prompt **wording** lives here as data, not in Python string
literals. The coordinate vocabulary (which modes/values exist) stays in the
ontology (`ontology/anchor_ontology.v4.json`); this directory holds the text.

Two families, one contract ("wording is data, coordinates are the ontology's"):

- `system_prompt/` — the wording that asks the input generator to **write a
  system message** (one file per `system_prompt_mode` value).
- `axis_instruction/` — the wording that tells the input generator what the
  **user message it writes must also satisfy**, for the six axes the ontology
  labels `layer = "instruction"` (see `docs/architecture.md` §7.2): one JSON file
  per axis, keyed by that axis's value.

## Layout

- `system_prompt/<system_prompt_mode>.md` — one file per value of the ontology's
  `system_prompt_mode` axis, keyed by the axis value: the location the ontology
  declares in `wording_policy.prompt_wording_location_recommendation.target`
  (in `ontology/anchor_ontology.v4.json`, with the
  `<system_prompt_mode>` placeholder resolved to the axis value).
- `axis_instruction/<axis>.json` — one file per instruction axis:
  `response_style.json`, `output_format.json`, `difficulty.json`,
  `context_length.json`, `input_condition.json`, `answer_mode.json`.

## File format (one definition, no variations)

### `system_prompt/<mode>.md`

1. **Name** — exactly `<mode>.md`, where `<mode>` is a value of the
   `system_prompt_mode` axis. No other file name is read.
2. **Body** — UTF-8 Markdown, no front matter and no comment syntax. The
   **whole file content, stripped of surrounding whitespace, is the template**.
   Nothing is added or interpreted.
3. **Placeholders** — optional, and the only allowed ones are `{language}`,
   `{capability}`, `{domain}`, substituted with Python `str.format` semantics
   from the anchor's metadata (`language`, `knowledge_domain`, `capability`;
   defaults `English`, `general`, `qa`). Any other placeholder, or unbalanced
   braces, is a load error.
4. **`none.md`** — the exception: `none` is the absence case (no system message
   is generated), so its body documents that absence and is never rendered.
   Every file, this one included, must be non-empty.

### `axis_instruction/<axis>.json`

1. **Name** — exactly `<axis>.json`, where `<axis>` is one of the six
   instruction axes (`src/ard/core/axis_instruction.py`, `INSTRUCTION_AXES`).
   No other file name is read.
2. **Body** — UTF-8 JSON, one object:
   `{"axis": "<axis>", "values": {"<axis value>": "<instruction>"}}`. The
   `axis` field must name the same axis as the file name (a copy-pasted file
   cannot masquerade as another axis), and `values` must cover
   **every value of that ontology axis** — one sentence per value.
3. **Instruction text** — the sentence is used verbatim (stripped). It is
   inserted into the input generator's system message after the coordinate
   sentence, under the lead-in *"The user message you write must also satisfy
   this:"*. No placeholders are substituted here.
4. **No absence case** — unlike `system_prompt`, every value of every
   instruction axis carries a sentence. An empty instruction is a load error.

**Coverage** is a contract test, not a convention: `tests/backends/test_axis_instruction_loader.py`
asserts each file's keys equal the ontology's value list for that axis, so a new
ontology value cannot ship without wording.

## Failure semantics

For both families, a missing directory, a missing file, an empty file, or a
malformed document is a **hard error** naming the path and what was expected —
never a silent fallback to a built-in string (that would restore a second source
of truth). An anchor value that carries no instruction is also a hard error
naming the axis, the value and the file. Errors carry only paths and axis/mode
names, never credentials.


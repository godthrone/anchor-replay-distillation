# Prompt wording files

Generator-facing prompt **wording** lives here as data, not in Python string
literals. The coordinate vocabulary (which modes exist) stays in the ontology
(`ontology/anchor_ontology.v4.json`); this directory holds the text.

## Layout

- `system_prompt/<system_prompt_mode>.md` — one file per value of the ontology's
  `system_prompt_mode` axis, keyed by the axis value: the location the ontology
  declares in `wording_policy.prompt_wording_location_recommendation.target`
  (`ontology/anchor_ontology.v4.json:1319-1323`, with the
  `<system_prompt_mode>` placeholder resolved to the axis value).

## File format (one definition, no variations)

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

## Failure semantics

A missing directory, a missing mode file, or an empty file is a **hard error**
naming the path and what was expected — never a silent fallback to a built-in
string (that would restore a second source of truth). Errors carry only paths
and mode names, never credentials.

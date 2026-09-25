The absence case of the axis: an anchor whose `system_prompt_mode` is `none`
carries **no system message**, so there is no generator-facing wording and the
input generator is never asked to write one.

This file therefore holds no prompt text. It exists because the wording set
covers every value of the `system_prompt_mode` axis — one file per value — and
it states that absence in place of a template. It is never rendered:
`build_system_prompt_prompt` refuses `none` explicitly, and the generator call
that uses it returns `None` instead of building a prompt.

See `configs/prompts/README.md` for the file format shared by this directory.

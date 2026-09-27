# Sample images — placeholders, not domain content

These ten small JPEGs exist for **one** purpose: after a fresh `git clone`, a
`--smoke` run can be started without the user having to supply pictures first.
They are **placeholders**: a picture filed under `animals/` does **not** show
what the `animals` visual domain means, and none of these files is a curated
representative of its domain. Never use them as training content or as a
statement about what a domain contains.

## Layout (the addressing convention)

```
examples/images/
├── everyday_objects/   # placeholder images for the everyday_objects visual domain
├── animals/
├── plants/
└── vehicles/
```

One subdirectory per `visual_domain`; a run picks images from the subdirectory
named by each anchor's own `visual_domain` coordinate:

```
<image_dir>/<visual_domain>/<image file>
```

These four directories cover the four visual domains that the **checked-in sample's** `--smoke`
run required (`everyday_objects`, `animals`, `plants`, `vehicles`). Under the v5 cycle-shuffle
rule a smoke run takes the first four units of each modality from round 0's shuffled order, so
*which* domains it needs depends on the seed: these four are what that run drew, not a fixed set.
A full run needs all 21 visual domains: supply your own directory and pass it with `--image-dir`.
A missing domain is refused by default, listing the domains it could not find.

The pictures are used **only** as runnable placeholders. Their provenance is not
documented in this repository, so anyone repackaging the project for
redistribution must confirm the sample images' licensing independently — the
code, configuration and documentation carry no such dependency.

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

These four directories are the **pool** the checked-in sample's `--smoke` run drew on. Under the v5
cycle-shuffle rule a smoke run takes the first four units of each modality from round 0's shuffled
order, so *which* domains it needs depends on the seed: the checked-in sample drew `clothing`,
`indoor_scenes`, `outdoor_scenes` and `urban_scenes`, none of which has a directory here, so all
four of its image anchors were served by the tree-wide fallback — a picture reused from the whole
tree, recorded as `fallback_visual_domains` / `fallback_anchor_count` in `manifest.json` (see
`examples/README.md`).

A domain with no directory of its own is therefore **not** refused while the tree holds any usable
picture: the run reuses one and declares the reuse. Only a tree with no usable image at all leaves
a domain without a picture, and that state is refused by default, listing the domains it could not
find. A full run needs all 21 visual domains: supply your own directory and pass it with
`--image-dir`.

The pictures are used **only** as runnable placeholders. Their provenance is not
documented in this repository, so anyone repackaging the project for
redistribution must confirm the sample images' licensing independently — the
code, configuration and documentation carry no such dependency.

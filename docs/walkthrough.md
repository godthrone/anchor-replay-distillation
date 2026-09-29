# ARD walkthrough: the plain-language picture

This page is the plain-language layer between the two READMEs and the technical docs. Its job is
the half the READMEs keep short: **what the program does after you press Enter**, where the
results land, and the questions that come next. For the exact construction rule see
[docs/algorithm.md](algorithm.md) §2; for the metric definitions see
[docs/measurement.md](measurement.md); for the module boundaries behind these steps see
[docs/architecture.md](architecture.md) §2.

**What you have to do lives in exactly one place** — the
[Quick Start](../README.md#quick-start) section of `README.md` (copy the credentials template and
fill it in, then run one command; Docker is the only prerequisite for the default path). This page
never repeats them.

A few words used below:

- **anchor** — one generated example: a question and the answer it gets. It becomes one line of
  the dataset.
- **ontology** — `ontology/anchor_ontology.v4.json`, the file that lists every axis and the values
  it may take. It says *what may be combined*, not how to word it.
- **round** — one pass over every legal combination. How many anchors a round holds is counted
  from the ontology at run time, never written down in the code; `docs/algorithm.md` §1 carries
  the recompute command and the current count.
- **manifest** — `manifest.json`, the run's own record of what it built and how it went.

## What the machine does, step by step

1. **Reads the config and checks it before doing anything.** A misspelled field, an unknown field,
   a count that is not an integer `>= 1` — all of it is refused at load, with a message naming the
   field. A refused run creates no output directory at all.
2. **Reads the ontology.** Which task types exist, which knowledge domains and other values exist,
   and which combinations are legal. **Every count is enumerated at run time** — no total is
   hard-coded in the code, so editing the ontology changes the plan automatically.
3. **Builds the list of legal combinations — that is one round.** One unit is a legal combination
   plus whether it carries an image; the number of units is the round size `U`
   (`docs/algorithm.md` §1 recomputes it from your ontology).
4. **Draws the anchors you asked for.** A run of N anchors is a sequence over rounds: each round the
   whole unit list is shuffled and walked without replacement, then reshuffled for the next round.
   So within one round each unit appears exactly once, and across rounds a unit can return. Asking
   for more than a round is not an error — the plan just rolls into round 1, 2, …
5. **Fills in the remaining values for every combination** — knowledge domain, language, response
   style, difficulty, context length, and a visual domain for image units. The knowledge and visual
   domains rotate round by round; the other values are drawn from the run's random source. Turn
   counts are not a setting: each conversation type declares its own in the ontology
   (`docs/algorithm.md` §4).
6. **Gives every anchor an id.** An id is the plan *position*,
   `<run key>-c<round>p<position in round>` (for example `1a2b3c4d-c00000p00137`). It labels *where
   in the plan* the anchor sits, not what its content is, and it does not contain N — that is what
   lets a later run append without rewriting earlier records (`docs/algorithm.md` §3).
7. **Finds the picture each image anchor needs.** The preferred source is
   `<image_dir>/<visual_domain>/<file>`. When that domain has no picture of its own, the run does
   not drop the anchor: it takes one from all images under `--image-dir` and rotates through them by
   round. If the whole tree holds only one image, every image anchor gets that one. The reuse is
   recorded, never hidden.
8. **Generates the anchors.** For each anchor the settings are turned into an instruction; the
   input-generator model writes the user's request (with the picture, where there is one), and the
   target model writes the answer. Multi-turn conversations alternate that way. The last target
   answer is the one kept as the training target; when thinking is enabled for the target model, its
   reasoning is kept in a separate field.
9. **Writes as it goes.** Each finished anchor is appended immediately as one line of
   `anchor_bank.jsonl`, so an interrupted run keeps what it already produced. At the end the run
   writes `manifest.json`: the plan it used, what it covered, how the pictures were used, and the
   failure counters. The plan it used is named by a digest, not by the seed
   (`docs/algorithm.md` §6).
10. **Computes the readout.** *Coverage* asks how much of one round's grid the run touched (a ratio
    that saturates at 1.0). *Density* asks how many anchors there are per grid cell (`N / U`) and
    keeps growing with N. *`q95`* is a distance readout — the 95th percentile of each target point's
    distance to its nearest anchor — and smaller means the anchors land closer to the target set.
    The structure readout needs no model call; `q95` needs an embeddings endpoint and a target set,
    and without them the run announces the gap in a WARNING instead of pretending
    (`docs/measurement.md` §6 and §7).
11. **When something fails.** A network error, or a model that spends its whole budget on hidden
    thinking and returns no answer, drops **that anchor**: it is not written, and it is counted in
    the manifest's failure report. Nothing is faked. Re-running the same output directory resumes —
    the run compares the plan against the bank and generates only the positions still missing. It
    does not regenerate anchors already on disk, and a plan that does not match the existing records
    is refused before anything is written (so a smoke run and a full run are never mixed into one
    directory).

## Where the results land

Everything goes to `outputs/<run name>/` — default `ard_dataset_<timestamp>`, with the `_smoke`
suffix for a smoke run; set `[output] directory` to pin the name. The full layout, and which file
is the authoritative declaration, are in [docs/architecture.md](architecture.md) §4 and in the
README's Output section.

| File | What it is | Look here when you want to know |
|---|---|---|
| `anchor_bank.jsonl` | the dataset: one anchor per line | what was actually generated |
| `manifest.json` | the run's own declaration: plan, composition, health, image use | whether the run is complete, and what it was meant to be |
| `config.toml` | the merged settings this run used (secret values masked) | which settings produced this output — it is itself a usable `--config` |
| `images/<visual_domain>/<file>` | the pictures placed for this run | which image an image anchor uses |
| `results/coverage.json`, `results/coverage.md` | coverage, density and `q95`, machine- and human-readable | how well the run covers the space |
| `logs/` | the run's log files | the detail behind a warning or a failure |
| `plan_identity.in_progress.json` | only while a run is unfinished: binds the directory to its plan | what plan an interrupted directory belongs to |

Two parts of `manifest.json` are worth knowing by name. The `plan` section carries the readouts
(`unit_total`, `coverage_ratio`, `density`, …). The `images` section carries how pictures were
resolved: `fallback_visual_domains` and `fallback_anchor_count` say which domains were shown a
reused picture and how many anchors that affected, and each row of `resolved_images` carries a
`fallback` flag.

## Common questions

The README's FAQ answers these at more length; these are the short versions.

**What happens if I have very few pictures?** They are reused, not skipped. A domain without a
picture of its own takes one from the whole image tree and rotates through it, so a single picture
is enough for a complete run. The manifest records every reuse, so you can judge whether the
repetition is acceptable for your dataset.

**Can I stop and continue later?** Yes. Re-running the same output directory fills in only the
missing anchors; everything already on disk is kept. Keep the same settings — if the plan changed,
the run stops before writing anything rather than mixing two plans into one directory.

**Can N be larger than one round?** Yes, N has no upper bound. Larger than a round simply means the
plan enters the next round; a coordinate that comes back in a later round is a new sample and is
kept.

**What does `[images] skip_missing_images` do?** Only one situation leaves an anchor without a
picture: the whole `--image-dir` tree holds no usable image at all — reuse covers every other case.
That situation is refused by default. Set `skip_missing_images = true` to drop those anchors
instead; each drop is logged and declared in the manifest.

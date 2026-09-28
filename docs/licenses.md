# Runtime dependency licences

**Facts only.** This page records *which packages are installed, under which licence, and how to
reproduce the list*. It does **not** judge whether any of those licences is acceptable for a
given use — that call belongs to whoever ships or redistributes the software, and it depends on
their own obligations.

## 1. Facts that bear on such a judgement

- **A default install contains no `rawpy` and no LibRaw.** A plain `uv sync` installs the
  project's runtime closure only. Support for RAW camera formats lives in the optional `raw`
  extra (`uv sync --extra raw`). Only then does `rawpy` enter the environment, and only then
  does its bundled LibRaw decoder (**LGPL-2.1 / CDDL-1.0**) enter with it.
- **The default closure is not uniformly MIT/Apache.** Most of it is permissive
  (MIT, BSD-3-Clause, MIT-CMU, PSF-2.0). The entries that are *not* MIT/Apache are:
  - `certifi` — **MPL-2.0**, pulled in transitively by `httpx` (which depends on it directly, as
    does `httpcore`);
  - `tqdm` — **`MPL-2.0 AND MIT`** (a dual licence: the MIT option is available);
  - `typing-extensions` — **PSF-2.0**, pulled in transitively through the pydantic stack;
  - `numpy` — `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0`.
- Read from those metadata fields, the default closure contains **no Apache-2.0 package** and
  **no GPL/AGPL package**. The `raw` extra adds `rawpy` (MIT) and, inside its wheel, LibRaw
  (LGPL-2.1 / CDDL-1.0) — no further distribution.

## 2. Reproducing the complete list

The list is derived from the installed environment, never from a hand-written table. In the
project root, for the **default** environment:

```bash
uv sync
uv run python - <<'PY'
import importlib.metadata as md

seen: dict[str, tuple[str, str, str, str]] = {}
for dist in md.distributions():
    name = dist.metadata.get("Name") or "?"
    expr = dist.metadata.get("License-Expression")
    plain = dist.metadata.get("License")
    value = expr or plain
    source = "License-Expression" if expr else ("License" if plain else "")
    if not value:
        hits = [c for c in (dist.metadata.get_all("Classifier") or [])
                if c.startswith("License ::")]
        value = "; ".join(hits) or "unclassified"
        source = "Classifier"
    seen.setdefault(name.lower(), (name, dist.version, value, source))

for key in sorted(seen):
    name, version, value, source = seen[key]
    print(f"{name}\t{version}\t{value}\t({source})")
PY
```

For the environment **with the `raw` extra**, run the same block after `uv sync --extra raw`:
`rawpy` is the only package it adds, and LibRaw is not a separate distribution — it is declared
by rawpy's own `METADATA` (`License-File: LICENSE.LibRaw`) and ships as a shared library inside
rawpy's wheel.

The root package may show up twice in a raw `importlib.metadata` walk (an editable install plus
its installed metadata); the snippet above de-duplicates by name.

## 3. This page is not the authority

Only the notes above and the way to reproduce them are maintained here. The authority is
`uv.lock` (the exact dependency closure) together with each installed distribution's own
metadata — `License-Expression`, then `License`, then a `License ::` classifier, in that order,
and "unclassified" when all three are absent. If the dependencies change, the output of §2 wins
over the snapshot below.

## 4. Snapshot (goes stale — regenerate with §2)

Generated from the tracked `uv.lock` at tag `v5.0.0`, environment: **default** (`uv sync`, no
extra), Python 3.11.15 on Linux x86-64. Versions are that `uv.lock` resolution; the root package's
version is derived from git tags by `setuptools_scm`.

| Package | Version | Licence (as read) | Read from |
|---|---|---|---|
| `anchor-replay-distillation` | 5.0.0 | MIT | `License` |
| `annotated-types` | 0.8.0 | MIT | `License-Expression` |
| `anyio` | 4.13.0 | MIT | `License-Expression` |
| `certifi` | 2026.5.20 | **MPL-2.0** | `License` |
| `h11` | 0.16.0 | MIT | `License` |
| `httpcore` | 1.0.9 | BSD-3-Clause | `License-Expression` |
| `httpx` | 0.28.1 | BSD-3-Clause | `License` |
| `idna` | 3.16 | BSD-3-Clause | `License-Expression` |
| `numpy` | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | `License-Expression` |
| `Pillow` | 12.3.0 | MIT-CMU | `License-Expression` |
| `pydantic` | 2.13.5 | MIT | `License-Expression` |
| `pydantic-core` | 2.46.5 | MIT | `License-Expression` |
| `tomli-w` | 1.2.0 | MIT | `Classifier` |
| `tqdm` | 4.67.3 | **`MPL-2.0 AND MIT`** | `License` |
| `typing-extensions` | 4.15.0 | **PSF-2.0** | `License-Expression` |
| `typing-inspection` | 0.4.4 | MIT | `License-Expression` |

The `raw` extra snapshot is not repeated here: it adds `rawpy` (MIT) only, and the LibRaw
licence text travels inside that wheel as described in §2.

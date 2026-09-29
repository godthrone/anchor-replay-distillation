# Contributing to ARD

Thanks for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/godthrone/anchor-replay-distillation.git
cd anchor-replay-distillation
# Keep uv's managed Python inside this checkout.  `.venv/bin/python` points into
# uv's managed-Python directory, and an interpreter installed on a volatile disk
# (a `/tmp` path, wiped by a reboot or a WSL restart) leaves the environment
# broken.  uv has no settings-file key for that directory — the environment
# variable below is the only knob — so export it before any `uv` command.
export UV_PYTHON_INSTALL_DIR="$PWD/.local/uv-python"
uv sync --extra dev
```

## Code Quality

```bash
# Run tests
uv run pytest tests/ -v

# Type checking
uv run mypy src/ard/

# Linting
uv run ruff check src/ tests/

# Formatting (check only — an unformatted file fails the gate)
uv run ruff format --check src/ tests/
```

## Branch and Review Workflow

ARD has **one active maintainer**, and its history reflects that: changes are committed
**directly to `main`**, without a `feature/` branch and without a pull request. That is a
deliberate single-contributor choice, not an oversight — a PR opened and reviewed by the same
person adds a round trip and no reviewer.

So, for a change you make yourself:

1. Make the change (a local branch is fine), run the checks under "Code Quality".
2. Add or update tests for new functionality.
3. Commit it to `main` with a conventional-commit subject (see below).
4. One commit = one change.

**Sending a change from outside this repository:** open an issue first if it is more than a
small fix, then send it as a **pull request against `main`**. The maintainer reviews the diff,
asks for the checks to pass, and merges. Pull requests become mandatory for everyone — including
the maintainer — as soon as the project gains a second regular contributor; this section is
updated at that point.

## Commit Convention

Use conventional commits: `type: description`

- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation
- `refactor:` — code restructuring
- `test:` — test additions/changes
- `chore:` — build/config changes

## Referencing Code in New Text

When a new code comment, docstring, test, or document points at a spot in the source,
name the symbol (function, class, constant, config key) — do **not** write `file:line`.
Line numbers drift with every edit, so a `file:line` reference is stale the moment the
file changes.

## Versioning

The version comes from git tags through `setuptools_scm` (`MAJOR.MINOR.PATCH`). The construction
rule is settled, so later improvements do not raise the major version number.

## Third-party Licences

**Facts only.** This section records *which packages are installed, under which licence, and how to
reproduce the list*. It does **not** judge whether any of those licences is acceptable for a
given use — that call belongs to whoever ships or redistributes the software, and it depends on
their own obligations.

### Facts that bear on such a judgement

- **A default install contains no `rawpy`, no LibRaw and no `numpy`.** A plain `uv sync` installs
  the project's runtime closure only. Support for RAW camera formats — and the `numpy` array the
  decoder hands to Pillow — lives in the optional `raw` extra (`uv sync --extra raw`). Only then
  do `numpy` and `rawpy` enter the environment, and only then does rawpy's bundled LibRaw decoder
  (**LGPL-2.1 / CDDL-1.0**) enter with it.
- **The default closure is not uniformly MIT/Apache.** Most of it is permissive
  (MIT, BSD-3-Clause, MIT-CMU, PSF-2.0). The entries that are *not* MIT/Apache are:
  - `certifi` — **MPL-2.0**, pulled in transitively by `httpx` (which depends on it directly, as
    does `httpcore`);
  - `tqdm` — **`MPL-2.0 AND MIT`** (a dual licence: the MIT option is available);
  - `typing-extensions` — **PSF-2.0**, pulled in transitively through the pydantic stack.
- Read from those metadata fields, the default closure contains **no Apache-2.0 package** and
  **no GPL/AGPL package**. The `raw` extra adds `numpy` (`BSD-3-Clause AND 0BSD AND MIT AND Zlib
  AND CC0-1.0`), `rawpy` (MIT) and, inside the rawpy wheel, LibRaw (LGPL-2.1 / CDDL-1.0) — no
  further distribution.

### Reproducing the complete list

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
it adds `numpy` and `rawpy`, and LibRaw is not a separate distribution — it is declared by
rawpy's own `METADATA` (`License-File: LICENSE.LibRaw`) and ships as a shared library inside
rawpy's wheel.

The root package may show up twice in a raw `importlib.metadata` walk (an editable install plus
its installed metadata); the snippet above de-duplicates by name.

### This section is not the authority

Only the notes above and the way to reproduce them are maintained here. The authority is
`uv.lock` (the exact dependency closure) together with each installed distribution's own
metadata — `License-Expression`, then `License`, then a `License ::` classifier, in that order,
and "unclassified" when all three are absent. If the dependencies change, the output of the
"Reproducing the complete list" section wins over the snapshot below.

### Snapshot (goes stale — regenerate as above)

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
| `Pillow` | 12.3.0 | MIT-CMU | `License-Expression` |
| `pydantic` | 2.13.5 | MIT | `License-Expression` |
| `pydantic-core` | 2.46.5 | MIT | `License-Expression` |
| `tomli-w` | 1.2.0 | MIT | `Classifier` |
| `tqdm` | 4.67.3 | **`MPL-2.0 AND MIT`** | `License` |
| `typing-extensions` | 4.15.0 | **PSF-2.0** | `License-Expression` |
| `typing-inspection` | 0.4.4 | MIT | `License-Expression` |

The `raw` extra snapshot is not repeated here: it adds `numpy` (`BSD-3-Clause AND 0BSD AND MIT
AND Zlib AND CC0-1.0`) and `rawpy` (MIT), and the LibRaw licence text travels inside the rawpy
wheel as described above. The `dev` extra carries `numpy` too, for the test suite.

## License

By contributing, you agree that your contributions will be licensed
under the MIT License.
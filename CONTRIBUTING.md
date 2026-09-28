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
file changes; the repository has deliberately dropped its line-reference guard, and no
test scanning for `file:line` references should be added back.

## License

By contributing, you agree that your contributions will be licensed
under the MIT License.
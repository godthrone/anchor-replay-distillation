"""Guard: every ``file:line`` reference in the docs points at a real line.

What this catches
-----------------
For each reference found in ``docs/`` + ``README*`` + ``CONTRIBUTING.md`` whose
target file has a known extension (``.py`` / ``.md`` / ``.toml`` / ``.json`` /
``.sh``), it asserts that:

1. the target file exists in this tree;
2. every line number is inside that file; and
3. the referenced **start** line is non-empty.

What this cannot catch (capability boundary, stated on purpose)
--------------------------------------------------------------
It **cannot detect an off-by-N drift onto another real, non-empty line** — that
is exactly what a ``file:line`` reference silently becomes when a line is
inserted above it.  The wrong line is still a valid, non-blank line, so only a
human (or a symbol-aware check) can see that the cited text no longer matches
the prose.  This guard is a *hard-error* net (deleted file, renamed file, line
past the end, reference left dangling on a blank line), not a correctness proof
for line numbers.  A reference that lands on a blank line — which is how a
``+1`` shift past a section separator shows up — *is* caught.

**Range references are outside its coverage.**  A ``file:12-30`` reference is
bounds-checked only: the guard asserts that line 30 exists and that line 12 is
non-blank.  It never asserts that the range still brackets the symbol the prose
names, nor that either endpoint belongs to it.  A range that has drifted by N
lines onto other real lines — or one that was wrong from the time it was written
— therefore passes silently, even though a wrong range is the more damaging
error: it is the form used to cite a whole function or data block.  Only a human
(or a symbol-aware check) can see that its endpoints no longer match the prose.

Not covered at all: targets without one of the extensions above (e.g.
``docker/Dockerfile:41``), and relative ``:123`` references whose target file is
only clear from the surrounding prose rather than from an explicit reference
earlier on the same line.  Those are counted and skipped rather than guessed.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: ``path/to/file.py:12`` / ``path/to/file.py:12-30``, with the path optional
#: (a bare ``:12`` inherits the last explicit reference on the same line).
REF = re.compile(
    r"(?P<path>[A-Za-z0-9_./-]+\.(?:py|md|toml|json|sh))?:(?P<start>\d+)(?:-(?P<end>\d+))?"
)

DOC_TARGETS = ("docs", "README.md", "README.zh-CN.md", "CONTRIBUTING.md")
#: Directories never searched when resolving a bare basename such as
#: ``pipeline.py``: virtualenvs, caches, run outputs and VCS metadata.
SKIP_DIRS = {"outputs", "node_modules"}


def _doc_files() -> list[Path]:
    files: list[Path] = []
    for name in DOC_TARGETS:
        target = REPO_ROOT / name
        if target.is_dir():
            files.extend(sorted(target.rglob("*.md")))
        elif target.is_file():
            files.append(target)
    return files


def _skipped(path: Path) -> bool:
    return any(
        part.startswith(".") or part in SKIP_DIRS for part in path.relative_to(REPO_ROOT).parts
    )


def _unique_basename(name: str) -> Path | None:
    """``pipeline.py`` → ``src/ard/pipeline.py`` when the basename is unique."""
    if "/" in name:
        return None
    hits = [path for path in REPO_ROOT.rglob(name) if not _skipped(path)]
    return hits[0].relative_to(REPO_ROOT) if len(hits) == 1 else None


def _resolve(raw: str) -> Path:
    if raw.startswith("/"):
        return Path(raw)
    if "/" in raw:
        return REPO_ROOT / raw
    hit = _unique_basename(raw)
    return REPO_ROOT / (hit if hit is not None else raw)


def _is_clock_time(line: str, match: re.Match[str]) -> bool:
    return match.start() > 0 and line[match.start() - 1].isdigit()


def _problems() -> list[str]:
    problems: list[str] = []
    for doc in _doc_files():
        doc_name = doc.relative_to(REPO_ROOT)
        for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            same_line: str | None = None
            for match in REF.finditer(line):
                if _is_clock_time(line, match):
                    continue
                raw = match.group("path")
                if raw is not None:
                    if raw.startswith(("http://", "https://")):
                        continue
                    same_line = raw
                    target = _resolve(raw)
                elif same_line is not None:
                    target = _resolve(same_line)
                else:
                    continue  # document-level relative reference: not resolved here
                start = int(match.group("start"))
                end = int(match.group("end")) if match.group("end") else start
                where = f"{doc_name}:{lineno} -> {match.group(0)}"
                if not target.is_file():
                    problems.append(f"{where}: target file does not exist ({target})")
                    continue
                lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
                if start < 1 or end > len(lines):
                    problems.append(f"{where}: out of range ({target} has {len(lines)} lines)")
                elif not lines[start - 1].strip():
                    problems.append(f"{where}: referenced line {start} is blank")
    return problems


def test_doc_file_line_references_are_resolvable() -> None:
    problems = _problems()
    assert not problems, (
        "docs cite lines that do not exist / are out of range / are blank:\n"
        + "\n".join(problems)
        + "\n\n(An off-by-N drift onto another non-blank line is NOT detectable "
        "here — re-read the cited line against the prose.)"
    )

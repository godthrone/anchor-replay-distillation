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

**Range references are only partly covered.**  A ``file:12-30`` reference is
always bounds-checked: the guard asserts that line 30 exists and that line 12 is
non-blank.  On top of that, when line 12 is itself a ``def`` / ``class`` line,
the guard parses the target with :mod:`ast` and asserts that the range reaches
the end of a definition: either the one it starts at (one line of slack covers
the blank line the repo's style leaves after a block) or one it deliberately
spans and ends on.  A stale range whose end now sits inside the function it
starts at — the shape a range takes when the function grew underneath it — is
therefore caught, and both its endpoints are still real, non-blank lines, so no
amount of bounds checking would ever see it.

What that still cannot catch: a range whose *start* line is no longer a
``def`` / ``class`` at all (a symbol that moved down onto other code), a range
that starts on a data block rather than a definition, and an end that is merely
*long* (the check does not assume the cited block ends at the next definition).
Those remain for a human to re-read against the prose.

Not covered at all: targets without one of the extensions above (e.g.
``docker/Dockerfile:41``), and relative ``:123`` references whose target file is
only clear from the surrounding prose rather than from an explicit reference
earlier on the same line.  Those are counted and skipped rather than guessed.
"""

from __future__ import annotations

import ast
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


#: How far short of a definition's last line a range may end and still count as
#: reaching it.  One line: the repository's citation style sometimes ends a range
#: on the blank line after a block (and sometimes omits that blank line).
RANGE_END_SLACK = 1


def _definitions(target: Path) -> list[tuple[int, int, str]]:
    """``(first line, last line, name)`` of every ``def`` / ``class`` that ended."""
    tree = ast.parse(target.read_text(encoding="utf-8", errors="replace"))
    return sorted(
        (node.lineno, node.end_lineno, node.name)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.end_lineno is not None
    )


def _stale_range(where: str, target: Path, start: int, end: int) -> list[str]:
    """Symbol-aware check for a range that *starts* on a definition.

    A range starting at ``def f`` / ``class C`` is expected to bracket a whole
    definition: either it reaches that definition's last line (within
    ``RANGE_END_SLACK`` above), or it spans several definitions and ends on a
    later one's last line (within the same slack) — the class-cited-with-one-
    method shape.
    An end inside the definition it starts at, on no definition boundary, is
    what a range looks like after the function grew underneath it — both
    endpoints are still real, non-blank lines, so ``_problems``' bounds check
    cannot see it.  Ranges that start elsewhere are out of this check's scope.
    """
    if target.suffix != ".py" or end <= start:
        return []
    try:
        defs = _definitions(target)
    except SyntaxError:
        return []
    starts = [entry for entry in defs if entry[0] == start]
    if not starts:
        return []
    _, first_end, name = starts[0]
    if end >= first_end - RANGE_END_SLACK:
        return []
    if any(
        start < other_start <= end and other_end <= end <= other_end + RANGE_END_SLACK
        for other_start, other_end, _ in defs
    ):
        return []
    return [
        f"{where}: range ends at line {end}, inside `{name}` ({start}-{first_end}) "
        f"and on no definition boundary — the range no longer brackets a whole "
        f"definition (stale after the code moved?)"
    ]


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
                else:
                    problems.extend(_stale_range(where, target, start, end))
    return problems


def test_doc_file_line_references_are_resolvable() -> None:
    problems = _problems()
    assert not problems, (
        "docs cite lines that do not exist / are out of range / are blank / stop "
        "inside the definition they start at:\n"
        + "\n".join(problems)
        + "\n\n(An off-by-N drift onto another non-blank line is still not "
        "detectable for single-line references — re-read those against the prose.)"
    )


def test_a_range_that_stops_inside_its_definition_is_reported(tmp_path: Path) -> None:
    """Negative control: the stale end the bounds check cannot see."""
    target = tmp_path / "sample.py"
    target.write_text(
        'def f() -> int:\n    """Doc."""\n    x = 1\n    y = 2\n    return x + y\n',
        encoding="utf-8",
    )
    where = "doc.md:1 -> sample.py:1-3"
    assert _stale_range(where, target, 1, 3), "an end inside `f` must be reported"
    assert _stale_range(where, target, 1, 1) == [], "a single-line reference is not a range"
    assert _stale_range(where, target, 1, 4) == [], "one line of slack is allowed"
    assert _stale_range(where, target, 1, 5) == [], "the definition's last line is fine"


def test_a_range_may_end_on_a_nested_definition(tmp_path: Path) -> None:
    """A class cited together with one of its methods is not a stale range."""
    target = tmp_path / "sample.py"
    target.write_text(
        "class C:\n"
        '    """Doc."""\n'
        "\n"
        "    def m(self) -> int:\n"
        "        return 1\n"
        "\n"
        "    def n(self) -> int:\n"
        "        return 2\n",
        encoding="utf-8",
    )
    where = "doc.md:1 -> sample.py:1-5"
    assert _stale_range(where, target, 1, 5) == [], "ends on `m`'s last line"
    assert _stale_range(where, target, 1, 6) == [], "one line of slack after `m`"
    assert _stale_range(where, target, 1, 3), "an end on no boundary inside `C` is reported"

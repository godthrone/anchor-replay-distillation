"""Boundary guard: ``ard.core`` is pure computation (§1.3 计算与设施分离).

Responsibility: fail the build if any module under ``src/ard/core/`` gains
filesystem, network, environment or subprocess access.  The core docstrings
claim that separation; this test is the防呆 device that keeps the claim true
(§2.1 契约即防呆) now that the file-reading halves live in
:mod:`ard.backends.ontology_loader` and :mod:`ard.backends.prompt_loader`.

The scan is AST-based, not textual, so a mention of ``read_text`` inside a
docstring or a comment is not a violation — only real code is.
"""

import ast
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parents[2] / "src" / "ard" / "core"

#: Modules that give a core module access to the outside world.
FORBIDDEN_MODULES = frozenset(
    {
        "aiohttp",
        "http",
        "httpx",
        "os",
        "requests",
        "shutil",
        "socket",
        "subprocess",
        "tempfile",
        "urllib",
    }
)

#: Builtin that opens a file.
FORBIDDEN_BUILTINS = frozenset({"open"})

#: ``pathlib.Path`` methods (and friends) that touch the filesystem.  Path
#: *arithmetic* — ``Path(...)``, ``/``, ``.name``, ``.parent`` — stays legal:
#: core declares where wording files are, it just must not read them.
FORBIDDEN_PATH_METHODS = frozenset(
    {
        "chmod",
        "cwd",
        "exists",
        "glob",
        "home",
        "is_dir",
        "is_file",
        "iterdir",
        "mkdir",
        "open",
        "read_bytes",
        "read_text",
        "rglob",
        "stat",
        "touch",
        "unlink",
        "walk",
        "write_bytes",
        "write_text",
    }
)


def _module_root(name: str) -> str:
    """Return the top-level package of a dotted module name."""
    return name.split(".", 1)[0]


def _scan(path: Path) -> list[str]:
    """Return one ``file:line: reason`` string per access found in *path*."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        imported: list[str] = []
        if isinstance(node, ast.Import):
            imported = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported = [node.module or ""]
        for name in imported:
            if _module_root(name) in FORBIDDEN_MODULES:
                found.append(f"{path}:{node.lineno}: imports {name!r}")
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in FORBIDDEN_BUILTINS:
            found.append(f"{path}:{node.lineno}: calls {func.id}()")
        elif isinstance(func, ast.Attribute) and func.attr in FORBIDDEN_PATH_METHODS:
            found.append(f"{path}:{node.lineno}: calls .{func.attr}()")
    return found


def test_core_package_is_present_where_this_guard_expects_it() -> None:
    """Sanity: the scan points at the real core package, not an empty directory."""
    modules = sorted(CORE_DIR.glob("*.py"))
    names = {module.name for module in modules}
    assert {"ontology.py", "system_prompt.py", "constraints.py"} <= names


def test_core_modules_do_no_file_network_or_subprocess_io() -> None:
    """Every ``src/ard/core/*.py`` module is free of facility access (§1.3)."""
    problems: list[str] = []
    for module in sorted(CORE_DIR.glob("*.py")):
        problems.extend(_scan(module))
    assert problems == [], "core must stay pure computation (§1.3):\n" + "\n".join(problems)

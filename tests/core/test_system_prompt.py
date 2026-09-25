"""Tests for the system-prompt sampling dimension (:mod:`ard.core.system_prompt`).

Covers the vocabulary contract — the modes are the v4 ontology's
``system_prompt_mode`` axis values, all five of them — the wording-files
contract (the directory the ontology declares, one non-empty file per axis
value, no silent fallback), and the generation prompt handed to the input
generator, which must stay byte-identical to the wording that used to be
hardcoded (WP-S6c).

Reading the wording files is facility work (§1.3), so
``build_system_prompt_prompt`` comes from :mod:`ard.backends.prompt_loader`; the
pure contract it feeds (path arithmetic, template validation, rendering) stays
in :mod:`ard.core.system_prompt` and is tested directly as well.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ard.backends.prompt_loader import build_system_prompt_prompt
from ard.core.ontology import FlatAxisWithDefinitions, OntologyV4
from ard.core.system_prompt import (
    SYSTEM_PROMPT_NONE,
    SYSTEM_PROMPT_TEMPLATE_DIR,
    SYSTEM_PROMPT_TEMPLATE_FIELDS,
    SystemPromptTemplateError,
    render_system_prompt_prompt,
    require_present_mode,
    system_prompt_template_path,
    validate_system_prompt_template,
)

# ── Vocabulary: the v4 axis is the single source ────────────────────────────


def test_modes_are_the_five_axis_values(ontology: OntologyV4) -> None:
    """The modes are the ontology axis values — all five of them."""
    values = list(ontology.axis_values("system_prompt_mode"))
    assert len(values) == 5
    assert len(set(values)) == 5
    assert values[0] == SYSTEM_PROMPT_NONE
    # The v3 shape fell back to a single ``("none", "none")`` *pair*; the v4
    # modes are plain coordinate values, never tuples.
    assert all(isinstance(value, str) for value in values)


def test_axis_defines_every_mode_the_module_hands_out(ontology: OntologyV4) -> None:
    """Each mode is documented on the axis it came from."""
    axis = ontology.axes.spec("system_prompt_mode")
    assert isinstance(axis, FlatAxisWithDefinitions)
    for mode in ontology.axis_values("system_prompt_mode"):
        assert axis.value_definitions.get(mode, "").strip()


# ── Wording files: declared location and coverage ───────────────────────────


def test_template_dir_is_the_ontology_declared_location(ontology: OntologyV4) -> None:
    """The wording directory is the one the ontology declares (§1.4).

    The ontology names a per-mode file with a ``<system_prompt_mode>``
    placeholder; the runtime states the directory once.  This test is what keeps
    the two from drifting apart.
    """
    target = ontology.wording_policy.prompt_wording_location_recommendation.target
    assert "<system_prompt_mode>" in target
    declared = Path(target.replace("<system_prompt_mode>", "none"))
    assert declared == system_prompt_template_path("none")
    assert declared.parent == SYSTEM_PROMPT_TEMPLATE_DIR


def test_directory_holds_one_non_empty_file_per_axis_value(ontology: OntologyV4) -> None:
    """Every ``system_prompt_mode`` value has a non-empty wording file."""
    modes = list(ontology.axis_values("system_prompt_mode"))
    on_disk = sorted(path.stem for path in SYSTEM_PROMPT_TEMPLATE_DIR.glob("*.md"))
    assert on_disk == sorted(modes)
    for mode in modes:
        body = system_prompt_template_path(mode).read_text(encoding="utf-8").strip()
        assert body, f"{mode}: wording file is empty"
        if mode != SYSTEM_PROMPT_NONE:
            assert "{" in body, f"{mode}: template uses no placeholder at all"


# ── Generation prompt ───────────────────────────────────────────────────────

#: Fixed anchor metadata used for the byte-identity evidence.
RENDER_META = {
    "language": "日本語",
    "knowledge_domain": "software_engineering",
    "capability": "debugging",
}

#: sha256 of the prompt each present mode rendered **before** the wording moved
#: into data files (the WP-S6c byte-identity evidence).  A change here means the
#: wording changed, not just its location.
RENDERED_SHA256 = {
    "minimal_persona": "543cb1bc58576a326614dd00f6c333d623bdcbaf65855aa14a9cf7184af56ee4",
    "detailed_persona": "055e69c7fa4f49e697e1dbb50240db2e38d84df4ea01d286a0e82e099474f8f2",
    "task_constraint": "281eb68cd6e3115c9c89ab4fbcaadc11115063e43412a53774966b406e5a164e",
    "domain_style": "adeed39ad80f61b92be5611aeb9602d1307eb27d0ae417f4270bd329743d74a0",
}


def test_generation_prompt_is_byte_identical_to_the_old_in_code_wording() -> None:
    """The extraction moved the wording; it did not rewrite a single byte."""
    for mode, digest in RENDERED_SHA256.items():
        rendered = build_system_prompt_prompt(RENDER_META, mode)
        assert hashlib.sha256(rendered.encode()).hexdigest() == digest, mode


def test_generation_prompt_echoes_domain_and_capability() -> None:
    """The prompt ties the system text to the anchor's own domain/capability."""
    prompt = build_system_prompt_prompt(RENDER_META, "task_constraint")
    assert "software_engineering" in prompt
    assert "debugging" in prompt
    assert "日本語" in prompt
    assert "task-constraint" in prompt


def test_generation_prompt_applies_the_metadata_defaults() -> None:
    """An anchor without those metadata keys still renders, with the defaults."""
    prompt = build_system_prompt_prompt({}, "minimal_persona")
    assert "English" in prompt
    assert "'qa'" in prompt
    assert "'general'" in prompt


def test_generation_prompt_accepts_a_directory_override(tmp_path: Path) -> None:
    """The wording directory can be pointed elsewhere (used by these tests)."""
    (tmp_path / "custom_style.md").write_text("Be terse in {language}.", encoding="utf-8")
    prompt = build_system_prompt_prompt({"language": "German"}, "custom_style", tmp_path)
    assert prompt == "Be terse in German."


# ── Failure paths: loud, named, never a silent fallback ─────────────────────


def test_missing_wording_directory_is_a_hard_error(tmp_path: Path) -> None:
    """A missing directory names path and expectation instead of falling back."""
    missing = tmp_path / "system_prompt"
    with pytest.raises(SystemPromptTemplateError) as excinfo:
        build_system_prompt_prompt({}, "task_constraint", directory=missing)
    message = str(excinfo.value)
    assert str(missing) in message
    assert "task_constraint.md" in message


def test_missing_mode_file_is_a_hard_error(tmp_path: Path) -> None:
    """A mode without a file is an error naming the file it expected."""
    with pytest.raises(SystemPromptTemplateError) as excinfo:
        build_system_prompt_prompt({}, "task_constraint", directory=tmp_path)
    message = str(excinfo.value)
    assert str(tmp_path / "task_constraint.md") in message
    assert "no wording file" in message


def test_empty_mode_file_is_a_hard_error(tmp_path: Path) -> None:
    """An empty (or whitespace-only) file is an error, not empty wording."""
    (tmp_path / "task_constraint.md").write_text("  \n\t\n", encoding="utf-8")
    with pytest.raises(SystemPromptTemplateError) as excinfo:
        build_system_prompt_prompt({}, "task_constraint", directory=tmp_path)
    assert "is empty" in str(excinfo.value)
    assert str(tmp_path / "task_constraint.md") in str(excinfo.value)


def test_unknown_placeholder_is_a_hard_error(tmp_path: Path) -> None:
    """A placeholder outside the declared set cannot be filled — so it fails."""
    (tmp_path / "task_constraint.md").write_text("Be {tone}.", encoding="utf-8")
    with pytest.raises(SystemPromptTemplateError) as excinfo:
        build_system_prompt_prompt({}, "task_constraint", directory=tmp_path)
    message = str(excinfo.value)
    assert "tone" in message
    assert str(list(SYSTEM_PROMPT_TEMPLATE_FIELDS)) in message


def test_malformed_placeholder_is_a_hard_error(tmp_path: Path) -> None:
    """Unbalanced braces are reported as a template defect."""
    (tmp_path / "task_constraint.md").write_text("Be terse in {language", encoding="utf-8")
    with pytest.raises(SystemPromptTemplateError):
        build_system_prompt_prompt({}, "task_constraint", directory=tmp_path)


def test_mode_must_be_a_bare_file_name(tmp_path: Path) -> None:
    """A mode is a file name: a path-like value cannot escape the directory."""
    with pytest.raises(SystemPromptTemplateError):
        build_system_prompt_prompt({}, "../elsewhere", directory=tmp_path)


def test_generation_prompt_rejects_unknown_mode() -> None:
    """An unknown mode cannot silently fall back to a default style."""
    with pytest.raises(SystemPromptTemplateError) as excinfo:
        build_system_prompt_prompt({}, "casual_vibes")
    assert str(system_prompt_template_path("casual_vibes")) in str(excinfo.value)


def test_generation_prompt_refuses_the_absence_case() -> None:
    """``none`` has no generator-facing wording, so asking for it is an error."""
    with pytest.raises(SystemPromptTemplateError) as excinfo:
        build_system_prompt_prompt({}, SYSTEM_PROMPT_NONE)
    assert "absence case" in str(excinfo.value)


# ── The pure half: validation and rendering of already-read text ────────────


def test_required_present_mode_rejects_only_the_absence_case() -> None:
    """The single statement of the absence-case rule accepts every style."""
    require_present_mode("task_constraint")
    with pytest.raises(SystemPromptTemplateError, match="absence case"):
        require_present_mode(SYSTEM_PROMPT_NONE)


def test_validate_strips_and_returns_the_body() -> None:
    """Valid wording text is returned stripped, ready for the renderer."""
    body = validate_system_prompt_template("\n  Be terse in {language}.  \n", "w.md")
    assert body == "Be terse in {language}."


def test_validate_rejects_empty_and_unknown_placeholders() -> None:
    """Empty wording and a placeholder outside the declared set are hard errors."""
    with pytest.raises(SystemPromptTemplateError, match="is empty"):
        validate_system_prompt_template("   \n", "w.md")
    with pytest.raises(SystemPromptTemplateError, match="tone"):
        validate_system_prompt_template("Be {tone}.", "w.md")


def test_render_fills_placeholders_from_metadata() -> None:
    """Rendering is pure: same template plus same metadata, same string."""
    rendered = render_system_prompt_prompt(
        {"language": "German"}, "custom_style", "Be terse in {language}."
    )
    assert rendered == "Be terse in German."


def test_render_refuses_the_absence_case() -> None:
    """The pure renderer states the absence case too — no wording exists for it."""
    with pytest.raises(SystemPromptTemplateError, match="absence case"):
        render_system_prompt_prompt({}, SYSTEM_PROMPT_NONE, "unused {language}")

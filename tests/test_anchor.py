import json
from collections import Counter
from pathlib import Path

from ard.anchor import (
    AnchorGenerationConfig,
    TargetAnswerAnchor,
    filter_target_answer_anchors,
    generate_anchor_prompts,
    load_anchor_ontology,
    split_target_answer_anchors,
)
from ard.anchor.manifest import build_anchor_manifest


def test_anchor_generation_is_deterministic(tmp_path):
    ontology_path = tmp_path / "anchor_ontology.json"
    ontology_path.write_text(
        json.dumps(
            {
                "languages": ["English"],
                "knowledge_domains": {"domain": {"topic": ["leaf"]}},
                "capabilities": {"knowledge_response": ["qa"]},
                "conversation_types": {"single_turn": ["single_turn"]},
                "safety_boundaries": {"normal": ["standard_helpful_answer"]},
                "language_features": {"style": ["concise", "formal"]},
            }
        ),
        encoding="utf-8",
    )

    ontology = load_anchor_ontology(ontology_path)
    config = AnchorGenerationConfig(count=4, seed=7, languages=["English"], task_types=["qa"])

    first = generate_anchor_prompts(
        ontology.knowledge,
        ontology.language_features,
        config,
        ontology.capabilities,
        ontology.conversation_types,
        ontology.safety_boundaries,
    )
    second = generate_anchor_prompts(
        ontology.knowledge,
        ontology.language_features,
        config,
        ontology.capabilities,
        ontology.conversation_types,
        ontology.safety_boundaries,
    )

    assert [item.id for item in first] == [item.id for item in second]
    assert len(first) == 4
    assert first[0].anchor_meta["task_type"] == "qa"


def test_anchor_ontology_file_exposes_all_sampling_dimensions():
    ontology = load_anchor_ontology(Path("data/anchor_seed/anchor_ontology.json"))

    assert ontology.languages == ["English", "简体中文", "bilingual_zh_en"]
    assert ontology.knowledge.leaves
    assert ontology.language_features.leaves
    assert ontology.capabilities.leaves
    assert ontology.conversation_types.leaves
    assert ontology.safety_boundaries.leaves
    assert {leaf["top_level"] for leaf in ontology.knowledge.leaves}
    assert {leaf["leaf"] for leaf in ontology.capabilities.leaves} >= {"qa", "coding"}


def test_anchor_ontology_rejects_missing_required_sections(tmp_path):
    path = tmp_path / "bad_ontology.json"
    path.write_text(json.dumps({"languages": ["English"]}), encoding="utf-8")

    try:
        load_anchor_ontology(path)
    except ValueError as exc:
        assert "missing required sections" in str(exc)
    else:
        raise AssertionError("expected missing section error")


def test_extended_anchor_sampling_is_deterministic_and_balanced():
    ontology = load_anchor_ontology(Path("data/anchor_seed/anchor_ontology.json"))
    config = AnchorGenerationConfig(
        count=100,
        seed=11,
        languages=ontology.languages,
        input_generator_model="strong-input-generator",
        target_model="open-anchor-target",
    )

    first = generate_anchor_prompts(
        ontology.knowledge,
        ontology.language_features,
        config,
        ontology.capabilities,
        ontology.conversation_types,
        ontology.safety_boundaries,
    )
    second = generate_anchor_prompts(
        ontology.knowledge,
        ontology.language_features,
        config,
        ontology.capabilities,
        ontology.conversation_types,
        ontology.safety_boundaries,
    )

    assert [item.id for item in first] == [item.id for item in second]
    buckets = Counter(item.anchor_meta["sampling_bucket"] for item in first)
    assert buckets == {
        "general": 60,
        "code_math_reasoning": 20,
        "business_agent": 15,
        "safety_refusal": 5,
    }
    assert sum(item.anchor_meta["is_multi_turn"] for item in first) == 30
    assert {item.anchor_meta["language"] for item in first} == {
        "English",
        "简体中文",
        "bilingual_zh_en",
    }
    assert all(
        item.anchor_meta["input_generator_model"] == "strong-input-generator" for item in first
    )
    assert all(item.anchor_meta["target_model"] == "open-anchor-target" for item in first)


def test_task_type_filter_limits_capability_sampling():
    ontology = load_anchor_ontology(Path("data/anchor_seed/anchor_ontology.json"))
    config = AnchorGenerationConfig(
        count=20,
        seed=13,
        languages=["English"],
        task_types=["qa", "coding"],
    )

    prompts = generate_anchor_prompts(
        ontology.knowledge,
        ontology.language_features,
        config,
        ontology.capabilities,
        ontology.conversation_types,
        ontology.safety_boundaries,
    )

    assert {item.anchor_meta["task_type"] for item in prompts} <= {"qa", "coding"}


def test_prompt_shape_instructions_for_single_and_multi_turn():
    ontology = load_anchor_ontology(Path("data/anchor_seed/anchor_ontology.json"))
    config = AnchorGenerationConfig(count=20, seed=3, languages=["English"])

    prompts = generate_anchor_prompts(
        ontology.knowledge,
        ontology.language_features,
        config,
        ontology.capabilities,
        ontology.conversation_types,
        ontology.safety_boundaries,
    )
    single = next(
        item for item in prompts if item.anchor_meta["conversation_type"] == "single_turn"
    )
    multi = next(item for item in prompts if item.anchor_meta["is_multi_turn"])

    assert "Generate one realistic user message only" in single.messages[0]["content"]
    assert len(single.messages) == 1
    assert single.messages[0]["role"] == "user"
    assert "JSON array of messages" in multi.messages[0]["content"]
    assert "final message must be from user" in multi.messages[0]["content"]
    assert "Do not include the final assistant answer" in multi.messages[0]["content"]


def test_safety_specs_add_safe_boundary_language():
    ontology = load_anchor_ontology(Path("data/anchor_seed/anchor_ontology.json"))
    config = AnchorGenerationConfig(count=20, seed=5, languages=["English"])

    prompts = generate_anchor_prompts(
        ontology.knowledge,
        ontology.language_features,
        config,
        ontology.capabilities,
        ontology.conversation_types,
        ontology.safety_boundaries,
    )
    safety_prompts = [
        item
        for item in prompts
        if item.anchor_meta["safety_top_level"] in {"regulated_domain", "boundary"}
    ]

    assert safety_prompts
    assert all(
        "non-authoritative guidance" in item.messages[0]["content"] for item in safety_prompts
    )
    assert all(
        "avoid asking for dangerous procedural details" in item.messages[0]["content"]
        for item in safety_prompts
    )


def test_anchor_manifest_counts_extended_dimensions():
    anchors = [
        TargetAnswerAnchor(
            id="a",
            messages=[{"role": "user", "content": "question"}],
            target_answer="answer",
            target_model="target",
            anchor_meta={
                "knowledge_top_level": "software_engineering",
                "capability": "debugging",
                "task_type": "debugging",
                "language": "English",
                "conversation_type": "single_turn",
                "safety_boundary": "standard_helpful_answer",
                "input_generator_model": "strong-input-generator",
                "target_model": "target",
            },
        )
    ]

    manifest = build_anchor_manifest(
        target_model="target",
        generation_config={},
        anchors=anchors,
        seed=7,
        input_generation_stats={"input_count": 1, "kept_count": 1},
        target_answer_stats={"input_count": 1, "kept_count": 1},
    )

    assert manifest["input_generator_model"] == "strong-input-generator"
    assert manifest["target_model"] == "target"
    assert manifest["counts"]["by_capability"] == {"debugging": 1}
    assert manifest["counts"]["by_conversation_type"] == {"single_turn": 1}
    assert manifest["counts"]["by_safety_boundary"] == {"standard_helpful_answer": 1}
    assert manifest["input_generation_stats"] == {"input_count": 1, "kept_count": 1}
    assert manifest["target_answer_stats"] == {"input_count": 1, "kept_count": 1}


def test_anchor_filter_and_split():
    anchors = [
        TargetAnswerAnchor(
            id="a",
            messages=[{"role": "user", "content": "question a"}],
            target_answer="a useful answer",
            target_model="base",
        ),
        TargetAnswerAnchor(
            id="b",
            messages=[{"role": "user", "content": "question b"}],
            target_answer="",
            target_model="base",
        ),
        TargetAnswerAnchor(
            id="c",
            messages=[{"role": "user", "content": "question a"}],
            target_answer="another answer",
            target_model="base",
        ),
    ]

    kept, stats = filter_target_answer_anchors(anchors, min_answer_chars=4)
    assert [item.id for item in kept] == ["a"]
    assert stats.dropped_empty == 1
    assert stats.dropped_duplicate_prompt == 1

    train, eval_items = split_target_answer_anchors(kept, eval_ratio=0.5, seed=1)
    assert len(train) + len(eval_items) == 1


def test_anchor_filter_keeps_long_answers_by_default():
    long_answer = "x" * 5000
    anchors = [
        TargetAnswerAnchor(
            id="a",
            messages=[{"role": "user", "content": "question a"}],
            target_answer=long_answer,
            target_model="base",
        )
    ]

    kept, stats = filter_target_answer_anchors(anchors)

    assert kept[0].target_answer == long_answer
    assert stats.dropped_too_long == 0


def test_anchor_filter_can_drop_long_answers_when_limit_is_explicit():
    anchors = [
        TargetAnswerAnchor(
            id="a",
            messages=[{"role": "user", "content": "question a"}],
            target_answer="x" * 5000,
            target_model="base",
        )
    ]

    kept, stats = filter_target_answer_anchors(anchors, max_answer_chars=4096)

    assert kept == []
    assert stats.dropped_too_long == 1

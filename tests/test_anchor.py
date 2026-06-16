import json
from pathlib import Path

import pytest

from ard.anchor import (
    AnchorGenerationConfig,
    TargetAnswerAnchor,
    filter_target_answer_anchors,
    generate_anchor_prompts,
    load_anchor_ontology,
    split_target_answer_anchors,
)
from ard.anchor.embeddings import (
    build_ontology_embeddings,
    hash_embed_texts,
    load_ontology_embeddings,
    write_ontology_embeddings,
)
from ard.anchor.manifest import build_anchor_manifest


def _minimal_ontology_payload():
    return {
        "languages": ["English"],
        "knowledge_domains": {
            "domain_a": {"topic": ["alpha"]},
            "domain_b": {"topic": ["beta"]},
            "domain_c": {"topic": ["gamma"]},
        },
        "capabilities": {
            "knowledge_response": ["qa"],
            "reasoning": ["reasoning"],
            "coding_and_data": ["coding"],
            "language_work": ["rewriting"],
            "agentic_behavior": ["tool_use"],
        },
        "conversation_types": {"single_turn": ["single_turn"]},
        "language_features": {
            "style": ["concise"],
            "format": ["paragraph"],
            "difficulty": ["basic"],
            "context_length": ["short"],
            "noise": ["clean"],
            "answer_expectation": ["direct_answer"],
        },
    }


def test_anchor_generation_is_deterministic(tmp_path):
    ontology_path = tmp_path / "anchor_ontology.json"
    ontology_path.write_text(json.dumps(_minimal_ontology_payload()), encoding="utf-8")

    ontology = load_anchor_ontology(ontology_path)
    config = AnchorGenerationConfig(count=4, seed=7, languages=["English"], task_types=["qa"])

    first = generate_anchor_prompts(
        ontology.knowledge,
        ontology.language_features,
        ontology.capabilities,
        ontology.conversation_types,
        config,
    )
    second = generate_anchor_prompts(
        ontology.knowledge,
        ontology.language_features,
        ontology.capabilities,
        ontology.conversation_types,
        config,
    )

    assert [item.id for item in first] == [item.id for item in second]
    assert len(first) == 4
    assert first[0].anchor_meta["task_type"] == "qa"
    removed_meta_key = "safety" + "_boundary"
    removed_prompt_phrase = "non" + "-authoritative"
    assert removed_meta_key not in first[0].anchor_meta
    assert removed_prompt_phrase not in first[0].messages[0]["content"]


def test_anchor_ontology_file_exposes_research_neutral_dimensions():
    ontology = load_anchor_ontology(Path("configs/anchor_ontology.json"))

    assert ontology.languages == ["English", "简体中文", "Español", "日本語"]
    assert ontology.knowledge.leaves
    assert ontology.language_features.leaves
    assert ontology.capabilities.leaves
    assert ontology.conversation_types.leaves
    assert {leaf["top_level"] for leaf in ontology.knowledge.leaves} >= {
        "science_exploration",
        "art_aesthetics",
        "philosophy_worldviews",
        "religion_myth_folklore",
        "esoterica_belief_systems",
        "society_events",
        "history_civilization",
        "culture_anthropology",
        "human_oddities",
        "future_speculation",
    }
    assert {leaf["leaf"] for leaf in ontology.capabilities.leaves} >= {"qa", "coding"}


def test_anchor_ontology_rejects_missing_required_sections(tmp_path):
    path = tmp_path / "bad_ontology.json"
    path.write_text(json.dumps({"languages": ["English"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required sections"):
        load_anchor_ontology(path)


def test_balanced_anchor_sampling_is_deterministic_and_covers_capability_groups():
    ontology = load_anchor_ontology(Path("configs/anchor_ontology.json"))
    config = AnchorGenerationConfig(
        count=100,
        seed=11,
        languages=ontology.languages,
        task_types=[str(leaf["leaf"]) for leaf in ontology.capabilities.leaves],
        input_generator_model="strong-input-generator",
        target_model="open-anchor-target",
        sampling_strategy="balanced",
    )

    first = generate_anchor_prompts(
        ontology.knowledge,
        ontology.language_features,
        ontology.capabilities,
        ontology.conversation_types,
        config,
    )
    second = generate_anchor_prompts(
        ontology.knowledge,
        ontology.language_features,
        ontology.capabilities,
        ontology.conversation_types,
        config,
    )

    assert [item.id for item in first] == [item.id for item in second]
    assert {item.anchor_meta["language"] for item in first} == set(ontology.languages)
    assert {item.anchor_meta["capability_top_level"] for item in first} == {
        "knowledge_response",
        "reasoning",
        "coding_and_data",
        "language_work",
        "agentic_behavior",
    }
    assert all(
        item.anchor_meta["input_generator_model"] == "strong-input-generator" for item in first
    )
    assert all(item.anchor_meta["target_model"] == "open-anchor-target" for item in first)


def test_task_type_filter_limits_capability_sampling():
    ontology = load_anchor_ontology(Path("configs/anchor_ontology.json"))
    config = AnchorGenerationConfig(
        count=20,
        seed=13,
        languages=["English"],
        task_types=["qa", "coding"],
    )

    prompts = generate_anchor_prompts(
        ontology.knowledge,
        ontology.language_features,
        ontology.capabilities,
        ontology.conversation_types,
        config,
    )

    assert {item.anchor_meta["task_type"] for item in prompts} <= {"qa", "coding"}


def test_prompt_shape_instructions_for_single_and_multi_turn():
    ontology = load_anchor_ontology(Path("configs/anchor_ontology.json"))
    config = AnchorGenerationConfig(
        count=20,
        seed=3,
        languages=["English"],
        task_types=[str(leaf["leaf"]) for leaf in ontology.capabilities.leaves],
    )

    prompts = generate_anchor_prompts(
        ontology.knowledge,
        ontology.language_features,
        ontology.capabilities,
        ontology.conversation_types,
        config,
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


def test_farthest_sampling_uses_embedding_sidecar(tmp_path):
    ontology_path = tmp_path / "anchor_ontology.json"
    ontology_path.write_text(json.dumps(_minimal_ontology_payload()), encoding="utf-8")
    ontology = load_anchor_ontology(ontology_path)
    embeddings = build_ontology_embeddings(
        ontology_path,
        embed_fn=lambda texts: hash_embed_texts(texts, dimensions=8),
        embedding_model="test-hash",
    )
    config = AnchorGenerationConfig(
        count=5,
        seed=1,
        languages=["English"],
        task_types=[str(leaf["leaf"]) for leaf in ontology.capabilities.leaves],
        sampling_strategy="farthest",
        ontology_embeddings=embeddings,
        ontology_sha256=embeddings.ontology_sha256,
        embedding_model=embeddings.embedding_model,
        embedding_distance=embeddings.distance,
    )

    prompts = generate_anchor_prompts(
        ontology.knowledge,
        ontology.language_features,
        ontology.capabilities,
        ontology.conversation_types,
        config,
    )

    assert len(prompts) == 5
    assert {item.anchor_meta["knowledge_domain"] for item in prompts[:3]} == {
        "domain_a -> topic -> alpha",
        "domain_b -> topic -> beta",
        "domain_c -> topic -> gamma",
    }
    assert all(item.anchor_meta["sampling_strategy"] == "farthest" for item in prompts)
    assert all(item.anchor_meta["embedding_model"] == "test-hash" for item in prompts)


def test_stale_embedding_sidecar_reports_regeneration_hint(tmp_path):
    ontology_path = tmp_path / "anchor_ontology.json"
    ontology_path.write_text(json.dumps(_minimal_ontology_payload()), encoding="utf-8")
    embeddings = build_ontology_embeddings(
        ontology_path,
        embed_fn=lambda texts: hash_embed_texts(texts, dimensions=8),
        embedding_model="test-hash",
    )
    sidecar_path = tmp_path / "embeddings.json"
    write_ontology_embeddings(embeddings, sidecar_path)
    payload = _minimal_ontology_payload()
    payload["knowledge_domains"]["domain_d"] = {"topic": ["delta"]}
    ontology_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="ontology-embed"):
        load_ontology_embeddings(sidecar_path, ontology_path)


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
    removed_count_key = "by_" + "safety" + "_boundary"
    assert removed_count_key not in manifest["counts"]
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


def test_system_persona_is_sampled_and_stored_in_meta():
    ontology = load_anchor_ontology(Path("configs/anchor_ontology.json"))
    config = AnchorGenerationConfig(
        count=20,
        seed=3,
        languages=["English"],
        task_types=[str(leaf["leaf"]) for leaf in ontology.capabilities.leaves],
        system_personas=["none", "one_sentence", "appropriate", "detailed"],
    )

    prompts = generate_anchor_prompts(
        ontology.knowledge,
        ontology.language_features,
        ontology.capabilities,
        ontology.conversation_types,
        config,
    )

    personas = {item.anchor_meta.get("system_persona") for item in prompts}
    assert personas <= {"none", "one_sentence", "appropriate", "detailed"}
    assert len(personas) >= 1
    assert all("system_persona" in item.anchor_meta for item in prompts)


def test_system_persona_none_by_default_when_not_configured():
    ontology = load_anchor_ontology(Path("configs/anchor_ontology.json"))
    config = AnchorGenerationConfig(
        count=5,
        seed=3,
        languages=["English"],
        task_types=["qa"],
        system_personas=None,
    )

    prompts = generate_anchor_prompts(
        ontology.knowledge,
        ontology.language_features,
        ontology.capabilities,
        ontology.conversation_types,
        config,
    )

    assert all(item.anchor_meta.get("system_persona") == "none" for item in prompts)


def test_system_persona_not_in_prompt_text():
    ontology = load_anchor_ontology(Path("configs/anchor_ontology.json"))
    config = AnchorGenerationConfig(
        count=5,
        seed=3,
        languages=["English"],
        task_types=["qa"],
        system_personas=["one_sentence", "detailed"],
    )

    prompts = generate_anchor_prompts(
        ontology.knowledge,
        ontology.language_features,
        ontology.capabilities,
        ontology.conversation_types,
        config,
    )

    for item in prompts:
        text = item.messages[0]["content"]
        assert "## Role" not in text
        assert "## Expertise" not in text
        assert "## Guidelines" not in text
        assert "## Constraints" not in text
        assert "one_sentence" not in text.lower()


def test_system_persona_multiple_runs_produces_varied_output():
    ontology = load_anchor_ontology(Path("configs/anchor_ontology.json"))
    config = AnchorGenerationConfig(
        count=100,
        seed=3,
        languages=["English"],
        task_types=["qa"],
        system_personas=["none", "one_sentence", "appropriate", "detailed"],
    )

    prompts = generate_anchor_prompts(
        ontology.knowledge,
        ontology.language_features,
        ontology.capabilities,
        ontology.conversation_types,
        config,
    )

    personas = [item.anchor_meta.get("system_persona") for item in prompts]
    # With 100 samples and 4 options, all should appear at least once
    assert set(personas) == {"none", "one_sentence", "appropriate", "detailed"}

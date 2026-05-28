import json
import threading
import time
from argparse import Namespace

import pytest

from ard.anchor.api_client import ChatAPIConfig, chat_api_config_from_env
from ard.anchor import api_client
from ard.anchor.bank import AnchorPrompt, GeneratedInputAnchor, write_jsonl
from ard.anchor.ontology import load_anchor_ontology
from ard.anchor.input_generator import parse_input_generator_messages, generate_anchor_inputs
from ard.anchor.pipeline import build_anchor_dataset_api
from ard.anchor.target import answer_generated_inputs_api


def _write_test_anchor_ontology(
    tmp_path,
    *,
    knowledge: dict,
    language_features: dict | None = None,
    capabilities: dict | None = None,
    conversation_types: dict | None = None,
):
    path = tmp_path / "anchor_ontology.json"
    path.write_text(
        json.dumps(
            {
                "languages": ["English"],
                "knowledge_domains": knowledge,
                "language_features": language_features or {"style": ["concise"]},
                "capabilities": capabilities or {"knowledge_response": ["qa"]},
                "conversation_types": conversation_types or {"single_turn": ["single_turn"]},
            }
        ),
        encoding="utf-8",
    )
    return load_anchor_ontology(path)


def test_api_env_file_parsing_and_no_key_serialization(tmp_path):
    env_path = tmp_path / "api.env"
    env_path.write_text(
        "\n".join(
            [
                "ARD_INPUT_GENERATOR_API_BASE=https://api.example.com/",
                "ARD_INPUT_GENERATOR_MODEL_NAME=test-model",
                "ARD_INPUT_GENERATOR_API_KEY=secret-key",
            ]
        ),
        encoding="utf-8",
    )
    input_path = tmp_path / "prompts.jsonl"
    output_path = tmp_path / "generated_inputs.jsonl"
    write_jsonl(
        [
            AnchorPrompt(
                id="a",
                messages=[{"role": "user", "content": "make a question"}],
                anchor_meta={"conversation_type": "single_turn"},
            )
        ],
        input_path,
    )
    config = chat_api_config_from_env(env_path, env_prefix="ARD_INPUT_GENERATOR")

    generate_anchor_inputs(
        api_config=config,
        input_path=input_path,
        output_path=output_path,
        chat_fn=lambda *_args: "A real user question?",
    )

    assert config.api_key == "secret-key"
    assert "secret-key" not in output_path.read_text(encoding="utf-8")


def test_chat_completion_falls_back_to_reasoning_content(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "reasoning_content": "fallback answer",
                            }
                        }
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request, **_kwargs):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(api_client.urllib.request, "urlopen", fake_urlopen)

    content = api_client.chat_completion(
        ChatAPIConfig("https://api.example.com", "model", "secret"),
        [{"role": "user", "content": "q"}],
    )

    assert content == "fallback answer"
    assert "max_tokens" not in captured["payload"]


def test_parse_input_generator_messages_single_turn_text():
    assert parse_input_generator_messages("How do I debug this?", "single_turn") == [
        {"role": "user", "content": "How do I debug this?"}
    ]


def test_parse_input_generator_messages_valid_multi_turn_json():
    text = json.dumps(
        [
            {"role": "user", "content": "My script fails."},
            {"role": "assistant", "content": "What error do you see?"},
            {"role": "user", "content": "It says KeyError."},
        ]
    )

    messages = parse_input_generator_messages(text, "troubleshooting_3_turn")

    assert messages[-1]["role"] == "user"
    assert len(messages) == 3


def test_parse_input_generator_messages_extracts_json_array_from_text():
    text = """
Here is the requested conversation:
[
  {"role": "user", "content": "My script fails."},
  {"role": "assistant", "content": "What error do you see?"},
  {"role": "user", "content": "It says KeyError."}
]
"""

    messages = parse_input_generator_messages(text, "troubleshooting_3_turn")

    assert messages[-1]["content"] == "It says KeyError."


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        json.dumps([{"role": "tool", "content": "bad"}]),
        json.dumps([{"role": "user", "content": ""}]),
        json.dumps(
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "final answer"},
            ]
        ),
    ],
)
def test_parse_input_generator_messages_rejects_invalid_multi_turn(text):
    with pytest.raises(ValueError):
        parse_input_generator_messages(text, "troubleshooting_3_turn")


def test_anchor_answer_api_writes_final_anchor_shape(tmp_path, monkeypatch):
    input_path = tmp_path / "generated_inputs.jsonl"
    output_path = tmp_path / "anchor_bank.jsonl"
    write_jsonl(
        [
            GeneratedInputAnchor(
                id="a",
                messages=[{"role": "user", "content": "What is async await?"}],
                input_generator_model="input-generator",
                anchor_meta={
                    "capability": "explanation",
                    "input_generator_model": "input-generator",
                },
            )
        ],
        input_path,
    )

    answer_generated_inputs_api(
        api_config=ChatAPIConfig(
            api_base="https://api.example.com",
            model_name="target",
            api_key="secret-key",
        ),
        input_path=input_path,
        output_path=output_path,
        chat_fn=lambda **_kwargs: "target answer",
    )

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert record["messages"] == [{"role": "user", "content": "What is async await?"}]
    assert record["target_answer"] == "target answer"
    assert record["input_generator_model"] == "input-generator"
    assert record["target_model"] == "target"
    assert "secret-key" not in output_path.read_text(encoding="utf-8")


def test_anchor_generate_inputs_cli_uses_input_generator_stage(tmp_path, monkeypatch):
    input_path = tmp_path / "prompts.jsonl"
    output_path = tmp_path / "generated_inputs.jsonl"
    env_path = tmp_path / "api.env"
    stats_path = tmp_path / "input_generation_stats.json"
    env_path.write_text(
        "ARD_INPUT_GENERATOR_API_BASE=https://api.example.com/\n"
        "ARD_INPUT_GENERATOR_MODEL_NAME=input-generator\n"
        "ARD_INPUT_GENERATOR_API_KEY=secret-key\n",
        encoding="utf-8",
    )
    write_jsonl(
        [
            AnchorPrompt(
                id="a",
                messages=[{"role": "user", "content": "make a question"}],
                anchor_meta={"conversation_type": "single_turn"},
            )
        ],
        input_path,
    )

    def fake_generate_anchor_inputs(**kwargs):
        write_jsonl(
            [
                GeneratedInputAnchor(
                    id="a",
                    messages=[{"role": "user", "content": "real question"}],
                    input_generator_model=kwargs["api_config"].model_name,
                    anchor_meta={"conversation_type": "single_turn"},
                )
            ],
            kwargs["output_path"],
        )
        stats_path.write_text(json.dumps({"input_count": 1, "kept_count": 1}), encoding="utf-8")

        class Stats:
            def to_dict(self):
                return {"input_count": 1, "kept_count": 1}

        return [], Stats()

    monkeypatch.setattr(
        "ard.anchor.input_generator.generate_anchor_inputs", fake_generate_anchor_inputs
    )
    from ard.cli import cmd_anchor_generate_inputs

    cmd_anchor_generate_inputs(
        Namespace(
            api_env_file=str(env_path),
            input=str(input_path),
            output=str(output_path),
            input_generator_model=None,
            max_new_tokens=10,
            temperature=0.1,
            top_p=1.0,
            timeout=1.0,
            max_retries=0,
            limit=None,
            stats_output=str(stats_path),
        )
    )

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert record["messages"] == [{"role": "user", "content": "real question"}]


def test_build_anchor_dataset_api_end_to_end_with_fake_chat(tmp_path):
    ontology = _write_test_anchor_ontology(
        tmp_path,
        knowledge={"general": {"topic": ["alpha", "beta", "gamma"]}},
        language_features={
            "style": ["concise"],
            "format": ["paragraph"],
            "difficulty": ["basic"],
            "context_length": ["short"],
            "noise": ["clean"],
            "answer_expectation": ["direct_answer"],
        },
    )

    input_counter = {"value": 0}

    def fake_input_generation_chat(_config, messages, *_args):
        input_counter["value"] += 1
        domain = messages[0]["content"].split("Domain: ", 1)[1].split(".", 1)[0]
        return f"Input {input_counter['value']}: {domain}"

    def fake_answer_chat(**kwargs):
        return "Answer for " + kwargs["messages"][-1]["content"]

    logs = []
    result = build_anchor_dataset_api(
        output_dir=tmp_path / "dataset",
        target_count=3,
        seed=1,
        knowledge=ontology.knowledge,
        language=ontology.language_features,
        capability=ontology.capabilities,
        conversation=ontology.conversation_types,
        languages=["English"],
        task_types=["qa"],
        input_generator_config=ChatAPIConfig(
            "https://api.example.com", "input-generator", "secret"
        ),
        target_config=ChatAPIConfig("https://api.example.com", "target", "secret"),
        batch_size=3,
        max_batches=1,
        input_generation_chat_fn=fake_input_generation_chat,
        target_answer_chat_fn=fake_answer_chat,
        logger=logs.append,
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "dataset" / "anchor_bank.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    manifest = json.loads((tmp_path / "dataset" / "manifest.json").read_text(encoding="utf-8"))

    assert result.final_count == 3
    assert len(records) == 3
    assert all(record["input_generator_model"] == "input-generator" for record in records)
    assert all(record["target_model"] == "target" for record in records)
    assert all("secret" not in json.dumps(record) for record in records)
    assert manifest["input_generation_stats"]["kept_count"] == 3
    assert manifest["target_answer_stats"]["kept_count"] == 3
    assert any("stage=build status=start" in line for line in logs)
    assert any("stage=input_generation" in line for line in logs)
    assert any("stage=target_answer" in line for line in logs)
    assert any("progress=" in line and "eta=" in line for line in logs)
    assert any("stage=build status=done" in line for line in logs)


def test_build_anchor_dataset_streams_target_answers_before_all_inputs_finish(tmp_path):
    ontology = _write_test_anchor_ontology(
        tmp_path,
        knowledge={"general": {"topic": ["alpha", "beta", "gamma", "delta"]}},
    )

    lock = threading.Lock()
    input_completed = {"count": 0}
    target_start_input_counts: list[int] = []
    logs: list[str] = []

    def fake_input_generation_chat(_config, _messages, *_args):
        time.sleep(0.05)
        with lock:
            input_completed["count"] += 1
            count = input_completed["count"]
        return f"unique input {count}"

    def fake_answer_chat(**kwargs):
        with lock:
            target_start_input_counts.append(input_completed["count"])
        return "useful answer for " + kwargs["messages"][-1]["content"]

    result = build_anchor_dataset_api(
        output_dir=tmp_path / "dataset",
        target_count=4,
        seed=1,
        knowledge=ontology.knowledge,
        language=ontology.language_features,
        capability=ontology.capabilities,
        conversation=ontology.conversation_types,
        languages=["English"],
        task_types=["qa"],
        input_generator_config=ChatAPIConfig(
            "https://api.example.com", "input-generator", "secret"
        ),
        target_config=ChatAPIConfig("https://api.example.com", "target", "secret"),
        batch_size=4,
        max_batches=1,
        input_generator_concurrency=2,
        target_concurrency=2,
        input_generation_chat_fn=fake_input_generation_chat,
        target_answer_chat_fn=fake_answer_chat,
        logger=logs.append,
    )
    manifest = json.loads((tmp_path / "dataset" / "manifest.json").read_text(encoding="utf-8"))
    target_log_positions = [
        index for index, line in enumerate(logs) if "stage=target_answer" in line
    ]
    input_log_positions = [
        index for index, line in enumerate(logs) if "stage=input_generation" in line
    ]

    assert result.final_count == 4
    assert min(target_start_input_counts) < 4
    assert target_log_positions[0] < input_log_positions[-1]
    assert manifest["generation_config"]["input_generator_concurrency"] == 2
    assert manifest["generation_config"]["target_concurrency"] == 2


def test_cli_logger_prefixes_timestamp(capsys):
    from ard.cli import _print_log

    _print_log("stage=build status=start")

    captured = capsys.readouterr()
    assert captured.out.startswith("ts=")
    assert " stage=build status=start\n" in captured.out


def test_build_anchor_dataset_attempt_count_does_not_refill_by_default(tmp_path):
    ontology = _write_test_anchor_ontology(
        tmp_path,
        knowledge={"general": {"topic": ["alpha", "beta", "gamma"]}},
    )

    answer_counter = {"value": 0}
    input_counter = {"value": 0}

    def fake_input_generation_chat(_config, _messages, *_args):
        input_counter["value"] += 1
        return f"unique input {input_counter['value']}"

    def fake_answer_chat(**_kwargs):
        answer_counter["value"] += 1
        if answer_counter["value"] == 1:
            return "bad"
        return f"useful answer {answer_counter['value']}"

    result = build_anchor_dataset_api(
        output_dir=tmp_path / "dataset",
        target_count=3,
        seed=1,
        knowledge=ontology.knowledge,
        language=ontology.language_features,
        capability=ontology.capabilities,
        conversation=ontology.conversation_types,
        languages=["English"],
        task_types=["qa"],
        input_generator_config=ChatAPIConfig(
            "https://api.example.com", "input-generator", "secret"
        ),
        target_config=ChatAPIConfig("https://api.example.com", "target", "secret"),
        batch_size=3,
        max_batches=3,
        input_generation_chat_fn=fake_input_generation_chat,
        target_answer_chat_fn=fake_answer_chat,
    )
    manifest = json.loads((tmp_path / "dataset" / "manifest.json").read_text(encoding="utf-8"))

    assert result.attempted_count == 3
    assert result.final_count == 2
    assert result.batches == 1
    assert manifest["attempted_count"] == 3
    assert manifest["final_count"] == 2
    assert manifest["filter_stats"]["dropped_too_short"] == 1


def test_build_anchor_dataset_refuses_non_empty_output_dir(tmp_path):
    output_dir = tmp_path / "dataset"
    output_dir.mkdir()
    (output_dir / "old.txt").write_text("previous run", encoding="utf-8")
    ontology = _write_test_anchor_ontology(tmp_path, knowledge={"general": ["alpha"]})

    with pytest.raises(FileExistsError, match="Output directory already exists"):
        build_anchor_dataset_api(
            output_dir=output_dir,
            target_count=1,
            seed=1,
            knowledge=ontology.knowledge,
            language=ontology.language_features,
            capability=ontology.capabilities,
            conversation=ontology.conversation_types,
            languages=["English"],
            task_types=["qa"],
            input_generator_config=ChatAPIConfig(
                "https://api.example.com", "input-generator", "secret"
            ),
            target_config=ChatAPIConfig("https://api.example.com", "target", "secret"),
            input_generation_chat_fn=lambda *_args: "unique input",
            target_answer_chat_fn=lambda **_kwargs: "useful answer",
        )


def test_build_anchor_dataset_exact_count_refills_when_enabled(tmp_path):
    ontology = _write_test_anchor_ontology(
        tmp_path,
        knowledge={"general": {"topic": ["alpha", "beta", "gamma"]}},
    )

    answer_counter = {"value": 0}
    input_counter = {"value": 0}

    def fake_input_generation_chat(_config, _messages, *_args):
        input_counter["value"] += 1
        return f"unique input {input_counter['value']}"

    def fake_answer_chat(**_kwargs):
        answer_counter["value"] += 1
        if answer_counter["value"] == 1:
            return "bad"
        return f"useful answer {answer_counter['value']}"

    result = build_anchor_dataset_api(
        output_dir=tmp_path / "dataset",
        target_count=3,
        seed=1,
        knowledge=ontology.knowledge,
        language=ontology.language_features,
        capability=ontology.capabilities,
        conversation=ontology.conversation_types,
        languages=["English"],
        task_types=["qa"],
        input_generator_config=ChatAPIConfig(
            "https://api.example.com", "input-generator", "secret"
        ),
        target_config=ChatAPIConfig("https://api.example.com", "target", "secret"),
        batch_size=3,
        max_batches=3,
        require_exact_count=True,
        input_generation_chat_fn=fake_input_generation_chat,
        target_answer_chat_fn=fake_answer_chat,
    )

    assert result.attempted_count == 4
    assert result.final_count == 3
    assert result.batches == 2


def test_build_anchor_dataset_filters_duplicates_across_exact_batches(tmp_path):
    ontology = _write_test_anchor_ontology(
        tmp_path,
        knowledge={"general": {"topic": ["alpha", "beta", "gamma"]}},
    )

    input_counter = {"value": 0}

    def fake_input_generation_chat(_config, _messages, *_args):
        input_counter["value"] += 1
        if input_counter["value"] <= 2:
            return "same generated input"
        return f"unique generated input {input_counter['value']}"

    def fake_answer_chat(**kwargs):
        return "useful answer for " + kwargs["messages"][-1]["content"]

    result = build_anchor_dataset_api(
        output_dir=tmp_path / "dataset",
        target_count=2,
        seed=1,
        knowledge=ontology.knowledge,
        language=ontology.language_features,
        capability=ontology.capabilities,
        conversation=ontology.conversation_types,
        languages=["English"],
        task_types=["qa"],
        input_generator_config=ChatAPIConfig(
            "https://api.example.com", "input-generator", "secret"
        ),
        target_config=ChatAPIConfig("https://api.example.com", "target", "secret"),
        batch_size=1,
        max_batches=3,
        require_exact_count=True,
        input_generation_chat_fn=fake_input_generation_chat,
        target_answer_chat_fn=fake_answer_chat,
    )
    manifest = json.loads((tmp_path / "dataset" / "manifest.json").read_text(encoding="utf-8"))

    assert result.attempted_count == 3
    assert result.final_count == 2
    assert result.batches == 3
    assert manifest["filter_stats"]["dropped_duplicate_prompt"] == 1


def test_anchor_build_dataset_command_is_registered():
    from ard.cli import build_parser

    args = build_parser().parse_args(
        [
            "anchor-build-dataset",
            "--output-dir",
            "out",
            "--target-count",
            "3",
        ]
    )

    assert args.func.__name__ == "cmd_anchor_build_dataset"
    assert args.target_count == 3
    assert args.input_generator_concurrency == 4
    assert args.target_concurrency == 4
    assert args.sampling_strategy == "farthest"


def test_ontology_embed_command_writes_hash_sidecar(tmp_path):
    from ard.cli import cmd_ontology_embed

    ontology = _write_test_anchor_ontology(
        tmp_path,
        knowledge={"general": {"topic": ["alpha", "beta"]}},
    )
    ontology_path = tmp_path / "anchor_ontology.json"
    output_path = tmp_path / "embeddings.json"

    cmd_ontology_embed(
        Namespace(
            ontology=str(ontology_path),
            output=str(output_path),
            backend="hash",
            model=None,
            api_env_file=None,
            batch_size=2,
            timeout=1.0,
            dimensions=8,
        )
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["embedding_model"] == "research-neutral-lexical-bootstrap-v1"
    assert payload["embedding_dimension"] == 8
    assert {item["section"] for item in payload["items"]} >= {
        "languages",
        "knowledge_domains",
        "capabilities",
        "conversation_types",
        "language_features",
    }
    assert ontology.knowledge.leaves

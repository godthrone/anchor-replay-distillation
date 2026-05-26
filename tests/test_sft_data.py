import json

from ard.sft.data import load_mixed_sft_samples
from ard.sft.collator import ARDDataCollator
from ard.sft.data import load_sft_samples


def test_load_mixed_sft_samples(tmp_path):
    hard_path = tmp_path / "hard.jsonl"
    anchor_path = tmp_path / "anchor.jsonl"
    hard_path.write_text(
        json.dumps(
            {
                "sample_type": "hard",
                "messages": [{"role": "user", "content": "extract"}],
                "target": '{"field":"value"}',
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    anchor_path.write_text(
        json.dumps(
            {
                "sample_type": "anchor",
                "messages": [{"role": "user", "content": "general question"}],
                "target_answer": "general answer",
                "target_model": "base",
                "anchor_meta": {"domain": "general"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    samples = load_mixed_sft_samples(hard_path, anchor_path)

    assert [sample.sample_type for sample in samples] == ["hard", "anchor"]
    assert samples[0].target == '{"field":"value"}'
    assert samples[1].target == "general answer"


def test_multi_turn_anchor_loads_and_collator_masks_prompt(tmp_path):
    anchor_path = tmp_path / "anchor.jsonl"
    anchor_path.write_text(
        json.dumps(
            {
                "sample_type": "anchor",
                "messages": [
                    {"role": "user", "content": "My script fails."},
                    {"role": "assistant", "content": "What error do you see?"},
                    {"role": "user", "content": "It says KeyError."},
                ],
                "target_answer": "Check the missing dictionary key.",
                "target_model": "base",
                "anchor_meta": {"conversation_type": "troubleshooting_3_turn"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    class CharTokenizer:
        pad_token_id = 0
        eos_token = "<eos>"
        chat_template = None

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(char) for char in text]}

    sample = load_sft_samples(anchor_path, "anchor")[0]
    batch = ARDDataCollator(CharTokenizer(), max_length=4096)([sample])
    labels = batch["labels"][0].tolist()
    non_masked = [value for value in labels if value != -100]

    assert len(sample.messages) == 3
    assert sample.target == "Check the missing dictionary key."
    assert len(non_masked) == len("Check the missing dictionary key.<eos>")
    assert labels[: labels.index(non_masked[0])] == [-100] * labels.index(non_masked[0])

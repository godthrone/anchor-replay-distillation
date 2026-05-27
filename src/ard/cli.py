from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any


def _read_yaml(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _read_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_optional_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _optional_int(value: str | None) -> int | None:
    if value is None or not str(value).strip():
        return None
    return int(value)


def _print_log(message: str) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"ts={timestamp} {message}", flush=True)


def _load_seed_paths(args: argparse.Namespace, config_data: dict[str, Any]) -> dict[str, Any]:
    ontology_cfg = config_data.get("ontology", {})
    return {
        "knowledge_path": args.knowledge_ontology or ontology_cfg.get("knowledge_path"),
        "language_path": args.language_ontology or ontology_cfg.get("language_path"),
        "capability_path": args.capability_ontology or ontology_cfg.get("capability_path"),
        "conversation_path": args.conversation_ontology or ontology_cfg.get("conversation_path"),
        "safety_path": args.safety_ontology or ontology_cfg.get("safety_path"),
        "knowledge_root": args.knowledge_root or ontology_cfg.get("knowledge_root"),
        "language_root": args.language_root or ontology_cfg.get("language_root"),
        "capability_root": args.capability_root or ontology_cfg.get("capability_root"),
        "conversation_root": args.conversation_root or ontology_cfg.get("conversation_root"),
        "safety_root": args.safety_root or ontology_cfg.get("safety_root"),
    }


def cmd_anchor_generate(args: argparse.Namespace) -> int:
    from ard.anchor import AnchorGenerationConfig, generate_anchor_prompts, load_ontology
    from ard.anchor.bank import write_jsonl
    from ard.anchor.sampler import DEFAULT_TASK_TYPES

    config_data = _read_yaml(args.config)
    generation_cfg = config_data.get("generation", {})
    paths = _load_seed_paths(args, config_data)
    if not paths["knowledge_path"] or not paths["language_path"]:
        raise SystemExit("--knowledge-ontology and --language-ontology are required")
    output_path = args.output or generation_cfg.get("output_path")
    if not output_path:
        raise SystemExit("--output is required")

    languages = (
        _csv(args.languages)
        or generation_cfg.get("languages")
        or ["English", "简体中文", "bilingual_zh_en"]
    )
    task_types = (
        _csv(args.task_types) or generation_cfg.get("task_types") or list(DEFAULT_TASK_TYPES)
    )
    prompt_config = AnchorGenerationConfig(
        count=args.count if args.count is not None else int(generation_cfg.get("count", 100)),
        seed=args.seed if args.seed is not None else int(generation_cfg.get("seed", 42)),
        languages=list(languages),
        task_types=list(task_types),
        language_features_per_prompt=(
            args.language_features_per_prompt
            if args.language_features_per_prompt is not None
            else int(generation_cfg.get("language_features_per_prompt", 2))
        ),
        input_generator_model=args.input_generator_model
        or str(generation_cfg.get("input_generator_model", "unspecified_input_generator_model")),
        target_model=args.target_model
        or str(generation_cfg.get("target_model", "unspecified_target_model")),
    )
    prompts = generate_anchor_prompts(
        knowledge=load_ontology(paths["knowledge_path"], root_key=paths["knowledge_root"]),
        language=load_ontology(paths["language_path"], root_key=paths["language_root"]),
        config=prompt_config,
        capability=(
            load_ontology(paths["capability_path"], root_key=paths["capability_root"])
            if paths["capability_path"]
            else None
        ),
        conversation=(
            load_ontology(paths["conversation_path"], root_key=paths["conversation_root"])
            if paths["conversation_path"]
            else None
        ),
        safety=(
            load_ontology(paths["safety_path"], root_key=paths["safety_root"])
            if paths["safety_path"]
            else None
        ),
    )
    write_jsonl(prompts, output_path)
    print(f"Wrote {len(prompts)} anchor prompts to {output_path}")
    return 0


def cmd_anchor_generate_inputs(args: argparse.Namespace) -> int:
    from ard.anchor.api_client import chat_api_config_from_env
    from ard.anchor.input_generator import generate_anchor_inputs

    api_config = chat_api_config_from_env(
        args.api_env_file, args.input_generator_model, env_prefix="ARD_INPUT_GENERATOR"
    )
    generated_inputs, stats = generate_anchor_inputs(
        api_config=api_config,
        input_path=args.input,
        output_path=args.output,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.timeout,
        max_retries=args.max_retries,
        limit=args.limit,
        stats_output=args.stats_output,
    )
    print(json.dumps(stats.to_dict(), ensure_ascii=False))
    print(f"Wrote {len(generated_inputs)} generated input anchors to {args.output}")
    return 0


def cmd_anchor_generate_target_answers(args: argparse.Namespace) -> int:
    if args.backend == "api":
        from ard.anchor.api_client import chat_api_config_from_env
        from ard.anchor.target import answer_generated_inputs_api

        api_config = chat_api_config_from_env(
            args.api_env_file, args.target_model, env_prefix="ARD_TARGET"
        )
        answered = answer_generated_inputs_api(
            api_config=api_config,
            input_path=args.input,
            output_path=args.output,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
            limit=args.limit,
            stats_output=args.stats_output,
        )
    else:
        from ard.anchor.target import answer_anchor_prompts

        if not args.model_path:
            raise SystemExit("--model-path is required when --backend hf")
        answered = answer_anchor_prompts(
            model_path=args.model_path,
            input_path=args.input,
            output_path=args.output,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=args.torch_dtype,
            max_new_tokens=args.max_new_tokens or 512,
            max_prompt_length=args.max_prompt_length,
            temperature=args.temperature,
            top_p=args.top_p,
            limit=args.limit,
        )
    print(f"Wrote {len(answered)} answered anchors to {args.output}")
    return 0


def cmd_anchor_filter(args: argparse.Namespace) -> int:
    from ard.anchor.bank import (
        filter_target_answer_anchors,
        read_target_answer_anchors,
        write_jsonl,
    )
    from ard.anchor.manifest import build_anchor_manifest, write_manifest

    anchors = read_target_answer_anchors(args.input)
    kept, stats = filter_target_answer_anchors(
        anchors,
        min_answer_chars=args.min_answer_chars,
        max_answer_chars=args.max_answer_chars,
    )
    write_jsonl(kept, args.output)
    stats_payload = stats.to_dict()
    print(json.dumps(stats_payload, ensure_ascii=False))
    if args.stats_output:
        _write_json(args.stats_output, stats_payload)
    if args.manifest_output:
        manifest = build_anchor_manifest(
            target_model=args.target_model or (kept[0].target_model if kept else ""),
            generation_config={},
            anchors=kept,
            filter_stats=stats,
            seed=args.seed,
            input_generation_stats=_read_optional_json(args.input_generation_stats),
            target_answer_stats=_read_optional_json(args.target_answer_stats),
        )
        write_manifest(manifest, args.manifest_output)
    return 0


def cmd_anchor_split(args: argparse.Namespace) -> int:
    from ard.anchor.bank import read_target_answer_anchors, split_target_answer_anchors, write_jsonl

    anchors = read_target_answer_anchors(args.input)
    train, eval_items = split_target_answer_anchors(
        anchors, eval_ratio=args.eval_ratio, seed=args.seed
    )
    write_jsonl(train, args.train_output)
    write_jsonl(eval_items, args.eval_output)
    print(
        json.dumps(
            {"input": len(anchors), "train": len(train), "eval": len(eval_items)},
            ensure_ascii=False,
        )
    )
    return 0


def cmd_anchor_build_dataset(args: argparse.Namespace) -> int:
    from ard.anchor.api_client import chat_api_config_from_env
    from ard.anchor.ontology import load_ontology
    from ard.anchor.pipeline import build_anchor_dataset_api
    from ard.anchor.sampler import DEFAULT_TASK_TYPES

    config_data = _read_yaml(args.config)
    generation_cfg = config_data.get("generation", {})
    paths = _load_seed_paths(args, config_data)
    if not paths["knowledge_path"] or not paths["language_path"]:
        raise SystemExit("--knowledge-ontology and --language-ontology are required")

    languages = (
        _csv(args.languages)
        or generation_cfg.get("languages")
        or ["English", "简体中文", "bilingual_zh_en"]
    )
    task_types = (
        _csv(args.task_types) or generation_cfg.get("task_types") or list(DEFAULT_TASK_TYPES)
    )
    result = build_anchor_dataset_api(
        output_dir=args.output_dir,
        target_count=args.target_count,
        seed=args.seed if args.seed is not None else int(generation_cfg.get("seed", 42)),
        knowledge=load_ontology(paths["knowledge_path"], root_key=paths["knowledge_root"]),
        language=load_ontology(paths["language_path"], root_key=paths["language_root"]),
        capability=(
            load_ontology(paths["capability_path"], root_key=paths["capability_root"])
            if paths["capability_path"]
            else None
        ),
        conversation=(
            load_ontology(paths["conversation_path"], root_key=paths["conversation_root"])
            if paths["conversation_path"]
            else None
        ),
        safety=(
            load_ontology(paths["safety_path"], root_key=paths["safety_root"])
            if paths["safety_path"]
            else None
        ),
        languages=list(languages),
        task_types=list(task_types),
        input_generator_config=chat_api_config_from_env(
            args.api_env_file, args.input_generator_model, env_prefix="ARD_INPUT_GENERATOR"
        ),
        target_config=chat_api_config_from_env(
            args.api_env_file, args.target_model, env_prefix="ARD_TARGET"
        ),
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        input_generator_max_tokens=args.input_generator_max_tokens,
        target_max_tokens=args.target_max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.timeout,
        max_retries=args.max_retries,
        input_generator_concurrency=args.input_generator_concurrency,
        target_concurrency=args.target_concurrency,
        require_exact_count=args.require_exact_count,
        min_target_answer_chars=args.min_target_answer_chars,
        max_target_answer_chars=args.max_target_answer_chars,
        logger=None if args.quiet else _print_log,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if (not args.require_exact_count or result.final_count >= result.target_count) else 2


def cmd_train_sft_ard(args: argparse.Namespace) -> int:
    from ard.sft import ARDSFTTrainer
    from ard.sft.schema import ARDSFTConfig

    ARDSFTTrainer(ARDSFTConfig.from_yaml(args.config)).train()
    return 0


def cmd_eval_forgetting(args: argparse.Namespace) -> int:
    anchors = _read_jsonl_records(args.anchor_eval)
    completions = {
        str(item.get("id", idx)): item
        for idx, item in enumerate(_read_jsonl_records(args.completions))
    }
    exact = contains = missing = scored = 0
    length_ratios: list[float] = []
    for idx, anchor in enumerate(anchors):
        key = str(anchor.get("id", idx))
        completion_record = completions.get(key)
        if completion_record is None:
            missing += 1
            continue
        target = str(anchor.get("target_answer", "")).strip()
        completion = str(
            completion_record.get(
                "completion", completion_record.get("response", completion_record.get("output", ""))
            )
        ).strip()
        if not target or not completion:
            continue
        scored += 1
        norm_target = " ".join(target.lower().split())
        norm_completion = " ".join(completion.lower().split())
        exact += int(norm_target == norm_completion)
        contains += int(norm_target in norm_completion or norm_completion in norm_target)
        length_ratios.append(len(completion) / max(len(target), 1))
    payload = {
        "anchors": len(anchors),
        "scored": scored,
        "missing": missing,
        "exact_match": exact / scored if scored else 0,
        "contains_match": contains / scored if scored else 0,
        "mean_length_ratio": sum(length_ratios) / len(length_ratios) if length_ratios else 0,
    }
    print(json.dumps(payload, ensure_ascii=False))
    if args.output:
        _write_json(args.output, payload)
    return 0


def _add_seed_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--knowledge-ontology")
    parser.add_argument("--language-ontology")
    parser.add_argument("--capability-ontology")
    parser.add_argument("--conversation-ontology")
    parser.add_argument("--safety-ontology")
    parser.add_argument("--knowledge-root")
    parser.add_argument("--language-root")
    parser.add_argument("--capability-root")
    parser.add_argument("--conversation-root")
    parser.add_argument("--safety-root")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ard", description="Anchor Replay Distillation utilities."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    anchor_generate = subparsers.add_parser("anchor-generate", help="Generate anchor meta prompts.")
    anchor_generate.add_argument("--config", default="configs/anchor_generation.yaml")
    _add_seed_args(anchor_generate)
    anchor_generate.add_argument("--output", "-o")
    anchor_generate.add_argument("--count", type=int)
    anchor_generate.add_argument("--seed", type=int)
    anchor_generate.add_argument("--languages")
    anchor_generate.add_argument("--task-types")
    anchor_generate.add_argument("--language-features-per-prompt", type=int)
    anchor_generate.add_argument("--input-generator-model")
    anchor_generate.add_argument("--target-model")
    anchor_generate.set_defaults(func=cmd_anchor_generate)

    anchor_generate_inputs = subparsers.add_parser(
        "anchor-generate-inputs",
        help="Generate real user messages with an API input generator model.",
    )
    anchor_generate_inputs.add_argument("--api-env-file")
    anchor_generate_inputs.add_argument("--input", "-i", required=True)
    anchor_generate_inputs.add_argument("--output", "-o", required=True)
    anchor_generate_inputs.add_argument("--input-generator-model")
    anchor_generate_inputs.add_argument(
        "--max-new-tokens", type=int, help="Explicit override only; omitted by default."
    )
    anchor_generate_inputs.add_argument("--temperature", type=float, default=0.7)
    anchor_generate_inputs.add_argument("--top-p", type=float, default=0.95)
    anchor_generate_inputs.add_argument("--timeout", type=float, default=60)
    anchor_generate_inputs.add_argument("--max-retries", type=int, default=2)
    anchor_generate_inputs.add_argument("--limit", type=int)
    anchor_generate_inputs.add_argument("--stats-output")
    anchor_generate_inputs.set_defaults(func=cmd_anchor_generate_inputs)

    anchor_target_answers = subparsers.add_parser(
        "anchor-generate-target-answers",
        help="Generate target answers with a local HF model or API target model.",
    )
    anchor_target_answers.add_argument("--backend", choices=["hf", "api"], default="hf")
    anchor_target_answers.add_argument("--model-path")
    anchor_target_answers.add_argument("--api-env-file")
    anchor_target_answers.add_argument("--target-model")
    anchor_target_answers.add_argument("--input", "-i", required=True)
    anchor_target_answers.add_argument("--output", "-o", required=True)
    anchor_target_answers.add_argument(
        "--trust-remote-code", action=argparse.BooleanOptionalAction, default=True
    )
    anchor_target_answers.add_argument("--torch-dtype", default="bfloat16")
    anchor_target_answers.add_argument(
        "--max-new-tokens",
        type=int,
        help="Explicit override only; omitted by default for API backend.",
    )
    anchor_target_answers.add_argument("--max-prompt-length", type=int, default=2048)
    anchor_target_answers.add_argument("--temperature", type=float, default=0.7)
    anchor_target_answers.add_argument("--top-p", type=float, default=0.95)
    anchor_target_answers.add_argument("--timeout", type=float, default=60)
    anchor_target_answers.add_argument("--limit", type=int)
    anchor_target_answers.add_argument("--stats-output")
    anchor_target_answers.set_defaults(func=cmd_anchor_generate_target_answers)

    anchor_filter = subparsers.add_parser(
        "anchor-filter", help="Filter answered anchors and optionally write manifest."
    )
    anchor_filter.add_argument("--input", "-i", required=True)
    anchor_filter.add_argument("--output", "-o", required=True)
    anchor_filter.add_argument("--min-answer-chars", type=int, default=8)
    anchor_filter.add_argument("--max-answer-chars", type=_optional_int)
    anchor_filter.add_argument("--stats-output")
    anchor_filter.add_argument("--manifest-output")
    anchor_filter.add_argument("--input-generation-stats")
    anchor_filter.add_argument("--target-answer-stats")
    anchor_filter.add_argument("--target-model")
    anchor_filter.add_argument("--seed", type=int)
    anchor_filter.set_defaults(func=cmd_anchor_filter)

    anchor_split = subparsers.add_parser(
        "anchor-split", help="Split filtered anchors into train/eval JSONL."
    )
    anchor_split.add_argument("--input", "-i", required=True)
    anchor_split.add_argument("--train-output", required=True)
    anchor_split.add_argument("--eval-output", required=True)
    anchor_split.add_argument("--eval-ratio", type=float, default=0.1)
    anchor_split.add_argument("--seed", type=int, default=42)
    anchor_split.set_defaults(func=cmd_anchor_split)

    anchor_build = subparsers.add_parser(
        "anchor-build-dataset", help="Build a two-hop API ARD dataset end to end."
    )
    anchor_build.add_argument("--config", default="configs/anchor_generation.yaml")
    anchor_build.add_argument("--api-env-file")
    anchor_build.add_argument("--output-dir", required=True)
    anchor_build.add_argument("--target-count", type=int, default=100)
    anchor_build.add_argument("--batch-size", type=int, default=50)
    anchor_build.add_argument("--max-batches", type=int, default=5)
    anchor_build.add_argument("--seed", type=int)
    anchor_build.add_argument("--input-generator-model")
    anchor_build.add_argument("--target-model")
    _add_seed_args(anchor_build)
    anchor_build.add_argument("--languages")
    anchor_build.add_argument("--task-types")
    anchor_build.add_argument(
        "--input-generator-max-tokens",
        type=int,
        help="Explicit override only; omitted by default.",
    )
    anchor_build.add_argument(
        "--target-max-tokens", type=int, help="Explicit override only; omitted by default."
    )
    anchor_build.add_argument("--temperature", type=float, default=0.7)
    anchor_build.add_argument("--top-p", type=float, default=0.95)
    anchor_build.add_argument("--timeout", type=float, default=60)
    anchor_build.add_argument("--max-retries", type=int, default=2)
    anchor_build.add_argument("--input-generator-concurrency", type=int, default=4)
    anchor_build.add_argument("--target-concurrency", type=int, default=4)
    anchor_build.add_argument("--require-exact-count", action="store_true")
    anchor_build.add_argument("--min-target-answer-chars", type=int, default=8)
    anchor_build.add_argument("--max-target-answer-chars", type=_optional_int)
    anchor_build.add_argument("--quiet", action="store_true")
    anchor_build.set_defaults(func=cmd_anchor_build_dataset)

    train_sft = subparsers.add_parser(
        "train-sft-ard", help="Run experimental ARD-SFT LoRA training."
    )
    train_sft.add_argument("--config", "-c", required=True)
    train_sft.set_defaults(func=cmd_train_sft_ard)

    eval_forgetting = subparsers.add_parser(
        "eval-forgetting", help="Compare anchor eval completions to target answers."
    )
    eval_forgetting.add_argument("--anchor-eval", required=True)
    eval_forgetting.add_argument("--completions", required=True)
    eval_forgetting.add_argument("--output")
    eval_forgetting.set_defaults(func=cmd_eval_forgetting)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()

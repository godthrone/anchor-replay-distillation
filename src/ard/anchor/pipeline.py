from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from ard.anchor.api_client import ChatAPIConfig, chat_completion
from ard.anchor.bank import (
    TargetAnswerAnchor,
    filter_target_answer_anchors,
    normalize_text,
    write_jsonl,
)
from ard.anchor.manifest import build_anchor_manifest, write_manifest
from ard.anchor.ontology import Ontology
from ard.anchor.input_generator import generate_anchor_inputs
from ard.anchor.sampler import AnchorGenerationConfig, generate_anchor_prompts
from ard.anchor.target import answer_generated_inputs_api

ChatFn = Callable[..., str]
LogFn = Callable[[str], None]


@dataclass(slots=True)
class DatasetBuildResult:
    output_dir: str
    target_count: int
    attempted_count: int
    final_count: int
    batches: int
    anchor_bank_path: str
    filtered_path: str
    manifest_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_anchor_dataset_api(
    output_dir: str | Path,
    target_count: int,
    seed: int,
    knowledge: Ontology,
    language: Ontology,
    capability: Ontology | None,
    conversation: Ontology | None,
    safety: Ontology | None,
    languages: list[str],
    task_types: list[str],
    input_generator_config: ChatAPIConfig,
    target_config: ChatAPIConfig,
    batch_size: int = 50,
    max_batches: int = 5,
    input_generator_max_tokens: int | None = None,
    target_max_tokens: int | None = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    timeout: float = 60,
    max_retries: int = 2,
    require_exact_count: bool = False,
    min_target_answer_chars: int = 8,
    max_target_answer_chars: int | None = None,
    input_generation_chat_fn: ChatFn | None = None,
    target_answer_chat_fn: ChatFn | None = None,
    logger: LogFn | None = None,
) -> DatasetBuildResult:
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_batches <= 0:
        raise ValueError("max_batches must be positive")
    if min_target_answer_chars < 0:
        raise ValueError("min_target_answer_chars must be non-negative")
    if max_target_answer_chars is not None and max_target_answer_chars < min_target_answer_chars:
        raise ValueError(
            "max_target_answer_chars must be greater than or equal to min_target_answer_chars"
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_kept: list[TargetAnswerAnchor] = []
    attempted_count = 0
    batches_run = 0
    input_generation_stats_total = {
        "input_count": 0,
        "kept_count": 0,
        "failed_count": 0,
        "parse_failed_count": 0,
        "api_failed_count": 0,
    }
    target_answer_stats_total = {
        "input_count": 0,
        "kept_count": 0,
        "failed_count": 0,
        "api_failed_count": 0,
    }
    filter_stats_total = {
        "input_count": 0,
        "kept_count": 0,
        "dropped_empty": 0,
        "dropped_failure": 0,
        "dropped_too_short": 0,
        "dropped_too_long": 0,
        "dropped_duplicate_prompt": 0,
        "dropped_duplicate_answer": 0,
    }

    run_start = perf_counter()
    if logger:
        logger(
            "stage=build status=start "
            f"target={target_count} batch_size={batch_size} max_batches={max_batches} "
            f"require_exact_count={require_exact_count} "
            f"input_generator_model={input_generator_config.model_name} target_model={target_config.model_name}"
        )

    batch_limit = max_batches if require_exact_count else 1
    for batch_idx in range(batch_limit):
        remaining = target_count - len(all_kept)
        if require_exact_count and remaining <= 0:
            break
        current_count = min(batch_size, remaining) if require_exact_count else target_count
        batches_run += 1
        batch_start = perf_counter()
        batch_dir = out / f"batch_{batch_idx + 1:03d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        prompts_path = batch_dir / "anchor_prompts.jsonl"
        generated_inputs_path = batch_dir / "anchor_generated_inputs.jsonl"
        answered_path = batch_dir / "anchor_target_answers.jsonl"
        input_generation_stats_path = batch_dir / "input_generation_stats.json"
        target_answer_stats_path = batch_dir / "target_answer_stats.json"

        config = AnchorGenerationConfig(
            count=current_count,
            seed=seed + batch_idx,
            languages=languages,
            task_types=task_types,
            input_generator_model=input_generator_config.model_name,
            target_model=target_config.model_name,
        )
        prompts = generate_anchor_prompts(
            knowledge=knowledge,
            language=language,
            config=config,
            capability=capability,
            conversation=conversation,
            safety=safety,
        )
        write_jsonl(prompts, prompts_path)
        if logger:
            logger(
                "stage=batch "
                f"batch={batch_idx + 1}/{batch_limit} status=prompts_generated "
                f"prompts={len(prompts)} cumulative_kept={len(all_kept)} "
                f"elapsed={perf_counter() - batch_start:.1f}s total={perf_counter() - run_start:.1f}s"
            )
        attempted_count += len(prompts)
        _generated_inputs, q_stats = generate_anchor_inputs(
            api_config=input_generator_config,
            input_path=prompts_path,
            output_path=generated_inputs_path,
            max_tokens=input_generator_max_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            max_retries=max_retries,
            stats_output=input_generation_stats_path,
            chat_fn=input_generation_chat_fn or chat_completion,
            logger=logger,
        )
        if logger:
            logger(
                "stage=batch "
                f"batch={batch_idx + 1}/{batch_limit} status=input_generation_done "
                f"input={q_stats.input_count} kept={q_stats.kept_count} "
                f"failed={q_stats.failed_count} api_failed={q_stats.api_failed_count} "
                f"parse_failed={q_stats.parse_failed_count} "
                f"elapsed={perf_counter() - batch_start:.1f}s total={perf_counter() - run_start:.1f}s"
            )
        answered = answer_generated_inputs_api(
            api_config=target_config,
            input_path=generated_inputs_path,
            output_path=answered_path,
            max_tokens=target_max_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            stats_output=target_answer_stats_path,
            chat_fn=target_answer_chat_fn or chat_completion,
            logger=logger,
        )
        existing_prompts = {
            normalize_text("\n\n".join(msg.get("content", "") for msg in item.messages))
            for item in all_kept
        }
        existing_answers = {normalize_text(item.target_answer.strip()) for item in all_kept}
        kept, f_stats = filter_target_answer_anchors(
            answered,
            min_answer_chars=min_target_answer_chars,
            max_answer_chars=max_target_answer_chars,
            existing_normalized_prompts=existing_prompts,
            existing_normalized_answers=existing_answers,
        )
        all_kept.extend(kept)

        for key, value in q_stats.to_dict().items():
            input_generation_stats_total[key] += value
        for key, value in json.loads(target_answer_stats_path.read_text(encoding="utf-8")).items():
            target_answer_stats_total[key] += int(value)
        for key, value in f_stats.to_dict().items():
            filter_stats_total[key] += value
        if logger:
            logger(
                "stage=batch "
                f"batch={batch_idx + 1}/{batch_limit} status=filter_done "
                f"answered={len(answered)} kept_after_filter={len(kept)} "
                f"cumulative_kept={len(all_kept)} target={target_count} "
                f"dropped_empty={f_stats.dropped_empty} dropped_failure={f_stats.dropped_failure} "
                f"dropped_too_long={f_stats.dropped_too_long} "
                f"dropped_duplicate_prompt={f_stats.dropped_duplicate_prompt} "
                f"elapsed={perf_counter() - batch_start:.1f}s total={perf_counter() - run_start:.1f}s"
            )

    final = all_kept[:target_count] if require_exact_count else all_kept
    anchor_bank_path = out / "anchor_bank.jsonl"
    filtered_path = out / "anchor_filtered.jsonl"
    manifest_path = out / "manifest.json"
    stats_path = out / "build_stats.json"
    input_generation_stats_path = out / "input_generation_stats.json"
    target_answer_stats_path = out / "target_answer_stats.json"

    write_jsonl(final, anchor_bank_path)
    write_jsonl(final, filtered_path)
    input_generation_stats_path.write_text(
        json.dumps(input_generation_stats_total, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    target_answer_stats_path.write_text(
        json.dumps(target_answer_stats_total, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = build_anchor_manifest(
        target_model=target_config.model_name,
        generation_config={
            "target_count": target_count,
            "batch_size": batch_size,
            "max_batches": max_batches,
            "seed": seed,
            "require_exact_count": require_exact_count,
            "min_target_answer_chars": min_target_answer_chars,
            "max_target_answer_chars": max_target_answer_chars,
        },
        anchors=final,
        seed=seed,
        requested_count=target_count,
        attempted_count=attempted_count,
        input_generation_stats=input_generation_stats_total,
        target_answer_stats=target_answer_stats_total,
        filter_stats=filter_stats_total,
    )
    write_manifest(manifest, manifest_path)
    result = DatasetBuildResult(
        output_dir=str(out),
        target_count=target_count,
        attempted_count=attempted_count,
        final_count=len(final),
        batches=batches_run,
        anchor_bank_path=str(anchor_bank_path),
        filtered_path=str(filtered_path),
        manifest_path=str(manifest_path),
    )
    stats_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if logger:
        logger(
            "stage=build status=done "
            f"attempted={attempted_count} final_count={result.final_count} target={target_count} "
            f"batches={result.batches} "
            f"total={perf_counter() - run_start:.1f}s anchor_bank={result.anchor_bank_path}"
        )
        filter_input_count = filter_stats_total["input_count"]
        if filter_input_count:
            filter_rate = (
                filter_input_count - filter_stats_total["kept_count"]
            ) / filter_input_count
            if filter_rate > 0.05:
                logger(
                    "stage=build status=filter_rate_high "
                    f"filter_rate={filter_rate:.2%} filter_stats={json.dumps(filter_stats_total, ensure_ascii=False)}"
                )
    return result

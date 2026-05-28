from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from ard.anchor.api_client import ChatAPIConfig, chat_completion
from ard.anchor.bank import (
    AnchorPrompt,
    GeneratedInputAnchor,
    TargetAnswerAnchor,
    filter_target_answer_anchors,
    normalize_text,
    write_jsonl,
)
from ard.anchor.manifest import build_anchor_manifest, write_manifest
from ard.anchor.ontology import Ontology
from ard.anchor.input_generator import InputGenerationStats, generate_one_anchor_input
from ard.anchor.sampler import AnchorGenerationConfig, SamplingStrategy, generate_anchor_prompts
from ard.anchor.embeddings import OntologyEmbeddings
from ard.anchor.target import TargetAnswerStats, answer_one_generated_input_api

ChatFn = Callable[..., str]
LogFn = Callable[[str], None]


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _progress_fields(completed: int, total: int, start: float) -> str:
    now = perf_counter()
    elapsed = max(now - start, 0.0)
    percent = 100.0 if total == 0 else min(100.0, completed / total * 100.0)
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = max(total - completed, 0)
    eta = remaining / rate if rate > 0 else None
    return (
        f"progress={percent:.1f}% done={completed}/{total} "
        f"rate={rate:.2f}/s eta={_format_duration(eta)}"
    )


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


def _run_streaming_batch(
    prompts: list[AnchorPrompt],
    input_generator_config: ChatAPIConfig,
    target_config: ChatAPIConfig,
    input_generator_max_tokens: int | None,
    target_max_tokens: int | None,
    temperature: float,
    top_p: float,
    timeout: float,
    max_retries: int,
    input_generator_concurrency: int,
    target_concurrency: int,
    input_generation_chat_fn: ChatFn,
    target_answer_chat_fn: ChatFn,
    logger: LogFn | None,
    start: float,
) -> tuple[
    list[GeneratedInputAnchor], list[TargetAnswerAnchor], InputGenerationStats, TargetAnswerStats
]:
    input_stats = InputGenerationStats(input_count=len(prompts))
    target_stats = TargetAnswerStats()
    generated: list[tuple[int, GeneratedInputAnchor]] = []
    answered: list[tuple[int, TargetAnswerAnchor]] = []
    input_started_at: dict[Future[GeneratedInputAnchor], float] = {}
    input_indexes: dict[Future[GeneratedInputAnchor], int] = {}
    target_started_at: dict[Future[TargetAnswerAnchor], float] = {}
    target_indexes: dict[Future[TargetAnswerAnchor], int] = {}
    total_work_units = len(prompts) * 2

    def completed_units() -> int:
        input_done = len(generated) + input_stats.failed_count
        target_done_or_skipped = (
            len(answered) + target_stats.failed_count + input_stats.failed_count
        )
        return input_done + target_done_or_skipped

    with (
        ThreadPoolExecutor(
            max_workers=input_generator_concurrency,
            thread_name_prefix="ard-input-generator",
        ) as input_pool,
        ThreadPoolExecutor(
            max_workers=target_concurrency,
            thread_name_prefix="ard-target",
        ) as target_pool,
    ):
        for idx, item in enumerate(prompts, start=1):
            input_future = input_pool.submit(
                generate_one_anchor_input,
                item=item,
                api_config=input_generator_config,
                max_tokens=input_generator_max_tokens,
                temperature=temperature,
                top_p=top_p,
                timeout=timeout,
                max_retries=max_retries,
                chat_fn=input_generation_chat_fn,
            )
            input_indexes[input_future] = idx
            input_started_at[input_future] = perf_counter()

        while input_indexes or target_indexes:
            pending_futures: set[Future[Any]] = set(input_indexes)
            pending_futures.update(target_indexes)
            done, _pending = wait(
                pending_futures,
                return_when=FIRST_COMPLETED,
            )
            for completed_future in done:
                if completed_future in input_indexes:
                    idx = input_indexes.pop(completed_future)
                    try:
                        generated_item = completed_future.result()
                    except Exception as exc:  # noqa: BLE001
                        input_stats.failed_count += 1
                        if "request failed" in str(exc):
                            input_stats.api_failed_count += 1
                        else:
                            input_stats.parse_failed_count += 1
                        if logger:
                            logger(
                                "stage=input_generation "
                                f"idx={idx}/{len(prompts)} status=failed "
                                f"kept={len(generated)} failed={input_stats.failed_count} "
                                f"api_failed={input_stats.api_failed_count} "
                                f"parse_failed={input_stats.parse_failed_count} "
                                f"error={type(exc).__name__} "
                                f"elapsed={perf_counter() - input_started_at[completed_future]:.1f}s "
                                f"total={perf_counter() - start:.1f}s "
                                f"{_progress_fields(completed_units(), total_work_units, start)}"
                            )
                        continue

                    generated.append((idx, generated_item))
                    if logger:
                        logger(
                            "stage=input_generation "
                            f"idx={idx}/{len(prompts)} status=kept "
                            f"conversation={generated_item.anchor_meta.get('conversation_type', 'unknown')} "
                            f"kept={len(generated)} failed={input_stats.failed_count} "
                            f"elapsed={perf_counter() - input_started_at[completed_future]:.1f}s "
                            f"total={perf_counter() - start:.1f}s "
                            f"{_progress_fields(completed_units(), total_work_units, start)}"
                        )

                    target_future = target_pool.submit(
                        answer_one_generated_input_api,
                        item=generated_item,
                        api_config=target_config,
                        max_tokens=target_max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        timeout=timeout,
                        chat_fn=target_answer_chat_fn,
                    )
                    target_stats.input_count += 1
                    target_indexes[target_future] = idx
                    target_started_at[target_future] = perf_counter()
                    continue

                idx = target_indexes.pop(completed_future)
                try:
                    answered_item = completed_future.result()
                except Exception as exc:  # noqa: BLE001
                    target_stats.failed_count += 1
                    target_stats.api_failed_count += 1
                    if logger:
                        logger(
                            "stage=target_answer "
                            f"idx={idx}/{len(prompts)} status=failed "
                            f"kept={len(answered)} failed={target_stats.failed_count} "
                            f"error={type(exc).__name__} "
                            f"elapsed={perf_counter() - target_started_at[completed_future]:.1f}s "
                            f"total={perf_counter() - start:.1f}s "
                            f"{_progress_fields(completed_units(), total_work_units, start)}"
                        )
                    continue

                answered.append((idx, answered_item))
                if logger:
                    logger(
                        "stage=target_answer "
                        f"idx={idx}/{len(prompts)} status=kept "
                        f"kept={len(answered)} failed={target_stats.failed_count} "
                        f"answer_chars={len(answered_item.target_answer)} "
                        f"elapsed={perf_counter() - target_started_at[completed_future]:.1f}s "
                        f"total={perf_counter() - start:.1f}s "
                        f"{_progress_fields(completed_units(), total_work_units, start)}"
                    )

    input_stats.kept_count = len(generated)
    target_stats.kept_count = len(answered)
    generated_items = [item for _idx, item in sorted(generated, key=lambda pair: pair[0])]
    answered_items = [item for _idx, item in sorted(answered, key=lambda pair: pair[0])]
    return generated_items, answered_items, input_stats, target_stats


def build_anchor_dataset_api(
    output_dir: str | Path,
    target_count: int,
    seed: int,
    knowledge: Ontology,
    language: Ontology,
    capability: Ontology,
    conversation: Ontology,
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
    input_generator_concurrency: int = 4,
    target_concurrency: int = 4,
    require_exact_count: bool = False,
    overwrite_output: bool = False,
    min_target_answer_chars: int = 8,
    max_target_answer_chars: int | None = None,
    sampling_strategy: SamplingStrategy = "balanced",
    ontology_embeddings: OntologyEmbeddings | None = None,
    ontology_sha256: str | None = None,
    embedding_model: str | None = None,
    embedding_distance: str | None = None,
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
    if input_generator_concurrency <= 0:
        raise ValueError("input_generator_concurrency must be positive")
    if target_concurrency <= 0:
        raise ValueError("target_concurrency must be positive")
    if min_target_answer_chars < 0:
        raise ValueError("min_target_answer_chars must be non-negative")
    if max_target_answer_chars is not None and max_target_answer_chars < min_target_answer_chars:
        raise ValueError(
            "max_target_answer_chars must be greater than or equal to min_target_answer_chars"
        )

    out = Path(output_dir)
    if out.exists() and any(out.iterdir()) and not overwrite_output:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {out}. "
            "Use a new output directory or pass --overwrite-output."
        )
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
            f"input_generator_concurrency={input_generator_concurrency} "
            f"target_concurrency={target_concurrency} "
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
            sampling_strategy=sampling_strategy,
            ontology_embeddings=ontology_embeddings,
            ontology_sha256=ontology_sha256,
            embedding_model=embedding_model,
            embedding_distance=embedding_distance,
        )
        prompts = generate_anchor_prompts(
            knowledge=knowledge,
            language=language,
            capability=capability,
            conversation=conversation,
            config=config,
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
        generated_inputs, answered, q_stats, t_stats = _run_streaming_batch(
            prompts=prompts,
            input_generator_config=input_generator_config,
            target_config=target_config,
            input_generator_max_tokens=input_generator_max_tokens,
            target_max_tokens=target_max_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            max_retries=max_retries,
            input_generator_concurrency=input_generator_concurrency,
            target_concurrency=target_concurrency,
            input_generation_chat_fn=input_generation_chat_fn or chat_completion,
            target_answer_chat_fn=target_answer_chat_fn or chat_completion,
            logger=logger,
            start=batch_start,
        )
        write_jsonl(generated_inputs, generated_inputs_path)
        write_jsonl(answered, answered_path)
        input_generation_stats_path.write_text(
            json.dumps(q_stats.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        target_answer_stats_path.write_text(
            json.dumps(t_stats.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if logger:
            logger(
                "stage=batch "
                f"batch={batch_idx + 1}/{batch_limit} status=streaming_done "
                f"input={q_stats.input_count} kept={q_stats.kept_count} "
                f"failed={q_stats.failed_count} api_failed={q_stats.api_failed_count} "
                f"parse_failed={q_stats.parse_failed_count} "
                f"target_input={t_stats.input_count} target_kept={t_stats.kept_count} "
                f"target_failed={t_stats.failed_count} "
                f"elapsed={perf_counter() - batch_start:.1f}s total={perf_counter() - run_start:.1f}s"
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
        for key, value in t_stats.to_dict().items():
            target_answer_stats_total[key] += value
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
            "input_generator_concurrency": input_generator_concurrency,
            "target_concurrency": target_concurrency,
            "min_target_answer_chars": min_target_answer_chars,
            "max_target_answer_chars": max_target_answer_chars,
            "sampling_strategy": sampling_strategy,
            "ontology_sha256": ontology_sha256,
            "embedding_model": embedding_model,
            "embedding_distance": embedding_distance,
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

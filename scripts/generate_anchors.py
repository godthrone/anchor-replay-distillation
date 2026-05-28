from __future__ import annotations

from datetime import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
ENV_EXAMPLE_PATH = PROJECT_ROOT / ".env-example"

REQUIRED_API_KEYS = (
    "ARD_INPUT_GENERATOR_API_BASE",
    "ARD_INPUT_GENERATOR_MODEL_NAME",
    "ARD_INPUT_GENERATOR_API_KEY",
    "ARD_TARGET_API_BASE",
    "ARD_TARGET_MODEL_NAME",
    "ARD_TARGET_API_KEY",
)
PLACEHOLDER_VALUES = {"", "...", "replace-me", "your-api-key", "sk-..."}


def log(message: str) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"ts={timestamp} {message}", flush=True)


def fail(message: str) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"ts={timestamp} level=error message={message}", file=sys.stderr)
    raise SystemExit(1)


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        fail(f"Missing {path.name}. Copy {ENV_EXAMPLE_PATH.name} to {path.name} and fill it in.")

    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip().lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            fail(f"Invalid .env line {line_no}: expected KEY=VALUE.")
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip('"').strip("'")
        if not key:
            fail(f"Invalid .env line {line_no}: empty key.")
        values[key] = value
    return values


def get_required(env_values: dict[str, str], key: str) -> str:
    value = env_values.get(key, os.environ.get(key, "")).strip()
    if value.lower() in PLACEHOLDER_VALUES:
        fail(f"Set {key} in .env before generating data.")
    return value


def get_value(env_values: dict[str, str], key: str, default: str) -> str:
    return env_values.get(key, os.environ.get(key, default)).strip() or default


def get_optional(env_values: dict[str, str], key: str) -> str | None:
    value = env_values.get(key, os.environ.get(key, "")).strip()
    return value or None


def get_int(env_values: dict[str, str], key: str, default: int, minimum: int = 1) -> int:
    raw = get_value(env_values, key, str(default))
    try:
        value = int(raw)
    except ValueError:
        fail(f"{key} must be an integer, got {raw!r}.")
    if value < minimum:
        fail(f"{key} must be >= {minimum}, got {value}.")
    return value


def get_float(env_values: dict[str, str], key: str, default: float) -> float:
    raw = get_value(env_values, key, str(default))
    try:
        return float(raw)
    except ValueError:
        fail(f"{key} must be a number, got {raw!r}.")


def get_bool(env_values: dict[str, str], key: str, default: bool = False) -> bool:
    raw = get_value(env_values, key, str(default)).lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    fail(f"{key} must be true or false, got {raw!r}.")


def default_output_dir(target_count: int, seed: int) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "outputs" / f"ard_anchor_dataset_{timestamp}_n{target_count}_seed{seed}"


def run(command: list[str], env: dict[str, str]) -> None:
    log("cmd=" + " ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def count_jsonl(path: Path) -> int:
    if not path.exists():
        fail(f"Expected output file was not created: {path}")
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def validate_outputs(
    output_dir: Path,
    target_count: int,
    exact_final_count_enabled: bool,
    api_keys: tuple[str, ...],
) -> None:
    anchor_bank_path = output_dir / "anchor_bank.jsonl"
    manifest_path = output_dir / "manifest.json"
    input_generation_stats_path = output_dir / "input_generation_stats.json"
    target_answer_stats_path = output_dir / "target_answer_stats.json"

    anchor_count = count_jsonl(anchor_bank_path)
    if exact_final_count_enabled and anchor_count != target_count:
        fail(f"Expected {target_count} anchors, found {anchor_count} in {anchor_bank_path}.")
    if anchor_count > target_count:
        fail(
            f"Expected at most {target_count} anchors, found {anchor_count} in {anchor_bank_path}."
        )

    if not manifest_path.exists():
        fail(f"Expected manifest file was not created: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generation_config = manifest.get("generation_config", {})
    if generation_config.get("target_count") != target_count:
        fail(f"Manifest does not confirm target_count={target_count}: {manifest_path}")
    if manifest.get("attempted_count") != target_count and not exact_final_count_enabled:
        fail(f"Manifest does not confirm attempted_count={target_count}: {manifest_path}")

    for stats_path in (input_generation_stats_path, target_answer_stats_path):
        if not stats_path.exists():
            fail(f"Expected stats file was not created: {stats_path}")

    for output_file in (
        anchor_bank_path,
        manifest_path,
        input_generation_stats_path,
        target_answer_stats_path,
    ):
        content = output_file.read_text(encoding="utf-8")
        for api_key in api_keys:
            if api_key and api_key in content:
                fail(f"API key leaked into output file: {output_file}")

    log(f"status=validated generated={anchor_count} output_dir={output_dir}")
    log(f"status=done anchor_bank={anchor_bank_path}")


def main() -> int:
    uv = shutil.which("uv")
    if uv is None:
        fail("uv is required. Install it from https://docs.astral.sh/uv/")

    env_values = parse_env_file(ENV_PATH)
    input_generator_api_base = get_required(env_values, "ARD_INPUT_GENERATOR_API_BASE")
    input_generator_model_name = get_required(env_values, "ARD_INPUT_GENERATOR_MODEL_NAME")
    input_generator_api_key = get_required(env_values, "ARD_INPUT_GENERATOR_API_KEY")
    target_api_base = get_required(env_values, "ARD_TARGET_API_BASE")
    target_model_name = get_required(env_values, "ARD_TARGET_MODEL_NAME")
    target_api_key = get_required(env_values, "ARD_TARGET_API_KEY")

    target_count = get_int(env_values, "ARD_TARGET_COUNT", 10)
    exact_final_count_batch_size = get_int(env_values, "ARD_EXACT_FINAL_COUNT_BATCH_SIZE", 10)
    exact_final_count_max_batches = get_int(env_values, "ARD_EXACT_FINAL_COUNT_MAX_BATCHES", 3)
    seed = get_int(env_values, "ARD_SEED", 42, minimum=0)
    temperature = get_float(env_values, "ARD_TEMPERATURE", 0.7)
    top_p = get_float(env_values, "ARD_TOP_P", 0.95)
    timeout = get_float(env_values, "ARD_TIMEOUT", 60)
    max_retries = get_int(env_values, "ARD_MAX_RETRIES", 2, minimum=0)
    input_generator_concurrency = get_int(env_values, "ARD_INPUT_GENERATOR_CONCURRENCY", 4)
    target_concurrency = get_int(env_values, "ARD_TARGET_CONCURRENCY", 4)
    exact_final_count_enabled = get_bool(env_values, "ARD_EXACT_FINAL_COUNT_ENABLED", False)

    configured_output_dir = get_optional(env_values, "ARD_OUTPUT_DIR")
    output_dir = (
        PROJECT_ROOT / configured_output_dir
        if configured_output_dir is not None
        else default_output_dir(target_count, seed)
    )
    overwrite_output = get_bool(env_values, "ARD_OVERWRITE_OUTPUT", False)

    process_env = os.environ.copy()
    process_env.update(env_values)
    process_env["ARD_INPUT_GENERATOR_API_BASE"] = input_generator_api_base
    process_env["ARD_INPUT_GENERATOR_MODEL_NAME"] = input_generator_model_name
    process_env["ARD_INPUT_GENERATOR_API_KEY"] = input_generator_api_key
    process_env["ARD_TARGET_API_BASE"] = target_api_base
    process_env["ARD_TARGET_MODEL_NAME"] = target_model_name
    process_env["ARD_TARGET_API_KEY"] = target_api_key

    run([uv, "sync", "--extra", "dev"], process_env)

    command = [
        uv,
        "run",
        "ard",
        "anchor-build-dataset",
        "--api-env-file",
        str(ENV_PATH),
        "--output-dir",
        str(output_dir),
        "--target-count",
        str(target_count),
        "--batch-size",
        str(exact_final_count_batch_size),
        "--max-batches",
        str(exact_final_count_max_batches),
        "--seed",
        str(seed),
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--timeout",
        str(timeout),
        "--max-retries",
        str(max_retries),
        "--input-generator-concurrency",
        str(input_generator_concurrency),
        "--target-concurrency",
        str(target_concurrency),
    ]

    optional_cli_args = {
        "ARD_ONTOLOGY_PATH": "--ontology",
        "ARD_LANGUAGES": "--languages",
        "ARD_TASK_TYPES": "--task-types",
        "ARD_SAMPLING_STRATEGY": "--sampling-strategy",
        "ARD_ONTOLOGY_EMBEDDINGS_PATH": "--ontology-embeddings",
        "ARD_INPUT_GENERATOR_MAX_TOKENS": "--input-generator-max-tokens",
        "ARD_TARGET_MAX_TOKENS": "--target-max-tokens",
        "ARD_MIN_TARGET_ANSWER_CHARS": "--min-target-answer-chars",
        "ARD_MAX_TARGET_ANSWER_CHARS": "--max-target-answer-chars",
    }
    for env_key, cli_arg in optional_cli_args.items():
        value = get_optional(env_values, env_key)
        if value is not None:
            command.extend([cli_arg, value])
    if exact_final_count_enabled:
        command.append("--require-exact-count")
    if overwrite_output:
        command.append("--overwrite-output")

    run(command, process_env)
    validate_outputs(
        output_dir,
        target_count,
        exact_final_count_enabled,
        (input_generator_api_key, target_api_key),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

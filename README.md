# ARD — Anchor Replay Distillation

> Multi-modal anchor data generation with log-prob export for OPD training.

**ARD** generates anchor datasets that prevent catastrophic forgetting during
continual fine-tuning. It samples diverse prompts from an ontology, generates
answers using a teacher model, and exports token-level log-probabilities
for On-Policy Distillation (OPD) training.

## Quick Start

```bash
# 1. Prepare config
cp configs/config.toml my_config.toml
# Edit my_config.toml: fill in API endpoint and model name
# Create config.override.toml with your API key:
cat > config.override.toml << 'EOF'
[input_generator]
api_key = "your-api-key"

[target]
api_key = "your-api-key"
EOF

# 2. Install
uv sync

# 3. Generate text anchors
uv run ard --config my_config.toml

# 4. Generate text + multimodal anchors
uv run ard --config my_config.toml --image-dir /path/to/images

# 5. Docker (alternative)
bash run.sh --config configs/my_config.toml --image-dir /data/images
```

## Configuration

All settings in a single TOML file. See `configs/config.toml` for the complete
template with all fields and defaults.

| Section | Purpose |
|---------|---------|
| `[input_generator]` | VLM/LLM that generates user questions |
| `[target]` | Teacher model that generates answers with log-probs |
| `[generation]` | Target count, seed, language and task type filters |
| `[ontology]` | Ontology path |
| `[output]` | Output directory settings |

Secrets (API keys) go in `config.override.toml` (gitignored).

## CLI

```
ard --config <path> [--image-dir <path>]
```

A single command handles everything:

- `--config` — Path to config.toml (required)
- `--image-dir` — Image directory for multimodal anchors (optional)

## Output

```
outputs/<dataset_name>/
├── anchor_bank.jsonl          # Unified anchor data (graspo-compatible)
├── images/                    # Multimodal images (if any)
└── manifest.json              # Summary statistics
```

### Data Format

All anchors are written to a single `anchor_bank.jsonl` file in a format
compatible with graspo:

```json
{
  "id": "anchor_a1b2c3d4",
  "source": "ard",
  "messages": [{"role": "user", "content": "Explain entropy..."}],
  "targets": [{
    "id": "primary",
    "output": {
      "content": "Entropy is a measure of disorder...",
      "logprobs": {
        "token_ids": [1, 2, 3],
        "log_probs": [-0.1, -0.2, -0.3]
      }
    }
  }],
  "anchor_meta": {"language": "English", "knowledge_domain": "science"},
  "teacher_id": "Qwen3.8-27B"
}
```

## Docker

```bash
# Build
bash docker/build.sh

# Run
bash run.sh --config configs/my_config.toml
```

## Development

```bash
uv sync --dev
uv run pytest tests/ -v
```

## License

MIT
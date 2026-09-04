# ARD Architecture

## Module Boundaries

```mermaid
graph TD
    subgraph "Entry Layer"
        CLI["CLI (cli.py)"]
        CFG["Config (config.py)"]
    end

    subgraph "Orchestration Layer"
        PL["Pipeline (pipeline.py)"]
    end

    subgraph "Domain Layer"
        TA["Text Anchor (text_anchor.py)"]
        IS["Image Store (image_store.py)"]
        BK["Bank (bank.py)"]
    end

    subgraph "Backends Layer"
        API["API Client (api_client.py)"]
    end

    subgraph "Core Layer"
        TY["Types (types.py)"]
        ON["Ontology (ontology.py)"]
        SM["Sampler (sampler.py)"]
        EM["Embeddings (embeddings.py)"]
        QT["Quota (quota.py)"]
    end

    CLI --> CFG
    CLI --> PL
    PL --> TA
    PL --> IS
    PL --> SM
    PL --> ON
    PL --> BK
    PL --> QT
    TA --> API
    TA --> BK
    SM --> ON
    SM --> QT
    SM --> EM
    IS --> EM
```

## Data Flow

```mermaid
flowchart LR
    subgraph "Input"
        TOML["config.toml"]
        ONT["anchor_ontology.json"]
        IMG["Image Directory"]
    end

    subgraph "ARD Pipeline"
        LOAD["1. Load Config + Ontology"]
        SMPL["2. Sample Combinations + Allocate Images"]
        GEN["3. Generate Anchors (Input Generator + Target Model)"]
        EXP["4. Export"]
    end

    subgraph "Output"
        AB["anchor_bank.jsonl"]
        IMGS["images/"]
        MF["manifest.json"]
    end

    TOML --> LOAD
    ONT --> LOAD
    LOAD --> SMPL
    SMPL --> GEN
    IMG --> GEN
    GEN --> EXP
    EXP --> AB
    EXP --> IMGS
    EXP --> MF
```

## Layer Responsibilities

| Layer | Directory | Responsibility | Dependencies |
|-------|-----------|---------------|-------------|
| **Core** | `src/ard/core/` | Pure computation: data types, ontology, sampling, quota, embeddings | `numpy` |
| **Backends** | `src/ard/backends/` | Infrastructure: API calls (urllib) to Input Generator and Target Model APIs | `core` |
| **Domain** | `src/ard/domain/` | Business logic: text anchor generation, image storage, bank I/O | `core`, `backends` |
| **Pipeline** | `src/ard/pipeline.py` | Orchestration: wires config → domain modules → output | `domain` |
| **Entry** | `src/ard/cli.py`, `config.py` | CLI parsing, config loading + validation | `pipeline` |

> **Note:** `multimodal_anchor.py` was removed in Phase 4. All anchors (text + multimodal) now flow through the unified `generate_text_anchors()` function in `text_anchor.py`. Image utilities (`scan_images`, `sample_images`, `copy_images_to_output`) were extracted into `image_store.py` and are called directly by the pipeline.

## Key Design Decisions

1. **Three-layer separation**: Core (pure computation) → Backends (I/O) → Domain (business logic). Core modules can be tested without network, GPU, or file system access.

2. **Single CLI command**: `ard --config <path> [--image-dir <path>]` handles everything. All anchors are generated through a unified pipeline; multimodal data is handled by attaching image paths to anchor specs before generation.

3. **Dual-model architecture**: Two separate API backends are used — the **Input Generator** (simulates user questions) and the **Target Model** (provides assistant answers). Each has its own endpoint, model name, and temperature configuration. Token-level log-probabilities are obtained from the Target Model API on the final turn of each conversation. See `api_client.py` for implementation details.

4. **Unified output format**: All anchors are written to a single `anchor_bank.jsonl` file in a format compatible with graspo. Each record includes `id`, `source`, `messages`, `targets` (with `logprobs`), `anchor_meta`, and `teacher_id` (historical field name retained for graspo compatibility — contains the target model name).

5. **TOML config with override**: Base config (`config.toml`) contains all fields with defaults. Override config (`config.override.toml`) contains only secrets. Both are deep-merged into a single `ARDConfig` instance at startup.

6. **User-provided resources**: Images are provided by the user (directory). All model inference goes through the user's API endpoints. ARD does not bundle models or images.

7. **Phase 3 unified flow**: Text and multimodal anchors share a single code path. Images are sampled and allocated to `AnchorSpec` objects via `allocate_images()`, then all specs (with or without images) are processed by `generate_text_anchors()`. The `multimodal_anchor.py` module is no longer invoked by the pipeline.

## Anchor Generation Flow

### Single-turn / Multi-turn Generation

```mermaid
sequenceDiagram
    actor User
    participant CLI as "CLI (cli.py)"
    participant Pipeline as "Pipeline (pipeline.py)"
    participant Domain as "Domain Layer (text_anchor.py)"
    participant InputGen as "Input Generator API"
    participant Target as "Target Model API"

    User->>CLI: ard --config config.toml [--image-dir /path]
    CLI->>CLI: Parse CLI args
    CLI->>Pipeline: load_config() → ARDConfig
    Pipeline->>Pipeline: Validate output directory
    Pipeline->>Pipeline: Load ontology
    Pipeline->>Pipeline: Sample AnchorSpec objects
    opt Image directory provided
        Pipeline->>Pipeline: Scan, sample, copy images
        Pipeline->>Pipeline: allocate_images() → attach images to specs
    end
    Pipeline->>Domain: generate_text_anchors(specs, input_client, target_client)
    loop For each AnchorSpec
        loop For each turn
            Domain->>InputGen: chat() — generate user message
            InputGen-->>Domain: User message
            alt Intermediate turn
                Domain->>Target: chat() — generate assistant reply
                Target-->>Domain: Assistant message
            else Final turn
                Domain->>Target: chat_with_logprobs() — generate answer + logprobs
                Target-->>Domain: Answer + token logprobs
            end
        end
        Domain->>Domain: Validate answer length
        Domain->>Domain: append_anchor() → anchor_bank.jsonl
    end
    Domain-->>Pipeline: List of generated anchors
    Pipeline->>Pipeline: build_manifest_from_records() → manifest.json
    Pipeline-->>CLI: Output directory path
    CLI-->>User: Done! Output: outputs/ard_dataset_YYYYMMDD_HHMMSS/
```

### Multi-turn Conversation Example (max_turns=3)

```mermaid
sequenceDiagram
    participant InputGen as "Input Generator API"
    participant Target as "Target Model API"
    participant Pipeline as "Pipeline"

    Note over Pipeline: AnchorSpec with 3 turns

    Pipeline->>InputGen: chat() — generate user_msg_1
    InputGen-->>Pipeline: "What is machine learning?"
    Pipeline->>Target: chat() — generate assistant_msg_1 (intermediate)
    Target-->>Pipeline: "Machine learning is..."

    Pipeline->>InputGen: chat() — generate user_msg_2 (follow-up)
    InputGen-->>Pipeline: "Can you give an example?"
    Pipeline->>Target: chat() — generate assistant_msg_2 (intermediate)
    Target-->>Pipeline: "Sure! Here's an example..."

    Pipeline->>InputGen: chat() — generate user_msg_3 (final turn)
    InputGen-->>Pipeline: "How does it compare to traditional programming?"
    Pipeline->>Target: chat_with_logprobs() — generate assistant_msg_3 (final, with logprobs)
    Target-->>Pipeline: "Traditional programming..." + logprobs

    Note over Pipeline: Anchor complete: 3 user messages + 3 assistant replies
```

### Multimodal Anchor Flow

```mermaid
sequenceDiagram
    participant Pipeline as "Pipeline (pipeline.py)"
    participant IS as "Image Store (image_store.py)"
    participant Domain as "Domain Layer (text_anchor.py)"
    participant InputGen as "Input Generator API"
    participant Target as "Target Model API"

    Pipeline->>IS: scan_images(image_dir)
    IS-->>Pipeline: List of image paths
    Pipeline->>IS: sample_images(images, count, seed)
    IS-->>Pipeline: Sampled image paths
    Pipeline->>IS: copy_images_to_output(sampled, output_dir)
    IS-->>Pipeline: Relative paths in output/images/
    Pipeline->>Pipeline: allocate_images() — attach image paths to AnchorSpec turns

    Note over Pipeline: All specs (with/without images) go through unified flow

    Pipeline->>Domain: generate_text_anchors(specs, input_client, target_client)
    loop For each AnchorSpec with image
        Domain->>Domain: encode_image_to_base64(image_path)
        Domain->>InputGen: chat() — generate user message about image
        InputGen-->>Domain: User message
        Domain->>Target: chat_with_logprobs() — answer + logprobs (final turn)
        Target-->>Domain: Answer + logprobs
    end
    Domain-->>Pipeline: Generated anchors
```

## Output Format

Each line in `anchor_bank.jsonl` follows the graspo-compatible format:

```json
{
  "id": "anchor_<sha256_hex16>",
  "source": "ard",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "targets": [{
    "id": "primary",
    "output": {
      "content": "<target_answer>",
      "logprobs": {"token_ids": [...], "log_probs": [...]}
    }
  }],
  "anchor_meta": {
    "language": "English",
    "knowledge_domain": "computer_science",
    "capability": "qa",
    "conversation_type": "single_turn"
  },
  "teacher_id": "<target_model_name>"
}
```

> **Note on `teacher_id`:** The field name `teacher_id` is retained for backward compatibility with graspo. In the current dual-model architecture, it contains the **target model** name (the model that generated the final answer with logprobs). The field name is historical and does not reflect the current architecture's distinction between Input Generator and Target Model.
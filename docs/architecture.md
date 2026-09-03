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
        MA["Multimodal Anchor (multimodal_anchor.py)"]
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
    end

    CLI --> CFG
    CLI --> PL
    PL --> TA
    PL --> MA
    TA --> API
    TA --> BK
    MA --> API
    MA --> IS
    MA --> BK
    PL --> BK
    TA --> SM
    MA --> SM
    SM --> ON
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
        SMPL["2. Sample Combinations"]
        GEN["3. Generate Anchors (with log-probs)"]
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
| **Core** | `src/ard/core/` | Pure computation: data types, ontology, sampling, embeddings | `numpy` |
| **Backends** | `src/ard/backends/` | Infrastructure: API calls (urllib) with log-prob support | `core` |
| **Domain** | `src/ard/domain/` | Business logic: assembles core + backends for each feature | `core`, `backends` |
| **Pipeline** | `src/ard/pipeline.py` | Orchestration: wires config → domain modules → output | `domain` |
| **Entry** | `src/ard/cli.py`, `config.py` | CLI parsing, config loading + validation | `pipeline` |

## Key Design Decisions

1. **Three-layer separation**: Core (pure computation) → Backends (I/O) → Domain (business logic). Core modules can be tested without network, GPU, or file system access.

2. **Single CLI command**: `ard --config <path> [--image-dir <path>]` handles everything. Text anchors are always generated; multimodal anchors are generated when `--image-dir` is provided.

3. **Log-probs via API**: Token-level log-probabilities are obtained directly from the vLLM API by adding `logprobs=True` to the target model call. No local HF model is needed.

4. **Unified output format**: All anchors are written to a single `anchor_bank.jsonl` file in a format compatible with graspo. Each record includes `id`, `source`, `messages`, `targets` (with `logprobs`), `anchor_meta`, and `teacher_id`.

5. **TOML config with override**: Base config (`config.toml`) contains all fields with defaults. Override config (`config.override.toml`) contains only secrets. Both are deep-merged into a single `ARDConfig` instance at startup.

6. **User-provided resources**: Images are provided by the user (directory). All model inference goes through the user's API endpoints. ARD does not bundle models or images.

## Anchor Generation Flow

```mermaid
sequenceDiagram
    actor User
    participant CLI as "CLI (cli.py)"
    participant Pipeline as "Pipeline (pipeline.py)"
    participant Domain as "Domain Layer"
    participant Backend as "API Client (api_client.py)"
    participant Teacher as "Teacher Model API"

    User->>CLI: ard --config config.toml [--image-dir /path]
    CLI->>CLI: Parse CLI args
    CLI->>Pipeline: load_config() → ARDConfig
    Pipeline->>Pipeline: Validate output directory
    Pipeline->>Pipeline: Load ontology
    Pipeline->>Domain: generate_text_anchors(ontology, config, clients)
    Domain->>Domain: Sample anchor combinations
    loop For each anchor
        Domain->>Backend: chat() — generate user question
        Backend->>Teacher: POST /chat/completions
        Teacher-->>Backend: User question
        Backend-->>Domain: User message
        Domain->>Backend: chat_with_logprobs() — generate answer
        Backend->>Teacher: POST /chat/completions (logprobs=True)
        Teacher-->>Backend: Answer + token log-probs
        Backend-->>Domain: Answer + logprobs
        Domain->>Domain: Validate answer length
    end
    Domain-->>Pipeline: List of text anchors
    opt Image directory provided
        Pipeline->>Domain: generate_multimodal_anchors(image_dir, ...)
        Domain->>Domain: Scan & sample images
        loop For each image
            Domain->>Backend: chat_batch() — VLM question
            Backend->>Teacher: POST /chat/completions (multimodal)
            Teacher-->>Backend: Question
            Domain->>Backend: chat_with_logprobs() — answer
            Backend->>Teacher: POST /chat/completions (logprobs=True)
            Teacher-->>Backend: Answer + logprobs
            Backend-->>Domain: Answer + logprobs
        end
        Domain-->>Pipeline: List of multimodal anchors
    end
    Pipeline->>Pipeline: write_anchor_bank() → anchor_bank.jsonl
    Pipeline->>Pipeline: build_manifest() → manifest.json
    Pipeline-->>CLI: Output directory path
    CLI-->>User: Done! Output: outputs/ard_dataset_YYYYMMDD_HHMMSS/
```
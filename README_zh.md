# Anchor Replay Distillation

[English](README.md) | 简体中文

Anchor Replay Distillation（ARD）用于生成研究中性的、可直接用于监督微调
（SFT）的聊天数据。你配置两个 OpenAI-compatible 模型：
`ARD_INPUT_GENERATOR_*` 根据采样到的 anchor spec 写出真实用户侧请求，
`ARD_TARGET_*` 以目标模型身份回答这些请求，产出 SFT target。

本仓库专注数据生成，不包含模型训练或微调运行器。ARD 的目标是覆盖和保留
模型的一般能力，而不是在训练框架内注入服务级合规、拒绝或价值对齐策略。

## 快速开始

1. 安装 uv，并确认 Python 3.11+ 可用：

```bash
uv --version
uv run --no-project --python 3.11 python --version
```

Windows PowerShell 如果阻止安装脚本，可先运行：

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

2. 配置 API：

```bash
cp .env-example .env
```

填写必需值：

```env
ARD_INPUT_GENERATOR_API_BASE=https://example.com/v1
ARD_INPUT_GENERATOR_MODEL_NAME=deepseek-v4-flash
ARD_INPUT_GENERATOR_API_KEY=replace-me

ARD_TARGET_API_BASE=https://example.com/v1
ARD_TARGET_MODEL_NAME=deepseek-v4-flash
ARD_TARGET_API_KEY=replace-me
```

`.env` 包含密钥，不要提交；`.env-example` 是提交用模板。

3. 生成默认数据集：

```bash
uv run python scripts/generate_anchors.py
```

默认会在 `outputs/ard_anchor_dataset_*` 下尝试生成 10 个候选 anchor。

主要输出：

- `anchor_bank.jsonl`：最终 SFT-ready 数据
- `manifest.json`：分布、seed、模型和采样元数据
- `input_generation_stats.json` / `target_answer_stats.json`：生成统计
- `batch_*/`：中间 prompts、generated inputs、target answers

示例格式见 [examples/anchor_bank.sample.jsonl](examples/anchor_bank.sample.jsonl)。

仓库文本产物和生成的 JSON/JSONL 输出均使用 UTF-8。非 ASCII 语言内容直接保留为
可读文本，不写成 `\uXXXX`，但 JSON 必需的换行和反斜杠转义仍会保留。

## 输出格式

`anchor_bank.jsonl` 是 JSONL 文件，每行是一条 SFT 样本。`messages` 表示单轮或
多轮聊天上下文，`anchor_meta` 记录研究中性的采样维度。

| 字段 | 含义 |
| --- | --- |
| `id` | 稳定样本 ID。 |
| `messages` | 聊天输入上下文。 |
| `target_answer` | 目标模型回答，也就是 SFT 监督文本。 |
| `target_model` | 生成 `target_answer` 的模型。 |
| `input_generator_model` | 生成真实用户侧输入的模型。 |
| `anchor_meta` | 采样元数据和能力标签。 |

关键 `anchor_meta` 字段：

| 字段 | 含义 |
| --- | --- |
| `language` | 语言桶，例如 `English`、`简体中文`、`Español`、`日本語`。 |
| `knowledge_domain` | 采样到的知识或研究领域。 |
| `capability` | 能力标签，例如解释、比较、推理、工具选择或写作。 |
| `task_type` | 用于过滤或均衡的任务标签。 |
| `conversation_type` | 对话形态，例如 `single_turn` 或 `constraint_update_4_turn`。 |
| `is_multi_turn` | `messages` 是否包含多轮对话。 |
| `sampling_strategy` | `farthest`、`balanced` 或 `random`。 |
| `ontology_sha256` | 使用 embedding sidecar 时校验 ontology 的哈希。 |
| `seed` | 可复现采样 seed。 |

## 修改默认参数

运行参数放在 `.env` 和 CLI 参数里。ontology 文件是
`configs/anchor_ontology.json`，默认预计算 embedding sidecar 是
`configs/anchor_ontology_embeddings.json`。

```env
ARD_TARGET_COUNT=10
ARD_EXACT_FINAL_COUNT_ENABLED=false
ARD_EXACT_FINAL_COUNT_BATCH_SIZE=10
ARD_EXACT_FINAL_COUNT_MAX_BATCHES=3
ARD_SEED=42
ARD_OUTPUT_DIR=
ARD_OVERWRITE_OUTPUT=false
ARD_ONTOLOGY_PATH=configs/anchor_ontology.json
ARD_ONTOLOGY_EMBEDDINGS_PATH=configs/anchor_ontology_embeddings.json
ARD_SAMPLING_STRATEGY=farthest
ARD_LANGUAGES=
ARD_TASK_TYPES=
ARD_TEMPERATURE=0.7
ARD_TOP_P=0.95
ARD_TIMEOUT=60
ARD_MAX_RETRIES=2
ARD_INPUT_GENERATOR_CONCURRENCY=100
ARD_TARGET_CONCURRENCY=100
ARD_MIN_TARGET_ANSWER_CHARS=8
ARD_MAX_TARGET_ANSWER_CHARS=
ARD_INPUT_GENERATOR_MAX_TOKENS=
ARD_TARGET_MAX_TOKENS=
```

`ARD_LANGUAGES` 和 `ARD_TASK_TYPES` 是逗号分隔过滤器，留空使用 ontology 默认值：

```env
ARD_LANGUAGES=English,简体中文,Español,日本語
ARD_TASK_TYPES=qa,explanation,reasoning,coding,debugging
```

严格 API 要求：默认不要设置 `ARD_INPUT_GENERATOR_MAX_TOKENS` 或
`ARD_TARGET_MAX_TOKENS`。留空时 ARD 不会向大模型 API 发送 `max_tokens`。

## 采样 Ontology

ARD 从 `configs/anchor_ontology.json` 采样。dict 表示路径，list 中的字符串表示
叶子节点，每个叶子节点都是一个可采样选项。

顶层 section：

| Section | 含义 |
| --- | --- |
| `languages` | 输出语言桶。 |
| `knowledge_domains` | 广义研究和工作领域，包括科学探索、艺术、哲学、宗教/民俗、玄学作为文化现象、社会事件、历史、文化、人类奇闻、未来推演、技术工作、财经、法律、医学、写作和工具使用。 |
| `capabilities` | 能力/任务标签。 |
| `conversation_types` | 单轮和多轮对话形态。 |
| `language_features` | 风格、格式、难度、上下文长度、噪声和回答预期。 |

默认 `farthest` 策略读取 `configs/anchor_ontology_embeddings.json`，并用其中的
`ontology_sha256` 校验 `configs/anchor_ontology.json`。校验通过后，它会在
`knowledge_domains` 上用 cosine 最远点采样，再均衡分配语言、能力、对话形态和语言特征。
修改 ontology 节点后，需要运行 `ard ontology-embed` 重新生成 sidecar；在新 sidecar
生成前，可以先用 `--sampling-strategy balanced`。

为自定义 ontology 重新生成 sidecar：

```bash
uv run --extra embed ard ontology-embed \
  --ontology configs/anchor_ontology.json \
  --output configs/anchor_ontology_embeddings.json \
  --backend local \
  --model Qwen/Qwen3-Embedding-0.6B
```

也可以使用 OpenAI-compatible embedding API：

```bash
uv run ard ontology-embed \
  --ontology configs/custom_ontology.json \
  --output configs/custom_ontology_embeddings.json \
  --backend api \
  --api-env-file .env
```

当前提交的 sidecar 已使用 `Qwen/Qwen3-Embedding-0.6B` 生成，包含 1024 维真实
embedding。普通用户无需下载 embedding 模型即可运行；修改 ontology 节点后，建议用
Qwen 或 API backend 重新生成。

## 验证快照

当前提交的 ontology 和 Qwen sidecar 已用默认 `farthest` 策略跑通过一次端到端验证：
`ARD_TARGET_COUNT=1000`，attempted `1000`，input kept `1000`，最终 target-answer
样本 `986`，其中 target API timeout/failure `14`。完整 `outputs/` 目录保持 ignored，
不会提交。

## 开发检查

```bash
uv run ruff format .
uv run ruff check --fix .
uv run mypy
uv run pytest -q
```

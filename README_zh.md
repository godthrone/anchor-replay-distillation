# Anchor Replay Distillation

[English](README.md) | 简体中文

Anchor Replay Distillation (ARD) 用来自动生成可直接用于监督微调（SFT）的聊天数据。你只需要配置两个 OpenAI-compatible 模型：`ARD_INPUT_GENERATOR_*` 负责把采样到的 anchor spec 写成真实用户请求，`ARD_TARGET_*` 负责以你要训练、评估或保留能力的目标模型身份回答。最终输出是 SFT-ready 的 `anchor_bank.jsonl`，可以混入你的微调数据，降低模型在微调后遗忘通用问答、推理、翻译、代码和安全边界等能力的风险。

默认流程尽量简单：一条命令创建 `.venv`、安装依赖、调用 API，并写出生成数据。

## 快速开始

1. 安装 uv 并验证工具链。

本项目需要 Python 3.11 或更新版本，但使用 uv standalone installer 时，不需要先安装 Python 或 `pipx`。uv 可以自动安装和管理本项目使用的 Python。

Windows PowerShell:

如果 PowerShell 提示 execution policy 阻止安装脚本，先允许当前用户运行已签名脚本。这不需要管理员权限：

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

然后安装 uv：

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

如果你只想在当前终端会话临时放行，也可以在安装前使用 `Set-ExecutionPolicy Bypass -Scope Process -Force`。

macOS/Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

如果安装后找不到 `uv`，关闭并重新打开终端，然后验证：

```bash
uv --version
uv run --no-project --python 3.11 python --version
```

这个 Python 检查应输出 Python 3.11 或更新版本，并且不会安装本项目依赖。如果本机没有可用 Python，uv 通常会自动下载。你也可以显式安装项目 Python：

```bash
uv python install 3.11
```

如果你已经有 Python 和 `pipx`，也可以使用 `pipx install uv`。更多安装方式见 uv 官方文档：https://docs.astral.sh/uv/

2. 配置 API：

```bash
cp .env-example .env
```

打开 `.env`，填入：

```env
ARD_INPUT_GENERATOR_API_BASE=https://example.com/v1
ARD_INPUT_GENERATOR_MODEL_NAME=deepseek-v4-flash
ARD_INPUT_GENERATOR_API_KEY=replace-me

ARD_TARGET_API_BASE=https://example.com/v1
ARD_TARGET_MODEL_NAME=deepseek-v4-flash
ARD_TARGET_API_KEY=replace-me
```

`ARD_INPUT_GENERATOR_*` 是更强的输入生成模型，用来把 anchor spec 写成真实用户消息。`ARD_TARGET_*` 是你要训练、评估或保持能力的目标模型，它的回答会成为 SFT 监督目标。

`.env` 包含密钥，不要提交；`.env-example` 是可提交的配置模板。

所有面向用户的运行参数都在 `.env` 里。默认 seed 是 `ARD_SEED=42`；如果你想得到另一组可复现的样本分布，就修改它。

3. 一键生成默认数据：

```bash
uv run python scripts/generate_anchors.py
```

脚本会自动运行 `uv sync --extra dev`，然后默认尝试生成 10 个候选 anchors 到带时间戳的运行目录：

```text
outputs/ard_anchor_dataset_YYYYMMDD_HHMMSS_n10_seed42/
```

主要输出：

- `anchor_bank.jsonl`：最终 SFT-ready 数据
- `manifest.json`：数据分布、seed、模型元数据
- `input_generation_stats.json` / `target_answer_stats.json`：生成统计
- `batch_*/`：中间 prompts/generated-inputs/target-answers，便于排查

如果只想看 JSONL 格式，不想使用真实 API 输出，可以参考 [examples/anchor_bank.sample.jsonl](examples/anchor_bank.sample.jsonl)。

生成过程中，日志会包含时间戳、进度百分比、吞吐量和预计剩余时间（ETA）。input generation 和 target answer 会流水线并发执行，因此两类日志会交错出现。

默认是“尝试数量语义”：`ARD_TARGET_COUNT=10` 表示尝试 10 个候选，过滤后 `anchor_bank.jsonl` 可能少于 10 条。过滤只筛明显坏样本；如果你必须让最终保留行数达到 `ARD_TARGET_COUNT`，把 `ARD_EXACT_FINAL_COUNT_ENABLED=true`。

## 输出格式

`anchor_bank.jsonl` 是 JSONL 文件，每一行是一条可用于 SFT 的聊天样本。看 `messages` 可以判断单轮/多轮；看 `anchor_meta` 可以判断语言、能力、任务类型、安全边界和采样分布。

| 字段 | 含义 |
| --- | --- |
| `id` | 稳定的 anchor 样本 ID。 |
| `messages` | 聊天输入上下文。单轮样本通常只有一条 user 消息；多轮样本包含历史 user/assistant 轮次和最后的 user 请求。 |
| `target_answer` | 目标模型生成的回答，也就是 SFT 监督文本。 |
| `target_model` | 生成 `target_answer` 的目标模型。 |
| `input_generator_model` | 用来生成真实感用户输入的强模型。 |
| `anchor_meta` | 采样元数据和能力标签。 |

关键 `anchor_meta` 字段：

| 字段 | 含义 |
| --- | --- |
| `language` | 语言桶，例如 `English`、`简体中文`、`bilingual_zh_en`。 |
| `knowledge_domain` | 采样到的知识或工作领域。 |
| `capability` | 要保留的能力，例如解释、比较、推理、工具选择、不确定性处理。 |
| `task_type` | 用于过滤或分布平衡的任务标签。 |
| `conversation_type` | 对话形态，例如 `single_turn`、`troubleshooting_3_turn`、`constraint_update_4_turn`。 |
| `is_multi_turn` | `messages` 是否包含多轮对话。 |
| `safety_boundary` | 期望的安全/权威边界行为，例如标准回答、澄清、拒绝或安全替代方案。 |
| `seed` | 用于可复现采样的 seed。 |

## 修改默认参数

运行参数以 `.env` 为主要配置入口。`.env-example` 列出了所有支持的 `ARD_*` 环境变量，包括默认 seed `ARD_SEED=42`。生成问题时所有可采样范围都在一个 JSON 文件里：`data/anchor_seed/anchor_ontology.json`。如果你希望 ARD 生成自己的领域、语言、能力、对话形态、安全边界或语言风格，复制并编辑这个文件，再把 `ARD_ONTOLOGY_PATH` 指向你的副本即可。

常用参数：

```env
ARD_TARGET_COUNT=10
ARD_EXACT_FINAL_COUNT_ENABLED=false
ARD_EXACT_FINAL_COUNT_BATCH_SIZE=10
ARD_EXACT_FINAL_COUNT_MAX_BATCHES=3
ARD_SEED=42
ARD_OUTPUT_DIR=
ARD_OVERWRITE_OUTPUT=false
ARD_CONFIG_PATH=configs/anchor_generation.yaml
ARD_ONTOLOGY_PATH=data/anchor_seed/anchor_ontology.json
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

`ARD_INPUT_GENERATOR_CONCURRENCY` 和 `ARD_TARGET_CONCURRENCY` 分别控制两个 API 阶段的并发度。默认值假设服务商/模型档位允许较高并发；如果遇到限流或超时，请调低。端到端构建器采用流水线执行：一条 input 生成完成后，可以立刻开始 target answer，不需要等全部 input 都生成完。

`ARD_OUTPUT_DIR` 留空可以避免覆盖上一次运行结果。如果你设置固定的 `ARD_OUTPUT_DIR`，ARD 默认会拒绝写入非空目录；只有设置 `ARD_OVERWRITE_OUTPUT=true` 时才会覆盖。

`ARD_LANGUAGES` 和 `ARD_TASK_TYPES` 是本次运行过滤器。留空时使用 ontology 的全部默认值；如果只想缩小本次运行范围，可以这样写：

```env
ARD_LANGUAGES=English,简体中文,bilingual_zh_en
ARD_TASK_TYPES=qa,explanation,reasoning,coding,debugging
```

`ARD_MAX_TARGET_ANSWER_CHARS` 留空表示不过滤长答案。严格要求：默认不要设置 `ARD_INPUT_GENERATOR_MAX_TOKENS` 或 `ARD_TARGET_MAX_TOKENS`。留空时，ARD 不会向大模型 API 发送 `max_tokens` 或等价 token cap，让模型/API 使用默认输出预算。只有你明确要覆盖默认预算时才填写。

## 采样 Ontology

ARD 从单一 JSON 文件采样：`data/anchor_seed/anchor_ontology.json`。文件格式是普通嵌套 JSON：dict 表示路径，list 里的字符串表示叶子节点，每个叶子节点都是一个可采样选项。

顶层 section：

| Section | 含义 | 默认顶层值 |
| --- | --- | --- |
| `languages` | 输出语言桶。 | `English`, `简体中文`, `bilingual_zh_en` |
| `knowledge_domains` | 知识/工作领域。 | `software_engineering`, `systems_devops`, `data_ai_ml`, `math_logic`, `science_engineering`, `business_operations`, `finance_economics`, `law_policy_safety`, `medicine_health_safety`, `humanities_world_knowledge`, `language_writing_translation`, `agent_tool_use` |
| `capabilities` | 能力/任务标签。 | `knowledge_response`, `reasoning`, `coding_and_data`, `language_work`, `agentic_behavior` |
| `conversation_types` | 单轮和多轮对话形态。 | `single_turn`, `clarification`, `troubleshooting`, `revision`, `tool_and_safety` |
| `safety_boundaries` | 安全和权威边界行为。 | `normal`, `regulated_domain`, `boundary` |
| `language_features` | 风格、格式、难度、上下文长度、噪声、回答预期。 | `style`, `format`, `difficulty`, `context_length`, `noise`, `answer_expectation` |

如果要定制采样空间，复制 `data/anchor_seed/anchor_ontology.json`，编辑副本，然后把 `ARD_ONTOLOGY_PATH` 指向该文件。如果只想缩小某次运行范围，用 `ARD_LANGUAGES` 或 `ARD_TASK_TYPES` 即可，不需要编辑 ontology。

## 开发检查

```bash
uv run ruff format .
uv run ruff check --fix .
uv run mypy
uv run pytest -q
```

## 环境变量附录

所有运行配置都放在 `.env` 中。`.env-example` 是简短模板；这里是完整说明。

| 变量 | 默认值 | 含义 / 什么时候修改 |
| --- | --- | --- |
| `ARD_INPUT_GENERATOR_API_BASE` | 必填 | 强输入生成模型的 OpenAI-compatible base URL。 |
| `ARD_INPUT_GENERATOR_MODEL_NAME` | 必填 | 用来生成真实用户请求的强模型。 |
| `ARD_INPUT_GENERATOR_API_KEY` | 必填 | 输入生成模型 API key。不要提交 `.env`。 |
| `ARD_TARGET_API_BASE` | 必填 | 目标模型的 OpenAI-compatible base URL。 |
| `ARD_TARGET_MODEL_NAME` | 必填 | 你要训练、评估或保持能力的目标模型；它的回答会成为 SFT target。 |
| `ARD_TARGET_API_KEY` | 必填 | 目标模型 API key。不要提交 `.env`。 |
| `ARD_TARGET_COUNT` | `10` | 尝试生成的候选 input 数量。不是最终行数保证，因为过滤可能丢弃坏样本。 |
| `ARD_EXACT_FINAL_COUNT_ENABLED` | `false` | 只有你必须让最终保留行数达到 `ARD_TARGET_COUNT` 时才设为 `true`。默认 `false` 可以让 API 调用量更可预期。 |
| `ARD_EXACT_FINAL_COUNT_BATCH_SIZE` | `10` | 仅 exact-final-count 模式启用时使用的额外候选 batch 大小。 |
| `ARD_EXACT_FINAL_COUNT_MAX_BATCHES` | `3` | 仅 exact-final-count 模式启用时最多尝试多少批。 |
| `ARD_SEED` | `42` | 采样 seed。修改它可以得到另一组可复现的样本分布。 |
| `ARD_OUTPUT_DIR` | 留空 | 留空时自动在 `outputs/` 下创建时间戳目录。需要固定输出位置时再填写。 |
| `ARD_OVERWRITE_OUTPUT` | `false` | 只有明确要写入非空输出目录时才设为 `true`。 |
| `ARD_CONFIG_PATH` | `configs/anchor_generation.yaml` | 基础生成配置。大多数用户保持默认即可。 |
| `ARD_ONTOLOGY_PATH` | `data/anchor_seed/anchor_ontology.json` | 单一 ontology JSON，包含语言、领域、能力、对话形态、安全边界和语言特征。 |
| `ARD_LANGUAGES` | 留空 | 可选语言过滤，逗号分隔，例如 `English,简体中文`。留空使用 ontology 默认值。 |
| `ARD_TASK_TYPES` | 留空 | 可选任务类型过滤，逗号分隔，例如 `qa,explanation,reasoning`。留空使用 ontology 默认值。 |
| `ARD_TEMPERATURE` | `0.7` | 两个模型调用使用的采样 temperature。 |
| `ARD_TOP_P` | `0.95` | 两个模型调用使用的 top-p。 |
| `ARD_TIMEOUT` | `60` | 单次请求超时时间，单位秒。如果服务商长回答较慢，可以调大。 |
| `ARD_MAX_RETRIES` | `2` | API 调用失败后的重试次数。 |
| `ARD_INPUT_GENERATOR_CONCURRENCY` | `100` | input generation 阶段并发数；遇到服务商限流或超时时调低。 |
| `ARD_TARGET_CONCURRENCY` | `100` | target answer 阶段并发数；遇到服务商限流或超时时调低。 |
| `ARD_MIN_TARGET_ANSWER_CHARS` | `8` | 低于该字符数的回答会被过滤。 |
| `ARD_MAX_TARGET_ANSWER_CHARS` | 留空 | 可选最长回答过滤。留空表示保留长答案。 |
| `ARD_INPUT_GENERATOR_MAX_TOKENS` | 留空 | 可选 input generation token cap。留空时 ARD 不发送 `max_tokens`。 |
| `ARD_TARGET_MAX_TOKENS` | 留空 | 可选 target answer token cap。留空时 ARD 不发送 `max_tokens`。 |

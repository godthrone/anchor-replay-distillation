# ARD — Anchor Replay Distillation

[English](README.md)

> 锚点数据生成框架，专为大模型持续微调中的灾难性遗忘问题设计。
> 本体驱动采样、教师模型蒸馏导出 token 级 log-probabilities、
> 原生兼容 Graspo/OPD 训练流程。

**ARD** 生成高质量锚点数据集，防止大模型持续微调过程中的灾难性遗忘。
它从结构化知识本体中采样多样化提示词，调用教师模型生成答案并导出
完整 token 级 log-probabilities，输出格式直接对接 On-Policy Distillation
(OPD) 和 Graspo 强化学习训练流程。

**核心能力：**
- 🧠 **本体驱动采样** — 基于知识领域、语言、能力、任务类型四维本体，
  结合 embedding 最远点采样，确保锚点覆盖全面且分布均匀。
- 🎯 **教师模型蒸馏** — 强模型生成答案并导出完整 token 级 logprobs，
  为 OPD 训练提供高质量 replay 信号。
- 🖼️ **多模态支持** — 文本锚点和图文锚点统一流程、统一输出，一条命令
  覆盖纯文本和多模态场景。
- 📦 **Graspo 兼容** — 输出格式与 Graspo anchor bank 无缝对接，生成即训练。
- ⚡ **一条命令** — 单 CLI、TOML 配置驱动、机密分离、Docker 一键启动。

## 快速开始

### 环境要求

- Docker

### 1. 构建 Docker 镜像

```bash
git clone https://github.com/your-org/anchor-replay-distillation.git
cd anchor-replay-distillation
bash docker/build.sh
```

### 2. 创建覆写配置

```bash
mkdir -p .local
cp configs/config.override.sample.toml .local/config.override.toml
# 编辑 .local/config.override.toml: 填入 api_base, model_name, api_key
```

编辑 `.local/config.override.toml`，填入你的 API 凭证：

```toml
[input_generator]
api_base = "https://your-api.example.com/v1"
model_name = "your-model-name"
api_key = "your-api-key"

[target_model]
api_base = "https://your-api.example.com/v1"
model_name = "your-model-name"
api_key = "your-api-key"
```

### 3. 生成锚点

```bash
# 纯文本锚点
bash run.sh --config configs/config.toml --override .local/config.override.toml

# 多模态锚点（使用样例图片）
bash run.sh --config configs/config.toml --override .local/config.override.toml \
    --image-dir examples/images
```

### 4. 查看输出

```bash
ls outputs/
```

找到 `ard_dataset_*` 目录（如 `ard_dataset_20240101_120000`）。
参考 `examples/anchor_bank.sample.jsonl` 了解输出格式。

## 配置

ARD 采用**分层 TOML 配置**模型。有两个配置文件：

| 文件 | 用途 | 是否 git 跟踪？ |
|------|------|----------------|
| `configs/config.toml` | 基础配置 — 含全部字段和默认值 | ✅ 是 |
| `.local/config.override.toml` | 覆写配置 — 部署环境机密信息 | ❌ 否（`.gitignore` 排除） |

**`configs/config.toml`** 是配置结构的唯一权威来源。它定义了所有字段及其
合理的默认值。非机密字段（如 `temperature`、`max_tokens`、`seed`）开箱即用。
机密字段（`api_base`、`model_name`、`api_key`）留空，由覆写文件填充。

**`.local/config.override.toml`** 只包含你需要覆写的字段——通常是
`[input_generator]` 和 `[target_model]` 的 `api_base`、`model_name`、
`api_key`。它存放在 `.local/` 目录中，不会被提交到 git。不能添加基础配置
中不存在的字段。

启动时，覆写文件深度合并到基础配置中。合并后的结果是一份唯一的配置 dict，
程序中不存在两个配置源。

| Section | 用途 |
|---------|------|
| `[input_generator]` | 生成用户提问的 VLM/LLM |
| `[target_model]` | 生成答案（含 log-prob）的教师模型 |
| `[generation]` | 目标数量、随机种子、并发数、最大轮数、系统角色、语言和任务类型过滤 |
| `[ontology]` | 本体论路径 |
| `[output]` | 输出目录设置 |

## CLI

```
ard --config <路径> [--override <路径>] [--image-dir <路径>] [--max-turns <n>]
```

单一命令完成所有操作：

- `--config` — 基础配置 TOML 文件路径（必填）
- `--override` — 覆写配置 TOML 文件路径（可选；未提供时自动检测 `--config` 同目录下的 `config.override.toml`）
- `--image-dir` — 多模态锚点的图片目录（可选）
- `--max-turns` — 最大对话轮数，覆盖配置中的值（可选；1 = 单轮, 2-10 = 多轮）

## 输出

```
outputs/<dataset_name>/
├── anchor_bank.jsonl          # 统一锚点数据（graspo 兼容）
├── images/                    # 多模态图片（如有）
└── manifest.json              # 统计摘要
```

### 数据格式

所有锚点写入单个 `anchor_bank.jsonl` 文件，格式与 graspo 兼容：

```json
{
  "id": "anchor_a1b2c3d4",
  "source": "ard",
  "messages": [
    {"role": "user", "content": "解释熵的概念..."},
    {"role": "assistant", "content": "熵是衡量系统无序程度的物理量..."},
    {"role": "user", "content": "能举个例子吗？"},
    {"role": "assistant", "content": "当然！冰融化成水..."}
  ],
  "targets": [{
    "id": "primary",
    "output": {
      "content": "当然！冰融化成水...",
      "logprobs": {
        "token_ids": [1, 2, 3],
        "log_probs": [-0.1, -0.2, -0.3]
      }
    }
  }],
  "anchor_meta": {"language": "简体中文", "knowledge_domain": "science"},
  "teacher_id": "Qwen3.8-27B"
}
```

## 示例

`examples/` 目录包含样例输入和输出，帮助你无需运行即可了解项目：

```
examples/
├── images/                    # 多模态模式样例图片
│   ├── sample_01.jpg
│   └── ...
└── anchor_bank.sample.jsonl   # 样例输出（3 条锚点）
```

你可以在 GitHub 上直接浏览 `examples/` 查看输入输出格式。

## Docker

Docker 镜像通过 `docker/build.sh` 构建，使用当前 git 版本号作为标签。
镜像采用两层构建（依赖层 + 源码层），配合 BuildKit 缓存挂载实现快速重建。

### 自定义镜像标签

```bash
IMAGE_NAME=ard:latest bash docker/build.sh
ARD_IMAGE=ard:latest bash run.sh --config configs/config.toml --override .local/config.override.toml
```

### 无 git 环境

```bash
VERSION=1.0.0 bash docker/build.sh
```

## 开发

```bash
# 安装依赖
uv sync --extra dev

# 运行测试
uv run pytest tests/ -v

# 类型检查
uv run mypy src/ard/

# 代码检查
uv run ruff check src/ tests/
```

## 常见问题

### 如何配置 API key？

从样例模板创建 `.local/config.override.toml` 并填入 API key：

```bash
mkdir -p .local
cp configs/config.override.sample.toml .local/config.override.toml
```

然后编辑文件：

```toml
[input_generator]
api_base = "https://your-api.example.com/v1"
model_name = "your-model-name"
api_key = "sk-..."

[target_model]
api_base = "https://your-api.example.com/v1"
model_name = "your-model-name"
api_key = "sk-..."
```

启动时覆写文件会与 `configs/config.toml` 深度合并。只有机密信息放这里，
其他配置仍在 `configs/config.toml` 中。

### 如何添加自定义本体论？

将你的本体论 JSON 文件放入 `data/` 目录（或任意路径），
然后在 `configs/config.toml` 中设置 `path`：

```toml
[ontology]
path = "data/my_ontology.json"
```

本体论必须遵循预期的 schema，包含 `knowledge_domains`、`capabilities`
和 `languages` 字段。

### 输出格式是什么？

所有锚点写入单个 `anchor_bank.jsonl` 文件，JSONL 格式（每行一个 JSON 对象）。
每条记录包含 `id`、`source`、`messages`、`targets`（含 `content` 和 token 级
`logprobs`）、`anchor_meta` 和 `teacher_id`。格式与 graspo 的锚点库格式兼容。
完整结构见[数据格式](#数据格式)章节。

### 如何生成多模态锚点？

使用 `--image-dir` 参数指定图片目录：

```bash
bash run.sh --config configs/config.toml --override .local/config.override.toml \
    --image-dir examples/images
```

流水线会从目录中采样图片，生成基于 VLM 的提问，
并在文本锚点之外产出多模态锚点。

### 文本锚点和多模态锚点有什么区别？

文本锚点是从本体论生成的对话（单轮或多轮），不含图片。多模态锚点在对话消息中包含图片，
需要提供 `--image-dir` 参数才会生成。两种类型共享相同的输出格式，
写入同一个 `anchor_bank.jsonl` 文件。对话轮数由配置中的 `max_turns`（或 CLI 的 `--max_turns`）控制。

### 如何断点续传？

ARD 自动从上次已完成的锚点恢复。只需重新运行相同的命令——流水线会检测
`anchor_bank.jsonl` 中已有的锚点，只生成剩余数量以达到 `target_count`。

## 许可证

MIT
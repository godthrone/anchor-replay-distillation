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
  结合 embedding 分层最远点采样，确保锚点覆盖全面且分布多样。
- 🎯 **教师模型蒸馏** — 强模型生成答案并导出完整 token 级 logprobs，
  为 OPD 训练提供高质量 replay 信号。
- 🖼️ **多模态支持** — 文本锚点和图文锚点统一流程、统一输出，一条命令
  覆盖纯文本和多模态场景。
- 📦 **Graspo 兼容** — 输出格式与 Graspo anchor bank 无缝对接，生成即训练。
- ⚡ **一条命令** — 单 CLI、TOML 配置驱动、机密分离、Docker 一键启动。

## 快速开始

### 环境要求

- Docker
- 嵌入数据：`ontology/anchor_ontology_embeddings.json` —— **已随仓库跟踪**，
  clone 后即存在（49 个预计算嵌入向量，1024 维，供分层 FPS 采样器使用）。
  它是只读的流水线输入，不是生成产物。
- **可选：** `rawpy`（已包含在 `pyproject.toml` 依赖中）用于 RAW 图片格式支持
  （CR2、NEF、ARW、DNG 等 19 种）。该库以 manylinux wheel 发布，自带
  `libraw.so`，无需安装系统包。如果不需要 RAW 格式，rawpy 的导入是懒加载的，
  不会被触发。

### 1. 构建 Docker 镜像

```bash
git clone https://github.com/godthrone/anchor-replay-distillation.git
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
| `[generation]` | 目标数量、随机种子、并发数、最大轮数、系统角色、语言和任务类型过滤、背压阈值 |
| `[ontology]` | 本体论文件路径 |
| `[output]` | 输出目录设置 |

## 配置参考

所有参数定义在 `configs/config.toml` 中。机密字段（`api_base`、`model_name`、`api_key`）
留空，在 `.local/config.override.toml` 中填写。

### `[input_generator]` — 输入生成器（出题模型）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `api_base` | string | `""` | API 端点 URL（OpenAI 兼容） |
| `model_name` | string | `""` | 模型名称 |
| `api_key` | string | `""` | API 密钥（机密，在 override 中填写） |
| `temperature` | float | `0.8` | 采样温度，越高越随机 |
| `max_tokens` | int \| null | `null`（已注释） | 最大生成 token 数。基础配置中**未设置**——该行被注释掉，由 API 服务商决定；取消注释即可强制限制 |
| `connect_timeout` | float | `10.0` | TCP 连接 + TLS 握手超时秒数 |
| `first_token_timeout` | float | `300.0` | 等待首个 token 的最大秒数（prefill + 排队） |
| `inter_token_timeout` | float | `15.0` | 首个 token 后 token 间最大等待秒数 |
| `retry_on_timeout` | bool | `false` | 超时后是否重试（需 `max_retries > 0`） |
| `max_retries` | int | `3` | 请求失败重试次数 |

> **注意：** 问题生成模型永远不会进入推理模式。它总是以 `enable_thinking = false`
> 显式发送到服务端，且**不可配置** —— `[input_generator]` 没有 `enable_thinking`
> 字段，`[target_model].enable_thinking` 只影响教师模型。生成出的 user 轮会被原样
> 存为锚点的提问，因此推理 token 会先把 `max_tokens` 花在**不该出现在提问里**的
> 文字上。

### `[target_model]` — 目标模型（教师模型）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `api_base` | string | `""` | API 端点 URL |
| `model_name` | string | `""` | 教师模型名称 |
| `api_key` | string | `""` | API 密钥（机密） |
| `temperature` | float | `0.0` | 采样温度，0.0 = 确定性输出 |
| `max_tokens` | int \| null | `null`（已注释） | 最大生成 token 数。基础配置中**未设置**——该行被注释掉，由 API 服务商决定；取消注释即可强制限制 |
| `connect_timeout` | float | `10.0` | TCP 连接 + TLS 握手超时秒数 |
| `first_token_timeout` | float | `300.0` | 等待首个 token 的最大秒数（prefill + 排队） |
| `inter_token_timeout` | float | `15.0` | 首个 token 后 token 间最大等待秒数 |
| `retry_on_timeout` | bool | `false` | 超时后是否重试（需 `max_retries > 0`） |
| `max_retries` | int | `3` | 请求失败重试次数 |
| `enable_thinking` | bool | `false` | 启用推理模式（Qwen3/DeepSeek-R1 等）。开启后模型先输出 `...` 推理过程再输出答案，`content` 和 `logprobs` 均包含推理 token。**仅当蒸馏目标为推理模型时开启**。该值**总会发送到服务端** — `false` 显式关闭推理，`true` 显式开启 |

> **⚠️ 行为变更（v0.3+）：** 旧版本在 `enable_thinking = false` 时**不发送**该参数到服务端
> — 服务端的 chat template 将"未定义"视为"推理开启"，因此 `false` 被静默忽略。
> 自本版本起，`enable_thinking` 会**始终显式发送**。如果你依赖旧行为（`enable_thinking = false` 时
> 推理仍开启），请将其设为 `true`。

### `[generation]` — 生成控制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `target_count` | int | `100` | 目标锚点数量。建议 ≥200 以保证能力维度全覆盖 |
| `seed` | int | `42` | 随机种子 |
| `concurrency` | int | `4` | 并发请求数，过高可能触发限流 |
| `languages` | list | `[]` | 语言过滤（空=全部）。可选：`zh-CN`, `en`, `ja`, `ko` |
| `task_types` | list | `[]` | 任务类型过滤（空=全部） |
| `max_turns` | int | `1` | 最大对话轮数（1=单轮，2-10=多轮） |
| `system_persona` | string | `"none"` | 系统角色模式：`none` / `one_sentence` / `appropriate` / `detailed` |
| `max_turns_with_image` | int | `1` | 含图片的最大轮数（≤ `max_turns`） |
| `embeddings_path` | string | `"ontology/anchor_ontology_embeddings.json"` | 预计算本体论 embedding 文件路径 |
| `backpressure_threshold` | int | `3` | 连续服务端类失败达到该次数即触发冷却 |
| `backpressure_cooldown` | float | `60.0` | 触发阈值后的冷却暂停秒数 |

### `[ontology]` — 本体论

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `path` | string | `"ontology/anchor_ontology.json"` | 本体论 JSON 文件路径 |

### `[output]` — 输出

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `directory` | string | `""` | 输出目录（空=自动生成时间戳目录） |
| `overwrite` | bool | `false` | 是否覆盖已有输出目录 |

### 将字段设为"空"/使用 API 默认值

TOML 不支持 `null` 值。要让 API 服务商自行决定某个值（如 `max_tokens`），
请在 `config.toml` 中**注释掉或删除对应行**：

```toml
[input_generator]
# max_tokens = 4096   ← 注释掉 → API 使用自己的默认值
```

pydantic 配置模型使用 `None` 作为可选字段的默认值。当字段为 `None` 时，
它会被完全从 API 请求中省略。

## CLI

```
ard --config <路径> [--override <路径>] [--image-dir <路径>] [--no-convert]
```

单一命令完成所有操作：

- `--config` — 基础配置 TOML 文件路径（必填）
- `--override` — 覆写配置 TOML 文件路径（可选；未提供时自动检测 `--config` 同目录下的 `config.override.toml`）
- `--image-dir` — 多模态锚点的图片目录（可选）
  - ⚠️ `input_generator` 和 `target_model` 均需支持多模态输入。如果模型不支持多模态，API 会直接报错。
- `--no-convert` — 关闭图片格式自动转换（可选）
  - 默认情况下，流水线会将所有图片自动转换为 JPG/PNG（PNG → 直接复制，RAW/BMP/TIFF/GIF/WebP → JPG quality=95）。
  - 使用此参数时仅接受 PNG/JPEG/GIF/WEBP；**其他格式的文件会被目录扫描静默忽略**——既不转换，也不报错。如果因此没有可用图片，流水线会打 WARNING（`No images found in <dir>. All anchors will be pure text.`），并照常产出**纯文本锚点**。若期望多模态产出，请检查启动日志中的这一行。

## 输出

```
outputs/<dataset_name>/
├── anchor_bank.jsonl          # 统一锚点数据（graspo 兼容）
├── config.json                # 合并后配置快照（保证可复现）
├── images/                    # 多模态图片（如有）
├── logs/
│   ├── ard.log                # 人类可读流水线日志（INFO+）
│   ├── ard_debug.log          # 机器可解析调试日志（DEBUG+）
│   └── ard_error.log          # 错误日志（ERROR+）
└── manifest.json              # 统计摘要 + 生成健康计数
```

### `manifest.json`

除锚点库摘要（`total_anchors` / `domains` / `languages` / `capabilities` /
`output_dir`）外，manifest 还回答"这轮生成健康吗"：

```json
"generation": {
  "counters": {
    "requested": 120,
    "succeeded": 101,
    "abandoned_total": 19,
    "abandoned_by_reason": {"timeout": 12, "empty_content": 4, "logprobs_error": 3},
    "written": 100,
    "rejected_invalid_shape": 0,
    "duplicate_ids": 1,
    "backpressure_events": 2
  },
  "failures": {
    "key_missing": 3,
    "empty_content": 4,
    "reasoning_only_responses": 4,
    "truncated_empty": 4
  }
}
```

* `counters` —— 每条被请求的 anchor 的归宿：产出、落盘、被放弃（含机器可读原因）、
  被消息形状门拦下、被 id 去重拦下，以及触发背压冷却的次数。
* `failures` —— 进程级失败计数：log-probs 提取失败按 reason 计数，
  以及能暴露"思考吃光预算"的 reasoning / 空正文计数。

两个子对象**只在非空时写出**，且**零值条目被丢弃**——健康的一轮不会因为新增字段多出噪音，
而不健康的一轮**不可能**看起来健康。

**向后兼容**：`generation` 是**可选**字段。在它出现之前写下的 manifest 仍然可读——
字段缺失应理解为"该轮没有记录生成统计"，而不是全部为零
（用 `manifest.get("generation") is None` 判断，不要用真值判断）。
既有六个字段（`total_anchors` / `domains` / `languages` / `capabilities` /
`output_dir` / `config`，最后一个为合并后配置快照）的语义与形状**未变**。

### 数据格式

所有锚点写入单个 `anchor_bank.jsonl` 文件，格式与 graspo 兼容。多轮锚点总是以
`user` 消息开头**并**以 `user` 消息结尾，角色严格交替，因此消息形状恒为
`U` 或 `UAU`（落盘前强制校验——违反者被拒绝并计入 `rejected_invalid_shape`）。
完整真实记录见 `examples/anchor_bank.sample.jsonl`。

```json
{
  "id": "anchor_a1b2c3d4",
  "source": "ard",
  "messages": [
    {"role": "user", "content": "解释熵的概念..."},
    {"role": "assistant", "content": "熵是衡量系统无序程度的物理量..."},
    {"role": "user", "content": "能举个例子吗？"}
  ],
  "targets": [{
    "id": "primary",
    "output": {
      "content": "当然！冰融化成水...",
      "logprobs": {
        "token_ids": ["当然", "！", "冰", "融化"],
        "log_probs": [-0.1, -0.2, -0.3, -0.05]
      }
    }
  }],
  "anchor_meta": {"language": "简体中文", "knowledge_domain": "science"},
  "teacher_id": "your-model-name"
}
```

多模态锚点的图片内嵌在 `user` 消息的 `content` 中，此时 `content` 是部件列表
而非纯字符串：

```json
{"role": "user", "content": [
  {"type": "image", "image": "images/sample_01.jpg"},
  {"type": "text", "text": "请根据图片内容，判断这张图片拍摄的场景类型。"}
]}
```

## 示例

`examples/` 目录包含样例输入和输出，帮助你无需运行即可了解项目：

```
examples/
├── images/                    # 多模态模式样例图片
│   ├── sample_01.jpg
│   └── ...
└── anchor_bank.sample.jsonl   # 样例输出（5 条锚点：3 条纯文本 + 2 条多模态）
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

当无法获取 git 元数据时，`build.sh` 回退到版本号 `1.0.0`，因此构建出的镜像是 `ard:1.0.0`。
`VERSION` 只覆盖记录在镜像**内部**的软件包版本——它**不会**重命名镜像；重命名请用 `IMAGE_NAME`。

```bash
VERSION=1.0.0 IMAGE_NAME=ard:latest bash docker/build.sh
```

脚本最后一行会打印结果镜像名（`Built: <image>`）。

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

将你的本体论 JSON 文件放入 `ontology/` 目录（或任意路径），
然后在 `configs/config.toml` 中设置 `path`：

```toml
[ontology]
path = "ontology/my_ontology.json"
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

### 多模态锚点生成需要什么模型？

`input_generator` 和 `target_model` 均需支持多模态（视觉-语言）输入。
如果模型不支持多模态，API 会直接返回错误——无需手动维护配置开关。

### 文本锚点和多模态锚点有什么区别？

文本锚点是从本体论生成的对话（单轮或多轮），不含图片。多模态锚点在对话消息中包含图片，
需要提供 `--image-dir` 参数才会生成。两种类型共享相同的输出格式，
写入同一个 `anchor_bank.jsonl` 文件。对话轮数由配置中的 `max_turns` 控制。

### 如何断点续传？

ARD 自动从上次已完成的锚点恢复。只需重新运行相同的命令——流水线会检测
`anchor_bank.jsonl` 中已有的锚点，只生成剩余数量以达到 `target_count`。

### 多模态锚点的多样性从哪里来？

多模态锚点的多样性来自三个独立来源：

1. **Ontology 多样性**（FPS 保证）：FPS 采样器从 10,080 个本体组合中选出
   最优锚点规格，语言/知识域/能力/会话类型等元数据通过 VLM prompt 传递，
   确保 prompt 多样性。即使图片池单一，Ontology 元数据仍然保证 prompt 不重复。

2. **图片池多样性**：从用户指定的图片目录中随机采样图片，
   图片内容本身构成视觉输入的多样性。

3. **VLM 随机性**（temperature=0.8）：Input Generator 的采样温度默认 0.8，
   即使相同的元数据和图片，VLM 也会生成不同措辞的问题。

### 生成的 token_ids 是什么格式？

vLLM API 返回的 `token_ids` 优先使用整数 token ID
（`logprobs.content[].token_id`），fallback 为字符串 token
（`logprobs.content[].token`）。当 teacher 和 student 使用相同 tokenizer 时，
整数 ID 可直接用于 OPD 训练；不同 tokenizer 则字符串形式更通用，
下游可按自己的 tokenizer 重新编码。

### 多模态锚点和文本锚点的比例怎么控制？

当前是**二选一开关**：提供 `--image-dir` 则生成多模态锚点，不提供则生成纯文本锚点。
如需混合比例（如 30% 图片 + 70% 纯文本），需要运行两次并手动合并输出：
一次不带 `--image-dir` 生成纯文本锚点，一次带 `--image-dir` 生成多模态锚点，
然后合并两个 `anchor_bank.jsonl` 文件。

### 推荐的目标数量是多少？

`target_count = 100` 时，知识域和语言的覆盖率可达 100%，
但**能力（capability）的覆盖率约 75-90%**，是覆盖的短板
（20 种能力在 100 个样本中难以全部覆盖）。
建议 `target_count ≥ 200` 以保证能力维度的全覆盖。
各维度实际总数：18 知识域 × 20 能力 × 4 语言 × 7 会话类型 = 10,080 个组合。

## 许可证

MIT
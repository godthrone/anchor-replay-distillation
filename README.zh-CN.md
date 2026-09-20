# ARD — Anchor Replay Distillation

[English](README.md)

> 锚点数据生成框架，专为大模型持续微调中的灾难性遗忘问题设计。
> 本体驱动采样、教师模型蒸馏并把推理过程作为独立字段落盘、
> 原生兼容 Graspo/OPD 训练流程。

**ARD** 生成高质量锚点数据集，防止大模型持续微调过程中的灾难性遗忘。
它从结构化知识本体中采样多样化提示词，调用教师模型生成答案
（教师推理过程与答案分开存放），输出格式直接对接 On-Policy Distillation
(OPD) 和 Graspo 强化学习训练流程。

**核心能力：**
- 🧠 **本体驱动采样** — 基于知识领域、语言、能力、会话类型、system prompt 模式
  五维本体，结合 embedding 分层最远点采样，确保锚点覆盖全面且分布多样。
- 🎯 **教师模型蒸馏（含推理过程）** — 强模型同时给出答案与其推理过程，推理过程
  作为独立的 `reasoning` 字段落盘，为学生提供两个监督目标而不是一个。
- 🖼️ **多模态支持** — 文本锚点和图文锚点统一流程、统一输出，一条命令
  覆盖纯文本和多模态场景。
- 📦 **Graspo 兼容** — 输出格式与 Graspo anchor bank 无缝对接，生成即训练。
- ⚡ **一条命令** — 单 CLI、TOML 配置驱动、机密分离、Docker 一键启动。

## 快速开始

**Clone → 第一个数据集，三条命令。** 你只需要两样东西：Docker，以及一个
OpenAI 兼容的 chat-completions 端点。

### 你需要什么——没有会怎样

| 你需要 | 没有会怎样 | 能否降级照跑？ |
|--------|-----------|----------------|
| **Docker** | `bash run.sh` 完全跑不起来——它首次运行会自动构建镜像，所以 `bash docker/build.sh` 同样跑不起来 | 不能——没有容器运行时什么都跑不了。请先装 Docker。 |
| **一个 OpenAI 兼容端点 + 模型名**（`api_base`、`model_name`），提问与作答各一份 | 运行会在**动手之前**就被拒绝，报一条字段级错误 `ERROR: [input_generator] is missing ...`（或 `[target_model] ...`），点明缺哪个字段；**不会创建任何输出目录** | 可以——两个角色可以指向**同一个**服务/模型，也可以指向本地 vLLM/Ollama/llama.cpp 服务。项目**不提供**离线演示模式。 |
| **API key** | 只有当你的端点校验密钥时才必要 | 可以——`api_key` 不填（或删掉该行）就不会发送 `Authorization` 头。 |
| **本体与嵌入数据**（`ontology/*.json`） | 需要重新用 embedding API 生成 | 不是问题：两个文件都**已随仓库跟踪**，clone 后即存在（55 个预计算嵌入向量，1024 维，供分层 FPS 采样器使用）。只读输入，不是生成产物。 |
| **RAW 图片支持**（`rawpy`，已在 `pyproject.toml` 依赖中） | 无法读取 RAW（CR2/NEF/ARW/DNG/…）输入 | 可以——文本锚点与 JPG/PNG/GIF/WebP 多模态锚点均不受影响。该库以 manylinux wheel 发布，自带 `libraw.so`，导入是懒加载。 |

### 1. clone 仓库（一条命令）

```bash
git clone https://github.com/godthrone/anchor-replay-distillation.git && cd anchor-replay-distillation
```

**不需要单独的构建步骤**：首次运行 `run.sh` 时会发现镜像不存在，于是先构建镜像、
再启动容器（构建委托给 `docker/build.sh`，所以镜像标签仍由你掌控——见
[自定义镜像标签](#自定义镜像标签)）。首次构建会编译依赖层（`uv.lock` 锁定的
75 个包 + `python:3.11-slim` 基础镜像），需要能访问 PyPI，**耗时几分钟**；
之后的每次运行都直接复用镜像、立即开始。脚本打印的 `Built: ard:<version>`
就是构建确实发生过的确认。

`run.sh` 自己**从不从 registry 拉取**——它只在本地不存在该镜像时才构建。
如果你想用**预构建 / 来自 registry 的镜像**，先 `docker pull <tag>` 把它拉到
本地：镜像存在后 `run.sh` 会整块跳过构建，行为与从前完全一致。

### 2. 创建覆写配置并填入字段（一条命令 + 一次编辑）

```bash
mkdir -p .local && cp configs/config.override.sample.toml .local/config.override.toml
```

> **关于模板的先说一句。** `configs/config.override.sample.toml` 为基础配置
> `configs/config.toml` 定义的**每个**字段都列了一行，因此可以兼作字段文档
> ——它是手册，不是清单：你并不需要保留它、也不必填满它。其中有些行**并没有
> 注释掉**（如 `target_count = 100`、`concurrency = 4`、`max_turns = 1`、
> `[ontology].path`）。覆写文件里任何未注释的行都会**覆盖基础配置**——如果你
> 想保留默认值，请把该行注释掉或删除。**绝不要重复声明文件中已有的段**——
> 同一个段写成两份（如两个 `[generation]`）是 TOML 解析错误，不会被合并。
> 与其改整个模板，直接把下面这段写进
> `.local/config.override.toml` 也可以：加载器会把它深度合并进
> `configs/config.toml`，**部分字段的覆写文件是合法的**。

`.local/config.override.toml` 里**必填的是四个值**——两个角色各自的
`api_base` 与 `model_name`。`api_key` **按需**：只有当你的端点校验密钥时才填，
否则删掉该行即可，此时不会发送 `Authorization` 头。其余字段在
`configs/config.toml` 里都有可用的默认值：

```toml
[input_generator]        # 出题端
api_base = "https://your-api.example.com/v1"
model_name = "your-model-name"
api_key = "REPLACE_ME"   # 端点不需要密钥就删掉这行

[target_model]           # 作答端（= 教师：它的答案就是监督目标）
api_base = "https://your-api.example.com/v1"
model_name = "your-model-name"
api_key = "REPLACE_ME"
```

### 3. 生成锚点（一条命令）

```bash
bash run.sh --config configs/config.toml --override .local/config.override.toml
```

加上 `--image-dir examples/images` 就产出多模态锚点（而不是纯文本）。
默认一轮要 **100 个锚点**，耗时取决于你的端点。

### 4. 几秒钟就看到第一个产出（可选）

想在完整跑之前先确认整条链路通，就让某一轮跑一个很小的目标——把这两个值填进
覆写文件里**已经存在的那个 `[generation]` 段**。**不要新增第二个 `[generation]`
表**：如果你按第 2 步拷了整份模板，里面本来就有一个，而 TOML 不允许同一个表被
声明两次（`TOMLDecodeError: Cannot declare ('generation',) twice`）。就地在原有
位置改这两行：

```toml
[generation]             # 这是已经存在的段——不要再写一遍这个表头
target_count = 2
concurrency = 1
```

...然后用同一条命令运行，看产出。以下四条全都命中，就说明流水线跑到了终点：

```bash
ls -t outputs/ | head -1                 # 最新的 ard_dataset_<时间戳>/ 目录
wc -l outputs/ard_dataset_*/anchor_bank.jsonl   # == target_count 行，每行一个 JSON 对象
python3 -c "import json,glob; print(json.load(open(sorted(glob.glob('outputs/ard_dataset_*/manifest.json'))[-1]))['total_anchors'])"
tail -3 outputs/ard_dataset_*/logs/ard.log      # 结尾是：Done! Output: outputs/ard_dataset_...
```

一轮只要通过了凭证校验、却没产出任何锚点，也照样会写出 `config.json` 和
`logs/`——真正区分成功与失败的是 `anchor_bank.jsonl` 的行数。多模态运行还要多看一处：记录的
`data_source` 必须是 `ard_multi`（纯文本运行则是 `ard_text`）。

### 第 2 步或第 3 步失败时

- **`ERROR: [input_generator] is missing ...`**（或 **`[target_model] ...`**）——
  那个配置段的 `api_base` / `model_name` 还是空的（`configs/config.toml` 里两者
  都写作 `""`，空值一律按"未设置"处理）。消息会点明每个缺失字段，而且运行在
  **创建任何输出目录、写出任何文件之前**就停下，没有残留需要清理。按第 2 步把
  字段填进 `.local/config.override.toml` 即可。
- **`ERROR: override config not found: <路径>`** —— `--override` 路径写错了。
  如果**省略** `--override`，它只会自动检测**与 `--config` 同目录**的
  `config.override.toml`——那是 `configs/`，不是 `.local/`。请像上面那样显式
  传 `--override .local/config.override.toml`。
- **首次运行自动构建镜像失败** —— 要么是 `run.sh: docker not found in PATH`
  （没装 Docker），要么是
  `run.sh: building image '<tag>' failed (docker/build.sh exit N)`（Docker 没运行，
  或构建无法访问 PyPI）。构建配方就在本仓库里，镜像绝不会从任何 registry 拉取；
  排除原因后手动构建一次即可：`bash docker/build.sh`。

以下内容都是"想调参时再看"的参考资料——**上手第一份数据集不需要读它们**。

## 配置

ARD 采用**分层 TOML 配置**模型。有两个配置文件：

| 文件 | 用途 | 是否 git 跟踪？ |
|------|------|----------------|
| `configs/config.toml` | 基础配置 — 含全部字段和默认值 | ✅ 是 |
| `.local/config.override.toml` | 覆写配置 — 部署环境机密信息 | ❌ 否（`.gitignore` 排除） |

**`configs/config.toml`** 是配置结构的唯一权威来源。它定义了所有字段及其
合理的默认值。非机密字段（如 `temperature`、`max_tokens`、`seed`）开箱即用。
端点字段（`api_base`、`model_name`）留空，必须由覆写文件填充。`api_key` 是
可选的——它同样留空，只有当你的端点校验密钥时才需要填。

**`.local/config.override.toml`** 只包含你需要覆写的字段——通常是
`[input_generator]` 和 `[target_model]` 的 `api_base`、`model_name`、
`api_key`。它存放在 `.local/` 目录中，不会被提交到 git。不能添加基础配置
中不存在的字段。

启动时，覆写文件深度合并到基础配置中。合并后的结果是一份唯一的配置 dict，
程序中不存在两个配置源。

| Section | 用途 |
|---------|------|
| `[input_generator]` | 生成用户提问的 VLM/LLM（出题端） |
| `[target_model]` | 目标作答模型（=教师端）：生成答案（含推理过程） |
| `[generation]` | 目标数量、随机种子、并发数、最大轮数、系统角色、语言和任务类型过滤、背压阈值 |
| `[ontology]` | 本体论文件路径 |
| `[output]` | 输出目录设置 |

## 配置参考

所有参数定义在 `configs/config.toml` 中。端点字段（`api_base`、`model_name`）
留空，在 `.local/config.override.toml` 中填写；端点不需要 bearer token 时
`api_key` 可以一直不填。

### 模型术语对照表 — ARD vs 下游 SFT/OPD

一张表 + 一句话规则，确保 ARD 里的两个角色永远不会和下游框架谈论的角色混淆：

| ARD 内名称 | 下游 SFT/OPD 语境 | 职责 |
|-----------|-------------------|------|
| `input_generator`（提问/输入生成器，即"出题端"） | 无对应（SFT/OPD 不"出题"） | 生成 `user` 轮（出题端） |
| `target_model`（目标作答模型） | **教师模型 (teacher)** | 生成目标答案 `targets[0].output`（教学信号） |
| — | 学生模型 (student) | 学习者——**ARD 不参与**（由 graspo/SFT 负责） |

> **一句话规则："谁的输出是目标答案，谁就是教师"。** 在 ARD 里目标答案只有
> 一个来源——`target_model`；输入生成器只提问、从不产出监督信号。ARD 从不
> 运行学生模型。

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

> **注意：** 问题生成模型永远不会进入推理模式。它总是以
> `chat_template_kwargs.enable_thinking = false` 显式发送到服务端，且**不可配置**
> —— `[input_generator]` 没有 `enable_thinking` 字段，`[target_model].enable_thinking`
> 只影响教师模型。生成出的 user 轮会被原样
> 存为锚点的提问，因此推理 token 会先把 `max_tokens` 花在**不该出现在提问里**的
> 文字上。

### `[target_model]` — 目标作答模型（=教师端）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `api_base` | string | `""` | API 端点 URL |
| `model_name` | string | `""` | 教师模型名称 |
| `api_key` | string | `""` | API 密钥（机密） |
| `temperature` | float | `0.1` | 教师作答的采样温度。默认 0.1（R12 裁决，原 0.0）：接近贪心，但让教师在不同锚点间变化措辞——既有答案多样性，又不至于漂移成不一致的答案 |
| `max_tokens` | int \| null | `null`（已注释） | 最大生成 token 数。基础配置中**未设置**——该行被注释掉，由 API 服务商决定；取消注释即可强制限制 |
| `connect_timeout` | float | `10.0` | TCP 连接 + TLS 握手超时秒数 |
| `first_token_timeout` | float | `300.0` | 等待首个 token 的最大秒数（prefill + 排队） |
| `inter_token_timeout` | float | `15.0` | 首个 token 后 token 间最大等待秒数 |
| `retry_on_timeout` | bool | `false` | 超时后是否重试（需 `max_retries > 0`） |
| `max_retries` | int | `3` | 请求失败重试次数 |
| `enable_thinking` | bool | `false` | 启用推理模式（Qwen3/DeepSeek-R1 等）。开启后，模型的推理过程流入 `targets[0].output` 中独立的 `reasoning` 字段，**绝不**并入 `content`；关闭时 `reasoning` 为 `null`。**仅当蒸馏目标为推理模型时开启**。该值**总是以 `chat_template_kwargs.enable_thinking` 发送**——嵌套一层。请求体里**没有**顶层的 `enable_thinking` 键，写顶层也无效：读取这个开关的是服务端的 **chat template**，不是 OpenAI API，而只有 `chat_template_kwargs` 会被转发进去。`false` 显式关闭推理，`true` 显式开启 |

> **⚠️ 行为变更（v0.3+）：** 旧版本在 `enable_thinking = false` 时**不发送**
> `chat_template_kwargs.enable_thinking` 到服务端 — 服务端的 chat template 将
> "未定义"视为"推理开启"，因此 `false` 被静默忽略。
> 自本版本起，该键会**始终显式发送**，且**嵌套在 `chat_template_kwargs` 内**。
> 为什么必须嵌套：顶层的 `enable_thinking` 键是**无效的**——只有
> `chat_template_kwargs` 会被转发进 chat template，写在顶层等同于没写，
> 推理会静默重新开启。如果你依赖旧行为（`enable_thinking = false` 时
> 推理仍开启），请把该配置字段设为 `true`。

### `[generation]` — 生成控制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `target_count` | int | `100` | 目标锚点数量。能力维度是覆盖短板：单次运行约覆盖 30% 的能力（**单 seed 实测，该比例随 seed 波动**），且随 `target_count` 上升缓慢（见[常见问题](#推荐的目标数量是多少)） |
| `seed` | int | *省略 → 随机* | 采样种子。省略则每轮取一个新的随机种子；显式设为整数固定该轮的采样顺序 |
| `concurrency` | int | `4` | 并发请求数，过高可能触发限流 |
| `languages` | list | `[]` | 语言过滤（空=全部）。取值必须与本体**精确匹配**：`English`、`简体中文`、`Español`、`日本語` |
| `task_types` | list | `[]` | 任务类型过滤（空=全部） |
| `max_turns` | int | `1` | 最大对话轮数（1=单轮，2-10=多轮） |
| — | — | — | 系统提示词不再是配置开关：由本体采样决定（`system_prompt_presence` / `system_prompt_style`），带 system 的锚点其文本在运行时生成并写入 `messages[0]` |
| `max_turns_with_image` | int | `1` | 含图片的最大轮数（≤ `max_turns`） |
| `embeddings_path` | string | `"ontology/anchor_ontology_embeddings.json"` | 预计算本体论 embedding 文件路径 |
| `criterion` | string | `"max"` | 域内最远点选择的贪心规则。`"max"` = 经典最远点（保持 `"max"` 可逐位复现此前所有版本）；`"sum"` = total-blankness / 贪心设施选址，它需要完整的 (n, n) 距离矩阵，因此超过约 11,585 行（1 GiB）时会被**拒绝**，而不是静默降级。**而在生产路径上它本来就会被拒绝：** 常规运行的余量填充那一段云约 **50,310** 行，需要约 **18.9 GiB**，因此在当前本体组合云的生产路径上被**规模预算拒绝**。也就是说 `"sum"` **在生产路径上不可选**——预算直接取消该选项，而不是假装结果没变。该预算**只**约束 `"sum"`（`"max"` 根本不会物化距离矩阵，因此没有这个上限），所以这属于规模限制，而不是"`max` 还是 `sum`"的偏好选择。其他取值会在**配置加载时被拒绝**——不存在静默回退 |
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
├── config.json                # 合并后配置快照（可复现性）
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
    "abandoned_by_reason": {"empty_content": 4, "timeout": 12},
    "written": 100,
    "rejected_invalid_shape": 0,
    "duplicate_ids": 1,
    "backpressure_events": 2
  },
  "failures": {
    "reasoning_only_responses": 4,
    "truncated_empty": 4
  }
}
```

* `counters` —— 每条被请求的 anchor 的归宿：产出、落盘、被放弃（含机器可读原因）、
  被消息形状门拦下、被 `data_source` 词表门拦下、被 id 去重拦下，以及触发背压冷却的次数。
  完整键集为 `requested` / `succeeded` / `abandoned_total` / `abandoned_by_reason` /
  `written` / `rejected_invalid_shape` / `rejected_invalid_data_source` /
  `duplicate_ids` / `backpressure_events`（零值条目被丢弃）。
  `abandoned_by_reason` 的键就是代码实际记录的标签，例如服务端不稳定类的
  `timeout` / `transport_error`，模型输出类的 `empty_content`、
  `answer_too_short` / `answer_too_long`、`invalid_shape`、`role_mismatch`、
  `unknown_role`、`history_not_advanced`、`no_final_turn`。
* `failures` —— 针对教师模型**返回的响应**的进程级计数：`responses` /
  `reasoning_responses` / `reasoning_chars` / `reasoning_only_responses` /
  `empty_content` / `truncated_empty`。它们能暴露"思考吃光预算"
  （`truncated_empty` = 正文为空 **且** `finish_reason == "length"`；
  `empty_content` 是 `reasoning_only_responses` 的超集）。

两个子对象**只在非空时写出**，且**零值条目被丢弃**——健康的一轮不会因为新增字段多出噪音，
而不健康的一轮**不可能**看起来健康。

**如何读日志行：这些计数器不是"只算教师端"。** 生成结束时 ARD 会打印
`Target model reasoning stats: N response(s), M with reasoning …`，但它背后的计数器
（`responses` / `reasoning_responses` / `reasoning_chars` / …）是**模块级、由所有走流式的
客户端共享**的：它记录的是锚点生成期间的增量，因此**同时包含输入生成器（出题）与目标模型
（作答）的完成数**。所以一个目标模型每次作答都带思考的运行，仍可能读到
`378 response(s)` 中只有 `149 with reasoning`——其余是输入生成器的响应，它们只返回正文。
manifest 里的 `generation.failures` 报告的是同一批计数器，口径相同。若要单独判断**教师端**，
请依据记录自身的 reasoning 字段（`targets[0].output.reasoning`），或让输入生成器也开启思考；
仅凭这行日志无法把计数归给某一个模型。

`generation` 的子对象的键就是代码实际记录的计数器；上面的示例取自该集合
（`generation.counters` 的字段见 `AnchorGenerationStats.to_manifest_dict`，
`generation.failures` 的字段见 `reasoning_stats()`）。

**向后兼容**：`generation` 是**可选**字段。在它出现之前写下的 manifest 仍然可读——
字段缺失应理解为"该轮没有记录生成统计"，而不是全部为零
（用 `manifest.get("generation") is None` 判断，不要用真值判断）。
既有六个字段（`total_anchors` / `domains` / `languages` / `capabilities` /
`output_dir` / `config`，最后一个为合并后配置快照）的语义与形状**未变**。
v3 写出的 manifest 顶层还会带 `system_prompt_modes` 与 `data_sources`。

### 数据格式

所有锚点写入单个 `anchor_bank.jsonl` 文件，格式与 graspo 兼容。多轮锚点总是以
`user` 消息开头**并**以 `user` 消息结尾，角色严格交替，因此消息形状恒为
`U` 或 `UAU`（落盘前强制校验——违反者被拒绝并计入 `rejected_invalid_shape`）。
完整真实记录见 `examples/anchor_bank.sample.jsonl`。

**单次运行 = 单个 `data_source`。** 一轮写出的每条记录携带同一个
`data_source`：不带 `--image-dir` 时为 `ard_text`，带 `--image-dir` 时为
`ard_multi`。这是代码级结构保证，不是约定。注意它是**按运行而非按文件**：
两次运行若共用同一个输出目录，两种形态会落进同一个 bank（见下方续跑边界），
因此要跑两种形态请使用**独立的输出目录**，各得一个 `anchor_bank.jsonl`。
另有两种情况即使传了 `--image-dir` 也全为 `ard_text`：图片目录中没有可用图片，
或 `max_turns_with_image = 0` 完全关掉了图片轮。

**边界——续跑（resume）可能让同一个 bank 混装两种形态。** 续跑会把后一次运行
产出的记录追加到前一次写出的 bank 里。若 run 1 写了 `ard_text`，run 2 带
`--image-dir` 续跑同一目录，该 `anchor_bank.jsonl` 就会同时含两种 `data_source`。
**因此消费端必须以记录级 `data_source` 为路由键，不得按 bank 级假设单一取值。**

anchor id 由五维元数据哈希得到，因此只在**单次运行内唯一**，跨 run 不承诺唯一：
合并多个 bank 时，不要期望一个 id 能全局标识一条记录。

`seed` 是可选字段，默认不设置——此时每轮从系统随机源取一个新的种子。显式设为
整数固定该轮的采样顺序。

```json
{
  "id": "anchor_a1b2c3d4",
  "source": "ard",
  "data_source": "ard_text",
  "schema_version": "3.0.0",
  "messages": [
    {"role": "system", "content": "你是一位严谨的科学讲师。"},
    {"role": "user", "content": "解释熵的概念..."},
    {"role": "assistant", "content": "熵是衡量系统无序程度的物理量..."},
    {"role": "user", "content": "能举个例子吗？"}
  ],
  "targets": [{
    "id": "primary",
    "output": {
      "content": "当然！冰融化成水...",
      "reasoning": "用户要一个具体例子，所以挑一个相变过程..."
    }
  }],
  "anchor_meta": {"language": "简体中文", "knowledge_domain": "science_exploration"},
  "input_generator_model": "your-question-model",
  "teacher_id": "your-model-name"
}
```

开头的 `system` 消息是**可选**的（最多一条，且只能在位置 0），只有当本体为该
anchor 采样到 system prompt 时才存在——`messages` 数组是它的唯一真相源，不存在
单独的顶层字段。教师未思考时（`enable_thinking = false`）`reasoning` 为 `null`
（绝不是 `""`）。

v3 记录中**不存在** `logprobs` / `token_ids` / `log_probs` 任何键：ARD 不请求、
不采集、不落盘 token 级 log-probabilities（为什么见 `docs/architecture.md` §5 与
§9.3）。教师 logprob 属于 OPD 训练，在训练时由原版 LLM 教师现场给出。

多模态锚点的图片内嵌在 `user` 消息的 `content` 中，此时 `content` 是部件列表
而非纯字符串：

```json
{"role": "user", "content": [
  {"type": "image", "image": "images/sample_01.jpg"},
  {"type": "text", "text": "请根据图片内容，判断这张图片拍摄的场景类型。"}
]}
```

## 示例

`examples/` 目录包含**真实**的输入与输出，帮助你无需运行即可了解项目。
这里的内容全部由管线真实运行产出，没有任何手工编造的样例：

```
examples/
├── README.md                    # 本目录内容说明与格式解读
├── images/                      # 多模态模式样例图片
│   ├── sample_01.jpg
│   └── ...                      # 10 张小 JPEG（宽 400 px，高度不一），适合冒烟试跑
├── anchor_bank.sample.jsonl     # 样例输出：6 条真实锚点
│                                #   （2 条单轮 `U` + 4 条 `SU` = system + user、
│                                #    4 种语言、每条多模态记录 1 张图）
└── manifest.sample.json         # 真实运行会一并写出的 manifest 清单
```

建议先读 `examples/README.md`——它逐字段解释了产物 schema，并说明为什么
`messages` 总是以 `user` 轮开头、以 `user` 轮结尾。

你可以在 GitHub 上直接浏览 `examples/` 查看输入输出格式。

## 原理（算法总览）

在翻配置参考之前，建议先看 [`docs/ard-algorithm.md`](docs/ard-algorithm.md)：它先给**算法的宏观全貌**，
再逐个看完**九个环节各自的具体操作**；不用公式、不写参数取值、不展开代码结构，每一步都带一行
"实现状态"（即当前真正在跑的是哪一步）。它与 `docs/architecture.md` 互补——后者写的是模块边界与数据流。

[`docs/measurement-methods.md`](docs/measurement-methods.md) 回答的是另一个问题：**哪个变量真的在移动语言空间、
下一份预算该花在哪**。它存放采样权重背后的**测量方法学**（口径、基线、零分布、可复现纪律），
适合要重推或重跑一次测量时查阅；只跑 ARD 不需要它。

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
api_key = "REPLACE_WITH_YOUR_API_KEY"

[target_model]
api_base = "https://your-api.example.com/v1"
model_name = "your-model-name"
api_key = "REPLACE_WITH_YOUR_API_KEY"
```

启动时覆写文件会与 `configs/config.toml` 深度合并。只有机密信息放这里，
其他配置仍在 `configs/config.toml` 中。

覆写文件可以是**部分字段**的——不需要把样例模板里的每个字段都留着。但如果你
确实整份复制了模板，注意其中未注释的行（`target_count`、`concurrency`、
`max_turns`、`[ontology].path` 等）会覆盖基础配置；想保留默认值的请注释掉。

### 没有 API key / 没有模型端点，能先试试吗？

**没有离线演示模式**：每条锚点都要调用所配置的两个模型，项目不自带任何端点。
没有凭证时，你仍能验证镜像能构建、配置能加载——随后运行会停在配置边界上，
报一条字段级错误 `ERROR: [input_generator] is missing ...`，点明是哪个配置段、
缺哪些字段，而且是在**创建输出目录、写出任何文件之前**停下。如果你的端点不需要
密钥，`api_key` 留空即可，请求不会带认证头。两个角色也可以指向同一个本地
OpenAI 兼容服务（vLLM、Ollama、llama.cpp、LM Studio 等），这是看到完整一轮
产出最省事的办法。

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
每条记录包含 `id`、`source`、`data_source`、`schema_version`、`messages`、
`targets`（其 `output` 含 `content` 和 `reasoning`）、`anchor_meta`、
`input_generator_model` 和 `teacher_id`。格式与 graspo 的锚点库格式兼容。
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
需要提供 `--image-dir` 参数才会生成。两种类型的输出记录格式相同，但单次运行只会产出
其中一种（见**数据格式**）：每种模式各跑一次，各有自己的 `anchor_bank.jsonl`。
对话轮数由配置中的 `max_turns` 控制。

### 如何断点续传？

ARD 自动从上次已完成的锚点恢复。只需重新运行相同的命令——流水线会检测
`anchor_bank.jsonl` 中已有的锚点，只生成剩余数量以达到 `target_count`。

凭证校验发生在**续跑判断之前**，所以即使 bank 已经补齐，端点字段也必须保持填写：
带着缺失的 `api_base` / `model_name` 重跑一个已完成的目录会被拒绝（见**第 2 步或第 3 步失败时**），
而不是静默地报告"没有剩余工作"。

### 已知行为边界

以下两个行为是**如实记录**而非已修复的缺陷。它们都是当前设计的后果，也都**不改变**
`data_source` / `id` 字段的含义。

**1. 续跑不保证补满 `target_count`。** 续跑按行数差额重采样
（`remaining = target_count - existing_count`），而 anchor **id** 只由五个采样维度哈希而来
——既不含 seed 也不含运行身份。因此**同一个 seed** 下，续跑会沿同一采样顺序再次抽到 bank
里已有的组合，这些记录被 id 门拦下并计入 `duplicate_ids`（bank 侧的 `duplicate_skipped`），
bank 可能停在 `target_count` **之下**。实测：bank 已有 60 条、请求再补 40 条，实际只写入
5 条，终值 65。默认 `seed`（省略 → 每轮取新随机种子）每次抽的是新组合，通常可正常补满；
显式固定 seed 时该现象更明显。重复运行**不会破坏** bank，只是可能不会让它变大。若必须补满，
请以未设置的 seed 续跑，或删除 bank 并用 `output.overwrite = true` 重新生成。

**2. v3 修复前的历史产物可能把图片记录标错。** 早期版本存在一个 bug：会把对话中确实含图片的
记录写成 `data_source = "ard_text"`（或 `anchor_meta` 中缺 `has_image` 键）。这些目录
**不在 git 内**——它们是 `.gitignore` 下的本地 `outputs/` 产物。消费旧产物前请逐条核实：
把 `data_source` 与 `messages` 中是否真含图片部分对照检查；否则请用当前版本重新生成该 bank。

### 多模态锚点的多样性从哪里来？

多模态锚点的多样性来自三个独立来源：

1. **Ontology 多样性**（FPS 促成）：FPS 采样器从 50,400 个本体组合中选出
   最优锚点规格，元数据（语言/知识域/能力/会话类型/system prompt 模式）参与锚点
   采样，其中语言/能力/会话类型会写进 VLM prompt，从而促成 prompt 多样性——
   即使图片池单一，语言、能力、会话类型不同时 VLM 拿到的指令也不同。注意这
   只是促成而非保证：两类锚点若语言、能力、会话类型相同，VLM 拿到的指令文本
   完全相同（因此依赖第 3 条的随机性）；system prompt 的正文也不参与提问生成，
   它由输入生成模型在运行时单独写出并存入 `messages[0]`。

2. **图片池多样性**：从用户指定的图片目录中随机采样图片，
   图片内容本身构成视觉输入的多样性。

3. **VLM 随机性**（`[input_generator].temperature`，默认 `0.8`）：输入生成器
   按**配置**的采样温度采样——`configs/config.toml` 里的值就是实际发往 API 的值——
   即使相同的元数据和图片，VLM 也会生成不同措辞的问题。

### 生成的 token_ids 是什么格式？

**没有这个字段：v3 记录不含 `token_ids`。** ARD 不向 API 请求
`logprobs`/`top_logprobs`，也不落盘任何 token 级数据，因此 `targets[0].output`
只有 `content`（答案）和 `reasoning`（教师推理过程，未思考时为 `null`）。

历史上（v2，logprob 链路被移除之前）采集到的 token 列表**优先字符串 token**
（`logprobs.content[].token`），当字符串形式缺失时 fallback 到整数 ID
（`logprobs.content[].token_id`）——字符串 token 跨 tokenizer 可移植，
下游可用自己的 tokenizer 重新编码。该机制连同字段本身在 v3 已移除
（见 `docs/architecture.md` §5.2）；若需要 token 级监督信号，请由 OPD 训练时
的原版 LLM 教师现场给出。

### 多模态锚点和文本锚点的比例怎么控制？

当前是**二选一开关**：提供 `--image-dir` 则生成多模态锚点，不提供则生成纯文本锚点。
如需混合比例（如 30% 图片 + 70% 纯文本），请**运行两次**——一次不带 `--image-dir`
生成纯文本锚点，一次带 `--image-dir` 生成多模态锚点——并让两次运行**各用独立的
`output_dir`**。**不要**把第二次运行指向第一次的输出目录：续跑会追加到已有 bank，
从而把两种 `data_source` 混在同一个文件里（见**数据格式**）。

能保持两个 bank 各自独立就最好：这样每个文件都仍然满足"单次运行 = 单个 `data_source`"。
若确实需要合并成单一文件，技术上可行，但**不建议作为默认做法**——因为消费端此后必须按
**记录级** `data_source` 字段路由，不得假设整个 bank 只含一种取值。

### 推荐的目标数量是多少？

**能力（capability）是覆盖短板。** 单次运行实测（`k = 100`，`seed = 42`）
五维覆盖如下：

| 维度 | 总数 | `k = 100` 时覆盖 | 说明 |
|------|------|------------------|------|
| 知识域 | 18 | **18（100%）** | **结构性**：第 1 层 FPS 逐域配额，18/18 由构造保证，不是采样器的功劳 |
| system prompt | 5 | **5（100%）** | 同样是结构性的：每个采样出的 anchor 必带 5 种模式之一 |
| 会话类型 | 7 | 6 | |
| 语言 | 4 | 3 | |
| **能力** | 20 | **6（30%）** | 短板；且分布极不均——最热的 `translation` 拿到 37 条，最冷的 `uncertainty_handling` 只有 1 条，**37:1** |

能力覆盖率随 `target_count` 只是缓慢上升，而且**不保证单调** —— 每域配额是整数
除法（`max(1, target_count // 18)`），`k` 一变采样链就变。实测：

| `target_count` | 200 | 600 | 2800 | 5040 |
|----------------|-----|-----|------|------|
| 覆盖能力数 | 7/20 | 10/20 | 15/20 | 18/20 |

> 以上是**单 seed 实测**，会随 seed 波动。除组合空间本身外，不存在任何一个
> `target_count` 能保证能力维度覆盖完整——提高 `target_count` 只有缓慢收益，
> 且没有单调性保证。

组合空间总数：18 知识域 × 20 能力 × 4 语言 × 7 会话类型 × 5 个 system prompt 取值
= **50,400**；把两种输出形态（文本与多模态）也算上 = **100,800**。如果训练确实需要
能力维度全覆盖，请大幅提高 `target_count`，或增加按维度分层的配额——当前采样器
不提供后者。

## 许可证

MIT
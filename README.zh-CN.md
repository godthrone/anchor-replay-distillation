# anchor-replay-distillation

**Anchor Replay Distillation（ARD，锚点回放蒸馏）把一套由本体定义的坐标空间变成可复现的
监督微调语料。** 对每一个锚点坐标，它先让问题生成模型产出用户轮，再让目标（教师）模型作答，
然后把教师的回答——以及（开启时的）推理过程——与产生它的那个问题一起保存下来。ARD 是蒸馏流水线
中**负责产数据**的一半：它不训练、不蒸馏、也不评测学生模型。

计划来自 v4 本体（`ontology/anchor_ontology.v4.json`），且是**推导出来的，不是配置出来的**：
**935 个合法文本态受限块 + 891 个合法影像态受限块 = 1,826 条锚点**，每个合法块恰好一条，
209 个 `knowledge_domain` 叶（影像态再加 21 个 `visual_domain` 叶）在其上轮转。目标集固定，
运行才可审计：每一个被接受的坐标要么恰好被覆盖一次，要么整个运行拒绝继续。

## 快速开始

三步，最后拿到一份真实产物。这三步会调用真实端点——`--smoke` 只是让第一份产物变小。

### 1. 安装依赖

```bash
uv sync
```

需要 [uv](https://docs.astral.sh/uv/) 与 Python 3.11（由 `.python-version` 锁定）；
`uv.lock` 锁定了全部依赖版本。

### 2. 准备端点凭证

```bash
cp configs/config.override.sample.toml .local/config.override.toml
```

然后填写 `[input_generator]` 与 `[target_model]` 的 `api_base`、`model_name`、`api_key`
（只有在需要 `q95` 读数时才填 `[coverage.embedding]`）。

**如果 `.local/config.override.toml` 已存在，请直接编辑它，不要覆盖复制**——照抄这一步会
静默覆盖那里已经部署好的凭证。

覆写以固定三级优先级合并到 `configs/config.toml` 之上，先命中者胜：

1. 显式 `--override PATH`（文件不存在即报错）；
2. `<project_root>/.local/config.override.toml`——推荐的、被 gitignore 的位置；
3. `<--config 所在目录>/config.override.toml`——向后兼容。

这个选择**从不静默**：每次运行都会在 INFO 日志里写明用了哪个覆写文件，或者写明"只用基础配置"。
字段名按 `extra="forbid"` 校验，拼错的字段是报错，而不是被静默忽略。

**缺凭证时会发生什么**：运行会在**创建任何输出目录之前**停下，给出字段级报文并以退出码 1 结束：

```
ERROR: [input_generator] is missing `api_base`, `model_name`.
```

### 3. 跑冒烟

```bash
bash run.sh --config configs/config.toml --smoke --image-dir examples/images
```

（`run.sh` 入库时未带可执行位，所以用 `bash` 调用。）
`run.sh` 在首次使用时构建 Docker 镜像（需要一次 PyPI 访问），随后在容器里运行
`python -m ard`：`configs/`、`ontology/`、`examples/`、`.local/` 以只读方式挂载，
`outputs/` 可写。没有 Docker 时，用第 1 步装好的环境跑同一个入口：

```bash
uv run python -m ard --config configs/config.toml --smoke --image-dir examples/images
```

**`--smoke` 的语义。** 它以**同一条构造规则**的缩小规模物化计划——1,826 条中的 8 条
（4 个文本块 + 4 个影像块，按枚举序等距取块、含首尾）——让新 clone 的仓库几分钟内就能看到
一份完整形态的产物。这份产物**刻意不完整**，并在三个互相独立的位置自我申报：运行目录名带
`_smoke` 后缀、日志打 WARNING 说明规模、`manifest.json` 里 `smoke: true` 外加一个
`smoke_plan` 块。`--smoke` 是 CLI 的**运行边界参数**，不是配置字段，对不带它的运行不产生任何影响。

产物落在 `outputs/<run_name>/`（默认 `ard_dataset_<YYYYmmdd_HHMMSS>_smoke`；用配置里的
`[output] directory` 可以钉住目录名）。几分钟后应当看到：

- `anchor_bank.jsonl`——8 条锚点，`schema_version 4.0.0`；
- `results/coverage.json` + `results/coverage.md`——验收读数。**结构读数**（计划计数 vs 构造规则）
  零模型调用、恒产出；未配置目标集时它对完整计划报 `within_rule: false`、`metrics: null`，
  这正是冒烟运行该说的话——8 条不是 1,826 条计划，报告不假装它是；
- `manifest.json`——库构成 + 冒烟申报；
- `config.json` 与 `logs/`。

**完整运行**——全部 1,826 条锚点，每一轮一次端点调用。请自备图片目录，按
`<image_dir>/<visual_domain>/<图片文件>` 布局，且 **21 个视觉域齐备**：

```bash
bash run.sh --config configs/config.toml --image-dir /path/to/images
```

完整运行要花掉数小时的端点时间；先用冒烟验证配置是否正确。

## 数据格式

`anchor_bank.jsonl` 每行一个 JSON 对象：

```jsonc
{
  "id": "anchor_…",                 // 对抽中轴取值求 sha256
  "source": "ard",
  "data_source": "ard_text",        // 受控词表：ard_text | ard_multi
  "schema_version": "4.0.0",
  "messages": [ /* 对话 */ ],
  "targets": [
    { "id": "primary",
      "output": { "content": "…教师的回答…",
                  "reasoning": "…教师的推理，或 null…" } }
  ],
  "anchor_meta": { /* 坐标：每个轴一个键 */ },
  "input_generator_model": "…",
  "teacher_id": "…"                 // 其输出即监督目标的模型
}
```

`anchor_meta` 带上采样字段 `modality` 以及每一个轴的取值：`language`、`knowledge_domain`、
`capability`、`system_prompt_mode`、`conversation_type`、`response_style`、`output_format`、
`difficulty`、`context_length`、`input_condition`、`answer_mode`；影像态锚点另有
`visual_domain`。文本态坐标**整个省略** `visual_domain` 键（不放 `null` 占位），代之以
`has_image: false` / `image_count: 0`；影像态锚点则带 `visual_domain` 与
`has_image: true` / `image_count: 1`。
`data_source` 是**逐条**的：`ard_multi` 表示该对话至少带一张图，下游按这个键路由。

`reasoning` 的类型是 `str | null`：当教师未开思考（`target_model.enable_thinking = false`）时
它是 `null`、绝不是 `""`。配置要求推理却拿不到的锚点会被**丢弃并计数**（记在 `manifest.json`），
不会以空值写入。

图片在消息的 content 列表里以**相对路径**引用，绝不内联 base64：

```jsonc
{ "role": "user",
  "content": [ { "type": "image", "image": "images/<visual_domain>/sample_XX.jpg" },
               { "type": "text",  "text": "…生成的用户问题…" } ] }
```

图片取自 `<image_dir>/<visual_domain>/<图片文件>`——图属于锚点自己的 `visual_domain` 坐标，
绝不从一个扁平图片池里抽：扁平池无法保证"图"与贴在它身上的标签一致。路径相对于运行目录，
所以 `outputs/<run_name>/images/<visual_domain>/…` 直接可解析；运行时会把选中的图复制到那里，
并（除非加 `--no-convert`）转码。这些记录的一份小而实用的样例已入库，见
[examples/README.md](examples/README.md)。

## 配置参考

全部字段都定义在 `configs/config.toml`（英文注释、字段齐全），覆写只负责填值。
**产物一律由 config 决定**——CLI 只传输入定位参数与运行边界参数：`--config`、`--override`、
`--image-dir`、`--no-convert`、`--smoke`。

| 段 | 控制什么 |
|---|---|
| `[input_generator]` | 问题生成端点（`api_base`、`model_name`、`api_key`）、采样温度、超时、重试 |
| `[target_model]` | 教师端点——它的回答就是监督目标。`enable_thinking` 打开推理通道（`targets[0].output.reasoning`） |
| `[generation]` | `concurrency`；可选 `seed`（省略则每次运行从系统随机源抽新种子，实际使用的种子记在本次运行的 `config.json` 里）；背压阈值。锚点条数与对话轮数**不可配置**——它们由本体推导 |
| `[ontology]` | v4 本体路径（`ontology/anchor_ontology.v4.json`） |
| `[output]` | `directory`（留空 = `outputs/ard_dataset_<timestamp>`）、`overwrite`（默认 `false`：已有锚点库是续跑，不是替换） |
| `[images]` | `skip_missing_images`（默认 **`false`**）：缺某个 `visual_domain` 图片目录时报错，而不是跳过 |
| `[coverage]` | `enabled`（默认 `true`）与 `target_set_path`——指标读数用的目标集；留空即只出结构读数 |
| `[coverage.embedding]` | 指标读数用的 OpenAI 兼容 `/embeddings` 端点、模型、期望 `dimension`、批大小与超时 |

`configs/config.override.sample.toml` 以注释模板镜像了全部字段，也就是第 2 步复制的那个文件。

## 输出说明

```text
outputs/<run_name>/          # 默认 ard_dataset_<YYYYmmdd_HHMMSS>；--smoke 追加 _smoke
├── anchor_bank.jsonl        # 每行一条记录，schema_version 4.0.0
├── config.json              # 合并后的配置快照，凭证已脱敏
├── logs/                    # ard.log / ard_debug.log / ard_error.log
├── results/
│   ├── coverage.json        # 机器可读的验收读数
│   └── coverage.md          # 同一份读数的人读版
└── manifest.json            # 库构成、运行健康、配置、acceptance 指针
```

`results/coverage.{json,md}` 是**验收读数，不是训练数据**：

- **结构读数**（计划计数 vs 构造规则）恒产出，且零成本——不调用任何模型；
- **指标读数**（`q95`、带 ε±5% 敏感带的 `Extent(ε)`、配对 bootstrap CI、噪声带）需要同时配置
  `coverage.target_set_path` 与 `[coverage.embedding]`。

两者缺一时，运行只写结构读数，并用 WARNING 明说缺口（`metric readout not measured …`），
manifest 里 `metric_readout: false`、`q95: null`——绝不产出一份"看起来干净"的报告。
尺子的定义与分辨力边界见 [docs/measurement.md](docs/measurement.md)。

## 开发指南

```bash
uv sync --extra dev     # 测试、类型检查、lint 工具
uv run pytest
uv run ruff check src/ tests/
uv run mypy src/ard/
```

同一套流程也写在 [CONTRIBUTING.md](CONTRIBUTING.md) 里。

**`--smoke` 与完整运行的区别**：构造规则、采样与逐条生成路径完全相同，只有计划规模不同
（8 条 vs 1,826 条）。冒烟产物刻意不完整、不得当作数据集交付——`_smoke` 后缀、日志 WARNING 与
`manifest.json: smoke` 三处申报就是为此。两条路径都会调用已配置的端点；它们都不是离线演示。

## FAQ

**缺图会怎样？**
默认整个运行会被拒绝，且发生在**创建输出目录之前**：报错逐项列出缺失的视觉域、它的期望路径
（`<image_dir>/<visual_domain>/`）以及受影响条数。可以补齐图片，也可以设
`[images] skip_missing_images = true`：那些锚点会被丢掉、逐条打 WARNING，跳过条数与涉及的域在
`manifest.json` 里申报——跳过永不静默。完全不传 `--image-dir` 时，运行根本不会附图：影像态锚点
仍会被生成成纯文本对话，同时保留自己的 `visual_domain` 坐标。想要图片就传 `--image-dir`。

**为什么 q95 是主尺子而覆盖率只作参考？**
`q95` 是每个目标点到其最近锚点距离的 95 分位（Hyndman–Fan type-7）：一个尾部统计量，读作
"目标集里最难覆盖的那 5% 到底有多远"。`Extent(ε)`（落在 ε 半径内的目标点比例）只作次级读数，
因为它随 ε 的取法而变——所以它与自己的 ε±5% 敏感带并列发布，而不是当成一个过/不过的数字。
定义与这条区分背后的敏感性说明见 [docs/measurement.md](docs/measurement.md)。

**噪声带是什么？**
同一个坐标重复生成并不会得到同样的文本，同一坐标的两个回答之间因此存在一段距离。噪声带就是这些
同坐标配对距离的分布 `[q50, max]`。落在这个带内的 `q95` 差值**无法**与生成随机性区分开；所以该
读数用于自证与回归，不用来主张细微的算法优劣，落在带内的读数会写成"不可分辨"，而不是"更好"。
细节见 [docs/measurement.md](docs/measurement.md)。

**`MULTI_TURN_DEFAULT = 4` 是什么口径？**
本体对两个 `conversation_type` 只写了 `turns: "multi"`，没有数值上界。实现把"multi"读作**本体
自己声明过的最大轮数**（`constraint_update` = 4），把该数字收在单一常量处，并在本体一旦声明更大
轮数时直接报错——绝不静默截断。轮数不是配置项：要改就改本体（或那个常量），不是改
`configs/config.toml`。见 [docs/algorithm.md](docs/algorithm.md) §3。

**本体 md5 变过——既有读数要重算吗？**
不用。v4 本体的 md5 因为一行 `supersedes` provenance 文本改动而变过一次；四个计数、无解对分组与
1,826 行计划都没有变，所以既有的 `q95` 读数、计数与计划都不需要重算。指纹与逐项核对见
[docs/algorithm.md](docs/algorithm.md) §5。

## 许可证与贡献

MIT——见 [LICENSE](LICENSE)。欢迎贡献：开发环境、质量命令与提交约定见
[CONTRIBUTING.md](CONTRIBUTING.md)。

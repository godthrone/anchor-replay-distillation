# anchor-replay-distillation

**Anchor Replay Distillation（ARD，锚点回放蒸馏）把一套由本体定义的坐标空间变成可复现的
监督微调语料。** 对每一个锚点坐标，它先让问题生成模型产出用户轮，再让目标（教师）模型作答，
然后把教师的回答——以及（开启时的）推理过程——与产生它的那个问题一起保存下来。ARD 是蒸馏流水线
中**负责产数据**的一半：它不训练、不蒸馏、也不评测学生模型。

计划来自 v4 本体（`ontology/anchor_ontology.v4.json`），且是**推导出来的，不是配置出来的**：
**935 个合法文本态受限块 + 891 个合法影像态受限块 = 1,826 条锚点**——**每个模态组内每块一条**，
209 个 `knowledge_domain` 叶（影像态再加 21 个 `visual_domain` 叶）在其上轮转。891 个影像态块是
935 个的子集，因此这些坐标各出现两次（每模态一次），二者靠 `modality` 区分，而 `modality` 正是
查重所用的坐标身份的一部分。目标集固定，
运行才可审计：每一个被接受的坐标要么恰好被覆盖一次，要么整个运行拒绝继续。

## 快速开始

三步，最后拿到一份真实产物。这三步会调用真实端点——`--smoke` 只是让第一份产物变小。

### 1. 安装依赖

```bash
uv sync
```

需要 [uv](https://docs.astral.sh/uv/) 与 Python 3.11（由 `.python-version` 锁定）；
`uv.lock` 锁定了全部依赖版本。包版本由 `setuptools_scm` 从 git 标签推导：`git clone` 得到
标签对应的确切版本；而不带 `.git` 的源码归档（GitHub 的 "Download ZIP"、`git archive`）
回退到固定版本 `1.0.0`——与 `run.sh`、`docker/build.sh` 在没有可描述标签时使用的值一致。

**默认安装不含 RAW 支持。** 一次普通 `uv sync` 不会装上 `rawpy`——它在 `raw` extra 里——所以默认
安装不含 LibRaw。默认依赖闭包以宽松许可证为主（MIT / BSD / MIT-CMU / PSF-2.0），但**并非只有
MIT/Apache**：`certifi` 是 **MPL-2.0**（由 `httpx` 传递引入）、`tqdm` 是 **`MPL-2.0 AND MIT`**
（双许可，可按 MIT 选用）、`typing-extensions` 是 **PSF-2.0**。RAW 相机格式（`.cr2`、`.nef`、
`.arw`、`.dng` 等）需要 `rawpy`，而它的 wheel 捆绑了 LibRaw（LGPL-2.1 / CDDL-1.0）——这正是 RAW
支持做成可选 extra 的原因。要处理 RAW 文件，请显式安装该 extra：

```bash
uv sync --extra raw
```

未安装时，RAW 输入会以一条点名该文件的 WARNING 被跳过，而不会让运行崩溃——原因（许可证）
与细节见下面的 FAQ。完整逐包清单与"如何从自己装好的环境复现它"见
[`docs/licenses.md`](docs/licenses.md)；那里只陈述事实，不作合规判定。

### 2. 准备端点凭证

```bash
mkdir -p .local && cp configs/config.override.sample.toml .local/config.override.toml
```

`mkdir -p` 不是装饰：`.local/` 被 gitignore，全新 clone 里没有这个目录，直接 `cp` 会以
`No such file or directory` 失败。两版 README 都把它作为一行执行。

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
./run.sh --config configs/config.toml --smoke --image-dir examples/images
```

`run.sh` 在首次使用时构建 Docker 镜像（需要一次 PyPI 访问），随后在容器里运行
`python -m ard`：`configs/`、`ontology/`、`examples/`、`.local/` 以只读方式挂载，
`outputs/` 可写。没有 Docker 时，用第 1 步装好的环境跑同一个入口：

```bash
uv run python -m ard --config configs/config.toml --smoke --image-dir examples/images
```

**`--smoke` 的语义。** 它以**同一条构造规则**的缩小规模物化计划——8 个受限块
（4 个文本块 + 4 个影像块，按枚举序等距取块、含首尾）——8 个计划槽位，但只有 **6 个不同
`RestrictedBlock`**：其中两块各以"每模态一次"出现，正是上文 891⊂935 的同一子集口径。这些块是全量计划所枚举块集合的**子集**
（块级覆盖 8/8），但它们的坐标在冒烟尺度上**重新轮转**，所以这 8 条锚点**不是** 1,826 条计划
中的行（坐标级仅 1/8 重合）。它的用意是让新 clone 的仓库几分钟内就能看到
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
- `config.toml` 与 `logs/`。运行尚未结束时还会有一份 `plan_identity.in_progress.json`：
  把目录绑定到计划的中间记录；运行完成后只保留 `manifest.json` 作为申报。

**完整运行**——全部 1,826 条锚点，每一轮一次端点调用。请自备图片目录，按
`<image_dir>/<visual_domain>/<图片文件>` 布局，且 **21 个视觉域齐备**：

```bash
./run.sh --config configs/config.toml --image-dir /path/to/images
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
并（除非设 `[images] convert = false`）转码。这些记录的一份小而实用的样例已入库，见
[examples/README.md](examples/README.md)。

## 配置参考

全部字段都定义在 `configs/config.toml`（英文注释、字段齐全），覆写只负责填值。
**产物一律由 config 决定**——CLI 只传输入定位参数与运行边界参数：`--config`、`--override`、
`--image-dir`、`--smoke`。

| 段 | 控制什么 |
|---|---|
| `[input_generator]` | 问题生成端点（`api_base`、`model_name`、`api_key`）、采样温度、超时、重试 |
| `[target_model]` | 教师端点——它的回答就是监督目标。`enable_thinking` 打开推理通道（`targets[0].output.reasoning`） |
| `[generation]` | `concurrency`；可选 `seed`（省略则每次运行从系统随机源抽新种子，实际使用的种子记在本次运行的 `config.toml` 里）；背压阈值。锚点条数与对话轮数**不可配置**——它们由本体推导 |
| `[ontology]` | v4 本体路径（`ontology/anchor_ontology.v4.json`） |
| `[output]` | `directory`（留空 = `outputs/ard_dataset_<timestamp>`）、`overwrite`（默认 `false`：已有锚点库是续跑，不是替换） |
| `[images]` | `convert`（默认 **`true`**）：接受 RAW/BMP/TIFF/GIF/WebP 并把选中的图统一转成 JPEG（RAW 另需可选 `raw` extra）；设 `false` 则只接受已适合网络的格式并原样复制。`skip_missing_images`（默认 **`false`**）：缺某个 `visual_domain` 图片目录时报错，而不是跳过 |
| `[coverage]` | `enabled`（默认 `true`）与 `target_set_path`——指标读数用的目标集；留空即只出结构读数 |
| `[coverage.embedding]` | 指标读数用的 OpenAI 兼容 `/embeddings` 端点、模型、期望 `dimension`、批大小与超时 |

`configs/config.override.sample.toml` 以注释模板镜像了全部字段，也就是第 2 步复制的那个文件。

### 接口变更：图片转码开关移入 config

新增 `[images] convert`，并删除 `--no-convert` CLI 参数。该参数决定 `outputs/<run>/images`
里落盘的内容，按项目"一份 config 描述一份产物"的工程规则，这个决定必须放进 config：留档的
`config.toml` 必须能同时复现图片字节与锚点。旧参数**不保留**为兼容别名——同一决策同时存在 CLI
参数与 config 字段就是双真相源。旧参数想要的行为今天写作 `convert = false`。

留档文件本身就是合法的 `--config`：`python -m ard --config outputs/<run>/config.toml
--override config.override.toml` 即可用同一份配置重跑，凭证由覆写文件提供。

## 输出说明

```text
outputs/<run_name>/          # 默认 ard_dataset_<YYYYmmdd_HHMMSS>；--smoke 追加 _smoke
├── anchor_bank.jsonl        # 每行一条记录，schema_version 4.0.0
├── config.toml              # 合并后的配置快照，凭证已脱敏
├── plan_identity.in_progress.json  # 仅运行未结束时存在：中途的计划身份记录
├── images/                  # 影像态锚点图片的落点：images/<visual_domain>/<图片文件>（转码或复制而来）；只有本次带影像态锚点时才创建
├── logs/                    # ard.log / ard_debug.log / ard_error.log
├── results/
│   ├── coverage.json        # 机器可读的验收读数
│   └── coverage.md          # 同一份读数的人读版
└── manifest.json            # 库构成、运行健康、配置、acceptance 指针（权威申报）
```

`manifest.json` 是**权威**申报：`status: "complete"`、库构成、运行健康计数器与 `acceptance` 指针。
`plan_identity.in_progress.json` 是中间态记录：计划固定、第一次端点调用之前落盘，运行正常结束时删除。
它带 `status: "in_progress"` 与本次的 `plan_identity`，**不含**运行健康（那时还不存在）。它让
**中断的运行也能被审计**——1,826 条那条路本来就会被反复中断/续跑，这份记录把目录绑定到产出它的计划。
它的 `counters` 在第一次端点调用前定稿、此后不再刷新，描述的是**计划**而不是进度：`existing` = 库里已
有多少，`new` 与 `written` = 本次计划生成、写出的锚点数（两者同值）。**不得**当作完成度、进度或验收申报
来读——真正已落库的条数见 `manifest.json` 的 `generation.counters.written`；其中的计划身份可以用
`PlanIdentity.of(plan)` 独立复算、逐位比对摘要。见 [docs/architecture.md](docs/architecture.md) 第 4 节。

`results/coverage.{json,md}` 是**验收读数，不是训练数据**：

- **结构读数**（计划计数 vs 构造规则）恒产出，且零成本——不调用任何模型；
- **指标读数**（`q95`、带 ε±5% 敏感带的 `Extent(ε)`、配对 bootstrap CI、噪声带）需要同时配置
  `coverage.target_set_path` 与 `[coverage.embedding]`。

两者缺一时，运行只写结构读数，并用 WARNING 明说缺口（`metric readout not measured …`），
manifest 里 `metric_readout: false`、`q95: null`——绝不产出一份"看起来干净"的报告。
尺子的定义与分辨力边界见 [docs/measurement.md](docs/measurement.md)。

指标空间**只含文本**：图像模态锚点以其最终 user 轮的**文本部分**参与，**图像像素不进该空间**，
所以读数把锚点字段声明为 `messages[last].content(text parts only)`。

### 自己复跑指标读数（三步）

指标读数由配置驱动，没有 CLI 开关：

1. 在 `.local/config.override.toml`（被 gitignore 的本机覆写文件）里配 `[coverage.embedding]`。
   下面只给**字段名与占位符**，绝不要把真实端点、模型名或密钥写进仓库：

   ```toml
   [coverage.embedding]
   api_base = "<OpenAI 兼容的 /embeddings 基址，含 /v1>"
   model = "<嵌入模型名>"
   dimension = <该模型的向量维度>
   # api_key = "<服务端需要时填>"
   # normalize = true      # 必须为 true：尺子要求 L2 归一化行
   ```
2. 用 `coverage.target_set_path` 指向目标集。仓库自带一个小而确定的样例
   `examples/target_set.sample.jsonl`（32 条；构造规则写在其文件头与
   [docs/measurement.md](docs/measurement.md)），开箱可用：

   ```toml
   [coverage]
   target_set_path = "examples/target_set.sample.jsonl"
   ```
3. 跑 `--smoke`（8 条锚点，几分钟）或完整运行。读数落在
   `<output_dir>/results/coverage.{json,md}`。

三种配置组合是**契约**，不是建议：**两者都未设** ⇒ 结构读数 + WARNING；**只设
`target_set_path`、`[coverage.embedding]` 不全** ⇒ 在 `load_config` 即被拒，报文列出缺失字段，
且早于创建任何输出目录；**两者都设** ⇒ 产出指标读数。

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

**RAW 相机文件为什么要装 `.[raw]`？**
RAW 解码走 `rawpy`，而它的 wheel 捆绑了 LibRaw 解码器（LGPL-2.1 / CDDL-1.0）。一次普通
`uv sync` 完全不装 LibRaw——`raw` extra 只新增 `rawpy` 这一个包——所以 RAW 解码器做成可选
extra，而不进默认依赖集：`uv sync --extra raw`（见上面的安装步骤）。默认依赖闭包并非清一色
MIT/Apache：`certifi` 是 MPL-2.0、`tqdm` 是 `MPL-2.0 AND MIT`、`typing-extensions` 是
PSF-2.0，其余为 MIT / BSD-3-Clause / MIT-CMU。完整清单与读取它的命令见
[`docs/licenses.md`](docs/licenses.md)；本仓库只陈述这些事实，合规判断留给你。默认安装遇到 RAW
输入不会崩——它打一条点名该文件的 WARNING（"`rawpy not installed, cannot convert RAW image:
…`"），丢掉这张图，其余运行照常继续。

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

# anchor-replay-distillation

[English](README.md) | [简体中文](README.zh-CN.md)

## 简介

**Anchor Replay Distillation（ARD，锚点回放蒸馏）把一套由本体定义的坐标空间变成可复现的
监督微调语料。** 对每一个锚点坐标，它先让问题生成模型产出用户轮，再让目标（教师）模型作答，
然后把教师的回答——以及（开启时的）推理过程——与产生它的那个问题一起保存下来。ARD 是蒸馏流水线
中**负责产数据**的一半：它不训练、不蒸馏、也不评测学生模型。

计划由 v4 本体（`ontology/anchor_ontology.v4.json`）按**唯一一条规则**构造——**随机顺序轮转
（cycle-shuffle）**；单元总数 `U` 在运行时从本体穷举得出，既不写进本体也不写进代码。规则本身与
`U` 的复算命令见 [docs/algorithm.md](docs/algorithm.md) §1–§2。

**一次运行产出多少条锚点由你决定**，入口只有一个——`configs/config.toml` 的
`[generation] count`。不设它就是一个完整轮（`U`），而且**没有上限**：更大的 `N` 只是滚进第 1 轮、
第 2 轮……详见 [如何选择锚点条数（N）](#如何选择锚点条数n)。

想先看一遍平实的全貌？读 [docs/walkthrough.md](docs/walkthrough.md)——按下回车之后，机器每一步做什么。

本文反复用到的几个词：

| 术语 | 含义 |
|---|---|
| 锚点（anchor） | 一份生成样本——一个坐标对应的问题与教师给出的回答；`anchor_bank.jsonl` 的一行 |
| 本体（ontology） | `ontology/anchor_ontology.v4.json`：各条轴、它们的取值，以及哪些组合合法 |
| 坐标（coordinate） | 标注一条锚点的取值元组：每条轴一个值，外加采样字段 `modality` |
| 单元 / 轮（unit / round） | 一个 `(模态, 合法受限块)` 对；一轮把每个单元恰好走一遍 |
| 计划（plan） | 一次运行要生成的坐标有序列表，在第一次端点调用之前就已固定 |
| N | 一次运行产出多少条锚点：`[generation] count` 字段；不设 = 一个完整轮 = `U` |
| manifest | `manifest.json`：一次运行的权威申报 |

## 快速开始

默认用 Docker 跑 ARD。全新 clone 的仓库，**两条命令**就能拿到一份真实产物。**ARD 没有已发布的
镜像**：第一次 `./run.sh` 会在本机用 `docker/Dockerfile` 构建 `ard:<最新 git 标签>`（只建一次，
几分钟）。构建需要能访问 PyPI，也需要基础镜像 `python:3.11.15-slim`：本机没有缓存时 Docker 会从
registry 拉取它——受限网络或离线机器需要先在本机备好这个基础镜像。

### 1. 准备端点凭证

```bash
mkdir -p .local && cp configs/config.override.sample.toml .local/config.override.toml
```

然后填写 `[input_generator]` 与 `[target_model]` 的 `api_base`、`model_name`、`api_key`。

`mkdir -p` 不是装饰：`.local/` 被 gitignore，全新 clone 里没有这个目录，直接 `cp` 会以
`No such file or directory` 失败。**如果 `.local/config.override.toml` 已存在，请直接编辑它，
不要覆盖复制**——照抄这一步会静默覆盖那里已经部署好的凭证。

覆写以固定三级优先级合并到 `configs/config.toml` 之上，先命中者胜：

1. 显式 `--override PATH`（文件不存在即报错）；
2. `<project_root>/.local/config.override.toml`——推荐的、被 gitignore 的位置；
3. `<--config 所在目录>/config.override.toml`——向后兼容。

这个选择**从不静默**：每次运行都会在 INFO 日志里写明用了哪个覆写文件，或者写明"只用基础配置"。
字段名按 `extra="forbid"` 校验，拼错的字段是报错，而不是被静默忽略。

**缺凭证时**：运行会在**创建任何输出目录之前**停下，给出字段级报文并以退出码 1 结束。
报文首行如下（后面还有修法示例）：

```
ERROR: [input_generator] is missing `api_base`, `model_name`.
```

### 2. 跑冒烟

```bash
./run.sh --config configs/config.toml --smoke --image-dir examples/images
```

第一次调用会构建镜像 `ard:<git describe --tags --abbrev=0>`（也就是当前标签；可用
`ARD_IMAGE=<名字>` 换名），随后以 `docker run --rm --network=host` 在容器里运行
`python -m ard`：`configs/`、`ontology/`、`examples/`、`.local/` 以只读方式挂载，`outputs/`
可写，所以产物直接落在你的工作树里。`--network=host` 让容器能访问跑在宿主机 `localhost` 上的端点；
在 Docker Desktop 上需要在设置里启用 host networking。

想先单独构建镜像、看清构建日志？

```bash
bash docker/build.sh
```

它构建同一个标签（`IMAGE_NAME=<名字>` 可换）就结束。找不到 docker 时，`run.sh` 会打印提示并以
127 退出；构建失败时会打印重新构建的命令。

**`--smoke` 的语义。** 它以**同一条 cycle-shuffle 规则**的缩小规模物化计划：
**8 条锚点——4 条文本态 + 4 条影像态**，取自第 0 轮洗牌序列（每个模态的前 4 个单元）。它的用意是让
新 clone 的仓库几分钟内就能看到一份完整形态的产物，而且它是一次**真实的端点运行**，不是离线演示。
这份产物刻意很小，并在三个互相独立的位置自我申报：运行目录名带 `_smoke` 后缀、日志打 WARNING
说明规模、`manifest.json` 里 `smoke: true` 外加一个 `smoke_plan` 块。`--smoke` 是 CLI 的
**运行边界参数**，不是配置字段，对不带它的运行不产生任何影响。

冒烟计划**不是全量计划的前缀**：全量计划取第 0 轮单一洗牌序的前 `N` 个单元，而冒烟取的是
*每个模态各自*的前 4 个单元——所以同一个 id 在两者中可能指向不同的坐标。也正因如此，冒烟目录与
全量目录**不得混用**。

产物落在 `outputs/<run_name>/`（默认 `ard_dataset_<YYYYmmdd_HHMMSS>_smoke`；用配置里的
`[output] directory` 可以钉住目录名）。几分钟后应当看到：

- `anchor_bank.jsonl`——8 条锚点，`schema_version` 为 `5.0.0`；
- `results/coverage.json` + `results/coverage.md`——零模型的结构读数（`report_schema
  "ard-acceptance-4"`）：计划计数对照构造规则，含 `within_rule` 与多样性计数。不调用模型、
  不访问端点、恒产出。报告绝不假装 8 条锚点就是一份完整轮的交付物；
- `manifest.json`——库构成，外加 `plan_identity`（v2）与 `plan`、`images` 两段；
- `config.toml` 与 `logs/`。运行尚未结束时还会有一份 `plan_identity.in_progress.json`：
  把目录绑定到计划的中间记录；运行完成后它会被删掉，目录里只留 `manifest.json` 作为申报。

**完整运行。** 一个完整轮是 `U` 条锚点（`U` 由运行时穷举；要别的 `N` 就设 `[generation] count`），
每一轮对话一次端点调用。请自备图片目录，按 `<image_dir>/<visual_domain>/<图片文件>` 布局：

```bash
./run.sh --config configs/config.toml --image-dir /path/to/images
```

仓库之外的目录会以只读方式挂到容器的 `/data/images`；仓库之内的目录（例如 `examples/images`）
则直接按相对路径使用。图片按轮转使用，某个域自己的目录里没有图时**复用**整棵图片树里的一张、
而不是丢掉锚点——所以**只给一张图，也能覆盖所有域、所有轮次**（复用了多少在 `manifest.json`
里计数并点名）。只有整棵树一张可用图都没有才会中止运行。完整运行要花掉数小时的端点时间；
请先用冒烟验证配置是否正确。

### 不用 Docker 的话（可选）

同一个入口也能在本地环境里跑。需要 [uv](https://docs.astral.sh/uv/) 与 Python 3.11
（由 `.python-version` 锁定；`uv.lock` 锁定了全部依赖版本）：

```bash
uv sync
uv run python -m ard --config configs/config.toml --smoke --image-dir examples/images
```

**默认安装不含 RAW 支持——镜像里也没有。** 官方镜像与普通 `uv sync` 都只装默认依赖闭包，其中不含
`rawpy`；因此 RAW 相机格式（`.cr2`、`.nef`、`.arw`、`.dng` 等）需要在本地安装时显式加上 `raw`
extra：

```bash
uv sync --extra raw
```

不装它时，RAW 输入会以一条点名该文件的 WARNING 被跳过，而不会让运行崩溃。为什么 RAW 做成可选，
以及完整的依赖许可证清单，见 [`CONTRIBUTING.md`](CONTRIBUTING.md#third-party-licences)。

## 如何选择锚点条数（N）

`N` 只有一处来源——`configs/config.toml` 的 `[generation] count`：

```toml
[generation]
# 省略（缺省）= 一个完整轮 = U，即运行时穷举出的单元数。
# 设任意 >= 1 的整数即产出恰好那么多条锚点；N 按轮滚动
# （走完一轮就进入下一轮）。
count = 5000
```

- **没有 CLI 参数、没有 `run.sh` 透传、没有环境变量。** `./run.sh --count 10` 会把 `--count 10`
  原样交给 `python -m ard`，后者以"无法识别的参数"拒绝。同一取值若同时有 CLI 参数与 config 字段，
  就是双真相源，所以这个值只存在于 config。
- **没有上限。** 大 `N` 不会被拒绝；运行会在日志里写明它跨了几个完整轮、末轮有多少条。
- **会被拒绝的只有**：`count < 1`，以及非整数类型（`0`、负数、`true`、`"1826"`、`1.5`）。
  报错会点名该字段，并提示"删掉这一行即可得到一个完整轮"。
- **在已有目录上调大 `N` 是追加。** 锚点 id 是计划位置序号、**不含 `N`**，所以用更大的 `count`
  重跑同一个 `output.directory` 会保留全部已有记录、只追加新记录。
- **覆盖与密度必须分开读。** *覆盖率*数坐标，一旦每个单元都被访问过就饱和于 `1.0`
  （`min(distinct, U) / U`）；*密度*数条数，随 `N` 持续增长（`N / U`）。在后续轮次再次抽到同一坐标是
  **合法的**新样本——生成器是随机的，同一组标签会问出不同的问题。定义见
  [docs/algorithm.md](docs/algorithm.md) §7。

## 数据格式

`anchor_bank.jsonl` 每行一个 JSON 对象：

```jsonc
{
  "id": "1a2b3c4d-c00000p00137",    // 计划位置序号：run_key + 轮次 + 轮内序号
  "source": "ard",
  "data_source": "ard_text",        // 受控词表：ard_text | ard_multi
  "schema_version": "5.0.0",
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

`id` 是**计划位置**，不是坐标指纹：`run_key` = `H(本体哈希, seed, 采样算法)[:8]`，`c00000` 是轮次，
`p00137` 是轮内序号。它**刻意不含 `N`**，这正是"调大 `count` 再跑同一目录"能干净追加的原因。
因此两条记录可以携带同一个 `anchor_meta` 却有不同的 id，两条都会保留。

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

图片取自 `<image_dir>/<visual_domain>/<图片文件>`——归档在锚点自己 `visual_domain` 坐标下的图，
才是用户为"内容与标签相符"背书的那张，所以永远优先用它。该目录里没有图时**不会丢掉锚点**：
运行会退回整棵图片树（`--image-dir` 下所有可用文件，确定性排序）并逐轮轮转。每一次复用都会留痕
——`manifest.json` 里的 `fallback_visual_domains` / `fallback_anchor_count`，以及
`resolved_images` 每行的 `fallback` 标记。路径相对于运行目录，所以
`outputs/<run_name>/images/<visual_domain>/…` 直接可解析；运行时会把选中的图复制到那里，
并（除非设 `[images] convert = false`）转码。这些记录的一份真实样例已入库在 `examples/` 下，见
[examples/README.md](examples/README.md)。

## 配置参考

全部字段都定义在 `configs/config.toml`（英文注释、字段齐全），覆写只负责填值。
**产物一律由 config 决定**——CLI 只传输入定位参数与运行边界参数：`--config`（必需）、
`--override`、`--image-dir`、`--smoke`。

| 段 | 控制什么 |
|---|---|
| `[input_generator]` | 问题生成端点（`api_base`、`model_name`、`api_key`）、采样温度、超时、重试 |
| `[target_model]` | 教师端点——它的回答就是监督目标。`enable_thinking` 打开推理通道（`targets[0].output.reasoning`） |
| `[generation]` | `count`（锚点条数 `N`：省略 = 一个完整轮 = `U`；任意 `>= 1` 的整数，**无上限**）、`concurrency`；可选 `seed`（省略则每次运行从系统随机源抽新种子，实际使用的种子记在本次运行的 `config.toml` 里）；背压阈值。对话轮数**不可配置**——它们由本体推导 |
| `[ontology]` | v4 本体路径（`ontology/anchor_ontology.v4.json`） |
| `[output]` | `directory`（留空 = `outputs/ard_dataset_<timestamp>`）、`overwrite`（默认 `false`：已有锚点库是续跑，不是替换） |
| `[images]` | `convert`（默认 **`true`**）：接受 RAW/BMP/TIFF/GIF/WebP 并把选中的图统一转成 JPEG（RAW 另需可选 `raw` extra）；设 `false` 则只接受已适合网络的格式并原样复制。`skip_missing_images`（默认 **`false`**）：某个域自己的目录里没有图时，会从整棵图片树复用一张；整棵树一张可用图都没有时，运行直接报错。设成 `true` 则改为丢弃这些锚点——每丢一条打一条 WARNING，条数与涉及的域在 `manifest.json` 里申报 |
| `[coverage]` | `enabled`（默认 `true`）：把零模型的结构读数写进 `results/`。设成 `false` 则整个相位跳过 |

`configs/config.override.sample.toml` 就是第 1 步复制的那个注释模板；部署值（端点、模型名、凭证）
放这里，绝不写进 `configs/config.toml` 本身。

### 图片转码是配置字段

`[images] convert` 决定 `outputs/<run>/images` 里落盘的内容：`true` 把选中的图统一转成 JPEG，
`false` 则把已适合网络的格式原样复制。这个决定**没有对应的 CLI 参数**——一份 config 必须描述
一份产物，留档的 `config.toml` 才能同时复现图片字节与锚点。

留档文件本身就是合法的 `--config`：`python -m ard --config outputs/<run>/config.toml
--override config.override.toml` 即可用同一份配置重跑，凭证由覆写文件提供。

## 输出说明

```text
outputs/<run_name>/          # 默认 ard_dataset_<YYYYmmdd_HHMMSS>；--smoke 追加 _smoke
├── anchor_bank.jsonl        # 每行一条记录，schema_version 5.0.0
├── config.toml              # 合并后的配置快照，凭证已脱敏
├── plan_identity.in_progress.json  # 仅运行未结束时存在：中途的计划身份记录
├── images/                  # 影像态锚点图片的落点：images/<visual_domain>/<图片文件>（转码或复制而来，同一源文件只落一份）；只有本次带影像态锚点时才创建
├── logs/                    # ard.log / ard_debug.log / ard_error.log
├── results/
│   ├── coverage.json        # 机器可读的结构读数（report_schema ard-acceptance-4）
│   └── coverage.md          # 同一份读数的人读版
└── manifest.json            # 库构成、运行健康、配置、plan_identity v2、plan 段、images 段、acceptance 指针（权威申报）
```

`manifest.json` 是**已完成运行**的权威申报：`status`、库构成、运行健康计数器，以及计划的身份与形状
（采样规则、本体哈希、`seed` 与 `count`、`U`、轮分解、计划条数与实际写入条数、覆盖率与密度）。
运行尚未结束时，改由 `plan_identity.in_progress.json` 把目录绑定到它的计划；manifest 写好后这份
记录会被删除，所以完成的目录里只有一份申报。

去哪里读真相源：磁盘上真实存在的记录在 `anchor_bank.jsonl`；申报读运行结束时的 `manifest.json`；
`--smoke` 产物会在目录名、日志与 `manifest.smoke` 三处自报身份。逐字段说明、计数器口径与续跑守卫见
[docs/architecture.md](docs/architecture.md) 第 4 节。

`results/coverage.{json,md}` 是**结构读数**，不是训练数据（`report_schema "ard-acceptance-4"`）：
计划计数对照本次运行自己的采样空间与磁盘上真实落库的记录——`coverage`、`density`、轮分解、
`within_rule` 与两个多样性计数。它**不调用模型、不访问端点**，只要 `[coverage] enabled = true` 就产出。
库里少了某条计划坐标时读数会如实写出来——`within_rule: false`，并在 `warnings` 里点名缺失坐标——
绝不产出一份看似干净的报告。定义见 [docs/algorithm.md](docs/algorithm.md) §5–§6。

## 添加一个知识领域

**加一个 `knowledge_domain` 叶，只需要往 `ontology/anchor_ontology.v4.json` 的
`knowledge_domain_tree` 里添一项，别的什么都不用动。** 没有任何手写计数要更新、没有常量要改：覆盖单元、
叶轮转与验收期望全部在运行时从本体穷举得出，所以下一个叶在下一次运行就被自动接住，不用碰 `configs/`、
`src/` 或代码。（加叶会改变本体哈希，因此它是一份**不同的计划**——改动前产出的目录不能续跑；请换新的
`output.directory`。）

## 开发指南

```bash
uv sync --extra dev     # 测试、类型检查、lint 工具
uv run pytest
uv run ruff check src/ tests/
uv run mypy src/ard/
```

可选 extra 都声明在 `pyproject.toml` 里：`dev`（上面的工具）与 `raw`（RAW 相机解码）。
包版本由 `setuptools_scm` 从 git 标签推导；不带 `.git` 的源码归档
（GitHub 的 "Download ZIP"、`git archive`）回退到 `1.0.0`——与 `run.sh`、`docker/build.sh`
在没有可描述标签时使用的值一致。

同一套流程也写在 [CONTRIBUTING.md](CONTRIBUTING.md) 里。

**`--smoke` 与完整运行的区别**：构造规则、采样与逐条生成路径完全相同，只有计划规模不同
（8 条 vs 一个完整轮）。冒烟产物刻意不完整、不得当作数据集交付——`_smoke` 后缀、日志 WARNING 与
`manifest.json: smoke` 三处申报就是为此。两条路径都会调用已配置的端点；它们都不是离线演示。

## 常见问题（FAQ）

**我把 N 设在别处了，为什么没生效？**
`N` 只有一处来源：config 里的 `[generation] count` 字段。没有 `--count` 参数、没有 `run.sh` 选项、
也没有环境变量——`./run.sh --count 10` 会被交给 `python -m ard`，后者以"无法识别的参数"拒绝。
见 [如何选择锚点条数（N）](#如何选择锚点条数n)。

**续跑时我的输出目录被拒绝了。**
续跑守卫会逐条比对已有记录与它当前计划位置上应有的坐标，所以计划变过的运行——换了 `seed`、改了本体、
或在 `--smoke` 与完整运行之间切换——会在写入任何东西之前被拒绝。在同一目录上**调大 `count`** 会追加
新锚点；否则请换新的 `output.directory`，或设 `[output] overwrite = true` 清空已有锚点库重新开始。

**端点缺失或连不上怎么办？**
缺 `api_base` / `model_name` 会在 config 加载阶段被拒，报文点名缺失字段，且早于创建任何输出目录。端点连不上或调用失败会按
`max_retries`、`retry_on_timeout` 重试，连续失败达到背压阈值后按配置的冷却时间暂停。

**缺图会怎样？**
图片是复用而非跳过。只要 `<image_dir>/<visual_domain>/` 里有图，就用这个域自己的；没有，就退回
整棵图片树取一张并逐轮轮转——所以**你只给一张图，所有域、所有轮次都复用这一张**，运行照样完整，
重复对数据集影响有多大由你判断。每次复用都看得见：`manifest.json.images` 里有
`fallback_visual_domains`、`fallback_anchor_count`、`pool_candidate_count`，以及
`resolved_images` 每行的 `fallback` 标记，日志里也逐条记录。

只有整棵树**一张可用图都没有**时才会被拒绝，且发生在**创建输出目录之前**：报错逐项列出没有图的
视觉域。可以至少补一张图，也可以设 `[images] skip_missing_images = true`：那些锚点会被丢掉、
逐条打 WARNING，跳过条数与涉及的域在 `manifest.json` 里申报——跳过永不静默。完全不传
`--image-dir` 时，运行根本不会附图：影像态锚点仍会被生成成纯文本对话，同时保留自己的
`visual_domain` 坐标。

**RAW 相机文件为什么要装 `raw` extra——Docker 里有吗？**
RAW 解码走 `rawpy`，而它的 wheel 捆绑了 LibRaw 解码器（LGPL-2.1 / CDDL-1.0），所以做成可选
extra，而不进默认依赖集：`uv sync --extra raw`（见 [不用 Docker 的话（可选）](#不用-docker-的话可选)）。
官方 Docker 镜像只装默认依赖闭包，因此其中**同样没有** `rawpy`——在 Docker 下，RAW 输入会以一条点名
该文件的 WARNING 被跳过，其余运行照常继续。默认依赖闭包并非清一色 MIT/Apache：`certifi` 是
MPL-2.0、`tqdm` 是 `MPL-2.0 AND MIT`、`typing-extensions` 是 PSF-2.0，其余为
MIT / BSD-3-Clause / MIT-CMU。完整清单与读取它的命令见 [`CONTRIBUTING.md`](CONTRIBUTING.md#third-party-licences)；
本仓库只陈述这些事实，合规判断留给你。

**`MULTI_TURN_DEFAULT = 4` 是什么口径？**
本体对两个 `conversation_type` 只写了 `turns: "multi"`，没有数值上界；实现把"multi"读作**本体
自己声明过的最大轮数**（`constraint_update` = 4），并在本体一旦声明更大轮数时直接报错——绝不静默
截断。轮数不是配置项。见 [docs/algorithm.md](docs/algorithm.md) §4。

## 许可证与贡献

MIT——见 [LICENSE](LICENSE)。欢迎贡献：开发环境、质量命令与提交约定见
[CONTRIBUTING.md](CONTRIBUTING.md)。

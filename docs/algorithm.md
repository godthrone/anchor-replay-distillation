# ARD 算法：目标集口径与构造规则

> 职责：说明 ARD 一轮锚点计划的**目标集口径**、**构造规则**（坐标如何被选出）、**轮数口径**、**坐标与措辞的边界**，
> 以及本体指纹沿革。纯计算实现见 `src/ard/core/sampling.py` / `constraints.py` / `ontology.py`；判据与读数定义见 `docs/measurement.md`。
> 基线：本页所有 `文件:行` 已按提交 `a94e7b3` 的树逐条核对；其中指向 `src/ard/pipeline.py` 的引用
> 已按提交 `70af65e`（S11c resume 修复系列）逐条按符号内容重定位，指向 `src/ard/core/sampling.py`
> 的一处引用同期核正（原起点落在空行）。其余引用由 `tests/test_doc_line_references.py` 守卫检查
> "文件存在 / 行号在范围内 / 引用行非空"——它**不能**发现错位到另一条非空行上的情况。

## 1. 目标集口径

计划来自本体 v4（`ontology/anchor_ontology.v4.json`），它声明 **12 轴** = 11 个通用轴 + 1 个模态条件轴
（`ontology/anchor_ontology.v4.json:6-25`）。轴分两类：

- **6 个受限轴**（`capability` / `system_prompt_mode` / `conversation_type` / `output_format` / `input_condition` / `answer_mode`）——受约束 `allowed_pairs`（R1–R4b）与模态门 R5 限制，合法组合是**有限集合**；
- **5 个自由轴**（`language` / `knowledge_domain` / `response_style` / `difficulty` / `context_length`）——彼此正交（R7），组合数即自由轴积；
- **1 个条件轴** `visual_domain`——仅当采样字段 `modality == image` 时取值，文本态缺席（`:1222-1229`、`:1303-1309`）。

| 计数 | 值 | 出处 |
|---|---:|---|
| 自由轴积 `free_axis_product` | **52,668** | `ontology/anchor_ontology.v4.json:1340`（4 × 209 × 7 × 3 × 3） |
| 受限轴原始组合 `raw_restricted_block` | 100,800 | `:1341` |
| 文本态合法受限块 | **935** | `:1342`；`src/ard/core/sampling.py:88` |
| 影像态合法受限块（18 个 image-capable capability） | **891** | `:1343`；`src/ard/core/sampling.py:91` |
| `knowledge_domain` 叶 | **209**（18 domain / 36 subdomain） | `:63-67` |
| `visual_domain` 叶 | **21** | `:307-311` |
| **一轮计划总数** | **1,826**（935 + 891） | `src/ard/core/sampling.py:100` |

**总数不是配置项**：它由构造规则与本体唯一推导。`configs/config.toml` 中没有 `target_count` 一类字段，
`sampling.sample_anchors` 的返回长度就是计划长度；本体计数一旦与规则不符，采样器**报错退出**而不是产出更短/更长的计划
（`src/ard/core/sampling.py:255-277`）。

## 2. 构造规则

规则两态相同，差别只在 `capability` 的取值域与条件轴是否参与：

```mermaid
flowchart TD
    A["本体 v4"] --> B["ConstraintEvaluator<br/>枚举合法受限块"]
    B --> C["文本态：935 块<br/>（20 个 capability）"]
    B --> D["影像态：891 块<br/>（18 个 image-capable capability）"]
    C --> E["每块 ≥ 1 条<br/>knowledge_domain 209 叶轮转"]
    D --> F["每块 ≥ 1 条<br/>knowledge_domain 209 叶轮转<br/>visual_domain 21 叶轮转"]
    E --> G["language / response_style /<br/>difficulty / context_length 按 seed 随机"]
    F --> G
    G --> H["1,826 条坐标<br/>（0 重复，重复即报错）"]
```

逐轴取值方式：

| 轴 | 取值方式 | 证据 |
|---|---|---|
| 6 个受限轴（`capability`/`system_prompt_mode`/`conversation_type`/`output_format`/`input_condition`/`answer_mode`） | **穷举合法受限块，每块恰好 1 条** | `src/ard/core/sampling.py:373-453`；`src/ard/core/constraints.py:212-264` |
| `knowledge_domain` | **轮转**：第 i 条取 `leaves[i % 209]` | `src/ard/core/sampling.py:79`、`:420-422` |
| `visual_domain` | **轮转**：第 i 条取 `leaves[i % 21]`（仅影像态；文本态缺席） | `src/ard/core/sampling.py:79`、`:420-422` |
| `language` / `response_style` / `difficulty` / `context_length` | **按 run seed 随机抽取** | `src/ard/core/sampling.py:81-86`、`:418-419`；`_draw` `src/ard/core/sampling.py:326-329` |
| `modality` | 采样字段（非轴）：前 935 条 `text_only`，后 891 条 `image` | `src/ard/core/sampling.py:429-449`；本体 `ontology/anchor_ontology.v4.json:1303-1309` |

已知的两个失败模式都做成**显式报错**而非静默降级：坐标重复（`src/ard/core/sampling.py:361-370`）、本体叶数与期望不符
（`src/ard/core/sampling.py:255-277`，报文给出 `expected (received: …)`）。

**"每块恰好 1 条"的作用域是模态组内**：891 个影像态合法块是 935 个文本态合法块的子集，因此这 891 个
受限坐标各出现**两次**——文本态组一次、影像态组一次——二者靠采样字段 `modality` 区分，而 `modality` 正是
查重身份（`identity()` → `as_dict()`）的一部分，故不构成重复。断言口径应为"每个模态组内每块一条"，
而不是"935 个互异块各一条"。

### 已不再使用的算法

本口径**不使用**：FPS / 最远点采样 / 贪心选点 / 覆盖选点 / 叶子权重 / 候选池 / 最大接近采样 / 三维空间 / 组合云。
对应旧模块（`core/_fps.py`、`core/cloud.py`、`core/embeddings.py`、`core/sampler.py`）与旧配置字段
（`criterion` / `embeddings_path` / `target_count` / `task_types`）已从 `src/` 与 `configs/` 删除。
可复核：`git grep -niE "farthest|_fps|criterion|proximity" -- src/` 在 `src/` 内 0 命中
（残留的 `image_pool`/`ThreadPoolExecutor` 是图片池与并发设施，与选点无关）。

## 3. 轮数口径

**两个"轮数"不是一回事，必须分开读：**

- 本体 `conversation_type.value_attributes.turns` 数的是**交换**（exchange = 一个用户问题 + 它得到的回答）；
- `AnchorSpec.messages`（生成产物中的消息数组）数的是**消息**，其长度 = `2n − 1`：`user` 开头、`user` 结尾、角色交替，
  因为**最后一轮必须是 user**——它的回答才是训练目标，不作为 spec 轮存在（`src/ard/core/types.py:62-75`）。

例：`single_turn`(1) → 1 条消息；`clarification`(2) → 3 条；`constraint_update`(4) → 7 条。
映射实现见 `src/ard/core/sampling.py:456-487`（`_spec_turns`）与 `:490-520`（`turn_counts_by_conversation_type`）。

**`MULTI_TURN_DEFAULT = 4` 的取值依据与本体缺口**（`src/ard/core/sampling.py:163-176`）：

- 依据：本体对 `tool_assisted` 与 `source_review` 只写 `turns: "multi"`，**没有数值上界**；常量取本体自身声明的最大轮数
  `constraint_update = 4`，即"`multi` = 本体已声明的最长交换数"。
- 缺口：这是**本体缺口，不是设计选择**——本体没有表达"multi 的上界"。代码把该数字收在单一常量处，
  并设硬门：本体一旦声明比它更大的轮数，`_spec_turns` 直接报错（`:481-486`），不静默采用。
- 轮数**不是配置项**：`configs/config.toml` 无对应字段，改轮数只能改本体（或该常量）。

## 4. 坐标与措辞

本体自述 **"Coordinates only. No prompt wording, no sampling/weighting/experiment policy."**
（`ontology/anchor_ontology.v4.json:5`）。`wording_policy`（`:1312-1324`）自述本体只持坐标、不得含 prompt 模板
（`:1314-1318`），并把措辞位置写成 `configs/prompts/system_prompt/<system_prompt_mode>.md`（`:1319-1323`）。

**该位置的效力**：`prompt_wording_location_recommendation.status` 逐字为
"recommendation only; no such file was created in this round (ontology-only round)"，即**推荐，非硬性要求**；
有约束力的是"坐标与措辞分离"本身。本工程按该推荐位置把措辞落地为数据文件（提交 `c8ffd44`），以下为**已实施的事实**。

**数据位置**：`configs/prompts/system_prompt/` 下 5 个文件，文件名 = 本体轴 `system_prompt_mode` 的取值：
`none.md`、`minimal_persona.md`、`detailed_persona.md`、`task_constraint.md`、`domain_style.md`。

**文件格式**（唯一格式定义，另见 `configs/prompts/README.md`）：

- 文件名恰为 `<system_prompt_mode>.md`，不读任何其他文件名；
- 正文 = UTF-8 Markdown，无 front matter、无注释语法；**整份内容去掉首尾空白即模板**，不做任何解释；
- 占位符**可选用且仅允许** `{language}` / `{capability}` / `{domain}`，按 `str.format` 从 `anchor_meta` 取
  `language` / `knowledge_domain` / `capability`（默认 `English` / `general` / `qa`）；其他占位符或不配对花括号 = 加载错误；
- `none.md` 是唯一例外：`none` 是"无 system message"的缺省态，没有生成措辞，该文件只**陈述**这一事实、
  **永不被渲染**（`build_system_prompt_prompt` 对 `none` 显式报错）；每个文件（含它）都必须非空。

**加载与报错语义**：措辞的契约与渲染是纯计算，落在 `src/ard/core/system_prompt.py`（目录常量 `:64`；错误类型、模板路径校验、模板校验、渲染 `:95-226`），**不含任何内置措辞串**；
**读文件**是设施动作，落在 `src/ard/backends/prompt_loader.py:38`（`load_system_prompt_template`）与组装入口 `:75`
（`build_system_prompt_prompt`）——`core/` 内零文件访问（§1.3，由 `tests/core/test_core_is_pure.py` 守卫）。目录缺失、mode 无对应文件、
文件为空、占位符非法**一律硬报错**，报文含**路径与期望**（§2.3），**不静默回退到硬编码**（那等于重建第二个真相源，§1.4）；
报文只含路径与 mode 名，无机密（§15）。`system_prompt_mode = none` 时 `_generate_system_message` 直接返回 `None`，
不发起任何请求（`src/ard/domain/text_anchor.py:496`）。

**模板目录来源单点**：`SYSTEM_PROMPT_TEMPLATE_DIR`（`src/ard/core/system_prompt.py:64`）是运行时**唯一**一处
声明该目录的地方（`git grep -n "configs/prompts" -- src` 仅此 1 命中）；契约测试
`test_template_dir_is_the_ontology_declared_location` 把它与本体声明的 `target`（`<system_prompt_mode>` 替换后）
逐字对齐，任一侧漂移即在 CI 报错。**未新增 config 字段**：生成路径拿不到 config 对象，加字段就没有消费者（§7.2）。

**行为不变**：5 个 mode 的最终渲染 prompt 与迁数据文件前**逐字节相同**，由契约测试
`test_generation_prompt_is_byte_identical_to_the_old_in_code_wording` 以 sha256 固定
（`tests/core/test_system_prompt.py`）。

**`get_system_prompt_values` 与 `SYSTEM_PROMPT_GENERATION_INSTRUCTIONS` 已具名删除**（§18.1 不留死代码）：
前者只是 `ontology.axis_values("system_prompt_mode")` 的 1:1 包装、生产路径 0 消费者；后者是被数据文件取代的
内置措辞表。测试改用本体访问器（`tests/core/test_system_prompt.py`）。

- **有措辞的轴**（轴的取值会进入发给模型的 prompt 文本）：

| 轴 | 措辞来源 | 证据 |
|---|---|---|
| `language` | 内联模板串（user 侧）+ 数据文件占位符 `{language}`（system-prompt 侧） | `src/ard/domain/text_anchor.py:185`、`:272`、`:280`；`configs/prompts/system_prompt/*.md` |
| `knowledge_domain` | 内联模板串 + 数据文件占位符 `{domain}` | `src/ard/domain/text_anchor.py:186`、`:281`；`configs/prompts/system_prompt/*.md` |
| `capability` | 内联模板串 + 数据文件占位符 `{capability}` | `src/ard/domain/text_anchor.py:187`、`:274`、`:282`；`configs/prompts/system_prompt/*.md` |
| `conversation_type` | 内联模板串（作为 "conversation style" 回显；其 `turns` 另决定消息条数） | `src/ard/domain/text_anchor.py:188`、`:275`、`:283` |
| `system_prompt_mode` | **数据文件** `configs/prompts/system_prompt/<mode>.md`（5 个取值各一份；present 模式 = 完整生成 prompt 模板） | `configs/prompts/system_prompt/`、`configs/prompts/README.md:1`、`src/ard/core/system_prompt.py:64`、`:95-226`；消费点 `src/ard/domain/text_anchor.py:511`（`none` 不生成 system message） |
| `response_style` | **数据文件** `configs/prompts/axis_instruction/response_style.json`（7 取值各一句） | 同上；组装 `src/ard/domain/text_anchor.py:242`、`:289` |
| `output_format` | **数据文件** `configs/prompts/axis_instruction/output_format.json`（6 取值各一句） | 同上；组装 `src/ard/domain/text_anchor.py:242`、`:289` |
| `difficulty` | **数据文件** `configs/prompts/axis_instruction/difficulty.json`（3 取值各一句） | 同上；组装 `src/ard/domain/text_anchor.py:242`、`:289` |
| `context_length` | **数据文件** `configs/prompts/axis_instruction/context_length.json`（3 取值各一句） | 同上；组装 `src/ard/domain/text_anchor.py:242`、`:289` |
| `input_condition` | **数据文件** `configs/prompts/axis_instruction/input_condition.json`（6 取值各一句） | 同上；组装 `src/ard/domain/text_anchor.py:242`、`:289` |
| `answer_mode` | **数据文件** `configs/prompts/axis_instruction/answer_mode.json`（4 取值各一句） | 同上；组装 `src/ard/domain/text_anchor.py:242`、`:289` |

**6 条 instruction 轴的措辞数据**（WP-S17 新增；"wording is data" 的第二个数据族，与 `system_prompt/` 平级，
格式另见 `configs/prompts/README.md`）：`configs/prompts/axis_instruction/<axis>.json` 下 6 个文件，
键 = 轴的取值，值 = 发给输入生成器的一句要求。纯契约在 `src/ard/core/axis_instruction.py`
（`INSTRUCTION_AXES`、`AXIS_INSTRUCTION_DIR`、`validate_axis_instructions`、`render_axis_instructions`），
读文件在 `src/ard/backends/axis_instruction_loader.py`（`load_axis_instructions` / `build_axis_requirements`）。
取值缺指令 = **硬报错并点名轴、取值与文件**，无内置回退（§1.4、§2.3）；坐标不携带该轴（`None`）时该轴不加文本。
逐值来源（`[本体有据]` / `[我方拟定]`）与渲染后片段见 WP-S17 的 `AXIS-WORDING-MAP.md`。

**落点语义**：这 6 条轴约束的是**生成出的用户消息**，不是别的通道——`input_condition` 决定消息本身带何种缺陷，
其余 5 条要求用户在提问里**索要**该风格 / 格式 / 难度 / 长度 / 回答方式。`context_length` 按
**用户消息的长度**落地（本体 R7 逐字称该轴为 "input length"），而不是答案长度：锚点这一侧生成的就是用户消息；
若改按答案长度落地，需要目标模型的 system 通道，会与 `system_prompt_mode` 的语义冲突。

> 注：`text_anchor.py` 的 import 段历经增删（最近一次是 WP-S17 加一行 axis-wording import），其后的行号整体上移；
> 本页行号已按 WP-S17 提交逐条重核，无需再自行换算。

- **仅坐标的轴**（取值进入 `anchor_meta` / id / 统计，但**不改变 prompt 的文本措辞**）：

| 轴 | 取值仍被谁使用（非措辞） | 证据 |
|---|---|---|
| `visual_domain` | **决定影像态图片目录** `<image_dir>/<visual_domain>/`；进 manifest 分组标签 | `src/ard/domain/image_store.py:125-211`；调用点 `src/ard/pipeline.py:1242-1301`；分组标签 `src/ard/domain/bank.py:453` |

> 上表在 WP-S14 审计时还有 6 行（`response_style` / `difficulty` / `context_length` / `output_format` /
> `input_condition` / `answer_mode`）：当时它们的取值只进采样坐标与约束求解，**一个字符都没进 prompt**。WP-S17 已为这 6 条
> `layer = "instruction"` 的轴补齐措辞，故它们上移到"有措辞的轴"表。它们原先的非措辞用途不变：
> 采样坐标（`src/ard/core/sampling.py:352-356`）与受限块合法性判定（`src/ard/core/constraints.py:34-36`、`:53-55`、`:228-237`）。

**边界声明**：12 轴中 **11 轴的取值改变发给模型的 prompt 文本**（4 条 base 轴 + `system_prompt_mode` + 本轮补齐的 6 条
instruction 轴）；唯一例外是 `visual_domain`——它不进文本，而是经图片目录影响**输入图像的内容**，属内容而非措辞
（WP-S14 已实测其生效）。**不允许"吉祥物轴"**：守卫测试 `tests/domain/test_axis_influence_guard.py` 对 12 轴逐轴取两个
只差该轴的坐标，断言生成侧请求四元组 `(I, S, T, V)` 不同；负向对照在临时去掉 6 条 instruction 轴的措辞时**恰好点名这 6 条**。

**待决清单沿革**：曾单列"需上级决策"的**措辞数据位置**项——**已决**：按本体推荐落地为数据文件（提交 `c8ffd44`）。
其后单列的 `output_format` / `input_condition` / `answer_mode` 语义缺口（以及同类的 3 条自由轴）——**已决**（WP-S17，
数据位置见上面的 instruction 轴表；用户裁定原话："这就是 bug……确保所有的轴都有提示词应用，能切实的影响生成"）。

### 4.1 `layer` 词表（本体无定义，本工程定义）

本体（`ontology/anchor_ontology.v4.json`）给每条轴一个 `layer` 字段，取值 `content` / `form` / `instruction` /
`modality`，但**全文无 legend、0 代码消费者**（`git grep -n "\.layer" -- src` 0 命中），即该词表此前
**只有名字没有定义**（WP-S14 审计 §1、§4.2）。此处按四值对 12 轴的实际划分**给出定义**——**这是我方拟定的定义，不是本体的声明**
（本体与本体指纹均未改动）：

| `layer` | 定义（我方拟定） | 该层的轴 |
|---|---|---|
| `content` | 取值决定**内容主题与语言** | `language`、`knowledge_domain`、`capability` |
| `form` | 取值决定**对话形态**（system message 的存在与风格、轮数） | `system_prompt_mode`、`conversation_type` |
| `instruction` | 取值是**对生成内容的要求/指令**，必须作为措辞进入 prompt | `response_style`、`output_format`、`difficulty`、`context_length`、`input_condition`、`answer_mode` |
| `modality` | 取值是**模态条件轴**，经输入通道（图片）生效 | `visual_domain`（R5 逐字称其为 "modality-conditional axis"） |

定义与代码的一致性由测试锁定：`instruction` 层 = `src/ard/core/axis_instruction.py` 的 `INSTRUCTION_AXES`
（契约测试 `test_instruction_axes_are_the_ontology_instruction_layer`），且 12 轴逐轴守卫见
`tests/domain/test_axis_influence_guard.py`。

## 5. 本体指纹沿革

本体 v4 在 v3 清账后有一行 provenance 文本改动，指纹因此变化：

| 项 | 值 |
|---|---|
| 旧 md5 | `8100af028ce1ada5dcf6da02f3b47026` |
| 新 md5 | `3c24b00927a1fe21d287d18df3af82b7`（复核：`md5sum ontology/anchor_ontology.v4.json`） |
| 改动 | **仅 `supersedes` 一行**：`ontology/anchor_ontology.v4.json:4` |

**为何 md5 变了但既有读数不作废**：`supersedes` 是 `OntologyV4` 的一个普通 `str` 字段，不参与任何解析、约束求值或采样规则；
v3 本体文件被删除后，原文 "left untouched" 已成假话，故改。语义不变性可逐项复核：

- 该字段是 `OntologyV4` 的普通 `str` 字段，**不被任何解析/约束/采样代码读取**（`git grep "supersedes" -- src tests` 0 命中）；
- **四个计数**在改动前后逐项相同，且与本体自述一致（`ontology/anchor_ontology.v4.json:1340-1343`：`52668 / 100800 / 935 / 891`）；
- **55 组无解对**（`reachability.zero_solution_pairs`，`ontology/anchor_ontology.v4.json:1356`）未受影响——它由约束求值决定；
- 采样计划只依赖本体坐标与 seed，**不含 provenance 字段**；同 seed 的 1,826 行计划逐字节复现由契约测试锁定
  （`tests/core/test_sampling.py`，`git grep -n "def test_" -- tests/core/test_sampling.py`）。

⇒ 该改动是 provenance 文本修订，**不构成口径变更**：既有 `q95` 读数、计数与计划都不需要重算。

## 6. 复现与"计划身份"

**计划由 `(本体, seed, 代码)` 唯一决定。** 采样只用显式的 `random.Random(seed)`（`src/ard/core/sampling.py:418`、`_draw` `:326-328`），
自由轴取值来自本体的有序数组（`ontology/anchor_ontology.v4.json`），受限块按固定嵌套顺序枚举
（`src/ard/core/constraints.py:212-241`）——没有未播种随机源，没有 `set`/`dict` 迭代顺序依赖。
实测（S22）：同一 seed 的 1,826 条计划在 `PYTHONHASHSEED` 取 `0/1/42/12345/random/未设` 的 7 个进程里摘要逐字节相同。

- 采样由 `generation.seed` 决定；不设该字段则每次运行从系统随机源抽新 seed，实际使用的 seed 记入输出目录的 `config.toml`（`configs/config.toml:53-56`）。
- 同一 `(本体, seed)` 必然得到同一计划；构造规则本身无随机性，自由轴之外的取值完全确定。

**但 `seed` 不是计划的名字。** `config.toml`/`manifest.json` 的 `config` 段**每次运行都被覆盖**（含什么都没生成的空转续跑），
因此它记录的 seed 描述的是**最后一次调用**，不一定是产出 `anchor_bank.jsonl` 的那次计划——S22 定位到的正是这个
"记录身份错位"：一个历史冒烟产物目录被调用 3 次、后两次空转，用记录 seed `1488279264` 复算该库自由轴仅 11/32 相同
（同 seed 重算在本仓库是确定的，所以差异只可能来自"记录的不是采样那次"）。

**计划身份是一等公民**：`PlanIdentity.of(plan)`（`src/ard/core/sampling.py:677-726`）对**有序坐标列表**做
`sha256`，连同**计划条数**、**算法**与**版本**（`PLAN_IDENTITY_VERSION = 1`）写进每次运行的 `manifest.json` `plan_identity` 字段。
下面是本仓库真实可复算的一例（`--smoke` 计划、`seed = 1488279264`、8 条）：

```json
"plan_identity": {"algorithm": "sha256", "version": 1, "plan_size": 8,
                  "digest": "25dae0cd0112fa3a5bccde01f8fd279e0c6be6e53b2187acad23ca353ccc4b1a"}
```

**复算**（任意进程、任意 `PYTHONHASHSEED` 得到同一摘要；下面是上面那一例，1,826 条全量把 `scale` 去掉即可）：

```bash
uv run python -c "from ard.backends.ontology_loader import load_ontology_v4; \
from ard.core.sampling import PlanIdentity, SMOKE_SCALE, sample_anchors; \
from ard.core.types import AnchorGenerationConfig; \
o = load_ontology_v4('ontology/anchor_ontology.v4.json'); \
print(PlanIdentity.of(sample_anchors(o, AnchorGenerationConfig(seed=1488279264), scale=SMOKE_SCALE)).as_dict())"
```

**续跑判据**（`src/ard/pipeline.py:625-703`）：记录的 `plan_identity` 与本次不同 ⇒ 报错，绝不把两个计划写进同一份
`anchor_bank.jsonl`；相同 ⇒ 允许续跑。同一计划的判据是摘要，不是 seed——**不同 seed 可以是同一计划，同一 seed 也可以是
不同计划**（换本体/换 scale）。若产物没有记录 `plan_identity`（旧库），守卫退化为结构校验：库里出现不属于本次计划的
anchor id 即拒绝，全部属于则允许并记 WARNING。


# ARD 验收尺子：口径与可分辨性边界

> 职责：定义 ARD 的验收尺子——距离、分位、覆盖读数、ε 敏感带、配对 bootstrap、噪声带、空间声明要求与退化行为，
> 并写明该尺子的**可分辨性边界**。纯计算实现见 `src/ard/core/coverage.py`；读数组装见 `src/ard/core/acceptance.py`
> 与 `src/ard/backends/coverage_wiring.py`；产物见 `results/coverage.json` / `coverage.md`。
> 基线：`src/ard/core/acceptance.py` 的行号已按 **WP-S18** 的树重新逐条核对（S18 新增
> `prompt_signature_distinct` / `effective_projection_distinct` / `diversity` 并把噪声带分组改为
> **prompt 签名**，行号随之整体平移）；更早一次核对在提交 `16ee1fd`（修正多模态锚点取文）；
> 本页指向 `src/ard/pipeline.py` 的引用沿用以提交 `70af65e`（S11c resume 修复系列）为准的核对结果；
> 其余被引用的代码文件沿用以 `a94e7b3` 为准的核对结果。
> `tests/test_doc_line_references.py` 守卫检查"文件存在 / 行号在范围内 / 引用行非空"，但它**不能**
> 发现引用错位到另一条非空行上的情况。

## 1. 距离与分位

- **距离**：`d(x) = min_{a ∈ A} (1 − cos(x, a))`，逐目标点取其到**最近锚点**的距离。两侧向量必须 **L2 归一化**，
  这样点积即余弦、`1 − dot` 即距离；结果夹到 `[0, 2]` 以免浮点噪声产生负距离。
  实现：`src/ard/core/coverage.py:338-377`（`nearest_anchor_distances`）；归一化门 `src/ard/core/coverage.py:292-324`（容差 `1e-6`，常量 `:43`）。
- **分位**：**Hyndman–Fan type-7**（`numpy.percentile(method="linear")`，`src/ard/core/coverage.py:33`），全项目统一口径。
- **主尺子 `q95`**：最近锚点距离的 **type-7 95 分位**（`src/ard/core/coverage.py:380-397`）。并报 `q50` / `q90` / `r_max`（最大距离），
  以及样本量 `|T|`（目标点个数）。
- `Extent(ε) = mean(d ≤ ε)`：落在半径 ε 内的目标点比例，**仅作参考**，从不作为判定尺子（`src/ard/core/coverage.py:399-412`）。

## 2. ε 敏感带

`Extent` 对 ε 的选择敏感，故并列三点（`src/ard/core/coverage.py:416-435`，乘子常量 `src/ard/core/coverage.py:37`）：

| 字段 | 含义 |
|---|---|
| `extent_at_095` | `Extent(0.95 × ε)` |
| `extent_at_100` | `Extent(1.00 × ε)` = 报告的主覆盖读数 |
| `extent_at_105` | `Extent(1.05 × ε)` |

ε 的来源被显式落在空间声明里（`src/ard/backends/coverage_wiring.py:180-191`）：

1. 目标集文件头部声明了 `epsilon` ⇒ 原样使用；
2. 未声明 ⇒ 用目标集**自身尺度**：目标点之间最近邻距离的 type-7 中位数（`src/ard/core/acceptance.py:705-726`，`intrinsic_epsilon`）；
   目标点少于 2 个时**报错**，不猜测。

## 3. 配对 bootstrap

用于比较两臂的 `q95` 差（`src/ard/core/coverage.py:471-548`）：

- **单位 = 目标点**：每次重采样抽的是目标点下标；
- **两臂共用同一次索引抽取**（同一 `positions` 同时索引两臂）——这是"配对"的定义，也是它与独立重采样的区别；
- **B = 2000**（`DEFAULT_BOOTSTRAP_RESAMPLES`，`:39`），置信水平 0.95（`:41`）；
- 点估计 = `q95(A) − q95(B)`（原始样本上的 type-7 分位差）；CI = bootstrap 差值的 type-7 2.5% / 97.5% 分位；
- 两臂长度不一致、或 `unit_id` 不同的目标集 ⇒ `PairingMismatchError`，不静默产出。

## 4. 噪声带

同格重复生成（**同一请求**被生成多次）的距离分布，下沿 `q50`、上沿 `max`（`src/ard/core/coverage.py:578-595`）。
原料是同一格的**互异对**距离（`pairwise_distances`，`src/ard/core/coverage.py:549-576`）。判据 `within_noise_band(value, band)`
为闭区间判定：落在 `[q50, max]` 内 = 与重复生成噪声**不可分辨**（`src/ard/core/coverage.py:597-608`）。

组装：`src/ard/core/acceptance.py:729-769`（`noise_section`）从库记录中找**同 prompt 签名**的分组
（`repeat_groups`，`src/ard/core/acceptance.py:676-702`），无重复生成数据时按 `NOISE_UNAVAILABLE_REASON`
（`src/ard/core/acceptance.py:85-87`）显式写 **`unavailable`**，绝不省略。

### 4.1 分组键 = prompt 签名（不是整份 `anchor_meta`）

**这是 WP-S18 修正的口径。** 分组的定义是"两次生成发给输入生成器的请求逐字节相同"，因此分组键必须是
**prompt 组装实际读到的字段序列**，而不是 `anchor_meta` 的全部键：

- 签名 = `PROMPT_SIGNATURE_AXES`（`src/ard/core/acceptance.py:123-136`）的取值**有序元组**；缺轴 = `None`，
  不是"取该轴第一个值"（`prompt_signature`，`src/ard/core/acceptance.py:401-413`）。
- 这 12 个轴就是**全部 12 个本体轴**，逐字段的消费点见 `src/ard/core/acceptance.py:89-122` 的注释：
  `language` / `knowledge_domain` / `capability` / `conversation_type`
  （`src/ard/domain/text_anchor.py:233-236` 读，`:266-284` 拼进指令，`:286-291` 组装 system 串）、
  `system_prompt_mode`（`src/ard/backends/prompt_loader.py:99-100` 选措辞文件，
  `src/ard/core/system_prompt.py:83-87` 填占位符）、
  六个 instruction 轴（`src/ard/domain/text_anchor.py:242-243` → `src/ard/backends/axis_instruction_loader.py:96-97`）、
  `visual_domain`（`src/ard/domain/image_store.py:128-130` 选图目录，即请求里带哪张图）。
- **不在**签名里的是计划记账字段 `modality` / `has_image` / `image_count`：它们不改变发给生成器的请求文本。
  `image_count = min(用户轮数, IMAGES_PER_ANCHOR)` 且 `IMAGES_PER_ANCHOR = 1`（`src/ard/pipeline.py:357`），
  信息量为零；契约测试 `tests/core/test_acceptance_prompt_signature.py` 双向渲染两遍（列出轴必须改变渲染、
  未列字段必须不改变渲染）来守住这个集合，轴一旦变成"吉祥物"即失败。
- **判据没有放宽**：仍然只在"同一签名出现 **≥2** 条"时才标定噪声带。v4 计划里每条签名都只出现一次，
  所以正常运行的读数**就是** `unavailable`；重复组是因为"同一格被生成多次"才出现，不是为了"能出数"而放宽。

**为什么旧口径会低估噪声**：按整份 `anchor_meta` 分组时，两条**只差不进 prompt 的字段**的记录会被分到两格，
于是"同一请求的两次生成"被算成"两个格子"，重复对消失、噪声带被报成 `unavailable` 或偏窄，
而 `q95` 的臂间差反而显得"可分辨"。这正是 WP-S14 §3 记下的口径缺陷。

## 5. 空间声明要求

每次指标读数必须**声明它是在什么空间里测的**（`src/ard/core/acceptance.py:302-334`，`SpaceDeclaration`）：

| 字段 | 要求 | 值/来源 |
|---|---|---|
| `anchors_source` | 锚点向量来自哪个产物 | 锚点库路径（如 `outputs/<run>/anchor_bank.jsonl`） |
| `anchor_field` | 嵌入的是该产物的哪个字段 | `src/ard/core/acceptance.py:70`：`messages[last].content(text parts only)`（最后一个 user 轮的**文本部分**） |
| `targets_source` | 目标集文件路径 | `coverage.target_set_path` |
| `target_field` | 嵌入目标条目的哪个字段 | `src/ard/core/acceptance.py:82`：`text` |
| `n_anchor` / `n_target` | `|A|` / `|T|` | 实测矩阵行数 |
| `embedder` | **嵌入器身份 = model + dimension + normalize** | `src/ard/core/acceptance.py:292-299`；由 `[coverage.embedding]` 解析 |
| `distance` / `quantile_method` | 距离与分位定义 | `"1 - cos"`、`"linear"`（type-7） |
| `epsilon` / `epsilon_source` | ε 及其来源 | 头部声明或目标集自身尺度 |

**指标空间只含文本；图像模态以其文本部分参与，图像像素不进该空间。** 图像模态锚点的最终 user 轮
`content` 是多模态 part 列表（`{"type": "image", ...}` 与 `{"type": "text", "text": ...}` 并列），
取文只取其中 `type == "text"` 的 `text`，故 `anchor_field` 写作 `messages[last].content(text parts only)`，
而不是笼统的 `messages[last].content`——否则读数的空间声明会被误读成"图像也进了这个空间"。
多个文本部分按声明分隔符 `"\n"`（`src/ard/core/acceptance.py:79`）连接；非文本 part 一律忽略。

**没有任何文本部分 ⇒ 报错，不静默跳过、不用空串占位。** 某条锚点的最终 user 轮若一个可用文本部分都没有
（纯图像锚点、或文本部分全为空白），`user_turn_text`（`src/ard/core/acceptance.py:578-634`）与
`anchor_texts`（`src/ard/core/acceptance.py:637-673`）以 `AcceptanceError` 终止，报文含**记录 id、part 数量与 part 类型**。
静默跳过会让 `q95` 落在一个比运行产物更小的锚点集上，空串占位则是在空间里伪造一个点——两者都是伪读数。

**不含密钥**：空间声明只写 model 与 dimension，不写 `api_base`、不写任何 key；输出目录里的配置快照另有脱敏
（`src/ard/pipeline.py:203`，`_redact_secrets`）。

## 6. 读数流水线与退化行为

```mermaid
flowchart TD
    A["锚点库 anchor_bank.jsonl"] --> B["anchor_texts()<br/>取最后一个 user 轮的文本部分<br/>（image 像素不入空间）"]
    T["覆盖目标集 target_set_path"] --> U["load_target_set()<br/>计数/维度/epsilon 头校验"]
    B --> E["EmbeddingClient /embeddings<br/>backends/embedding_client.py"]
    U --> E
    E --> N["L2 归一化 VectorSet"]
    N --> D["nearest_anchor_distances<br/>d = min(1 - cos)"]
    D --> Q["q50/q90/q95/r_max + |T|"]
    D --> X["Extent(ε) 与 ε±5% 敏感带"]
    N --> W["同格重复生成对距离 → 噪声带 [q50,max]"]
    Q --> R["results/coverage.json + coverage.md"]
    X --> R
    W --> R
```

**三种配置组合的契约**（`src/ard/config.py:352-404`，`resolved_embedding`；`src/ard/pipeline.py:656-698`，`_prepare_coverage`）。

| `coverage.target_set_path` | `[coverage.embedding]` | 行为 |
|---|---|---|
| **两者都未设** | 任意 | **结构读数 + 明确 WARNING**：`metric readout not measured …`（`src/ard/pipeline.py:674-680`）；报告 `metrics: null`、manifest `metric_readout: false` / `q95: null`。不静默、不崩溃 |
| **已设** | **缺** `api_base` / `model` / `dimension`（或全空） | **fail-fast 报错**（`ConfigError`），报文逐项列出缺失字段并要求"补齐或清空 `target_set_path`"；在 `load_config` 即触发（`src/ard/config.py:381-387`），**早于创建任何输出目录**。这是**刻意严格**的行为：目标集已声明要测指标，却没有可用嵌入器，属于配置错误而非可退化的缺省 |
| **已设** | 完备，`normalize = true` | 产出**指标读数**（`q95` / `Extent(ε)` / ε 带 / 噪声带）；`normalize = false` 同样 fail-fast（尺子要求 L2 归一化，`src/ard/config.py:389-393`） |

其余退化行为（都**显式**、绝不静默，`src/ard/pipeline.py:747-844`，`_run_acceptance`）：

| 情形 | 行为 |
|---|---|
| 目标集缺失/不可解析/计数或维度与配置矛盾 | 在**创建输出目录之前**拒绝整个运行（`src/ard/pipeline.py:656-698`，`CoverageWiringError`；`:941` 在输出目录存在之前调用），不产生半成品产物 |
| 嵌入调用失败 | 结构性读数**先已落盘**（`src/ard/pipeline.py:813-820`），失败照常抛出，但不会抹掉零成本的结构报告 |
| 无同格重复生成数据 | 噪声带写 `unavailable` 并附原因（`src/ard/core/acceptance.py:85-87`；`src/ard/pipeline.py:827-828` 把原因并入 warnings） |
| 库比计划少一条计划坐标 | 结构读数仍描述计划，但读数被如实改为 `within_rule: false`，并在 warnings 里**列出缺失坐标**（`src/ard/pipeline.py:716-745`；一条计划坐标没落库时，计划再合规也不算"产物合规"） |

## 7. 已知边界：指标层的分辨力上限

**这是本尺子的硬约束，读任何 `q95` 数字前必须先读它：**

- 噪声带的语义是"**同格重复生成**这条管道自身产生的距离散布"。当两臂的 `q95` 差值**小于**该噪声时，
  指标层**无法区分**这两臂——差异可能来自生成随机性，而非被测的设计因素。
- **实验结论（设计阶段的可分辨性实验）**：在"多臂（六臂）× 多尺子层（四层）"的格子中，
  **24/24 的 `q95` 都落在同格重复生成噪声带 `[q50, max]` 内**；配对 bootstrap 的臂间差 CI 大多含 0。
  ⇒ 尾部读数（`q95`、`r_max`）已被噪声主导，**指标层已到顶**。
  （出处：v4 设计阶段的本地实验记录，**未纳入版本控制**，此处只作方法论沿革；不引用任何部署/运行编号，
  也不构成对某个具体运行的断言。）
- **由此得出的口径**：在该分辨力上限内，**结构保证才是硬约束**——
  即"935/891 个合法受限块每块恰好 1 条、**935/891 个互异 prompt 签名**、**935/891 个有效受限投影**、
  `knowledge_domain` 209 叶轮转、`visual_domain` 21 叶轮转、坐标 0 重复"
  这类可零成本复核的结构事实，比臂间 `q95` 排名更可靠。`q95` 用于**自证与回归**（同一构造的读数是否稳定），
  不用于主张细微的算法优劣。
  后两个数不是装饰（§11）：它们正是"935 个格子"与"935 个不同规格"之间那道必须被读出来的差别。
- 报告的措辞要求：当读数落在噪声带内时，不得写"某臂更优/更差"，只能报
  **效应量 + CI 宽度 + 最小可辨差（≈ CI 半宽）**，并明确声明"不可分辨"。

## 8. 产物字段

- `results/coverage.json`：`AcceptanceReport` 的完整机器可读序列化（`report_schema = "ard-acceptance-2"`），
  含 `structure` / `metrics`（`space` / `quantiles` / `extent` / `epsilon_band` / `noise`）/ `warnings`。
- `results/coverage.md`：同一报告的人读版渲染（`src/ard/core/acceptance.py:815-934`），
  结构读数表、Diversity declaration（两个 distinct 读数的定义）、Conventions（`MULTI_TURN_DEFAULT` 与轮数映射）、
  空间声明、分位/覆盖读数、噪声带、warnings。
- `manifest.json` → `plan_identity`：**这份读数描述的是哪个计划**。`{algorithm, version, plan_size, digest}`，
  其中 `digest` 是有序坐标列表的 sha256（`src/ard/core/sampling.py:677-726`）。它是续跑守卫的键（§12），
  也是"读数绑定到哪个计划"的锚点；`manifest.json` 的 `config` 段里的 `seed` **不是**这个锚点。

**`ard-acceptance-1` → `ard-acceptance-2`（WP-S18，字段语义变更 ⇒ 版本号递增）。** 两处变化：
`structure` 新增 `prompt_signature_distinct` / `effective_projection_distinct` / `diversity`（§11），
且 `metrics.noise.n_repeat_groups` 的含义从"整份 `anchor_meta` 相同的格子数"改为"prompt 签名相同的格子数"（§4.1）。
**这两个版本的 `noise` 字段不可直接比较**：同名不同义，所以老报告必须按 `report_schema` 区分后再读
（`src/ard/core/acceptance.py:62-69`）。

## 9. 复核指标路径：三步

指标读数**完全由配置驱动**（没有 CLI 开关，`src/ard/config.py:335-404`）。要让复核员能零成本复跑：

1. **配嵌入器**：在 `.local/config.override.toml`（被 gitignore 的本机覆写文件，**绝不入库**）里写
   `[coverage.embedding]`。下文只给**字段名与占位符**，不给真实端点/模型/密钥：

   ```toml
   [coverage.embedding]
   api_base = "<OpenAI 兼容的 /embeddings 基址，含 /v1>"
   model = "<嵌入模型名>"
   dimension = <该模型的向量维度>
   # api_key = "<服务端需要时填>"
   # batch_size = 32
   # normalize = true      # 必须为 true：尺子要求 L2 归一化行
   ```

2. **指向目标集**：设 `coverage.target_set_path`。仓库自带一个小而确定的样例
   （`examples/target_set.sample.jsonl`，32 条、约 4.5 KB，构造规则见其文件头与 §10），开箱可用：

   ```toml
   [coverage]
   target_set_path = "examples/target_set.sample.jsonl"
   ```

3. **跑一次**：`--smoke`（8 条锚点，几分钟）或完整运行。产物在 `<output_dir>/results/coverage.json`
   与 `coverage.md`：`metrics.space` 是空间声明，`metrics.quantiles.q95` 是主读数。

三种配置组合的行为见 §6 的契约表：都没设 ⇒ 结构读数 + WARNING；只设目标集不配嵌入器 ⇒ fail-fast 报错；
两者都设 ⇒ 指标读数。只想验证接线、不想调用生成端点时，可以对**已有的** `anchor_bank.jsonl` 直接调用
`pipeline._prepare_coverage` + `pipeline._run_acceptance`（`src/ard/pipeline.py:656-698` / `:747-844`）。

## 10. 目标集样例的构造规则

`examples/target_set.sample.jsonl` 是**接线样例（wiring sample），不是产品目标集**。它的构造规则写在文件头
（`axes` / `selection` / `template` 三个字段）里，机器可读、可复算：

- **用到的轴**：`knowledge_domain` 与 `language`（取自 `ontology/anchor_ontology.v4.json`）；
- **`knowledge_domain` 取法**：按 `knowledge_domain_tree` 的文档序展开成 **209 个叶值**，取下标
  `floor(k × (209 − 1) / 7)`，`k = 0..7` —— 等距 8 个，**含首尾**；
- **`language` 取法**：`axes.language.values` 的**全部 4 个值**，按本体序；
- **文本模板**（唯一一条，逐字）：
  `Explain {leaf} in depth: what it is, how it is studied, and why it matters. Answer in {language}.`
- **顺序与数量**：`knowledge_domain` 外层、`language` 内层，共 **8 × 4 = 32 条**；文件头声明
  `expected_count: 32`，条目数不符即被 `load_target_set` 拒绝。

该文件不含端点、密钥、模型名或个人信息。文件是**一行 JSON 文档**（外层是带 `targets` 数组的对象），
这同时满足 JSONL 的"一行一个 JSON 值"与 `load_target_set` 的"对象文档可带头部"两种读法——JSONL 逐行读取
时不支持头部，头部因此只能由对象文档承载。

**它不保证读出的数字有意义**：32 条同模板句子的自身尺度很小（本机离线 MiniLM 384-d 下 ε ≈ 0.021，
故 `Extent(ε)` 通常为 0）。样例只证明"目标集 → 嵌入 → 距离 → 读数"这条**接线**通，不构成任何覆盖结论；
产品级读数需要真正的目标集与真正的锚点库。

## 11. 结构读数里的"有效多样性"——以及读数如何被误读

**为什么非要报这两个数（WP-S14 审计的教训）。** WP-S14 的只读审计（**工作区产物，未入库**；其脚本只读
import 本体加载器与两个真实 prompt 渲染器，不调用任何模型）发现：把 935 个合法受限块投影到
**prompt 真正读到的受限轴**上，只剩 **112** 种（影像是 102 种）；**4282 对**块只差在
`output_format` / `input_condition` / `answer_mode` 上——只要自由轴相同，它们发出的 prompt **逐字节相同**。
也就是说：**"935 个块"曾经被读成"935 个不同规格"，而当时只有 112 个规格真的影响生成。**
WP-S17 把这六个 instruction 轴的措辞真的渲染进 prompt 之后，"有效投影"才等于块数（112/102 → 935/891，
见本页 §4.1 逐字段消费点）。

**于是结构读数并列报四个数**（`structure` 里，全部按模态分开；定义随读数一起落盘在 `structure.diversity`）：

| 字段 | 含义 | 当前全量计划 |
|---|---|---|
| `text_block_count` / `image_block_count` | 覆盖到的**合法受限块**数（标称块数） | 935 / 891 |
| `prompt_signature_distinct` | 不同 **prompt 签名**数 = 不同生成侧请求数 | 935 / 891 |
| `effective_projection_distinct` | 受限轴里**真正进 prompt**的那些轴的**不同投影**数 = 能改变生成结果的不同规格数 | 935 / 891 |
| `knowledge_domain_leaves` / `visual_domain_leaves` | 轮转叶覆盖数 | 209 / 21 |

`prompt_signature_distinct` 与 `effective_projection_distinct` 都是 `{text_only, image}` 两个整数；
`structure.diversity` 写明签名包含哪些轴、按什么顺序拼、在什么集合上数（`counted_over`），
`docs/` 这页只讲"为什么要报"。任何一个 distinct 数**小于**同模态的块数，`within_rule` 即为 `false`、
`coverage.md` 对应行标 `MISMATCH`、`structure_mismatch` 的 warning 逐行点名——这是防呆，
不是给读数化妆。

**读数如何被误读（三种典型错误，都必须避免）**

1. **把标称块数当有效多样性**：`935/935 块覆盖` 只说明"计划枚举了 935 个合法受限组合"，
   **不说明**它们产生 935 个不同请求或 935 个不同规格。要读有效多样性，只能看上面后两个数。
2. **把 `prompt_signature_distinct` 当"设计格数"**：签名 distinct=935 只说"935 个不同 prompt"。但签名互异
   **可以靠单轴轮转撑起来**——WP-S14 在 S17 之前的口径下测得多样性分解是
   **有效受限投影 112 → 加 `language` 386 → 加 `knowledge_domain` 935**，即当时"935 个不同 prompt"
   由 `knowledge_domain` 逐条轮转贡献了后 549 个。所以**签名互异 ≠ 受限规格互异**，两个数必须一起读；
   只看签名数会把"轮转出来的差异"当成"实验格子本身的差异"。
3. **把 `noise.available=false` 当"没有噪声"**：`unavailable` 的含义是"**这批产物里没有同签名的重复生成**"，
   不是"噪声为零"。v4 计划每格只生成一次，所以正常运行的读数就是 `unavailable`（§4.1）；此时
   `q95` 的臂间差**没有标定过**可分辨性，不得据此写"某臂更优/更差"（§7）。

**复算**：前三个数都是零成本的纯结构计数，`acceptance.structure_readout(plan)` 即可复得；
本机既有产物上重跑读数的做法见 §9 结尾，不需要任何生成调用。

## 12. 计划身份：读数必须绑定到它，而不是绑定到 seed

**问题**：`manifest.json` 的 `config.generation.seed` 是**进程级配置值**，不是计划的身份。输出目录的
`config.toml` 快照**每次运行都被覆盖**（`src/ard/pipeline.py:1109`），包括什么都没生成的空转续跑；
不钉 seed 时每个进程还会现抽一个新 seed（`src/ard/config.py:204-205`）。所以"记录 seed + 代码"**不能**
复现历史产物里的自由轴分配——历史冒烟库被调用 3 次，后两次空转覆盖了真正采样那次的 seed，
用记录值复算自由轴只有 11/32 相同。
**结论：计划本身由 `(本体, seed, 代码)` 唯一决定，所以"钉 seed ⇒ 计划可复现"成立；但"产物里记录的 seed
⇒ 计划可复现"不成立**，因为记录值可能属于另一次调用。

**做法**：每次运行把 `plan_identity`（有序坐标列表 sha256 + `plan_size` + `algorithm` + `version`）写进
`manifest.json`（`src/ard/pipeline.py:490-496`）。同一计划在任何进程、任何 `PYTHONHASHSEED` 下得到同一摘要
（S22 实测：7 个进程、1,826 条全量计划，摘要逐字节相同）。

**绑定规则（全量 1,826 运行时）**：

1. **产品级读数（`q95` / `Extent` / 覆盖与多样性）只能绑定到 `plan_identity.digest`**：写报告时引用该摘要，
   而不是引用 seed；两个摘要相同才允许把两次读数并列比较。
2. **续跑守卫比为计划身份**（`src/ard/pipeline.py:420-487`）：摘要不同 ⇒ 拒绝续跑（明确报错"holds a different
   plan"），绝不把两份计划的锚点写进同一份 `anchor_bank.jsonl`；摘要相同 ⇒ 允许。旧库（无 `plan_identity`）
   走结构校验：库里存在不属于本次计划的 anchor id 即拒绝，绝不静默混合。
3. **`plan_identity` 没有被记录的老产物**（本字段引入之前的库）不能凭 seed 断言其计划身份，只能按结构校验
   （库里所有 id ⊆ 本次计划）续跑；若要做产品级读数，先重新采样并记录摘要。

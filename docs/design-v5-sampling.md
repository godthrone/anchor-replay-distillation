# ARD v5 采样设计：随机顺序轮转 + 覆盖优先

> **状态：v5 已实现**（本工作树；v4 基线为 `main` @ `3d4e545`，tag `v4.0.0`）。
> 本文是设计稿兼决策记录：它写结构与决策（宪法 §1.6），不写实现细节；代码位置一律用符号引用
> （如 `ard.core.sampling.sample_coordinates`），不写 `文件:行`（§17.2）。
> 文中的"裁定"是主席对本设计待决项的最终拍板；实现必须照裁定写，与本裁定冲突的早期草稿段落已删除。

---

## 一、三次主席裁定（回填）

设计稿原先有 3 个待主席拍板的问题。主席的最终裁定如下，**它们是 v5 的口径，不是候选方案**。

### 裁定 1：N 由用户设置且无上限，唯一入口是 config

- **N 的唯一入口**是 `configs/config.toml` 的 `[generation] count`，类型 `int | None`。
  **缺省 `None` = 一个完整轮 = 单元数 U**（运行时穷举给出，今天 1826）。
- **没有 CLI 参数、没有 `run.sh` 透传、没有环境变量**。这是宪法 §10.1 推论 1/2 的直接要求：
  CLI 与 config 的取值域零交集，不存在"config 有值、CLI 再覆盖"，也不存在第二真相源（§1.4）。
- **N 不作上限拦截**。越界判定只有两条：拒绝 `count < 1`、拒绝非法类型（含 `bool`、字符串、浮点）。
  **任何大 N 都不拒绝**——计划按轮滚动（第 0 轮、第 1 轮…），机制就是"取更长前缀"。
- **"上限"降级为信息提示**：当 N 很大时，日志报"跑满几轮 / 末轮几条 / 共多少条"
  （`sampling.plan_rounds` 的 `(full_cycles, last_cycle_size)`），而不是拒绝请求。
- N 不参与计划的前缀性质：`plan(seed, N)` 是 `plan(seed, N')` 的前缀（`N <= N'`），
  自由轴随机种子只依赖计划下标，不依赖 N（`sampling._free_rng`）。

### 裁定 2：锚点 id 与坐标彻底脱钩

- id 是**计划位置序号**（库的主键），不是内容指纹：

  ```
  run_key = H(本体哈希, seed, 采样算法版本)[:8]
  id      = f"{run_key}-c{轮次:05d}p{轮内序号:05d}"
  ```

  实现符号：`sampling.run_key` / `sampling.format_anchor_id`。
- **N 不参与 id**，所以把 `[generation] count` 调大后，已经写出的 id 一个都不变——续跑安全。
- id 与坐标无关，因此同一坐标跨轮出现两次会得到**两个不同的 id**，两条都留存。
  这不是缺陷：温度 0.8 下同一坐标能问出不同的题，重复采样是**合法样本**而不是重复数据。
- 旧 id 构造（对抽中的 5 个坐标轴求 sha256）**整体清除**：`sampling.generate_anchor_id`、
  `sampling.ANCHOR_ID_DIMENSIONS`、`sampling._reject_duplicates`、`Coordinate.identity()` 全部删除，
  仓内零残留（§18.1 不留负债）。

### 裁定 3：采样规则 = 随机顺序轮转；不按坐标去重；覆盖率按坐标、密度按条数

- 采样规则是 **cycle-shuffle**：每轮把覆盖单元集合重洗一遍、轮内不放回；轮尽重洗进入下一轮。
- **不再有任何"按坐标去重"**。同一坐标跨轮再次出现是设计允许的，不做运行时查重、不丢记录。
  坐标相同 ≠ 样本重复。
- **覆盖率按坐标统计**：同一坐标跨轮重复**不重复计入**，饱和于 `1.0`（`min(distinct, U) / U`）。
- **密度按条数统计**：`N / U`，重复**计入**（`N = 2U` ⇒ `2.0`）。两个口径必须分开读。
- 伴随裁定：**出题温度保持 0.8 不动**；**影像通道改 1 域多图、按轮次确定性轮转**
  （`ard.domain.image_store.select_domain_image` 的 `cycle` 参数，`cycle=0` 与 v4 旧行为逐字节一致）。

**续跑守卫（由裁定 1、2 推出的必要条件）**：既然 id 是位置序号，判据不能是"已有 id 集合 ⊆ 新计划 id 集合"
——那会把 smoke 计划与全量计划混进同一目录（两者 id 形状相同）。正确判据是
**逐条比对已有记录在其 id 所指位置上的坐标是否与新计划一致**（`pipeline._refuse_foreign_records_on_resume`）。
于是"把 count 调大再跑同一目录"顺畅追加；换 seed / 换本体 / smoke↔full 换形状都会被拒绝并提示换目录。

---

## 二、覆盖单元、轮转算法、不变量

**覆盖单元** =（模态，合法受限块）=（`text_only` | `image`，6 个受限轴的合法组合）。
清单由运行时穷举给出（`sampling.coverage_units`），本体完善时 U 自动变大；**本体里没有任何手写计数**
可以与之冲突。一轮 = U 条。

**算法（伪代码，与实现一一对应）**

```
units    = 文本合法块 ++ 影像合法块                 # 穷举；U = len(units)
order(c) = shuffle(units, H(seed, 本体哈希, "cycle", c))
coordinate(i):
    c, pos = divmod(i, U); u = order(c)[pos]
    k = leaves_k[(b(u) + c) % K]                   # b(u) = 单元在声明序中的下标
    v = leaves_v[(b(u) + c) % V] if u.modality == image else None
    free = draw(Random(H(seed, 本体哈希, "free", i)))   # 4 个自由轴
    return Coordinate(modality=u.modality, block=u.block, knowledge=k, visual=v, **free)
plan(seed, N) = [coordinate(i) for i in range(N)]
```

`b(u)` 是单元的固定下标，`K`/`V` 是知识叶 / 视觉叶数（运行时从本体读）。
用 `+c`（步长 1）而不是草稿的 `+37·c`：两者相位周期相同，但步长 1 永远与 K 互质，
用户加叶后不会失效。实现符号：`sampling._iter_plan` 的 `order` / `coordinate`。

**不变量**

| 标记 | 内容 | 依据 |
|---|---|---|
| I1 | 一轮内每个单元恰好出现一次 | `order(c)` 是排列 |
| I2 | `N >= U` 时 K 个知识叶、V 个视觉叶全覆盖 | 一轮内 `b(u)` 取遍 U 个连续整数，U >= K 且影像单元数 >= V |
| I3 | 同一 run 内 id 两两不同 | id 是 `(轮次, 轮内序号)` 的位置序号，构造即单射 |
| I4 | 同一坐标可以在不同轮再次出现，且两条都留存 | 裁定 3：坐标是内容不是身份；id 与坐标脱钩 |

干跑验证（设计算法 + 真实本体，400 轮）：每轮知识叶 209/209、视觉叶 21/21；
`plan(seed,100)` 是 `plan(seed,3U)` 的前缀。

**上限**：**没有坐标去重保证到期这一说，也没有 id 维度上限**。N 任意大都是合法请求，
越过 U 后照常进入下一轮；覆盖率在 `N >= U` 后饱和于 `1.0`，密度继续线性增长。
（早期草稿里的"第 K 轮 = 209 轮 = 381,634 条硬报错"与"5 轴 id 上限 585,200 条"两条边界
随裁定 2、3 一并作废——坐标不再查重，id 也不再由坐标维度决定。）

---

## 三、宪法审视后落到本设计的基线

1. **禁止 `--count`（§10.1 推论 1/2）**：N 只在 `configs/config.toml` 的 `[generation] count`；
   CLI 与 config 零交集。CLI 只有 `--config` / `--image-dir` / `--override` / `--smoke`，
   全部落在三原则内，**无违规**。`--smoke` 是宪法明文允许的运行边界参数，保留。
2. **不留负债（§18.1）**：不保留 v4 等价模式。v5 验证通过后删除遗留采样入口
   `PlanScale`、`FULL_SCALE` / `SMOKE_SCALE`、`_evenly_spaced_indices`、`_select_blocks`、
   `EXPECTED_*`、`_verify_rule_counts`、`_rotating`，以及 id 侧的 `generate_anchor_id` /
   `ANCHOR_ID_DIMENSIONS` / `_reject_duplicates`；库里只留一套采样路径与一套 id 公式。
3. **零步上手（§4）**：根目录已有 `run.sh`，`python -m ard` 可直接跑；实测 clone → 产出 3 步
   （Docker 路径 2 步），满足 ≤ 3，不需要新增脚本。无凭证时**没有降级模式**，在加载期拒绝。
4. **图一律 Mermaid（§17.2）**：本文无 ASCII 手绘图；正文只写结构，不写实现细节。
5. **单一真相源（§1.4）**：本体里的手写计数块必须收敛，采纳**方案 (a) 删除**，不采纳 (b) 加载时忽略——
   (b) 会在文件里留下一份用户看得见、以为要维护的陈旧数据，正是 §18.1 的负债。删除清单与全部消费者：
   - 本体文件：`axes.knowledge_domain.counts`、`axes.capability.counts`、`axes.visual_domain.counts`、
     顶层 `derived_counts`、顶层 `reachability`；
   - `ard.core.ontology`：对应的 `HierarchicalAxis.counts` / `GroupedAxis.counts` /
     `DerivedCounts` / `Reachability` 字段；
   - 消费者（须同步改）：`tests/core/test_constraints.py` 的"自述 vs 穷举"比对、
     `tests/core/test_ontology_v4.py` 的必填字段用例、`docs/algorithm.md` 对 `reachability` 的引用。
   保留本体里的轴值清单本身（叶子清单就是唯一权威）。
6. **空值语义（§2.2）**：`count: int | None = None`，`None` = 一个完整轮 = U；不用 `0` / `-1` 表示"未提供"。
7. **机密（§15/§16）**：本文不写任何真实端点、密钥或环境专用路径。
8. **符号引用（§17.2）**：本文不用 `文件:行`。
9. **不提交（§19.1）**：本轮只落文件，不做 `git add` / `git commit`。

---

## 四、简化清单（一行一条，[x] = 本次实现已落地）

1. [x] 不引入 `--count`；N 只进 config。
2. [x] 删 v4 等价模式与上列遗留采样入口（含 id 侧三个符号），只留一套采样路径。
3. [x] 删本体手写计数块（`counts` / `derived_counts` / `reachability`），运行时穷举为唯一来源。
4. [x] 本体计数不再写死期望值；改数本体后"跑的时候不报错"。
5. [x] `docs/` 与 README 的手抄计数表改成"命令 + 输出"，不手抄数字。
6. [x] 按裁定删掉那个"盯行号"的守卫测试（它自我声明抓不到行号漂移），文档改符号引用。
7. [x] 测试从写死 1826/935/891/209/21 改为对任意 N 的不变量测试 + 负控。
8. [x] 验收读数从"等于 1826"改为按本 run 的 N 与轮分解判定（smoke 不再误报 MISMATCH）。
9. [x] 本体指纹沿革表删除；本体哈希进 `manifest.json` / `plan_identity`，身份自动派生，用户零登记。
10. [x] 失败一律硬报错（缺凭证 / 非法 N / 续跑形状不符），不静默、不降级。

---

## 五、兼容与迁移

- 单条锚点记录的**字段格式不变**；`ard.domain.bank.SCHEMA_VERSION` 由 `4.0.0` 升为 `5.0.0`
  （采样规则与 id 公式均属不兼容变更，宪法 §18.1 升 MAJOR）。**历史产物不迁移、不回填**。
- `plan_identity` 升 v2（`sampling.PLAN_IDENTITY_VERSION`）：记录本体哈希、seed、count、U、
  采样算法与条数；它只描述计划，不描述部署。
- v4 运行目录**不能续跑**（id 公式变了、顺序也变了），换新目录即可；无数据迁移、不重算历史读数。

---

## 六、未核实与待办

- `[未核实]` `uv sync` 安装步骤未实跑（本环境无外网）；步数按干净检出结构与 README 清点。
- `[未核实]` 提高出题温度对覆盖/多样性指标的实际影响，未测（且本轮裁定温度保持 0.8，不再改）。
- 待办：无（设计条目已全部落地；后续如需改动，改的是实现与文档，不是本设计稿的口径）。

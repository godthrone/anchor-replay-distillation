# Compliance Exemptions — Class A (BADGE Constitution v2.3.0)

This file records **evidence-based exemptions** for `tools/check_all.sh --class=A`
findings that are tool false positives or sanctioned §12.2/§15.x designs, so that
a reviewer can judge them without re-deriving the evidence. It complements, and
does not weaken, the constitution: every entry below is a **recorded false
positive / documented design**, not a suppressed real violation.

- 检查基线（用户认可口径）: `bash tools/check_all.sh --class=A --no-history`
- 记录日期: 2026-09-15 · **最近复核: 2026-09-16（工具 T1–T5 补丁后）**
- 项目: ARD (anchor-replay-distillation), branch `main` (HEAD `8fe3828`, tag `v3.0.0`)
- 覆盖范围: 本文件只覆盖 **Class A** 检查项中经人工复核判定为"证据化豁免"的条目。

> 处理原则（用户/指挥官认可）: 不为了通过工具而谎改文件、不删除合法内容（例如
> 中文 README 的 `## 许可证` 段）；豁免必须以证据记录，供审查者复核。

---

## 状态速览（2026-09-16 复核）

工具侧 T1–T5 补丁落地后（BADGE 宪法仓库 `tools/check_readme_parity.sh` 与
`tools/check_secrets.sh`，工作区已改、未提交），本文件原先记录的四个 FAIL 项
**全部不再上报**：

```bash
# <repo-root>         = 本项目根目录，即 `git rev-parse --show-toplevel`
# <constitution-repo> = BADGE 宪法仓库本地检出（本机路径不写入本文件，§15.1）
bash <constitution-repo>/tools/check_all.sh --class=A --no-history <repo-root>
#   Passed: 25
#   Failed: 0
# [PASS] All checks passed.
```

| # | 检查项 | 2026-09-15 判定 | 2026-09-16 判定 | 结论 |
|---|--------|:---:|:---:|------|
| E1 | §12.2 class-path mirroring | FAIL → 已消除 | PASS | 留痕（违规已实质修复，R10） |
| E2 | §15.1 机密扫描（API key 占位符） | FAIL | **PASS** | 工具 T4 修复模式过宽 → 豁免留痕 |
| E3 | §15.1 PII 扫描（uv.lock 数字串） | FAIL | **PASS** | 工具 T2/T5 修复正则边界 → 豁免留痕 |
| E4 | §17.1 README 镜像（CN "License"） | FAIL | **PASS** | 工具 T1 接受中文章节名 → 豁免留痕 |

**因此 E2/E3/E4 均已失去豁免对象**：它们不再是任何 FAIL 的依据，条目保留仅为
**留痕**，说明"当时为何如此判定、后来如何消失"。下方各节保留原始证据。

> ⚠️ **另一个维度（提交元数据 PII）不在这四条豁免之内**，它有自己的 ref 级口径：
> 推送范围（`main` + tags）元数据全为平台 noreply；曾含个人邮箱的历史脏 ref 不在
> 推送范围、已备份到 `.local/` 并清理。见文末《提交元数据 PII》。

### 行号口径（取代旧快照行号）

本文件**不再写死行号**。README 持续被编辑，任何行号当场过期。复核一律用 grep，
按**命中内容**判定：

```bash
grep -n 'api_key' README.md README.zh-CN.md          # E2：是否仍为占位符形态
grep -n '^## ' README.md README.zh-CN.md             # E4：`## License` ↔ `## 许可证`
grep -c 'files.pythonhosted' uv.lock                 # E3：uv.lock 命中来源
```

（2026-09-16 实测定位：EN `api_key = "REPLACE_WITH_YOUR_API_KEY"` → `README.md:501`、
`README.zh-CN.md:460`；`## License` → `README.md:675`、`## 许可证` → `README.zh-CN.md:608`，
两处正文均为 `MIT`。）

---

## E1 — §12.2 类名-路径镜像：**已消除，豁免已撤销**

**原命中**（R6 快照）：`check_all.sh --class=A --no-history` 输出

```
[FAIL] src/ard/domain/bank.py: single class 'AppendOutcome' (full snake: append_outcome)
       → expected file name to be a suffix of 'append_outcome.py' (§12.2 class-path mirroring)
```

**复核裁定（推翻原豁免，2026-09-15）**：原 E1 条目按"数据容器依附函数族"（§12.2
豁免）记录为"类推豁免、工具与宪法的缝隙"。**该裁定错误**：

1. §12.2 豁免明文只列 **frozen dataclass、pydantic 模型**，**`Enum` 不在其内**；
   `AppendOutcome` 是 `Enum`，不能主张该豁免。
2. `check_class_file_naming.sh` 的豁免通道（`_*` / `*Mixin` / `@dataclass` /
   `BaseModel|BaseSettings` / 类目录 / 多类文件）与宪法文字**一致**——因此这是
   **真实的 §12.2 不合规**，不是工具误报。原条目第 3、4、5 点（"工具缝隙"、
   "改名违背设计"）均以错误前提成立，现一并作废。
3. 原第 3 点所述"使用点超出 bank 自身"是**改名的成本**，不是豁免的依据。

**修复（R10 工作包，改代码不改规则）**：

- `AppendOutcome`（4 成员 / 值 / docstring 一字未改）迁至新模块
  `src/ard/domain/append_outcome.py`，类名 ↔ 文件名精确镜像。
- `src/ard/domain/bank.py` 删除类定义，改为**自用导入**；文件头原"§12.2 数据容器
  豁免"注释（错误）已改为事实描述（Enum 不属豁免明文，已归位同名模块）。
- 不留兼容 shim（§18.1）：`bank.py` **不** re-export `AppendOutcome`，调用方直接
  从新模块导入 —— `src/ard/domain/text_anchor.py`、`tests/domain/test_bank.py`、
  `tests/domain/test_system_prompt_anchor.py`、
  `tests/domain/test_text_anchor_backpressure.py`。

**复跑证据**（2026-09-15）：

```bash
# <constitution-repo> = BADGE 宪法仓库本地检出（内网路径不写入本文件，§15.1）；
# <repo-root>         = 本项目根目录，即 `git rev-parse --show-toplevel`。
bash <constitution-repo>/tools/check_class_file_naming.sh <repo-root>
# Checking single-class file naming...
#   [OK] All single-class files match their class names.
# [PASS] check_class_file_naming: Class-path mirroring satisfied.   (EXIT=0)
```

条目保留仅为**留痕**（说明"豁免被撤销、违规已实质修复"），不再是任何 FAIL 的豁免依据。

---

## E2 — §15.1 机密扫描：README 的 api_key 占位符 → **已转 PASS（工具 T4 修复）**

**原命中**（2026-09-15）：两份 README 的 Quick Start 配置示例中各出现两处 `api_key`
值为显式占位符（值含 "REPLACE_WITH" 字样的占位形态，25+ 字符）。

**证据 / 为何当时判定为误报**:

1. 命中值是 **自描述的显式占位符**，不是任何真实凭证；其在 README 中的用途是
   引导部署者填写自己的密钥（§15.2 明确要求模板/文档使用占位符，如 `REPLACE_ME`）。
2. §15.1 PII 豁免清单本身包含"明确的占位符（`REPLACE_ME`、`your-email@…`）"；
   本占位符属同一语义类别。原工具 `check_secrets.sh` 的 `api_key` 通用正则
   只匹配"api_key=" 上下文与 16+ 字符值，未对占位符值单独豁免 —— 属模式覆盖过宽。
3. 真实配置模板不含该字面值: `configs/config.toml` 与
   `configs/config.override.sample.toml` 的 `api_key` 均为空串（TOML 序列化边界
   空值约定，加载时归一为 None，见 §2.2）；本地覆写走 gitignored 的
   `config.override.toml`。即"占位符只出现在文档层，凭证永不入库"符合 §15.2。

**现状（2026-09-16）**：工具侧 T4 补丁为 `check_secrets.sh` 增加了**占位符行过滤**
（`filter_placeholder_lines` / `PLACEHOLDER_VALUE_PATTERN`），并在 docstring 中写明
其安全性论证：一行只有在**取出的每个值都是占位符形态**时才被丢弃，任一真实值即保留
整行，因此"真实密钥与占位符同行"不会被掩盖。复跑结果：

```bash
bash tools/check_secrets.sh --no-history <repo-root>
# [PASS] check_secrets: No secrets or PII found in tracked files.
```

**豁免处置**：命中已消除，**条目转为留痕**。README 中的占位符**保留不动**——它是
§15.2 要求的模板形态，不是需要清理的内容。

**残余风险（如实记录，供后续复核）**：过滤依据是"值形如占位符"。若将来有人把示例值
改成**真实形态的密钥字面量**（例如 `sk-...`），工具会照常 FAIL——那是正确行为，不是
本豁免可以覆盖的情形。本条目**不授权**任何真实凭证出现在文档中。

**history 维度覆盖**：`--no-history` 口径只扫工作树。`check_secrets.sh` 默认含
`git log --all -p`，另报 1 处历史命中（commit `e6e9da0` 的 README 同样含该占位符
字面值）；其性质与工作树命中完全相同（显式占位符），§15.1 占位符豁免同样适用，
因此本豁免结论在 history 维度**继续成立**，不需要重写历史（§15.3 仅适用于真实
机密/PII 泄露）。原 R6 还记录了工具健壮性缺陷（全历史跑时 `head -20 | while read`
触发 SIGPIPE 中止、Phase 3 未执行），该缺陷已由 T3 修复。

---

## E3 — §15.1 PII 扫描：uv.lock 哈希/URL 数字串 → **已转 PASS（工具 T2/T5 修复）**

**原命中**: `uv.lock` 中 24 处 pythonhosted.org 包的 URL 与 SHA-256 哈希所含数字串
（扫描器归类为"个人信息（PII）"，实际为大陆座机正则
`0[0-9]{2,3}-?[0-9]{7,8}` 命中连续数字段）。

**证据 / 为何当时判定为误报**:

1. 逐条核对命中行：全部位于 `uv.lock` 的 `{ url = "https://files.pythonhosted...
   hash = "sha256:<64-hex>" ... }` 描述行。被命中的数字串（如哈希内
   `0xx-xxxxxxx` 形片段）是 SHA-256 十六进制哈希与 wheel 路径的**巧合数字序列**，
   既非手机号也非座机号，不指向任何自然人。
2. 对策场合: §6 环境可复现**强制要求提交 `uv.lock`** 锁定精确版本与哈希；该文件是
   依赖完整性的防呆装置，哈希不可删除。

**现状（2026-09-16）**：工具侧 T2/T5 补丁把三条数字串正则（手机 / 座机 / 身份证）
的边界统一为**前置 `[^0-9A-Fa-f.]` + 后置 `[^0-9A-Fa-f]`**，即十六进制哈希内部的
数字段不再被当作电话号码。复跑结果：

```bash
bash tools/check_all.sh --class=A --no-history <repo-root>
# [PASS] check_secrets: No secrets or PII found in tracked files.
```

**豁免处置**：命中已消除，**条目转为留痕**；`uv.lock` 保持原样入库。

**history 维度覆盖**：`--no-history` 口径报 24 处命中，全部位于工作树 `uv.lock`；
全历史扫描另报多个历史 commit 中的 `uv.lock` 哈希行，性质与工作树命中相同
（SHA-256 哈希 / wheel URL 的巧合数字串），`uv.lock` 在历史上从未出现过真实 PII，
且它**必须**入库（§6）。因此本豁免结论在 history 维度**继续成立**，不需要重写历史。

---

## E4 — §17.1 README 镜像：CN "License" 段 → **已转 PASS（工具 T1 修复）**

**原命中**（2026-09-15）: `check_readme_parity.sh` 输出 `[FAIL] CN README missing section: License`。

**证据 / 为何当时判定为误报**:

1. 原 `check_readme_parity.sh` 的 `REQUIRED_TERMS_CN` 数组硬编码了**英文字面
   "License"** 作为 CN README 的必需小节词，并对 CN 文件做 `grep "License"`。
2. CN README 实际存在对应小节 `## 许可证`（正文 "MIT"），与 EN README 的
   `## License`（正文 "MIT"）**内容镜像**。工具按字面匹配"License"，无法识别中文
   "许可证" —— 属工具误报。
3. §17.1 要求的是内容镜像：两版小节（EN "License" ↔ CN "许可证"）均存在且正文一致
   （MIT），语义已满足。为迎合工具在 CN 中强行加入英文 "License" 字样属于
   破坏合法内容，不予执行（用户认可口径）。

**现状（2026-09-16）**：工具侧 T1 补丁把该条改为**多语言别名正则**：

```
"License|Licence|许可证|許可證|许可协议|許可協議|授权协议|授權協議|开源协议|開源協議"
```

复跑结果（2026-09-16）：

```bash
bash tools/check_readme_parity.sh <repo-root>
# [PASS] check_readme_parity: READMEs are consistent.
```

**豁免处置**：命中已消除，**条目转为留痕**。两份 README 的 License 小节**保持原样**，
未为迎合工具加入英文字面。

**行号口径**：本条目原曾写死行号（R10 期间实测从 `:520` → `:592` → `:599` 漂移）。
现按上文的 grep 口径复核：`grep -n '^## ' README.md README.zh-CN.md` → EN `:675`、
CN `:608`，两处正文均为 `MIT`。

---

## 提交元数据 PII：**推送范围全净，历史脏 ref 不在推送范围且已清理**

> 本条与 E1–E4 性质不同：E1–E4 是"工具误报 / 已消除"；本条记录的是**元数据维度的
> 实际口径与处置**——结论必须连同 **ref 级限界**一起读，不能只写一个 PASS/FAIL。

**推送范围（`main` + tags `v1.0.0` / `v2.0.0` / `v3.0.0`）实测：0 条个人邮箱。**

```bash
cd <repo-root>
git log --format='%an <%ae>|%cn <%ce>' main refs/tags/v1.0.0 refs/tags/v2.0.0 refs/tags/v3.0.0 \
  | tr '|' '\n' | sort -u
#   GitHub <noreply@github.com>
#   godthrone <70987748+godthrone@users.noreply.github.com>
# → 全部为平台 noreply 身份，无个人邮箱命中。
```

**曾用于取证、且不在推送范围的历史 ref（`refs/remotes/remote-check/*`）**：其提交身份中
出现过**个人邮箱**（作者名 `godthrone` / `zzy`，地址形如 `63****@qq.com`——**本文件不复
录该地址的明文**，§15.1 同样约束记录隐私事件的文档本身）。这些 ref **不属于推送范围**，
并已按"保留证据、再清理"的方式处置：先 `git bundle` 备份到
`.local/backups/remote-before-forcepush-20260916.bundle`（`.local/` 已被 `.gitignore`
排除，**bundle 不入库、不推送**），再从本地删除这些 ref 并 prune。

> 归属更正：`refs/heads/backup-before-l10-cleanup` **不含**个人邮箱身份（实测 0 条），
> 也不属于推送范围；它不在此次清理之列。

**清理后实测**（2026-09-16 复核）：

```bash
cd <repo-root>
git log --all --format='%ae|%ce' | grep -c 'qq\.com'      # → 0
bash <constitution-repo>/tools/check_secrets.sh --no-history --meta-pii=fail <repo-root>
# [PASS] check_secrets (metadata): No personal email in commit/tag metadata.
```

**因此**：

- 「推送会把个人邮箱推上去」这一风险**不成立**——推送范围只有 `main` 与 tags，
  其元数据全为平台 noreply；本地全 ref 的 QQ 计数亦已归零；
- 未经清理的脏 ref 只存在于 `.local/` bundle 中，原始证据未丢失、且不入库不推送；
- 复核该维度时**必须写明 ref 范围**：在含 `remote-check/*` 取证 ref 的历史状态下跑会
  得到 FAIL，在"推送范围"或"清理后的当前本地状态"下跑为 PASS。旧版 E2 中"元数据
  PASS"一句的适用前提正是**干净状态**（无 `remote-check/*` 取证 ref），此处明确写出，
  避免误读。

---

## 遗留观察（非豁免、需他方跟进）

- **README 中英 token_ids 方向矛盾（§17.1 一致性）—— 已由 P1B 修复（复核 2026-09-15 23:30）**：
  原 EN README 表述"优先字符串 token、整数 token_id 兜底"、CN 表述"优先整数 token ID、
  字符串 token 兜底"，**二者方向相反**。现两版口径已一致：均说明"历史上 v2 的 token 列表
  **优先字符串 token**（`logprobs.content[].token`），字符串形式缺失时 fallback 到整数 ID
  （`logprobs.content[].token_id`）"（EN README FAQ "What format are the generated
  token_ids?"、CN README FAQ "生成的 token_ids 是什么格式？"）。词串位置随 README 编辑
  漂移，请用 `grep -n 'token_id' README.md README.zh-CN.md` 定位。本条改记为**已解决**，
  仅留痕说明原 P5 观察的处置结果。
- 本地文件 `configs/config.override.toml`（未跟踪）含一个本地"无鉴权"跑通用的
  `api_key` 占位值（无真实密钥语义）：该文件已被 `.gitignore`
  （`config.override.toml`）覆盖，不会入库，无泄露风险；仅提醒后续可统一归入
  `.local/`。

---

*本文件由 P5 合规收口工作包生成；R10（2026-09-15）撤销了 E1（§12.2 Enum 已归位
`append_outcome.py`，违规实质修复），并为 E2/E3 补上 history 维度说明、核准 E2/E4
行号口径。**2026-09-16 复核（F2 工作包）**：工具 T1–T5 补丁落地后 E2/E3/E4 亦转为
PASS，条目一并改为留痕；行号口径统一改为 grep 定位；新增《提交元数据 PII》一节，
按"推送范围全净 + 历史脏 ref 已备份清理"的口径记录。若后续工具/宪法升级进一步收敛
（占位符值、lockfile 数字串、中文小节匹配已由 T1–T5 覆盖），本条目的留痕部分可整体归档。*

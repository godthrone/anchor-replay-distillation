# Compliance Exemptions — Class A (BADGE Constitution v2.3.0)

This file records **evidence-based exemptions** for `tools/check_all.sh --class=A`
findings that are tool false positives or sanctioned §12.2/§15.x designs, so that
a reviewer can judge them without re-deriving the evidence. It complements, and
does not weaken, the constitution: every entry below is a **recorded false
positive / documented design**, not a suppressed real violation.

- 检查基线（用户认可口径）: `bash tools/check_all.sh --class=A --no-history`
- 记录日期: 2026-09-15
- 项目: ARD (anchor-replay-distillation), branch `feature/v3.0.0-b1`
- 覆盖范围: 本文件只覆盖 **Class A** 检查项中经人工复核判定为"证据化豁免"的条目。

> 处理原则（用户/指挥官认可）: 不为了通过工具而谎改文件、不删除合法内容（例如
> 中文 README 的 `## 许可证` 段）；豁免必须以证据记录，供审查者复核。

---

## 豁免摘要表

| # | 检查项 | 命中位置 | 工具判定 | 处置 | 原因 |
|---|--------|---------|:---:|:---:|------|
| E1 | §12.2 class-path mirroring | `src/ard/domain/append_outcome.py` → class `AppendOutcome` | **PASS** | **已消除（撤销豁免）** | 真实违规已修复：Enum 归位到同名模块 `append_outcome.py`（R10，2026-09-15） |
| E2 | §15.1 机密扫描（API key） | `README.md:454/459`、`README.zh-CN.md:424/429` 的 `api_key` 示例（2026-09-15 核准快照；行号会漂移，见 E2 §行号口径） | FAIL | 证据化豁免 | 显式占位符（§15.2 模板规范），非真实密钥；history 维度已覆盖（见 E2） |
| E3 | §15.1 PII 扫描（手机/座机） | `uv.lock` 内 pythonhosted URL 与 SHA-256 哈希 | FAIL | 证据化豁免 | 哈希/URL 数字串命中 "大陆座机" 正则，纯误报；§6 要求提交 uv.lock；history 维度已覆盖（见 E3） |
| E4 | §17.1 README 镜像 | CN README "License" 段（CN 侧为 `## 许可证`，正文 "MIT"；行号漂移，见 E4） | FAIL | 证据化豁免 | 工具按英文字面 "License" 匹配 CN；CN 已有 `## 许可证` 镜像 |

> 行号口径：本文件所有行号均为 **2026-09-15 快照时刻** 的行号。README 正在被
> P1B 工作包编辑、`uv.lock` 会随依赖同步变化，行号会漂移；复核时应按 **FAIL 类别
> 与命中内容**（占位符字面值、`api_key` 上下文）定位，不依赖行号本身。

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

## E2 — §15.1 机密扫描：README 的 api_key 占位符

**命中**: `README.md:454/459`、`README.zh-CN.md:424/429`（**2026-09-15 核准快照**；
原记 `README.md:415/420`、`README.zh-CN.md:395/400` 已随 README 编辑漂移、作废）——
两份 README 的 Quick Start 配置示例中各出现两处 `api_key` 值为显式占位符
（值含 "REPLACE_WITH" 字样的占位形态，25+ 字符）。

**证据 / 为何豁免**:

1. 命中值是 **自描述的显式占位符**，不是任何真实凭证；其在 README 中的用途是
   引导部署者填写自己的密钥（§15.2 明确要求模板/文档使用占位符，如 `REPLACE_ME`）。
2. §15.1 PII 豁免清单本身包含"明确的占位符（`REPLACE_ME`、`your-email@…`）"；
   本占位符属同一语义类别。工具 `check_secrets.sh` 的 `api_key` 通用正则
   `api[_-]?key\s*[=:]\s*["']?[a-zA-Z0-9_-]{16,}["']?` 只匹配"api_key=" 上下文与
   16+ 字符值，未对占位符值单独豁免 —— 属模式覆盖过宽。
3. 真实配置模板不含该字面值: `configs/config.toml` 与
   `configs/config.override.sample.toml` 的 `api_key` 均为空串（TOML 序列化边界
   空值约定，加载时归一为 None，见 §2.2）；本地覆写走 gitignored 的
   `config.override.toml`。即"占位符只出现在文档层，凭证永不入库"符合 §15.2。
4. 处置：不改 README（P1B 负责），记录豁免。

**行号口径（R6 瑕疵修正）**: 上述行号为快照值。README 由 P1B 工作包持续编辑，
行号必然漂移——复核时请用 `grep -n 'api_key' README.md README.zh-CN.md` 重新定位，
并按"值是否仍为 `REPLACE_WITH…` 形态占位符"判定，不按行号判定。

**history 维度覆盖（R6 瑕疵修正 · 补 E2）**: 本豁免**不止覆盖 `--no-history` 口径**。

- `check_all.sh --class=A --no-history`（项目默认协议）只扫工作树，本项 4 处命中
  （`probe-classA_run1.log` / `classA_run_full.log`）。
- **全历史扫描**（`check_secrets.sh` 默认含 `git log --all -p`）另报 1 处历史命中：
  commit `e6e9da0` 的 README 同样含该占位符字面值（R6
  `probe-secrets-history.log:44-48`）。
- 该历史命中的**性质与工作树命中完全相同**：显式占位符，非真实凭证，
  §15.1 占位符豁免同样适用；因此 E2 的豁免结论在 history 维度**继续成立**，
  不需要重写历史（§15.3 仅适用于真实机密/PII 泄露）。
- 已知工具健壮性缺陷（非本项目问题，见 R6 §2）：全历史跑时 `check_secrets.sh`
  因 `head -20 | while read` 触发 SIGPIPE 中止（`probe-secrets-history.log:56`），
  Phase 3 元数据段未执行；元数据段已用
  `check_secrets.sh --no-history --meta-pii=fail` 单独补跑，结果 **PASS**
  （提交元数据全为平台 noreply 地址）。

---

## E3 — §15.1 PII 扫描：uv.lock 哈希/URL 数字串

**命中**: `uv.lock` 中 24 处 pythonhosted.org 包的 URL 与 SHA-256 哈希所含数字串
（扫描器归类为"个人信息（PII）"，实际为大陆座机正则
`0[0-9]{2,3}-?[0-9]{7,8}` 命中连续数字段）。

**证据 / 为何豁免**:

1. 逐条核对命中行：全部位于 `uv.lock` 的 `{ url = "https://files.pythonhosted...
   hash = "sha256:<64-hex>" ... }` 描述行。被命中的数字串（如哈希内
   `0xx-xxxxxxx` 形片段）是 SHA-256 十六进制哈希与 wheel 路径的**巧合数字序列**，
   既非手机号也非座机号，不指向任何自然人。
2. 对策场合: §6 环境可复现**强制要求提交 `uv.lock` 锁定精确版本与哈希**；该文件是
   依赖完整性的防呆装置，哈希不可删除。
3. 处置：不改 uv.lock，记录豁免。`--no-history` 口径下本项不涉及 git 历史中的
   真实 PII（已单独跑元数据扫描 PASS）。

**history 维度覆盖（R6 瑕疵修正 · 补 E3）**: 本豁免**不止覆盖 `--no-history` 口径**。

- `check_all.sh --class=A --no-history` 报 24 处命中，全部位于工作树 `uv.lock`
  （R6 `classA_run_full.log:182-203`）。
- **全历史扫描**（`check_secrets.sh` 默认含 `git log --all -p`）另报多个历史 commit
  中的 `uv.lock` 哈希行（R6 `probe-secrets-history.log:55-76`）。
- 该历史命中的**性质与工作树命中完全相同**：SHA-256 哈希 / wheel URL 的巧合数字串，
  不指向任何自然人；`uv.lock` 在历史上从未出现过真实 PII，且它**必须**入库（§6）。
  因此 E3 的豁免结论在 history 维度**继续成立**，不需要重写历史。
- 为完整起见：历史维度的**提交元数据**扫描（`--meta-pii=fail`）单独跑为 **PASS**，
  即"历史中确无真实 PII"这一前提已被实测确认（不依赖本豁免推定）。

---

## E4 — §17.1 README 镜像：CN "License" 段

**命中**: `check_readme_parity.sh` 输出 `[FAIL] CN README missing section: License`。

**证据 / 为何豁免**:

1. `check_readme_parity.sh` 的 `REQUIRED_TERMS_CN` 数组硬编码了**英文字面
   "License"** 作为 CN README 的必需小节词，并对 CN 文件做 `grep "License"`。
2. CN README 实际存在对应小节 `## 许可证`（正文 "MIT"），与 EN README 的
   `## License`（正文 "MIT"）**内容镜像**。工具按字面匹配"License"，无法识别中文
   "许可证" —— 属工具误报。
   **行号不写入本条目**：R10 核查期间实测 `README.md` 的 `## License` 从 :520 → :592
   → :599 漂移（P1B 正在编辑 README），任何写死的行号当场过期。复核请用
   `grep -n '^## ' README.md README.zh-CN.md` 即时定位（2026-09-15 23:25 实测：
   EN `:599`、CN `:552`，均为 `##` 小节且正文 "MIT"）。
3. §17.1 要求的是内容镜像：两版小节（EN "License" ↔ CN "许可证"）均存在且正文一致
   （MIT），语义已满足。为迎合工具在 CN 中强行加入英文 "License" 字样属于
   破坏合法内容，不予执行（用户认可口径）。
4. 处置：不改两份 README（P1B 负责），记录豁免。

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
行号口径。若后续工具/宪法升级补上对应豁免通道（占位符值、lockfile 数字串、
中文小节匹配），相应条目应转为自动 PASS 并撤销豁免。*

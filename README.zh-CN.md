# ARD — Anchor Replay Distillation

> 多模态锚点数据生成工具，支持导出 log-prob 用于 OPD 训练。

**ARD** 生成锚点数据集，防止持续微调过程中的灾难性遗忘。
它从本体论中采样多样化提示词，使用教师模型生成答案，
并导出 token 级 log-prob 用于 On-Policy Distillation (OPD) 训练。

## 快速开始

```bash
# 1. 准备配置
cp configs/config.toml my_config.toml
# 编辑 my_config.toml：填入 API 地址和模型名
# 创建 config.override.toml 填入 API key：
cat > config.override.toml << 'EOF'
[input_generator]
api_key = "your-api-key"

[target_model]
api_key = "your-api-key"
EOF

# 2. 安装
uv sync

# 3. 生成文本锚点
uv run ard --config my_config.toml

# 4. 生成文本 + 多模态锚点
uv run ard --config my_config.toml --image-dir /path/to/images

# 5. Docker（备选）
bash run.sh --config configs/my_config.toml --image-dir /data/images
```

## 配置

所有配置在一个 TOML 文件中。完整模板见 `configs/config.toml`。

| Section | 用途 |
|---------|------|
| `[input_generator]` | 生成用户提问的 VLM/LLM |
| `[target_model]` | 生成答案（含 log-prob）的教师模型 |
| `[generation]` | 目标数量、随机种子、语言和任务类型过滤 |
| `[ontology]` | 本体论路径 |
| `[output]` | 输出目录设置 |

机密信息（API key）填入 `config.override.toml`（gitignored）。

## CLI

```
ard --config <路径> [--image-dir <路径>]
```

单一命令完成所有操作：

- `--config` — TOML 配置文件路径（必填）
- `--image-dir` — 多模态锚点的图片目录（可选）

## 输出

```
outputs/<dataset_name>/
├── anchor_bank.jsonl          # 统一锚点数据（graspo 兼容）
├── images/                    # 多模态图片（如有）
└── manifest.json              # 统计摘要
```

### 数据格式

所有锚点写入单个 `anchor_bank.jsonl` 文件，格式与 graspo 兼容：

```json
{
  "id": "anchor_a1b2c3d4",
  "source": "ard",
  "messages": [{"role": "user", "content": "解释熵的概念..."}],
  "targets": [{
    "id": "primary",
    "output": {
      "content": "熵是衡量系统无序程度的物理量...",
      "logprobs": {
        "token_ids": [1, 2, 3],
        "log_probs": [-0.1, -0.2, -0.3]
      }
    }
  }],
  "anchor_meta": {"language": "简体中文", "knowledge_domain": "science"},
  "teacher_id": "Qwen3.8-27B"
}
```

## Docker

```bash
# 构建
bash docker/build.sh

# 运行
bash run.sh --config configs/my_config.toml
```

## 开发

```bash
uv sync --dev
uv run pytest tests/ -v
```

## 许可证

MIT
# Session Handoff — llm-learning

> **本卡用途**：把项目交给「下一阶段（Session 003+）」的接手者，主题是 **P2 — Qwen3 从零实现**。
> **context 用满（≥70%）或长 session 结束前必须更新本卡**（见 `AGENTS.md` §9）。
> 规则：用指针代替贴全文；不带密钥；一次 session 只关心当前任务。

---

## 一句话现状

GPT-2 线的笔记与代码已就位但**训练链路因 torch 环境坏了没复验**；**下一阶段是从零手写 tiny Qwen3**，Qwen3 代码尚未开工，21 条 feature 已排好序。

---

## 当前阶段（P2：Qwen3 从零实现）

**范围**：从零手写 tiny Qwen3 —— RoPE + RMSNorm + QK-Norm + SwiGLU + GQA（后期 MoE）。
**不做什么**：不引入 `transformers` / `peft` / `accelerate`，不跑真 Qwen3-0.6B，不做 LoRA 微调。

**为什么这么定**：
1. 与 GPT-2 线同一套「从零手写 + 可 debugger」的做法，两个模型能互相印证
2. 依赖上限保持 `torch` / `numpy` / `nltk`，不让仓库变成模型动物园
3. tiny 版本足够看清每个组件的机制

**核心方法论 —— 组件全部做成可关掉的开关**：

```
use_gqa      = False   → 退化为 MHA
use_rope     = False   → 退化为无位置编码
use_qk_norm  = False   → 退化为裸注意力
mlp          = "gelu"  → 退化为 GPT-2 的 FFN
```

退化后的 Qwen3 应当与 GPT-2 在同输入下对齐（`max_abs_diff < 1e-4`，fp32）。
这把「Qwen3 到底改了什么、为什么改」从**文字断言**变成**可重跑的实验**（`feat-017`）—— 这是 Qwen3 线最有价值的产出。

---

## Verified Now（哪些是已验证的）

- **能用**：`bash init.sh` 的 L0 / L1 / L3 三层全绿（Python 3.11 可用、纯 Python 数据层可用、仓库忽略规则生效）
- **不能用**：torch 相关的一切 —— `import torch` 报 `WinError 127`（feat-001）
- **已跑过的验证命令**：`bash init.sh`（2026-10-18，PASS=11 / SKIP=2 / FAIL=0，退出码 0）

## Broken Or Unverified（坏的 / 没验证的）

- **已知缺陷（阶段硬前置）**：`import torch` 失败 → `OSError [WinError 127] Error loading torch_python.dll`。
  候选解法：① 重装匹配 Python 3.11 的 torch CPU wheel；② 安装 Microsoft Visual C++ Redistributable。**两条都没试过。**
- **未验证路径**：GPT-2 线的 `train_pretrain.py` / `train_sft.py` / `train_reward.py` / `train_ppo.py` / `evaluate.py` / `inference.py` 全部**没有在当前环境跑过**
- **不可信数据**：`gpt-learning/checkpoints/pretrain.pt`、`ppo.pt` 由旧 torch 产出，当前环境无法复验其指标。
  笔记里引用这些数字的地方（13.gpt2评估.md 等），在复跑前**不要当作事实**
- **风险**：`feat-011` ~ `feat-018`（Qwen3 全线）都依赖 feat-001；torch 不修，模型代码写完也验证不了

## Next Best Step（按顺序，不要跳）

| 顺序 | feature | 做什么 | 为什么是它 |
|------|---------|--------|-----------|
| 1 | `feat-001` | 修 torch，重跑 `bash init.sh` 确认 L2 变 PASS | 硬前置，不修则下面全部无法验证 |
| 2 | `feat-011` | 建 `qwen3-learning/qwen3-pytorch/` 骨架 + 独立 `config.py` + `README.md` | P2 的起点；同步把它挂进 `init.sh` 的 L4 |
| 3 | `feat-012` | RMSNorm + Pre-LN Block | 最简单的组件，先跑通「模块 → smoke → 笔记标注」这套循环 |
| 4 | `feat-013` | RoPE（含旋转等变性自测 + 外推长度接口） | 注意预留 `position_ids`，KV Cache 的增量位置要用 |
| 5 | `feat-014` | QK-Norm + GQA | **要确认 QK-Norm 在 RoPE 之前还是之后**，以代码注释写明 Qwen3 的选择 |
| 6 | `feat-015` | SwiGLU（hidden_dim 对齐到 128 的倍数） | 顺带算清与 GPT-2 的 FFN 参数量对比，写进笔记 |
| 7 | `feat-016` | 完整前向 + 权重绑定 | **返回接口刻意对齐 `GPT2.forward` 的 4 元组**，为 feat-017 铺路 |
| 8 | `feat-017` | 与 GPT-2 的退化对照测试 | 前面组件的「意义证明」，也是本线最有价值的产出 |
| 9 | `feat-018` | KV Cache 推理 + 采样 | 复用 GPT-2 线 `generate.py` 的 prefill/decode 组织方式 |

**第 2 步这一步不许动的**：不改 GPT-2 线的任何训练代码或超参；不在 Qwen3 目录外 import。

---

## 需要用户拍板的决策（别自行决定）

| 决策 | 出现在 | 说明 |
|------|--------|------|
| Qwen3 用 `head_dim` 与 `n_embd` 解耦后，层数 / 头数 / 维度的具体取值 | `feat-011` | 先给方案与理由，再问用户 |
| tokenizer 沿用 GPT-2 的字符级，还是引入 BPE | `feat-019` | 选 BPE 会动到依赖上限 |
| 是否做 MoE（`feat-021`），还是只做笔记对照 | `feat-021` | 时间紧可只做笔记 |

---

## Commands（常用命令）

```bash
# 启动 + 轻量验证（五层 L0-L4）
bash init.sh
bash init.sh --quiet            # 只看汇总行

# GPT-2 线（代码目录已存在）
cd gpt-learning/gpt-learning-pytorch
python data/build_vocab.py
python train_pretrain.py && python train_sft.py && python train_reward.py && python train_ppo.py
python evaluate.py --stage all --all_checkpoints checkpoints/

# Qwen3 线（P2 目标目录，尚未创建）
# cd qwen3-learning/qwen3-pytorch

# 状态
cat feature_list.json            # 含 phase 阶段标记
cat claude-progress.md           # 上次做到哪
git log --oneline -5
```

## 相关文件速查

| 想做什么 | 看哪个 |
|----------|--------|
| 开工前必读 | `AGENTS.md` §0 framing / §3 硬约束 / §5 验证路径 |
| 当前阶段与任务顺序 | `feature_list.json` 的 `phase` + priority 最小的非 passing 项 |
| Qwen3 笔记原文 | `qwen3-learning/Qwen3_*.md`（7 篇） |
| GPT-2 对照实现 | `gpt-learning/gpt-learning-pytorch/model/gpt2.py` |
| 收班自查 | `clean-state-checklist.md` |

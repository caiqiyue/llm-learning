# Session Handoff — llm-learning

> **本卡用途**：把项目交给「下一阶段（Session 005+）」的接手者，主题是 **P2 — Qwen3 从零实现**。
> **context 用满（≥70%）或长 session 结束前必须更新本卡**（见 `AGENTS.md` §9）。
> 规则：用指针代替贴全文；不带密钥；一次 session 只关心当前任务。

---

## 一句话现状

GPT-2 线**全链路已复验通过**（torch 环境已修好，训练 + 9 项评估全绿）；Qwen3 线的**组件层已全部从零手写完**（RMSNorm / RoPE / KVCache / QK-Norm+GQA 注意力 / SwiGLU / Qwen3Dense / Qwen3MoE / byte-level BPE），**训练侧（预训练 → LoRA → DPO）与评估侧尚未开工**。

---

## 当前阶段（P2：Qwen3 从零实现）

**范围**：从零手写 tiny Qwen3 —— RoPE + RMSNorm + QK-Norm + SwiGLU + GQA（后期 MoE）+ byte-level BPE，
然后走 **预训练 → LoRA 微调 → DPO**（路线已从 SFT/Reward Model/PPO 切到 LoRA+DPO，见 `feature_list.json` 的 `phase`）。
**不做什么**：不引入 `transformers` / `peft` / `accelerate` / `sentencepiece` / `tokenizers`，不跑真 Qwen3-0.6B。

**核心方法论 —— 组件全部做成可关掉的开关**（feat-017 的退化对照实验用）：

```
use_gqa      = False   → 退化为 MHA
use_rope     = False   → 退化为无位置编码
use_qk_norm  = False   → 退化为裸注意力
mlp          = "gelu"  → 退化为 GPT-2 的 FFN
```

退化后的 Qwen3 应当与 GPT-2 在同输入下对齐（`max_abs_diff < 1e-4`，fp32）。
这把「Qwen3 到底改了什么、为什么改」从**文字断言**变成**可重跑的实验**（`feat-017`）。

---

## Verified Now（哪些是已验证的）

- `bash init.sh` → **PASS=22 SKIP=0 FAIL=0**（2026-10-22）
- `cd qwen3-learning/qwen3-pytorch && python tests/run_all.py` → **PASS=8 FAIL=0 TOTAL=8**
- Qwen3 前向 smoke 已用**真实 BPE 词表**（vocab 778）跑通：`logits (2, 8, 778) loss 6.6282 params 985,856`
- BPE 端到端：语料 20000 句 / 785065 字节 → 词表 778（521 次合并）→ **压缩比 7.10 字节/token**
- GPT-2 线：`python evaluate.py --stage all` 退出码 0，9 项评估全过
- 各 feature 的详细证据在 `feature_list.json` 的 `evidence` 数组里（feat-011 ~ feat-022 都有）

## Broken Or Unverified（坏的 / 没验证的）

- **Qwen3 训练侧完全没跑过**：`train/` 目录下还没有任何脚本（预训练 / LoRA / DPO / 评估都待写，见 plan 的 Task 13–16）
- **BPE 词表只有 778 < 目标 1000**：合成语料的可合并对用尽（521 次后没有频次 ≥ 2 的对），属语料多样性问题，非实现 bug
- **BPE 语料与预训练语料是两个生成器**：`build_vocab.py` 与待写的 `data/pretrain_data.py` 各自合成语料，词表对预训练语料的覆盖率会打折
  → **Task 12 必须把语料生成器收敛成唯一真源**（建议放 `data/pretrain_data.py`，`build_vocab.py` 改 import 它）并重新生成词表
- **笔记缺口**：`Qwen3_BPE_学习笔记.md` / `Qwen3_LoRA与DPO_学习笔记.md` 未写（按计划在 feat-008 / Task 19 补）
- **MoE 的「参数多 N 倍、计算只多 k 倍」在本仓库不成立**：专家更窄（128 vs 384），全模型只多 1.21×（前提条件见笔记 §25.1）

## Next Best Step（按顺序，不要跳）

| 顺序 | feature | 做什么 | 为什么是它 |
|------|---------|--------|--------|
| 1 | `feat-019` | `data/pretrain_data.py` + `data/preference_data.py`：预训练切块（block_size+1，inputs/targets 错位一位）+ DPO 偏好对（至少 3 种「变差」策略：wrong_entity / unrelated / too_short / noisy） | 预训练和 DPO 的前置；同时把**语料生成器收敛成唯一真源**并重建词表 |
| 2 | `feat-020` | 预训练 smoke（`--model dense|moe`，超参全从 `config.py` 读） | 打通「数据 → 训练 → loss 下降」 |
| 3 | `feat-023` | 手写 LoRA（B 零初始化 + base 冻结 + inject/merge/enable 三件套） | 路线已切到 LoRA+DPO |
| 4 | `feat-024` | 手写 DPO（**π_ref 复用同一模型对象，靠关 LoRA 分支得到**，不训 Reward Model） | 用户明确要求 DPO |
| 5 | `feat-025` | Qwen3 精简评估（pretrain/lora/dpo/speedup 四段） | 验收 |
| 6 | `feat-018` | KV Cache 推理 + 采样（temperature/top-k/top-p） | 推理侧 |
| 7 | `feat-017` | 与 GPT-2 的**退化对照**（逐组件消融 + max_abs_diff 记录） | 本线最有价值的产出 |
| 8 | `feat-008` | 笔记 ↔ 源码交叉引用（补 BPE / LoRA / DPO 三篇笔记） | 收尾 |
| 9 | `feat-009` | 笔记数字与 `config.py` 全量核对 | 收尾 |

**这一步不许动的**：不改 GPT-2 线的任何训练代码或超参；不在 `qwen3-learning/qwen3-pytorch/` 之外 import；不削弱 `evaluate.py` 的任何指标与阈值。

---

## 已知的「文字断言 vs 可跑实验」陷阱（别再踩）

| 结论 | 关键点 |
|------|--------|
| QK-Norm 在 RoPE 前 vs 后 | **用均匀 gain 测不出差异**（max\|diff\| ≈ 3.6e-07）。`rotate_half` 把第 i 维与第 i+half 维配对，逐维 gain 与旋转不对易。必须用**非均匀 gain** 才暴露（7.38 ~ 14.9） |
| RoPE 是相对位置编码 | 绝对位置**整体平移**不改变输出（max\|diff\| < 1e-5）；只有改变**相对距离**才改变输出 |
| SwiGLU vs GPT-2 4×-GELU 参数量 | 差 = `矩阵数 × H × I`。I 严格取 8/3·H 时**完全相等**；本仓库取整到 384 → 比值 9/8 = 1.125×。Qwen3-0.6B 实际用 3×hidden，8/3 是 LLaMA 的规则 |
| MoE aux_loss | Switch 论文版均匀时 = 1；HF Mixtral 版（tokens×k 求均值、系数 E）均匀时 = **1/K**（K=2 → 0.5）。本仓库用 Mixtral 版 |
| BPE 的 id 分配 | 合并 token 从 **257** 开始（不是 256 —— 256 是字节 0xFF 的位置，撞车会把它静默吞掉） |

---

## 需要用户拍板的决策（别自行决定）

| 决策 | 出现在 | 说明 |
|------|--------|------|
| 预训练要跑多少 step / 学习率是否要调 | `feat-020` | 先给方案与理由，再问用户 |
| DPO 偏好数据要不要引入真实数据集 | `feat-019` | 目前是合成数据，引入真实数据会动依赖上限 |
| 评估要不要和 GPT-2 线的 9 项对齐 | `feat-025` | Qwen3 线目前规划的是 4 段精简评估 |

---

## Commands（常用命令）

```bash
# 启动 + 轻量验证（L0-L4）
bash init.sh
bash init.sh --quiet            # 只看汇总行

# GPT-2 线（代码目录已存在，全链路已复验）
cd gpt-learning/gpt-learning-pytorch
python data/build_vocab.py
python train_pretrain.py && python train_sft.py && python train_reward.py && python train_ppo.py
python evaluate.py --stage all

# Qwen3 线（组件层已就位，训练侧待写）
cd qwen3-learning/qwen3-pytorch
python data/build_vocab.py       # 重建 BPE 词表（vocab.json / merges.txt）
python tests/run_all.py          # 8 个 smoke 汇总
python tests/test_bpe.py         # 单个 smoke

# 状态
cat feature_list.json            # 含 phase 阶段标记 + 每条的 evidence
cat claude-progress.md           # 上次做到哪
git log --oneline -5
```

## 相关文件速查

| 想做什么 | 看哪个 |
|----------|--------|
| 开工前必读 | `AGENTS.md` §0 framing / §3 硬约束 / §5 验证路径 |
| 当前阶段与任务顺序 | `feature_list.json` 的 `phase` + priority 最小的非 passing 项 |
| Qwen3 设计 / 实施计划 | `qwen3-learning/qwen3-pytorch设计.md`、`qwen3-learning/qwen3-pytorch开发计划.md`（Task 7–11 已就地纠错，含「实测修正记录」） |
| Qwen3 笔记原文 | `qwen3-learning/Qwen3_*.md`（7 篇，缺 BPE / LoRA / DPO） |
| GPT-2 对照实现 | `gpt-learning/gpt-learning-pytorch/model/gpt2.py` |
| 收班自查 | `clean-state-checklist.md` |

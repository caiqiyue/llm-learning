# Session Handoff — llm-learning

> **本卡用途**：把项目交给「下一阶段（Session 006+）」的接手者，主题是 **P2 — Qwen3 从零实现**。
> **context 用满（≥70%）或长 session 结束前必须更新本卡**（见 `AGENTS.md` §9）。
> 规则：用指针代替贴全文；不带密钥；一次 session 只关心当前任务。

---

## 一句话现状

GPT-2 线**全链路已复验通过**（torch 环境已修好，训练 + 9 项评估全绿）；
Qwen3 线的**组件层 + 数据层 + 预训练 + LoRA 已从零手写完并真跑通**
（RMSNorm / RoPE / KVCache / QK-Norm+GQA 注意力 / SwiGLU / Qwen3Dense / Qwen3MoE / byte-level BPE /
语料与偏好数据 / `--model dense|moe` 预训练 / 手写 LoRA 微调），**DPO / 评估侧尚未开工**。

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

- `bash init.sh` → **PASS=24 SKIP=0 FAIL=0**（2026-10-22，L4 已有四条：组件 smoke / 前向 smoke / **训练步 smoke** / **LoRA smoke**）
- `cd qwen3-learning/qwen3-pytorch && python tests/run_all.py` → **PASS=11 FAIL=0 TOTAL=11**
- Qwen3 前向 smoke 已用**真实 BPE 词表**（vocab 778）跑通：`logits (2, 8, 778) loss 6.6897 params 985,856`
- Qwen3 训练步 smoke：`[train-step] loss 5.4788 grads 46 个 lr0 6.00e-06 lr_end 3.00e-05`
- **预训练真跑通**（CPU，脚本在 `train/train_pretrain.py`，超参全从 `config.py` 读）：
  - `--model dense`：`params = 985,856`，`initial_loss=6.7154 → val_loss=1.3510 val_ppl=3.86`，46.8s
  - `--model moe`：`params = 1,184,512`，`initial_loss=6.6954 → val_loss=1.3607 val_ppl=3.90`，59.1s，
    `aux_loss 1.0253 → 0.9982`，`expert_load` 从 ±35% 偏差被拉平到 ±3.5%
- BPE 端到端：语料 20000 句 / 785065 字节 → 词表 778（521 次合并）→ **压缩比 7.10 字节/token**
- 语料同源已证明：`data/corpus.py` 是唯一真源，重建词表后 `vocab.json` / `merges.txt` **逐字节一致**
- **LoRA 真跑通**（CPU，`train/train_lora.py`）：
  - `trainable params = 8,704 / 994,560 (0.88%)`（基座 985,856 全冻结，只训 A/B）
  - `BEFORE win_rate=0.515 win_rate_per_token=0.400 mean_logp(chosen)=-603.26`
    → `AFTER win_rate=0.595 win_rate_per_token=0.745 mean_logp(chosen)=-389.48`
  - 「注入即无损」实测成立：注入前后 logits `torch.allclose(atol=1e-6)`
  - `tests/test_lora.py` 9 组断言全过 → `test_lora OK  replaced=8 r=4 alpha=8 scaling=2.0`
- **消融结论**：低秩约束 + 注入点决定「能学到什么」——
  全参微调 loss ~1.06 vs r=4 LoRA loss ~5.7（**内容型任务上 LoRA 明显吃力**，
  LoRA 的主场是风格/格式型任务）
- **偏好指标的长度偏置已量化**：`too_short` 策略（rejected 6.3 token vs chosen 72 token）
  的总 logprob 胜率**恒为 0.000**，把整体胜率天花板压到 ~0.75
  ⇒ **只看整体胜率会误判「模型没进步」**；这也正是 DPO 减 π_ref 的原因
- GPT-2 线：`python evaluate.py --stage all` 退出码 0，9 项评估全过
- 各 feature 的详细证据在 `feature_list.json` 的 `evidence` 数组里（feat-011 ~ feat-023 都有）

## Broken Or Unverified（坏的 / 没验证的）

- **Qwen3 的 DPO / 评估侧完全没跑过**：`train/` 下只有 `train_pretrain.py` / `train_lora.py`，其余待写（plan Task 15–16）
- **LoRA 的胜率是「训练内指标」**：合成偏好数据只有 5 个 prompt × 5 个 chosen，训练集与评估集同分布 ——
  它证明的是「LoRA 确实让模型更偏向 chosen」，**不能当泛化能力**
- **BPE 词表只有 778 < 目标 1000**：合成语料的可合并对用尽（521 次后没有频次 ≥ 2 的对），属语料多样性问题，非实现 bug
- **MoE 在本仓库体现不出容量优势**：val_loss 1.3607 vs dense 1.3510（MoE 反而略差）——
  语料太简单 + 专家更窄（参数量只多 1.21×）+ 只训 300 步。**写笔记时别暗示「MoE 一定更好」**（前提条件见笔记 §25.1）
- **笔记缺口**：`Qwen3_BPE_学习笔记.md` 仍缺；
  `Qwen3_LoRA与DPO_学习笔记.md` **已建且 LoRA 半部（§1–§9）已写**，但 **DPO 半部（§8）待 feat-024 补**
  （按计划在 feat-008 / Task 19 做全量交叉引用）
- **本仓库的 aux_loss 与 HF 数值不能直接比大小**：本实现 = HF Mixtral 同款公式的 1/K（见下面的陷阱表）

## Next Best Step（按顺序，不要跳）

| 顺序 | feature | 做什么 | 为什么是它 |
|------|---------|--------|--------|
| 1 | `feat-024` | 手写 DPO（**π_θ = LoRA 分支开着 / π_ref = LoRA 分支关着，同一个模型对象**，不训 Reward Model、不做 PPO） | 用户明确要求 DPO |
| 2 | `feat-025` | Qwen3 精简评估（pretrain/lora/dpo/speedup 四段） | 验收 |
| 3 | `feat-018` | KV Cache 推理 + 采样（temperature/top-k/top-p） | 推理侧 |
| 4 | `feat-017` | 与 GPT-2 的**退化对照**（逐组件消融 + max_abs_diff 记录） | 本线最有价值的产出 |
| 5 | `feat-008` | 笔记 ↔ 源码交叉引用（补 BPE 笔记 + LoRA-DPO 笔记的 DPO 半部） | 收尾 |
| 6 | `feat-009` | 笔记数字与 `config.py` 全量核对 | 收尾 |

**这一步不许动的**：不改 GPT-2 线的任何训练代码或超参；不在 `qwen3-learning/qwen3-pytorch/` 之外 import；不削弱 `evaluate.py` 的任何指标与阈值。

---

## 已知的「文字断言 vs 可跑实验」陷阱（别再踩）

| 结论 | 关键点 |
|------|--------|
| QK-Norm 在 RoPE 前 vs 后 | **用均匀 gain 测不出差异**（max\|diff\| ≈ 3.6e-07）。`rotate_half` 把第 i 维与 i+half 维配对，逐维 gain 与旋转不对易。必须用**非均匀 gain** 才暴露（7.38 ~ 14.9） |
| RoPE 是相对位置编码 | 绝对位置**整体平移**不改变输出（max\|diff\| < 1e-5）；只有改变**相对距离**才改变输出 |
| SwiGLU vs GPT-2 4×-GELU 参数量 | 差 = `矩阵数 × H × I`。I 严格取 8/3·H 时**完全相等**；本仓库取整到 384 → 比值 9/8 = 1.125×。Qwen3-0.6B 实际用 3×hidden，8/3 是 LLaMA 的规则 |
| MoE aux_loss | Switch 论文版均匀时 = 1；**本仓库版均匀时也 = 1**（f 对 tokens×k 槽位求均值、Σf=1）；HF Mixtral 版把槽位计数除以 N（Σf=K），均匀时 = **K**（K=2 → 2）。**本实现 = HF 的 1/K**。⚠️ 曾把这条写成「1/K = 0.5」，已被预训练日志（aux≈1.0）与 `tests/test_moe.py` 第 11 节实测纠正 |
| BPE 的 id 分配 | 合并 token 从 **257** 开始（不是 256 —— 256 是字节 0xFF 的位置，撞车会把它静默吞掉） |
| 量纲类结论 | **必须用代码实测锚定**，不能公式心算后直接写进文档（aux_loss 就是这么错了一倍） |
| Windows 子进程输出编码 | bash 按 UTF-8 解码、Windows Python 默认 stdout 是 GBK → 中文乱码。证据命令加 `PYTHONIOENCODING=utf-8` |
| 脚本放在子目录 | Qwen3 线脚本在 `train/` / `tests/` 里，`sys.path[0]` 不是项目根 → **必须显式 `sys.path.insert(0, <项目根>)`** |
| **偏好胜率的长度偏置** | 总 logprob 口径偏袒**短回复**（`too_short` 策略胜率**恒为 0.000**，因为 rejected 只有 6.3 token），把整体胜率天花板压到 ~0.75；长度归一口径反过来偏袒「可预测」的文本。**必须按策略分组看，否则误判「没进步」** |
| **旁路注入的「结构性 bug」** | `merge_lora` 合并后分支还开着 → ΔW 算两遍；`inject_lora` 只冻被包装的那一个 Linear → embedding/Norm/MLP/lm_head 仍可训；loss 掩码把 prompt 也算进去。**三条都不报错、只静默给错结论 ⇒ 必须用「等价性断言」守，不能靠看 loss 曲线** |
| **LoRA ≠ 弱化版全参** | 低秩约束 + 注入点决定「能学到什么」：内容型任务上全参 loss ~1.06 vs r=4 LoRA ~5.7。**LoRA 是风格/格式型任务的主场** |

---

## 需要用户拍板的决策（别自行决定）

| 决策 | 出现在 | 说明 |
|------|--------|------|
| ~~LoRA 的注入点与超参~~ | `feat-023` | **已定**：`lora_r=4 / lora_alpha=8 / lora_target_modules=("q_proj","v_proj")`；`lora_lr` 3e-4→**1e-3**、`lora_max_iters` 300→**1500**（消融实测依据） |
| DPO 的 β 与训练步数 | `feat-024` | `config.py` 已预留 `dpo_beta=0.1 / dpo_lr=1e-4` |
| DPO 偏好数据要不要引入真实数据集 | `feat-024` | 目前是合成数据，引入真实数据会动依赖上限 |
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

# Qwen3 线（组件 + 数据 + 预训练已就位）
cd qwen3-learning/qwen3-pytorch
python data/build_vocab.py       # 重建 BPE 词表（vocab.json / merges.txt）
python tests/run_all.py          # 11 个 smoke 汇总
python tests/test_pretrain.py    # 单个 smoke
python tests/test_lora.py        # LoRA 的 9 组断言
python train/train_pretrain.py --model dense   # 预训练（dense，~47s CPU）
python train/train_pretrain.py --model moe     # 预训练（MoE，~59s CPU）
python train/train_lora.py       # LoRA 微调（r=4 q/v, lr=1e-3, 1500 步）

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
| Qwen3 设计 / 实施计划 | `qwen3-learning/qwen3-pytorch设计.md`、`qwen3-learning/qwen3-pytorch开发计划.md`（Task 7–14 已就地纠错，含「实测修正记录」） |
| Qwen3 笔记原文 | `qwen3-learning/Qwen3_*.md`（8 篇，缺 BPE；LoRA-DPO 篇缺 DPO 半部） |
| GPT-2 对照实现 | `gpt-learning/gpt-learning-pytorch/model/gpt2.py` |
| 收班自查 | `clean-state-checklist.md` |

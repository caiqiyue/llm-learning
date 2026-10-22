# qwen3-pytorch

从零手写 tiny Qwen3：RoPE + RMSNorm + QK-Norm + SwiGLU + GQA，外加 MoE 变体，
走通 预训练 → LoRA → DPO 三段。

设计文档：[../qwen3-pytorch设计.md](../qwen3-pytorch设计.md) ｜ 开发计划：[../qwen3-pytorch开发计划.md](../qwen3-pytorch开发计划.md)

## 快速开始

```bash
python tests/run_all.py            # 全部组件 smoke
python data/build_vocab.py         # 训练 byte-level BPE，生成 vocab.json / merges.txt
python train/train_pretrain.py --model dense
python train/train_pretrain.py --model moe
python train/train_lora.py
python train/train_dpo.py
python evaluate.py --stage all
```

## 概念 → 笔记 → 源码

| 概念 | 笔记 | 源码 |
|---|---|---|
| RMSNorm | [RMSNorm_学习笔记](../Qwen3_RMSNorm_学习笔记.md) | `model/common/rmsnorm.py` |
| RoPE | [RoPE_学习笔记](../Qwen3_RoPE_学习笔记.md) | `model/common/rope.py` |
| QK-Norm | [QK-Norm_学习笔记](../Qwen3_QK-Norm_学习笔记.md) | `model/common/attention.py` |
| GQA / KV Cache | [GQA与KV_Cache](../Qwen3_GQA与KV_Cache_完整学习笔记.md) | `model/common/attention.py`、`model/common/kv_cache.py` |
| SwiGLU | [SwiGLU_MLP_学习笔记](../Qwen3_SwiGLU_MLP_学习笔记.md) | `model/qwen3_dense.py` |
| MoE | [MoE_学习笔记](../Qwen3_MoE_学习笔记.md) | `model/qwen3_moe.py` |
| 训练 → 推理 | [训练推理完整学习笔记](../Qwen3_训练推理完整学习笔记.md) | `train/`、`inference.py` |

## 与 GPT-2 线的对照

本线不跨目录 import GPT-2 线代码，但为对照实验保留了相同接口：
`data/pretrain_data.py::make_pretrain_loader` 与 `data/preference_data.py::generate_preference_dataset`
的签名与 GPT-2 线一致，见 `ablations/compare_with_gpt2.py`。

### 评估指标：与 GPT-2 线 9 项评估的差异（feat-025 的验收项）

GPT-2 线的 `evaluate.py` 有 9 个评估函数（8 个 `--stage` + `diversity` 里打包的 4 个），
整套是围绕 **Reward Model + PPO** 设计的。
Qwen3 线走的是 **LoRA + DPO**：没有显式奖励模型、没有 PPO、没有采样。
⇒ **两条线的评估不做逐项对标**，但 **pretrain 与 speedup 两项保留与 GPT-2 线逐字同名的列**。

| GPT-2 线 | GPT-2 的评估函数 | Qwen3 线 | 说明 |
|---|---|---|---|
| `pretrain` | `eval_perplexity` | ✅ `pretrain` | **同名列**：`stage / metric / value / loss / perplexity / n_tokens / checkpoint`；口径也相同（val set 的 token 加权平均 cross_entropy） |
| `speedup` | `eval_speedup` | ✅ `speedup` | **同名列**：`stage / metric / value / naive_avg_ms / kv_avg_ms / speedup / naive_tokens / kv_tokens / n_runs / max_new_tokens` |
| `sft` | `eval_sft` | 🔁 `lora` | 对应物：LoRA 前后偏好胜率对比（`baseline → win_rate`）。列名不通用（SFT 与 LoRA 是不同的训练方式） |
| `reward` | `eval_pairwise_accuracy`（RM） | 🔁 `dpo` | **本线没有 RM**。对应物是 DPO 的 **margin 判对率** `accuracy`（`margin > 0` 的比例） |
| `ppo` | `eval_ppo`（reward / KL） | 🔁 `dpo` | **本线没有 PPO**。对应物是 DPO 的**隐式奖励 margin**（`r_θ(y) = β(logπ_θ − logπ_ref)`） |
| `compare` | `eval_sft_vs_ppo` | 🔁 `lora` | **本线没有 PPO**。对应物是 LoRA 段里的「注入前（= 预训练模型）→ 训练后」胜率对比 |
| `benchmark` | `eval_benchmark`（QA 准确率） | ❌ 无 | 本线语料是**合成语料**，没有 QA benchmark 可做 |
| `diversity` | `eval_distinct_n` / `eval_self_bleu` / `eval_token_entropy` | ⏳ 未实现 | 属「生成质量」类，本线**尚未实现**（feat-025 的范围是精简评估，不是复制整套生成质量指标） |
| `response_quality` | `eval_response_quality` | ⏳ 未实现 | 同上 |
| — | — | ➕ `moe` | **Qwen3 独有**：专家负载分布 + `aux_loss`（GPT-2 线没有 MoE） |

**结论一句话**：GPT-2 线的评估是「**生成质量 + RLHF 三件套**」；
Qwen3 线的评估是「**训练质量（loss/ppl）+ 偏好对齐（margin/胜率）+ 推理效率（speedup）**」。
两条线只在 **pretrain** 与 **speedup** 两项上能直接并排比数字。

### 评估脚本的「不删/不改弱门槛」

`evaluate.py` 只设**可自证的门槛**，达不到就**报错退出**而不是打一个好看的数：

| 段 | 门槛 |
|---|---|
| `pretrain` / `moe` | `perplexity == exp(loss)` 自洽；`n_tokens > 0`；`aux_loss` 有限；专家负载之和 == `B*T` |
| `lora` | 胜率 ∈ [0, 1] |
| `dpo` | checkpoint 的 LoRA `B` 必须非零（B 全零 ⇒ DPO 没训过）；`margin_std >= 0`；准确率 ∈ [0, 1] |
| `speedup` | **有/无 KV Cache 的 greedy 输出必须逐位相同**，不一致直接报错退出 |

checkpoint 缺失时**跳过该段并打印原因**（那是「还没跑到那一步」，不是「指标不合格」）。

## 硬约束

- 依赖上限：`torch` / `numpy` / `nltk`，不引入 pytest / peft / transformers / sentencepiece
- 超参全部来自 `config.py`
- `checkpoints/` 不入库

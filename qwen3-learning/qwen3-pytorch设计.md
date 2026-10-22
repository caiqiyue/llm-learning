# qwen3-pytorch 设计文档

> 本文件是 `qwen3-learning/qwen3-pytorch/` 的设计说明，兼作该目录的学习入口。
> 与仓库其他设计文档冲突时，以本文件为准；超参数的唯一真值永远是 `qwen3-pytorch/config.py`。
>
> 状态：**已批准，待实施** ｜ 批准日期：2026-10-19

---

## 1. 背景与目标

仓库已有一条 GPT-2 线（`gpt-learning/gpt-learning-pytorch/`）：从零手写 tiny GPT-2，走通预训练 → SFT → Reward Model → PPO 四阶段 + 9 项评估。Qwen3 线延续同一套「笔记 ↔ 代码对照」的做法，但把架构换成 Qwen3 的现代组件，并改走 **LoRA + DPO** 这条微调路线。

Qwen3 线的核心产出不是「训得动」，而是 **把 Qwen3 相对 GPT-2 的每一个架构改动都变成一次可运行的实验**，而不是文字断言。

---

## 2. 范围

### 做

| 项 | 说明 |
|---|---|
| Dense Qwen3 | 完整手写：RoPE + RMSNorm + QK-Norm + SwiGLU + GQA |
| MoE Qwen3 | 完整手写：Dense 主干不变，FFN 换成 Router + 专家 |
| 预训练 | dense 与 MoE 各跑通一次，CPU 可训 |
| LoRA | 从零手写，微调 dense |
| DPO | 从零手写，微调 dense，不训 Reward Model |
| byte-level BPE | 从零手写 tokenizer（GPT-2 / Qwen 同款思路） |
| 精简评估 | `evaluate.py`：val loss/ppl、LoRA 胜率、DPO reward margin |
| KV Cache 推理 | prefill / decode 两阶段 + temperature/top-k/top-p |
| 退化对照 | 把 Qwen3 组件开关关掉后与 GPT-2 逐层对齐（feat-017 保留） |

### 不做

| 项 | 理由 |
|---|---|
| 全参 SFT | 路线已定为 LoRA；全参 SFT 在 GPT-2 线已有覆盖 |
| Reward Model + PPO | 路线已定为 DPO |
| MoE 的 LoRA / DPO | 范围收敛；MoE 只验证到预训练 |
| transformers / peft / sentencepiece / 任何新依赖 | 项目硬约束：依赖上限 `torch` / `numpy` / `nltk` |
| 分布式训练 / FSDP / FlashAttention | CPU 单机 tiny 模型 |
| MoE 的 expert 并行、all-to-all | 同上 |

> **约束确认**：手写 LoRA、手写 BPE、DPO 手写三者合计**不需要引入任何新包**，项目依赖上限无需破例。

---

## 3. 目录结构

```
qwen3-learning/qwen3-pytorch/
├── config.py                 # DENSE_CONFIG / MOE_CONFIG —— 超参唯一真值
├── README.md                 # 学习入口：概念 ↔ 笔记 ↔ 源码位置
├── data/
│   ├── bpe.py                # 从零手写 byte-level BPE：训 merge 表 + 编解码
│   ├── build_vocab.py        # 训练并落盘 vocab.json / merges.txt
│   ├── tokenizer.py          # BPETokenizer，接口对齐 GPT-2 线 tokenizer.py
│   ├── pretrain_data.py      # 预训练样本构造
│   └── preference_data.py    # DPO 的 chosen / rejected 偏好对
├── model/
│   ├── common/               # 两个模型唯一共享的部分
│   │   ├── rmsnorm.py        # 无 bias、无均值中心化、带可学习 gain
│   │   ├── rope.py           # theta 可配，支持 position_ids（增量解码预留）
│   │   ├── kv_cache.py       # prefill / decode 两态缓存
│   │   ├── attention.py      # QK-Norm + GQA + KV Cache
│   │   └── lora.py           # LoRALinear，从零手写
│   ├── qwen3_dense.py        # FFN = SwiGLU
│   └── qwen3_moe.py          # FFN = Router top-k + 专家
├── train/
│   ├── train_pretrain.py     # --model dense|moe
│   ├── train_lora.py         # 只 dense
│   └── train_dpo.py          # 只 dense
├── inference.py              # prefill/decode + temperature/top-k/top-p
├── evaluate.py               # 精简评估
└── ablations/
    └── compare_with_gpt2.py  # feat-017：退化对照实验
```

### 设计要点

**两个模型类的差别只在 FFN 一处。** attention / norm / rope / KV Cache 全部走 `model/common/`。这样 GQA、QK-Norm 这类难点的知识不重复，MoE 的差异干净地隔离在一个位置。

代码风格对齐 `gpt-learning-pytorch/model/gpt2.py`：中文 docstring + type hint，注释解释「为什么这么写」而不是「这行在做什么」。

---

## 4. 架构超参

tiny 尺寸，CPU 可训。以下是 `config.py` 的**初始值**，实施时以该文件为准。

### Dense

| 参数 | 值 | 说明 |
|---|---|---|
| `n_layer` | 4 | |
| `n_embd` | 128 | 残差流宽度 |
| `n_head` | 4 | Q head 数 |
| `n_head_kv` | 2 | GQA：KV head 数，为 Q head 的 1/2 |
| `head_dim` | 48 | **与 n_embd 解耦**：4 × 48 = 192 ≠ 128 |
| `intermediate_size` | 384 | `128 × 8/3 = 341.3` → 对齐到 128 的倍数；也正好 = 3 × `n_embd`，与 Qwen3-0.6B 的 3×hidden 取法一致。**8/3 规则来自 LLaMA，Qwen3 实际用 3×** |
| `rope_theta` | 1e6 | Qwen3 默认；真实值以官方 `config.json` 为准 |
| `rms_eps` | 1e-6 | |
| `block_size` | 128 | |
| `tie_embeddings` | True | 权重绑定，代码与注释中显式写明策略 |
| `vocab_size` | 由 BPE 训练结果决定 | |
| `attention_bias` | False | Qwen3 全线无 bias |

> `head_dim=48` 而非 `n_embd/n_head=32`，是为了**在代码里真实体现 head_dim 与 n_embd 解耦**这件事——这是 Qwen3 相对 GPT-2 的一个实际差异点（GPT-2 强制 `n_embd % n_head == 0`）。

### MoE

在 Dense 基础上替换 FFN：

| 参数 | 值 | 说明 |
|---|---|---|
| `n_experts` | 4 | |
| `experts_per_tok` | 2 | top-k 路由 |
| `moe_intermediate_size` | 128 | 每个专家的 FFN 中间维 |
| `norm_topk_prob` | True | 对 top-k 权重归一化，匹配 Qwen3-MoE |

---

## 5. 组件规格

### 5.1 `model/common/rmsnorm.py`

`RMSNorm(d, eps)`：`x / sqrt(mean(x²) + eps) * gain`。

- 无 bias
- 不减均值（与 LayerNorm 的核心差异，注释里写明）
- gain 是 `nn.Parameter`（可学习，形状 `[d]`）
- `forward(x) -> Tensor`，支持任意前置维度

**smoke**：随机输入归一化后，均值≈0、方差≈1；`rms_eps` 在极小输入下不产生 NaN。

### 5.2 `model/common/rope.py`

`RoPE(head_dim, max_seq_len, theta)`。

- 预计算 `inv_freq` / `cos` / `sin` 缓存，形状 `(max_seq_len, head_dim)`
- `forward(q, k, position_ids=None) -> (q_rot, k_rot)`
- `position_ids` 参数**为 KV Cache 增量解码预留**：`None` 表示 `0..T-1`，显式传入时按给定位置取缓存
- `theta` 可配，默认 1e6

**smoke**：`head_dim=8` 时，任意两对相对距离相同的 `(m, n)`，旋转后 `q_m · k_n` 相等（旋转等变性），误差 < 1e-5。

### 5.3 `model/common/kv_cache.py`

`KVCache` 类，两态：

- **prefill**：`append(k, v)` 写入长度为 `T` 的新块
- **decode**：每步只传入新 token 的 `(k, v)`，形状 `(B, n_head_kv, 1, head_dim)`，沿序列维拼接

按 **KV head 数**分配，不是 Q head 数。

**smoke**：`n_head=8 / n_head_kv=2` 时，缓存占用为同尺寸 MHA 的 1/4；不同 batch 元素的缓存互不污染。

### 5.4 `model/common/attention.py`

`CausalSelfAttention(n_embd, n_head, n_head_kv, head_dim, ...)`。

流程（顺序即 QK-Norm 的位置，**QK-Norm 在 RoPE 之前**，与 `Qwen3_QK-Norm_学习笔记.md` 一致）：

```
x → q_proj / k_proj / v_proj (no bias)
  → 拆 head: (B, n_head, T, head_dim)
  → q_norm / k_norm   ← QK-Norm，对每个 head 的 head_dim 做 RMSNorm
  → RoPE 旋转
  → GQA: KV 沿 head 维 repeat_interleave 到 n_head
  → QKᵀ/√head_dim → Causal Mask → Softmax → ×V
  → o_proj
```

- KV Cache 读写由本模块调用 `KVCache` 完成
- `forward(x, kv_cache=None, position_ids=None) -> (out, new_kv_cache)`

**smoke**：`n_head=8 / n_head_kv=2` 前向输出 `(B, T, n_embd)`；形状在 head_dim 解耦下正确。

### 5.5 `model/common/lora.py`

`LoRALinear(base: nn.Linear, r, alpha, dropout=0.0)`。

- 前向：`out = base(x) + (alpha / r) * dropout(x) @ A.T @ B.T`
- `A` 形状 `(r, in_features)`，Kaiming 初始化；`B` 形状 `(out_features, r)`，**零初始化**（保证初始时 LoRA 分支输出恒为 0）
- `base.weight` / `base.bias` 全部 `requires_grad = False`
- 提供 `inject_lora(model, target_modules, r, alpha)`：按模块名后缀（如 `q_proj`、`v_proj`）递归替换
- 提供 `merge_lora(model)`：把 `B @ A * (alpha/r)` 加回原权重，便于导出

**smoke**：注入前后的模型输出在 `B=0` 时**逐元素相等**；只有 A/B 的梯度非 None。

### 5.6 `model/qwen3_dense.py`

`Qwen3Dense(config)`，结构：

```
wte  →  N × Pre-LN Block  →  final RMSNorm  →  lm_head (tied)
Block = x + Attn(RMSNorm(x))  →  x + SwiGLU(RMSNorm(x))
```

`SwiGLU(x) = down( silu(gate(x)) * up(x) )`，三个矩阵均无 bias。

`forward(idx, targets=None, kv_cache=None, position_ids=None) -> (logits, loss, new_kv_cache, extras)`

- dense 的 `extras = {}`
- 有 `targets` 时返回交叉熵 loss（与 GPT-2 线一致）
- 提供 `num_params()` 统计，打印参数量并与笔记中的规模描述对得上

### 5.7 `model/qwen3_moe.py`

`Qwen3MoE(config)`，**主干与 Dense 完全一致**（导入 `common/` 的同一批模块），只有 FFN 换成 `SparseMoeBlock`：

```
Router (无 bias 线性) → softmax over n_experts → top-k
  → 若 norm_topk_prob 则对选中的 k 个权重重新归一化
  → 对每个 token 只过被选中的 k 个专家，其余专家不计算（节省计算）
  → 按 token_id 把专家输出加权求和回原位
```

负载均衡辅助损失（Switch Transformer 风格）：

```
aux_loss = n_experts * Σ_i  f_i * P_i
  f_i = 本 batch 中被路由到专家 i 的 token 比例
  P_i = Router 对专家 i 的平均概率
```

`forward` 返回同 Dense 的四元组，其中 `extras = {"aux_loss": Tensor, "expert_load": List[float]}`。

**smoke**：`dense` 与 `moe` 的 `forward` 签名与返回结构一致；MoE 的 `aux_loss` 非 None；专家负载分布打印后无专家饿死（每个专家分到 >0 token）；总 loss = `lm_loss + aux_loss_coef * aux_loss`。

### 5.8 统一接口约定

两个模型都返回 `(logits, loss, kv_cache, extras)`，差异**只在 `extras`**：

| 模型 | `extras` |
|---|---|
| `Qwen3Dense` | `{}` |
| `Qwen3MoE` | `{"aux_loss": ..., "expert_load": [...]}` |

训练脚本因此**不需要 `if is_moe` 分支去猜返回结构**，只需读 `extras.get("aux_loss")`。

---

## 6. 数据层

### 6.1 byte-level BPE（`data/bpe.py`，从零手写）

训练流程：

1. 文本 → UTF-8 字节 → 基础 token 集合（256 个字节值）
2. 统计所有相邻 token 对的出现频次
3. 取频次最高的对（并列时取字典序小的，保证确定性）合并为新 token
4. 重复直到 `vocab_size` 达到目标

接口：

- `train_bpe(texts, vocab_size) -> (vocab: dict, merges: list)`
- `encode(text, merges) -> list[int]`
- `decode(ids, vocab) -> str`
- merge 表落盘为 `merges.txt`（每行 `a b`），词表落盘为 `vocab.json`

**smoke**：随机中文/英文/代码片段编码后解码**逐字节还原**；`vocab_size` 达到目标时实际 merge 次数 = `vocab_size - 256`。

### 6.2 `data/tokenizer.py`

`BPETokenizer`，接口对齐 GPT-2 线 `data/tokenizer.py` 的 `CharTokenizer`（`encode` / `decode` / `vocab_size` / `stoi` / `itos`），使训练脚本写法一致。

> 对照：GPT-2 线是**字符级**（`vocab_size` 约 64），Qwen3 线是 **byte-level BPE**。两套词表**互不共享**，各自 checkpoint 各配各的词表。

同时提供 `<|endoftext|>` 特殊 token，作为文档分隔符。

### 6.3 `data/pretrain_data.py`

- 读取语料 → BPE 编码 → 按 `block_size + 1` 切块（多留 1 个 token 做 label shift）
- 输出 `(inputs, targets)`，供交叉熵使用

### 6.4 `data/preference_data.py`

DPO 需要成对偏好数据：

```python
{
  "prompt":     "...",       # 共享前缀
  "chosen":     "...",       # 更好的回复
  "rejected":   "..."        # 更差的回复
}
```

构造方式：同一 prompt 下，取 SFT 模型的若干次采样，按长度/重复度等启发式规则分成好坏两组。复用 GPT-2 线 `preference_data.py` 的组织方式，**但代码独立重写**，不跨目录 import。

**smoke**：pairwise 数据每条的 prompt 非空，chosen ≠ rejected。

---

## 7. 训练链路

### 7.1 `train/train_pretrain.py --model dense|moe`

唯一带 `--model` 开关的脚本。

- AdamW + cosine 或 warmup 学习率调度
- 每 N 步打印 train loss / val loss / val ppl
- MoE 额外打印专家负载分布与 aux_loss
- 落盘 `checkpoints/pretrain_{model}.pt`（`checkpoints/` 不入库）

### 7.2 `train/train_lora.py`（只 dense）

1. 加载 `pretrain_dense.pt`
2. `inject_lora(model, ["q_proj", "v_proj"], r=4, alpha=8)`
3. 冻结 base，只训 A/B
4. 在 pairwise 数据的 prompt 上做**自监督式**微调：给定 prompt，训练模型补全 chosen
5. 评估：偏好胜率（chosen 的 logprob 高于 rejected 的样本比例）

**明确不做**：全参 SFT。

### 7.3 `train/train_dpo.py`（只 dense）

DPO 目标函数：

```
L_DPO = -E[ log σ( β * ( (logπ_θ(y_w|x) - logπ_ref(y_w|x))
                       - (logπ_θ(y_l|x) - logπ_ref(y_l|x)) ) ) ]
```

其中 `β` 为 KL 惩罚系数（默认 0.1，config 可配）。

**reference model 的取法**：**不单独加载一份冻结模型**，而是复用同一个 model 对象，把 LoRA 分支关闭即得 `π_ref`。这既省一份显存与一份 checkpoint，也把这个技巧变成可读的代码。该技巧单独写进笔记。

流程：

1. 加载 LoRA 微调后的 dense 模型
2. 同一模型跑两遍：开启 LoRA 得 `logπ_θ`，关闭 LoRA 得 `logπ_ref`
3. 计算 DPO loss，只回传开启分支的梯度
4. 评估：reward margin 分布、chosen>rejected 准确率

---

## 8. 评估指标（`evaluate.py`）

精简版，对应三条链路：

| 阶段 | 指标 |
|---|---|
| 预训练 | val loss、val perplexity |
| MoE 预训练 | 同上 + 专家负载分布 + aux_loss |
| LoRA | 偏好数据胜率（chosen logprob > rejected 的比例） |
| DPO | reward margin 均值/标准差、chosen>rejected 准确率 |
| 推理 | 有/无 KV Cache 的输出一致性、加速比 |

用法：`python evaluate.py --stage {pretrain,lora,dpo,all}`

> 与 GPT-2 线的 9 项评估**不做逐项对标**（那套是围绕 RM+PPO 设计的，DPO 阶段不适用），但 pretrain 与 speedup 两项**保留同名列**，便于横向对比。

---

## 9. 验证策略

### 每个组件的可跑 smoke

见 §5 各模块的 smoke 条目，全部挂进 `init.sh` 的 **L4 层**（`init.sh` 已预留该层，当前因目录未创建而 SKIP）。

L4 层按 feature 完成度逐步加重：

- feat-011 落地 → 检查目录与 `config.py` 齐全
- feat-012 → RMSNorm smoke
- feat-013 → RoPE 旋转等变性 smoke
- feat-014 → GQA 形状与 KV 缓存 1/4 smoke
- feat-015 → SwiGLU 逐元素一致性 smoke
- feat-016 → 完整前向 + 参数量 smoke
- feat-017 → 退化对照 `max_abs_diff < 1e-4`
- BPE 落地 → 编解码往返 smoke
- LoRA → 注入前后输出一致 smoke
- DPO → aux_loss / margin 非 None smoke

### 退化对照实验（feat-017，保留）—— **已完成**

目的：把「Qwen3 相对 GPT-2 改了什么」变成实证。

构造退化开关：`use_gqa=False` / `use_rope=False` / `use_qk_norm=False` / `mlp_type='gelu'` /
`norm_type='layernorm'`。开关全部关闭后，Qwen3 与 GPT-2 在同样输入、同权重加载下
`max_abs_diff < 1e-4`（fp32）。

**实测（2026-10-22，`ablations/compare_with_gpt2.py`）**：

| 配置 | max_abs_diff | 相对差 |
|---|---|---|
| 全关（退化成 GPT-2） | **0.000e+00**（逐位相同） | 0 |
| 只开 `use_gqa` | 7.514e-02 | 1.01e-01 |
| 只开 `use_rope` | 1.541e+00 | 2.08e+00 |
| 只开 `use_qk_norm` | 9.334e-03 | 1.26e-02 |
| 只开 `mlp_type=swiglu` | 7.187e-02 | 9.68e-02 |
| 只开 `norm_type=rmsnorm` | 5.449e-01 | 7.34e-01 |

三个**前提**（少一个就对不齐，详见笔记 §12.2）：
① `head_dim` 必须回退成 `n_embd // n_head`（主线是解耦的 48）；
② 中间层必须回退成 GPT-2 的 `n_inner = 4·n_embd`（主线是 8/3 规则的 384）；
③ 归一化必须**两边同步**切到 LayerNorm，且 eps 取 GPT-2 默认的 `1e-5`。

结论已写进 `Qwen3_训练推理完整学习笔记.md` §12，并在 5 篇组件笔记开头加了指针。

---

## 10. `feature_list.json` 改动清单

| 改动 | 内容 |
|---|---|
| 保留 | `feat-011` ~ `feat-018`（骨架 / RMSNorm / RoPE / QK-Norm+GQA / SwiGLU / 前向 / 退化对照 / KV Cache） |
| `feat-021` | MoE 从「P3 可选」升为 **P2 必做**（`priority` 前移） |
| `feat-003` | 全参 SFT → **改为 deferred**，理由：路线改为 LoRA |
| `feat-004` / `feat-005` | RM+PPO → **改为 deferred**，理由：路线改为 DPO |
| 新增 | **手写 byte-level BPE tokenizer**（`feat-022`，优先级排在 `feat-019` 之前） |
| `feat-019` | 「数据与 tokenizer 方案」被 **feat-022 取代** → 状态 `deferred`，理由：tokenizer 选型已由 feat-022 直接落地，只剩「预训练语料选型」这一小段未做，并入 `feat-020` |
| 新增 | **手写 LoRA 微调**（`feat-023`） |
| 新增 | **DPO 训练**（`feat-024`） |
| `feat-001` | 状态 `blocked` → `passing`（torch 2.12.0+cpu 已可正常导入，`init.sh` L2 已转 PASS） |
| `init.sh` | L4 层逐 feature 加重 |

新增 feature 的顺序遵循「先能跑通最短闭环，再加专项」：
`feat-011` 骨架 → `feat-012` RMSNorm → `feat-013` RoPE → `feat-014` QK-Norm/GQA → `feat-015` SwiGLU → `feat-016` 前向 → BPE → `feat-020` 预训练 → `feat-021` MoE → `feat-023` LoRA → `feat-024` DPO → `feat-017` 对照 → `feat-018` KV Cache → `feat-008` 笔记交叉引用

---

## 11. 风险与已知约束

| 风险 | 缓解 |
|---|---|
| CPU 上训练慢 | tiny 尺寸（n_layer=4, n_embd=128, block_size=128），单次预训练控制在分钟级 |
| byte-level BPE 训练慢（O(n·merges)） | 语料先截断到合理长度；`vocab_size` 取 1000–4000 量级 |
| DPO 在 tiny 模型上可能不收敛 | 加大 `β` 或降低学习率；接受「指标上升即可」而非「训得很好」 |
| MoE 专家塌缩 | 打印负载分布作为 smoke；必要时提高 `aux_loss_coef` |
| 退化对照难以和 GPT-2 逐层对齐（GPT-2 用 **固定的 sinusoidal PE** + GELU + LayerNorm） | ✅ **已解决**（feat-017）。映射关系在 `compare_with_gpt2.py` 注释中显式列出；实测全关时 `max_abs_diff = 0.000e+00`，比 `1e-4` 的阈值还严 |
| `feat-017` 时间成本高 | 排在训练链路之后，属于「有富余时间再做」 |

---

## 12. 关联笔记

| 笔记 | 对应代码 |
|---|---|
| `Qwen3_RMSNorm_学习笔记.md` | `model/common/rmsnorm.py`、`model/common/norm.py` |
| `Qwen3_RoPE_学习笔记.md` | `model/common/rope.py`、`model/common/pos_emb.py` |
| `Qwen3_QK-Norm_学习笔记.md` | `model/common/attention.py` |
| `Qwen3_GQA与KV_Cache_完整学习笔记.md` | `model/common/attention.py`、`model/common/kv_cache.py`、`inference.py` |
| `Qwen3_SwiGLU_MLP_学习笔记.md` | `model/common/mlp.py` |
| `Qwen3_MoE_学习笔记.md` | `model/qwen3_moe.py` |
| `Qwen3_BPE_学习笔记.md` | `data/bpe.py`、`data/tokenizer.py`、`data/build_vocab.py` |
| `Qwen3_LoRA与DPO_学习笔记.md` | `model/common/lora.py`、`train/train_lora.py`、`train/train_dpo.py` |
| `Qwen3_训练推理完整学习笔记.md` | `train/train_pretrain.py`、`inference.py`、`evaluate.py`、`ablations/compare_with_gpt2.py`（§12 退化对照） |

GPT-2 主线参照：`gpt-learning/gpt-learning-pytorch/model/gpt2.py`

> **每篇笔记开头都有「对应 GPT-2 实现位置 / 对应 Qwen3 源码位置 / 对应可跑测试」三行**（feat-008），
> 链接的可点开性由 `qwen3-learning/qwen3-pytorch/tests/test_note_refs.py` 自动校验 ——
> 链接指向的文件不存在、或某篇笔记漏了引用块，测试直接红。

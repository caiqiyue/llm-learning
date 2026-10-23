# GPT-2 训练数学公式与超参详解

> 本项目的"数学手册"——把所有学习率、优化器、损失函数、评估指标、关键数学公式**集中整理**。
>
> 配套文档：
> - 架构与流程：[`GPT全流程详解.md`](GPT全流程详解.md)
> - 算法详解：[`docs/superpowers/specs/2026-06-09-gpt2-training-explained.md`](docs/superpowers/specs/2026-06-09-gpt2-training-explained.md)
>
> 配套代码：[`gpt-learning-pytorch/`](gpt-learning-pytorch/) + [`gpt-learning-tensorflow/`](gpt-learning-tensorflow/)

---

## 目录

- [第 1 部分：超参与训练配置](#第-1-部分超参与训练配置)
  - [1.1 模型架构超参](#11-模型架构超参)
  - [1.2 4 阶段训练超参汇总](#12-4-阶段训练超参汇总)
  - [1.3 优化器（AdamW）](#13-优化器adamw)
  - [1.4 学习率调度：Warmup + Cosine](#14-学习率调度warmup--cosine)
  - [1.5 梯度裁剪](#15-梯度裁剪)
- [第 2 部分：核心数学函数](#第-2-部分核心数学函数)
  - [2.1 Softmax](#21-softmax)
  - [2.2 GELU 激活](#22-gelu-激活)
  - [2.3 LayerNorm](#23-layernorm)
  - [2.4 Sinusoidal 位置编码](#24-sinusoidal-位置编码)
  - [2.5 Scaled Dot-Product Attention](#25-scaled-dot-product-attention)
- [第 3 部分：4 阶段损失函数](#第-3-部分4-阶段损失函数)
  - [3.1 Pretrain：Causal LM Cross-Entropy](#31-pretraincausal-lm-cross-entropy)
  - [3.2 SFT：Prompt-Masked Loss](#32-sftprompt-masked-loss)
  - [3.3 RM：Bradley-Terry Pairwise Loss](#33-rmbradley-terry-pairwise-loss)
  - [3.4 PPO：Clipped Policy + Value Loss](#34-ppoclipped-policy--value-loss)
- [第 4 部分：PPO 强化学习公式](#第-4-部分ppo-强化学习公式)
  - [4.1 KL 散度近似（k3 估计）](#41-kl-散度近似k3-估计)
  - [4.2 Reward Shaping](#42-reward-shaping)
  - [4.3 Reward Normalization](#43-reward-normalization)
  - [4.4 GAE 广义优势估计](#44-gae-广义优势估计)
  - [4.5 Advantage Normalization（可选）](#45-advantage-normalization可选)
- [第 5 部分：9 项评估指标公式](#第-5-部分9-项评估指标公式)
  - [5.1 Perplexity (PPL)](#51-perplexity-ppl)
  - [5.2 Pairwise Accuracy](#52-pairwise-accuracy)
  - [5.3 Distinct-N](#53-distinct-n)
  - [5.4 Self-BLEU](#54-self-bleu)
  - [5.5 Token Entropy](#55-token-entropy)
  - [5.6 Response Quality](#56-response-quality)
  - [5.7 SFT vs PPO 对比](#57-sft-vs-ppo-对比)
  - [5.8 Benchmark Accuracy](#58-benchmark-accuracy)
  - [5.9 KV Cache Speedup](#59-kv-cache-speedup)
- [第 6 部分：采样策略公式](#第-6-部分采样策略公式)
  - [6.1 Temperature Scaling](#61-temperature-scaling)
  - [6.2 Top-K](#62-top-k)
  - [6.3 Top-P (Nucleus)](#63-top-p-nucleus)
- [附录：关键数字速查表](#附录关键数字速查表)

---

## 第 1 部分：超参与训练配置

### 1.1 模型架构超参

来源：[`gpt-learning-pytorch/config.py`](gpt-learning-pytorch/config.py)

```python
n_layer   = 4        # Transformer Block 层数
n_head    = 4        # 多头注意力的头数（n_embd 必须能整除 n_head）
n_embd    = 128      # 隐藏维度 / token embedding 维度
n_inner   = 512      # FFN 中间层维度（4 × n_embd）
block_size= 128      # 最大上下文长度（位置编码最大位置数）
dropout   = 0.1      # Dropout 比例
bias      = False    # Linear/Conv1D 是否使用 bias（True 还原 GPT-2；False 对齐 nanoGPT）
```

**派生量**：

| 量 | 公式 | 值 |
|---|---|---|
| `head_dim` | `n_embd // n_head` | 32 |
| `vocab_size` | 实际 token 数 | ~45（5 special + 40 字符）|
| 总可训练参数 | 见 [§1.9 of 主文档](GPT全流程详解.md#19-参数量级) | **794,496** |
| 上下文 token 数 | `block_size` | 128 |
| 理论 FLOPs/token | `6 × N_params`（forward）| ~4.8M |

---

### 1.2 4 阶段训练超参汇总

| 阶段 | learning rate | batch size | max_iters | warmup | 起始权重 | 优化器 |
|---|---|---|---|---|---|---|
| **① Pretrain** | `3e-4` | `32` | `200` | `50` | 随机初始化 | AdamW |
| **② SFT** | `3e-4` | `32` | `1000` | `~33`（min(200, max_iters//5)）| `pretrain.pt` | AdamW |
| **③ RM** | `3e-4` | `32` (`rm_batch_size`) | `1000` | `~200`（min(200, max_iters//5)）| `sft.pt` + 新 value_head | AdamW |
| **④ PPO** | `1e-5` (policy) + `1e-5` (value) | `8` rollout + `4` minibatch | `100` | — | `sft.pt` + 新 value_head | AdamW |

**学习率设计哲学**：
- Pretrain 用较大 lr（`3e-4`），从头学
- SFT/RM 与 pretrain 同 lr，但 iter 少 + 起始预训练权重 → 不会灾难性遗忘
- PPO 必须**小 lr**（`1e-5`），因为 RL 训练对步长极敏感

---

### 1.3 优化器（AdamW）

**所有 4 阶段都用 AdamW**：

```python
optimizer = torch.optim.AdamW(
    params,
    lr=lr,                      # 3e-4 或 1e-5
    weight_decay=0.1,           # 权重衰减
    betas=(0.9, 0.95),          # β1, β2
)
```

**AdamW vs Adam 区别**：

| | Adam | AdamW |
|---|---|---|
| 权重衰减 | L2 正则（耦合 loss）| 解耦（直接衰减权重）|
| 公式 | $\mathcal{L}_{total} = \mathcal{L} + \lambda \|\theta\|_2^2$ | $\theta_t \leftarrow \theta_t - \eta (\nabla\mathcal{L} + \lambda \theta_t)$ |
| 效果 | 自适应学习率被 weight_decay 影响 | 干净解耦，效果更好 |

**Adam 公式**（来自 Kingma & Ba 2014）：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**AdamW 改进**（Loshchilov & Hutter 2019）：

$$\theta_t = \theta_{t-1} - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)$$

**本项目超参含义**：

| 超参 | 值 | 含义 |
|---|---|---|
| `lr` | 3e-4 / 1e-5 | 初始学习率（峰值）|
| `betas` | (0.9, 0.95) | 0.9 用于一阶矩（梯度均值），0.95 用于二阶矩（梯度方差，比默认 0.999 更激进）|
| `weight_decay` | 0.1 | 权重衰减率（AdamW 关键参数）|
| `eps` | 1e-8 | 数值稳定性 |

---

### 1.4 学习率调度：Warmup + Cosine

**两段式调度**（所有 4 阶段都用）：

```
lr
│       ╱╲
│      ╱  ╲          peak = lr
│     ╱    ╲
│    ╱      ╲
│   ╱        ╲___   min_lr
│  ╱
│ ╱
└────────────────── iter
   warmup_iters    lr_decay_iters
   (线性上升)      (cosine 下降)
```

**公式**：

$$\text{lr}(t) = \begin{cases}
\text{lr}_{\max} \cdot \dfrac{t}{T_{\text{warmup}}} & t < T_{\text{warmup}} \\
\text{lr}_{\min} + \frac{1}{2}(\text{lr}_{\max} - \text{lr}_{\min}) \left(1 + \cos\left(\pi \cdot \dfrac{t - T_{\text{warmup}}}{T_{\text{decay}} - T_{\text{warmup}}}\right)\right) & T_{\text{warmup}} \le t \le T_{\text{decay}} \\
\text{lr}_{\min} & t > T_{\text{decay}}
\end{cases}$$

**代码实现**：

> ⚠️ **本篇的 `get_lr` / `_build_sinusoidal_pos_emb` 由机器守卫核对**：
> `qwen3-learning/qwen3-pytorch/tests/test_note_code_sync.py` 会把**这段笔记里的代码**和
> **仓库里的源码**放在一起跑，在同样输入上比输出 —— 教学上可以简化（改写成局部变量、多加一行 clamp），
> 但**行为必须一样**。改了笔记没改源码（或反之），测试立刻红。

```python
def get_lr(it: int) -> float:
    base_lr = config.learning_rate
    min_lr  = config.min_lr
    warmup  = config.warmup_iters
    decay_end = config.lr_decay_iters
    
    if it < warmup:
        return base_lr * (it + 1) / max(1, warmup)             # 线性 warmup
    if it > decay_end:
        return min_lr                                          # 衰减后保持 min
    decay_ratio = (it - warmup) / max(1, decay_end - warmup)
    decay_ratio = min(1.0, max(0.0, decay_ratio))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))     # cosine 系数
    return min_lr + coeff * (base_lr - min_lr)                 # min_lr + (lr-min_lr)*coeff
```

**各阶段参数表**：

| 阶段 | lr_max | lr_min | warmup_iters | decay_iters |
|---|---|---|---|---|
| Pretrain | 3e-4 | 3e-5 | 50 | 200（= max_iters）|
| SFT | 3e-4 | 3e-5 | ~33（min(200, 1000//5)）| 1000 |
| RM | 3e-4 | 3e-5 | ~200（min(200, 1000//5)）| 1000 |
| PPO | 1e-5 | 1e-6（按比例缩小 10x）| — | — |

**为什么需要 warmup？**
- 训练初期，模型权重接近随机
- 大学习率 + 随机梯度 → **梯度爆炸**
- 慢慢升温 → 让优化器先"探明地形"

**为什么 cosine decay？**
- 比 step decay 更平滑
- 训练后期需要小学习率精细调优
- cosine 形状让下降率"先快后慢"——符合训练 loss 下降规律

---

### 1.5 梯度裁剪

**所有 4 阶段都用梯度裁剪**：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**公式**：

$$\text{if } \|\mathbf{g}\| > \text{max\_norm}: \quad \mathbf{g} \leftarrow \mathbf{g} \cdot \frac{\text{max\_norm}}{\|\mathbf{g}\|}$$

其中 $\|\mathbf{g}\| = \sqrt{\sum_i g_i^2}$ 是全局梯度范数。

**为什么需要？**
- RNN / Transformer 训练中梯度可能爆炸
- 防止单步更新过大破坏训练稳定性
- **PPO 特别需要**：RL 训练中 gradient 经常异常大

**clip=1.0 的设计含义**：
- 1.0 是经验甜点（类似 GPT-3 用的）
- 大于 1.0 → 裁剪太松，可能仍爆
- 小于 0.5 → 训练极慢

---

## 第 2 部分：核心数学函数

### 2.1 Softmax

**公式**：

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{V} e^{x_j}}$$

**性质**：
- 输出范围 (0, 1)，且 $\sum_i \text{softmax}(x_i) = 1$（合法概率分布）
- 不变平移：$\text{softmax}(x + c) = \text{softmax}(x)$
- 当 $x_i \gg x_j$ 时，$\text{softmax}(x_i) \approx 1$，其余 ≈ 0（饱和）

**数值稳定性实现**：

```python
def softmax(x):
    x_max = x.max(dim=-1, keepdim=True).values          # 减最大值
    exp_x = torch.exp(x - x_max)                        # 防溢出
    return exp_x / exp_x.sum(dim=-1, keepdim=True)
```

**在本项目的作用**：
- attention scores → attention weights
- logits → 下一个 token 概率分布

---

### 2.2 GELU 激活

**精确公式**（Hendrycks & Gimpel 2016）：

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

其中 $\Phi(x)$ 是标准正态 CDF。

**近似公式**（GPT-2 / 我们的实现）：

$$\text{GELU}(x) \approx 0.5 x \left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715 x^3\right)\right]\right)$$

**GELU vs ReLU 对比**：

| x | ReLU | GELU |
|---|---|---|
| -2 | 0 | -0.046 |
| -1 | 0 | -0.159 |
| 0 | 0 | 0 |
| 1 | 1 | 0.841 |
| 2 | 2 | 1.954 |

**关键区别**：
- ReLU：硬截断（负半轴 = 0）
- GELU：软截断（负半轴有微小值，按概率"通过"）

**为什么 GPT 用 GELU？**
- 平滑梯度（在 0 附近可微）
- 经验上 NLP 任务表现更好

**对应代码**（[model/gpt2.py:222](gpt-learning-pytorch/model/gpt2.py#L222)）：

```python
# PyTorch 内置精确 GELU（基于 erf）
x = F.gelu(x)

# TF 端用近似版本（更高效）
x = tf.nn.gelu(x, approximate=True)
```

---

### 2.3 LayerNorm

**公式**（Ba et al. 2016）：

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

其中（对每个样本的最后一维）：
- $\mu = \frac{1}{C} \sum_{i=1}^{C} x_i$（均值）
- $\sigma^2 = \frac{1}{C} \sum_{i=1}^{C} (x_i - \mu)^2$（方差）
- $\gamma, \beta \in \mathbb{R}^C$（可学习参数，初始化为 $\gamma=1, \beta=0$）

**与 BatchNorm 的区别**：

| | BatchNorm | LayerNorm |
|---|---|---|
| 归一化维度 | 跨 batch 同一 channel | 每个样本自身跨 feature |
| 依赖 batch size | ✅ 需要 ≥ 2 才有意义 | ❌ 单样本可算 |
| 推理时 | 需 moving average | 直接算 |
| 主流使用 | CNN | **Transformer** |

**为什么 Transformer 用 LayerNorm？**
- 不依赖 batch 统计量
- 训练和推理行为一致
- 对 batch size 不敏感（适合大模型）

**对应代码**（[model/gpt2.py:262](gpt-learning-pytorch/model/gpt2.py#L262)）：

```python
self.ln_1 = nn.LayerNorm(n_embd)  # 128 维，γ 和 β 各 128 参数
```

---

### 2.4 Sinusoidal 位置编码

**公式**（Transformer 2017 原始）：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

其中：
- `pos`：token 位置（0 ~ block_size-1）
- `i`：维度索引（0 ~ d/2-1）
- `d`：n_embd（本项目 = 128）

**直觉**：
- 维度 0（i=0）：$\sin(pos)$，高频，编码"近的位置差异"
- 维度 d/2（i=d/2-1）：$\sin(pos/10000)$，低频，编码"远的位置差异"
- 任意 pos 由一组不同频率的正弦波唯一表示（类 Fourier 分解）

**对应代码**（[model/gpt2.py:33-57](gpt-learning-pytorch/model/gpt2.py#L33-L57)）：

```python
def _build_sinusoidal_pos_emb(block_size, n_embd):
    pe = torch.zeros(block_size, n_embd)
    position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
    # div_term = exp(-log(10000) * 2i / n_embd) = 1/10000^(2i/n_embd)
    div_term = torch.exp(
        torch.arange(0, n_embd, 2).float() * -(math.log(10000.0) / n_embd)
    )
    pe[:, 0::2] = torch.sin(position * div_term)   # 偶数维
    pe[:, 1::2] = torch.cos(position * div_term)   # 奇数维
    return pe
```

**与 GPT-2 原版的区别**：
- GPT-2 原文：**Learned positional embedding**（每个位置学一个向量）
- 本项目：**Sinusoidal**（固定公式，不参与训练）

**为什么本项目用 sinusoidal？**
1. 不增加参数
2. 可外推到训练时未见过的更长位置（理论性质）
3. Karpathy nanoGPT 风格（更易理解）

---

### 2.5 Scaled Dot-Product Attention

**核心公式**（Transformer 2017）：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

**各符号含义**：
- $Q \in \mathbb{R}^{T \times d_k}$：query 矩阵
- $K \in \mathbb{R}^{T \times d_k}$：key 矩阵
- $V \in \mathbb{R}^{T \times d_v}$：value 矩阵
- $d_k = \text{head\_dim}$：query/key 维度（本项目 = 32）
- $M$：mask 矩阵（GPT 中是 causal mask）

**为什么除以 $\sqrt{d_k}$？**
- 点积 $Q \cdot K$ 的方差随 $d_k$ 线性增长
- 方差大 → softmax 饱和（一个值接近 1，其他接近 0）→ 梯度消失
- 除以 $\sqrt{d_k}$ 让点积方差 ≈ 1，softmax 梯度健康

**详细推导**：
- 设 $q_i, k_i$ 独立同分布，均值 0，方差 1
- $q \cdot k = \sum_i q_i k_i$ 的方差 $= d_k$
- 标准差 $= \sqrt{d_k}$
- 除以 $\sqrt{d_k}$ 后点积 ≈ 标准正态

**Causal Mask**：

GPT 中 $M$ 是上三角（不含对角线） = $-\infty$ 的矩阵：

$$M_{ij} = \begin{cases} 0 & j \le i \text{（允许看）}\\ -\infty & j > i \text{（mask 掉）} \end{cases}$$

softmax 后 $- \infty$ 位置 = 0 概率，保证位置 $i$ 看不到位置 $j > i$。

**多头并行**：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

每个 head 学习不同子空间的关系，总参数 = $h \cdot d_v \cdot d_{model}$。

---

## 第 3 部分：4 阶段损失函数

### 3.1 Pretrain：Causal LM Cross-Entropy

**任务**：给定前 $i-1$ 个 token，预测第 $i$ 个 token。

**单样本 loss**：

$$\mathcal{L}_{\text{CLM}} = -\frac{1}{T} \sum_{i=1}^{T} \log P_\theta(t_i | t_{<i})$$

**整 batch loss**：

$$\mathcal{L} = \frac{1}{B} \sum_{b=1}^{B} \mathcal{L}_{\text{CLM}}^{(b)}$$

**实现**（[model/gpt2.py](gpt-learning-pytorch/model/gpt2.py)）：

```python
logits = ...                              # (B, T, V)
logits = logits[:, :-1, :].contiguous()  # 去掉最后 1 位（没有 target）
targets = targets[:, 1:].contiguous()    # 错位 1 位（logits[i] 预测 tokens[i+1]）
loss = F.cross_entropy(
    logits.view(-1, V),                   # [(B*(T-1)), V]
    targets.view(-1),                     # [(B*(T-1))]
    ignore_index=-1                       # padding 不算
)
```

**为什么 logits 比 targets 少 1 位？**
- logits[i] 是基于 tokens[0..i-1] 预测 tokens[i]
- 所以 logits[0..T-2] 预测 tokens[1..T-1]
- 去掉最后 1 位（logits[T-1] 没有对应的 target）

**为什么 `ignore_index=-1`？**
- padding token 设为 -1（不合法 id）
- `F.cross_entropy` 自动忽略这些位置
- 只算真实 token 的 loss

**直觉图**：

```
Input:  [t₁] [t₂] [t₃] [t₄]
            ↓    ↓    ↓    ↓
Logits: [l₁] [l₂] [l₃] [l₄]
            ↓    ↓    ↓    ↓
Target:     t₂   t₃   t₄   (eos)  ← 错位
```

---

### 3.2 SFT：Prompt-Masked Loss

**任务**：在 SFT 数据上微调，但**只算 response 部分的 loss**。

**关键创新**：把 prompt 部分的 `labels` 设为 `-100`：

```python
labels = full_ids.clone()
labels[: len(prompt_ids)] = -100        # prompt 部分 mask 掉
loss = model(full_ids, targets=labels)  # 内部 cross_entropy 跳过 labels=-100
```

**等价的 loss 公式**：

$$\mathcal{L}_{\text{SFT}} = -\frac{1}{|R|} \sum_{i \in R} \log P_\theta(t_i | t_{<i})$$

其中 $R$ = response 部分的位置集合（不含 prompt 和 padding），$|R|$ 是 response 长度。

**为什么不训练 prompt？**
- prompt 是用户输入，模型不需要"生成 prompt"
- 只训练 response 让模型学"看完 prompt 后怎么接"
- 避免模型学到"复读 prompt"的坏习惯

**InstructGPT 论文标准做法**：
- Loss mask: prompt → -100
- Loss mask: response → 真实 token id
- Loss mask: padding → -100

---

### 3.3 RM：Bradley-Terry Pairwise Loss

**任务**：给 (chosen, rejected) 对子打分，让 $r_{\text{chosen}} > r_{\text{rejected}}$。

**Bradley-Terry 概率模型**（1952）：

$$P(\text{chosen better}) = \frac{e^{r_c}}{e^{r_c} + e^{r_r}} = \sigma(r_c - r_r)$$

其中 $\sigma$ 是 sigmoid 函数。

**Cross-entropy loss**：

$$\mathcal{L}_{\text{BT}} = -\log \sigma(r_c - r_r) = -\log \frac{e^{r_c}}{e^{r_c} + e^{r_r}}$$

**等价形式**（更直观）：

$$\mathcal{L}_{\text{BT}} = \log(1 + e^{r_r - r_c}) = \text{softplus}(r_r - r_c)$$

**直觉表**：

| $r_c - r_r$ | $\sigma(r_c - r_r)$ | $\mathcal{L}$ | 含义 |
|---|---|---|---|
| +5 | 0.993 | 0.007 | RM 答对了：chosen 好很多 |
| +1 | 0.731 | 0.313 | RM 答对了：chosen 好一点 |
| 0 | 0.500 | 0.693 | RM 在猜 |
| -1 | 0.269 | 1.313 | RM 答错了：chosen 略差 |
| -5 | 0.0067 | 5.0 | RM 答错了：chosen 差很多 |

**梯度推导**（对 $r_c$）：

$$\frac{\partial \mathcal{L}_{\text{BT}}}{\partial r_c} = -\sigma(r_r - r_c) = \sigma(r_c - r_r) - 1$$

- 当 $r_c > r_r$（RM 答对）：梯度小，几乎不更新
- 当 $r_c < r_r$（RM 答错）：梯度大，强烈推动 $r_c$ 上升

**代码**（[train_reward.py:203](gpt-learning-pytorch/train_reward.py#L203)）：

```python
loss = -F.logsigmoid(r_chosen - r_rejected).mean()
# 等价于 softplus(r_rejected - r_chosen).mean()
```

---

### 3.4 PPO：Clipped Policy + Value Loss

**总 PPO loss**：

$$\mathcal{L}_{\text{PPO}} = \mathcal{L}_{\text{policy}} + c_1 \mathcal{L}_{\text{value}}$$

其中 $c_1 = 0.5$（value loss 系数）。

#### 3.4.1 Policy Loss（Clipped Surrogate Objective）

**单一 token 的 surrogate objective**：

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[\min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

其中：
- $r_t(\theta) = \dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$：新旧策略的概率比
- $A_t$：advantage（GAE 算出）
- $\epsilon = 0.2$：clip 范围

**Loss 形式（最小化）**：

$$\mathcal{L}_{\text{policy}} = -\frac{1}{|R|} \sum_{t \in R} L^{\text{CLIP}}_t$$

**Clipped Objective 详解**：

设 $A_t > 0$（好动作）：
- $r_t$ 越大越好（让 policy 更倾向这个动作）
- 但 $r_t > 1+\epsilon$ 时 clip 到 $1+\epsilon$（限制更新幅度）

设 $A_t < 0$（坏动作）：
- $r_t$ 越小越好（让 policy 远离这个动作）
- 但 $r_t < 1-\epsilon$ 时 clip 到 $1-\epsilon$（限制更新幅度）

→ **min() 的作用：取悲观值，防止 policy 单步更新过大**

#### 3.4.2 Value Loss

$$\mathcal{L}_{\text{value}} = \frac{1}{|R|} \sum_{t \in R} (V_\theta(s_t) - R_t)^2$$

其中：
- $V_\theta(s_t)$：value_head 在状态 $s_t$ 的估计（标量）
- $R_t = A_t + V_t$：实际收益（GAE 返回值）

#### 3.4.3 代码实现（[train_ppo.py](gpt-learning-pytorch/train_ppo.py)）

```python
# policy loss
ratio = torch.exp(logp_new - old_logprobs)                    # (B, T-1)
loss_unclipped = ratio * advantages
loss_clipped   = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
loss_policy    = -torch.min(loss_unclipped, loss_clipped).mean()

# value loss
loss_value = F.mse_loss(values, returns)

# 总 loss
loss = loss_policy + ppo_vf_coef * loss_value                # c1=0.5
```

**4 个超参**：

| 超参 | 值 | 含义 |
|---|---|---|
| `ppo_cliprange` | 0.2 | policy ratio clip 范围 ε |
| `ppo_cliprange_value` | 0.2 | value clip 范围（可选）|
| `ppo_vf_coef` | 0.5 | value loss 系数 $c_1$ |
| `ppo_epochs` | 4 | 同一批数据反复更新的 epoch 数 |

---

## 第 4 部分：PPO 强化学习公式

### 4.1 KL 散度近似（k3 估计）

**精确 KL**：

$$D_{\text{KL}}(\pi \| \pi_{\text{ref}}) = \mathbb{E}_{a \sim \pi} \left[\log \frac{\pi(a|s)}{\pi_{\text{ref}}(a|s)}\right]$$

**k3 估计**（一阶 Taylor 近似）：

$$\hat{D}_{\text{KL}} \approx \log \pi_\theta(a|s) - \log \pi_{\text{ref}}(a|s)$$

**Token-level 形式**（在 PPO 中）：

$$\text{kl}_t = \log p_\theta(t | t_{<t}) - \log p_{\text{ref}}(t | t_{<t})$$

**整句 KL**：

$$\text{KL}_{\text{seq}} = \sum_{t \in R} \text{kl}_t$$

**直觉**：KL 散度 = policy 在每个 token 上的 log 概率差之和。

**代码**（[train_ppo.py:262-268](gpt-learning-pytorch/train_ppo.py#L262-L268)）：

```python
def compute_kl(policy_lp, ref_lp):
    return (policy_lp - ref_lp).sum(dim=-1)  # (B,)
```

**为什么用 k3 估计？**
- 严格 KL 需要在 policy 分布上求期望（要采样）
- k3 估计只用"已采样动作"的 log 概率差（一阶 Taylor）
- 工程上便宜得多，效果接近

---

### 4.2 Reward Shaping

**总 reward = RM 分数 - KL 惩罚**：

$$r_t^{\text{shaped}} = \begin{cases}
-\beta \cdot \text{kl}_t & t < T_{\text{response}} \\
r_{\text{RM}} - \beta \cdot \text{kl}_T & t = T_{\text{response}} \\
0 & t \ge T_{\text{response}} \text{（已结束）}
\end{cases}$$

**对应代码**（[train_ppo.py:392-393](gpt-learning-pytorch/train_ppo.py#L392-L393)）：

```python
_, _, _, r_total = reward_model(rm_ids, targets=None)         # (B,) RM 分数
r_total = r_total - config.ppo_kl_coef * kl_per_sample       # KL 惩罚
# r_total 现在 = RM 分数 - β * KL
```

**β 系数设计哲学**：

| β 值 | 效果 |
|---|---|
| β = 0 | 纯 RM 优化 → 容易 reward hacking |
| β = 0.05 | 轻惩罚 → 快速适应 RM，但可能风格漂移 |
| **β = 0.1（本项目）**| **中等 → 平衡** |
| β = 0.5 | 重惩罚 → 保守，policy 几乎不变 |
| β → ∞ | policy 完全冻结 = ref model（SFT）|

---

### 4.3 Reward Normalization

**问题**：RM 输出 scale 不一致（PT/TF 不同），PPO 训练中 reward 容易爆炸。

**解法**：每个 batch 内做标准化：

$$\hat{r}_b = \frac{r_b - \mu_r}{\sigma_r + \epsilon}$$

其中 $\mu_r = \frac{1}{B} \sum_b r_b$，$\sigma_r = \sqrt{\frac{1}{B} \sum_b (r_b - \mu_r)^2}$。

**代码**（[train_ppo.py:397-399](gpt-learning-pytorch/train_ppo.py#L397-L399)）：

```python
r_mean = r_total.mean()
r_std = r_total.std() + 1e-8
r_total = (r_total - r_mean) / r_std                          # (B,) 标准化后
```

**为什么有效？**
- 把 reward 强制拉到 ~N(0, 1)
- 防止个别极端值主导梯度
- 训练稳定（PT 端实测：reward 不再爆炸到 ±100+）

---

### 4.4 GAE 广义优势估计

**TD-error（Temporal Difference error）**：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

含义：在状态 $s_t$ 看到的"实际奖励 + 下一状态预估"减去"当前预估"= **惊喜程度**。

**GAE 递归**（Schulman et al. 2015）：

$$A_t^{\text{GAE}} = \delta_t + \gamma \lambda A_{t+1}^{\text{GAE}}$$

边界条件：$A_T^{\text{GAE}} = 0$（序列末尾）。

**展开形式**（递归展开）：

$$A_t^{\text{GAE}} = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}$$

**超参含义**：

| 超参 | 值 | 效果 |
|---|---|---|
| γ（折扣因子）| 1.0 | 未来奖励不衰减（语言任务常用）|
| λ（GAE λ）| 0.95 | bias-variance 平衡，λ→0 高 bias 低 variance；λ→1 低 bias 高 variance |
| 经验甜点 | — | λ=0.95 是 Schulman 推荐 |

**Returns（value head 学习目标）**：

$$R_t = A_t + V(s_t)$$

**代码**（[train_ppo.py:271-303](gpt-learning-pytorch/train_ppo.py#L271-L303)）：

```python
def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=rewards.device)
    
    # 反向递归
    for t in reversed(range(T)):
        next_v = values[:, t+1] if t < T-1 else torch.zeros(B, device=rewards.device)
        delta = rewards[:, t] + gamma * next_v - values[:, t]
        last_gae = delta + gamma * lam * last_gae * mask[:, t]   # mask 修正
        advantages[:, t] = last_gae
    
    returns = advantages + values
    return advantages, returns
```

**为什么反向递归？**
- $A_t$ 依赖 $A_{t+1}$（后一时刻）
- 从 $t=T-1$ 开始往前算，保证依赖已计算
- 边界 $A_T = 0$ 自然满足

**mask[:, t] 的作用？**
- 不同样本序列长度可能不同
- padding 位置不应参与 GAE 累积
- mask = 0 让 last_gae 重置，避免污染有效位置

---

### 4.5 Advantage Normalization（可选）

**公式**：

$$\hat{A}_t = \frac{A_t - \mu_A}{\sigma_A + \epsilon}$$

**作用**：
- 把 advantage 拉到 ~N(0, 1)
- 让 loss 量级与 lr 解耦
- 在大 batch PPO 中常用

**本项目**：未启用（每个 rollout batch size = 8 较小，不需要）

---

## 第 5 部分：9 项评估指标公式

### 5.1 Perplexity (PPL)

**公式**：

$$\text{PPL} = \exp\left(\mathcal{L}_{\text{CLM}}\right) = \exp\left(-\frac{1}{T} \sum_{i=1}^{T} \log P(t_i | t_{<i})\right)$$

**直觉**：模型"等效从多少个候选 token 里挑"——PPL=20 意味着平均相当于从 20 个里挑。

**Baseline 对照**：

| PPL | 含义 |
|---|---|
| 1 | 完美预测（从不困惑）|
| V（vocab_size = 45）| 随机猜测 |
| ∞ | 模型完全失败 |

**代码**（[train_pretrain.py](gpt-learning-pytorch/train_pretrain.py)）：

```python
val_loss = sum(losses) / n_samples
val_ppl = math.exp(val_loss)                                  # ← 关键一行
```

**项目实测**：PT 20.23（200 iter）/ TF 1.92（2000 iter）/ random 45。

---

### 5.2 Pairwise Accuracy

**公式**：

$$\text{Acc} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[r(x_i^{\text{chosen}}) > r(x_i^{\text{rejected}})]$$

**等价形式**：

$$\text{Acc} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\sigma(r_c - r_r) > 0.5]$$

**Baseline 对照**：

| Acc | 含义 |
|---|---|
| 0.5 | RM 在猜 |
| 0.7+ | RM 学到了偏好 |
| 1.0 | RM 完美（实际达不到）|

**项目实测**：PT 73% / TF 78%。

---

### 5.3 Distinct-N

**公式**（[Li et al. 2016](https://arxiv.org/abs/1510.03055)）：

$$\text{Distinct-N} = \frac{\text{unique n-grams}}{\text{total n-grams}}$$

**直觉**：去重率——越高越多样。

**Baseline 对照**：

| Distinct-1 | 含义 |
|---|---|
| 接近 0 | 完全重复（如 SFT 过拟合）|
| 0.5-0.8 | 健康生成 |
| 接近 1 | 完全不同 token（可能胡言乱语）|

**项目实测**：PT distinct_1 = 0.0027（SFT 严重过拟合，多样性极低）。

---

### 5.4 Self-BLEU

**公式**（生成样本两两之间 BLEU 分数的平均）：

$$\text{Self-BLEU} = \frac{2}{N(N-1)} \sum_{i=1}^{N} \sum_{j \ne i} \text{BLEU}(x_i, x_j)$$

**BLEU 公式**（Papineni et al. 2002）：

$$\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

其中：
- $p_n$ = n-gram 精确率（modified n-gram precision）
- $w_n = 1/N$（uniform weights）
- $BP = \min(1, e^{1 - r/c})$：brevity penalty（$r$ 参考长度，$c$ 候选长度）

**直觉**：Self-BLEU = "生成的不同样本之间有多相似"——越高越单一。

**Baseline 对照**：

| Self-BLEU | 含义 |
|---|---|
| 接近 1 | 所有样本几乎相同（严重 mode collapse）|
| 0.3-0.7 | 健康 |
| 接近 0 | 完全不同（多样，但可能不连贯）|

**实现**：基于 [nltk.translate.bleu_score](https://www.nltk.org/api/nltk.translate.bleu_score.html)，需 `nltk.download('punkt')`。

**项目实测**：PT self_bleu = 1.0（SFT 过拟合，生成高度相似）。

---

### 5.5 Token Entropy

**公式**：

$$H(p) = -\sum_{v=1}^{V} p(v) \log p(v)$$

其中 $p(v) = \text{softmax}(\text{logits})_v$。

**性质**：
- $H \in [0, \log V]$
- $H = 0$：确定性分布（一个 token 概率=1）
- $H = \log V$：均匀分布（完全随机）

**项目实测**：PT token_entropy = 0.0000（SFT 输出概率极度集中，model collapse 迹象）。

---

### 5.6 Response Quality

**规则评估**（项目自定义）：

$$Q = \frac{1}{N} \sum_{i=1}^{N} \text{rule}(x_i^{\text{prompt}}, x_i^{\text{response}})$$

**本项目的规则**（[evaluate.py](gpt-learning-pytorch/evaluate.py)）：
- 检查 response 是否包含特定关键词（如 "fox"、"owl"）
- 检查 response 长度是否在合理区间
- 规则集合因 prompt 类型而异

**Baseline 对照**：

| Q | 含义 |
|---|---|
| 接近 0 | 生成的 response 不符合任务格式 |
| 0.5-0.8 | 健康 |
| 1.0 | 全部符合规则 |

**项目实测**：PT quality = 0.0000（SFT 过拟合，response 几乎相同 → 不满足多样性规则）。

---

### 5.7 SFT vs PPO 对比

**公式**：

$$\Delta R = \frac{1}{N} \sum_{i=1}^{N} \left[ r_{\text{PPO}}(x_i) - r_{\text{SFT}}(x_i) \right]$$

**解读**：

| $\Delta R$ | 含义 |
|---|---|
| > 0 | PPO 比 SFT 好（PPO 改善了）|
| ≈ 0 | 一样 |
| < 0 | PPO 比 SFT 差（PPO 退化了）|

**项目实测**：PPO 比 SFT 在 RM 分数上高 +1.2 ~ +2.2 ✅

---

### 5.8 Benchmark Accuracy

**公式**：

$$\text{Benchmark Acc} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{model}(x_i^{\text{question}}) = x_i^{\text{answer}}]$$

**对比 baseline**：

| Acc | 含义 |
|---|---|
| 0.25 (4-选 1) | 随机 |
| 0.5+ | 学到了 |
| 1.0 | 完美 |

**项目实测**：PT 30% / TF 32%（>25% 随机 baseline）✅

---

### 5.9 KV Cache Speedup

**公式**：

$$\text{Speedup} = \frac{T_{\text{naive}}}{T_{\text{KV}}}$$

**理论对比**：

| 方案 | 总 FLOPs | 复杂度 |
|---|---|---|
| **Naive（每次从头跑）** | $\sum_{t=1}^{T} t^2 = O(T^3)$ | $T^3/3$ |
| **KV Cache（用 prefill）** | $T_p^2 + (T-T_p) \cdot T = O(T^2)$ | $T^2/2$ |

**实际加速比**：$T / 3$（理论）/ 实测 1.5-1.8x（PyTorch，受 Python 开销影响）

**项目实测**：
- PyTorch：1.5-1.8x ✅
- TensorFlow：1.0x（Python loop 开销主导，tiny scale 下不显著）

---

## 第 6 部分：采样策略公式

### 6.1 Temperature Scaling

**公式**：

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

**温度效应**：

| T 值 | 分布 | 效果 |
|---|---|---|
| T → 0 | one-hot（贪心）| 确定性，最大概率 |
| T = 1 | 标准 softmax | 平衡 |
| T → ∞ | 均匀分布 | 完全随机 |
| **T = 0.8（本项目默认）** | 略锐 | 稍微保守一点 |

**代码**（[inference.py:_sample_from_logits](gpt-learning-pytorch/inference.py#L334-L366)）：

```python
if temperature <= 0.0:
    return logits.argmax(dim=-1)               # greedy
logits = logits / temperature                  # 温度缩放
```

---

### 6.2 Top-K

**公式**：

$$p_i^{\text{top-k}} = \begin{cases}
p_i & i \in \text{top-k}(p) \\
0 & \text{otherwise}
\end{cases}$$

然后重新归一化。

**直觉**：只保留概率最大的 k 个，其余设为 $-\infty$。

**项目默认**：`top_k = 50`

---

### 6.3 Top-P (Nucleus)

**公式**（Holtzman et al. 2020）：

1. 把 logits 按降序排序：$z_{(1)} \ge z_{(2)} \ge ... \ge z_{(V)}$
2. 计算 softmax 累积概率：$\sum_{i=1}^{k} p_{(i)} \le p$ 的最小 $k$
3. 保留前 $k$ 个，其余 mask 掉

**直觉**：保留"累计概率到 p"的最小候选集。

**项目默认**：`top_p = 0.95`

**示例**：

```
原始概率:   [0.5, 0.3, 0.1, 0.05, 0.05]  (V=5)
累积:       [0.5, 0.8, 0.9, 0.95, 1.00]
top_p=0.9:  保留前 3 个 [0.5, 0.3, 0.1]，重新归一化 → [0.55, 0.33, 0.11]
```

**代码**：

```python
sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
sorted_probs = sorted_logits.softmax(dim=-1)
cumsum = sorted_probs.cumsum(dim=-1)
mask = cumsum - sorted_probs > top_p                       # 超出 top_p 的 mask
sorted_logits = sorted_logits.masked_fill(mask, float('-inf'))
logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
```

**何时用哪个？**

| 场景 | 推荐 |
|---|---|
| 贪心（评测）| temperature=0 |
| 开放生成（默认）| temperature=0.8 + top_k=50 + top_p=0.95 |
| PPO rollout | temperature=1.0 + top_k=50 + top_p=0.95（探索）|
| 短文本 QA | temperature=0.3-0.5（保守）|

---

## 附录：关键数字速查表

### 模型超参（[config.py](gpt-learning-pytorch/config.py)）

| 超参 | 值 | 备注 |
|---|---|---|
| `n_layer` | 4 | |
| `n_head` | 4 | |
| `n_embd` | 128 | |
| `n_inner` | 512 | 4×n_embd |
| `block_size` | 128 | |
| `vocab_size` | 45 | 5 special + 40 chars |
| `dropout` | 0.1 | |
| `bias` | False | nanoGPT 风格 |
| 总可训练参数 | **794,496** | 实测值 |

### 训练超参（4 阶段）

| | Pretrain | SFT | RM | PPO |
|---|---|---|---|---|
| lr_max | 3e-4 | 3e-4 | 3e-4 | 1e-5 |
| lr_min | 3e-5 | 3e-5 | 3e-5 | ~1e-6 |
| batch_size | 32 | 32 | 32 | 8 rollout / 4 minibatch |
| max_iters | 200 | 1000 | 1000 | 100 |
| warmup_iters | 50 | min(200, 200) | min(200, 200) | — |
| weight_decay | 0.1 | 0.1 | 0.1 | 0.1 |
| betas | (0.9, 0.95) | (0.9, 0.95) | (0.9, 0.95) | (0.9, 0.95) |
| grad_clip | 1.0 | 1.0 | 1.0 | 1.0 |

### PPO 专属超参

| 超参 | 值 | 公式符号 |
|---|---|---|
| `ppo_gamma` | 1.0 | $\gamma$ |
| `ppo_lam` | 0.95 | $\lambda$ |
| `ppo_cliprange` | 0.2 | $\epsilon$ |
| `ppo_cliprange_value` | 0.2 | — |
| `ppo_vf_coef` | 0.5 | $c_1$ |
| `ppo_kl_coef` | 0.1 | $\beta$ |
| `ppo_epochs` | 4 | $K$ |

### 评估指标（项目实测）

| 指标 | PT | TF | 解读 |
|---|---|---|---|
| PPL（pretrain）| 20.23 | 1.92 | 远低于 random=45 ✅ |
| Pairwise acc（RM）| 73% | 78% | 显著高于 50% ✅ |
| Distinct-1 | 0.0027 | — | SFT 过拟合 ⚠️ |
| Self-BLEU | 1.0 | — | mode collapse ⚠️ |
| Token entropy | 0.0000 | — | 概率集中 ⚠️ |
| Response quality | 0.0000 | — | 规则不满足 ⚠️ |
| SFT vs PPO | +1.2 ~ +2.2 | — | PPO 改善 ✅ |
| Benchmark acc | 30% | 32% | >25% random ✅ |
| KV speedup | 1.5-1.8x | 1.0x | PT 显著 |

### 关键数学常数

| 常数 | 值 | 含义 |
|---|---|---|
| $\log 2$ | 0.693 | 二分类等概率 loss |
| $\log 45$ | 3.807 | random 45-vocab loss |
| $\sigma(5)$ | 0.993 | BT 模型"很强正确" |
| $\sigma(-5)$ | 0.0067 | BT 模型"很强错误" |
| $-\log\sigma(5)$ | 0.007 | BT loss at 极正 |
| $-\log\sigma(-5)$ | 5.0 | BT loss at 极负 |
| $e^1$ | 2.718 | PPL=1 → loss=1 |

---

## 关键论文 / 参考

| 主题 | 论文 / 资源 |
|---|---|
| Transformer | [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762) |
| GPT-2 | [Radford et al. 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) |
| Adam | [Kingma & Ba 2014](https://arxiv.org/abs/1412.6980) |
| AdamW | [Loshchilov & Hutter 2019](https://arxiv.org/abs/1711.05101) |
| LayerNorm | [Ba et al. 2016](https://arxiv.org/abs/1607.06450) |
| GELU | [Hendrycks & Gimpel 2016](https://arxiv.org/abs/1606.08415) |
| Bradley-Terry | [Bradley & Terry 1952](https://www.jstor.org/stable/2334029) |
| InstructGPT (RLHF) | [Ouyang et al. 2022](https://arxiv.org/abs/2203.02155) |
| PPO | [Schulman et al. 2017](https://arxiv.org/abs/1707.06347) |
| GAE | [Schulman et al. 2015](https://arxiv.org/abs/1506.02438) |
| Top-P (Nucleus) | [Holtzman et al. 2020](https://arxiv.org/abs/1904.09751) |
| BLEU | [Papineni et al. 2002](https://www.aclweb.org/anthology/P02-1040.pdf) |
| Distinct-N | [Li et al. 2016](https://arxiv.org/abs/1510.03055) |

---

**作者**：caiqiyue
**项目地址**：https://github.com/caiqiyue/gpt-learning
**最后更新**：2026-06-11
**配套阅读**：
- 主文档：[`GPT全流程详解.md`](GPT全流程详解.md)（架构 + 流程）
- spec 文档：[`docs/superpowers/specs/2026-06-09-gpt2-training-explained.md`](docs/superpowers/specs/2026-06-09-gpt2-training-explained.md)（算法详解 1184 行）
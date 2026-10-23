# Qwen3-LoRA 与 DPO 学习笔记

> **对应 GPT-2 实现位置**：[`gpt-learning/gpt-learning-pytorch/train_sft.py`](../gpt-learning/gpt-learning-pytorch/train_sft.py)（**全参 SFT**）/ [`gpt-learning/gpt-learning-pytorch/train_ppo.py`](../gpt-learning/gpt-learning-pytorch/train_ppo.py)（**RLHF 三件套**）—— **GPT-2 线没有 LoRA / DPO**（原理另见 [`gpt-learning/9.强化学习.md`](../gpt-learning/9.强化学习.md)）
> **对应 Qwen3 源码位置**：[`qwen3-learning/qwen3-pytorch/model/common/lora.py`](qwen3-pytorch/model/common/lora.py) → `LoRALinear` / `inject_lora` / `merge_lora` / `strip_lora`；[`qwen3-learning/qwen3-pytorch/train/train_lora.py`](qwen3-pytorch/train/train_lora.py)；[`qwen3-learning/qwen3-pytorch/train/train_dpo.py`](qwen3-pytorch/train/train_dpo.py) → `dpo_loss`
> **对应可跑测试**：[`qwen3-learning/qwen3-pytorch/tests/test_lora.py`](qwen3-pytorch/tests/test_lora.py)、[`qwen3-learning/qwen3-pytorch/tests/test_dpo.py`](qwen3-pytorch/tests/test_dpo.py)

> 代码位置：`qwen3-learning/qwen3-pytorch/model/common/lora.py`、
> `qwen3-learning/qwen3-pytorch/train/train_lora.py`、
> `qwen3-learning/qwen3-pytorch/tests/test_lora.py`
>
> 本文所有数字都来自**本仓库代码实跑输出**（CPU，2026-10-22），
> 不是从论文或别处抄的。跑法见 §6。

---

## 1. LoRA 要解决什么问题

全参微调一个模型，要为每个参数额外存一份梯度 + 一份优化器状态（AdamW 是两份）：

\[
\text{全参微调显存} \approx W \times (1_{\text{权重}} + 1_{\text{梯度}} + 2_{\text{Adam状态}})
\]

对一个大模型来说，**适配一个下游任务的开销与模型本身同阶**。
这在「一个基座要适配几十上百个任务」时是不可接受的。

LoRA（LoRA: Low-Rank Adaptation, Hu et al. 2021）的假设是：

> **微调时权重的变化量 ΔW 是低秩的。**

于是不去直接更新 W，而是学一个低秩的旁路：

\[
W' = W + \Delta W, \qquad \Delta W = BA, \qquad B \in \mathbb{R}^{d_{out}\times r},\ A \in \mathbb{R}^{r\times d_{in}}
\]

其中 \(r \ll \min(d_{in}, d_{out})\)。

参数量对比（单个 Linear）：

| 方案 | 可训练参数量 |
|---|---|
| 全参 | \(d_{in}\times d_{out}\) |
| LoRA | \(r(d_{in}+d_{out})\) |

本仓库 `q_proj` 是 \(d_{in}=128,\ d_{out}=192\)，`v_proj` 是 \(d_{in}=128,\ d_{out}=96\)：

- 全参：\(128\times192 + 128\times96 = 24576 + 12288 = 36864\)
- LoRA r=4：\(4(128+192) + 4(128+96) = 1280 + 896 = 2176\)（每层）
- 4 层：全参 147456 → LoRA **8704**，约为 **1/17**

## 2. 核心公式

LoRA 层的前向：

\[
\boxed{y = Wx + \frac{\alpha}{r}\, B A x}
\]

代码里写成（`model/common/lora.py`）：

```python
out = self.base(x)
delta = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
return out + self.scaling * delta        # self.scaling = alpha / r
```

三个设计点，缺一不可：

### 2.1 B 零初始化 —— 「注入即无损」

`lora_B` 初始化为**全零**（`torch.zeros(out_features, r)`），`lora_A` 用 Kaiming 初始化。

初始时：

\[
BAx = 0 \quad\Longrightarrow\quad y = Wx
\]

也就是说：

> **刚注入 LoRA 的模型，输出与预训练模型逐元素完全相同。**

这不是「差不多」，是**严格相等**（float 误差在 `atol=1e-6` 内）。
这条性质被 `tests/test_lora.py` 第 4 条守住，也是 `init.sh` L4 的 LoRA smoke 在测的东西。

为什么必须这样：如果 A、B 都随机初始化，注入的瞬间模型输出会被打乱，
微调就得先「把模型拉回来」再学新东西；而且**同一次微调的不同随机种子起点不同**，
实验不可复现。B 零初始化让起点唯一。

### 2.2 缩放因子 α/r

`scaling = alpha / r`，本仓库默认 `alpha = 2r`（r=4 → alpha=8 → scaling=2.0）。

- `alpha` 固定而 `r` 变小时，实际更新幅度被放大 ⇒ **同一套 alpha 下，r 越小有效学习率越大**。
- 这也是为什么 LoRA 的学习率通常比全参微调**大一个量级**（本仓库 `lora_lr=1e-3` vs 预训练 `learning_rate=3e-4`）。

### 2.3 注入点

LoRA 可以注入任何 Linear。本仓库默认注入 **`q_proj` 与 `v_proj`**（`config.lora_target_modules`）：

```python
lora_target_modules: tuple = ("q_proj", "v_proj")
```

选 attention 的 q/v 而不是 MLP，理由与结论见 §7 的消融表。

## 3. 实现：三个基本动作 + 两个补丁

`model/common/lora.py` 提供 5 个函数。**三个是 LoRA 论文里就有的，两个是本仓库实测踩坑后补的。**

### 3.1 三个基本动作

| 函数 | 作用 |
|---|---|
| `inject_lora(model, target_modules, r, alpha, dropout)` | 按模块名后缀递归把 `nn.Linear` 换成 `LoRALinear` |
| `set_lora_enabled(model, bool)` | 一键开关全模型的 LoRA 分支 |
| `merge_lora(model)` | 把 ΔW 折回 base 权重，导出成普通 Linear |

`set_lora_enabled` 是 **DPO 的关键**，见 §8。

### 3.2 补丁 ①：`mark_only_lora_trainable`

**问题**：`inject_lora` 只冻结了**被包装的那一个** Linear 的 `weight` / `bias`。
模型里其余的参数 —— `wte`（embedding）、`q_norm` / `k_norm` / `rms_norm`、
MLP 的 `gate_proj` / `up_proj` / `down_proj`、`lm_head` —— **全部仍然是 `requires_grad=True`**。

于是如果按最直觉的写法取可训练参数：

```python
trainable = [p for p in model.parameters() if p.requires_grad]
```

会把 embedding 一起训进去 —— **「LoRA 微调」悄悄变成了「大部分参数都在训」**，
既不是 LoRA，显存也没省下来，而且实验结论全错。

**修法**：

> ⚠️ **本篇的 `mark_only_lora_trainable` / `set_lora_enabled` 由机器守卫核对**：
> `qwen3-learning/qwen3-pytorch/tests/test_note_code_sync.py` 会把**这段笔记里的代码**和
> **仓库里的源码**放在一起跑，在同样输入上比输出 —— 教学上可以简化，但**行为必须一样**。
> 改了笔记没改源码（或反之），测试立刻红。

```python
def mark_only_lora_trainable(model):
    lora_ids = {id(p) for p in lora_parameters(model)}
    for p in model.parameters():
        p.requires_grad = id(p) in lora_ids
    return [p for p in model.parameters() if p.requires_grad]
```

实测效果（vocab=778 的真实模型）：

```
trainable params = 8,704 / 994,560 (0.88%)
```

其中 8,704 = 4 层 × (1280 + 896)，而 994,560 = 基座 985,856 + LoRA 8,704。

> 这个 bug **不会报错**，只会静默地给出错误结论。属于「结构性 bug」——
> 跑得通、loss 也在降，但训的根本不是 LoRA。

### 3.3 补丁 ②：`merge_lora` 合并后必须关分支

**问题**：LoRA 的最终权重是 `W + (α/r)·BA`。如果把它折回 base：

```python
module.base.weight.data.copy_(module.base.weight.data + module.scaling * (B @ A))
```

**但分支还开着**，前向变成：

\[
y = (W + \tfrac{\alpha}{r}BA)x + \tfrac{\alpha}{r}BAx
\]

**ΔW 被算了两遍**。

**修法**：`merge_lora` 合并完把 `scaling` 置 0：

```python
module.base.weight.data.copy_(module.merged_weight())
module.scaling = 0.0        # 合并即收编，分支不再参与前向
```

`tests/test_lora.py` 第 6 条「merge 前后输出必须一致」就是防这个的。
（顺带一提：**初稿的 `merge_lora` 配初稿的测试，本来就是红的** —— 这条测试是初稿自带的，
但初稿的实现过不了它。）

### 3.4 `set_lora_enabled` 与 DPO 的关系

```python
def set_lora_enabled(model, enabled):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.scaling = module.alpha / module.r if enabled else 0.0
```

DPO 需要两个模型：

- \(\pi_\theta\)：正在训练的策略
- \(\pi_{ref}\)：参考模型（通常是 SFT 后的模型，作用是「不要偏离太远」）

常规实现要**存两份完整权重**。本仓库不这么做：

> **同一个模型对象，LoRA 分支开着是 \(\pi_\theta\)，关掉就是 \(\pi_{ref}\)。**
> 因为 base 权重从头到尾没被改过，关掉分支就精确等于原始预训练模型。

省一份显存、省一份 checkpoint。§8 会用到这一点。

## 4. 训练流程

`train/train_lora.py` 的流程：

1. 加载 `checkpoints/pretrain_dense.pt`（`strict=True`，且**必须在注入之前**加载 ——
   注入后模块名会变，state_dict 对不上）
2. `inject_lora` → `mark_only_lora_trainable`
3. 评估**注入后但未训练**的基线（此时 B=0，行为 == 预训练模型）
4. AdamW 只对 A/B 优化，训练：给定 prompt，补全 chosen
5. 再次评估
6. 落盘 `checkpoints/lora.pt`

### 4.1 「只对回复部分计分」

LoRA 与 DPO 都需要序列的**回复段** logprob：

\[
\log \pi(y\mid x) = \sum_{t \in \text{回复位置}} \log p(y_t \mid x, y_{<t})
\]

所以 prompt 段与 response 段**分别编码再拼接**：

```python
def encode_prompt_response(tk, prompt, response):
    prefix_ids = tk.encode(PROMPT_TEMPLATE.format(prompt=prompt))
    resp_ids = tk.encode(response)
    return prefix_ids + resp_ids, len(prefix_ids)
```

为什么不编码拼接后的整串？因为边界处 BPE 可能把分隔符和回复第一个字符合并，
那「回复从哪个 token 开始」就不再精确了。分开编码，这个下标是**解析出来的、精确的**。

掩码（注意：模型内部的 loss 是 `cross_entropy(logits, targets)`，**不做内部 shift**，
所以 `targets[i]` 必须是「位置 i 预测出来的 token」，即 `ids[i+1]`）：

```python
targets = torch.full_like(inputs, -100)
targets[0, n_prompt - 1: len(ids) - 1] = inputs[0, n_prompt:]
```

prompt 段与最后一个位置都是 -100（忽略）。

> **踩坑**：初稿的掩码是 `targets = cat([idx[:,1:], -100])` —— 只在最后一个位置忽略，
> 于是 **prompt 的 token 也进了 loss**，与它自己 docstring 写的「给定 prompt 补全 chosen」矛盾。
> 这也是结构性 bug：loss 照常下降，但模型学到的是「续写 prompt」而不是「回答问题」。

## 5. 评估口径：长度偏置（本次最值得记住的一条）

评估用的是「偏好胜率」：chosen 的 logprob 高于 rejected 的样本比例。

但这里有**两个互相打架的口径**：

### 5.1 总 logprob 胜率（`win_rate`）

\[
\text{win\_rate} = \frac{1}{N}\sum \mathbb{1}\left[\log\pi(y_w) > \log\pi(y_l)\right]
\]

**问题**：总 logprob 与回复长度强相关。回复越短，总 logprob 越接近 0（越大）。
所以这个口径**系统性偏袒短回复**。

### 5.2 长度归一胜率（`win_rate_per_token`）

\[
\text{win\_rate\_per\_token} = \frac{1}{N}\sum \mathbb{1}\left[\frac{\log\pi(y_w)}{|y_w|} > \frac{\log\pi(y_l)}{|y_l|}\right]
\]

**问题**：反过来**偏袒「可预测」的文本**（重复、常见词）—— 每 token 平均 logprob 高的
往往是那些「不用想也能接上」的内容。

### 5.3 实测：按策略分组看

本仓库的合成偏好数据有 4 种「变差策略」。实测（预训练基线，未微调）：

| 策略 | n | 总 logprob 胜率 | 长度归一胜率 | len(chosen) | len(rejected) |
|---|---|---|---|---|---|
| `noisy`（加噪声） | 43 | 0.767 | 0.163 | 71.4 | 86.9 |
| `too_short`（截短） | 47 | **0.000** | 0.404 | 72.0 | **6.3** |
| `unrelated`（答非所问） | 66 | 0.576 | 0.545 | 67.4 | 69.3 |
| `wrong_entity`（实体错） | 44 | 0.727 | 0.409 | 69.5 | 70.1 |
| **整体** | 200 | **0.515** | **0.400** | | |

`too_short` 的总 logprob 胜率**恒为 0.000** —— 因为 rejected 只有 6.3 个 token，
而 chosen 有 72 个。总 logprob 天然被短文本占便宜，这条策略**与模型好坏无关地必输**。

于是：

- 用总 logprob 口径，胜率的**天花板被压到 0.75 左右**（0.25 的样本必输），
  后面无论模型怎么变好都看不出来；
- 用长度归一口径，胜率**天然偏低**（0.400），因为它偏袒「可预测」的文本。

### 5.4 这正是 DPO 要减去参考模型的原因

> **DPO 的损失里有 \(-\log\sigma(\beta[(\log\pi_\theta(y_w) - \log\pi_{ref}(y_w)) - (\log\pi_\theta(y_l) - \log\pi_{ref}(y_l))])\)。**
> 括号里是**差值**：`π_ref` 把「回复长度」「文本可预测性」这类**静态偏置**同时减掉了，
> 剩下的才是「相对参考模型，模型更偏好了哪个」。

所以 §5 这两种口径的偏差，**在 DPO 里是自动抵消的** —— 这就是 DPO 相对
「直接用 SFT + 打分」的根本优势，也是本仓库下一节要手写的东西。

## 6. 实测结果（全部可复现）

### 6.1 单测

```bash
cd qwen3-learning/qwen3-pytorch
python tests/test_lora.py
```

```
[可训练参数] 8,704 / 927,744 (0.94%)
test_lora OK  replaced=8 r=4 alpha=8 scaling=2.0
```

（这里用 vocab=256 的小模型，所以总参数 927,744；比例 0.94%。）

`tests/test_lora.py` 的 9 组断言：

1. B 零初始化 ⇒ 注入前后输出逐元素相等
2. base 的 `weight` / `bias` 都被冻结
3. 只有 A / B 拿到梯度
4. 注入的模块数 = `n_layer × len(target_modules)` = 8
5. `set_lora_enabled(False)` 后回到注入前的输出
6. **merge 前后输出必须一致**（防 §3.3 的双算）
7. ΔW 的秩 ≤ r（用 `.double()` 算，否则 float32 舍入噪声会让 `matrix_rank` 判错）
8. `mark_only_lora_trainable` 后可训练参数占比 < 20%（单测）/ < 10%（`init.sh` smoke 更严）
9. 边界情况：没有匹配模块返回空列表；没有 LoRA 时 `merge_lora` 返回 0

### 6.2 训练

```bash
python train/train_lora.py
```

**BEFORE**（注入后未训练 == 预训练模型）：

```
[lora] preference pairs = 2000
[lora] BEFORE  win_rate=0.515  win_rate_per_token=0.400  mean_logp(chosen)=-603.26
[lora]   noisy          n=43   总logprob胜率=0.767  长度归一胜率=0.163  len(chosen)=71.4 len(rejected)=86.9
[lora]   too_short      n=47   总logprob胜率=0.000  长度归一胜率=0.404  len(chosen)=72.0 len(rejected)=6.3
[lora]   unrelated      n=66   总logprob胜率=0.576  长度归一胜率=0.545  len(chosen)=67.4 len(rejected)=69.3
[lora]   wrong_entity   n=44   总logprob胜率=0.727  长度归一胜率=0.409  len(chosen)=69.5 len(rejected)=70.1
```

**训练中**（r=4, alpha=8, 注入 q/v, lr=1e-3, 1500 步）：

```
[lora] step    0/1500 loss 8.5754 lr 1.00e-03
[lora] step  400/1500 loss 5.8941 lr 1.00e-03
[lora] step  850/1500 loss 5.7164 lr 1.00e-03
[lora] step 1450/1500 loss 5.7050 lr 1.00e-03
```

**AFTER**：

```
[lora] AFTER   win_rate=0.605  win_rate_per_token=0.745  mean_logp(chosen)=-385.99
[lora]   noisy          n=43   总logprob胜率=0.767  长度归一胜率=0.767  len(chosen)=71.4 len(rejected)=86.9
[lora]   too_short      n=47   总logprob胜率=0.000  长度归一胜率=0.128  len(chosen)=72.0 len(rejected)=6.3
[lora]   unrelated      n=66   总logprob胜率=0.667  长度归一胜率=1.000  len(chosen)=67.4 len(rejected)=69.3
[lora]   wrong_entity   n=44   总logprob胜率=1.000  长度归一胜率=1.000  len(chosen)=69.5 len(rejected)=70.1
[lora] initial_loss=8.5754
[lora] 胜率 0.515 -> 0.605 (随机基线 0.5)，长度归一口径 0.400 -> 0.745
[lora] saved -> checkpoints\lora.pt
```

**怎么读这组数**：

- 总 logprob 口径：0.515 → 0.605（+0.09）
- 长度归一口径：0.400 → **0.745**（+0.345）← **真实提升在这里**
- `mean_logp(chosen)`：-603.26 → -385.99（模型对 chosen 的信心明显提高）
- `too_short` 的长度归一胜率从 0.404 **掉到 0.128** —— 不是模型变差了，
  而是模型学会了「认真写满一段」，而 6 个 token 的短回复在长度归一口径下
  每 token logprob 极高（可预测），所以更"像"赢家。**口径的偏向被放大了。**
- `wrong_entity` 0.727 → **1.000**、`unrelated` 0.576 → 0.667 —— 内容型的错误确实被修正了。

> ⚠️ **这个胜率是「训练内指标」**：合成偏好数据只有 5 个 prompt × 5 个 chosen，
> 训练集与评估集同分布。**它证明的是「LoRA 确实让模型更偏向 chosen」，
> 不能当成泛化能力。**

### 6.3 消融：低秩约束 + 注入点决定「能学到什么」

见 §7。

### 6.4 ★ 一个静默的 dropout 覆盖 bug（以及它逼出来的一次全量重测）

**怎么发现的**：做 feat-024（DPO）的消融时，同一份配方跑两遍，
`margin` 差了 0.028（+0.1859 vs +0.1582）。既然 `train/train_dpo.py` 自己重跑两遍
结果**完全一致**，那差异就只能来自「消融脚本多做了一点事」——
消融脚本在训练前多跑了一次 `win_flags`（200 次前向）。

**根因**：`Qwen3Dense.__init__` 算了覆盖值 `d` 并用它建 `self.drop`（embedding 的 Dropout），
但 `Block` 里写的是 `dropout=cfg.dropout` —— **覆盖值根本没传到注意力/MLP**。
于是 `train_lora.py` / `train_dpo.py` 里的 `dropout=0.0` 只关掉了 embedding 的 dropout，
**注意力与 MLP 一直是 `cfg.dropout = 0.1`**。

**为什么它会毁掉可复现性**：

```python
# 实测：nn.Dropout 只有 p>0 且处于 train 模式时才推进全局 torch RNG
Dropout(0.0) train 模式推进 RNG: False
Dropout(0.0) eval  模式推进 RNG: False
Dropout(0.1) train 模式推进 RNG: True
```

于是「训练前多做 200 次 train 模式的前向」= 多消耗 200 份随机数 ⇒
**训练轨迹变了**。这才是那 0.028 的来源 —— 不是 DPO 的随机性，是 dropout 的随机性。

**修法**：`Block` / `MoEBlock` 接受 `dropout` 覆盖参数，模型把 `d` 传下去。
修完之后所有 `Dropout` 的 `p` 都是 0 ⇒ train 模式的前向不再推进 RNG ⇒
**整个实验对「评估跑了几次」免疫**（`win_flags` 放不放 eval 模式都一样）。

**修完之后的全量重测**（A/B/C 三种配置，同样的 1500 步 LoRA）：

| 配置 | 注意力/MLP dropout | embedding dropout | win_rate | 长度归一胜率 | mean_logp(chosen) | initial_loss |
|---|---|---|---|---|---|---|
| A 修复前的现状 | 0.1 | 0.0 | 0.595 | 0.745 | -389.48 | 8.4630 |
| **B 修好后（本仓库采用）** | **0.0** | **0.0** | **0.605** | 0.745 | **-385.99** | **8.5754** |
| C 修好后但全用 0.1 | 0.1 | 0.1 | 0.580 | 0.735 | -401.18 | 8.2925 |

选 **B** 的理由：

- 训练脚本本来就写着 `dropout=0.0`（作者意图），修好后**代码与意图一致**；
- SFT / DPO 阶段数据少、要求确定性，**关掉 dropout** 更合适；
- 预训练阶段继续用 `cfg.dropout = 0.1` 正则化（`train_pretrain.py` 显式传 `cfg.dropout`），
  **预训练结果一个字节都没变**：`initial_loss=6.7154 → val_loss=1.3510 val_ppl=3.86`（重跑核对过）。

顺带一个副产品：**A 配置下各配方的 `initial_loss` 是不一样的**（8.463 / 8.415 / 8.348 / 8.450），
因为第 0 步的前向带 dropout，loss 是随机的；修好之后所有配方的 `initial_loss` 都是 **8.575**，
一模一样 —— 这本身就是「dropout 真的关掉了」的可运行证据。

**回归断言**（写在 `tests/test_dense.py` 第 13 条 / `tests/test_moe.py` 末尾）：

```python
# 结构断言：dropout=0.0 时所有 Dropout 的 p 都必须是 0
ps = sorted({m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)})
assert ps == [0.0]
# 行为断言：dropout=0 时 train 模式下两次前向必须逐元素相同
model.train()
assert torch.equal(model(idx)[0], model(idx)[0])
```

**两条断言缺一不可**：只写结构断言，将来有人把 `Block` 里的参数名改错、
但 `d` 恰好等于 `cfg.dropout` 时仍会漏过；只写行为断言，则在 `p` 很小（如 1e-9）时看不出来。

> **这一节的教训（也是本仓库对「静默 bug」的一次完整示范）**：
> 覆盖参数（`dropout` / `use_gqa` / `use_rope` / `use_qk_norm`）没传到子模块，
> **不会报错、不会崩，只是静默地用默认值继续跑**。
> 这次它伪装成「DPO 有随机性」，实际是「模型的 dropout 没关掉」。
> ⇒ **每个覆盖开关都要配一条等价性断言**（结构 + 行为各一条）。

## 7. 消融表

**同一份偏好数据、同一份预训练权重（`checkpoints/pretrain_dense.pt`）、同一份评估集（200 对），
只改配方。** 复现脚本：`E:/tmp/lora_ablation_v2.py`（临时脚本，不入库）。

| 配方 | loss | 总 logprob 胜率 | 长度归一胜率 | 可训练参数 |
|---|---|---|---|---|
| r=4 只注入 q/v，lr 1e-3，300 步 | 8.575 → 6.073 | 0.515 → 0.575 | 0.400 → **0.825** | 8,704 |
| r=4 只注入 q/v，lr 3e-4，1500 步 | 8.575 → 5.743 | 0.515 → 0.590 | 0.400 → 0.745 | 8,704 |
| **r=4 只注入 q/v，lr 1e-3，1500 步（本仓库默认）** | 8.575 → **5.517** | 0.515 → **0.605** | 0.400 → 0.745 | 8,704 |
| r=16 注入 q/v + MLP 三矩阵，lr 1e-3，300 步 | 8.575 → 4.868 | 0.515 → 0.620 | 0.400 → 0.745 | 133,120 |
| r=16 注入 q/v + MLP 三矩阵，lr 1e-3，1500 步 | 8.575 → **3.964** | 0.515 → **0.660** | 0.400 → **0.865** | 133,120 |
| r=64 注入 q/v + MLP 三矩阵，lr 1e-3，300 步 | 8.575 → 4.174 | 0.515 → 0.615 | 0.400 → **0.865** | 532,480 |
| r=64 注入 q/v + MLP 三矩阵，lr 3e-3，300 步（**发散**） | 8.575 → 6.257 | 0.515 → 0.520 | 0.400 → 0.665 | 532,480 |
| **全参微调（对照）**，lr 1e-3，300 步 | 8.575 → **0.744** | 0.515 → **0.715** | 0.400 → 0.835 | 985,856 |

> 这张表是**修完 §6.4 的 dropout 覆盖 bug 之后**重测的（修复前的那张见 git 历史）。
> 结论方向没变，具体数字都变了一轮 —— 这就是「结论必须挂着实跑输出」的代价。

四条结论：

### 7.1 低秩约束 + 注入点决定「能学到什么」

全参微调 300 步就能把 loss 打到 **0.744**，而最好的 LoRA 配方（r=16 + MLP，1500 步）只能到 **3.964**。

差 5 倍多。这不是「LoRA 没调好」，而是**任务类型决定的**：

- 本任务是**内容型**的 —— 要模型说出某段特定的文本，**需要新增信息**（记住内容）；
- LoRA 的低秩旁路**只能表达「已有功能的重新加权」**，很难凭空写入新内容；
- 全参微调有 985,856 个自由度可以写新东西，LoRA r=4 只有 8,704 个。

> **LoRA 是风格 / 格式型任务的主场（改语气、换输出格式、适配新领域风格），
> 内容型任务上明显吃力。** 这条结论是本仓库实测出来的，不是从论文里搬的。

### 7.2 步数不够是主要瓶颈（而不是学习率）

r=4 从 300 步加到 1500 步：loss 6.073 → 5.517，总 logprob 胜率 0.575 → 0.605。
**步数翻了 5 倍，loss 只降了 0.56** —— 曲线很平，说明瓶颈在「能表达的信息量」，
不在「没走够路」。

### 7.3 学习率：1e-3 比 3e-4 好，但 3e-3 会发散

同样 r=4 / attn / 1500 步：lr 3e-4 → loss 5.743，lr 1e-3 → **5.517**。
而 r=64 时 lr 3e-3 直接发散（loss 6.257，胜率 0.520，等于没训）。

**秩越大，允许的学习率上限越低** —— 因为 ΔW = BA 的梯度里带着 1/r 因子，
r 大时同样的 lr 相当于更大的实际步长。

据此 `config.py` 定为 `lora_lr = 1e-3`、`lora_max_iters = 1500`
（**r=4 / alpha=8 / 注入点 (q_proj, v_proj) 保持原样**）。

### 7.4 ★ 两个口径对配方的排序不一样（这才是本次最扎人的发现）

看 r=4 的前两行：

| | loss | 总 logprob 胜率 | 长度归一胜率 |
|---|---|---|---|
| r=4, 300 步 | 6.073（更差） | 0.575（更差） | **0.825（更好）** |
| r=4, 1500 步 | **5.517（更好）** | **0.605（更好）** | 0.745（更差） |

**训得更久的那个，loss 和总 logprob 口径都更好，长度归一口径反而更差。**

原因见 §5.2：长度归一口径偏袒「可预测」的文本。
训得久 → 模型对 chosen 的每 token 概率都更高 → 输出的**风格更稳、更可预测** → 长度归一胜率更高。
但同时，训得久也让模型在**总 logprob** 上更占便宜（长度相同的两个回复，谁更"像训练目标"谁赢）。

> **所以「长度归一胜率更高」≠「模型更好」。**
> **这说明评估口径本身是需要交代的事 —— 换一组超参，两个口径能给出相反的排名。**
>
> 也正因如此，DPO 不直接用「谁的 logprob 大」当奖励，而是用
> **相对 π_ref 的 logprob 差值** —— 差值把「长度」「风格稳不稳」这类**静态偏置**
> 同时减掉，剩下的才是「相对参考模型更偏好了谁」。

**本仓库的默认口径**：`win_rate`（总 logprob）为主 —— 因为 **DPO 用的正是这个口径**，
`win_rate_per_token` 作为辅助观测（用来监控长度偏置），并始终**按策略分组打印**。

## 8. DPO —— 从零手写

代码：`qwen3-learning/qwen3-pytorch/train/train_dpo.py`、
`qwen3-learning/qwen3-pytorch/tests/test_dpo.py`

### 8.1 为什么 DPO 可以不要奖励模型

经典 RLHF（也就是 GPT-2 线走的路）要四步：

```text
SFT  →  采样  →  训 Reward Model  →  PPO（带 KL 约束地优化 RM 的分数）
```

DPO（Direct Preference Optimization, Rafailov et al. 2023）的想法是：
**既然最终只用「偏好对」训练，那能不能直接把偏好对的似然写出来，跳过 RM 和 PPO？**

答案是可以，而且推导只有三步。

### 8.2 目标函数推导

RLHF 的目标是

\[
\max_\pi\ \mathbb{E}_{x,y\sim\pi}[r(x,y)] - \beta\, \mathrm{KL}(\pi\Vert\pi_{ref})
\]

带 KL 约束的最优解有闭式：

\[
\pi^*(y\mid x) = \pi_{ref}(y\mid x)\,\frac{\exp(r(x,y)/\beta)}{Z(x)}
\]

反解出奖励：

\[
r(x,y) = \beta\log\frac{\pi^*(y\mid x)}{\pi_{ref}(y\mid x)} + \beta\log Z(x)
\]

`log Z(x)` 与 y 无关，是常数，训练时直接丢掉。
再把 Bradley-Terry 的偏好模型代进去（\(P(y_w \succ y_l) = \sigma(r(x,y_w)-r(x,y_l))\)）：

\[
P(y_w \succ y_l) = \sigma\Big(\beta\big[\log\tfrac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\tfrac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\big]\Big)
\]

对它做极大似然（把「chosen 更好」当成标签 1），得到 DPO 的损失：

\[
\boxed{
L_{DPO} = -\mathbb{E}\Big[\log\sigma\Big(\beta\big[(\log\pi_\theta(y_w)-\log\pi_{ref}(y_w)) - (\log\pi_\theta(y_l)-\log\pi_{ref}(y_l))\big]\Big)\Big]
}
\]

**注意括号里是差值。** 这一步是 DPO 的全部魔法所在：

- 不需要采样（不用从策略里 rollout）
- 不需要显式的奖励模型（奖励被**隐式**地表达成 `β·log(π_θ/π_ref)`）
- 不需要 PPO（没有 KL 约束的优化过程，KL 是被解析地写进损失里的）
- 只需要：策略模型、参考模型、偏好对

### 8.3 隐式奖励与 reward margin

把上面的奖励写出来：

\[
r_\theta(y\mid x) = \beta\big(\log\pi_\theta(y\mid x) - \log\pi_{ref}(y\mid x)\big)
\]

**它是个相对量** —— 「相对参考模型，这个序列的概率高了多少」。

于是 DPO 的判据就是两个隐式奖励之差：

\[
\text{margin} = r_\theta(y_w) - r_\theta(y_l) = \beta\big[(\Delta_w^\theta - \Delta_w^{ref}) - (\Delta_l^\theta - \Delta_l^{ref})\big]
\]

`margin > 0` ⇒ 策略相对参考模型**更偏爱 chosen**。这正是 `train_dpo.py` 里打印的量。

**「相对」两个字是本节所有结论的根**：
margin 变大 **不等于** 绝对 logprob 排序改变。§8.7 用两组实测把这句话钉死 ——
一组是「margin 涨了但绝对胜率一动不动」，另一组是「两者一起涨」。

### 8.4 ★ π_ref 是谁？——本节最容易搞错、也最不容易被发现的地方

标准 DPO 里，**π_ref 就是 SFT 模型**：DPO 从 SFT 出发，KL 项把它锚在 SFT 上，
不让它为了讨好偏好数据而跑偏。

本仓库「不加载第二份权重」的技巧本来是这样写的：

```python
set_lora_enabled(model, True)    # π_θ
set_lora_enabled(model, False)   # π_ref
```

**但这里有个陷阱**：LoRA 微调阶段 base 权重全程没被改过，所以「关掉分支」= **预训练模型**。
拿它当 DPO 的 π_ref，锚的就不是 SFT 模型，而是**没微调过的那个**。
于是 DPO 的 KL 项会跟刚做完的 LoRA SFT 互相拉扯
（SFT 刚把 chosen 的 logprob 抬上去，DPO 的 KL 项又往回压）。

**解法：三步把 π_ref 变出来。**

```text
1. inject_lora → load lora.pt        # 此时模型 = 预训练 base + SFT 的 A/B
2. merge_lora                       # ΔW 折进 base ⇒ 模型 == SFT 模型
3. strip_lora → inject_lora(新的)    # base 保持 SFT 权重，新 LoRA 只承载 DPO 的增量
```

之后 `set_lora_enabled(model, False)` 就精确等于 **SFT 模型**。

- `merge_lora`：把 \(\frac{\alpha}{r}BA\) 加进 base，并把分支的 `scaling` 置 0（否则算两遍，见 §3.3）
- `strip_lora`（本次新增）：把 `LoRALinear` 换回普通 `nn.Linear`，**保留 base 权重**

第 3 步 inject 的新 LoRA **B 零初始化** ⇒ 训练开始时 \(\pi_\theta \equiv \pi_{ref}\)。

> **这个 bug 初稿自己看不出来**：代码跑得通、loss 也降、指标也涨，
> 只是整个实验**锚在错的基线上**。
> 所以「π_ref 到底是什么」必须有一条**可运行的断言**来钉 —— 本仓库用的是「初始 margin 恒为 0」。

### 8.5 初始状态的不变量

因为 B 零初始化 ⇒ \(\pi_\theta \equiv \pi_{ref}\) ⇒

\[
\text{margin} = 0, \qquad L_{DPO} = -\log\sigma(0) = \log 2 = 0.6931
\]

这条不变量在本仓库被守了**三次**：

| 位置 | 断言 |
|---|---|
| `train/train_dpo.py` | 训练前 `evaluate_dpo` 的 `margin_mean` 必须 \|\|< 1e-4 |
| `tests/test_dpo.py` 第 3 组 | 初始 margin = 0 且 loss = log 2 |
| `init.sh` 的 DPO smoke | 同上 |

实测（**修完 §6.4 的 dropout bug 之后，训练完全确定**）：

```
[dpo] BEFORE  margin=+0.0000 ± 0.0000  dpo_acc=0.000  win_rate=0.605  win_rate_per_token=0.745
[dpo] step    0/1500 loss 0.6931 margin +0.0000 (avg50 +0.0000)
```

第 0 步的 loss **恰好是 0.6931**、margin **恰好是 0.0000** —— 一个数都没抖动。
（修复前这里打印的是 `loss 0.7201 margin -0.0532`，因为 dropout 在动。）

### 8.6 实测结果

```bash
python train/train_dpo.py
```

```
[dpo] device=cpu max_iters=1500 beta=0.1 ref_mode=sft
[dpo] π_ref = SFT 模型：merge 8 个模块 / strip 8 个模块，base 已含 SFT 的 ΔW
[dpo] trainable params = 8,704 / 994,560 (0.88%)
[dpo] preference pairs = 2000
[dpo] BEFORE  margin=+0.0000 ± 0.0000  dpo_acc=0.000  win_rate=0.605  win_rate_per_token=0.745
[dpo]   noisy          n=43   dpo_acc=0.000  margin=+0.0000  win_rate=0.767
[dpo]   too_short      n=47   dpo_acc=0.000  margin=+0.0000  win_rate=0.000
[dpo]   unrelated      n=66   dpo_acc=0.000  margin=+0.0000  win_rate=0.667
[dpo]   wrong_entity   n=44   dpo_acc=0.000  margin=+0.0000  win_rate=1.000
[dpo] step    0/1500 loss 0.6931 margin +0.0000 (avg50 +0.0000)
[dpo] step  300/1500 loss 0.0106 margin +4.5396 (avg50 +0.8348)
[dpo] step  600/1500 loss 0.6931 margin +0.0000 (avg50 +3.3949)
[dpo] step  900/1500 loss 2.0980 margin -1.9671 (avg50 +4.2674)
[dpo] step 1200/1500 loss 0.0617 margin +2.7540 (avg50 +4.6878)
[dpo] AFTER   margin=+4.6512 ± 4.3952  dpo_acc=0.780  win_rate=0.665  win_rate_per_token=0.900
[dpo]   noisy          n=43   dpo_acc=0.767  margin=+4.1993  win_rate=0.767
[dpo]   too_short      n=47   dpo_acc=0.277  margin=-0.4568  win_rate=0.000
[dpo]   unrelated      n=66   dpo_acc=1.000  margin=+8.0222  win_rate=0.848
[dpo]   wrong_entity   n=44   dpo_acc=1.000  margin=+5.4928  win_rate=1.000
[dpo] margin 首50均 +0.1230 → 末50均 +3.2850
[dpo] saved -> checkpoints\dpo.pt
```

**先看一个交叉验证**：`BEFORE` 的 `win_rate = 0.605`，
与 §6.2 里 LoRA SFT 跑完后的 `0.605` **完全一致**。

这说明 **merge → strip 的往返没有丢掉 SFT 学到的东西** ——
SFT 模型被完整地搬进了 DPO 的 base 里。若 merge / strip 有半点误差，这个数会漂。

### 8.7 ★ margin 和绝对胜率是两回事（两组实测）

**第一组：margin 涨了，绝对胜率一动不动。**

消融表里 `300 步 / lr 1e-4` 那一行：margin 从 0 涨到 **+0.2831**、dpo_acc 从 0 涨到 **0.945**，
可绝对 `win_rate` 是 **0.605 → 0.605**，逐对比较下来**翻转 0 / 200**。

**第二组：两者一起涨。**

同一份数据、同一份 SFT 权重，只把配方换成 `1500 步 / lr 1e-3`：
margin **+4.6512**、dpo_acc 0.780，而绝对 `win_rate` **0.605 → 0.665**、
长度归一 `0.745 → **0.900**`，**翻转 12 / 200**。

把两行放一起看：

| 配方 | reward margin | dpo_acc | 绝对 win_rate | 长度归一 win_rate | 翻转数 |
|---|---|---|---|---|---|
| 300 步 / lr 1e-4 | +0.2831 | 0.945 | 0.605 → 0.605 | 0.745 → 0.780 | **0**/200 |
| 1500 步 / lr 1e-3 | **+4.6512** | 0.780 | 0.605 → **0.665** | 0.745 → **0.900** | 12/200 |

> **结论：DPO 优化的是「相对 π_ref 的间隔」，不是「绝对排序」。**
> 间隔变大 ≠ 绝对胜率变大 —— 但**间隔够大时，绝对胜率一定会跟着动**。
> 第一行不是「DPO 没用」，是**配方太弱**（300 步、lr 1e-4，间隔只推了 0.28），
> 推不动那个被长度偏置主导的绝对分布。
>
> ⇒ **报告 DPO 结果必须同时给 reward margin（目标口径）与绝对 win_rate（业务口径）**：
> 只给前者会自我欺骗（第一行 dpo_acc 0.945 看着很美，业务上一条没动）；
> 只给后者会误判「没训好」（第二行 dpo_acc 只有 0.78，但业务指标涨了 0.06 + 0.155）。

**第三组观察：训得越狠，单对的 margin 越极端。**

看训练日志里的单对 margin：`+8.8754`、`-8.8753`、`+4.5396`、`-6.7379`……
最终 `margin = +4.6512 ± 4.3952` —— **标准差比均值还大**。
DPO 在 `lr=1e-3` 下是在**逐对硬拟合**（loss 打到 0.0001 的对，就是彻底记住了这一对）。
这不是 bug，但它解释了为什么 `dpo_acc` 反而比弱配方低：均值被少数几对拉高，
而 `dpo_acc` 统计的是「有多少对 margin > 0」。

**第四组观察：`too_short` _subset 被训坏了。**

| | dpo_acc | margin | 绝对 win_rate |
|---|---|---|---|
| `too_short`（n=47） | 0.000 → **0.277** | **-0.4568** | 0.000（恒定） |

弱配方下 `too_short` 的 dpo_acc 是 **1.000**（DPO 判得很对），
强配方下掉到 **0.277**（DPO 反而判错了 72%）。
而它的绝对 win_rate 从头到尾恒为 **0.000**（§5.4 的长度偏置，rejected 只有 6.3 token）。

**假设（未进一步验证）**：DPO 的梯度要求「chosen 相对参考模型更被偏爱」。
`too_short` 的 chosen 是 72 token 的完整回答、rejected 是 6.3 token 的残句；
参考模型（SFT 模型）本来就偏爱短文本，DPO 要压过这个先验就得把 chosen 的相对概率抬得很高，
抬到某个点之后**过冲**，反而把 chosen 的相对优势压成负的。

> **这与 §5.4 互为镜像**：§5.4 说 DPO 减 π_ref 能减掉长度偏置；
> 这里看到的是**减掉偏置之后，DPO 自己又会按新目标重新造一个偏置**。

### 8.8 消融：步数 / lr / β / π_ref 取法

同一份偏好数据、同一份 SFT 权重、同一份评估集（200 对），只改配方：

| π_ref | 步数 | lr | β | reward margin | dpo_acc | 绝对 win_rate | 翻转数 |
|---|---|---|---|---|---|---|---|
| sft | 300 | 0.0001 | 0.1 | +0.0000 → +0.2831 | 0.000 → 0.945 | 0.605 → 0.605 | 0 |
| sft | 1500 | 0.0001 | 0.1 | +0.0000 → +1.6338 | 0.000 → 0.710 | 0.605 → 0.630 | 5 |
| sft | 1500 | 0.001 | 0.1 | +0.0000 → +4.6512 | 0.000 → 0.780 | 0.605 → 0.665 | 12 |
| sft | 1500 | 0.001 | 1.0 | +0.0000 → +8.4384 | 0.000 → 0.950 | 0.605 → 0.625 | 4 |
| sft | 1500 | 0.001 | 0.01 | +0.0000 → +1.0109 | 0.000 → 0.710 | 0.605 → 0.715 | 22 |
| pretrained | 1500 | 0.001 | 0.1 | +0.0000 → +8.4640 | 0.000 → 0.930 | 0.515 → 0.650 | 27 |
| sft | 1500 | 0.0003 | 0.1 | +0.0000 → +4.1526 | 0.000 → 0.715 | 0.605 → 0.665 | 12 |
| sft | 3000 | 0.001 | 0.1 | +0.0000 → +5.0101 | 0.000 → 0.920 | 0.605 → 0.665 | 12 |

四条结论：

**① β 越大，margin 越大，但绝对胜率不一定越好。**
β=1.0 时 margin 冲到 **+8.44**、dpo_acc 0.950，可绝对胜率只有 0.625（翻转 4/200）；
β=0.1 时 margin +4.65、绝对胜率 0.665（翻转 12/200）；
β=0.01 时 margin 只 +1.01（KL 太松，间隔被摊薄），但绝对胜率最高 0.715（翻转 22/200）。
**margin 的大小和业务指标不是一回事**（同 §8.7）。

**② lr 1e-3 明显优于 1e-4。**
同样 1500 步 / β=0.1：lr 1e-4 → margin +1.63 / 胜率 0.630；lr 1e-3 → margin +4.65 / 胜率 0.665。

**③ 步数不够 → margin 涨了但业务指标不动。**
300 步 / lr 1e-4：margin +0.28、dpo_acc 0.945，但胜率 0/200 翻转。
1500 步 / lr 1e-4：margin +1.63、翻转 5/200。

**④ ★ π_ref 用 SFT 模型 vs 用预训练模型：结论出人意料。**
标准做法是锚 SFT 模型。但实测里 **π_ref = 预训练模型**（跳过 SFT 直接 DPO）那一行
把绝对胜率从 **0.515 拉到 0.650**（翻转 **27/200**，全表最多），
而 π_ref = SFT 模型那行是 0.605 → **0.665**（翻转 12/200）。

**两个都能把业务指标推上去，终点几乎一样（0.650 vs 0.665）。**
也就是说：在本仓库这个合成数据的小场景里，**SFT 这一步不是 DPO 的必要前提** ——
DPO 自己就能把模型从「预训练」推到「偏好正确」。

> **但这不代表「可以不做 SFT」**。两行的 margin 不可直接比
> （π_ref 不同 ⇒ 间隔的参照系不同），而且 `ref=pretrained` 那一行跳过了整个 SFT 阶段，
> 它的 `win_rate` 起点 0.515 是**预训练模型**的分数，与 SFT 后的 0.605 不是同一个基线。
> 真正的结论是：**在这个小场景里 SFT 与 DPO 各自能独立地把偏好做对；
> 标准流程（SFT → DPO）依然更稳，因为它给 DPO 提供了一个已经「会说人话」的起点。**
> 本仓库的默认仍取 `--ref-mode sft`（标准做法）。

据此 `config.py` 定为 `dpo_lr = 1e-3`、`dpo_max_iters = 1500`、`dpo_beta = 0.1`。

### 8.9 与 GPT-2 线的路线对比

| | GPT-2 线 | Qwen3 线 |
|---|---|---|
| 监督微调 | 全参 SFT | **LoRA**（只训 0.88% 的参数） |
| 偏好数据 | SFT 模型采样 → 启发式打分 | 合成偏好对（prompt / chosen / rejected 三段分开存） |
| 奖励模型 | **要**（显式训一个标量 RM） | **不要**（DPO 直接吃偏好对） |
| 策略优化 | PPO（采样 + KL 约束 + 价值网络） | **DPO**（一次前向、一次反向） |
| 采样 | PPO 必须 rollout | **不需要** |
| 参考模型 | 单独存一份冻结权重 | **同一个模型对象**（关掉 LoRA 分支） |
| 显存开销 | 策略 + RM + 价值 + 参考 ≈ 4 份 | 1 份 |

DPO 省掉的不是一点算力，是**一整条流水线**（采样 + RM 训练 + PPO 的 KL 约束调参）。

---

## 9. 本节的可迁移教训

1. **旁路注入类框架里，「哪些参数被冻结」「旁路合并后是否关掉」「loss 算在哪一段」
   三件事都是结构性的** —— 初稿三条全错，而且其中两条**不报错**，
   只会静默地给出错误结论。**结构性 bug 必须用「等价性断言」守，不能靠看 loss 曲线。**
2. **评估口径决定结论**。同一个模型、同一份数据，换个口径结论能反过来
   （0.515→0.605 vs 0.400→0.745）；甚至**两个口径能对同一组超参给出相反的排名**（§7.4）。
   **评估指标被某个子集系统性压死时，要把子集拆开单独看**，
   否则天花板效应会掩盖一切进展。
3. **参数量的宣称必须打印出来**。「只训了 0.88%」不是形容词，是
   `8,704 / 994,560` 这个比值。没有这个数字，读者无法判断是不是真的只训了 A/B。
4. **★ 「参考模型是哪一个」是最隐蔽的 bug**。DPO 里锚错 π_ref 不会让代码崩、
   不会让 loss 变 NaN、指标还会照涨 —— 只是整个实验锚在错的基线上。
   **所以「π_ref 到底是什么」必须有可运行断言来钉**（本仓库用「初始 margin 恒为 0」）。
5. **★ 目标指标与业务指标必须同时报**。DPO 的 reward margin 从 0 涨到 +4.65、
   dpo_acc 从 0 涨到 0.78，绝对胜率 0.605 → 0.665、长度归一 0.745 → 0.900；
   但**弱配方下 margin 涨了、dpo_acc 涨到 0.945，绝对胜率却是 0/200 翻转**。
   **只报 margin 是自我欺骗，只报胜率会误判「没训好」。**
6. **★ 覆盖参数没传到子模块 = 静默的半配置**。`dropout=0.0` 只关掉了 embedding 的 dropout，
   注意力/MLP 一直是 0.1 —— 于是 train 模式的前向持续推进全局 RNG，
   「训练前多做几次评估」就会改变训练轨迹（§6.4，同一配方 margin 差了 0.028）。
   **每个覆盖开关都要配一条等价性断言（结构 + 行为各一条）。**
7. **★ 同一个 bug 会伪装成别的问题**。§6.4 那个 dropout bug，
   一开始表现为「DPO 消融两次结果对不上」—— 看起来像 DPO 本身有随机性。
   **当实验「不可复现」时，先怀疑配置与随机性来源，再怀疑算法。**

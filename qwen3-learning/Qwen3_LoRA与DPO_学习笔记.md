# Qwen3-LoRA 与 DPO 学习笔记

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
[lora] step  400/1500 loss 6.0329 lr 1.00e-03
[lora] step  850/1500 loss 5.7365 lr 1.00e-03
[lora] step 1450/1500 loss 5.7798 lr 1.00e-03
```

**AFTER**：

```
[lora] AFTER   win_rate=0.595  win_rate_per_token=0.745  mean_logp(chosen)=-389.48
[lora]   noisy          n=43   总logprob胜率=0.767  长度归一胜率=0.767  len(chosen)=71.4 len(rejected)=86.9
[lora]   too_short      n=47   总logprob胜率=0.000  长度归一胜率=0.128  len(chosen)=72.0 len(rejected)=6.3
[lora]   unrelated      n=66   总logprob胜率=0.667  长度归一胜率=1.000  len(chosen)=67.4 len(rejected)=69.3
[lora]   wrong_entity   n=44   总logprob胜率=0.955  长度归一胜率=1.000  len(chosen)=69.5 len(rejected)=70.1
[lora] initial_loss=8.4630
[lora] 胜率 0.515 -> 0.595 (随机基线 0.5)，长度归一口径 0.400 -> 0.745
[lora] saved -> checkpoints\lora.pt
```

**怎么读这组数**：

- 总 logprob 口径：0.515 → 0.595（+0.08）
- 长度归一口径：0.400 → **0.745**（+0.345）← **真实提升在这里**
- `mean_logp(chosen)`：-603.26 → -389.48（模型对 chosen 的信心明显提高）
- `too_short` 的长度归一胜率从 0.404 **掉到 0.128** —— 不是模型变差了，
  而是模型学会了「认真写满一段」，而 6 个 token 的短回复在长度归一口径下
  每 token logprob 极高（可预测），所以更"像"赢家。**口径的偏向被放大了。**
- `wrong_entity` 0.727 → 0.955、`unrelated` 0.576 → 0.667 —— 内容型的错误确实被修正了。

> ⚠️ **这个胜率是「训练内指标」**：合成偏好数据只有 5 个 prompt × 5 个 chosen，
> 训练集与评估集同分布。**它证明的是「LoRA 确实让模型更偏向 chosen」，
> 不能当成泛化能力。**

### 6.3 消融：低秩约束 + 注入点决定「能学到什么」

见 §7。

## 7. 消融表

**同一份偏好数据、同一份预训练权重（`checkpoints/pretrain_dense.pt`）、同一份评估集（200 对），
只改配方。** 复现脚本：`E:/tmp/lora_ablation_v2.py`（临时脚本，不入库）。

| 配方 | loss | 总 logprob 胜率 | 长度归一胜率 | 可训练参数 |
|---|---|---|---|---|
| r=4 只注入 q/v，lr 1e-3，300 步 | 8.463 → 6.121 | 0.515 → 0.570 | 0.400 → **0.820** | 8,704 |
| r=4 只注入 q/v，lr 3e-4，1500 步 | 8.463 → 5.926 | 0.515 → 0.590 | 0.400 → 0.810 | 8,704 |
| **r=4 只注入 q/v，lr 1e-3，1500 步（本仓库默认）** | 8.463 → 5.701 | 0.515 → **0.595** | 0.400 → 0.745 | 8,704 |
| r=16 注入 q/v + MLP 三矩阵，lr 1e-3，300 步 | 8.415 → 5.319 | 0.515 → 0.605 | 0.400 → 0.730 | 133,120 |
| r=16 注入 q/v + MLP 三矩阵，lr 1e-3，1500 步 | 8.415 → **4.153** | 0.515 → **0.640** | 0.400 → **0.865** | 133,120 |
| r=64 注入 q/v + MLP 三矩阵，lr 1e-3，300 步 | 8.348 → 4.460 | 0.515 → 0.575 | 0.400 → 0.800 | 532,480 |
| r=64 注入 q/v + MLP 三矩阵，lr 3e-3，300 步（**发散**） | 8.348 → 6.258 | 0.515 → 0.520 | 0.400 → 0.655 | 532,480 |
| **全参微调（对照）**，lr 1e-3，300 步 | 8.450 → **1.217** | 0.515 → **0.715** | 0.400 → 0.835 | 985,856 |

四条结论：

### 7.1 低秩约束 + 注入点决定「能学到什么」

全参微调 300 步就能把 loss 打到 **1.217**，而最好的 LoRA 配方（r=16 + MLP，1500 步）只能到 **4.153**。

差 3 倍多。这不是「LoRA 没调好」，而是**任务类型决定的**：

- 本任务是**内容型**的 —— 要模型说出某段特定的文本，**需要新增信息**（记住内容）；
- LoRA 的低秩旁路**只能表达「已有功能的重新加权」**，很难凭空写入新内容；
- 全参微调有 985,856 个自由度可以写新东西，LoRA r=4 只有 8,704 个。

> **LoRA 是风格 / 格式型任务的主场（改语气、换输出格式、适配新领域风格），
> 内容型任务上明显吃力。** 这条结论是本仓库实测出来的，不是从论文里搬的。

### 7.2 步数不够是主要瓶颈（而不是学习率）

r=4 从 300 步加到 1500 步：loss 6.121 → 5.701，总 logprob 胜率 0.570 → 0.595。
**步数翻了 5 倍，loss 只降了 0.42** —— 曲线很平，说明瓶颈在「能表达的信息量」，
不在「没走够路」。

### 7.3 学习率：1e-3 比 3e-4 好，但 3e-3 会发散

同样 r=4 / attn / 1500 步：lr 3e-4 → loss 5.926，lr 1e-3 → **5.701**。
而 r=64 时 lr 3e-3 直接发散（loss 6.258，胜率 0.520，等于没训）。

**秩越大，允许的学习率上限越低** —— 因为 ΔW = BA 的梯度里带着 1/r 因子，
r 大时同样的 lr 相当于更大的实际步长。

据此 `config.py` 定为 `lora_lr = 1e-3`、`lora_max_iters = 1500`
（**r=4 / alpha=8 / 注入点 (q_proj, v_proj) 保持原样**）。

### 7.4 ★ 两个口径对配方的排序不一样（这才是本次最扎人的发现）

看 r=4 的前两行：

| | loss | 总 logprob 胜率 | 长度归一胜率 |
|---|---|---|---|
| r=4, 300 步 | 6.121（更差） | 0.570（更差） | **0.820（更好）** |
| r=4, 1500 步 | **5.701（更好）** | **0.595（更好）** | 0.745（更差） |

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

## 8. 与 DPO 的接口

DPO 需要：

- \(\pi_\theta\)：LoRA 分支**开**
- \(\pi_{ref}\)：LoRA 分支**关**（= 原始预训练模型，因为 base 从没被改过）

对应本仓库的实现：

```python
set_lora_enabled(model, True)    # π_θ
set_lora_enabled(model, False)   # π_ref
```

DPO 部分见 `train/train_dpo.py`（feat-024，待补）。

---

## 9. 本节的三条可迁移教训

1. **旁路注入类框架里，「哪些参数被冻结」「合并后分支是否关掉」「loss 算在哪一段」
   三件事都是结构性的** —— 初稿三条全错，而且其中两条**不报错**，
   只会静默地给出错误结论。**结构性 bug 必须用「等价性断言」守，不能靠看 loss 曲线。**
2. **评估口径决定结论**。同一个模型、同一份数据，换个口径结论能反过来
   （0.515→0.595 vs 0.400→0.745）。**评估指标被某个子集系统性压死时，
   要把子集拆开单独看**，否则天花板效应会掩盖一切进展。
3. **参数量的宣称必须打印出来**。「只训了 0.88%」不是形容词，是
   `8,704 / 994,560` 这个比值。没有这个数字，读者无法判断是不是真的只训了 A/B。

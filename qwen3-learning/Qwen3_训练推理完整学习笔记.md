# Qwen3 学习笔记：从训练到 KV Cache 推理，再到 GQA 的 Head 映射

> **对应 GPT-2 实现位置**：[`gpt-learning/gpt-learning-pytorch/train_pretrain.py`](../gpt-learning/gpt-learning-pytorch/train_pretrain.py) / [`gpt-learning/gpt-learning-pytorch/train_sft.py`](../gpt-learning/gpt-learning-pytorch/train_sft.py) / [`gpt-learning/gpt-learning-pytorch/inference.py`](../gpt-learning/gpt-learning-pytorch/inference.py) / [`gpt-learning/gpt-learning-pytorch/evaluate.py`](../gpt-learning/gpt-learning-pytorch/evaluate.py)
> **对应 Qwen3 源码位置**：[`qwen3-learning/qwen3-pytorch/train/train_pretrain.py`](qwen3-pytorch/train/train_pretrain.py) / [`qwen3-learning/qwen3-pytorch/train/train_lora.py`](qwen3-pytorch/train/train_lora.py) / [`qwen3-learning/qwen3-pytorch/train/train_dpo.py`](qwen3-pytorch/train/train_dpo.py) / [`qwen3-learning/qwen3-pytorch/inference.py`](qwen3-pytorch/inference.py) / [`qwen3-learning/qwen3-pytorch/evaluate.py`](qwen3-pytorch/evaluate.py) / [`qwen3-learning/qwen3-pytorch/ablations/compare_with_gpt2.py`](qwen3-pytorch/ablations/compare_with_gpt2.py) / [`qwen3-learning/qwen3-pytorch/quality.py`](qwen3-pytorch/quality.py) / [`qwen3-learning/qwen3-pytorch/ablations/compare_quality.py`](qwen3-pytorch/ablations/compare_quality.py)
> **对应可跑测试**：[`qwen3-learning/qwen3-pytorch/tests/test_pretrain.py`](qwen3-pytorch/tests/test_pretrain.py) / [`qwen3-learning/qwen3-pytorch/tests/test_inference.py`](qwen3-pytorch/tests/test_inference.py) / [`qwen3-learning/qwen3-pytorch/tests/test_evaluate.py`](qwen3-pytorch/tests/test_evaluate.py) / [`qwen3-learning/qwen3-pytorch/tests/test_compare_with_gpt2.py`](qwen3-pytorch/tests/test_compare_with_gpt2.py) / [`qwen3-learning/qwen3-pytorch/tests/test_quality.py`](qwen3-pytorch/tests/test_quality.py)

> 学习目标：把 Transformer / GPT 基础与 Qwen3 的 RMSNorm、QK-Norm、RoPE、GQA、SwiGLU 和 KV Cache 串起来，重点跟踪 **Tensor Shape**。本笔记是我们多轮问答的综合版。示例是**微型 Qwen3 风格模型**，用于讲解机制，并不是声称能复现真实 Qwen3 的训练结果。

## 0. 阅读约定和最重要的 Shape

统一用两种维度顺序：

- 隐藏状态：`[B,T,C]`（Batch、Token/Sequence、Hidden）。
- Attention 按头拆分：`[B,H,T,D]`（Batch、Heads、Token/Sequence、每头维度）。

需要分清三个不同的数量：**Token 数 T、Query Head 数 Hq、KV Head 数 Hkv**。KV Head 少，不代表历史 Token 少。

普通全局 Causal Attention 的逻辑 Score Shape：

- 训练：`[B,Hq,T,T]`；
- Prefill：`[B,Hq,Tprompt,Tprompt]`；
- 单步 Decode：`[B,Hq,1,Tcache]`。

FlashAttention 等优化实现不一定在显存中实际生成整个 Score 矩阵；上述是**数学上的逻辑形状**。

---

## 1. 建立微型模型

为了能从头跑一遍，设置如下：

| 参数 | 数值 | 意义 |
|---|---:|---|
| `B` | 1 | 一条样本 |
| `vocab_size` | 6 | 六个 Token |
| `hidden_size = C` | 4 | 每个 Token 四维 |
| `n_layers = L` | 1 | 一个 Decoder Block |
| `Hq` | 2 | 两个 Query Head |
| `Hkv` | 1 | 一个 Key / Value Head |
| `D` | 2 | 每头两维 |
| `intermediate_size` | 8 | SwiGLU 中间特征数 |
| `n_ctx` | 8 | 例子中上下文容量 |

此例 `2Q/1KV` 也是 MQA（GQA 的单 KV Head 特例），共享/映射规律与普通 GQA 一致。定义词表：`我=0，爱=1，学习=2，AI=3，编程=4，EOS=5`。

单层整体：

```text
IDs → Embedding → RMSNorm → Q/K/V projection → per-head QK-Norm
   → RoPE(Q,K) → GQA causal attention → O projection → residual
   → RMSNorm → SwiGLU(gate/up/down) → residual → final RMSNorm
   → LM Head → vocabulary logits
```

对这个微型模型：`Q` 输出宽度 `Hq*D=4`，`K/V` 输出宽度 `Hkv*D=2`。三个线性层的权重若使用本文的数学写法 `X @ W`，分别是 `[4,4]`、`[4,2]`、`[4,2]`；**PyTorch `nn.Linear` 存储的 weight 转置为 `[out_features,in_features]`**，不要直接与数学写法混淆。

---

## 2. 第一阶段：训练 Training（一次并行计算整个序列）

### 2.1 Teacher forcing / shifted next-token labels

训练样本：`我 爱 学习 AI EOS`，可采用对齐的四步示例：

```text
input_ids   = [我, 爱, 学习, AI] = [0,1,2,3]   shape [1,4]
target_ids  = [爱, 学习, AI, EOS] = [1,2,3,5] shape [1,4]
```

每个位置预测下一个 Token。实际 CausalLM API 也常传完整序列，让实现内部 shift logits/labels；本例显式拆出输入/目标，避免混淆。

### 2.2 Embedding 与第一处 RMSNorm

Embedding table（若不讨论权重绑定）为 `[vocab,C]=[6,4]`。

```text
input_ids                  [1,4]
Embedding                  [1,4,4]
RMSNorm weight             [4]
RMSNorm output             [1,4,4]
```

四个 Token 各自沿最后一个长度为 4 的特征维计算一个 RMS；所有 Token 共用这 4 个可学习 weight。RMSNorm **不改变 Shape**：

\[
\operatorname{RMSNorm}(x)=w\odot\frac{x}{\sqrt{\frac1C\sum_{i=1}^C x_i^2+\epsilon}}
\]

### 2.3 Q/K/V 投影与拆头

```text
Input X                              [1,4,4]
Q = X @ Wq                           [1,4,4]
K = X @ Wk                           [1,4,2]
V = X @ Wv                           [1,4,2]

Reshape + transpose to [B,H,T,D]:
Q                                    [1,2,4,2]
K                                    [1,1,4,2]
V                                    [1,1,4,2]
```

`Hq=2`、`Hkv=1`，所以 Q/K/V 投影输出宽度**不要求相同**。Q-Norm、K-Norm 按各头的最后一个 `D=2` 维归一化，分别共享该类 Norm 的 `[2]` weight；RoPE 对 Q/K 加入位置 0、1、2、3 的旋转，**Shape 均不变**。V 不做 RoPE。

### 2.4 GQA 的训练 Attention

两只 Q Head 共享一只 KV Head。计算上可将 K/V **逻辑地**沿 Heads 维重复：

```text
Q                                    [1,2,4,2]
K before repeat                      [1,1,4,2]
K logically expanded                 [1,2,4,2]
K transpose(-2,-1)                   [1,2,2,4]
Q @ K.T                              [1,2,4,4]
```

`[4,2] @ [2,4] → [4,4]`；前两个维度 `[B,H]` 是 batch-like 维，**对应 Head 按索引逐一配对，不会自动把某个 Q 与所有 Head 的 K 做交叉组合**。

Causal Mask 使第 `t` 个位置只能看 `0..t`：

```text
        K0 K1 K2 K3
Q0      ✓  ×  ×  ×
Q1      ✓  ✓  ×  ×
Q2      ✓  ✓  ✓  ×
Q3      ✓  ✓  ✓  ✓
```

Score、Mask、Softmax 权重逻辑 shape 都为 `[1,2,4,4]`。乘 V：

```text
Attention weights                   [1,2,4,4]
V logically expanded               [1,2,4,2]
weights @ V                        [1,2,4,2]
heads concat (transpose + reshape) [1,4,4]
Output projection                  [1,4,4]
Residual                           [1,4,4]
```

### 2.5 SwiGLU、LM Head、Loss、Backward

```text
Second RMSNorm                     [1,4,4]
Gate projection + SiLU             [1,4,8]
Up projection                      [1,4,8]
Elementwise product                [1,4,8]
Down projection                    [1,4,4]
Residual                           [1,4,4]
Final RMSNorm                      [1,4,4]
LM Head (vocab=6)                  [1,4,6]
Targets                            [1,4]
Cross entropy                      scalar
```

四个位置各预测下一个 Token，四份有效位置的损失共同贡献梯度，`backward` 求梯度，`optimizer.step` 更新可训练权重。训练阶段通常**不使用跨生成步骤的增量 KV Cache**；这一轮 Forward 内计算出的 K/V 是用于本轮 Attention/反向传播的激活。Causal Mask 使模型虽并行处理整个序列，仍然不能偷看未来位置。

**训练总表：**

| 步骤 | 逻辑 Shape |
|---|---|
| IDs → Embedding | `[1,4] → [1,4,4]` |
| RMSNorm | `[1,4,4]` |
| Q / K / V | `[1,2,4,2]` / `[1,1,4,2]` / `[1,1,4,2]` |
| QK-Norm、RoPE | 不改变对应 Shape |
| QK Score（按组共享 K） | `[1,2,4,4]` |
| Mask + Softmax | `[1,2,4,4]` |
| Attention × V | `[1,2,4,2]` |
| Concat → O projection | `[1,4,4]` |
| SwiGLU intermediate → Down | `[1,4,8] → [1,4,4]` |
| Final Norm → LM Head | `[1,4,4] → [1,4,6]` |
| CE Loss | 标量 |

---

## 3. 第二阶段：推理 Inference，带 KV Cache

使用训练后固定的参数，不算 Loss、不反向传播。输入 Prompt `我 爱 学习`（3 Token），假设模型接着生成 `AI 编程 EOS`。下面跟踪缓存中**已经实际 Forward 过的 Token**，不是单纯已采样出的 Token。

### 3.1 Prefill：一次处理三个 Prompt Token

```text
IDs                                [1,3]
Embedding / RMSNorm                [1,3,4]
Q                                  [1,2,3,2]
K                                  [1,1,3,2]
V                                  [1,1,3,2]
QK-Norm / RoPE                     shape unchanged
Attention score                    [1,2,3,3]
Attention output after concat/O    [1,3,4]
LM Head logits                     [1,3,6]
Take logits[:, -1, :]              [1,6]
Sample next Token                  AI
```

此时 K/V 对三个 Prompt Token 缓存：

```text
K_cache (effective)                [1,1,3,2]
V_cache (effective)                [1,1,3,2]
含：我、爱、学习
```

**重要时间差**：Prefill 最后预测出了 `AI`，但 `AI` 还没有作为输入执行 Forward，因此此刻缓存有效长度仍是 **3**，尚未写入 AI 的 K/V。

### 3.2 Decode 第一步：输入刚生成的 AI

```text
AI input IDs                       [1,1]
Hidden states                      [1,1,4]
new Q                              [1,2,1,2]
new K / V                          [1,1,1,2] each
position ID                        3
```

对新 Q/K 做 QK-Norm、RoPE，把 **AI 的新 K/V** 追加到缓存：

```text
old cache                          [1,1,3,2]
new KV                             [1,1,1,2]
updated cache                      [1,1,4,2]
```

当前 Q 查询历史三个加当前一个、共四个 K：

```text
Q                                  [1,2,1,2]
K_cache                            [1,1,4,2]
Score                              [1,2,1,4]
Softmax                            [1,2,1,4]
Attention × V                      [1,2,1,2]
Concat / projection / MLP          [1,1,4]
LM Head                            [1,1,6]
```

假设预测出 `编程`，但它**还未写入** KV Cache；当前有效缓存长度 **4**。

### 3.3 Decode 第二步：输入编程

```text
Input                              [1,1]
new Q                              [1,2,1,2]
new K / V                          [1,1,1,2]
position ID                        4
updated K/V cache                  [1,1,5,2]
Score                              [1,2,1,5]
Logits                             [1,1,6]
```

预测 `EOS` 后结束，无需再将 EOS 喂入模型计算其 KV（普通终止流程）。

### 3.4 关键时间线

| 阶段 | 本轮实际输入 | Q | 本轮处理后有效 K/V 长度 | Score | 刚生成的 Token |
|---|---|---|---:|---|---|
| Prefill | 我、爱、学习 | `[1,2,3,2]` | 3 | `[1,2,3,3]` | AI |
| Decode 1 | AI | `[1,2,1,2]` | 4 | `[1,2,1,4]` | 编程 |
| Decode 2 | 编程 | `[1,2,1,2]` | 5 | `[1,2,1,5]` | EOS |

推理的特殊之处：**每一步通常只算新增一个 Token 的 Q/K/V，但新的 Q 可以查询缓存的全部有效历史 K/V**。所以数学上的 Score 是 `[B,Hq,1,Tcache]`。

---

## 4. GQA 的问答：为什么 Q Heads 多于 KV Heads 还可以算？

### Q1：假如 Q 有 16 Heads，KV 只有 4 Heads，谁和谁算？随意吗？

不是随意的，标准 GQA 按模型配置固定连续分组。每组：

\[
group\_size=H_q/H_{kv}=16/4=4
\]

使用从 **1 开始**的直观编号：

| Query Heads | 对应共享的 KV Head |
|---|---|
| Q1、Q2、Q3、Q4 | K1、V1 |
| Q5、Q6、Q7、Q8 | K2、V2 |
| Q9、Q10、Q11、Q12 | K3、V3 |
| Q13、Q14、Q15、Q16 | K4、V4 |

每个 Q **分别算一次 Attention**，而不是先把同组 Q 合并成一个。共享的是 K/V 数据，Q Head 的 Attention Score 和输出依旧可以不同。实际代码一般从 **0 开始**编号：

```python
kv_head_idx = q_head_idx // group_size
```

如索引 7 的 Q 使用 `7//4=1` 号 KV Head。

### Q2：先 Expand K，不就变成 Q 和 K 随意计算了吗？

不会。假设：

```text
Q Heads: Q1 Q2 Q3 Q4 | Q5 Q6 Q7 Q8
K Heads before expand: K1 K2
K Heads after repeat_interleave(4):
         K1 K1 K1 K1 | K2 K2 K2 K2
```

原始形状：

```text
Q: [1,8,3,2]
K: [1,2,3,2]
```

对 K 按 Head 维 `repeat_interleave(4, dim=1)`：`[1,8,3,2]`，转置最后两维：`[1,8,2,3]`。

`Q @ K.transpose(-2,-1)` 的前两个 `[B,H]` 是 **batch-like 维**，并不是用于各 Head 两两笛卡尔积的矩阵乘法维。第 `h` 个 Q **只与索引为 h 的扩展 K** 相乘：

```text
Q1@K1ᵀ  Q2@K1ᵀ  Q3@K1ᵀ  Q4@K1ᵀ
Q5@K2ᵀ  Q6@K2ᵀ  Q7@K2ᵀ  Q8@K2ᵀ
```

其输出 `[1,8,3,3]`，**不是** `[1,8,2,3,3]`；不产生 Q1×K2、Q5×K1 这种额外交叉 Head 配对。

### Q3：那 `[1,16,4,2]` 和 `[1,4,4,2]` 怎么算？

先显式扩展，分组大小 `16//4=4`：

```text
Q:            [1,16,4,2]
K:            [1, 4,4,2]
K repeated:   [1,16,4,2]
K transposed: [1,16,2,4]
Q @ K.T:      [1,16,4,4]
```

这里 `matmul` 自动逐批、逐 Head 配对，**但标准的 `matmul` 不会替你猜测 16Q/4KV 的分组规则；扩展必须先显式做，或调用支持 GQA 的专门 Attention API。**

V 也按同样规则扩展：

```text
attention weights: [1,16,4,4]
V repeated:        [1,16,4,2]
attention @ V:     [1,16,4,2]
transpose+reshape: [1, 4,32]
```

### Q4：逻辑上 repeat 四次会不会让 KV Cache 的节省白费？

示教代码中的 `repeat_interleave` 通常会产生重复的 Tensor；**高效 GQA Kernel 可以通过 Head 索引映射直接复用原始 K/V，而无须把缓存中的 KV 真实存成四份**。重复是理解数学配对的一种实现，绝非 GQA 必须采用的物理布局。低层算子是否产生临时缓冲区取决于具体后端。

### Q5：这套过程 PyTorch 自动完成哪些？

- `repeat_interleave(group_size, dim=1)`：开发者**指定如何按固定次序扩展**。
- `@`/`torch.matmul`：PyTorch 负责**相同 batch/head 索引的成批矩阵乘法**。
- 也可以考虑 `torch.nn.functional.scaled_dot_product_attention(..., enable_gqa=True)`，使用支持 GQA 的后端，并遵循其限制；这时无需自己显式重复 K/V。

### Q6：一份 K/V，共享后为什么两个 Q 输出还不同？

示意：

\[
Q_0=[1,0],\quad Q_1=[0,1]
\]

两个历史 Key 与 Value：

\[
K=\begin{bmatrix}1&0\\0&1\end{bmatrix},\quad
V=\begin{bmatrix}10&0\\0&20\end{bmatrix}
\]

两个 Q 各自对**同一套 K** 点积（以下仅示意可访问这两个 Key 的查询情形）：

\[
Q_0K^T=[1,0],\quad Q_1K^T=[0,1]
\]

除以 `√2` 并 Softmax：

\[
a_0\approx[0.670,0.330],\quad a_1\approx[0.330,0.670]
\]

再与同一套 V 相乘：

\[
O_0\approx[6.70,6.60],\quad O_1\approx[3.30,13.40]
\]

**共享 K/V，不共享 Q，也不共享最终 Attention 权重。**

---

## 5. 一份最小、可运行的 PyTorch GQA Shape 示例

这段用于演示分组与矩阵尺寸，不训练权重，也不假装随机数据经过 QK-Norm/RoPE。

```python
import torch
import torch.nn.functional as F

B, T, D = 1, 4, 2
Hq, Hkv = 16, 4
assert Hq % Hkv == 0
g = Hq // Hkv  # 4

q = torch.randn(B, Hq, T, D)
k = torch.randn(B, Hkv, T, D)
v = torch.randn(B, Hkv, T, D)

# 教学版：显式复制到对应的 Head 位置
k_exp = k.repeat_interleave(g, dim=1)
v_exp = v.repeat_interleave(g, dim=1)
score = q @ k_exp.transpose(-2, -1) / (D ** 0.5)

# 训练/Prefill 的 causal mask：未来 key 不可见
causal = torch.ones(T, T, dtype=torch.bool).tril()
score = score.masked_fill(~causal, float('-inf'))
attention = score.softmax(dim=-1)
out = attention @ v_exp

print('q', tuple(q.shape))          # (1, 16, 4, 2)
print('k', tuple(k.shape))          # (1,  4, 4, 2)
print('k_exp', tuple(k_exp.shape))  # (1, 16, 4, 2)
print('score', tuple(score.shape))  # (1, 16, 4, 4)
print('out', tuple(out.shape))      # (1, 16, 4, 2)

# 对应后端支持时，可直接：
# F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

若要模拟单步 Decode，只把本轮 `q` 的 Token 维设为 1，K/V 改成**包含当前及历史 Token 的有效缓存长度**，此时 Score 是 `[B,Hq,1,Tcache]`。不能把训练中的 `T×T` causal mask 机械地套到 `1×Tcache` 上；需要按正确的**绝对 position / cache position**构造可见性，或者在只有历史及当前、没有未来 Token 时直接允许访问所有有效 K。

---

## 6. 换成真实 Qwen3-0.6B 的 Shape

这里使用此前学习的经典 Qwen3-0.6B 参数：

```text
hidden_size = 1024
num_hidden_layers = 28
num_attention_heads = 16
num_key_value_heads = 8
head_dim = 128
```

这个模型尤其要注意：`Q 的总投影宽度 = 16×128 = 2048`，**并不等于** `hidden_size=1024`；K/V 总投影宽度各为 `8×128=1024`。以 `B=1`、Prefill Prompt 长 100 为例：

| 过程 | 单层 Tensor Shape |
|---|---|
| Hidden | `[1,100,1024]` |
| Q | `[1,16,100,128]` |
| K / V | 各 `[1,8,100,128]` |
| Score（逻辑） | `[1,16,100,100]` |
| Concat Attention Heads | `[1,100,2048]` |
| O Projection | `[1,100,1024]` |

生成第一个 Token 后，下一个 Decode step（当前 Token 真正被 Forward 处理）为：

| 过程 | 单层 Tensor Shape |
|---|---|
| Hidden | `[1,1,1024]` |
| Q | `[1,16,1,128]` |
| new K/V | 各 `[1,8,1,128]` |
| K/V 有效缓存（长度101） | 各 `[1,8,101,128]` |
| Score（逻辑） | `[1,16,1,101]` |
| Concat → O | `[1,1,2048] → [1,1,1024]` |

28 个层都有各自 K/V；表中只给出**一个层**的逻辑尺寸。llama.cpp 内部布局可以不采用相同的四维顺序。

---

## 7. KV Cache 内存量：GQA 为什么节省

对常规全局 Attention，暂不考虑分配器/对齐等开销、K/V 相同 dtype 时：

\[
M_{KV}=2\times L\times B\times T_{cache}\times H_{kv}\times D\times S
\]

`2` 代表 Key 和 Value 两份；`S` 是每元素字节数，如 FP16/BF16 为 2 bytes。关键是缓存大小随 **Hkv 而非 Hq** 缩放。若同样 16Q：

```text
MHA：16 KV Heads    → 基准 100%
GQA： 4 KV Heads    → 25%，减少 75%
MQA： 1 KV Head     → 6.25%，减少 93.75%
```

例如 Qwen3-0.6B 的 `L=28,B=1,T=4096,Hkv=8,D=128,S=2`，理论 K+V 总量为 **448 MiB**；假设只将对应 KV Heads 改为 16、其余相同，成为 **896 MiB**。这里讨论的是理论缓存数据容量，**不等于** GPU 进程全部显存、也不代表吞吐量按相同比例增加。

GQA 优化的是 KV 存储和读取；单步 Decode 仍然有多个独立 Q Head 计算、MLP、Projection、权重读取等成本。长上下文下 KV 读取可能成为显著的带宽瓶颈，GQA 因而有利于带宽受限场景。

**权重量化 ≠ KV Cache 量化**：模型权重用 NVFP4，并不意味着 KV Cache 自动采用 NVFP4。KV 的 dtype/格式由框架和配置决定；进一步把 KV Cache 从 FP16 量化为低位宽，可能减少缓存，但存在 scale 元数据、算子支持与精度取舍。

---

## 8. KV Cache 生命周期、满了怎么办

### 8.1 一次 Token / 一次请求结束会清理吗？

- **同一次 Decode 的一个 Token 结束**：一般保留历史 K/V，并追加新处理 Token 的 K/V。
- **整个请求结束**：可以释放或标记为可复用；多轮会话/公共前缀缓存机制也可能留下旧 KV。
- **缓存逻辑内容失效**，不意味着预分配的 GPU KV Buffer 一定马上归还给操作系统/驱动。许多服务器保留内存给后续请求复用。

第二轮对话能否复用上轮 KV，取决于**输入前缀匹配、槽位/缓存策略、位置一致性**等条件；不是只要属于同一会话就一定自动命中。

### 8.2 两种常混淆的“满”

| 情况 | 含义 | 可能的办法 |
|---|---|---|
| Context / KV Capacity 满 | 达到配置的可容纳 Token 数 | 停止、缩短上下文、在支持时 Context Shift/Sliding Window、增大 `n_ctx` |
| GPU OOM | 模型权重、KV、计算临时缓冲等总显存不够 | 减少并发/上下文、KV 量化、调整部署、调度抢占 |

Prefill Prompt 本身超过 Context：可以按应用策略拒绝或截断；`Chunked Prefill` 可缓解一次计算的工作内存/调度压力，**不能仅靠分块就让最终超容量的完整 KV Cache 装下**。

Decode 把缓存写满时：可以停止生成，或在适合的架构/框架中采用上下文移位或滑动窗口。注意**原生局部窗口 Attention**与给全局 Attention 模型临时删除最早 KV 并非完全等价；前文一旦移除，该信息可能无法再直接访问，位置也需要正确维护。

多请求争用 KV 资源时，服务框架可以调度/抢占并重计算，被抢占请求产生额外计算/延迟。若 KV Buffer 在启动时预分配，Decode 有效长度增长也未必对应每步 GPU 已分配显存同步增长。

---

## 9. 最终串联：必须牢记的三个不同问题

**问题 A：训练与推理为何 Shape 不同？** 训练及 Prefill 对一个序列并行构造 Q；Decode 每轮通常只对一个新增 Token 计算 Q，历史 K/V 来自 Cache。所以 Score 从 `[B,Hq,T,T]` 变为 `[B,Hq,1,Tcache]`。

**问题 B：Q 多，KV 少，怎么计算？** 固定连续分组 `group_size=Hq/Hkv`，第 h 个 Q 使用编号 `h//group_size` 的 K/V；可以逻辑 repeat 对齐；`matmul` 在前导维按相同索引配对，**不会随意跨 Head 计算**。

**问题 C：为什么少 K/V 又不损失 Q Head 数？** Q 各自生成独立查询及注意力权重，多只 Q 可以复用同一套 K/V，减少 K/V 投影输出、缓存容量和相应读取压力。

### 自测题

1. `[1,16,4,2]` 的 Q、`[1,4,4,2]` 的 K，`group_size` 是多少？展开 K 后的 Shape 呢？
2. `Q7`（从 0 编号）对应哪个 KV Head？
3. `Q @ K_expanded.transpose(-2,-1)` 的 Shape 是什么？会不会额外生成 Q7 对所有 KV Head 的交叉结果？
4. Prefill 长度为 3，预测 AI 但还没有 Forward AI，此时缓存有效长度是多少？
5. Decode 输入 AI 后缓存长度变成 4，Q 和 Score 分别是什么 Shape？
6. 为什么 Chunked Prefill 本身不能解决完整 KV Cache 总容量不足？

**答案：** ① `4`、`[1,16,4,2]`；② `7//4=1`；③ `[1,16,4,4]`，不会；④ `3`；⑤ `[1,2,1,2]`、`[1,2,1,4]`（使用微型模型）；⑥ 它只分块执行计算，仍需保存全部被保留 Token 的最终 K/V。

---

## 10. 推理实现：inference.py 与采样（对照 `gpt-learning/12.采样方式.md`）

代码：`qwen3-learning/qwen3-pytorch/inference.py`
测试：`qwen3-learning/qwen3-pytorch/tests/test_inference.py`（12 组断言）

### 10.1 与 GPT-2 线的函数对应

**组织方式完全一致，代码独立重写**（AGENTS.md 硬约束：Qwen3 线不跨目录 import）。

| 用途 | GPT-2 线（`gpt-learning/gpt-learning-pytorch/generate.py`） | Qwen3 线（`qwen3-pytorch/inference.py`） |
|---|---|---|
| 采样 | `_sample_from_logits` | `sample_from_logits` |
| 无缓存生成 | `generate_naive` | `generate_naive` |
| 有缓存生成 | `generate_kv`（包 `GPT2WithKVCache`） | `generate_with_cache`（直接传 `kv_cache`） |
| 加速比 | `benchmark_speed` | `benchmark_speed` |
| KV 占用统计 | —— | `kv_cache_bytes` / `kv_cache_report`（新增） |

`benchmark_speed` 返回的**列名与 GPT-2 线逐字相同**：`naive_avg_ms` / `kv_avg_ms` /
`speedup` / `naive_tokens` / `kv_tokens` ⇒ 两条线的数字可以直接并排放进一张表。

### 10.2 采样链：执行顺序与 GPT-2 线完全一致

```text
logits
  ↓ temperature <= 0 ? ── 是 ──▶ greedy（argmax，直接返回，不再走下面三步）
  ↓ 否
  ↓ logits / temperature          改尖锐程度
  ↓ top-k                         固定候选数量（第 k 名之后置 -inf）
  ↓ top-p                         动态候选集合（累计概率越界之后置 -inf）
  ↓ softmax
  ↓ torch.multinomial             随机抽取
```

`top-p` 的 mask 写法与 GPT-2 线**逐字相同**：

```python
cum = sorted_probs.cumsum(dim=-1)
remove = (cum - sorted_probs) > top_p
```

`cum - sorted_probs` 是「加入当前 token **之前**的累计概率」，
这样保证**让累计概率第一次超过 `top_p` 的那个 token 仍然被保留**（GPT-2 笔记 §12）。

`temperature <= 0` 时**直接 greedy 并跳过 top-k / top-p** —— 这条也被写成断言
（`tests/test_inference.py` 第 1 条：greedy 不受 `top_k` 影响）。

### 10.3 ★ GPT-2 笔记指出的 4 个问题，本线的处理

`gpt-learning/12.采样方式.md` 第十五节列了 4 个「这份代码中几个值得注意的问题」。
本线**逐条对照处理**：

| # | GPT-2 线的问题 | Qwen3 线的处理 |
|---|---|---|
| ① | 左 padding 没有 attention mask，短 prompt 会注意到 PAD，position_ids 也错位 | **本线只做 batch=1 的单条生成**，不引入 padding ⇒ 问题不存在（要批处理必须另加 mask） |
| ② | finished 样本没有真的冻结输出，EOS 后面仍会出随机 token | 本线在 eos 出现时**直接 break**；且**补上了 prefill 后第一个 token 的 eos 检查**（GPT-2 线的循环从第 2 个 token 开始查，第一个 token 若是 eos 不会被拦下） |
| ③ | `KVCache` 的 device / dtype 可能不匹配（默认 CPU + float32） | `make_kv_cache` **显式传模型参数的 dtype**，`append` 里也做 `.to(dtype)` 转换 |
| ④ | `cache.get()` 每次 clone 整段缓存 | 本线 `KVCache.get` **直接返回切片 view**，不复制（见 `model/common/kv_cache.py`） |

> 这条对照是 feat-018 的验证项之一：**不是「抄一遍 GPT-2 的实现」，
> 而是把 GPT-2 笔记里已经指出的坑，一条条在 Qwen3 这边堵上。**

### 10.4 ★ Qwen3 特有的两点

**① decode 阶段不施加因果 mask。**
`T_q = 1`、`T_kv = 已缓存的历史长度`，cache 天然只含历史与当前 token、不含未来，
所以**不需要 mask**。代码里体现为 `attention.py` 第 5 步的 `if T_q > 1:` 判断。

**可运行证据**（`tests/test_inference.py` 第 8 条，做的是「反向实验」）：
prefill 之后把 cache 里**除最后一个之外**的历史 K/V 全部清零，再走一步 decode：

- 若 decode 真的读了历史 → 输出**必须改变**
- 若 decode 错误地把历史屏蔽了 → 输出**不会变**

实测输出**变了** ⇒ decode 确实读了历史、不该有因果 mask。

**② KV Cache 按 `n_head_kv` 分配。**
占用公式：`n_layer × B × n_head_kv × T × head_dim × 2(K和V) × dtype_size`。
本仓库 `n_head=4 / n_head_kv=2` ⇒ **GQA 的占用是同尺寸 MHA 的 1/2**。

### 10.5 ★ 实测一：有/无 KV Cache 的输出**逐位相同**

```
$ python tests/test_inference.py
test_inference OK  greedy一致=逐位相同 logits_max|diff|=5.96e-07 speedup=1.33x
                    KV占用(GQA/MHA)=196608/393216B 比值=1/2
```

两条断言：

- **token 序列逐位相同**（greedy，12 步）：`torch.equal(seq_naive, seq_kv) == True`
- **逐步 logits 对得上**（更强，不依赖 argmax 不平局）：12 步的 `max|diff| = 5.96e-07`

> 为什么不是逐位相同而是「差 6e-7」？
> naive 路径每次把整条序列（T_q = P + s）一起前向，cache 路径每步只前向 1 个 token（T_q = 1）。
> **matmul 的规约顺序不同 ⇒ 浮点结果有末位差异**。
> 差异在 1e-7 量级（fp32 的 eps 量级），greedy 的 argmax 完全不受影响。

### 10.6 ★ 实测二：与 GPT-2 线的 speedup 横向对比

同一 prompt 长度（64）、同一生成长度（64）、同一台 CPU、同一份代码逻辑，
两边都是 greedy、各跑 3 次取平均：

| | naive_avg_ms | kv_avg_ms | speedup |
|---|---|---|---|
| **GPT-2**（MHA，n_head=4 / n_head_kv=4） | 207.92 | 118.94 | **1.748** |
| **Qwen3**（GQA，n_head=4 / n_head_kv=2） | 327.31 | 163.74 | **1.999** |

> ⚠️ **这一组是「实测值」，不是 config 里的超参**，因此**不受
> `tests/test_config_sync.py` 保护**，而且会随机器负载浮动。
> 同机复跑一次（`python evaluate.py --stage speedup`）得到的是
> GPT-2 **1.6992x**（230.54 / 135.68 ms）、Qwen3 **1.841x**（300.27 / 163.10 ms）。
> **结论不变**（Qwen3 绝对更慢、GQA 让加速比更高），但**小数点后的数字不可当常数引用**。
> `evaluate.py` 对 speedup 的门槛是**方向性**的（必须 > 1），不是固定阈值。

两条结论：

1. **Qwen3 的绝对速度更慢**（naive 327 ms vs 208 ms，慢了 ~57%）。
   原因是 Qwen3 单层做的事更多：三个独立投影（Q/K/V 不能合并）、
   **QK-Norm 两趟 RMSNorm**、**RoPE 旋转**、**SwiGLU 三矩阵**。
2. **但 Qwen3 的加速比反而更高**（2.00x vs 1.75x）。
   因为 GQA 让 cache 的**读取量减半** ⇒ cache 路径省下来的更多。

> **「模型更复杂」与「推理更快」是两回事。**
> Qwen3 用结构复杂度换来了 GQA 的内存/带宽优势 —— 这正是 Qwen3 真实部署里
> 「同尺寸下比 GPT-2 更快」的来源，而不是它算得更省。

同一个 prompt 长度下的 KV Cache 占用（T = 128，fp32）：

| | 占用 | 公式代入 |
|---|---|---|
| **Qwen3（GQA）** | **393,216 B**（384 KiB） | 4 × 1 × 2 × 128 × 48 × 2 × 4 |
| 同尺寸 MHA | 786,432 B（768 KiB） | 4 × 1 × 4 × 128 × 48 × 2 × 4 |

### 10.7 自测题（本节）

1. naive 路径和 cache 路径，为什么 logits 有 1e-7 量级的差异却能保证 token 逐位相同？
2. `top_p = 0.8`、概率为 `0.50 / 0.25 / 0.15 / 0.07 / 0.03` 时保留哪几个 token？
   `mask = cumsum - probs > top_p` 里为什么要减掉 `probs`？
3. 如果 decode 阶段错误地施加了因果 mask（把历史屏蔽掉），§10.5 的哪条断言会先炸？
4. GQA 让 Qwen3 的 speedup 比 MHA 高，但绝对速度更慢 —— 这两件事矛盾吗？

**答案：** ① matmul 规约顺序不同导致的 fp32 末位差异（1e-7 量级），远小于 logit 之间的间隔，
argmax 不受影响；② 保留 A、B、C（D、E 被删）—— 减掉 `probs` 是为了保证「让累计概率
第一次越过 0.8 的 C」仍被保留；③ 第 8 条（污染历史 K/V 后 decode 输出必须改变）；
④ 不矛盾 —— 复杂度影响**绝对耗时**，GQA 影响**cache 路径省下来的量**。

---

## 11. 评估：evaluate.py 与「两条线的评估不是同一套」

对应 feat-025。源码：[qwen3-pytorch/evaluate.py](qwen3-pytorch/evaluate.py)，
测试：[qwen3-pytorch/tests/test_evaluate.py](qwen3-pytorch/tests/test_evaluate.py)。

### 11.1 六段评估

| stage | 指标 | 对应 checkpoint |
|---|---|---|
| `pretrain` | val loss / val perplexity | `checkpoints/pretrain_dense.pt` |
| `moe` | 同上 **+ 专家负载分布 + aux_loss** | `checkpoints/pretrain_moe.pt` |
| `lora` | 偏好胜率（前 → 后对比） | `checkpoints/lora.pt` |
| `dpo` | reward margin 均值/标准差、chosen>rejected 准确率 | `checkpoints/dpo.pt` |
| `speedup` | 有/无 KV Cache 的输出一致性 + 加速比 | `checkpoints/pretrain_dense.pt` |
| `quality` | distinct-n / self-BLEU / token 熵 / response quality | `checkpoints/lora.pt` |

```bash
python evaluate.py --stage all        # 6 段一起跑
python evaluate.py --stage dpo        # 只跑 DPO 段
python evaluate.py --stage quality    # 只跑生成质量段（见 §13）
```

### 11.2 ★ 评估脚本的数字必须与训练日志对得上

| 指标 | 训练日志 | 评估脚本 |
|---|---|---|
| 预训练 val_loss / val_ppl | 1.3510 / 3.86 | 1.3510（batch 均值口径）/ 3.861 |
| LoRA 胜率（前 → 后） | 0.515 → 0.605 | 0.5150 → 0.6050 |
| DPO margin / dpo_acc / 绝对胜率 | +4.6512 / 0.780 / 0.665 | +4.6512 / 0.7800 / 0.6650 |

> **做法**：各段的评估函数**直接复用训练脚本里的实现**
> （`train.train_pretrain.estimate_loss` / `train.train_lora.evaluate_pairs` /
> `train.train_dpo.evaluate_dpo`），**不重写**。
> 重写就会出现「两边口径不同、数字对不上」的老问题 —— 本仓库已经吃过一次亏：
> 预训练 loss 一开始算成 batch 均值，而 GPT-2 线是 token 加权，两条线的 ppl 没法比。

### 11.3 ★ pretrain 的两种 loss 口径（都打印，是为了能跟训练日志对上）

| 口径 | 字段 | 跟谁对齐 |
|---|---|---|
| **token 加权** | `loss` / `perplexity` | GPT-2 线的 `eval_perplexity`（可直接横向比） |
| **batch 均值** | `loss_batch_mean` / `perplexity_batch_mean` | `train/train_pretrain.py` 打印的 `val_loss` |

实测 1.3485（token 加权）vs 1.3510（batch 均值），差 0.19% ——
差异只来自最后一个不满的 batch。

> **为什么「同名」还不够，「同口径」才够**：
> 同一个叫 `perplexity` 的字段，一个是 token 加权、一个是 batch 均值，
> 数字放在一起比就是在比两件不同的事。

### 11.4 ★ 与 GPT-2 线 9 项评估的差异：为什么不能逐项对标

GPT-2 线的 9 项是围绕 **Reward Model + PPO** 设计的；
Qwen3 线走 **LoRA + DPO**：没有显式奖励模型、没有 PPO、没有采样。

| GPT-2 线 | Qwen3 线 | 说明 |
|---|---|---|
| `pretrain` | ✅ `pretrain` | 同名列 + 同口径 |
| `speedup` | ✅ `speedup` | 同名列 |
| `sft` | 🔁 `lora` | 对应物：LoRA 前后胜率对比 |
| `reward`（RM pairwise acc） | 🔁 `dpo` | 本线没有 RM；对应物是 **margin 判对率** |
| `ppo`（reward / KL） | 🔁 `dpo` | 本线没有 PPO；对应物是**隐式奖励 margin** |
| `compare`（SFT vs PPO） | 🔁 `lora` | 本线没有 PPO；对应物是 LoRA 前→后对比 |
| `benchmark` | ❌ 无 | 语料是合成的，没有 QA benchmark |
| `diversity` / `response_quality` | ✅ `quality` | **同名同口径**（feat-026 补齐，见 §13）：共享一份实现 + 交叉验证 |
| — | ➕ `moe` | **Qwen3 独有**：专家负载分布 + aux_loss |

完整映射表在 [`qwen3-pytorch/README.md`](qwen3-pytorch/README.md)。

### 11.5 ★ 「不删 / 不改弱门槛」必须有可运行证据

`evaluate.py` 只设**可自证的门槛**，达不到就**报错退出**（不是打一个好看的数）：

| 段 | 门槛 |
|---|---|
| `pretrain` / `moe` | `perplexity == exp(loss)`；`n_tokens > 0`；`aux_loss` 有限；**专家负载之和 == B*T** |
| `lora` | 胜率 ∈ [0, 1] |
| `dpo` | **checkpoint 的 LoRA `B` 必须非零**（B 全零 ⇒ π_θ ≡ π_ref ⇒ DPO 压根没训过）；`margin_std >= 0`；准确率 ∈ [0, 1] |
| `speedup` | **有/无 KV Cache 的 greedy 输出必须逐位相同**，不一致直接报错退出 |

checkpoint 缺失 → **跳过该段并打印原因**（那是「还没跑到那一步」，不是「指标不合格」）。

> **门槛不是写在注释里的摆设**：`tests/test_evaluate.py` 第 5 组把
> `inference.generate_naive` 换成返回错误结果，断言 `eval_speedup` **必须抛 AssertionError**
> —— 这才是「门槛没被删」的证据。

### 11.6 MoE 段的两个新指标

- **`aux_loss`**：负载均衡辅助损失。本实现除以 `N·K`，
  所以**完全均衡时 = 1**（HF Mixtral 是 `K`，Switch 论文 top-1 版是 1）。
  实测 **0.9978** —— 已经很接近均衡值 1，说明 router 没塌。
- **专家负载分布**：实测 `[0.2439, 0.2395, 0.2437, 0.2729]`，
  **max/min = 1.140** —— 4 个专家的负载差 14%，没有出现「一个专家吃掉全部 token」的塌缩。

> 判读负载分布时要看 **max/min 比值**，不要只看单个值。
> 单个专家占 27% 看着不多，但如果另一个专家占 0%，max/min 就是 ∞ —— 那才是塌缩。

### 11.7 自测题（本节）

1. 为什么 pretrain 段要同时打印两种 loss 口径？
2. DPO 段的 `margin` 和 `win_rate` 为什么会打架（`too_short` 那一组）？
3. `dpo` 段为什么要检查 checkpoint 里 LoRA 的 `B` 是否全零？
4. `speedup` 段的「输出一致性」门槛，为什么不能改成「logits 逐位相同」？

**答案：** ① 两种口径分别对齐 GPT-2 线和本线训练日志，缺一个就没法横向比 / 纵向对；
② `margin` 是**相对 π_ref** 的间隔（长度偏置被抵消掉了），
`win_rate` 是**绝对** logprob 比较（长度偏置还在）⇒ `too_short` 的 rejected 极短、
总 logprob 天然接近 0，所以绝对胜率恒为 0，而 margin 能看出「模型确实更偏爱 chosen」；
③ B 全零 ⇒ π_θ ≡ π_ref ⇒ margin 恒为 0、loss 恒为 log 2 ——
那是「压根没训过」的 checkpoint，不该被评估脚本当成「训了但没效果」；
④ naive 每次前向整条序列（T_q = P+s），cache 路径每步只前向 1 个 token（T_q = 1），
**matmul 的规约顺序不同 ⇒ fp32 末位差异 ~1e-7**（实测 `max|diff| = 5.96e-07`）。
正确的门槛是「**token 序列逐位相同**」（§10.5 的实测），断言 logits 逐位相等会误报。

---

## 12. 退化对照：把「Qwen3 到底改了什么」变成一个数字（feat-017）

前面 7 篇笔记里全是这样的陈述：

> 「Qwen3 用了 RoPE」「Qwen3 用了 GQA」「Qwen3 用了 QK-Norm」「Qwen3 的 FFN 是 SwiGLU」
> 「Qwen3 用 RMSNorm 而不是 LayerNorm」

这些陈述**都是对的**，但它们都缺一个东西：**一个数字**。
「换成 RMSNorm 之后模型行为变了多少？」——没有数字，这个问题就没答案。

本节把它补上。做法是**反过来做**。

### 12.1 反过来做：把 Qwen3 退化成 GPT-2

正着做（从 GPT-2 出发，逐个加 Qwen3 的组件）有个麻烦：
每加一个组件，输出就变一次，你分不清是哪个组件造成的。

反过来做就干净得多：

1. 给 Qwen3 加 5 个**退化开关**，把它们**全部关掉** ⇒ Qwen3 与 GPT-2 结构完全相同；
2. 把 GPT-2 的权重按一张**显式映射表**搬进退化的 Qwen3；
3. 喂同一批输入，比两边的 logits。

| 开关 | 关掉后变成 | 对应 GPT-2 的什么 |
|---|---|---|
| `use_gqa` | `n_head_kv = n_head` | GPT-2 就是标准 MHA |
| `use_rope` | 不用 RoPE，改用加法式 `wpe` | GPT-2 的 sinusoidal 位置嵌入 |
| `use_qk_norm` | Q/K 不做 RMSNorm | GPT-2 没有 QK-Norm |
| `mlp_type` | `"swiglu"` → `"gelu"` | `c_fc → GELU → c_proj` |
| `norm_type` | `"rmsnorm"` → `"layernorm"` | GPT-2 用 `nn.LayerNorm` |

这 5 个开关**默认全部是「开」**（= 主线行为）。
也就是说：加它们**不改变任何一个既有训练 / 评估的数字**，
`config.py` 只是多了 5 个字段，主线代码路径一行都没变
（回归断言在 `tests/test_compare_with_gpt2.py` 第 4 组）。

### 12.2 三个「前提」：少一个就永远对不齐

这是整个实验最容易踩坑的地方，值得单独写。

**前提 ①：`head_dim` 必须回退成 GPT-2 的取法**

| | `n_head` | `head_dim` | `n_head · head_dim` |
|---|---|---|---|
| GPT-2 | 4 | `n_embd // n_head` = 32 | 128 = `n_embd` |
| Qwen3 主线 | 4 | **48**（故意解耦） | **192 ≠ 128** |

Qwen3 主线把 `head_dim` 与 `n_embd` **解耦**（为了体现真实 Qwen3-0.6B 的做法）。
但 GPT-2 的 `head_dim` 是 `n_embd // n_head` 算出来的。
`o_proj` 的输入维度是 `n_head · head_dim`，
不退回去的话连权重矩阵的形状都对不上 —— **这不是精度问题，是根本没法映射**。

**前提 ②：中间层维度必须回退成 GPT-2 的 `n_inner`**

| | 中间层维度 | 依据 |
|---|---|---|
| GPT-2 | 512 | `4 × n_embd` |
| Qwen3 主线 | 384 | `8/3 规则`后再对齐到 128 的倍数 |

`8/3 规则`是 SwiGLU 特有的（两个矩阵升维 ⇒ 宽度要砍）。
换成 GELU（一个矩阵升维）就必须用回 `4 × n_embd`。

**前提 ③：归一化必须两边同步切，而且 eps 也要一样**

RMSNorm（不减均值、无 bias）与 LayerNorm（减均值、有 bias）**在数学上不等价**。
只切 Qwen3 一侧 ⇒ 差的是「归一化方式」而不是「Qwen3 的其它组件」。

所以退化配置里 Qwen3 也用 `nn.LayerNorm`，**eps 取 GPT-2 的默认值 `1e-5`**。
（eps 放在 `sqrt(var + eps)` 内部，`1e-5` vs `1e-6` 会带来 ~1% 的相对差，
和「减不减均值」这个变量混在一起就分不清了。）

### 12.3 权重映射表（GPT-2 → 退化的 Qwen3）

| GPT-2 | → | 退化的 Qwen3 |
|---|---|---|
| `tok_emb.weight` | → | `wte.weight`（`lm_head` 与之绑定，两边都绑定，不用单独映射） |
| `pos_emb`（**buffer，不在 state_dict 里**） | → | `wpe.weight` |
| `blocks.{i}.ln_1.weight / .bias` | → | `blocks.{i}.ln_1.weight / .bias` |
| `blocks.{i}.attn.c_attn.weight[0:C]` | → | `blocks.{i}.attn.q_proj.weight` |
| `blocks.{i}.attn.c_attn.weight[C:2C]` | → | `blocks.{i}.attn.k_proj.weight` |
| `blocks.{i}.attn.c_attn.weight[2C:3C]` | → | `blocks.{i}.attn.v_proj.weight` |
| `blocks.{i}.attn.c_proj.weight` | → | `blocks.{i}.attn.o_proj.weight` |
| `blocks.{i}.ln_2.weight / .bias` | → | `blocks.{i}.ln_2.weight / .bias` |
| `blocks.{i}.mlp.c_fc.weight` | → | `blocks.{i}.mlp.up_proj.weight` |
| `blocks.{i}.mlp.c_proj.weight` | → | `blocks.{i}.mlp.down_proj.weight` |
| `ln_f.weight / .bias` | → | `ln_f.weight / .bias` |

两点值得记：

- **GPT-2 用一个 `c_attn` 一次投出 Q/K/V，Qwen3 必须拆成三个投影** ——
  因为 Qwen3 的 `head_dim` 与 `n_embd` 解耦，Q/K/V 的输出维度不再相等。
  退化时 `head_dim` 回退成 32，Q/K/V 的输出维度又相等了，但**代码仍然是三份**，
  所以映射是「把 `c_attn` 的 3 段行切开，分别塞进三个 Linear」。
- **GPT-2 的 `pos_emb` 是 `register_buffer(..., persistent=False)`** ——
  它**不在 `state_dict` 里**，必须从模型实例上直接取（`gpt2.pos_emb`）。
  > ⚠️ 开发计划 Task 18 的初稿写的是「GPT-2 用 **learned** PE（可训练参数）」——
  > **这是错的**。本仓 GPT-2 的位置表是**固定的 sinusoidal buffer**，不训练。
  > 对齐只关心数值不关心是否可训练，所以把同一张表拷进 `wpe` 即可。

### 12.4 实测结果：消融表

```bash
cd qwen3-learning/qwen3-pytorch
python ablations/compare_with_gpt2.py
```

```
[compare] Qwen3 ↔ GPT-2 退化对照
配置                          max_abs_diff       相对差    logit 量级   判定
----------------------------------------------------------------------------------
全关（退化成 GPT-2）                  0.000e+00  0.00e+00      0.7422   PASS（< 1e-04）
只开 use_gqa                     7.514e-02  1.01e-01      0.7422   OK（> 1e-04）
只开 use_rope                    1.541e+00  2.08e+00      0.7422   OK（> 1e-04）
只开 use_qk_norm                 9.334e-03  1.26e-02      0.7422   OK（> 1e-04）
只开 mlp=swiglu                  7.187e-02  9.68e-02      0.7422   OK（> 1e-04）
只开 norm=rmsnorm                5.449e-01  7.34e-01      0.7422   OK（> 1e-04）
```

（`vocab=293, seq_len=16, batch=2, seed=42`；`logit 量级` = GPT-2 那边 logits 的最大绝对值，
用来把绝对差换算成「相对信号有多大」。）

**这张表要这么读：**

- **第 1 行 `0.000e+00` 是全表的地基。** 退化后的 Qwen3 与 GPT-2
  **逐位相同**（不是「小于 1e-4」，是**一个 bit 都不差**）。
  这说明：① 退化开关真的把结构改成了 GPT-2；② 权重映射表没有漏项也没有多项；
  ③ 两边剩下的**数值差异来源为零**（连浮点累加顺序都一致）。
  > 有意思的是，GPT-2 的融合 `c_attn` 与 Qwen3 的三个独立 Linear，
  > 在 fp32 下算出了**逐位相同**的结果 —— 因为矩阵足够小，走的是同一套 GEMM 路径。
  > 所以「融合 vs 拆分」在这个尺寸上连浮点噪声都没有。

- **第 2–6 行每一行都只开一个组件**，diff 立刻从 `0` 跳到 `1e-3 ~ 1e0` 量级。
  这才是「每个组件到底改了什么」的量化答案：

| 组件 | 相对差 | 怎么读 |
|---|---|---|
| `use_rope` | **2.08** | **最大**。位置信息的表达方式换成 RoPE 后，logits 完全不是一回事 |
| `norm=rmsnorm` | **0.73** | 次大。减不减均值、有没有 bias，对数值的影响是 O(1) |
| `use_gqa` | **0.10** | K/V head 数减半 ⇒ 注意力看到的 K/V 变了 |
| `mlp=swiglu` | **0.097** | 多一个门控分支（gate 无 GPT-2 对应物，见 §12.5） |
| `use_qk_norm` | **0.013** | **最小**。Q/K 各过一次 RMSNorm，主要是「稳住数值尺度」，不改信息量 |

- **`norm=rmsnorm` 那一行是「前提」不是「组件」** ——
  它回答的是「为什么对照实验必须两边同步切换归一化」：
  不同步的话，误差就是 O(0.7) 量级，远超 `1e-4` 的阈值。

### 12.5 三条反证：证明它不是「碰巧通过」

一个只看「第 1 行 = 0」的实验是不可信的 —— 万一两边根本没跑呢？
所以 `tests/test_compare_with_gpt2.py` 里有三条**反证**：

| 反证 | 做法 | 实测 | 证明了什么 |
|---|---|---|---|
| A | **不加载权重**（退化 Qwen3 用随机权重） | `max_abs_diff = 1.013` | 差得远 ⇒ 第 1 行的 `0` 不是「拿自己跟自己比」 |
| B | **逐条核对映射表**（`wte` / `wpe` / `q,k,v,o` / `up,down` / `ln_f` 全部 `torch.equal`） | 全部相等 | 权重真的搬过去了，不是碰巧长得像 |
| C | **主线默认配置不变**（无 `wpe`、有 `q_norm`、FFN 是 `SwiGLUMLP`、`ln_f` 是 `RMSNorm`） | 全部符合 | 加开关**没有污染主线** |

还有两条「这一行的数字不能过度解读」的说明：

- **「只开 `use_rope`」这一行**：Qwen3 侧此时**完全没有加法式位置嵌入**
  （`use_rope=False` 才有 `wpe`），GPT-2 侧有 sinusoidal PE。
  所以这一行测的是「**加法式 sinusoidal PE → RoPE**」这个**整体替换**的影响，
  不是「在同样的位置信息上换一种编码方式」的差异。
- **「只开 `mlp=swiglu`」这一行**：SwiGLU 比 GELU 多一个 `gate_proj` 分支，
  GPT-2 里**没有对应物**，所以这一分支保持随机初值。
  也就是说这一行的 diff 里混着一份「随机的门控信号」，
  它测的是「有门控 vs 没门控」的行为差异，而不是「同一份门控的数值差异」。

### 12.6 这套方法能推广

同样的套路可以用在**任何两个模型之间**：

1. 给新模型加一组「退化开关」，默认值 = 新模型的行为；
2. 开关全关 ⇒ 结构退化成旧模型；
3. 写一张显式映射表 + 若干条**前提**（形状必须回退到旧模型的取法）；
4. 全关时 `max_abs_diff` 必须落到阈值内（否则说明映射表漏了东西）；
5. 逐个开开关，每个组件单独量化。

第 4 步特别有用：**它把「我以为我迁移对了」变成「差值是 0，所以确实迁移对了」**。
本节那个 `0.000e+00` 就是这么来的 —— 它不是「小于 1e-4」的口头保证，是可复现的实测。

### 12.7 自测题（本节）

1. 为什么 `use_rope=False` 时不能简单地「不给位置信息」？
2. 为什么退化配置里 `head_dim` 必须改回 32，改成 48 会怎样？
3. 「只开 `use_qk_norm`」那一行的相对差最小（1.3e-2），能说明 QK-Norm 没用吗？
4. 为什么第 1 行是 `0.000e+00` 而不是 `1e-7` 级别的浮点噪声？

**答案：**
① 不给位置信息的话，退化后的模型**连「位置」这个概念都没有了**，
与 GPT-2 的「有位置表」不是同一个模型，对齐必然失败；
必须把 GPT-2 的那张 sinusoidal 表搬进 `wpe`；
② `head_dim=48` 时 `o_proj` 的输入是 `4×48=192`，
而 GPT-2 的 `c_proj` 输入是 128 ⇒ **形状对不上，权重没法映射**，
连前向都跑不起来（不是精度问题）；
③ 不能。相对差小只说明 QK-Norm **不改变信息量**（它只缩放 Q/K 的尺度），
它改的是**数值稳定性**；要判断「有没有用」得看训练稳定性，
不能拿这个表当「QK-Norm 不重要」的证据；
④ 因为两边在 fp32 下走的是**同一套 GEMM / 逐元素算子的顺序**：
融合 `c_attn` 与三个独立 Linear 在这个尺寸下结果逐位相同，
LayerNorm / softmax / GELU 都是确定性算子 ⇒ 差异不是「小于浮点噪声」，
而是**真的一个 bit 都不差**。

---

## 13. 生成质量评估：把「两条线能并排」变成一条断言（feat-026）

对应 feat-026。源码：[qwen3-pytorch/quality.py](qwen3-pytorch/quality.py)、
[qwen3-pytorch/data/quality_prompts.py](qwen3-pytorch/data/quality_prompts.py)、
[qwen3-pytorch/ablations/compare_quality.py](qwen3-pytorch/ablations/compare_quality.py)，
测试：[qwen3-pytorch/tests/test_quality.py](qwen3-pytorch/tests/test_quality.py)。

### 13.1 缺口从哪来

§11.4 那张表里有一行刺眼：`diversity` / `response_quality` 在 feat-025 收班时还是
「⏳ 未实现」。当时的理由是「feat-025 的范围是精简评估」——
但那留下了一个**很不对称**的局面：

- GPT-2 线有 `diversity`（4 个生成质量指标）和 `response_quality`；
- Qwen3 线没有。

⇒ 两条线**别的段都不逐项对标**，唯独「模型说出来的话好不好」这件事，
一条线能测、另一条线不能测 ⇒ **没法并排**。

feat-026 就是补这半块拼图。但补的时候有个陷阱：
**「把 GPT-2 的 4 个函数抄一遍」不叫做补上了** ——
抄出来的两份代码迟早会分叉，而数字看起来一样却量着不同的事（§11.3 那个 loss 口径的坑）。

### 13.2 4 个指标的定义

| 指标 | 定义 | 越大越好？ |
|---|---|---|
| `distinct_1` / `distinct_2` | 字符级 n-gram 多样性 = 不同 n-gram 数 / n-gram 总数 | 越大越多样 |
| `self_bleu` | 批内每条回答对其它 N-1 条的 BLEU 平均（nltk + smoothing method1） | 越小越多样 |
| `token_entropy` | 生成每步 next-token 分布熵的均值（自然对数，单位 nat） | — |
| `response_quality` | 4 条规则各 0.25：长度 10~50 字符 / 含动物名 / 含常见动词 / 以 `.!?` 结尾 | 越大越好 |

两个容易写错的细节（GPT-2 线是这么写的，本线必须**一模一样**）：

1. **熵要「先样本内平均，再样本间平均」**。
   把所有样本的所有步摊平一起平均，结果在**样本长度不一致**时会不一样
   （提前 EOS 的短样本权重会变）。本线的 `token_entropy_from_logits` 按样本分组，
   `tests/test_quality.py` 第 3b 组用一个手算例子钉死了这一点
   （分组平均 2.7726 vs 摊平平均 2.3105）。
2. **响应质量的长度规则按**字符**算，不是按 token 算** —— 这条是后面 §13.5 的坑。

### 13.3 ★ 实现放在 `quality.py`：两条线共用同一份代码

`quality.py` 里只有 4 个**纯函数**（吃「一批回答」或「一批 logits」），
顶层只 import `torch` / `nltk` / 标准库 ——
Qwen3 特有的 import 全部**惰性**（函数体内才 import）。

这样 GPT-2 线的进程就能**按文件路径**加载它来算同一批指标
（`ablations/_gpt2_quality_runner.py` 里的 `_load_shared_quality()`），
而不必把 Qwen3 的项目根塞进 `PYTHONPATH` ——
两个线都有 `data` 包、都有 `model` 包，塞进去就抢包名了。

> **同包名的坑在本仓库出现过两次**：
> ① `data` / `model` 同名 ⇒ 跨线复用只能按**绝对路径**加载，或开**子进程**并把
> `PYTHONPATH` 指向要用的那一条线；
> ② GPT-2 线的 `evaluate.py` 在**模块层**就 `argparse.parse_args()` 并按 `args.stage` 分派
> ⇒ 直接 `import evaluate` 会顺手把某一整个 stage 跑一遍。
> runner 里用 `argparse.ArgumentParser.parse_args` 打补丁的办法把它变成「只定义函数、不执行」。

### 13.4 ★「同名同口径」不是承诺，是一条**断言**

`ablations/compare_quality.py` 的做法：

1. 两边喂**同一批 prompt**（Qwen3 线的 `data/quality_prompts.py` 生成的动物探针）；
2. 两边的指标**都走** `quality.py`（不是各写一份）；
3. 把 GPT-2 侧生成的回答**固定住**，再喂给 **GPT-2 线自己的 `evaluate.py`**
   的 `eval_distinct_n` / `eval_self_bleu` / `eval_response_quality` / `eval_token_entropy`；
4. 两边的值必须**完全相等**，不相等 ⇒ **报错退出**（退出码 1），报告里的并排数字一律不算数。

实测（两对模型各跑一次）：

```
表 2  口径交叉验证（GPT-2 自己的 evaluate.py  vs  共享实现 quality.py）
指标                GPT-2 自己算  共享实现算   差值
distinct_1        0.0187     0.0187  0.000e+00
distinct_2        0.1745     0.1745  0.000e+00
self_bleu         0.6959     0.6959  0.000e+00
response_quality  0.2750     0.2750  0.000e+00
token_entropy     2.7251     2.7251  0.000e+00
  ✅ 5 个指标全部逐值相等（容差 1e-09）⇒ 两条线量的是同一件事。
```

两处为了让这个断言成立而做的技术处理，值得单独记一笔：

- **熵要能逐值相等 ⇒ 路径必须确定**。GPT-2 的 `eval_token_entropy` 是**采样**的
  （`torch.multinomial`），两次跑路径不同 ⇒ 熵不同。
  runner 把 `torch.multinomial` 临时换成 `argmax`（= 贪心），
  再用**同一段贪心路径**的 logits 喂给共享实现 ⇒ 两边才可能算出同一个数。
- **样本集必须一样**。GPT-2 的 `eval_token_entropy(n_samples=20)` 取的是**前 20 条** prompt，
  所以 runner 里把 `get_ppo_prompts` 改成「前 n 条」的切片 ——
  不截断的话它会遍历全部 30 条，两边样本集不同 ⇒ 熵差了 2.8%（0.0000129698 vs 0.0000133310）。

### 13.5 ★ 三条不可比性：数字可信，但结论要打折

并排数字本身已被 §13.4 证明「量的是同一件事」。但**它们回答不了**
「哪个模型更好」，原因有三个，都必须写在报告里：

| 不可比性 | 具体表现 | 后果 | 对策 |
|---|---|---|---|
| **探针的域内外** | 动物探针对 GPT-2 是**域内**（它的预训练语料就是动物童话），对 Qwen3 是**域外**（Qwen3 训在 transformer 主题语料上） | 表 1 的高低**不能**读成「架构谁更好」 | 另跑一组「Qwen3 自己的语料」（域内）做对照（表 3） |
| **tokenizer 粒度** | `response_quality` 的长度规则是 10~50 **字符**（按字符级 tokenizer 标定），Qwen3 用 BPE ⇒ 回答普遍更长 | 这一条规则对 Qwen3 **系统性不利** | 额外报 `response_quality_ex_length`（扣掉长度规则）+ 每条规则的命中率 |
| **词表大小** | 熵的绝对值依赖 V（均匀分布时 H = log V）；GPT-2 ~64 字符 vs Qwen3 778 子词 | `token_entropy` 的绝对值跨线**不可比** | 跨线只看 `token_entropy_normalized` = H / log(V) |

### 13.6 ★ 实测数字（seed 1234，两次跑逐字相同）

脚本会比**两对**模型：微调对（`sft.pt` vs `lora.pt`）与底模对（`pretrain.pt` vs `pretrain_dense.pt`）。
加底模对是因为下一节那个发现。

**微调对**（`python evaluate.py --stage quality` 的 Qwen3 侧数字与下表一致）：

| 指标 | GPT-2 (`sft.pt`) | Qwen3 (`lora.pt`) |
|---|---|---|
| `distinct_1` | 0.0076 | 0.0258 |
| `distinct_2` | 0.0092 | 0.1277 |
| `self_bleu` | 1.0000 | 0.7907 |
| `token_entropy` | 0.0000 | 5.7975 |
| `token_entropy_normalized` | 0.0000 | 0.8709 |
| `response_quality` | 0.2500 | 0.0833 |
| `response_quality_ex_length` | 0.0000 | 0.1111 |
| `mean_response_chars` | 48.0 | 139.6 |

规则命中率：GPT-2 侧 `length=1.00 / entity=0 / verb=0 / ending=0`，
Qwen3 侧 `length=0 / entity=0.13 / verb=0 / ending=0.20`。
**GPT-2 那 0.25 的总分全部来自「长度规则」** —— 它的回答恒为 48 字符。

**底模对**：

| 指标 | GPT-2 (`pretrain.pt`) | Qwen3 (`pretrain_dense.pt`) |
|---|---|---|
| `distinct_1` | 0.0187 | 0.0099 |
| `distinct_2` | 0.1745 | 0.0296 |
| `self_bleu` | 0.6959 | 0.9825 |
| `token_entropy_normalized` | 0.7159 | 0.1954 |
| `response_quality` | 0.2750 | 0.0750 |
| `response_quality_ex_length` | 0.0333 | 0.1000 |

读法：**底模对里 GPT-2 全面更高**（探针对它是域内，而且它的模型没塌）；
**微调对里 Qwen3 全面更高**（因为 GPT-2 那一侧塌了，见下节）。
两个数字放在一起才说得清 —— 只报其中一个都会误导。

### 13.7 ★ 顺手挖出来的一个 P1 线问题：SFT / PPO checkpoint 退化成常量输出

测「退化」这件事本身需要判据。一开始用「不同回答的数量」，
**被骗了**：GPT-2 的 `sft.pt` 会把 prompt 的最后一个字符（`?` / `.`）一起吐出来，
于是 30 条不同问题得到 30 条**看起来不同**的回答。

改用**公共后缀长度**（一批回答的最长公共后缀）就露馅了：

```
· GPT-2 这一侧：退化成常量输出（公共后缀 47 字符 / 平均 48.0 字符 = 98%）
· Qwen3 这一侧：正常（公共后缀 0 字符 / 平均 139.6 字符 = 0%）
```

GPT-2 的 `sft.pt` 与 `ppo.pt` 对**任何** prompt 都输出同一串常量
（`'? <unk>sep<unk>'` + 一堆空格，48 字符），而 `pretrain.pt` 的输出是**有变化的**。
⇒ **P1 线的 SFT 之后模型塌成了常量输出**。

这与 handoff 里早就记着的「PPO 相对 SFT 无提升、`rm_score = 0.0696`」是**同一个现象**：
模型只能吐一个常量 ⇒ 奖励模型只能给一个常量分 ⇒ PPO 没有可优化的方向。

> **本 feature 不修它**（修它要改 P1 线的训练代码并跑 L3 全量，
> 那是另一条 feature 的事）。这里只负责**如实报告**，
> 并把它写进 `session-handoff.md` 的「Broken Or Unverified」。

### 13.8 ★ 域内 vs 域外：结论必须由数据算出

第三张表拿 Qwen3 自己的语料当探针（域内）跟动物探针（域外）对比。
**踩过的坑**：脚本第一版把结论**写死**成「域内探针上多样性明显更高」，
而实测数字恰好相反：

```
distinct_1  域外 0.0258 > 域内 0.0232      self_bleu 域外 0.7907 < 域内 0.8241
```

⇒ 结论改成**由数据算出**（脚本里比大小再打印），
底模对那一组域内域外几乎持平（0.0099 vs 0.0099）。
**教训**：对照实验的结论文字也属于「实验输出」，必须由数据生成，
否则「脚本跑出来的结论」和「脚本跑出来的数字」会互相打脸。

### 13.9 自测题（本节）

1. 「两条线各写一份指标实现、长得一样」为什么不算「同名同口径」？
2. 交叉验证里为什么必须把 `torch.multinomial` 换成 `argmax`？
3. 为什么熵的绝对值跨线不可比，归一化之后就可比了？
4. `response_quality` 的长度规则为什么对 Qwen3 系统性不利？
5. 为什么「域内探针上多样性更高」这种结论不能写死在脚本里？

**答案：**
① 两份实现迟早分叉；**口径一致要靠「同一份实现」**，
且必须有一条**机器证明**（本节 §13.4 的交叉验证断言）；
② 采样路径不确定 ⇒ 两次跑的熵不同 ⇒ 两边永远对不上；
③ 均匀分布时 H = log V，V 不同 ⇒ 绝对值不可比；除掉 log V 之后
剩下的就是「分布有多尖」这个与词表大小无关的量；
④ 规则是 10~50 **字符**，按字符级 tokenizer 标定；Qwen3 用 BPE，
同样的信息量要更多字符 ⇒ 回答普遍超长 ⇒ 这条规则专挑 Qwen3 不利；
⑤ 写死的结论一旦与实测数字矛盾，整份报告就不可信了
（本节 §13.8 就是这个坑的真实例子）——
**结论必须由数据生成**。

# Qwen3 学习笔记：从训练到 KV Cache 推理，再到 GQA 的 Head 映射

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

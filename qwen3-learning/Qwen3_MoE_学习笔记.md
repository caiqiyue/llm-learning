# Qwen3-MoE 学习笔记

## 1. MoE 是什么

MoE 全称：

```text
Mixture of Experts
专家混合模型
```

对于已经学过 Qwen3-Dense 的情况，可以先这样理解：

> **Qwen3-MoE 的 Attention 部分基本沿用 Qwen3-Dense，主要把原来的单个 SwiGLU MLP 换成了 Router + 多个 Expert SwiGLU。**

也就是说，Dense Decoder Layer：

```text
hidden_states
      ↓
RMSNorm
      ↓
Attention
      ↓
Residual
      ↓
RMSNorm
      ↓
SwiGLU MLP
      ↓
Residual
```

MoE Decoder Layer：

```text
hidden_states
      ↓
RMSNorm
      ↓
Attention
      ↓
Residual
      ↓
RMSNorm
      ↓
Router
      ↓
Top-K Experts
      ↓
Expert SwiGLU
      ↓
Weighted Sum
      ↓
Residual
```

## 2. MoE 的本质目标是什么

MoE 的第一目标不是单纯“让推理更快”。

更准确地说：

> **MoE 的本质，是让模型拥有更大的总参数容量，但每个 Token 只激活其中一小部分参数。**

可以写成：

\[
\boxed{
更大的模型容量
+
相对可控的单 Token 计算量
}
\]

例如：

```text
Dense：
所有 Token 都经过同一个大 MLP

MoE：
模型拥有很多 Expert
每个 Token 只选择少数几个 Expert
```

所以 MoE 的核心价值是：

1. 提高模型总参数容量和表达能力。
2. 避免每个 Token 都使用全部参数。
3. 在模型能力和计算成本之间做更好的 trade-off。

## 3. MoE 不等于“推理一定更快”

MoE 会引入额外开销：

```text
Router 计算
Token Dispatch
Expert Gather / Scatter
不同 Expert 的负载不均
Expert 权重访问
Kernel 调度
多卡通信
```

所以更准确地说：

> **MoE 的目标是“更大参数容量 + 稀疏激活”，而不是单纯追求最低推理延迟。**

## 4. Qwen3-Dense 和 Qwen3-MoE 的结构区别

Dense 模型中，一个 Decoder Layer 的 MLP：

```text
RMSNorm
   ↓
SwiGLU
   ↓
Residual
```

每个 Token 都经过同一套：

```text
gate_proj
up_proj
down_proj
```

MoE 中：

```text
RMSNorm
   ↓
Router
   ↓
选择 Top-K Experts
   ↓
多个 Expert SwiGLU
   ↓
加权求和
   ↓
Residual
```

最大的结构变化发生在 FFN / MLP 部分。

## 5. Expert 到底是什么

一个 Expert 本质上就是一个独立的 SwiGLU MLP。

第 \(i\) 个 Expert：

\[
Expert_i(x)
=
W^{(i)}_{down}
\left[
SiLU(W^{(i)}_{gate}x)
\odot
(W^{(i)}_{up}x)
\right]
\]

每个 Expert 都有自己独立的：

```text
gate_proj
up_proj
down_proj
```

## 6. Router 是什么

Router 的作用是：

> **对每个 Token 判断应该送到哪些 Expert。**

假设：

```text
hidden_size = H
num_experts = E
```

一个 Token 的 hidden state：

\[
x\in\mathbb{R}^{H}
\]

Router 做一次线性映射：

\[
router\_logits=xW_r
\]

其中：

\[
W_r.shape=[H,E]
\]

得到：

\[
router\_logits.shape=[E]
\]

这 E 个数分别对应 E 个 Expert 的分数。

## 7. Router 的 Tensor Shape

假设：

```text
Batch = B
Sequence = T
Hidden = H
Experts = E
```

进入 Router：

\[
X.shape=[B,T,H]
\]

Router：

\[
[B,T,H]
\rightarrow
[B,T,E]
\]

## 8. Top-K Routing

Router 不会把一个 Token 送给所有 Expert，而是选择分数最高的 Top-K。

例如：

```text
Experts = 4
Top-K = 2
```

某个 Token 的 Router logits：

\[
[1.2,\ 3.8,\ 0.5,\ 2.4]
\]

Top-2：

```text
Expert 1
Expert 3
```

于是：

```text
Token
 ├─→ Expert 1
 └─→ Expert 3
```

## 9. 每个 Token 可以选择不同 Expert

MoE 的 Routing 是 Token-Level 的。

例如：

```text
Token 0 → Expert 3, 7
Token 1 → Expert 1, 8
Token 2 → Expert 3, 12
Token 3 → Expert 6, 9
```

> **不是整句话统一选择同一组 Expert，而是每个 Token 独立 Routing。**

## 10. Top-K 之后还需要权重

MoE 通常会根据 Router 概率对多个 Expert 输出做加权求和。

假设：

\[
p_1=0.7,\quad p_3=0.3
\]

两个 Expert 输出：

\[
E_1(x)=[1,2,1,0]
\]

\[
E_3(x)=[0,1,3,2]
\]

最终：

\[
y
=
0.7E_1(x)+0.3E_3(x)
\]

得到：

\[
\boxed{
y=[0.7,1.7,1.6,0.6]
}
\]

所以：

\[
\boxed{
MoE(x)
=
\sum_{i\in TopK(x)}
p_i(x)\cdot Expert_i(x)
}
\]

## 11. Router 的概率怎么得到

Router 先计算：

\[
z_t=x_tW_r
\]

然后对 Expert 维度做 Softmax：

\[
P_{t,i}
=
\frac{
e^{z_{t,i}}
}{
\sum_j e^{z_{t,j}}
}
\]

这些概率用于：

1. Top-K 选择；
2. 生成 Expert 加权系数；
3. 计算 Auxiliary Loss。

## 12. MoE 的完整 Forward

输入：

\[
X.shape=[B,T,H]
\]

步骤：

```text
[B,T,H]
   ↓ Router
[B,T,E]
   ↓ Softmax
router_probs
   ↓ Top-K
topk_indices  [B,T,K]
topk_weights  [B,T,K]
   ↓
Token Dispatch
   ↓
Experts
   ↓
Weighted Combine
   ↓
[B,T,H]
```

## 13. 为什么需要 Token Dispatch

实际实现不会一个 Token 一个 Token 调 Expert。

而是把被分到同一个 Expert 的 Token 收集起来：

```text
Expert 3:
Token 0
Token 2
Token 7
Token 12
...
```

然后作为一个 batch 一次性执行：

```text
gate_proj
up_proj
down_proj
```

因此 MoE 的工程流程是：

```text
Dispatch
→ Expert Compute
→ Combine
```

## 14. 为什么会出现负载不均衡

如果 Router 把大量 Token 都送去少数几个 Expert，就会造成：

```text
Expert 0：非常忙
其他 Expert：很闲
```

问题包括：

1. GPU 负载不均衡；
2. 某些 Expert 成为瓶颈；
3. 很多 Expert 学不到有效能力；
4. 分布式训练通信效率下降。

所以需要 Load Balancing。

## 15. Auxiliary Loss 是什么

Auxiliary Loss 的作用是：

> **约束 Router，不要把大量 Token 都送到少数 Expert。**

总 Loss：

\[
\boxed{
L_{total}
=
L_{LM}
+
\lambda L_{aux}
}
\]

其中：

- \(L_{LM}\)：语言模型主任务 Loss；
- \(L_{aux}\)：Router 负载均衡辅助损失；
- \(\lambda\)：辅助 Loss 权重。

## 16. Auxiliary Loss 中的 \(f_i\)

\(f_i\) 表示：

> **Expert i 实际收到了多少比例的 Token。**

例如 8 个 Token 的 Top-1 分配：

```text
4 个 → Expert 0
2 个 → Expert 1
1 个 → Expert 2
1 个 → Expert 3
```

则：

\[
f=[0.5,0.25,0.125,0.125]
\]

## 17. Auxiliary Loss 中的 \(p_i\)

\(p_i\) 表示：

> **Router 对 Expert i 的平均概率偏好。**

假设每个 Token 的 Router Softmax 概率：

```text
Token 1: [0.64, 0.24, 0.09, 0.03]
Token 2: [0.50, 0.32, 0.12, 0.06]
Token 3: [0.11, 0.60, 0.15, 0.14]
Token 4: [0.10, 0.14, 0.60, 0.16]
```

则：

\[
\boxed{
p_i
=
\frac1T
\sum_{t=1}^{T}
P_{t,i}
}
\]

也就是对所有 Token 对同一个 Expert 的概率取平均。

## 18. \(f_i\) 和 \(p_i\) 的区别

| 符号 | 含义 |
|---|---|
| \(f_i\) | Expert i 实际收到了多少 Token |
| \(p_i\) | Router 平均有多偏爱 Expert i |

可以理解成：

```text
f_i：实际流量
p_i：Router 偏好
```

## 19. 一种经典 Auxiliary Loss

一种经典的 Switch Transformer 风格负载均衡损失：

\[
\boxed{
L_{aux}
=
N
\sum_{i=1}^{N}
f_i p_i
}
\]

其中：

- \(N\)：Expert 数量；
- \(f_i\)：实际 Token 分配比例；
- \(p_i\)：Router 平均概率。

例如：

\[
f=[0.5,0.25,0.125,0.125]
\]

\[
p=[0.45,0.30,0.15,0.10]
\]

则：

\[
L_{aux}
=
4(
0.5\times0.45
+
0.25\times0.30
+
0.125\times0.15
+
0.125\times0.10
)
\]

得到：

\[
\boxed{
L_{aux}=1.325
}
\]

如果完全均匀：

\[
f=p=[0.25,0.25,0.25,0.25]
\]

则：

\[
\boxed{
L_{aux}=1
}
\]

### 19.1 本仓库代码用的是哪个版本（量纲差异，别和 §19 混用）

§19 写的是 **Switch Transformer 论文版**：`L_aux = N · Σ f_i p_i`，其中 `f_i` 按 **token** 计数
（top-1 路由时 `Σf_i = 1`；top-k 路由时一个 token 被计 k 次，`Σf_i = K`），
且带一个 `N`（专家数）的系数。

本仓库 `model/qwen3_moe.py::load_balancing_aux_loss` 的写法是：

```text
f_i  = 专家 i 的 (token, k槽) 对数 / (token 总数 × k)     ← 对 tokens×k 个槽位求均值，恒有 Σf_i = 1
p_i  = router 对专家 i 给出的平均概率                       ← 与 §17 一致，恒有 Σp_i = 1
aux  = E · Σ f_i · p_i                                    ← 系数是专家数 E
```

**均匀路由时（关键锚点）**：`f_i = 1/E`、`p_i = 1/E`

\[
L_{aux} = E \cdot \sum_i \frac1E \cdot \frac1E = E \cdot E \cdot \frac{1}{E^2} = \boxed{1}
\]

**三种写法的量纲对比（别混用）**

| 版本 | `f` 的归一化 | 均匀时 f | 均匀时 p | 均匀时 \(L_{aux}\) |
|---|---|---|---|---|
| §19 Switch 论文版（top-1，按 token 计一次） | Σf = 1 | 1/E | 1/E | **1** |
| §19 Switch 版直接用在 top-k 上（同一 token 被计 k 次） | Σf = K | K/E | 1/E | **K** |
| HF Mixtral（transformers `load_balancing_loss_func`） | Σf = K | K/E | 1/E | **K** |
| **本仓库代码** | Σf = 1 | 1/E | 1/E | **1** |

> HF 的 `load_balancing_loss_func` 把槽位计数除以 **N**（不是 N·K），
> 所以它的 `f` 满足 `Σf = K`，均匀时 aux = K（本仓库 K=2 → 2）。
> 本实现多除一个 K（把 `f` 归一化到 Σf = 1），因此 **aux = HF 的 1/K**，
> 好处是「完全均衡时 aux_loss = 1」这个直觉数值成立（与 Switch 论文 top-1 版同量纲），
> 代价是**不能拿我们的数值和 HF 的数值直接比大小**（要乘 K）。
>
> **实测锚点**（`python tests/test_moe.py`）：
> `[aux 量纲] 均衡时 aux=1.0000（=1）；错配时 aux=1.6000；HF 同款公式 = 2.0000（= K×2）`
> —— 这条断言写死在 `tests/test_moe.py` 第 11 节，代码一改量纲测试就会红。
>
> **看到别人论文里写 "aux loss = 1" 时，先确认他用的是哪一版。**

## 20. 为什么 \(f_i\) 和 \(p_i\) 都需要

只看 \(f_i\)，只能看到最终流量。

只看 \(p_i\)，只能看到 Router 的平均倾向。

二者一起才能更完整地刻画：

```text
实际流量
×
Router 偏好
```

## 21. MoE 训练阶段流程

```text
Input Tokens
      ↓
Embedding
      ↓
Decoder Attention
      ↓
RMSNorm
      ↓
Router
      ↓
Softmax
      ↓
Top-K
      ↓
Token Dispatch
      ↓
Experts
      ↓
Weighted Combine
      ↓
Residual
      ↓
LM Head
      ↓
LM Loss
```

同时计算：

\[
L_{aux}
\]

最后：

\[
L_{total}
=
L_{LM}
+
\lambda L_{aux}
\]

再执行 Backward，更新：

```text
Attention 参数
Expert 参数
Router 参数
```

## 22. MoE 推理阶段流程

推理时没有 Auxiliary Loss，也没有 Backward。

Prefill：

```text
[B,T,H]
   ↓
Router
   ↓
[B,T,E]
   ↓
Top-K
   ↓
每个 Token 进入自己的 Experts
```

Decode：

```text
新 Token
[1,1,H]
   ↓
Router
   ↓
[1,1,E]
   ↓
Top-K
   ↓
只计算当前 Token 选中的 Experts
```

## 23. MoE 和 KV Cache 的关系

MoE 本身不会产生类似 Attention KV Cache 的历史缓存。

| 模块 | 是否有历史 KV Cache |
|---|---|
| Attention | 有 |
| MoE Router | 无 |
| Expert MLP | 无 |

KV Cache 属于 Attention；Expert 输出不会跨 Token 长期缓存。

## 24. MoE 和量化的关系

MoE 总参数量通常很大，因为每个 Expert 都有：

```text
gate_proj
up_proj
down_proj
```

即使一个 Token 只激活少数 Expert：

> **所有 Expert 权重通常仍需要存储在 CPU/GPU/设备内存中，除非使用 offload 等策略。**

所以 MoE 对：

```text
模型权重存储
内存带宽
Expert 权重搬运
```

特别敏感。

因此 FP8、INT8、INT4、NVFP4 等低精度权重量化，对 MoE 有很高的工程价值。

## 25. Dense 与 MoE 对比

| 项目 | Dense | MoE |
|---|---|---|
| Attention | 正常 Attention/GQA | 正常 Attention/GQA |
| MLP | 单个 SwiGLU | 多个 Expert SwiGLU |
| Router | 无 | 有 |
| 每 Token 激活全部 FFN 参数 | 是 | 否 |
| 每 Token 激活部分 Expert | 否 | 是 |
| 总参数容量 | 相对有限 | 可以非常大 |
| Token Dispatch | 无 | 有 |
| Auxiliary Loss | 无 | 通常有 |
| KV Cache | 有 | 有 |
| MoE 自身产生 KV Cache | 否 | 否 |

## 25.1 本仓库的实测对照（代码跑出来的数字）

`tests/test_moe.py` 在 `n_embd=128 / n_layer=4 / vocab=512` 下的实测输出：

```text
[专家负载] ['12.5', '16.0', '16.0', '19.5']  理想=16.0  最大偏差=21.9%
[稀疏性] 专家处理总行数=128 (=N*K=64*2)，全算需 256，省下 50%
[参数量] dense=951,808  moe=1,150,464  比值=1.21x
```

三点解读：

1. **稀疏性是真省下来的**，不是「算了再丢」：用 forward hook 统计每个专家实际处理的行数，
   总计 128 = N·K（64 个 token × top-2）；若实现是全算 4 个专家再取 top-2，这里会是 256。
2. **参数量只多 1.21 倍**，因为本配置的 MoE 专家更窄（`moe_intermediate_size=128`，
   而 Dense 的 `intermediate_size=384`）。FFN 部分：Dense = 3×128×384 = 147,456/层，
   MoE = 4 专家 × 3×128×128 + router 512 = 197,120/层 ≈ 1.34 倍；
   摊到整个模型（含 attention、embedding）就是 1.21 倍。
   **MoE 的「参数多 N 倍、计算只多 k 倍」要在专家宽度与 dense 相同的前提下才成立。**
3. **专家负载天然是不均匀的**（最大偏差 21.9%）：这正是需要 aux_loss 的原因 ——
   随机初始化的 router 不会自动均匀分配。

## 26. 最终完整心智模型

```text
                    hidden_states
                          │
                          ↓
                       RMSNorm
                          │
                          ↓
                     Attention
                          │
                          ↓
                      Residual
                          │
                          ↓
                       RMSNorm
                          │
                          ↓
                        Router
                          │
                    [B,T,H] → [B,T,E]
                          │
                          ↓
                        Softmax
                          │
                          ↓
                         Top-K
                          │
                    ┌─────┼─────┐
                    ↓     ↓     ↓
                 Expert  Expert Expert
                    ↓     ↓     ↓
                    └── Weighted Sum ──┐
                                      ↓
                                   Residual
                                      ↓
                                 Layer Output
```

## 27. 最重要的几句话

1. **MoE 的核心不是单纯加速，而是提高总参数容量，同时只激活部分参数。**
2. **Expert 本质就是独立的 SwiGLU MLP。**
3. **Router 对每个 Token 单独计算 Expert 分数。**
4. **Top-K 决定每个 Token 实际进入哪些 Expert。**
5. **多个 Expert 输出按 Router 权重加权求和。**
6. **Token Dispatch 是 MoE 高效实现的重要工程步骤。**
7. **Auxiliary Loss 用来防止 Router 把 Token 过度集中到少数 Expert。**
8. **\(f_i\) 表示实际 Expert 流量，\(p_i\) 表示 Router 平均偏好。**
9. **MoE 和 KV Cache 是两件不同的事情。**
10. **MoE 对权重内存和内存带宽非常敏感，因此很适合结合低精度量化。**

# Qwen3 学习笔记：QK-Norm

## 1. QK-Norm 是什么

QK-Norm 可以理解成：

> **Qwen3 在 Q、K 通过线性投影生成以后，再分别对每个 Attention Head 的 Q 和 K 做一次 RMSNorm，用来稳定它们进入点积之前的数值尺度。**

整体位置：

```text
hidden_states
      ↓
RMSNorm
      ↓
Q / K / V Projection
      ↓
拆成多个 Attention Head
      ↓
Q-Norm / K-Norm
      ↓
RoPE
      ↓
QKᵀ / √d
      ↓
Softmax
      ↓
× V
```

---

## 2. 为什么 Q/K 数值过大会有问题

对于一个 Attention Head：

\[
Q=[q_1,q_2,\dots,q_d]
\]

\[
K=[k_1,k_2,\dots,k_d]
\]

Attention Score：

\[
QK^T=q_1k_1+q_2k_2+\cdots+q_dk_d
\]

标准 Attention 会计算：

\[
\boxed{score=\frac{QK^T}{\sqrt d}}
\]

假设：

\[
Q=[100,200],\quad K=[100,200]
\]

那么：

\[
QK^T=50000
\]

即使除以：

\[
\sqrt2
\]

仍然有：

\[
\frac{50000}{1.414}\approx35355
\]

如果 Softmax 的输入过大，分布会变得非常尖锐，可能影响训练和数值稳定性。

---

## 3. QK-Norm 怎么解决

假设：

\[
Q=[100,200]
\]

先计算 RMS：

\[
RMS(Q)=
\sqrt{
\frac{100^2+200^2}{2}
}
\approx158.1
\]

归一化：

\[
Q_{norm}
=
\left[
\frac{100}{158.1},
\frac{200}{158.1}
\right]
\approx
[0.632,1.265]
\]

K 同理：

\[
K_{norm}\approx[0.632,1.265]
\]

点积：

\[
0.632^2+1.265^2\approx2
\]

原来：

\[
50000
\]

现在约为：

\[
2
\]

所以 Q/K 的整体尺度被稳定住了。

---

## 4. 为什么前面已经有 RMSNorm，还需要 QK-Norm

前面 Decoder 已经做：

\[
X'=RMSNorm(X)
\]

但随后还有：

\[
Q=X'W_Q
\]

\[
K=X'W_K
\]

其中：

\[
W_Q,W_K
\]

是可学习矩阵。

即使输入稳定，经过 Linear Projection 后，Q/K 仍然可能重新出现尺度变化。

例如：

\[
X=[1,1]
\]

\[
W_Q=
\begin{bmatrix}
100&0\\
0&100
\end{bmatrix}
\]

那么：

\[
Q=XW_Q=[100,100]
\]

所以：

> **输入稳定，不代表 Linear Projection 的输出一定稳定。**

因此：

```text
hidden_states
      ↓
RMSNorm
      ↓
Q/K Linear Projection
      ↓
Q/K 尺度可能重新漂移
      ↓
QK-Norm
```

---

## 5. QK-Norm 本质还是 RMSNorm

Q-Norm：

\[
Q'
=
w_Q
\odot
\frac{Q}{
\sqrt{
\frac1d\sum_i q_i^2+\epsilon
}
}
\]

K-Norm：

\[
K'
=
w_K
\odot
\frac{K}{
\sqrt{
\frac1d\sum_i k_i^2+\epsilon
}
}
\]

所以从数学操作上看：

> **QK-Norm = 对 Q 和 K 分别做 RMSNorm。**

---

## 6. QK-Norm 是按 Attention Head 做的

假设：

```text
num_heads = 4
head_dim = 128
```

Q 的逻辑 shape 可以写成：

```text
[B, T, 4, 128]
```

QK-Norm 主要沿每个 Head 的：

```text
128
```

这个维度做。

也就是：

```text
Head 0:
[q0, q1, ..., q127]
      ↓
RMSNorm

Head 1:
[q0, q1, ..., q127]
      ↓
RMSNorm
```

不是把：

```text
4 × 128 = 512
```

全部混在一起归一化。

---

## 7. QK-Norm 的 weight 是多大

如果：

```text
head_dim = 128
```

那么 Q-Norm 的 weight 通常就是：

```text
[128]
```

K-Norm 同样：

```text
[128]
```

因为它归一化的是每个 Head 内部的 Q/K 向量。

---

## 8. QK-Norm 的参数共享

假设：

```text
num_heads = 3
head_dim = 4
```

三个 Q Head：

```text
Head 0: [q00 q01 q02 q03]
Head 1: [q10 q11 q12 q13]
Head 2: [q20 q21 q22 q23]
```

Q-Norm weight：

\[
w_Q=[w_0,w_1,w_2,w_3]
\]

所有 Q Head 共用：

```text
[w0 w1 w2 w3]
```

K-Norm 同理。

---

## 9. 为什么已经有 /√d，还要 QK-Norm

标准 Attention：

\[
\frac{QK^T}{\sqrt d}
\]

这里的：

\[
\sqrt d
\]

主要解决：

> **head dimension 变大以后，点积天然容易变大的问题。**

而 QK-Norm 解决的是：

> **Q/K 自身数值幅度漂移的问题。**

二者作用不同。

| 组件 | 主要解决的问题 |
|---|---|
| \(/\sqrt d\) | 维度增加带来的点积尺度变化 |
| QK-Norm | Q/K 自身幅度变化导致的 score 不稳定 |

---

## 10. 用向量长度理解 QK-Norm

点积可以写成：

\[
QK^T
=
||Q||\,||K||\cos\phi
\]

其中：

- \(||Q||\)：Q 的长度
- \(||K||\)：K 的长度
- \(\phi\)：Q/K 的夹角

所以 Attention Score 同时受：

\[
\boxed{
||Q||\times||K||\times\cos\phi
}
\]

影响。

如果 \(||Q||\) 和 \(||K||\) 很大，即使方向关系没变，score 也可能暴涨。

QK-Norm 的核心作用就是：

> **稳定 Q 和 K 的长度，让 Attention 更稳定地反映 Q/K 的方向匹配关系。**

---

## 11. QK-Norm 和 Cosine Similarity 的关系

Cosine Similarity：

\[
cos(Q,K)
=
\frac{QK^T}{||Q||||K||}
\]

QK-Norm 会明显削弱 Q/K 长度变化的影响，所以点积会更偏向反映方向关系。

但不能简单说：

> QK-Norm 后的 Attention 就等于 cosine attention。

因为真实计算里还有：

- RMSNorm 的可学习 weight
- \(/\sqrt d\)
- RoPE
- Multi-Head / GQA
- 具体实现细节

---

## 12. QK-Norm 和 RoPE 的顺序

Qwen3 中可以理解成：

```text
Q/K Projection
      ↓
QK-Norm
      ↓
RoPE
      ↓
Attention
```

即：

\[
Q
\rightarrow
Norm(Q)
\rightarrow
RoPE(Q)
\]

\[
K
\rightarrow
Norm(K)
\rightarrow
RoPE(K)
\]

---

## 13. 为什么先 Norm，再 RoPE

二维旋转有一个重要性质：

\[
||R(\theta)x||=||x||
\]

也就是说：

> **RoPE 改变向量方向，但不会改变向量长度。**

例如：

\[
x=[3,4]
\]

长度：

\[
||x||_2=5
\]

旋转 \(90^\circ\) 后：

\[
[-4,3]
\]

长度仍然是：

\[
5
\]

所以：

```text
QK-Norm
↓
先稳定长度

RoPE
↓
再改变方向、加入位置信息
```

---

## 14. 完整 Attention 链路

假设某个 Token 的 hidden state 是：

\[
x
\]

先：

\[
x'=RMSNorm(x)
\]

然后：

\[
Q=x'W_Q
\]

\[
K=x'W_K
\]

\[
V=x'W_V
\]

再：

\[
Q_n=RMSNorm_Q(Q)
\]

\[
K_n=RMSNorm_K(K)
\]

然后：

\[
Q_r=RoPE(Q_n,m)
\]

\[
K_r=RoPE(K_n,n)
\]

再计算：

\[
score
=
\frac{Q_rK_r^T}{\sqrt d}
\]

然后：

\[
A=Softmax(score)
\]

最后：

\[
O=AV
\]

整体：

```text
hidden_states
      ↓
RMSNorm
      ↓
Q / K / V Projection
      ↓
Q-Norm / K-Norm
      ↓
RoPE
      ↓
QKᵀ / √d
      ↓
Softmax
      ↓
× V
```

---

## 15. 几个容易混淆的组件

| 组件 | 作用对象 | 核心目的 |
|---|---|---|
| Decoder RMSNorm | hidden state | 稳定 Decoder 输入尺度 |
| QK-Norm | 每个 Head 的 Q/K | 稳定投影后的 Q/K 尺度 |
| RoPE | Q/K | 加入相对位置信息 |
| \(/\sqrt d\) | QK 点积 | 补偿维度导致的点积规模变化 |

---

## 16. 和 llama.cpp Tensor 对起来

以后如果在 llama.cpp / GGUF 中看到类似：

```text
attn_q_norm.weight
attn_k_norm.weight
```

假设：

```text
head_dim = 128
```

那么它们可能就是：

```text
shape = [128]
```

可以直接翻译成：

```text
attn_q_norm.weight
→ Q-Norm 的 128 个可学习缩放参数

attn_k_norm.weight
→ K-Norm 的 128 个可学习缩放参数
```

它们不是大矩阵，而是一维缩放向量。

---

## 17. 最重要的数学直觉

牢牢记住：

\[
\boxed{
QK^T
=
||Q||\,||K||\cos\phi
}
\]

如果：

\[
||Q||,||K||
\]

乱飘，

那么 Attention Score 也容易乱飘。

QK-Norm 就是在做：

```text
Q 的长度 → 稳定
K 的长度 → 稳定
```

让：

\[
QK^T
\]

更稳定地反映 Q 和 K 的匹配关系，而不是被整体数值大小过度影响。

---

## 18. 一句话总结

> **Qwen3 先对 hidden state 做 RMSNorm，但 Q/K 经过 Linear Projection 后数值尺度仍可能重新漂移，所以在 Attention 点积之前，再分别对每个 Head 的 Q 和 K 做 RMSNorm；这样可以稳定 Q/K 的长度。随后 RoPE 负责改变方向、注入位置信息，而 \(/\sqrt d\) 负责补偿维度对点积规模的影响。**

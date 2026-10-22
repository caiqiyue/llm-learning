# Qwen3 学习笔记：RoPE（Rotary Position Embedding）

> **量化实证（feat-017）**：本组件「到底改了什么」的实测数字见
> `qwen3-learning/Qwen3_训练推理完整学习笔记.md` **§12 退化对照**
> —— 把 Qwen3 退化成 GPT-2 结构、只开 `use_rope` 这一个组件，
> 与 GPT-2 的 `max_abs_diff` = **1.541e+00**（相对差 **2.08**）。
> 把位置信息从「加法式 sinusoidal PE」换成「RoPE」是**行为改变最大**的一个组件，相对差 2.08 —— 换句话说 logits 已经完全不是一回事。

## 1. RoPE 是什么

RoPE 全称 **Rotary Position Embedding（旋转位置编码）**。

它的目标和传统位置编码一样：让 Attention 知道 Token 的位置信息。

传统位置编码更像：

```text
Token Embedding
      +
Position Encoding
      ↓
Hidden State
      ↓
Attention
```

RoPE 则更接近：

```text
hidden_states
      ↓
生成 Q / K / V
      ↓
对 Q 和 K 做旋转
      ↓
Attention
```

即：

\[
Q' = RoPE(Q, position)
\]

\[
K' = RoPE(K, position)
\]

然后：

\[
Attention(Q',K',V)
\]

注意：**RoPE 主要作用于 Q 和 K，而不是直接加到 hidden state 上。**

---

## 2. 为什么 Attention 需要位置信息

标准 Attention：

\[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
\]

然后：

\[
Attention(Q,K,V)
=
Softmax\left(
\frac{QK^T}{\sqrt d}
\right)V
\]

单纯的 \(QK^T\) 更主要表示两个 Token 在内容上的匹配程度，但它本身不知道：

```text
Token A 在位置 2
Token B 在位置 5
```

也不知道：

```text
A 和 B 相距 3 个位置
```

所以需要额外注入位置信息。

---

## 3. RoPE 的数学基础：二维旋转

假设二维向量：

\[
x=
\begin{bmatrix}
x_1\\
x_2
\end{bmatrix}
\]

旋转角度为 \(\alpha\)，旋转矩阵：

\[
R(\alpha)
=
\begin{bmatrix}
\cos\alpha & -\sin\alpha\\
\sin\alpha & \cos\alpha
\end{bmatrix}
\]

旋转后：

\[
x'=R(\alpha)x
\]

展开：

\[
x_1'=x_1\cos\alpha-x_2\sin\alpha
\]

\[
x_2'=x_1\sin\alpha+x_2\cos\alpha
\]

### 一个最简单例子

\[
x=
\begin{bmatrix}
1\\
0
\end{bmatrix}
\]

旋转 \(90^\circ\)：

\[
\cos90^\circ=0,\quad \sin90^\circ=1
\]

因此：

\[
(1,0)\rightarrow(0,1)
\]

所以所谓“旋转”，本质上就是改变二维向量的方向。

---

## 4. position 怎么变成旋转角度

如果某个二维维度对的旋转速度是：

\[
\theta
\]

Token 的位置是：

\[
m
\]

那么实际旋转角度：

\[
\boxed{angle=m\theta}
\]

例如：

\[
\theta=0.5
\]

则：

```text
position 0 → angle = 0
position 1 → angle = 0.5
position 2 → angle = 1.0
position 3 → angle = 1.5
```

所以 \(\theta\) 可以理解为：

> 每向后移动一个 Token，这一对维度要多旋转多少弧度。

---

## 5. 为什么不同维度需要不同频率

如果所有维度对都使用同样的旋转速度，那么每一组表达的位置模式都一样，信息太单一。

因此 RoPE 为不同维度对分配不同的：

\[
\theta_i
\]

可以类比钟表：

```text
秒针 → 高频
分针 → 中频
时针 → 低频
```

多种频率组合起来，可以表达更丰富、更长范围的位置关系。

---

## 6. RoPE 的频率怎么计算

经典形式可写为：

\[
\boxed{
\theta_i
=
base^{-\frac{2i}{d}}
}
\]

其中：

- \(d\)：head dimension
- \(i\)：第几个二维维度对
- \(base\)：控制频率分布的常数

经典形式常见：

\[
base=10000
\]

注意：\(\theta_i\) 更适合理解为“旋转速度”或“频率”，不是最终角度。

真正的旋转角度是：

\[
\boxed{
angle_{m,i}=m\theta_i
}
\]

---

## 7. 完整手算一个 RoPE

假设：

```text
head_dim = 4
```

某个 Token 的：

\[
Q=[1,2,3,4]
\]

为了方便手算，设：

\[
base=100
\]

Token 位置：

\[
position=2
\]

### 7.1 两两分组

\[
Q=[1,2,3,4]
\]

拆成：

\[
(1,2)
\]

和：

\[
(3,4)
\]

也就是：

```text
Q = [1, 2, 3, 4]

     └─┬─┘ └─┬─┘
      第0对   第1对
```

### 7.2 计算两组频率

因为：

\[
d=4
\]

所以有两组：

```text
i = 0
i = 1
```

第 0 对：

\[
\theta_0
=
100^{-\frac{2\times0}{4}}
=
100^0
=
1
\]

第 1 对：

\[
\theta_1
=
100^{-\frac{2\times1}{4}}
=
100^{-1/2}
=
0.1
\]

因此：

\[
\boxed{
[\theta_0,\theta_1]
=
[1,0.1]
}
\]

可以理解成：

```text
第 0 对：每前进一个 Token，多旋转 1 rad
第 1 对：每前进一个 Token，多旋转 0.1 rad
```

### 7.3 position × frequency

当前：

\[
position=2
\]

所以：

\[
angle_{2,0}=2\times1=2\ rad
\]

\[
angle_{2,1}=2\times0.1=0.2\ rad
\]

即：

```text
(1,2) → rotate 2 rad
(3,4) → rotate 0.2 rad
```

### 7.4 第一对旋转

第一对：

\[
(1,2)
\]

使用：

\[
\cos2\approx-0.416
\]

\[
\sin2\approx0.909
\]

计算：

\[
x'
=
1\times(-0.416)
-
2\times0.909
\approx -2.234
\]

\[
y'
=
1\times0.909
+
2\times(-0.416)
\approx0.077
\]

所以：

\[
\boxed{
(1,2)
\rightarrow
(-2.234,0.077)
}
\]

### 7.5 第二对旋转

第二对：

\[
(3,4)
\]

旋转角度：

\[
0.2\ rad
\]

近似：

\[
\cos0.2\approx0.980
\]

\[
\sin0.2\approx0.199
\]

计算：

\[
x'
=
3\times0.980
-
4\times0.199
\approx2.144
\]

\[
y'
=
3\times0.199
+
4\times0.980
\approx4.517
\]

所以：

\[
\boxed{
(3,4)
\rightarrow
(2.144,4.517)
}
\]

### 7.6 拼接回来

因此：

\[
\boxed{
Q_{rope}
=
[-2.234,0.077,2.144,4.517]
}
\]

这就是一次完整的 RoPE。

---

## 8. 换回经典 base = 10000

如果仍然：

\[
d=4
\]

那么：

\[
\theta_0=10000^0=1
\]

\[
\theta_1=10000^{-1/2}=0.01
\]

所以：

\[
\boxed{
[\theta_0,\theta_1]
=
[1,0.01]
}
\]

如果：

\[
position=2
\]

则：

\[
angle=[2,0.02]
\]

也就是：

```text
第 0 对 → rotate 2 rad
第 1 对 → rotate 0.02 rad
```

---

## 9. 为什么 Q 和 K 都要做 RoPE

假设：

```text
Q Token position = 2
K Token position = 5
```

对于某一组频率：

\[
\theta_i
\]

Q 旋转：

\[
2\theta_i
\]

K 旋转：

\[
5\theta_i
\]

Attention 中最终做：

\[
Q'K'^T
\]

两者的相对角度是：

\[
5\theta_i-2\theta_i
=
(5-2)\theta_i
=
3\theta_i
\]

这里：

\[
3=5-2
\]

正好是两个 Token 的相对位置差。

---

## 10. 为什么 RoPE 最终表达的是相对位置

设：

\[
Q'_m=R(m\theta)Q
\]

\[
K'_n=R(n\theta)K
\]

则：

\[
(Q'_m)^T K'_n
\]

代入：

\[
(R(m\theta)Q)^T
R(n\theta)K
\]

得到：

\[
Q^T
R(m\theta)^T
R(n\theta)K
\]

旋转矩阵满足：

\[
R(\theta)^T=R(-\theta)
\]

因此：

\[
Q^T
R(-m\theta)
R(n\theta)
K
\]

又因为：

\[
R(a)R(b)=R(a+b)
\]

所以：

\[
R(-m\theta)R(n\theta)
=
R((n-m)\theta)
\]

最终：

\[
\boxed{
(Q'_m)^TK'_n
=
Q^TR((n-m)\theta)K
}
\]

最重要的是：

\[
\boxed{n-m}
\]

因此：

> Q/K 各自按照绝对位置旋转，但做 Attention 点积后，自然体现出 Token 之间的相对位置。

---

## 11. 为什么主要旋转 Q 和 K，不旋转 V

Attention Score 的核心：

\[
score_{ij}=Q_iK_j^T
\]

如果希望注意力分数与位置关系有关，最自然的方法就是调整：

\[
Q_i
\]

和：

\[
K_j
\]

因此：

\[
Q_i\rightarrow RoPE(Q_i,i)
\]

\[
K_j\rightarrow RoPE(K_j,j)
\]

而 V 更主要表示：

> 真正被 Attention 权重加权汇总的内容。

所以标准 RoPE 不需要对 V 做同样的旋转。

---

## 12. 真实高维 Q/K 怎么旋转

真实模型中一个 Attention Head 可能有：

```text
head_dim = 128
```

那么：

\[
Q=
[q_0,q_1,q_2,\dots,q_{127}]
\]

概念上两两分组：

```text
(q0, q1)
(q2, q3)
(q4, q5)
...
(q126, q127)
```

一共：

\[
128/2=64
\]

个二维维度对。

每一对使用自己的：

\[
\theta_i
\]

再根据：

\[
position=m
\]

得到：

\[
m\theta_i
\]

并完成二维旋转。

---

## 13. sin/cos 在 RoPE 中真正负责什么

RoPE 中：

\[
position
\rightarrow
m\theta_i
\rightarrow
\cos(m\theta_i),\sin(m\theta_i)
\rightarrow
二维旋转
\]

所以 sin/cos 的作用是：

> **作为二维旋转所需的系数。**

因为二维旋转天然需要：

\[
x'
=
x\cos\alpha-y\sin\alpha
\]

\[
y'
=
x\sin\alpha+y\cos\alpha
\]

其中：

\[
\alpha=m\theta_i
\]

---

## 14. 实际代码为什么常看不到旋转矩阵

理论上我们写：

\[
R(\theta)
=
\begin{bmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{bmatrix}
\]

但实现时通常直接计算：

```text
q0' = q0*cos - q1*sin
q1' = q0*sin + q1*cos
```

所以在 Transformers、llama.cpp 或 CUDA Kernel 中，通常不需要真正创建一个巨大的 rotation matrix。

---

## 15. RoPE 在 Qwen3 Attention 中的位置

可以把 Qwen3 Attention 简化为：

```text
hidden_states
      ↓
RMSNorm
      ↓
Q / K / V Projection
      ↓
QK-Norm
      ↓
RoPE(Q, K)
      ↓
Attention Score
      ↓
Softmax
      ↓
× V
      ↓
O Projection
      ↓
Residual
```

因此 RoPE 的位置就是：

> **Q/K 生成之后，Attention Score 计算之前。**

---

## 16. 最终心智模型

可以把 RoPE 理解成：

> **Q/K 的每两个维度是一根“小指针”；不同指针转速不同，Token 的 position 决定每根指针当前转到了什么角度。**

例如：

```text
第 0 对维度 → 秒针
第 1 对维度 → 分针
第 2 对维度 → 时针
...
```

position 越大，每根指针转得越多。

由于不同维度对速度不同，一个 position 会对应一组独特的角度组合。

Q 和 K 都这样旋转。

最终：

\[
Q'K'^T
\]

真正关键的是两者之间的角度差，而角度差最终对应：

\[
\boxed{n-m}
\]

也就是 Token 之间的相对位置。

---

## 17. 最重要的公式

### 频率

\[
\boxed{
\theta_i
=
base^{-\frac{2i}{d}}
}
\]

### 第 m 个位置在第 i 对维度上的旋转角度

\[
\boxed{
angle_{m,i}
=
m\theta_i
}
\]

### 二维旋转

\[
\boxed{
x'
=
x\cos\alpha-y\sin\alpha
}
\]

\[
\boxed{
y'
=
x\sin\alpha+y\cos\alpha
}
\]

### Q/K 点积中的相对位置关系

\[
\boxed{
(Q'_m)^TK'_n
=
Q^TR((n-m)\theta)K
}
\]

---

## 18. 一句话总结

> **RoPE 把 Q/K 的维度两两组成二维平面，不同二维平面使用不同旋转频率；Token 的 position 决定实际旋转角度，sin/cos 用来完成旋转；最终 Q/K 做点积时，自然把两个 Token 的相对位置差编码进 Attention Score。**

# Qwen3 学习笔记：RMSNorm

## 1. RMSNorm 是什么

RMSNorm（Root Mean Square
Normalization，均方根归一化）是现代大模型中常见的归一化方法。

对于已经学过 GPT / Transformer 的情况，可以先这样理解：

> **RMSNorm 是一种比 LayerNorm 更简化的归一化方式：它不减均值，重点控制
> hidden state 的整体数值尺度。**

Qwen3 的 Decoder Layer 使用 RMSNorm，而不是经典 GPT 中常见的 LayerNorm。

------------------------------------------------------------------------

## 2. 为什么需要归一化

Transformer 在经过很多层计算之后，hidden state
的数值可能不断变大或变小。

例如一个 Token 的 hidden state：

``` text
[1, 2, 3, 4]
```

如果某些计算让它整体放大 100 倍：

``` text
[100, 200, 300, 400]
```

虽然两个向量的相对关系相同，但是数值尺度完全不同。

归一化的一个重要目的就是：

> **控制每一层 hidden state 的数值尺度，使深层网络的计算更加稳定。**

------------------------------------------------------------------------

## 3. RMS 是什么

RMS 的全称：

``` text
Root Mean Square
均方根
```

对于一个长度为 (d) 的向量：

\[ x=\[x_1,x_2,`\dots`{=tex},x_d\] \]

RMS 定义为：

\[ `\boxed{
RMS(x)=
\sqrt{
\frac{1}{d}
\sum_{i=1}^{d}x_i^2+\epsilon
}
}`{=tex} \]

可以拆成四步：

``` text
x
↓
每个元素平方
↓
求平均
↓
开根号
↓
得到一个 RMS 数值
```

### 例子

假设：

\[ x=\[1,2,3,4\] \]

平方：

\[ \[1,4,9,16\] \]

求平均：

\[ `\frac{1+4+9+16}{4}`{=tex}=7.5 \]

开根：

\[ RMS(x)=`\sqrt{7.5}`{=tex}`\approx2.739`{=tex} \]

因此，这个向量的 RMS 是一个标量：

\[ `\boxed{2.739}`{=tex} \]

可以把 RMS 粗略理解为：

> **这个向量整体的典型数值大小。**

------------------------------------------------------------------------

## 4. 为什么要先平方

平方有两个重要作用。

### 4.1 消除正负号影响

例如：

\[ (-3)\^2=9 \]

\[ 3\^2=9 \]

对于数值尺度而言，`-100` 和 `100`
都属于绝对值很大的数，不能因为一个是负数就忽略它。

### 4.2 衡量整个向量的整体大小

平方之后求平均：

\[ `\frac{1}{d}`{=tex}`\sum`{=tex}\_i x_i\^2 \]

可以衡量整个向量的平均"能量"。

最后再开根号，把数值恢复到与原始数据相近的尺度。

------------------------------------------------------------------------

## 5. RMS 与 L2 Norm 的关系

L2 范数：

\[ \|\|x\|\|\_2= `\sqrt{\sum_{i=1}^{d}x_i^2}`{=tex} \]

RMS：

\[ RMS(x)= `\sqrt{
\frac{1}{d}\sum_{i=1}^{d}x_i^2
}`{=tex} \]

因此：

\[ `\boxed{
RMS(x)=\frac{||x||_2}{\sqrt d}
}`{=tex} \]

所以 RMSNorm 也可以理解为：

> **根据 hidden state 向量的整体长度，对它的数值尺度进行归一化。**

------------------------------------------------------------------------

## 6. RMSNorm 的完整公式

RMSNorm 的核心公式：

\[ `\boxed{
RMSNorm(x)
=
w\odot
\frac{x}{
\sqrt{
\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon
}
}
}`{=tex} \]

其中：

-   (x)：输入 hidden state
-   (d)：hidden size
-   (`\epsilon`{=tex})：防止除零并提高数值稳定性的很小常数
-   (w)：可学习的 weight
-   (`\odot`{=tex})：逐元素乘法

整体流程：

``` text
hidden state x
      ↓
计算 RMS(x)
      ↓
x / RMS(x)
      ↓
逐元素乘 weight
      ↓
输出
```

------------------------------------------------------------------------

## 7. 完整计算例子

假设：

\[ x=\[1,2,3,4\] \]

前面已经得到：

\[ RMS(x)`\approx2.739`{=tex} \]

首先进行归一化：

\[ `\hat{x}`{=tex} = `\frac{x}{RMS(x)}`{=tex} \]

得到：

\[ `\hat{x}`{=tex} `\approx
[0.365,\ 0.730,\ 1.095,\ 1.461]`{=tex}\]

假设暂时：

\[ w=\[1,1,1,1\] \]

最终输出就是：

\[ y= \[0.365, 0.730, 1.095, 1.461\] \]

可以看到：

``` text
原来：

[1, 2, 3, 4]

RMSNorm 后：

[0.365, 0.730, 1.095, 1.461]
```

相对关系基本没有改变，但整体尺度被控制住了。

------------------------------------------------------------------------

## 8. RMSNorm 对整体缩放不敏感

假设把输入整体放大 100 倍：

\[ x=\[100,200,300,400\] \]

此时：

\[ RMS(x)`\approx273.86`{=tex} \]

归一化：

\[ `\frac{x}{RMS(x)}`{=tex} `\approx
[0.365,\ 0.730,\ 1.095,\ 1.461]`{=tex}\]

和：

\[ \[1,2,3,4\] \]

归一化后的结果基本相同。

因此可以粗略理解为：

\[ RMSNorm(100x)`\approx `{=tex}RMSNorm(x) \]

这体现了 RMSNorm 的核心作用：

> **削弱 hidden state 整体尺度变化带来的影响。**

------------------------------------------------------------------------

## 9. epsilon 的作用

RMSNorm 实际计算时需要：

\[ RMS(x)= `\sqrt{
\frac1d\sum_i x_i^2+\epsilon
}`{=tex} \]

而不是简单地：

\[ `\sqrt{\frac1d\sum_i x_i^2}`{=tex} \]

原因是如果：

\[ x=\[0,0,0,0\] \]

那么：

\[ RMS(x)=0 \]

之后会出现：

\[ `\frac{x}{0}`{=tex} \]

这是非法计算。

所以加入一个很小的：

\[ `\epsilon`{=tex} \]

例如：

\[ 10\^{-6} \]

避免除零，并改善浮点计算的数值稳定性。

------------------------------------------------------------------------

## 10. RMSNorm 的 weight 到底是什么

这是理解 RMSNorm 最关键的地方之一。

假设：

``` text
hidden_size = 2048
```

那么一个 Token 的 hidden state 是：

\[ x= \[x_1,x_2,`\dots`{=tex},x\_{2048}\] \]

RMSNorm 的 weight 是：

\[ w= \[w_1,w_2,`\dots`{=tex},w\_{2048}\] \]

所以：

``` text
weight.shape = [2048]
```

也就是说：

> **hidden state 的每一个特征维度，对应一个可学习的缩放系数。**

对应关系：

``` text
x1       → × w1       → y1
x2       → × w2       → y2
x3       → × w3       → y3
...
x2048    → × w2048    → y2048
```

因此 RMSNorm 不需要一个：

``` text
2048 × 2048
```

的矩阵。

只需要：

``` text
2048
```

个参数。

------------------------------------------------------------------------

## 11. 为什么 weight 是一维，而不是二维矩阵

因为 RMSNorm **不会混合不同 hidden dimension 的信息**。

RMSNorm：

\[ y_i=`\hat{x}`{=tex}\_i w_i \]

本质上是：

``` text
x1 → w1 → y1
x2 → w2 → y2
x3 → w3 → y3
```

第 1 个维度不会和第 2、3、4 个维度做加权求和。

### 对比 Linear

Linear：

\[ y=xW \]

不同输入维度会共同参与输出维度的计算，因此需要二维权重矩阵。

例如：

``` text
Linear:

x1 ─┬→ y1
x2 ─┼→ y1
x3 ─┼→ y1
... ┘
```

所以一个 `2048 → 2048` 的 Linear 层可能需要：

\[ 2048`\times2048`{=tex} \]

级别的权重。

而 RMSNorm 只是逐元素缩放：

``` text
RMSNorm:

x1    → ×w1    → y1
x2    → ×w2    → y2
...
x2048 → ×w2048 → y2048
```

因此：

\[ `\boxed{weight.shape=[2048]}`{=tex} \]

就足够了。

------------------------------------------------------------------------

## 12. 多个 Token 为什么仍然只有 2048 个 weight

假设：

``` text
seq_len = 3
hidden_size = 4
```

hidden states：

\[ X=
```{=tex}
\begin{bmatrix}
x_{11}&x_{12}&x_{13}&x_{14}\\
x_{21}&x_{22}&x_{23}&x_{24}\\
x_{31}&x_{32}&x_{33}&x_{34}
\end{bmatrix}
```
\]

shape：

``` text
[3, 4]
 ↑  ↑
 T  hidden
```

RMSNorm weight：

\[ w=\[w_1,w_2,w_3,w_4\] \]

shape：

``` text
[4]
```

三个 Token 共用这一套参数：

``` text
Token 1:
[x11 x12 x13 x14]
  ↓   ↓   ↓   ↓
[w1  w2  w3  w4]

Token 2:
[x21 x22 x23 x24]
  ↓   ↓   ↓   ↓
[w1  w2  w3  w4]

Token 3:
[x31 x32 x33 x34]
  ↓   ↓   ↓   ↓
[w1  w2  w3  w4]
```

所以：

> **RMSNorm 的 weight 数量取决于 hidden size，而不是 Token 数量。**

------------------------------------------------------------------------

## 13. 加上 Batch 后怎么理解

假设：

``` text
batch_size = 2
seq_len = 10
hidden_size = 2048
```

hidden states：

``` text
X.shape = [2, 10, 2048]
           ↑   ↑    ↑
           B   T    hidden
```

RMSNorm：

``` text
weight.shape = [2048]
```

计算时可以理解为广播：

``` text
[2, 10, 2048]
       ×
    [2048]
       ↓
[2, 10, 2048]
```

一共有：

\[ 2`\times10`{=tex}=20 \]

个 Token hidden states。

每一个 Token 都独立计算自己的 RMS，但是：

> **20 个 Token 共用同一套 2048 个可学习 weight。**

这是神经网络中的参数共享。

------------------------------------------------------------------------

## 14. 一个非常重要的区分

对于：

``` text
X.shape = [B, T, 2048]
```

RMS 是沿最后一个 hidden dimension 计算的。

因此每个 Token：

``` text
[x1, x2, ..., x2048]
```

都会得到自己的一个 RMS 标量。

例如：

``` text
Token 1 → RMS1
Token 2 → RMS2
Token 3 → RMS3
...
```

但是它们使用的 weight 都是：

``` text
[w1, w2, ..., w2048]
```

所以：

> **RMS 是每个 Token 自己算的；weight 是所有 Token 共享的。**

这是非常值得记住的一句话。

------------------------------------------------------------------------

## 15. weight 为什么还要学习

如果只有：

\[ `\frac{x}{RMS(x)}`{=tex} \]

模型的每个 hidden dimension 都会被统一按照归一化规则处理。

加入：

\[ w_i \]

以后，模型可以自己学习：

> "整体尺度你帮我稳定，但是不同 hidden dimension
> 最终应该放大还是缩小，由训练来决定。"

例如：

\[ `\hat{x}`{=tex} = \[0.36,0.73,1.10,1.46\] \]

模型学到：

\[ w=\[0.5,1.2,0.8,1.5\] \]

那么：

\[ y=`\hat{x}`{=tex}`\odot `{=tex}w \]

不同维度就可以拥有不同的最终尺度。

这些 weight 和 Linear 权重一样，会通过：

``` text
Forward
↓
Loss
↓
Backward
↓
Gradient
↓
Optimizer
↓
更新 weight
```

不断学习。

------------------------------------------------------------------------

## 16. LayerNorm 与 RMSNorm 的核心区别

LayerNorm：

\[ `\hat{x}`{=tex} = `\frac{x-\mu}`{=tex}
{`\sqrt{\sigma^2+\epsilon}`{=tex}} \]

主要包含：

``` text
计算 mean
↓
减 mean
↓
计算 variance
↓
除标准差
↓
乘 γ
↓
加 β
```

RMSNorm：

\[ `\hat{x}`{=tex} = `\frac{x}`{=tex}
{`\sqrt{\frac1d\sum_i x_i^2+\epsilon}`{=tex}} \]

主要包含：

``` text
平方
↓
平均
↓
开根
↓
相除
↓
乘 weight
```

核心区别：

  项目            LayerNorm   RMSNorm
  --------------- ----------- ----------
  计算均值        是          否
  减均值          是          否
  控制数值尺度    是          是
  可学习 weight   是          是
  bias            通常有      通常没有
  计算            相对更多    更简单
  Qwen3           否          是

可以把 RMSNorm 的设计思想理解成：

> **保留归一化中非常重要的尺度控制能力，同时去掉
> re-centering（减均值）这一步。**

------------------------------------------------------------------------

## 17. RMSNorm 在 Qwen3 Decoder Layer 中的位置

Qwen3 使用 Pre-Norm 结构。

一个 Decoder Layer 可以简化成：

``` text
                  x
                  │
                  ├──────────────────┐
                  ↓                  │
               RMSNorm               │
                  ↓                  │
              Attention              │
                  ↓                  │
                  + ←────────────────┘
                  ↓
                  x'
                  │
                  ├──────────────────┐
                  ↓                  │
               RMSNorm               │
                  ↓                  │
               SwiGLU                │
                  ↓                  │
                  + ←────────────────┘
                  ↓
                output
```

对应公式：

\[ x' = x+Attention(RMSNorm(x)) \]

然后：

\[ y= x'+MLP(RMSNorm(x')) \]

模型最后还会有 Final RMSNorm。

------------------------------------------------------------------------

## 18. 和 llama.cpp 中的 Tensor 对起来

在 llama.cpp 中可能看到类似：

``` cpp
layer.attn_norm = create_tensor(
    tn(LLM_TENSOR_ATTN_NORM, "weight", i),
    {n_embd},
    0
);
```

假设：

``` text
n_embd = 2048
```

那么：

``` cpp
{n_embd}
```

就是创建一个长度为：

``` text
2048
```

的 RMSNorm weight。

从数学上就是：

\[ \[w_1,w_2,`\dots`{=tex},w\_{2048}\] \]

在 ggml Tensor 中可能看到：

``` text
ne = [2048, 1, 1, 1]
```

它并不是：

``` text
2048 × 1 × 1 × 1
```

有什么特殊神经网络含义。

本质上就是一个长度为 2048 的一维向量；后面的 `1,1,1` 是 ggml Tensor
维度表示中的其余维度。

------------------------------------------------------------------------

## 19. 最终心智模型

看到：

``` text
RMSNorm
```

脑子里直接翻译成：

``` text
一个 Token 的 hidden state
[x1, x2, ..., x2048]
          ↓
对这 2048 个数平方
          ↓
求平均
          ↓
开根号
          ↓
得到这个 Token 自己的一个 RMS
          ↓
2048 个 hidden value 全部除以这个 RMS
          ↓
分别乘：
[w1, w2, ..., w2048]
          ↓
得到新的 2048 维 hidden state
```

最重要的三个结论：

1.  **RMSNorm 是对每个 Token 的 hidden dimension 做归一化。**
2.  **每个 Token 自己计算一个 RMS，但所有 Token 共用同一套 weight。**
3.  **hidden size = 2048 时，RMSNorm 只需要 2048 个
    weight，因为它做逐元素缩放，而不是像 Linear 一样混合不同维度。**

最终公式：

\[ `\boxed{
RMSNorm(x)
=
w\odot
\frac{x}{
\sqrt{
\frac1d\sum_{i=1}^{d}x_i^2+\epsilon
}
}
}`{=tex} \]

这就是后续阅读 Qwen3、Transformers 和 llama.cpp 中 RMSNorm
实现时最核心的数学与工程基础。

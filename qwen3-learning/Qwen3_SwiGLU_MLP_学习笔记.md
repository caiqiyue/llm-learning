# Qwen3 学习笔记：SwiGLU MLP

> 学习主线：Transformer/GPT 的传统 MLP → SwiGLU 的 Gate/Up/Down → SiLU 与逐元素门控 → 手算 → Tensor Shape → Qwen3-0.6B / llama.cpp。

## 1. 它在 Qwen3 Decoder Layer 的哪里？

```text
hidden_states x
     ├──────────────────────────────┐
     ↓                              │
  RMSNorm                           │
     ↓                              │
  Q/K/V → QK-Norm → RoPE → GQA      │
     ↓                              │
  O Projection                      │
     ↓                              │
     + ←────────────────────────────┘
     ↓
     x'
     ├──────────────────────────────┐
     ↓                              │
  RMSNorm                           │
     ↓                              │
  SwiGLU MLP                        │
     ↓                              │
     + ←────────────────────────────┘
     ↓
  Layer Output
```

用行向量/批量 Tensor 的写法：

\[
x'=x+\operatorname{Attention}(\operatorname{RMSNorm}_1(x))
\]

\[
y=x'+\operatorname{SwiGLU}(\operatorname{RMSNorm}_2(x'))
\]

**分工：** Attention 在不同 Token 间汇聚信息；SwiGLU 对每个 Token 自己的 hidden 特征做非线性变换。普通 Dense MLP 不沿序列维度混合 Token，所有 Token 共用相同的 MLP 权重。

## 2. 和 GPT 传统 MLP 的区别

经典的两层 GPT MLP（省略 bias）：

```text
x [H] → Linear [I] → GELU [I] → Linear [H]
```

Qwen3 Dense 的 SwiGLU：

```text
                    ┌→ gate_proj [I] → SiLU ─┐
x [H] ─────────────┤                         × → down_proj [H]
                    └→ up_proj   [I] ─────────┘
```

- **gate_proj**：产生输入相关的调节信号。
- **up_proj**：产生候选中间特征。
- **SiLU + 按元素乘法**：用 Gate 分支调节 Up 分支。
- **down_proj**：把中间维度降回 hidden size，以便参与残差相加。

传统 MLP 通常有两个 Linear；这里有三个 Linear（两个并行升维投影 + 一个降维投影）。

### 2.1 参数量对照：三个矩阵怎么做到不爆

多一个矩阵就多一份参数，所以中间维 `I` 必须比 GPT-2 的 `4H` 窄。
参数量 = **矩阵个数 × H × I**（省略 bias）：

| 方案 | 矩阵个数 | 中间维 I | 参数量 | H=1024 时 |
|------|---------|---------|--------|-----------|
| GPT-2 `4H` + GELU | 2 | 4H | `8H²` | 8,388,608 |
| SwiGLU，I 精确取 `8/3·H` | 3 | 8/3·H | `8H²` | — |
| SwiGLU，I = 3H | 3 | 3H | `9H²` | 9,437,184 |

**结论一：8/3 规则的设计意图是「参数量持平」。**
`3 × (8/3) = 8 = 2 × 4`，两边都是 `8H²`。如果中间维能精确取 `8/3·H`，
SwiGLU 和 GPT-2 的 4×-GELU 参数量**完全相等**。

**结论二：一旦对齐到整数倍，SwiGLU 就会略贵，且超出量只来自对齐。**
本仓库 tiny 配置 `H=128`：`8/3·128 = 341.33`，向上对齐到 128 的倍数 → `I=384`。
实测（`tests/test_swiglu.py` 的输出）：

```text
[参数对比] n_embd=128: SwiGLU(中间维=384) 147,456 vs GPT-2 GELU(中间维=512) 131,072 -> 比值 1.125
```

`3×384 / (2×512) = 9/8 = 1.125`。这个 12.5% 的超出**全部来自向上对齐**
（`341.33 → 384`），与「三个矩阵」本身无关。

**结论三：Qwen3 实际用的不是 8/3 规则。**
本笔记 §8 里 Qwen3-0.6B 的真实尺寸是 `hidden=1024, intermediate=3072`，
即 **3×hidden**，不是 `8/3·1024 = 2730.67`。
8/3 规则来自 LLaMA 系；Qwen3 选了 3×。
两者算出的 SwiGLU/GPT-2 比值都是 `9/8 = 1.125`（因为最终都是 I = 3H）。
> 常见误解：「SwiGLU 参数比 GPT-2 多 50% / 少 25%」这类说法都站不住：
> 三矩阵的代价被更窄的中间维抵消掉了，净效果是**多 12.5%**（在 I=3H 时）。

## 3. 关键数学公式

采用 PyTorch 常见的行向量记法，设输入最后一维为 \(H\)，中间维度为 \(I\)：

\[
G=XW_{gate},\quad U=XW_{up}
\]

\[
M=\operatorname{SiLU}(G)\odot U
\]

\[
\boxed{\operatorname{SwiGLU}(X)=\bigl[\operatorname{SiLU}(XW_{gate})\odot(XW_{up})\bigr]W_{down}}
\]

其中 \(\odot\) 表示**逐元素相乘**，不是矩阵乘法。权重形状（此处按行向量右乘数学约定）：\(W_{gate},W_{up}\in\mathbb R^{H\times I}\)，\(W_{down}\in\mathbb R^{I\times H}\)。PyTorch `nn.Linear` 的 `.weight` 实际存储形状是 `[out_features, in_features]`。

## 4. SiLU 到底是什么？

\[
\boxed{\operatorname{SiLU}(z)=z\,\sigma(z)=\frac{z}{1+e^{-z}}}
\]

| z | Sigmoid(z) | SiLU(z) |
|---:|---:|---:|
| -3 | 0.0474 | -0.1423 |
| -2 | 0.1192 | -0.2384 |
| -1 | 0.2689 | -0.2689 |
| 0 | 0.5000 | 0 |
| 1 | 0.7311 | 0.7311 |
| 2 | 0.8808 | 1.7616 |
| 3 | 0.9526 | 2.8577 |

对比 ReLU：ReLU 会把所有负输入直接变成 0；SiLU 对负值进行平滑抑制，部分小负值仍可得到负输出。Gate 不是 0/1 的硬开关：其输出可接近零、为负或大于一，因而能够抑制、放大或改变 Up 分支特征的符号。

## 5. 完整手算：从输入走到输出

为便于手算，使用 \(H=I=2\)，输入：

\[
x=[1,2]
\]

**步骤 1：Gate Projection**

\[
W_{gate}=\begin{bmatrix}1&0\\0&-1\end{bmatrix},\quad
G=xW_{gate}=[1,-2]
\]

**步骤 2：Gate 经过 SiLU**

\[
\operatorname{SiLU}(G)=[0.7311,-0.2384]
\]

**步骤 3：Up Projection**

\[
W_{up}=\begin{bmatrix}2&1\\0&1\end{bmatrix},\quad
U=xW_{up}=[2,3]
\]

**步骤 4：逐元素门控**

\[
M=[0.7311,-0.2384]\odot[2,3]
=[1.4622,-0.7152]
\]

**步骤 5：Down Projection**

\[
W_{down}=\begin{bmatrix}1&0\\1&1\end{bmatrix}
\]

\[
y=MW_{down}=[0.7470,-0.7152]
\]

汇总：

```text
x               [1, 2]
gate_proj       [1, -2]
SiLU(gate)      [0.7311, -0.2384]
up_proj         [2, 3]
逐元素乘法       [1.4622, -0.7152]
down_proj       [0.7470, -0.7152]
```

这只是 SwiGLU 子模块输出。在完整 Decoder Layer 里，还要将这个输出加回 MLP 分支原始输入 \(x'\)，形成 Residual；本手算未模拟前面的 RMSNorm/Attention。

## 6. 为什么升维、为什么非线性和门控？

纯线性层连续叠加而不含非线性时，可合并为一次线性变换；**升维本身不自动产生非线性**。SiLU 和两条依赖输入的分支相乘，提供更丰富的输入相关特征变换。中间维度 \(I\) 使网络能够在较大的特征空间中进行加工，最后降回 \(H\) 以便残差相加。中间维度和 hidden size 的比例是架构设计选择，并非必须固定为 4 倍。

## 7. Tensor Shape 全程跟踪

教学小模型：\(B=1,T=4,H=4,I=8\)。

| 计算 | 输出 Shape |
|---|---|
| 输入 X | `[1,4,4]` |
| MLP 前 RMSNorm | `[1,4,4]` |
| gate_proj | `[1,4,8]` |
| up_proj | `[1,4,8]` |
| SiLU(gate) | `[1,4,8]` |
| elementwise × | `[1,4,8]` |
| down_proj | `[1,4,4]` |
| Residual + | `[1,4,4]` |

规律：**Token 数量 T 不变；只在 hidden/features 维度做 H → I → H。**

## 8. Qwen3-0.6B 真实尺寸

基础 Qwen3-0.6B Dense 配置：`hidden_size=1024`，`intermediate_size=3072`，`hidden_act=silu`。具体仍应以你正在使用的模型 `config.json` / GGUF 元数据为准。

### Prefill：B=1，T=100

| 操作 | Shape |
|---|---|
| 输入及 RMSNorm | `[1,100,1024]` |
| Gate / Up | 各 `[1,100,3072]` |
| SiLU(Gate) × Up | `[1,100,3072]` |
| Down | `[1,100,1024]` |
| Residual | `[1,100,1024]` |

### Decode：单步一个 Token

```text
[1,1,1024]
  ├→ Gate [1,1,3072] → SiLU ─┐
  └→ Up   [1,1,3072] ────────×
                               ↓
                         [1,1,3072]
                               ↓ Down
                         [1,1,1024]
```

**SwiGLU 不读取或追加 KV Cache**。KV Cache 属于 Attention 的跨生成步骤复用；MLP 对当前 Forward 提供的每个 Token 分别计算。

## 9. 在 llama.cpp / GGUF 中怎么看

常见张量名：

```text
blk.0.ffn_gate.weight
blk.0.ffn_up.weight
blk.0.ffn_down.weight
```

以 Qwen3-0.6B 为例：

| GGUF Tensor | 功能 | 常见 GGML `ne[]` |
|---|---|---|
| `ffn_gate.weight` | 1024 → 3072 | `[1024,3072,1,1]` |
| `ffn_up.weight` | 1024 → 3072 | `[1024,3072,1,1]` |
| `ffn_down.weight` | 3072 → 1024 | `[3072,1024,1,1]` |

**警惕维度约定：** `ggml ne[]` 与 PyTorch `nn.Linear.weight` 的形状书写顺序不同。不要仅看数值顺序判断投影方向，应结合调用代码和输入输出 Shape。

### 对模型量化的联系

三个投影都是带权重的 Linear，可涉及 NVFP4 等权重量化和矩阵乘法内核；SiLU 与逐元素乘法是激活/算子操作，本身没有对应的权重矩阵。具体是否融合、采用何种计算/累加精度，取决于框架实现和硬件。

## 10. 极简可运行的 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size=1024, intermediate_size=3072):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

mlp = Qwen3MLP()
x = torch.randn(1, 100, 1024)
y = mlp(x)
print(y.shape)  # torch.Size([1, 100, 1024])
```

这里实现的是 **MLP 子模块**；在 Decoder Layer 中，通常由外围代码执行 `x + mlp(rmsnorm(x))`。

## 11. 常见易错点速查

1. Gate 和 Up 是**两条不同的可学习线性投影**，不是把同一 Tensor 复制两遍。
2. `Gate × Up` 是**按元素乘**，不是 `@` 矩阵乘。
3. Gate 不是 0/1 掩码，SiLU 输出可为负，也可大于 1。
4. SwiGLU 的输出维度回到 `hidden_size`，以便 Residual。
5. Prefill 和 Decode 使用同一套 MLP 权重；只因输入 Token 数不同，形状由 `[B,T,H]` 变成 `[B,1,H]`。
6. Attention 负责跨 Token 交互；普通 Dense SwiGLU 只处理单个 Token 的特征，不维护 KV Cache。
7. `ffn_gate/up/down` 是权重矩阵；SiLU 和 elementwise multiply 是中间算子。

## 12. 自测题（附答案）

**Q1：输入 `[2,10,1024]`，中间层是 3072。Gate、Up、Down 的输出 Shape？**

答：Gate 和 Up 均为 `[2,10,3072]`；SiLU/逐元素乘后不变；Down 为 `[2,10,1024]`。

**Q2：Gate 输出 `[1,-2]`，Up 输出 `[2,3]`，门控结果大致是多少？**

答：SiLU(Gate) ≈ `[0.7311,-0.2384]`，相乘 ≈ `[1.4622,-0.7152]`。

**Q3：SwiGLU 是否需要读取历史 KV Cache？**

答：不需要，历史 KV Cache 由 Attention 使用；MLP 对本次输入的 hidden states 按 Token 独立计算。

**Q4：为什么有三个 Linear？**

答：Gate/Up 两个独立投影产生调节信号与候选特征，Down 投影返回 hidden size。

---

**一句话总结：** SwiGLU 用 `SiLU(gate_proj(x)) × up_proj(x)` 形成输入相关的非线性门控中间特征，再由 `down_proj` 还原到原始 hidden size；它是 Qwen3 Dense Decoder Layer 中与 Attention 并列的另一大核心计算模块。

# Qwen3 学习笔记：GQA 与 KV Cache

> **量化实证（feat-017）**：本组件「到底改了什么」的实测数字见
> `qwen3-learning/Qwen3_训练推理完整学习笔记.md` **§12 退化对照**
> —— 把 Qwen3 退化成 GPT-2 结构、只开 `use_gqa` 这一个组件，
> 与 GPT-2 的 `max_abs_diff` = **7.514e-02**（相对差 **1.01e-01**）。
> K/V head 数从 4 减到 2（GQA）后，注意力看到的 K/V 变了，相对差 1.01e-1 —— 换来的是 KV Cache 占用减半（见 §7）。

> 学习定位：已掌握 Transformer/GPT、RMSNorm、QK-Norm 和 RoPE；本章从 Attention Head 分配讲到自回归推理时的 KV Cache 生命周期、容量计算和溢出处理，并联系 Qwen3 与 llama.cpp。

## 0. 先记住五句话

1. **MHA**：每个 Q Head 对应独立的 K/V Head；**MQA**：所有 Q Heads 共用一组 K/V；**GQA**：若干 Q Heads 分为一组，共享一组 K/V。
2. GQA **保留多个 Q Heads**，减少的是 K/V Heads。共享同一组 K/V 的 Q Heads 依然分别算 Attention，结果可以不同。
3. 自回归推理通常把历史 Token 的 K/V 缓存起来，下一步只需要计算新 Token 的 Q/K/V，然后查询历史缓存，避免重复投影历史 K/V。
4. 同一次生成过程中 KV Cache 通常继续保留并增长；请求结束后可能清理、标记复用，或者保留匹配前缀，**不等于显存一定立刻归还 GPU 驱动**。
5. **上下文容量达到上限**和**GPU 显存 OOM**是两种不同的问题，必须分别处理。

---

## 1. 从 MHA 到 MQA、GQA

一个标准 Attention Head 的输出：

\[
O_h=\operatorname{Softmax}\!\left(\frac{Q_hK_h^\top}{\sqrt{d_h}}+M\right)V_h
\]

其中 \(d_h\) 是单个 Head 的特征维度，\(M\) 是 Causal Mask 等掩码。

假设有 8 个 Query Heads：

| 方案 | Q Heads | K Heads | V Heads | 分配方式 |
|---|---:|---:|---:|---|
| MHA | 8 | 8 | 8 | Q0↔K0/V0，…，Q7↔K7/V7 |
| MQA | 8 | 1 | 1 | Q0～Q7 全部共享 K0/V0 |
| GQA | 8 | 2 | 2 | Q0～Q3 共用 K0/V0；Q4～Q7 共用 K1/V1 |

在通常的均匀分组实现中，要求 Query Head 数能被 KV Head 数整除：

\[
\boxed{g=\frac{H_q}{H_{kv}}}
\]

一个 Query Head \(h\) 对应的 KV Head 索引：

\[
\boxed{h_{kv}=\left\lfloor\frac{h}{g}\right\rfloor}
\]

对于 8Q/2KV，\(g=4\)：

```text
Group 0: Q0 Q1 Q2 Q3 ──→ K0 / V0
Group 1: Q4 Q5 Q6 Q7 ──→ K1 / V1
```

**容易误解的点：**共享 K/V ≠ Q 的计算结果相同。各 Q 由不同投影输出构成，所以 Query 内容、Attention 权重、最终输出都可以不同。

## 2. GQA 中 Q/K/V 的 Tensor Shape

使用教学配置：

```text
hidden_size = 512
num_attention_heads = 8
num_key_value_heads = 2
head_dim = 64
```

一个形状为 `[B, T, 512]` 的输入，经线性投影后：

| 张量 | 线性投影输出 | 分 Head 后 |
|---|---|---|
| Q | `[B,T,512]` = 8×64 | `[B,T,8,64]` |
| K | `[B,T,128]` = 2×64 | `[B,T,2,64]` |
| V | `[B,T,128]` = 2×64 | `[B,T,2,64]` |

重要：**不要默认 GQA 的 Q/K/V 输出维度相等**；Head Dim 应读取模型配置，不能一概用 `hidden_size / num_heads` 推断。

### Qwen3-0.6B 的参数示例

以下对应经典 Qwen3-0.6B 配置（不同 Qwen3 派生型号要以各自 `config.json`/GGUF 元数据为准）：

```text
hidden_size = 1024
num_attention_heads = 16
num_key_value_heads = 8
head_dim = 128
num_hidden_layers = 28
```

因此每个 Token 的投影输出：

\[
D_Q=16\times128=2048
\]
\[
D_K=D_V=8\times128=1024
\]

这里 Q 投影输出 2048 **大于** hidden_size 1024，并不矛盾，属于模型的结构配置。

## 3. 完整手算：共享 K/V，Q 的结果依然不同

为了不被大矩阵干扰，取：2 个 Q Heads、1 个 KV Head、2 个历史 Token、`head_dim=2`，忽略 Mask（假定两条历史记录均可被当前 Query 访问）。

\[
Q_0=[1,0],\qquad Q_1=[0,1]
\]

共同使用的历史 K/V：

\[
K=\begin{bmatrix}1&0\\0&1\end{bmatrix},\qquad
V=\begin{bmatrix}10&0\\0&20\end{bmatrix}
\]

**Q0：**

\[
\frac{Q_0K^\top}{\sqrt2}=[0.707,0],\quad
A_0\approx[0.670,0.330]
\]

\[
O_0=A_0V\approx[6.70,6.60]
\]

**Q1：**

\[
\frac{Q_1K^\top}{\sqrt2}=[0,0.707],\quad
A_1\approx[0.330,0.670]
\]

\[
O_1=A_1V\approx[3.30,13.40]
\]

两组 Q **共用完全相同的历史 K/V**，输出仍然不同。最后各 Head 输出被拼接，并通过 O Projection 返回模型隐藏维度。

注意：上面 K 的“两行”代表**两个历史 Token**，不是两个 KV Heads；“序列长度”和“Head 数量”是两个不同的轴。

---

## 4. KV Cache 为什么存在？Prefill 与 Decode

自回归生成会反复执行“根据已有 Token 预测下一个 Token”。历史 Token 的 K/V 在相同层和位置设定下可以复用，没有必要每一步重新计算。

### Prefill：处理输入提示词

```text
Prompt（例如 100 个 Token）
        ↓
逐层计算 Q/K/V 和 Attention
        ↓
把历史 Token 的 K/V 写入 KV Cache
```

### Decode：每次追加新 Token

```text
当前新 Token → 计算新 Q/K/V
            → 新 Q 读取历史 K（算注意力权重）
            → 用权重聚合历史 V
            → 追加本次 K/V，供后续 Token 使用
```

**直观增长：**若 Prompt 有 100 个 Token，随着生成进行，可供后续步骤使用的有效缓存历史长度从约 100 逐步增长到 101、102……（实际写入时刻、末尾 token 和计数取决于具体调度实现）。

KV Cache **不是模型权重**，也不是把整份 Q 缓存起来：通常主要缓存每层历史位置的 **K 和 V**；当前 Q 用完即可，不需要同样长期保存。

## 5. KV Cache 的容量公式与 GQA 节省量

对标准全局 Attention，假设所有层同样的 KV 配置、每层缓存的有效 Token 数相同、K/V 都使用相同元素字节数：

\[
\boxed{M_{KV}=2\times L\times B\times T\times H_{kv}\times D\times S}
\]

- 2：K 与 V 两份；
- \(L\)：层数；\(B\)：并发序列/批大小；\(T\)：每条序列缓存长度；
- \(H_{kv}\)：KV Heads 数；\(D\)：单 Head 维度；\(S\)：每元素字节数（FP16/BF16 通常为 2）。

若 K 和 V 的维度/存储精度不相同，应分别算后相加，而不是直接套上面的 `2×`。

**Qwen3-0.6B 示例：**\(L=28,B=1,T=4096,H_{kv}=8,D=128,S=2\)：

\[
M_{GQA}=2\times28\times4096\times8\times128\times2
=469,762,048\ \mathrm{Bytes}=448\ \mathrm{MiB}
\]

保持其他参数一致，假设改成 16 个 KV Heads 的 MHA：

\[
M_{MHA}=896\ \mathrm{MiB}
\]

假设改成 1 个 KV Head 的 MQA：

\[
M_{MQA}=56\ \mathrm{MiB}
\]

对应 GQA 相对 MHA 减少：

\[
1-\frac{8}{16}=50\%
\]

而前面的教学例子 **8Q/2KV** 相对于 **8Q/8KV**：

\[
1-\frac{2}{8}=75\%
\]

**区分：8Q/2KV 的 75% 是教学例子；真实 Qwen3-0.6B 16Q/8KV 是 50%。** 上面仅计算 K/V 数据容量，不含分配器、对齐、额外缓冲区等开销。

### 为什么缓存减小能改善速度？

长上下文 Decode 往往需要反复读取历史 K/V。GQA 减少缓存尺寸，也有机会减少读取的数据量，降低带宽压力。**KV Cache 减少 75% 不代表推理整体提速 4 倍**：模型权重读取、MLP、Q/K/V 投影、Attention 算术、调度和 Kernel 开销仍在；具体收益取决于是否 memory-bound、Kernel 是否有效共享 K/V 以及并发情况。

---

## 6. KV Cache 的生命周期：什么时候保留？什么时候清理？

必须区分“一个 Decode Token 结束”和“一个完整 Request 结束”。

| 时刻 | 常见行为 |
|---|---|
| Prefill 结束 | 输入历史的 K/V 已建立 |
| 每生成一个 Token 后 | 缓存继续保留/追加，以供下一步使用 |
| 当前请求生成完 | 可以标记空闲、覆盖复用、清理，或保留前缀缓存 |
| 下一轮多轮对话 | 若提交的 token 前缀与保留缓存匹配且框架支持，可以复用相应 K/V |
| 服务进程退出 | 该进程持有的 GPU 分配通常释放 |

**两个重要区分：**

1. 对话窗口里“看得见历史消息”，不等于服务端一定保存了旧 K/V；应用也可能将历史文本重新送去 Prefill。
2. KV Cache 失效或 Slot 空闲，不等于显存占用立即减少：框架可能保留预分配 Buffer 给下一次请求使用。

### 多请求与前缀共享

请求 A 和请求 B 的私有生成历史通常不能胡乱共用。但如果框架支持 Prefix Cache，两个请求具有**完全匹配的公共 Token 前缀**且缓存状态可复用，就有机会共享或重新利用公共前缀对应的计算结果。具体依赖缓存键、位置、模型配置、Slot 管理等。

---

## 7. “KV Cache 溢出”首先要分成两类

| 类型 | 典型表现 | 含义 |
|---|---|---|
| Context overflow | Prompt + 所需生成超出 `n_ctx` / 可用上下文槽位 | 逻辑容量达到上限，哪怕 GPU 仍有空闲显存 |
| GPU OOM | 模型权重、缓存及工作区等总量无法装入显存 | 物理内存资源不足，哪怕 Token 数尚未到上下文上限 |

例如 `n_ctx=4096`，Prompt 有 3000 Token，希望再生成 2000 Token：总潜在长度 5000 超过 4096，这是**上下文预算问题**。相反 `n_ctx=32768` 但缓存分配/运行工作区导致 GPU 显存不足，是 OOM。

还应区分 **逻辑有效缓存长度** 与 **物理 Buffer 的预分配容量**：有的实现一开始就分配最大 KV 空间，Decode 只填充预留位置，显存不会随每个 Token 逐字节上涨。

## 8. Prefill 阶段超限：怎么办？

**A. 直接拒绝**：不丢失输入含义，返回上下文超限错误。

**B. 截断/摘要 Prompt**：只保留指定片段或进行应用层总结；代价是原始信息不再可完整访问。

**C. 调整上下文容量**：例如 llama.cpp 中使用 `-c 8192`，前提是模型支持相应长度且设备内存允许。设置更大的缓存容量通常需要更多空间；若超过原生训练上下文，还须关注模型长上下文策略和效果。

**D. Chunked Prefill**：分块处理长 Prompt，减少单次计算的峰值临时开销、改善请求调度；**它本身不降低最终需要存储的完整 K/V 总量，也不能单独突破有效上下文容量**。

## 9. Decode 阶段达到上限：怎么办？

假设 `n_ctx=4096`，输入已用 3000 个 Token，则剩余空间约 1096 个 Token（实际还受特殊 Token、输出上限及框架计数影响）。

**方案 1：停止生成**：可能返回 `finish_reason="length"` 一类状态；它也可能仅表示触及 `max_tokens`，不能单凭它判断 GPU OOM。

**方案 2：上下文移位（Context Shift）**：移除部分更早历史、保留部分前缀与最近内容，维护可继续使用的上下文。历史内容被移除后，模型不再能像完整上下文那样直接访问它。具体能否安全执行、位置/KV 如何调整由模型和框架决定。

**方案 3：原生 Sliding Window Attention**：若模型架构本身规定每步只看窗口内的历史，那么可只保留窗口相关 K/V。**这与随意把全局 Attention 模型的旧 K/V 删掉并不等价**；不能把它当成无代价的无限上下文。

## 10. GPU 真正 OOM：如何处理？

可以考虑：

- **KV Cache 量化**：FP16/BF16 的元素通常 2 Bytes，Q8 原始数据约 1 Byte/元素，Q4 约 0.5 Byte/元素；实际量化格式还有 scale、block 对齐等额外开销，压缩率及精度变化应实测。
- **限制并发/单请求上下文或生成长度**：多个请求的私有缓存可能叠加占用资源。
- **CPU/GPU KV 放置或卸载**：若推理框架支持，可用更低显存占用换取潜在传输开销与延迟。
- **调度与抢占**：在支持的服务引擎中，可暂停/驱逐某请求，再重计算被驱逐的缓存等，以缓解多请求争用；它不能突破模型固有上下文长度。
- **优化推理工作区、模型权重存储或批大小**：OOM 不一定只来自 KV Cache。

注意：**模型权重是 NVFP4，不代表 KV Cache 也自动是 NVFP4**。两者是独立的张量与精度配置，需要检查推理框架、硬件与对应 Kernel 是否支持。

### llama.cpp 中值得识别的选项（以当前公司分支 `--help` 为准）

```bash
# 指定上下文容量（示例）
./build/bin/llama-server -m model.gguf -ngl 999 -c 8192

# 某些支持相关功能的构建提供 KV 类型参数，例如：
-ctk q8_0
-ctv q8_0

# 可能涉及 Slot、Prompt Cache 与上下文移位的参数：
-np
--cache-prompt
--context-shift
```

这些参数的可用性、默认值、适用模型以及与 Flash Attention 等功能的组合约束，**必须以实际 `llama-server --help`、日志和对应源码分支为准**。不要假设每个 fork 都支持完全相同的行为。

---

## 11. llama.cpp / GGUF 调试对照

| 常见字段/权重 | 看它时应想到什么 |
|---|---|
| `n_head` | Q Heads 数量 |
| `n_head_kv` | KV Heads 数量 |
| `n_embd_head_k` | K Head 特征维度 |
| `n_embd_head_v` | V Head 特征维度 |
| `attn_q.weight` | 从 hidden state 生成 Q 的投影 |
| `attn_k.weight` | 生成 K 的投影 |
| `attn_v.weight` | 生成 V 的投影 |
| `attn_q_norm.weight` | 各 Q Head 使用的 Q-Norm 权重 |
| `attn_k_norm.weight` | 各 KV Head 使用的 K-Norm 权重 |
| `n_ctx` | 运行时上下文容量（要留意 Slot/并发分配方式） |
| KV Buffer / Cell / Slot | 物理缓存分配、有效条目与请求使用状态相关的内部管理概念 |

把已学的组件串起来：

```text
hidden_states
     │
     └→ RMSNorm
          │
          ├→ Q Projection → 分 Q Heads → Q-Norm → RoPE ──┐
          ├→ K Projection → 分 KV Heads → K-Norm → RoPE ─┤
          └→ V Projection → 分 KV Heads ──────────────────┤
                                                         ↓
                                                  GQA Attention
                                                         │
                                                 各 Q Head 输出
                                                         │
                                                  拼接 + O Projection
                                                         │
                                                      Residual
```

KV Cache 主要保存其中历史 Token、各层的 **K/V**；每次 Decode，新 Q 查询与其分组对应的历史 KV Head。

## 12. 一张排障决策表

```text
遇到“KV Cache 满了”
        │
        ├─ Token 已达 n_ctx / Slot 逻辑容量？
        │      ├─ Prefill：拒绝、摘要/截断、扩展合法上下文；
        │      │          分块 Prefill 只能处理峰值/调度，不能凭空省最终 KV
        │      └─ Decode：停止、合法的 context shift / 模型原生窗口
        │
        └─ GPU 报 OOM？
               ├─ 检查模型权重、KV Buffer、工作区及并发
               ├─ 调整 KV 精度、上下文分配、并发/批大小
               └─ 若支持，使用资源调度或缓存卸载
```

## 13. 自测题

1. 8Q/2KV 的 `group_size` 是多少？Q6 对应哪个 KV Head？
2. Q0 与 Q1 共用 K/V，为什么注意力输出仍然可以不同？
3. 一个模型有 28 层、8 KV Heads、Head Dim 128，批大小 1，缓存 4096 Token，FP16 K/V。纯 KV 数据占多少 MiB？
4. 一次请求完成后 KV 失效，为什么 `nvidia-smi` 里的进程显存可能没有立即下降？
5. Chunked Prefill 为什么不能单独解决“完整历史 KV 容量不足”？
6. `finish_reason="length"` 能否直接认定为 GPU OOM？为什么？
7. 权重量化为 NVFP4，是否足以判断 KV Cache 的实际精度？

**答案速查：**1. `g=4`，Q6→KV1；2. Q 不同、各自点积和 Softmax 不同；3. 448 MiB；4. 预分配/内存池/Slot 复用；5. 仍需保存最终完整历史；6. 不能，还可能触及 `max_tokens`/上下文长度；7. 不能，须单独检查 KV 类型配置。

---

## 14. 本章终极总结

**GQA 解决“历史 K/V 应该有多少份”**：多个 Query Head 分组共享更少的 KV Heads，缓解长上下文 Decode 时的缓存容量和读取带宽压力。

**KV Cache 解决“历史 K/V 是否重复计算”**：Prefill 建立缓存，Decode 增量使用和追加；请求结束后由服务端按策略复用或回收。

**溢出处理先辨别逻辑容量还是物理内存**：上下文超长考虑截断、合法扩展、停止或移位；显存不足考虑 KV 精度、并发、内存布局和调度。两类问题不是同一回事。

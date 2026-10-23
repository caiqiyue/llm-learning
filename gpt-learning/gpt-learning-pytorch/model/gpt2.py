"""
GPT-2 模型定义（PyTorch 端）。

文件结构：
  - CausalSelfAttention  : 因果多头自注意力（支持可选的 KV cache）
  - MLP                  : 前馈网络（Linear → GELU → Linear → Dropout）
  - Block                : Pre-LN Transformer Block（attn + mlp 两个子层）
  - GPT2                 : 主模型（tok_emb + pos_emb + N × Block + ln_f + lm_head）

详细 spec 见 design §3.4（架构）、gpt2_training_explained.md Part 1（讲解）、
以及 implementation-plan.md 的 P2 任务。

代码风格约定：
  - 详细中文 docstring（每个类、每个方法）
  - 完整 type hint
  - 顶层不写 `if __name__`
  - 延迟 import torch：避免 import 失败炸掉模块加载
  - 确定性：eval 模式下（dropout=0）相同输入应 bit-identical
"""

# 延迟 import torch：把 import 错误的影响范围缩小
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 工具函数：构造 sinusoidal 位置编码
# ---------------------------------------------------------------------------
def _build_sinusoidal_pos_emb(block_size: int, n_embd: int) -> "torch.Tensor":
    """
    构造固定的 Sinusoidal 位置编码（与原始 Transformer 2017 一致）。

    公式：
        PE(pos, 2i)   = sin(pos / 10000^(2i/n_embd))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/n_embd))

    参数：
        block_size: 最大序列长度（T）
        n_embd:     隐藏维度（C）
    返回：
        (block_size, n_embd) 的浮点张量
    """
    pe = torch.zeros(block_size, n_embd)
    # 位置索引 (block_size, 1)
    position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
    # 频率因子 exp(-log(10000) * 2i / n_embd)
    div_term = torch.exp(
        torch.arange(0, n_embd, 2).float() * -(math.log(10000.0) / n_embd)
    )
    # 偶数维用 sin，奇数维用 cos
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# ---------------------------------------------------------------------------
# 1. 因果多头自注意力
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """
    因果多头自注意力（Causal Multi-Head Self-Attention）。

    流程：
      1) 用一个 `nn.Linear(n_embd → 3 * n_embd, bias)` 同时投出 Q、K、V
         （等价于 3 个独立 Linear，但更高效）
      2) 把最后一维拆成 n_head 个头：(B, T, n_embd) → (B, n_head, T, head_dim)
      3) 缩放点积注意力 `att = softmax(Q @ K^T / sqrt(head_dim) + mask) @ V`
      4) 拼回 (B, T, n_embd) 并做输出投影

    可选 KV cache（推理加速）：
      - `kv_cache` 不为 None 时，把新算出的 K、V 追加到 cache，并从 cache 取全长 K、V 算 attention
      - 此时 Q 形状是 (B, n_head, T_q, head_dim)，K/V 是 (B, n_head, T_kv, head_dim)
        当 T_q == T_kv（prefill 阶段），用因果 mask
        当 T_q == 1 && T_kv > 1（decode 阶段），**不用 mask**（cache 已经天然只含历史 token）
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float,
        bias: bool,
    ) -> None:
        """
        参数：
            n_embd:     隐藏维度
            n_head:     多头注意力的头数（必须能整除 n_embd）
            block_size: 最大上下文长度（用于注册 causal mask）
            dropout:    attention 之后的 dropout 比例
            bias:       Q/K/V 投影与输出投影是否使用 bias
        """
        super().__init__()
        assert n_embd % n_head == 0, "n_embd 必须能整除 n_head"

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.block_size = block_size

        # Q/K/V 一次投出（高效做法；等价于 3 个独立 Linear）
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # 输出投影
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # 因果 mask：上三角（不含对角线）为 True，表示"被 mask 掉"
        # 形状 (block_size, block_size)；用 register_buffer 让它跟随模型设备移动
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def forward(
        self,
        x: "torch.Tensor",
        kv_cache: Optional["object"] = None,
        layer_idx: int = 0,
        position_ids: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", Optional["object"]]:
        """
        因果多头自注意力 forward。

        【形状速查表】（B=batch, T=time, H=head, D=head_dim, C=n_embd）
        ┌────────────────────────────────────────────────────────────┐
        │ 步骤                │ 输入形状              │ 输出形状         │
        ├────────────────────────────────────────────────────────────┤
        │ 输入 x              │ -                     │ (B, T_q, C)      │
        │ 1. c_attn (QKV投影) │ (B, T_q, C)           │ (B, T_q, 3C)     │
        │ 2. split            │ (B, T_q, 3C)          │ q,k,v: (B,T_q,C) │
        │ 3. view+transpose   │ (B, T_q, C)           │ (B, H, T_q, D)   │
        │    (D = C // H)     │                       │                  │
        │ 4. KV cache 取全长  │ (B, H, T_q, D)        │ (B, H, T_kv, D)  │
        │    (训练时 T_kv=T_q)│                       │                  │
        │ 5. Q @ K^T / √D    │ (B,H,T_q,D)×(B,H,D,T_kv)│ (B,H,T_q,T_kv)  │
        │ 6. causal mask      │ (B, H, T_q, T_kv)     │ (B, H, T_q, T_kv)│
        │ 7. softmax          │ (B, H, T_q, T_kv)     │ (B, H, T_q, T_kv)│
        │ 8. attn @ V         │ (B,H,T_q,T_kv)×(B,H,T_kv,D)│ (B,H,T_q,D)  │
        │ 9. transpose+view   │ (B, H, T_q, D)        │ (B, T_q, C)      │
        │ 10. c_proj 输出投影 │ (B, T_q, C)           │ (B, T_q, C)      │
        └────────────────────────────────────────────────────────────┘

        【关键点】
          - Q 的 T_q 与 K/V 的 T_kv 在 decode 阶段可能不同：
              · 训练 / prefill: T_q = T_kv = T（一次性算整段）
              · decode:         T_q = 1, T_kv = 已 cache 的全部历史
          - 因果 mask 仅在 T_q == T_kv 时使用（decode 阶段 cache 天然只含历史，无需 mask）

        参数：
            x:           (B, T_q, n_embd)  当前步的输入 hidden states
            kv_cache:    KVCache 实例（推理用）或 None（训练用）
            layer_idx:   当前层索引（写入 KVCache 时需要）
            position_ids:(B, T_q) 绝对位置 id；None 时用默认 0..T_q-1
                              （与 sinusoidal pos emb 对齐）
        返回：
            (output, kv_cache)：
                output:    (B, T_q, n_embd)
                kv_cache:  原样返回（None 或更新后的 KVCache）
        """
        B, T_q, C = x.size()#32,128,128 / 1,22,128 / 1,1,128
        # 当前 self 内部常用维度，方便后面引用
        H = self.n_head          # 多头数（= 4）    4个头 ，每个头 32维度   4 x 32 = 128
        D = self.head_dim        # head_dim = n_embd // n_head（= 32）

        # 1) 一次性投出 Q、K、V（高效做法；等价于 3 个独立 Linear）
        #    输入 x:    (B, T_q, C)
        #    权重 W:    (C, 3C) bias 可选
        #    输出 qkv:  (B, T_q, 3C)
        qkv = self.c_attn(x)  #qkv:32,128,384  128x3=384 /  1,1,384
        # 1.5) 沿最后一维切成 3 段
        #    q, k, v:  (B, T_q, C) 各
        q, k, v = qkv.split(self.n_embd, dim=2)#q,k,v 32,128,128    1,22,384

        # 2) 拆多头：(B, T, C) → (B, T, H, D) → 转置 → (B, H, T, D)
        #    view: 把最后一维 C 拆成 (H, D) 两个维度
        #    transpose(1, 2): 把 T 和 H 维度互换，让 head 维在 batch 后
        #    目的: 后续 matmul 在 (H, T_q, D) × (H, D, T_kv) 上做 batch matmul
        q = q.view(B, T_q, H, D).transpose(1, 2)   # (B, H, T_q, D)  32,4,128,32    1,4,22,32  1,1,128 --> 1,4,1,32
        k = k.view(B, T_q, H, D).transpose(1, 2)   # (B, H, T_q, D)  32,4,128,32
        v = v.view(B, T_q, H, D).transpose(1, 2)   # (B, H, T_q, D)  32,4,128,32
        # 形状一致：q, k, v 都是 (B, H, T_q, D)

        # 3) 处理 KV cache（推理时才有，训练时为 None）
        if kv_cache is not None:
            # 3.1) 把当前步新算出的 K、V 追加到 cache 的 layer_idx 位置
            #      cache 内部维护每层独立的 seq_len；append 后该层 seq_len += T_q
            kv_cache.append(layer_idx, k, v) #k and v : 1,4,22,32  decode 阶段加入新token的kv后，1,4,23,32    kv-cache 是 4,1,4,128,32
            # 3.2) 取该层累计的全部 K、V（可能是多步 decode 累积的）
            #      k_full, v_full: (B, H, T_kv, D)，T_kv = 该层 cache 当前长度
            k_full, v_full = kv_cache.get(layer_idx) # 1,4,23,32 
        else:
            # 训练模式：T_kv = T_q，不需要 cache
            k_full, v_full = k, v

        T_kv = k_full.size(2)    # 取 K 的时间维长度  k 和 v 的 字数 22，decode阶段就逐步增长，23

        # 4) 缩放点积注意力：att[b, h, i, j] = "位置 i 看位置 j 的相似度"
        #    q:           (B, H, T_q, D)
        #    k_full^T:    (B, H, D, T_kv)   ← k_full.transpose(-2, -1) 把最后两维转置
        #    q @ k^T:     (B, H, T_q, T_kv) ← batch matmul
        #    * 1/√D:      缩放（避免点积方差过大导致 softmax 饱和）  q:1,4,22,32 k_full:1,4,22,32  --> 1,4,22,22
        att = (q @ k_full.transpose(-2, -1)) * (1.0 / math.sqrt(D))   # (B, H, T_q, T_kv)  32,4,128,32  32,4,32,128  ---> 32,4,128,128  
        #decode阶段，q：1,4,1,32 和 k 1,4,23,32  --》 1,4,1,23 新token 的 q 和 过去及自己的23个token进行注意力计算
        # 5) 因果 mask：训练或 prefill 时 T_q == T_kv，需要把"未来位置"屏蔽
        #    self.mask: (block_size, block_size) 的 bool 上三角矩阵（True=被 mask）
        #    masked_fill: 把 True 的位置填 -inf，softmax 后这些位置概率 = 0
        if T_q == T_kv:  #表明是训练或者prefill阶段，decode 阶段就无需遮住
            # 训练或 prefill：标准上三角 mask（位置 i 看不到 j > i）
            att = att.masked_fill(self.mask[:T_q, :T_kv] == True, float("-inf")) #self.mask[:T_q, :T_kv]  128,128 是一个上三角矩阵
        # else: decode 阶段 T_q == 1, T_kv > 1
        #       cache 天然只包含历史 token，无需 mask（K 已经是从前到后的）

        # 6) 注意力权重：每行（位置 i）归一化到 1
        att = F.softmax(att, dim=-1)              # (B, H, T_q, T_kv)  32,4,128,128 
        att = self.attn_dropout(att)              # 训练时随机丢一些注意力（防过拟合）

        # 7) 加权求和：把 V 按注意力权重聚合
        #    att:       (B, H, T_q, T_kv)
        #    v_full:    (B, H, T_kv, D)
        #    att @ v:   (B, H, T_q, D)  ← 每个 query 位置对所有 key 位置做加权 V  1,4,22,22  1,4,22,32  --> 1,4,22,32
        y = att @ v_full  #32,4,128,128  32,4,128,32  ---> 32，4,128,32
        # 1,4,1,23 和 1,4，23,32 ---> 1,4,1,32 ---> 1,1,128 1个样本，1个token，128维
        # 8) 拼多头：把 H 个头拼回 n_embd 维
        #    (B, H, T_q, D) → transpose → (B, T_q, H, D) → contiguous+view → (B, T_q, C)
        #    contiguous: 让内存连续（transpose 后内存不连续，view 之前必须先 contiguous）
        y = y.transpose(1, 2).contiguous().view(B, T_q, C)   # (B, T_q, C)  32,128,128    1,22,128

        # 9) 输出投影 c_proj（C → C）+ 残差 dropout
        y = self.resid_dropout(self.c_proj(y))    # (B, T_q, C)  32,128,128   y:1,22,128

        return y, kv_cache


# ---------------------------------------------------------------------------
# 2. 前馈网络（MLP / FFN）
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """
    Transformer Block 中的前馈网络。

    公式：`y = Dropout(W2 · GELU(W1 · x))`
    形状：`(B, T, n_embd) → (B, T, n_inner) → (B, T, n_embd)`

    中间层维度 `n_inner` 通常是 `n_embd` 的 4 倍（设计选择）。
    """

    def __init__(self, n_embd: int, n_inner: int, dropout: float, bias: bool) -> None:
        """
        参数：
            n_embd:  输入/输出维度
            n_inner: 中间层维度
            dropout: 输出 dropout
            bias:    Linear 层是否使用 bias
        """
        super().__init__()
        self.c_fc = nn.Linear(n_embd, n_inner, bias=bias)
        self.c_proj = nn.Linear(n_inner, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        MLP forward（位置独立的非线性变换）。

        【形状速查表】
        ┌─────────────────────────────────────────────┐
        │ 步骤           │ 输入形状        │ 输出形状       │
        ├─────────────────────────────────────────────┤
        │ 输入 x         │ -               │ (B, T, C)      │
        │ 1. c_fc 升维   │ (B, T, C)       │ (B, T, n_inner)│
        │ 2. GELU        │ (B, T, n_inner) │ (B, T, n_inner)│
        │ 3. c_proj 降维 │ (B, T, n_inner) │ (B, T, C)      │
        │ 4. dropout     │ (B, T, C)       │ (B, T, C)      │
        └─────────────────────────────────────────────┘
        本项目 n_inner = 4 × n_embd = 512

        参数：
            x: (B, T, n_embd)
        返回：
            (B, T, n_embd)
        """
        # 1) 升维：每个位置独立做 C → n_inner 维的线性变换
        #    x: (B, T, C) @ W^T (C, n_inner) → (B, T, n_inner)
        x = self.c_fc(x)
        # 2) GELU 激活（高斯误差线性单元；平滑非线性，比 ReLU 在 NLP 上更好）
        #    形状不变：(B, T, n_inner)
        x = F.gelu(x)
        # 3) 降维：n_inner → C
        #    x: (B, T, n_inner) @ W^T (n_inner, C) → (B, T, C)
        x = self.c_proj(x)
        # 4) dropout（训练时随机丢一些神经元）
        x = self.dropout(x)
        return x   # (B, T, C)


# ---------------------------------------------------------------------------
# 3. Transformer Block（Pre-LN）
# ---------------------------------------------------------------------------
class Block(nn.Module):
    """
    一个 Pre-LN Transformer Block：

        x = x + Attn(LN1(x), kv_cache, layer_idx, position_ids)
        x = x + MLP(LN2(x))

    Pre-LN：LayerNorm 在子层**之前**（残差路径不经过 LN），训练更稳定。

    支持 KV cache：把 `kv_cache` 和 `position_ids` 透传给 attention 子层。
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_inner: int,
        dropout: float,
        bias: bool,
        block_size: int,
    ) -> None:
        """
        参数：
            n_embd:     隐藏维度
            n_head:     注意力头数
            n_inner:    FFN 中间层维度
            dropout:    dropout 比例
            bias:       Linear 是否使用 bias
            block_size: 最大上下文长度（注册 mask 用）
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout, bias)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, n_inner, dropout, bias)

    def forward(
        self,
        x: "torch.Tensor",
        kv_cache: Optional["object"] = None,
        layer_idx: int = 0,
        position_ids: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", Optional["object"]]:
        """
        Pre-LN Transformer Block forward。

        【形状速查表】所有步骤输入输出都是 (B, T, C)
        ┌───────────────────────────────────────────────────────┐
        │ 步骤                  │ 形状变化                │
        ├───────────────────────────────────────────────────────┤
        │ 输入 x                │ (B, T, C)              │
        │ ln_1 (LayerNorm)      │ (B, T, C)              │
        │ attn(LN_1(x))         │ (B, T, C)  ← 残差前     │
        │ x + attn_out          │ (B, T, C)  ← 第一个残差  │
        │ ln_2 (LayerNorm)      │ (B, T, C)              │
        │ mlp(LN_2(x))          │ (B, T, C)  ← 残差前     │
        │ x + mlp_out           │ (B, T, C)  ← 第二个残差  │
        └───────────────────────────────────────────────────────┘

        【Pre-LN vs Post-LN】
          Pre-LN (本项目):  x = x + Sublayer(LN(x))
          Post-LN:          x = LN(x + Sublayer(x))
          Pre-LN 训练更稳定（残差路径上无 LN 缩放，梯度能直通到输入）

        参数：
            x:           (B, T, n_embd)
            kv_cache:    KVCache 或 None
            layer_idx:   当前 block 索引
            position_ids:(B, T) 绝对位置 id（推理用）
        返回：
            (x, kv_cache)
        """
        # === 第一个子层：因果自注意力（带 Pre-LN）===
        # 1) Pre-LN：先 LN 再 attention
        #    ln_1(x): (B, T, C) → (B, T, C)
        # 2) Attention：内部走 QKV 拆分、attn @ V、c_proj
        #    attn_out: (B, T, C)
        attn_out, kv_cache = self.attn(
            self.ln_1(x), kv_cache=kv_cache, layer_idx=layer_idx, position_ids=position_ids
        )
        # 3) 第一个残差连接：x = x + attn(LN(x))  decode 阶段 1,1,128
        x = x + attn_out   # (B, T, C)  1,22,128

        # === 第二个子层：FFN/MLP（带 Pre-LN）===
        # 4) Pre-LN + MLP
        #    ln_2(x): (B, T, C)
        #    mlp(...): (B, T, C)  ← 升维 GELU 降维
        # 5) 第二个残差连接
        x = x + self.mlp(self.ln_2(x))   # (B, T, C)  1,22,128

        return x, kv_cache #x:32,128,128


# ---------------------------------------------------------------------------
# 4. GPT-2 主模型
# ---------------------------------------------------------------------------
class GPT2(nn.Module):
    """
    Tiny GPT-2 主模型。

    组件：
      - `tok_emb`:  token embedding (vocab_size, n_embd)
      - `pos_emb`:  固定 sinusoidal 位置编码（注册为 buffer，不参与训练）
      - `drop`:     embedding 之后的 dropout
      - `blocks`:   n_layer 个 Pre-LN Transformer Block
      - `ln_f`:     Final LayerNorm
      - `head`:     输出投影 Linear(n_embd, vocab_size, bias=False)，**与 tok_emb 共享权重**
      - `value_head`: 可选（None by default），PPO 用：Linear(n_embd, 1)

    forward 输入：
        idx: (B, T) LongTensor 的 token id
        targets: (B, T) LongTensor，与 idx **同下标**（训练时给，-100 表示忽略）；
                  错位（"logits[t] 预测 token t+1"）在 forward 内部完成，见下方说明
        kv_cache: KVCache 或 None（推理加速用）
        position_ids: (B, T) LongTensor，**绝对位置**（与 sinusoidal pos_emb 对齐）
                       若 kv_cache 不为 None，必须显式提供
    forward 返回：
        (logits, loss, new_kv_cache, value)：
            logits: (B, T, vocab_size)
            loss:   scalar（targets=None 时为 None）
            new_kv_cache: 同上
            value:  (B,) scalar value per sample（当 value_head 不为 None 时存在）
    """

    def __init__(
        self,
        vocab_size: int,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 128,
        n_inner: int = 512,
        block_size: int = 128,
        dropout: float = 0.1,
        bias: bool = False,
    ) -> None:
        """
        参数：
            vocab_size: 词表大小
            n_layer:    Transformer Block 层数
            n_head:     多头头数
            n_embd:     隐藏维度
            n_inner:    FFN 中间层维度
            block_size: 最大上下文长度
            dropout:    dropout 比例
            bias:       是否使用 bias（与 nanoGPT 对齐 → False）
        """
        super().__init__()
        # 把超参数也存为属性，方便外部（value_head、inference）访问
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.n_inner = n_inner
        self.block_size = block_size
        self.dropout_p = dropout
        self.bias_flag = bias

        # --- token embedding ---
        self.tok_emb = nn.Embedding(vocab_size, n_embd)

        # --- 固定 sinusoidal 位置编码（注册为 buffer）---
        # 形状 (block_size, n_embd)；forward 中按 position_ids 切片
        pe = _build_sinusoidal_pos_emb(block_size, n_embd)
        self.register_buffer("pos_emb", pe, persistent=False)

        # --- embedding 之后的 dropout ---
        self.drop = nn.Dropout(dropout)

        # --- N 个 Transformer Block ---
        self.blocks = nn.ModuleList(
            [
                Block(
                    n_embd=n_embd,
                    n_head=n_head,
                    n_inner=n_inner,
                    dropout=dropout,
                    bias=bias,
                    block_size=block_size,
                )
                for _ in range(n_layer)
            ]
        )

        # --- Final LayerNorm ---
        self.ln_f = nn.LayerNorm(n_embd)

        # --- 输出投影（bias=False，与 nanoGPT 对齐）---
        # 关键：与 tok_emb 共享权重（weight tying）
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        # 直接把 head 的 weight 替换为 tok_emb 的 weight
        # 这样优化器只更新一份参数；token embedding 矩阵同时充当输出投影
        self.head.weight = self.tok_emb.weight

        # --- PPO 用 value head（默认 None；由 train_ppo 阶段赋值）---
        self.value_head: Optional[nn.Linear] = None

        # 初始化权重（N(0, 0.02)，与 GPT-2 原文一致）
        self.apply(self._init_weights)
        # 残差投影层（c_proj）按 1/sqrt(2*n_layer) 缩小，缓解深层网络梯度爆炸
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    # ------------------------------------------------------------------
    # 权重初始化
    # ------------------------------------------------------------------
    def _init_weights(self, module: nn.Module) -> None:
        """
        GPT-2 风格初始化：所有 Linear/Embedding 的 weight 用 N(0, 0.02)，bias 用 0。
        LayerNorm 的 bias 设为 0、weight 设为 1。
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # 工具：从最后一层取 hidden state（给 reward/value head 用）
    # ------------------------------------------------------------------
    def forward_hidden(
        self,
        idx: "torch.Tensor",
        position_ids: Optional["torch.Tensor"] = None,
        kv_cache: Optional["object"] = None,
    ) -> Tuple["torch.Tensor", Optional["object"]]:
        """
        与 `forward` 几乎一致，但只返回最后一层 hidden state（不投 logits）。
        给 value head 用，避免重复算 lm_head。

        【形状速查表】
        ┌─────────────────────────────────────────────┐
        │ 步骤              │ 输入形状   │ 输出形状      │
        ├─────────────────────────────────────────────┤
        │ idx (token ids)   │ -          │ (B, T)       │
        │ tok_emb           │ (B, T)     │ (B, T, C)    │
        │ pos_emb[position_ids]│ (B, T) │ (B, T, C)    │
        │ tok + pos         │ (B, T, C)  │ (B, T, C)    │
        │ N × Block         │ (B, T, C)  │ (B, T, C)    │
        │ hidden (返回)     │ -          │ (B, T, C)    │
        │  ← 不走 ln_f       │            │              │
        └─────────────────────────────────────────────┘

        参数：
            idx: (B, T) LongTensor
            position_ids: (B, T) 绝对位置
            kv_cache: KVCache 或 None
        返回：
            (hidden, new_kv_cache)：
                hidden: (B, T, n_embd)  最后一层（ln_f 之前）的 hidden
                new_kv_cache: 同上
        """
        B, T = idx.size()

        if position_ids is None:
            # 默认用相对位置 0..T-1（训练 / 评估时）
            #   shape: (T,) → (B, T)  [broadcast 到所有 batch]
            position_ids = torch.arange(0, T, dtype=torch.long, device=idx.device)
            position_ids = position_ids.unsqueeze(0).expand(B, T)

        # === 1) Token embedding ===
        #    idx: (B, T) 的整数 token id
        #    tok_emb.weight: (V, C) 查表矩阵
        #    tok: (B, T) → (B, T, C)
        tok = self.tok_emb(idx)
        # === 2) Position embedding（用 position_ids 切片 sinusoidal 表）===
        #    self.pos_emb: (block_size, C) 的固定正弦/余弦表（buffer，不训练）
        #    position_ids: (B, T) 的绝对位置下标
        #    pos_emb[position_ids]:  按 position_ids 在第一维做 gather
        #    pos: (B, T, C)
        pos = self.pos_emb[position_ids]
        # === 3) 相加 + dropout ===
        #    tok + pos: 逐元素相加，形状不变 (B, T, C)
        x = self.drop(tok + pos)

        # === 4) 走 N 个 Transformer Block（每个 Block 内部 Pre-LN + 残差）===
        #    形状保持 (B, T, C)
        new_kv = kv_cache
        for layer_idx, block in enumerate(self.blocks):
            x, new_kv = block(x, kv_cache=new_kv, layer_idx=layer_idx, position_ids=position_ids)

        # === 5) 不走 ln_f，直接返回最后一层 hidden ===
        #    这里与 forward() 的差别：不经过 ln_f 也不投 logits
        #    value head 想用"Pre-LN 后的特征"（学界常用做法）
        return x, new_kv   # x: (B, T, C)

    # ------------------------------------------------------------------
    # 主 forward
    # ------------------------------------------------------------------
    def forward(
        self,
        idx: "torch.Tensor",
        targets: Optional["torch.Tensor"] = None,
        kv_cache: Optional["object"] = None,
        position_ids: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"], Optional["object"], Optional["torch.Tensor"]]:
        """
        GPT-2 主前向计算。

        【形状速查表】（V = vocab_size, C = n_embd, B = batch, T = time）
        ┌────────────────────────────────────────────────────────────┐
        │ 步骤                      │ 输入形状        │ 输出形状         │
        ├────────────────────────────────────────────────────────────┤
        │ 0. idx (token ids)        │ -               │ (B, T)          │
        │ 1. tok_emb                │ (B, T)          │ (B, T, C)       │
        │    pos_emb[position_ids]  │ (B, T)          │ (B, T, C)       │
        │    drop(tok + pos)        │ (B, T, C)       │ (B, T, C)       │
        │ 2. N × Block              │ (B, T, C)       │ (B, T, C)       │
        │ 3. ln_f (final LayerNorm) │ (B, T, C)       │ (B, T, C)       │
        │ 4. value_head[可选]        │ (B, C)         │ (B,) scalar     │
        │    ← 取最后 token hidden   │ (B, T, C)[:, -1, :]                  │
        │ 5. head (lm_head)         │ (B, T, C)       │ (B, T, V)       │
        │    ← weight tied with tok_emb                                │
        │ 6. cross_entropy[可选]    │ logits:(B,T,V)  │ loss: scalar    │
        │    logits[:,:-1,:] vs     │ targets:(B,T)   │                │
        │    targets[:,1:]          │ (错位 1 位预测)  │                │
        └────────────────────────────────────────────────────────────┘

        【关键点】
          - weight tying: head.weight == tok_emb.weight（共享同一份参数）
          - cross_entropy 的 ignore_index=-100 让 SFT 可以 mask 掉 prompt 部分
          - value_head 默认 None；调用 attach_value_head() 后才挂载（PPO/RM 用）
          - 当 value_head 存在时返回 4 元组，否则 3 元组（向后兼容）

        【targets 的口径（feat-027 定下来的唯一约定）】
          约定：**targets 与 idx 同下标**（targets[t] = idx[t]），只是 SFT 会把
          prompt 部分置 -100。**错位（"logits[t] 预测 token t+1"）在本函数内部做**：
              CE(logits[:, :-1], targets[:, 1:])
          · 预训练线：`data/pretrain_data.py` 的 target_ids = input_ids（同下标）✅
          · SFT 线：`data/sft_data.py` 的 labels = input_ids，prompt 部分 -100 ✅
            （feat-027 之前：模型不左移 + labels 同下标 ⇒ 模型学成"复制当前 token"
              ⇒ 推理塌成吐空格；且预训练的数据侧左移、SFT 不左移，两条线口径打架）
          ⇒ 推论：logits[t] 恒等于「下一个 token 的分布」，PPO 的
            log_softmax(logits[:, :-1]) 配 full_ids[:, 1:] 用的就是这个约定。

        参数：
            idx:         (B, T) LongTensor 的 token id
            targets:     (B, T) LongTensor，与 idx **同下标**；-100 表示该位置不算 loss
                         None 时不计算 loss
            kv_cache:    KVCache 或 None（推理时用）
            position_ids:(B, T) LongTensor 绝对位置；kv_cache 不为 None 时必须显式提供
        返回：
            (logits, loss, new_kv_cache, value)：
                logits: (B, T, vocab_size)
                loss:   scalar（targets=None 时为 None）
                new_kv_cache: 同上
                value:  (B,) 仅在 value_head 不为 None 时存在
        """
        B, T = idx.size()#32, 128 / 推理的时候 1,22
        # 输入长度防御：训练/prefill 阶段不允许超过 block_size
        # （KV cache decode 步 T=1，永远不会触发）
        if T > self.block_size:
            raise ValueError(
                f"输入序列长度 T={T} 超过 block_size={self.block_size}。"
                "训练/prefill 必须先截断到 block_size。"
            )

        # === 0) position_ids 构造（若未提供）===
        #    默认用相对位置 0..T-1
        #    训练时 T 较小，position_ids 是 (B, T)
        #    推理时（带 kv_cache）必须显式传，因为 cache 已经有历史 token
        if position_ids is None:
            position_ids = torch.arange(0, T, dtype=torch.long, device=idx.device) #向量 【0，1,2,3,4.。。。。。。。】
            position_ids = position_ids.unsqueeze(0).expand(B, T)   # (B, T) 32,128 矩阵 / infra 1,22

        # === 1) Embedding：token + position ===
        #    tok_emb: idx (B,T) 的整数 → (B, T, C) 浮点向量
        tok = self.tok_emb(idx)   # idx:32,128  tok 32,128,128    /  infra 1,22,128                      # (B, T, C)
        #    pos_emb: 用 position_ids 在 buffer 的 PE 表里 gather
        #    self.pos_emb: (block_size, C) 固定正弦/余弦
        #    pos_emb[position_ids]: (B, T, C)  ← 每个位置取一行 PE
        pos = self.pos_emb[position_ids] #pos 32,128,128                     # (B, T, C)
        #    逐元素相加（tok 和 pos 同形状，可直接加）+ dropout
        x = self.drop(tok + pos)    #    32,128,128                     # (B, T, C)

        # === 2) N 个 Pre-LN Transformer Block 堆叠 ===
        #    每层内部: x = x + attn(LN(x));  x = x + mlp(LN(x))
        #    形状在所有 Block 中保持 (B, T, C)
        new_kv = kv_cache
        for layer_idx, block in enumerate(self.blocks):
            x, new_kv = block(   #x:32,128,128    new_kv:
                x, kv_cache=new_kv, layer_idx=layer_idx, position_ids=position_ids
            )

        # === 3) Final LayerNorm（所有 Block 之后）===
        x = self.ln_f(x)   #32,128,128                / infra 1,22,128                   # (B, T, C)

        # === 4) [可选] Value head：取最后 token hidden → scalar value ===
        #    RM/PPO 阶段会先调 attach_value_head()，这时 self.value_head 不为 None
        #    关键：用最后一 token 的 hidden 算 V(s)——它"看过了"所有前文
        value: Optional[torch.Tensor] = None
        if self.value_head is not None:
            last_hidden = x[:, -1, :]                        # (B, C)  ← 取最后位置  32,128,128  32个样本128个token，每个token128维向量表示,last_hidden 32,128
            value = self.value_head(last_hidden).squeeze(-1) # (B, 1) → (B,)  128,1    32,128  和 128,1  算出 32,1

        # === 5) 输出 logits（vocab 维投影）===
        #    关键：head.weight == tok_emb.weight（weight tying 共享参数）
        #    x: (B, T, C) @ W^T (C, V) → (B, T, V)
        logits = self.head(x) #32,128,45             /infra 1,22,45                  # (B, T, V)

        # === 6) [可选] 损失计算：cross_entropy（错位 1 位）===
        #    口径：**错位在模型内部做**，数据侧不动。
        #    targets[t] = input[t]（同下标），于是
        #      logits[:, :-1]  ↔  targets[:, 1:]
        #    即「位置 t 的输出预测位置 t+1 的 token」。
        #    ⇒ logits[t] 恒等于「下一个 token 的分布」，
        #      PPO（log_softmax(logits[:, :-1]) 配 full_ids[:, 1:]）与
        #      eval_benchmark 用的都是这个约定。
        loss: Optional[torch.Tensor] = None
        if targets is not None:    #targets 32,128
            # logits: (B, T, V) → 去掉最后一个位置 → view → (B*(T-1), V)
            # targets: (B, T)  → 去掉第一个位置 → view → (B*(T-1),)
            # ignore_index=-100：SFT 的 prompt 部分 loss 自动被 mask 掉
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),  # (B*(T-1), V)
                targets[:, 1:].contiguous().view(-1),                        # (B*(T-1),)
                ignore_index=-100,
            )

        # === 7) 返回：3 元组（无 value_head） vs 4 元组（PPO 风格）===
        if value is not None:
            return logits, loss, new_kv, value
        return logits, loss, new_kv

    # ------------------------------------------------------------------
    # 采样生成（教学版，不带 KV cache）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        idx: "torch.Tensor",
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_id: Optional[int] = None,
    ) -> "torch.Tensor":
        """
        朴素的 autoregressive 生成（**不带** KV cache，每步重新 forward 整段 prefix）。

        【复杂度】O(L²) per generation
        - 朴素版每生成第 t 个 token，要重新 forward 长度为 t 的整段
        - 用 KV cache 版本（generate_kv）则每步 O(1)

        【每步形状变化】
        ┌─────────────────────────────────────────────┐
        │ 步骤            │ 输入形状        │ 输出形状       │
        ├─────────────────────────────────────────────┤
        │ idx (prompt+已生)│ -              │ (B, T)        │
        │ forward(idx_cond)│ (B, T≤128)    │ logits: (B,T,V)│
        │ logits[:, -1, :]│ (B, T, V)      │ (B, V)        │
        │ sampling        │ (B, V)         │ next_id: (B,1)│
        │ cat([idx, nxt]) │ (B, T) + (B,1) │ (B, T+1)     │
        │ (循环 N 次)     │ -              │ (B, T+N)     │
        └─────────────────────────────────────────────┘

        用于：
          - 教学演示（看清 KV cache 带来的加速）
          - 单元测试（与 generate_kv 字符级一致时验证正确性）

        参数：
            idx:            (B, T) 起始 token id（prompt）
            max_new_tokens: 要生成的最大 token 数
            temperature:    采样温度（T→0 趋近 greedy）
            top_k:          只保留概率前 k 大的 token
            top_p:          nucleus 采样，保留累计概率 ≥ p 的最小 token 集
            eos_id:         早停 token id（遇 eos 立即停，不 pad）
        返回：
            (B, T + 实际生成长度) LongTensor
        """
        self.eval()
        for step in range(max_new_tokens):
            # === 0) 截断到 block_size ===
            #    训练过的最大长度是 block_size=128；超出后只看最后 128 个
            #    idx_cond: (B, min(T, block_size))
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]

            # === 1) forward（朴素：跑整段 prefix，不带 cache）===
            #    self() → GPT2.forward() 返回 3 元组（value_head=None）
            #    logits: (B, T_cond, V)
            logits, _, _ = self(idx_cond)
            # === 2) 取最后位置的 logits（预测下一个 token）===
            #    logits[:, -1, :]: (B, V)   V = vocab_size
            logits = logits[:, -1, :]

            # === 3) 采样策略：greedy / temperature + top-k + top-p ===
            if temperature <= 0.0:
                # 3.1) Greedy：argmax 取概率最大
                #    next_id: (B, 1)
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                # 3.2) Temperature scaling：logits / T
                #    T 越小分布越尖锐（→ greedy），T 越大越均匀
                logits = logits / temperature   # (B, V)
                # 3.3) Top-k：只保留概率最大的 k 个
                if top_k is not None and top_k > 0:
                    #    kth: (B, 1)  第 k 大的 logits 值
                    kth = logits.topk(top_k, dim=-1).values[:, -1:]
                    #    把 < kth 的位置设为 -inf（softmax 后概率 0）
                    logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
                # 3.4) Top-p (nucleus)：保留累计概率到 p 为止
                if top_p is not None and 0.0 < top_p < 1.0:
                    #    sort 降序
                    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                    #    计算 softmax 概率
                    sorted_probs = sorted_logits.softmax(dim=-1)        # (B, V)
                    #    累积概率
                    cumsum = sorted_probs.cumsum(dim=-1)               # (B, V)
                    #    mask 掉"超过 p 的部分"（保留刚好越过 p 的那个）
                    mask = cumsum - sorted_probs > top_p                # (B, V) bool
                    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
                    #    scatter 回原 token 顺序
                    logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
                # 3.5) softmax + 采样一个 token
                probs = logits.softmax(dim=-1)                          # (B, V)
                next_id = torch.multinomial(probs, num_samples=1)      # (B, 1)

            # === 4) 把新 token 拼到末尾 ===
            #    idx: (B, T) → cat → (B, T+1)
            idx = torch.cat([idx, next_id], dim=1)

            # === 5) 早停（教学版：只对 batch 0 简化判断）===
            #    严谨的 batched 早停见 inference.batched_generate
            if eos_id is not None and (next_id == eos_id).all():
                break

        return idx   # (B, T + 实际生成长度)

    # ------------------------------------------------------------------
    # 工具：统计参数量
    # ------------------------------------------------------------------
    def num_params(self, non_embedding: bool = False) -> int:
        """
        统计可训练参数量（PyTorch 风格统计：所有 requires_grad=True 的 tensor 之和）。

        参数：
            non_embedding: 若为 True，则**不**算 token embedding 矩阵（用于消融实验对比）
        返回：
            参数数量（int）
        """
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    # ------------------------------------------------------------------
    # 工具：value head 挂载（PPO 用）
    # ------------------------------------------------------------------
    def attach_value_head(self) -> None:
        """
        在主模型上挂一个 scalar value head（PPO 用）。
        通常在 SFT 模型加载后调用一次。

        实现：`self.value_head = nn.Linear(n_embd, 1, bias=True)`
        默认权重初始化由 PyTorch 完成（Kaiming uniform）。
        """
        if self.value_head is not None:
            return  # 已挂
        self.value_head = nn.Linear(self.n_embd, 1, bias=True)

    # ------------------------------------------------------------------
    # 工具：KV cache 版 forward（与 GPT2WithKVCache 配合）
    # ------------------------------------------------------------------
    def forward_with_kv(
        self,
        idx: "torch.Tensor",
        kv_cache: Optional["object"] = None,
        position_ids: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", Optional["object"]]:
        """
        增量 forward（与 GPT2WithKVCache 配合使用）。

        【形状速查表】
        ┌─────────────────────────────────────────────┐
        │ 步骤             │ 输入形状       │ 输出形状      │
        ├─────────────────────────────────────────────┤
        │ idx (token ids)  │ -              │ (B, T_q)     │
        │ tok_emb + pos    │ (B, T_q)       │ (B, T_q, C)  │
        │ N × Block        │ (B, T_q, C)    │ (B, T_q, C)  │
        │ ln_f             │ (B, T_q, C)    │ (B, T_q, C)  │
        │ head             │ (B, T_q, C)    │ (B, T_q, V)  │
        └─────────────────────────────────────────────┘

        【与 forward 的区别】
          - 不接收 targets（KV cache decode 阶段用不到 loss）
          - 不挂 value head（推理时用不到）
          - 不算 loss，返回 2 元组 (logits, new_kv_cache)
          - decode 阶段 T_q = 1，但仍要跑完整 N 层（因为 attention 需读历史 cache）

        参数：
            idx:         (B, T_q) LongTensor；decode 步通常 T_q=1
            kv_cache:    KVCache 或 None
            position_ids:(B, T_q) 绝对位置
        返回：
            (logits, new_kv_cache)：
                logits: (B, T_q, vocab_size)
                new_kv_cache: 同上
        """
        B, T_q = idx.size() #1,22 、 decode 阶段 1,1

        # === 0) position_ids 构造（推理时通常已显式传入）===
        if position_ids is None: #1,22 之前prefill 是 0-21 ，现在decode 阶段就是 22 开始
            position_ids = torch.arange(0, T_q, dtype=torch.long, device=idx.device)
            position_ids = position_ids.unsqueeze(0).expand(B, T_q)   # (B, T_q)

        # === 1) Embedding（与 forward 一致）===
        #    idx:    (B, T_q)  1,22  推理的话，应该就不用 补齐为 统一长度  decode 阶段1,1
        #    tok:    (B, T_q, C)
        tok = self.tok_emb(idx)  #1,22,128  1,1,128
        pos = self.pos_emb[position_ids]   # (B, T_q, C) 1,22,128 , 1,1,128
        x = self.drop(tok + pos)          # (B, T_q, C)  1,1,128

        # === 2) N × Block（带 KV cache，attention 内部会读 cache）===
        new_kv = kv_cache # kv.values.size  4,1,4,128,32  4层，1个样本，4个头，每个头32维，kv最大长度为128
        for layer_idx, block in enumerate(self.blocks):
            x, new_kv = block(
                x, kv_cache=new_kv, layer_idx=layer_idx, position_ids=position_ids
            )

        # === 3) final LN + head（投到 vocab 维）===  decode阶段1,1,128
        x = self.ln_f(x)         # (B, T_q, C)
        logits = self.head(x)    # (B, T_q, V)  decode 阶段 1,1,45
        return logits, new_kv

"""
Qwen3 因果自注意力：QK-Norm + RoPE + GQA + KV Cache。

对应笔记：qwen3-learning/Qwen3_QK-Norm_学习笔记.md
          qwen3-learning/Qwen3_GQA与KV_Cache_完整学习笔记.md

【相对 GPT-2 线 attention 的四处改动】
  1) QK-Norm：Q/K 投影后、RoPE 前各过一次 RMSNorm（GPT-2 没有）
  2) RoPE：旋转 Q/K（GPT-2 用 sinusoidal 加法，且 Q 与 K **不**加位置）
  3) GQA：K/V head 数 < Q head 数，靠 repeat_interleave 扩开
  4) 三个独立投影：因 head_dim 与 n_embd 解耦，Q/K/V 的输出维度不再相等，
     无法像 GPT-2 那样用一个 c_attn 一次投出 3*n_embd

【执行顺序（本文件最重要的一行注释）】
    q_proj/k_proj/v_proj → 拆 head → **QK-Norm** → **RoPE** → GQA 扩开 → 注意力
    QK-Norm 在 RoPE **之前**：先把数值尺度稳定下来，再旋转。
    若反序，大数值与 cos/sin 累乘会放大误差。

    —— 关于「顺序真的重要吗」的一个实测结论（2026-10-22）：
    RMSNorm 与 RoPE 在 **gain 均匀** 时数值上几乎可交换（实测 max|diff| ≈ 5e-7，
    纯浮点噪声）；只有当 gain **非均匀**（可学习参数被训成非均匀）时两者才显著不同
    （实测 max|diff| ≈ 14.9）。原因是 rotate_half 把特征维的前后两半交叉配对，
    均匀缩放与旋转可交换，非均匀缩放则不可交换。
    所以「QK-Norm 必须在 RoPE 之前」这条约束在**训练早期（gain≈1）不产生数值差异**，
    它的价值在于：① 官方实现的一致性 ② gain 学成非均匀后顺序才会显现。
    tests/test_attention.py 第 3 条用非均匀 gain 做了区分性验证。
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common.rmsnorm import RMSNorm
from model.common.rope import RoPE


class CausalSelfAttention(nn.Module):
    """
    Qwen3 的因果自注意力。

    参数：
        n_embd:      隐藏维度（残差流宽度）
        n_head:      Q head 数
        n_head_kv:   KV head 数（GQA）。必须整除 n_head
        head_dim:    单 head 特征维度（**与 n_embd 解耦**）
        block_size:  最大上下文长度
        dropout:     dropout 比例
        bias:        投影层是否用 bias（Qwen3 为 False）
        rope_theta:  RoPE 频率基数
        rms_eps:     QK-Norm 的 epsilon
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_head_kv: int,
        head_dim: int,
        block_size: int,
        dropout: float,
        bias: bool,
        rope_theta: float = 1e6,
        rms_eps: float = 1e-6,
    ) -> None:
        """
        初始化 CausalSelfAttention。参数说明见类 docstring。
        """
        super().__init__()
        if n_head % n_head_kv != 0:
            raise ValueError(f"n_head({n_head}) 必须整除 n_head_kv({n_head_kv}) 的倍数")

        self.n_embd = n_embd
        self.n_head = n_head
        self.n_head_kv = n_head_kv
        self.head_dim = head_dim
        self.block_size = block_size
        # 每个 KV head 被几个 Q head 共享（GQA 的「组」大小）
        self._kv_group_size = n_head // n_head_kv

        # --- 三个独立投影（不能合并：Q/K/V 输出维度在 GQA 下不同）---
        self.q_proj = nn.Linear(n_embd, n_head * head_dim, bias=bias)
        self.k_proj = nn.Linear(n_embd, n_head_kv * head_dim, bias=bias)
        self.v_proj = nn.Linear(n_embd, n_head_kv * head_dim, bias=bias)
        # --- 输出投影：n_head*head_dim → n_embd（这正是解耦的地方）---
        self.o_proj = nn.Linear(n_head * head_dim, n_embd, bias=bias)

        # --- QK-Norm：对每个 head 的 head_dim 做 RMSNorm ---
        # 注意作用维度是 head_dim 而非 n_embd —— 逐 head 独立归一化
        self.q_norm = RMSNorm(head_dim, eps=rms_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_eps)

        # --- RoPE：作用在 head_dim 上 ---
        self.rope = RoPE(head_dim, max_seq_len=block_size, theta=rope_theta)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # 因果 mask：(block_size, block_size) 上三角，True = 被屏蔽
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def _project_and_rotate(
        self,
        x: "torch.Tensor",
        position_ids: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        投影 → 拆 head → **QK-Norm** → **RoPE**，返回 (q, k)。

        单独抽出来是为了让「QK-Norm 在 RoPE 之前」这条约束**可被数值验证**
        （见 tests/test_attention.py 第 3 条），而不是只靠注释断言。

        参数：
            x:            (B, T_q, n_embd)
            position_ids: (B, T_q) 绝对位置；None 时用 0..T_q-1
        返回：
            (q, k)，各 (B, n_head/n_head_kv, T_q, head_dim)
        """
        B, T_q, _ = x.size()
        H = self.n_head
        Hkv = self.n_head_kv
        D = self.head_dim

        # --- 1) 三个独立投影 ---
        q = self.q_proj(x).view(B, T_q, H, D).transpose(1, 2)      # (B, H,   T_q, D)
        k = self.k_proj(x).view(B, T_q, Hkv, D).transpose(1, 2)    # (B, Hkv, T_q, D)

        # --- 2) QK-Norm：在 RoPE 之前 ---
        # 逐 head 对 head_dim 做 RMSNorm，形状不变
        q = self.q_norm(q)
        k = self.k_norm(k)

        # --- 3) RoPE：按绝对位置旋转 ---
        q, k = self.rope(q, k, position_ids=position_ids)

        return q, k

    def forward(
        self,
        x: "torch.Tensor",
        kv_cache: Optional[object] = None,
        layer_idx: int = 0,
        position_ids: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", Optional[object]]:
        """
        因果自注意力 forward。

        【形状速查表】（B=batch, T=time, H=n_head, Hkv=n_head_kv, D=head_dim, C=n_embd）
        ┌───────────────────────────────────────────────────────────────────────┐
        │ 步骤                │ 输入形状              │ 输出形状                 │
        ├───────────────────────────────────────────────────────────────────────┤
        │ 0. 输入 x           │ -                     │ (B, T_q, C)              │
        │ 1. q_proj           │ (B, T_q, C)           │ (B, T_q, H·D)            │
        │    k_proj           │ (B, T_q, C)           │ (B, T_q, Hkv·D)          │
        │    v_proj           │ (B, T_q, C)           │ (B, T_q, Hkv·D)          │
        │ 2. 拆 head          │ (B, T, ·)             │ (B, H, T_q, D) / (B, Hkv,…)│
        │ 3. **QK-Norm**      │ (B, H, T_q, D)        │ 形状不变（scale）        │
        │ 4. **RoPE**         │ (B, H, T_q, D)        │ 形状不变（旋转）         │
        │ 5. GQA 扩开 KV      │ (B, Hkv, T_kv, D)     │ (B, H, T_kv, D)          │
        │ 6. QKᵀ/√D           │ (B,H,T_q,D)×(B,H,D,T_kv)│ (B,H,T_q,T_kv)        │
        │ 7. causal mask      │ (B, H, T_q, T_kv)     │ 同上                     │
        │ 8. softmax @ V      │ (B,H,T_q,T_kv)×(B,H,T_kv,D)│ (B,H,T_q,D)       │
        │ 9. 拼 head + o_proj │ (B, H, T_q, D)        │ (B, T_q, C)              │
        └───────────────────────────────────────────────────────────────────────┘

        【GQA 的形状要点】
          K/V 算出来只有 Hkv 个 head。第 5 步用 repeat_interleave 沿 head 维复制
          `H // Hkv` 份，让 K/V 也有 H 个 head，才能和 Q 做 batch matmul。
          这样**计算量不变**，但 K/V 的**缓存与投影计算量降到 1/(H/Hkv)**。

        【T_q ≠ T_kv 的情况】
          · 训练 / prefill：T_q = T_kv = T，需要因果 mask
          · decode：        T_q = 1，T_kv = 已缓存的全部历史，**不需要 mask**
            （cache 天然只含历史与当前 token，不含未来）

        参数：
            x:            (B, T_q, n_embd)
            kv_cache:     KVCache 实例（推理）或 None（训练）
            layer_idx:    层索引（写 cache 用）
            position_ids: (B, T_q) 绝对位置；None 时用 0..T_q-1
        返回：
            (output, kv_cache)，output 为 (B, T_q, n_embd)
        """
        B, T_q, _ = x.size()
        H = self.n_head
        Hkv = self.n_head_kv
        D = self.head_dim

        # --- 1) 投影 + QK-Norm + RoPE（顺序见 _project_and_rotate 的 docstring）---
        q, k = self._project_and_rotate(x, position_ids=position_ids)
        v = self.v_proj(x).view(B, T_q, Hkv, D).transpose(1, 2)    # (B, Hkv, T_q, D)

        # --- 2) KV Cache：写入新算出的 K/V，取出累计的全长 K/V ---
        if kv_cache is not None:
            kv_cache.append(layer_idx, k, v)
            k_full, v_full = kv_cache.get(layer_idx)
        else:
            k_full, v_full = k, v

        T_kv = k_full.size(2)

        # --- 3) GQA：把 Hkv 个 KV head 复制成 H 个 ---
        # repeat_interleave 而非 repeat：必须保证「连续的 Hkv 个 Q head 共享同一 KV head」
        # 这正是 GQA 的分组语义；若用 repeat（整体复制）会变成 MQA
        if Hkv != H:
            k_full = k_full.repeat_interleave(self._kv_group_size, dim=1)   # (B, H, T_kv, D)
            v_full = v_full.repeat_interleave(self._kv_group_size, dim=1)

        # --- 4) 缩放点积 ---
        att = (q @ k_full.transpose(-2, -1)) * (1.0 / math.sqrt(D))   # (B, H, T_q, T_kv)

        # --- 5) 因果 mask（仅 prefill / 训练需要）---
        # 取 mask[T_kv-T_q : T_kv, :T_kv] 而非 mask[:T_q, :T_kv]：
        #   · 无 cache 的 prefill：T_kv == T_q → 等价于 mask[:T_q, :T_kv] ✓
        #   · 有 cache 且 T_q>1：query i 的绝对位置是 (T_kv-T_q)+i，
        #     它应当能看到全部 T_kv 个 key（含 cache 里的历史），用 mask[:T_q] 会错杀
        if T_q > 1:
            att = att.masked_fill(self.mask[T_kv - T_q:T_kv, :T_kv], float("-inf"))

        # --- 6) softmax + 加权求和 ---
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v_full                                    # (B, H, T_q, D)

        # --- 7) 拼回 (B, T_q, H·D) 再输出投影到 (B, T_q, C) ---
        y = y.transpose(1, 2).contiguous().view(B, T_q, H * D)
        y = self.resid_dropout(self.o_proj(y))

        return y, kv_cache

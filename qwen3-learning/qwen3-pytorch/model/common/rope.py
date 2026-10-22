"""
RoPE（Rotary Position Embedding，旋转位置编码）。

对应笔记：qwen3-learning/Qwen3_RoPE_学习笔记.md
对照笔记：gpt-learning/2.位置编码.md（Sinusoidal 加法式）

【机制差异 —— 这是本文件最该记住的一件事】
  Sinusoidal（GPT-2 线）：位置信息**加到** embedding 上
      x = token_emb[i] + pos_emb[pos] → 送进 attention
      位置信息在投影成 Q/K 之前就被混进 hidden states，无法区分
  RoPE（Qwen3）：位置信息**旋转进** Q/K 本身
      q' = R(pos) · q,  k' = R(pos) · k
      Q/K 向量本身带着角度，注意力分数天然只依赖相对位置

【为什么有效】
  R 是块对角旋转矩阵，满足 R(m)ᵀR(n) = R(n-m)。
  于是 q_mᵀk_n = qᵀR(m)ᵀR(n)k = qᵀR(n-m)k —— **内积只依赖相对距离 n-m**。
  这就是 RoPE 的全部理论基础（见 tests/test_rope.py 第 5 条断言）。

【为什么 theta = 1e6】
  频率 inv_freq[i] = 1 / theta^(2i/d)。theta 越大，低频维度旋转越慢，
  波长越长，长上下文下的位置信息退化得越晚。GPT-2 用 1e4，Qwen3 用 1e6。
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


def _rotate_half(x: "torch.Tensor") -> "torch.Tensor":
    """
    把特征维的前后两半交换并取负：[-x2, x1]。

    这是 HF / 官方实现的写法。数学上等价于「按 i 交替施加 ±90° 旋转」，
    但实现代价更低（一次 cat 而非逐元素构造旋转矩阵）。

    参数：
        x: (..., head_dim)，head_dim 必须是偶数
    返回：
        (..., head_dim)
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class RoPE(nn.Module):
    """
    旋转位置编码模块。预计算 cos / sin 查找表，forward 时按位置索引并施加旋转。

    参数：
        head_dim:    单个 attention head 的特征维度（必须是偶数）
        max_seq_len: 预计算的最大位置数
        theta:       频率基数（Qwen3 用 1e6）
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 1e6) -> None:
        """
        初始化 RoPE。

        参数：
            head_dim:    head 特征维度，必须为偶数
            max_seq_len: 支持的最大序列长度
            theta:       频率基数
        """
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim 必须为偶数才能配对旋转，实得 {head_dim}")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # --- 预计算 cos / sin 表 ---
        # 维度索引 i = 0, 2, 4, ..., head_dim-2（每隔一维取一个）
        # inv_freq[i] = 1 / theta^(2i/head_dim)，i 越大频率越低
        dim_idx = torch.arange(0, head_dim, 2, dtype=torch.float32)   # (head_dim/2,)
        inv_freq = 1.0 / (theta ** (dim_idx / head_dim))              # (head_dim/2,)
        # 位置索引 0..max_seq_len-1
        positions = torch.arange(max_seq_len, dtype=torch.float32)    # (max_seq_len,)
        # 外积得到每个 (位置, 维度) 的相位角：(max_seq_len, head_dim/2)
        freqs = torch.outer(positions, inv_freq)                     # (max_seq_len, head_dim/2)

        # 把相位角复制两遍 → cos / sin 表都是 (max_seq_len, head_dim)
        # 复制两遍是为了让 forward 时能与 x[..., :half] 和 x[..., half:] 逐半配对
        emb = torch.cat((freqs, freqs), dim=-1)                      # (max_seq_len, head_dim)
        self.register_buffer("cos", emb.cos(), persistent=False)     # 固定表，不训练、不入 checkpoint
        self.register_buffer("sin", emb.sin(), persistent=False)

    def get_cos_sin(self, position_ids: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        按绝对位置取 cos / sin。

        【形状】
            position_ids: (..., T) LongTensor —— 末维是位置
            返回 cos/sin: (..., T, head_dim)

        参数：
            position_ids: 绝对位置 id
        返回：
            (cos, sin)，形状均为 position_ids.shape + (head_dim,)
        """
        max_pos = int(position_ids.max().item())
        if max_pos >= self.max_seq_len:
            raise IndexError(
                f"position 最大值 {max_pos} 超过 max_seq_len={self.max_seq_len}，"
                "请调大 config.block_size"
            )
        # 在位置维做 gather；索引后形状 = position_ids.shape + (head_dim,)
        return self.cos[position_ids], self.sin[position_ids]

    def _apply_rotation(
        self,
        x: "torch.Tensor",
        cos: "torch.Tensor",
        sin: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        对 x 施加旋转：`x' = x * cos + rotate_half(x) * sin`。

        【形状速查表】（D = head_dim）
        ┌───────────────────────┬────────────────┬────────────────┐
        │ 步骤                  │ 输入形状       │ 输出形状       │
        ├───────────────────────┼────────────────┼────────────────┤
        │ x                     │ (B, H, T, D)   │ (B, H, T, D)   │
        │ rotate_half(x)        │ (B, H, T, D)   │ (B, H, T, D)   │
        │ cos / sin (已 unsqueeze(1) 到 head 维)│ (B,1,T,D)│ (B,1,T,D)│
        │ x*cos + rot(x)*sin    │ (B, H, T, D)   │ (B, H, T, D)   │
        └───────────────────────┴────────────────┴────────────────┘

        参数：
            x:   (B, H, T, D)
            cos: (B, 1, T, D) 或可广播到该形状
            sin: 同 cos
        返回：
            (B, H, T, D)
        """
        return x * cos + _rotate_half(x) * sin

    def forward(
        self,
        q: "torch.Tensor",
        k: "torch.Tensor",
        position_ids: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        对 q 与 k 施加旋转位置编码。

        【与 attention 的关系】
        调用顺序必须是 **QK-Norm → RoPE**（见 model/common/attention.py）：
        先把 Q/K 的数值尺度稳定下来，再旋转，避免大数值下 cos 累加误差放大。

        【形状速查表】
        ┌────────────────────────────┬─────────────────┬─────────────────┐
        │ 步骤                       │ 输入形状        │ 输出形状        │
        ├────────────────────────────┼─────────────────┼─────────────────┤
        │ q, k                       │ (B, H, T, D)   │ (B, H, T, D)   │
        │ position_ids=None → arange │ -               │ (B, T)         │
        │ cos, sin = get_cos_sin     │ (B, T)          │ (B, T, D)      │
        │ unsqueeze(1) 插入 head 维  │ (B, T, D)       │ (B, 1, T, D)   │
        │ _apply_rotation(q)         │ (B, H, T, D)   │ (B, H, T, D)   │
        │ _apply_rotation(k)         │ (B, H, T, D)   │ (B, H, T, D)   │
        └────────────────────────────┴─────────────────┴─────────────────┘

        参数：
            q: (B, n_head,     T, head_dim)
            k: (B, n_head_kv,  T, head_dim)
            position_ids: 绝对位置，末维是序列维（如 (B, T) 或 (B, 1, T)）；
                None 时用 0..T-1。**KV Cache decode 阶段必须显式传入当前步的绝对位置**，
                否则每步都会从 0 重新编号，位置信息全乱。
        返回：
            (q_rot, k_rot)，形状均与输入相同
        """
        B, _, T, _ = q.shape
        if position_ids is None:
            position_ids = torch.arange(T, dtype=torch.long, device=q.device)
            position_ids = position_ids.unsqueeze(0).expand(B, T)   # (B, T)

        # 统一成 2D (B', T)：允许调用方传 (B, T) / (1, T) / (B, 1, T) 等形状
        pos_2d = position_ids.reshape(-1, position_ids.shape[-1])   # (B', T)

        cos, sin = self.get_cos_sin(pos_2d)                        # (B', T, D)
        # unsqueeze(1) 插入 head 维，让同一组 cos/sin 广播给所有 head
        cos = cos.unsqueeze(1)   # (B', 1, T, D)
        sin = sin.unsqueeze(1)
        return self._apply_rotation(q, cos, sin), self._apply_rotation(k, cos, sin)

"""
SwiGLU 前馈网络。

对应笔记：qwen3-learning/Qwen3_SwiGLU_MLP_学习笔记.md
对照主线：gpt-learning-pytorch/model/gpt2.py 的 MLP（GELU）

【公式】
    SwiGLU(x) = W_down · ( SiLU(W_gate · x) ⊙ (W_up · x) )

    其中 SiLU(z) = z · sigmoid(z)，⊙ 是逐元素相乘。
    「门控」的含义：SiLU(gate) 是一个由输入自己决定的软开关，逐元素
    放大/抑制 up 分支的每一维。

【相对 GPT-2 MLP 的三点差异】
  1) **三个矩阵而非两个**：多了 W_gate，用来产生门控信号
  2) **激活函数**：SiLU 而非 GELU
  3) **中间维更窄**：GPT-2 取 4×n_embd；SwiGLU 取 round(n_embd×8/3) 再对齐到
     硬件友好的倍数。因为要两个矩阵做升维，宽度必须减半才能让参数量持平。

【为什么要对齐到 128 的倍数】
    n_embd×8/3 通常不是整数（128×8/3 = 341.33）。向下取整会削弱容量，
    向上取整到任意数字又可能让矩阵维度不能被硬件的 128 宽度整除。
    对齐到 128 的倍数 = 容量不打折 + 张量核心友好。本项目 128×8/3 → 384。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def align_intermediate_size(n_embd: int, multiple: int = 128) -> int:
    """
    按 8/3 规则计算中间层维度，并对齐到 `multiple` 的倍数（向上取整）。

    【为什么是 8/3】
      SwiGLU 用**两个**矩阵做升维（gate + up），GPT-2 只用一个（c_fc）。
      若中间维不变，参数量直接翻倍。把宽度砍到 8/3 ≈ 2.67 倍，
      两个 2.67 宽的矩阵合起来才与一个 4 宽的矩阵参数量相当：
          2 × (8/3) ≈ 5.33 > 4  → 仍略多，故取略保守的 8/3（而非 16/3=5.33）。

    【为什么要对齐到 multiple 的倍数】
      向下取整到整数会削弱容量；向上取整到任意数字又可能让矩阵维度
      不能被硬件的 128 宽度整除。对齐到 128 的倍数 = 容量不打折 + 张量核心友好。

    参数：
        n_embd:   残差流宽度
        multiple: 对齐到的倍数
    返回：
        中间层维度（int）
    """
    raw = n_embd * 8 / 3
    return int(math.ceil(raw / multiple) * multiple)


class SwiGLUMLP(nn.Module):
    """
    Qwen3 的前馈网络（SwiGLU）。

    参数：
        n_embd:           输入/输出维度
        intermediate_size: 中间层维度（已按 8/3 规则对齐到 128 的倍数）
        dropout:          输出 dropout
        bias:             投影层是否用 bias（Qwen3 为 False）
    """

    def __init__(
        self,
        n_embd: int,
        intermediate_size: int,
        dropout: float,
        bias: bool = False,
    ) -> None:
        """
        初始化 SwiGLUMLP。参数说明见类 docstring。
        """
        super().__init__()
        self.n_embd = n_embd
        self.intermediate_size = intermediate_size
        # 三个投影，全部无 bias
        self.gate_proj = nn.Linear(n_embd, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(n_embd, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        SwiGLU forward（位置独立的非线性变换）。

        【形状速查表】（I = intermediate_size）
        ┌────────────────────┬──────────────────┬──────────────────┐
        │ 步骤               │ 输入形状         │ 输出形状         │
        ├────────────────────┼──────────────────┼──────────────────┤
        │ 输入 x             │ -                │ (B, T, C)        │
        │ 1. gate_proj       │ (B, T, C)        │ (B, T, I)        │
        │ 2. SiLU(gate)      │ (B, T, I)        │ (B, T, I)        │
        │ 3. up_proj         │ (B, T, C)        │ (B, T, I)        │
        │ 4. 逐元素相乘      │ (B,T,I)×(B,T,I)  │ (B, T, I)        │
        │ 5. down_proj       │ (B, T, I)        │ (B, T, C)        │
        │ 6. dropout         │ (B, T, C)        │ (B, T, C)        │
        └────────────────────┴──────────────────┴──────────────────┘

        参数：
            x: (B, T, n_embd)
        返回：
            (B, T, n_embd)  ← 必须回到 n_embd 才能与残差相加
        """
        # 1) 门控分支：决定每一维「开多大」
        gate = F.silu(self.gate_proj(x))          # (B, T, I)
        # 2) 内容分支：待被门控的信号
        up = self.up_proj(x)                      # (B, T, I)
        # 3) 逐元素门控
        hidden = gate * up                        # (B, T, I)
        # 4) 降维回残差流宽度
        return self.dropout(self.down_proj(hidden))   # (B, T, C)

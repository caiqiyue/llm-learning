"""
RMSNorm（Root Mean Square Normalization）。

对应笔记：qwen3-learning/Qwen3_RMSNorm_学习笔记.md

公式：
    RMSNorm(x) = x / sqrt(mean(x²) + eps) * gain

与 GPT-2 线 `nn.LayerNorm` 的三点差异（这是 Qwen3 的关键改动之一）：
  1) **不减均值**：LayerNorm 先算 (x - mean) 再除以 std；RMSNorm 直接用 x 的平方均值。
     少一次中心化，计算更省，且在 LLM 实践中效果不掉。
  2) **无 bias**：LayerNorm 有 weight + bias 两个参数，RMSNorm 只有 gain 一个。
  3) **eps 位置不同**：LayerNorm 是 (x - mean) / sqrt(var + eps)；
     RMSNorm 是 x / sqrt(mean(x²) + eps)——eps 加在分母的根号**内部**，
     量纲是「数值本身」而不是「方差」，所以 Qwen3 用 1e-6 比 LayerNorm 常用的 1e-5 更小。
"""

from typing import Optional

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    均方根归一化层。

    参数：
        dim:      归一化的维度（通常是 n_embd；QK-Norm 里则是 head_dim）
        eps:      数值稳定项，加在 sqrt 内部
        gain_init: gain 的初始值（默认 1.0，与 LayerNorm 的 weight 初值一致）
    """

    def __init__(self, dim: int, eps: float = 1e-6, gain_init: float = 1.0) -> None:
        """
        初始化 RMSNorm。

        参数：
            dim:       归一化维度
            eps:       数值稳定项
            gain_init: gain 初值
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        # gain 是唯一可学习参数；形状 [dim]，沿最后一维做逐元素缩放
        self.gain = nn.Parameter(torch.full((dim,), gain_init))
        # 刻意不注册 bias —— RMSNorm 没有偏置项
        self.bias: Optional[torch.Tensor] = None

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        RMSNorm forward。

        【形状速查表】（C = dim）
        ┌──────────────────────────────┬─────────────────┬─────────────────┐
        │ 步骤                         │ 输入形状        │ 输出形状        │
        ├──────────────────────────────┼─────────────────┼─────────────────┤
        │ 输入 x                       │ -               │ (B, T, C)       │
        │ x²                           │ (B, T, C)       │ (B, T, C)       │
        │ mean(x², -1)                 │ (B, T, C)       │ (B, T, 1)       │
        │ + eps                        │ (B, T, 1)       │ (B, T, 1)       │
        │ sqrt                          │ (B, T, 1)       │ (B, T, 1)       │
        │ x / sqrt(·)                  │ (B, T, C)       │ (B, T, C)       │
        │ * gain                       │ (B, T, C)       │ (B, T, C)       │
        └──────────────────────────────┴─────────────────┴─────────────────┘
        任意前置维度都可以：dim=-1 的 mean 与 gain 的广播都自适应。

        参数：
            x: (..., dim)
        返回：
            (..., dim)
        """
        # 1) 平方并沿最后一维求均值：x (..., C) -> (..., 1)
        #    注意用 x.float() 保证 fp16 训练时数值稳定
        dtype = x.dtype
        x32 = x.float()
        # 2) 加 eps 后开根号：eps 在根号内部（见模块 docstring 的第 3 点）
        rms_inv = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # (..., 1)
        # 3) 归一化后乘 gain：gain 形状 [C]，沿最后一维广播
        out = x32 * rms_inv * self.gain
        # 4) 还原原 dtype
        return out.to(dtype)

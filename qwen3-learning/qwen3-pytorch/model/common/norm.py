"""
归一化层的统一构造入口。

对应笔记：qwen3-learning/Qwen3_RMSNorm_学习笔记.md（§12 对照实验里的「归一化必须同步切换」）

【为什么需要这个工厂】
    Qwen3 主线用 **RMSNorm**（不减均值、无 bias），
    GPT-2 用 **nn.LayerNorm**（减均值、有 bias）。
    两者在数学上**不等价** —— 只改 Qwen3 一侧永远对不齐。

    feat-017 的退化对照因此需要一个「按名字造归一化层」的口子：
      - 主线配置 `norm_type="rmsnorm"` → RMSNorm（行为不变）
      - 对照配置 `norm_type="layernorm"` → nn.LayerNorm

【参数名不统一的坑（对照脚本必须知道）】
    RMSNorm 的可学习参数叫 **`gain`**（形状 [dim]，无 bias）
    LayerNorm 的可学习参数叫 **`weight` + `bias`**（各 [dim]）
    ⇒ 退化对照做权重映射时，这两套键名要对上，见
      `ablations/compare_with_gpt2.py` 的 `build_target_state`。
"""

from typing import Union

import torch.nn as nn

from model.common.rmsnorm import RMSNorm


def build_norm(kind: str, dim: int, eps: float) -> Union[RMSNorm, nn.LayerNorm]:
    """
    按名字构造归一化层。

    参数：
        kind: "rmsnorm" | "layernorm"
        dim:  归一化的维度
        eps:  数值稳定项。**对照实验里必须两边用同一个 eps**
              （GPT-2 的 nn.LayerNorm 默认 eps=1e-5）
    返回：
        RMSNorm 或 nn.LayerNorm
    """
    if kind == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    if kind == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    raise ValueError(f"未知的归一化类型：{kind!r}（只支持 'rmsnorm' / 'layernorm'）")

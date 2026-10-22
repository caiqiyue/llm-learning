"""
加法式位置嵌入（sinusoidal）。

对应笔记：qwen3-learning/Qwen3_RoPE_学习笔记.md（§12 对照实验里的「位置信息从哪来」）
对照主线：gpt-learning/gpt-learning-pytorch/model/gpt2.py 的 `_build_sinusoidal_pos_emb`

【为什么主线要有这个文件】
    Qwen3 主线的位置信息由 **RoPE** 承担（旋转进 Q/K，不需要加法）。
    但 feat-017 的退化对照要求「关掉 RoPE 之后，Qwen3 与 GPT-2 的输出能对齐」，
    而 GPT-2 用的是 **加法式 sinusoidal 位置嵌入**。
    所以这里提供一份与 GPT-2 公式**逐项一致**的构造器，供 `use_rope=False` 时使用。

【公式】
    PE(pos, 2i)   = sin(pos / 10000^(2i/n_embd))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/n_embd))

【与 GPT-2 实现的关系（重要，别记错）】
    GPT-2 线的 `pos_emb` 是 **register_buffer(..., persistent=False)**：
      - 它是**固定表，不训练**（不是 learned PE）
      - 它**不在 state_dict 里** ⇒ 对照脚本必须单独构造，不能指望 state_dict 里有
    本文件只提供「怎么算这张表」，不提供「这张表存哪」——
    存成可训练参数还是 buffer，由调用方（`model/qwen3_dense.py`）决定。
"""

import math

import torch


def build_sinusoidal_pos_emb(block_size: int, n_embd: int) -> "torch.Tensor":
    """
    构造固定的 sinusoidal 位置编码表（与原始 Transformer 2017 一致）。

    【公式】
        PE(pos, 2i)   = sin(pos / 10000^(2i/n_embd))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/n_embd))

    【形状】
        (block_size, n_embd) 的 float32 张量。

    【为什么要逐项对齐 GPT-2 的实现】
        对照实验要求 fp32 下 max_abs_diff < 1e-4。位置表只要有一处不一致
        （比如偶数维用 cos、奇数维用 sin），误差就会直接进到 embedding 上，
        后面 4 层放大后必然超标。所以下面三行与 GPT-2 的实现逐字一致：
          1. 频率因子用 `exp(arange(0, n_embd, 2) * -(log(10000) / n_embd))`
          2. 偶数维填 sin、奇数维填 cos
          3. 位置索引是 float 而不是 long

    参数：
        block_size: 最大序列长度（T）
        n_embd:     隐藏维度（C，必须是偶数）
    返回：
        (block_size, n_embd) 的浮点张量
    """
    if n_embd % 2 != 0:
        raise ValueError(f"n_embd 必须是偶数（sin/cos 两两配对），实得 {n_embd}")
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

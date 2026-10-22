"""
KV Cache：自回归推理时缓存历史 token 的 K/V。

对应笔记：qwen3-learning/Qwen3_GQA与KV_Cache_完整学习笔记.md

【与 GPT-2 线的关键差异】
  GPT-2 线（MHA）：K/V 缓存按 **n_head** 个 head 分配
  本线（GQA）：    K/V 缓存按 **n_head_kv** 个 head 分配

  同样 n_head=8、head_dim=128 的模型，MHA 要存 8 份，GQA(8Q/2KV) 只存 2 份，
  **K/V 缓存缩小为 1/4**。这就是 GQA 的主要动机（见 tests/test_kv_cache.py 第 2 条）。

【容量公式】
  单元素 bytes = n_layer × batch × n_head_kv × T × head_dim × 2(K和V) × dtype_size
  例如 n_layer=4, batch=1, n_head_kv=8, T=4096, head_dim=128, fp16：
      4 × 1 × 8 × 4096 × 128 × 2 × 2 = 67,108,864 B = 64 MiB
"""

from typing import List, Tuple

import torch


class KVCache:
    """
    预分配式 KV 缓存。所有层的 K/V 一次性分配到位，append 时原地写入。

    参数：
        n_layer:    层数
        batch_size: batch 大小
        n_head_kv:  KV head 数（GQA 下小于 n_head）
        head_dim:   单个 head 的特征维度
        max_len:    最多缓存多少个 token
        dtype:      缓存张量的 dtype
    """

    def __init__(
        self,
        n_layer: int,
        batch_size: int,
        n_head_kv: int,
        head_dim: int,
        max_len: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        初始化 KVCache。

        参数：
            n_layer:     层数
            batch_size:  batch 大小
            n_head_kv:   KV head 数
            head_dim:    head 特征维度
            max_len:     最大缓存长度
            dtype:       张量 dtype
        """
        self.n_layer = n_layer
        self.batch_size = batch_size
        self.n_head_kv = n_head_kv
        self.head_dim = head_dim
        self.max_len = max_len

        # 预分配两个大张量：(n_layer, B, n_head_kv, max_len, head_dim)
        # 用 zeros 而非 empty：首次 append 前读到的位置应是确定的零值，便于 debug
        shape = (n_layer, batch_size, n_head_kv, max_len, head_dim)
        self.k_buf = torch.zeros(shape, dtype=dtype)
        self.v_buf = torch.zeros(shape, dtype=dtype)

        # 每层独立的已写入长度：list[n_layer]
        self.lengths: List[int] = [0] * n_layer

    # ------------------------------------------------------------------
    # 容量统计
    # ------------------------------------------------------------------
    def nbytes(self) -> int:
        """
        缓存占用的字节数（K + V）。

        返回：
            字节数（int）
        """
        return self.k_buf.numel() * self.k_buf.element_size() * 2

    def describe(self) -> str:
        """
        人类可读的容量描述（打印用）。

        返回：
            形如 "KV Cache: layer=4 batch=1 n_head_kv=2 head_dim=48 max_len=128 -> 48.0 KiB (已用 0/128)"
        """
        used = max(self.lengths) if self.lengths else 0
        return (
            f"KV Cache: layer={self.n_layer} batch={self.batch_size} "
            f"n_head_kv={self.n_head_kv} head_dim={self.head_dim} max_len={self.max_len} "
            f"-> {self.nbytes() / 1024:.1f} KiB (已用 {used}/{self.max_len})"
        )

    # ------------------------------------------------------------------
    # 读写
    # ------------------------------------------------------------------
    def append(self, layer_idx: int, k: "torch.Tensor", v: "torch.Tensor") -> None:
        """
        把当前步（或 prefill 整段）的 K/V 写入指定层的缓存。

        【两种调用形态】
          · prefill: k, v 形状 (B, n_head_kv, T,    head_dim)  —— 一次写入整段
          · decode:  k, v 形状 (B, n_head_kv, 1,    head_dim)  —— 每步写入 1 个

        参数：
            layer_idx: 层索引
            k: (B, n_head_kv, T, head_dim)
            v: (B, n_head_kv, T, head_dim)
        """
        B, H, T, D = k.shape
        if H != self.n_head_kv:
            raise ValueError(f"KV head 数不符：缓存按 {self.n_head_kv} 分配，实得 {H}")
        if B != self.batch_size or D != self.head_dim:
            raise ValueError(
                f"形状不符：期望 (B={self.batch_size}, *, head_dim={self.head_dim})，"
                f"实得 ({B}, {H}, {T}, {D})"
            )
        start = self.lengths[layer_idx]
        end = start + T
        if end > self.max_len:
            raise RuntimeError(
                f"KV Cache 溢出：第 {layer_idx} 层已写入 {start}，本次要写 {T}，"
                f"超过 max_len={self.max_len}"
            )
        # 原地写入预分配张量的 [start:end) 切片
        self.k_buf[layer_idx, :, :, start:end, :] = k.to(self.k_buf.dtype)
        self.v_buf[layer_idx, :, :, start:end, :] = v.to(self.v_buf.dtype)
        self.lengths[layer_idx] = end

    def get(self, layer_idx: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        取出该层已缓存的全部 K/V。

        【形状速查表】
        ┌──────────┬───────────────────────────┬───────────────────────────┐
        │ 阶段     │ 写入 k 的形状            │ 取出的形状                │
        ├──────────┼───────────────────────────┼───────────────────────────┤
        │ prefill  │ (B, n_head_kv, T, D)     │ (B, n_head_kv, T, D)     │
        │ decode   │ (B, n_head_kv, 1, D)     │ (B, n_head_kv, T_i+1, D) │
        └──────────┴───────────────────────────┴───────────────────────────┘
        decode 时 T_q = 1 而 T_kv 随步数增长，这是 Q/K 长度可以不等的原因。

        参数：
            layer_idx: 层索引
        返回：
            (k_full, v_full)，各 (B, n_head_kv, L, head_dim)
        """
        L = self.lengths[layer_idx]
        return (
            self.k_buf[layer_idx, :, :, :L, :],
            self.v_buf[layer_idx, :, :, :L, :],
        )

    def seq_len(self, layer_idx: int) -> int:
        """
        该层当前已缓存的 token 数。

        参数：
            layer_idx: 层索引
        返回：
            序列长度（int）
        """
        return self.lengths[layer_idx]

    def reset(self) -> None:
        """清空所有层的长度计数（张量内容不主动清零，下次写入会覆盖）。"""
        self.lengths = [0] * self.n_layer

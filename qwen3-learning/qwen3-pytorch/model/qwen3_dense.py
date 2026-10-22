"""
Tiny Qwen3 Dense 主模型。

对应笔记：qwen3-learning/Qwen3_训练推理完整学习笔记.md
对照主线：gpt-learning/gpt-learning-pytorch/model/gpt2.py 的 GPT2

【结构】
    wte → N × Pre-LN Block → ln_f → lm_head（与 wte 权重绑定）

    Block = x + Attn(RMSNorm(x))        ← 注意力子层
          → x + SwiGLU(RMSNorm(x))      ← 前馈子层

【相对 GPT-2 主模型的五处差异】
  1) 无 sinusoidal pos_emb：位置信息由 RoPE 编码进 Q/K（注意力内部），不需要加法
  2) LayerNorm → RMSNorm（三个位置：ln_1 / ln_2 / ln_f）
  3) MLP → SwiGLU（三矩阵，中间维按 8/3 规则）
  4) 注意力加入 QK-Norm + GQA
  5) 无 value_head：Qwen3 线走 DPO，不需要 PPO 的 value 头
     → 因此 forward **恒定返回 4 元组**（GPT-2 线是 3 或 4 元组可变）

【为什么用 4 元组而不是像 GPT-2 那样可变】
    MoE 版要返回 aux_loss 与专家负载。用固定 4 元组
    (logits, loss, kv_cache, extras) 后，训练脚本可以统一写
    `aux = extras.get("aux_loss")`，dense 时自动为 None，不需要 if 分支猜结构。
"""

from dataclasses import asdict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Qwen3Config
from model.common.attention import CausalSelfAttention
from model.common.mlp import SwiGLUMLP
from model.common.rmsnorm import RMSNorm


class Block(nn.Module):
    """
    Qwen3 的 Pre-LN Transformer Block。

        x = x + Attn(RMSNorm(x), kv_cache, layer_idx, position_ids)
        x = x + SwiGLU(RMSNorm(x))

    Pre-LN：归一化在子层**之前**，残差路径上不经过任何缩放，梯度能直通，训练更稳。
    """

    def __init__(self, cfg: Qwen3Config, layer_idx: int) -> None:
        """
        初始化 Block。

        参数：
            cfg:       Qwen3 配置对象
            layer_idx: 层索引（MoE 版用于逐层统计，本层仅作标识）
        """
        super().__init__()
        self.layer_idx = layer_idx
        # --- 注意力子层 ---
        self.ln_1 = RMSNorm(cfg.n_embd, eps=cfg.rms_eps)
        self.attn = CausalSelfAttention(
            n_embd=cfg.n_embd,
            n_head=cfg.n_head,
            n_head_kv=cfg.n_head_kv,
            head_dim=cfg.head_dim,
            block_size=cfg.block_size,
            dropout=cfg.dropout,
            bias=cfg.attention_bias,
            rope_theta=cfg.rope_theta,
            rms_eps=cfg.rms_eps,
        )
        # --- 前馈子层 ---
        self.ln_2 = RMSNorm(cfg.n_embd, eps=cfg.rms_eps)
        self.mlp = SwiGLUMLP(
            n_embd=cfg.n_embd,
            intermediate_size=cfg.intermediate_size,
            dropout=cfg.dropout,
            bias=cfg.attention_bias,
        )

    def forward(
        self,
        x: "torch.Tensor",
        kv_cache: Optional[object] = None,
        layer_idx: int = 0,
        position_ids: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", Optional[object]]:
        """
        Pre-LN Block forward。输入输出形状均为 (B, T, n_embd)。

        【形状速查表】
        ┌──────────────────────────┬──────────────────┬──────────────────┐
        │ 步骤                     │ 输入形状         │ 输出形状         │
        ├──────────────────────────┼──────────────────┼──────────────────┤
        │ ln_1                     │ (B, T, C)        │ (B, T, C)        │
        │ attn（RoPE 在内部）      │ (B, T, C)        │ (B, T, C)        │
        │ 残差相加                 │ (B,T,C)×(B,T,C)  │ (B, T, C)        │
        │ ln_2                     │ (B, T, C)        │ (B, T, C)        │
        │ mlp (SwiGLU)            │ (B, T, C)        │ (B, T, C)        │
        │ 残差相加                 │ (B,T,C)×(B,T,C)  │ (B, T, C)        │
        └──────────────────────────┴──────────────────┴──────────────────┘

        参数：
            x:            (B, T, n_embd)
            kv_cache:     KVCache 或 None
            layer_idx:    当前层索引
            position_ids: (B, T) 绝对位置
        返回：
            (x, kv_cache)
        """
        # 第一个残差：注意力
        attn_out, kv_cache = self.attn(
            self.ln_1(x), kv_cache=kv_cache, layer_idx=layer_idx, position_ids=position_ids
        )
        x = x + attn_out
        # 第二个残差：前馈
        x = x + self.mlp(self.ln_2(x))
        return x, kv_cache


class Qwen3Dense(nn.Module):
    """
    Tiny Qwen3 Dense 主模型。

    参数：
        vocab_size: 词表大小
        cfg:        Qwen3 配置对象（默认取 config.DENSE_CONFIG）
        dropout:    覆盖 cfg.dropout（smoke 测试传 0.0）
    """

    def __init__(
        self,
        vocab_size: int,
        cfg: Optional[Qwen3Config] = None,
        dropout: Optional[float] = None,
    ) -> None:
        """
        初始化 Qwen3Dense。

        参数：
            vocab_size: 词表大小
            cfg:        配置对象；None 时用 config.DENSE_CONFIG
            dropout:    dropout 覆盖值；None 时用 cfg.dropout
        """
        super().__init__()
        if cfg is None:
            from config import DENSE_CONFIG
            cfg = DENSE_CONFIG
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.block_size = cfg.block_size

        d = cfg.dropout if dropout is None else dropout
        self.dropout_p = d

        # --- token embedding（唯一的位置无关参数入口）---
        self.wte = nn.Embedding(vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(d)

        # --- N 个 Pre-LN Block ---
        self.blocks = nn.ModuleList(
            [Block(cfg, layer_idx=i) for i in range(cfg.n_layer)]
        )

        # --- final RMSNorm ---
        self.ln_f = RMSNorm(cfg.n_embd, eps=cfg.rms_eps)

        # --- lm_head ---
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size, bias=False)
        if cfg.tie_embeddings:
            # 权重绑定：让 lm_head.weight 与 wte.weight 指向同一张量，
            # 优化器只更新一份参数，embedding 同时充当输出投影
            self.lm_head.weight = self.wte.weight

        # --- 初始化 ---
        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def _init_weights(self, module: nn.Module) -> None:
        """
        初始化：Linear / Embedding 的 weight 用 N(0, 0.02)；Linear 的 bias 用 0。
        RMSNorm 的 gain 保持 1.0（默认即可），不需要干预。

        注：权重绑定时 `lm_head.weight` 与 `wte.weight` 是同一张量，
        apply 会把它初始化两次（都是同分布的 N(0,0.02)），结果等价，无副作用。
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # 主前向
    # ------------------------------------------------------------------
    def forward(
        self,
        idx: "torch.Tensor",
        targets: Optional["torch.Tensor"] = None,
        kv_cache: Optional[object] = None,
        position_ids: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"], Optional[object], dict]:
        """
        主前向计算。

        【形状速查表】（V = vocab_size, C = n_embd）
        ┌──────────────────────────────┬──────────────┬──────────────────┐
        │ 步骤                         │ 输入形状     │ 输出形状         │
        ├──────────────────────────────┼──────────────┼──────────────────┤
        │ 0. idx                       │ -            │ (B, T)           │
        │ 1. wte                       │ (B, T)       │ (B, T, C)        │
        │ 2. drop                      │ (B, T, C)    │ (B, T, C)        │
        │ 3. N × Block                 │ (B, T, C)    │ (B, T, C)        │
        │ 4. ln_f                      │ (B, T, C)    │ (B, T, C)        │
        │ 5. lm_head [tied]            │ (B, T, C)    │ (B, T, V)        │
        │ 6. cross_entropy [可选]      │ (B*T,V)/(B*T)│ scalar           │
        └──────────────────────────────┴──────────────┴──────────────────┘

        【注意】没有 +pos_emb 这一步 —— 位置信息由 attention 内部的 RoPE 承担。

        【与 GPT2.forward 的差异】
          - **恒定返回 4 元组**：GPT-2 无 value_head 时返回 3 元组，本模型恒 4 元组
          - 第四项是 extras（dense 为空 dict），不是 value
          - 无 value_head，故也不需要 attach_value_head()

        参数：
            idx:         (B, T) LongTensor，token id
            targets:     (B, T) LongTensor，-100 表示忽略该位
            kv_cache:    KVCache 或 None
            position_ids:(B, T) LongTensor 绝对位置；kv_cache 不为 None 时必须显式给
        返回：
            (logits, loss, new_kv_cache, extras)
        """
        B, T = idx.size()
        if T > self.block_size:
            raise ValueError(
                f"输入序列长度 T={T} 超过 block_size={self.block_size}，"
                "训练/prefill 必须先截断。"
            )

        # --- 0) position_ids 默认 0..T-1 ---
        if position_ids is None:
            position_ids = torch.arange(T, dtype=torch.long, device=idx.device)
            position_ids = position_ids.unsqueeze(0).expand(B, T)

        # --- 1) token embedding（**不加位置编码**，位置由 RoPE 在 attention 内处理）---
        x = self.drop(self.wte(idx))          # (B, T, C)

        # --- 2) N × Pre-LN Block ---
        new_kv = kv_cache
        for layer_idx, block in enumerate(self.blocks):
            x, new_kv = block(x, kv_cache=new_kv, layer_idx=layer_idx,
                              position_ids=position_ids)

        # --- 3) final RMSNorm ---
        x = self.ln_f(x)                       # (B, T, C)

        # --- 4) lm_head（权重与 wte 绑定）---
        logits = self.lm_head(x)               # (B, T, V)

        # --- 5) 交叉熵损失（-100 忽略）---
        loss: Optional["torch.Tensor"] = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        # --- 6) dense 无额外信息，extras 为空 dict ---
        return logits, loss, new_kv, {}

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def num_params(self, non_embedding: bool = False) -> int:
        """
        统计可训练参数量。

        参数：
            non_embedding: True 时不计入 token embedding 矩阵
        返回：
            参数数量（int）
        """
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            n -= self.wte.weight.numel()
        return n

    def config_dict(self) -> dict:
        """返回当前配置的 dict 快照（存 checkpoint 用）。"""
        return asdict(self.cfg)

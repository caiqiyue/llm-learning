"""
Tiny Qwen3 MoE 主模型（Router + 多专家 SwiGLU）。

对应笔记：qwen3-learning/Qwen3_MoE_学习笔记.md
对照主线：model/qwen3_dense.py（**主干完全一致，只有 FFN 不同**）

【核心思想】
    Dense Block 的 FFN：每个 token 都过同一个 SwiGLU(x)
    MoE    Block 的 FFN：每个 token 只过它被路由到的 top-k 个 SwiGLU

    MoE 的收益：把 FFN 的**参数量**放大 N 倍（模型容量更大），
                但每个 token 的**计算量**只增加 k 倍（k ≪ N）。
                Qwen3-30B-A3B 就是 A3B —— 每 token 激活约 3B 参数。

【稀疏路由怎么算】
    router(x) → n_experts 个 logits → softmax → 取 top-k → （可选）权重重新归一化
    → 只把这 k 个专家的 FFN 作用于该 token → 按归一化权重加权求和

【负载均衡辅助损失（HuggingFace Mixtral 的写法）】
    aux_loss = n_experts · Σ_i  f_i · P_i
        f_i = 专家 i 被选中的 (token, k槽) 对数 / (总 token 数 × k)
              —— 也就是对 tokens×k 个槽位求均值
        P_i = router 对专家 i 给出的平均概率

    均匀路由时：f_i = 1/(K·E)，P_i = 1/E
    → aux_loss = E · Σ_i (1/(K·E))·(1/E) = **1/K**（K=2 时是 0.5，**不是 1**）
    某个专家被过度使用则 f_i ≫ P_i，aux_loss 上升，推动 router 重新分配。

    （Switch Transformer 论文里的原始式子 f_i = 「被路由到专家 i 的 token 比例」
      在 k>1 时每个 token 会被计多次，与 HF Mixtral 的实现不是同一个量纲。
      本实现采用 HF Mixtral 的量纲，训练脚本里的 aux_loss_coef 与之配套。）

【实现要点：真的省计算，而不是"算了再丢"】
    朴素写法会算全部 n_experts 个 FFN 再按 top-k 取用，等于没省。
    本实现按 token 分组：把同一个 token 的 top-k 专家算完后立刻散回原位。
    tests/test_moe.py 第 9 条用 forward hook 统计每个专家实际处理了多少行，
    断言总行数 = N·K（而不是 N·E），以此证明稀疏性是真的。
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Qwen3Config
from model.common.attention import CausalSelfAttention
from model.common.mlp import SwiGLUMLP
from model.common.rmsnorm import RMSNorm


class SparseMoeBlock(nn.Module):
    """
    一个 MoE 前馈块：Router + n_experts 个 SwiGLU 专家。

    参数：
        n_embd:             残差流宽度
        n_experts:          专家数量
        experts_per_tok:    top-k 路由的 k
        moe_intermediate_size: 每个专家的 FFN 中间维
        norm_topk_prob:     是否对 top-k 权重重新归一化
        dropout:            dropout 比例
        bias:               投影层是否用 bias
    """

    def __init__(
        self,
        n_embd: int,
        n_experts: int,
        experts_per_tok: int,
        moe_intermediate_size: int,
        norm_topk_prob: bool = True,
        dropout: float = 0.1,
        bias: bool = False,
    ) -> None:
        """
        初始化 SparseMoeBlock。参数说明见类 docstring。
        """
        super().__init__()
        if experts_per_tok > n_experts:
            raise ValueError(f"experts_per_tok({experts_per_tok}) 不能超过 n_experts({n_experts})")

        self.n_embd = n_embd
        self.n_experts = n_experts
        self.experts_per_tok = experts_per_tok
        self.norm_topk_prob = norm_topk_prob

        # Router：一个无 bias 的线性层，n_embd → n_experts
        self.router = nn.Linear(n_embd, n_experts, bias=bias)

        # n_experts 个独立的 SwiGLU 专家
        self.experts = nn.ModuleList(
            [
                SwiGLUMLP(n_embd, moe_intermediate_size, dropout=dropout, bias=bias)
                for _ in range(n_experts)
            ]
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        MoE FFN forward（只做前向，不返回 aux_loss —— 见 _forward_ffn）。

        参数：
            x: (B, T, n_embd)
        返回：
            (B, T, n_embd)
        """
        out, _ = self._forward_ffn(x)
        return out

    def _forward_ffn(self, x: "torch.Tensor") -> Tuple["torch.Tensor", dict]:
        """
        MoE FFN 的完整前向，同时返回路由统计与负载均衡辅助损失。

        【形状速查表】（E = n_experts, K = experts_per_tok, N = B*T）
        ┌────────────────────────────────┬──────────────────┬──────────────────────┐
        │ 步骤                           │ 输入形状         │ 输出形状             │
        ├────────────────────────────────┼──────────────────┼──────────────────────┤
        │ 输入 x                         │ -                │ (B, T, C)            │
        │ 1. router                      │ (N, C)           │ (N, E)               │
        │ 2. softmax                     │ (N, E)           │ (N, E)  ← 这是 P_i   │
        │ 3. topk                         │ (N, E)           │ w:(N,K) idx:(N,K)    │
        │ 4. norm_topk_prob（可选）      │ w (N, K)         │ w (N, K)，行和为1    │
        │ 5. gather token                │ (N, C)           │ (N*K, C)             │
        │ 6. 逐专家 FFN（只过被选的）    │ (N_i, C)         │ (N_i, C)             │
        │ 7. 加权散回原位                │ -                │ (N, C)               │
        │ 8. reshape                      │ (N, C)           │ (B, T, C)            │
        └────────────────────────────────┴──────────────────┴──────────────────────┘

        【为什么第 6 步是"真稀疏"】
            先按专家分组（sort → 每组一段连续切片），
            每个专家只处理真正路由到它的那些 (token, k槽)，
            总计算量 = N·K·moe_intermediate，而不是 N·E·moe_intermediate。

        参数：
            x: (B, T, n_embd)
        返回：
            (out, stats)，out 为 (B, T, n_embd)；
            stats 含 aux_loss / expert_load / topk_weights / routed_tokens /
            expert_token_pairs
        """
        B, T, C = x.size()
        x_flat = x.view(-1, C)              # (N, C)
        n_tokens = x_flat.size(0)           # N = B*T
        E = self.n_experts
        K = self.experts_per_tok

        # --- 1) 路由：算每个专家的概率 ---
        router_logits = self.router(x_flat)              # (N, E)
        router_probs = F.softmax(router_logits, dim=-1)  # (N, E) ← P_i 的来源

        # --- 2) top-k 选专家 ---
        topk_weights, topk_indices = torch.topk(router_probs, K, dim=-1)   # (N,K)
        # topk_indices: (N, K) 每个 token 选中的专家 id

        # --- 3) 可选：把 top-k 权重重新归一化到和为 1 ---
        #     不归一化时权重来自全局 softmax，和 < 1，会整体缩小 FFN 输出
        if self.norm_topk_prob and K > 1:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # --- 4) 负载均衡辅助损失（HF Mixtral 量纲）---
        #     f_i = 专家 i 的 (token, k槽) 对数 / (N·K)  —— 对 tokens×k 个槽位求均值
        #     P_i = 专家 i 的平均路由概率
        #     aux_loss = E · Σ_i f_i · P_i，均匀时 = 1/K
        expert_mask = F.one_hot(topk_indices, num_classes=E).float()      # (N, K, E)
        expert_mask = expert_mask.view(n_tokens * K, E)                   # (N*K, E)
        f_i = expert_mask.mean(dim=0)                                     # (E,) 实际占比
        P_i = router_probs.mean(dim=0)                                    # (E,) 平均概率
        aux_loss = E * torch.sum(f_i * P_i)                               # scalar

        # 统计：每个专家分到多少 token（把 top-k 的每个 slot 都算一次）
        expert_load = (expert_mask.sum(dim=0) / K).tolist()               # list[E]，和 = N

        # --- 5) 按专家分组做稀疏计算 ---
        #     对第 e 个专家，取出「被路由到它」的 (token 下标, k 槽下标)
        out_flat = torch.zeros_like(x_flat)              # (N, C) 先铺零再散回
        pair_expert = topk_indices.view(-1)              # (N*K,) 每个 (token,k) 对应的专家
        # sort 后同一专家的样本连续，于是可以切片批量计算
        sorted_experts, sort_indices = torch.sort(pair_expert)
        token_idx = sort_indices // K                    # 该样本属于哪个 token
        slot_idx = sort_indices % K                      # 该样本是该 token 的第几个槽
        # gather 出被路由样本的 token 向量：(N*K, C)
        gathered = x_flat[token_idx]                      # (N*K, C)

        # 逐专家处理（E 很小，Python 循环的常数开销可接受；生产实现会用 index_add）
        counts = torch.bincount(sorted_experts, minlength=E).tolist()
        start = 0
        for e in range(E):
            n_e = counts[e]
            if n_e == 0:
                continue  # 该专家这一步没被用到，跳过（这正是稀疏计算省下来的部分）
            end = start + n_e
            seg = gathered[start:end]                    # (n_e, C) 本专家要处理的样本
            expert_out = self.experts[e](seg)            # (n_e, C)
            # 按该专家分到的 top-k 权重缩放
            w = topk_weights.view(-1)[sort_indices[start:end]].unsqueeze(-1)   # (n_e, 1)
            expert_out = expert_out * w
            # 散回原 token 位置并累加（同一 token 的 K 个专家输出在这里相加）
            out_flat.index_add_(0, token_idx[start:end], expert_out)
            start = end

        stats = {
            "aux_loss": aux_loss,
            "expert_load": expert_load,
            "topk_weights": topk_weights.view(B * T, K).detach(),
            "routed_tokens": n_tokens,
            "expert_token_pairs": n_tokens * K,
        }
        return out_flat.view(B, T, C), stats


class MoEBlock(nn.Module):
    """
    Qwen3 MoE 的 Pre-LN Block。与 Qwen3Dense.Block 结构相同，
    只是第二个子层（FFN）换成 SparseMoeBlock。
    """

    def __init__(self, cfg: Qwen3Config) -> None:
        """
        初始化 MoEBlock。

        参数：
            cfg: Qwen3 配置对象（应为 MOE_CONFIG）
        """
        super().__init__()
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
        self.ln_2 = RMSNorm(cfg.n_embd, eps=cfg.rms_eps)
        self.moe = SparseMoeBlock(
            n_embd=cfg.n_embd,
            n_experts=cfg.n_experts,
            experts_per_tok=cfg.experts_per_tok,
            moe_intermediate_size=cfg.moe_intermediate_size,
            norm_topk_prob=cfg.norm_topk_prob,
            dropout=cfg.dropout,
            bias=cfg.attention_bias,
        )
        self._last_stats: Optional[dict] = None

    def forward(
        self,
        x: "torch.Tensor",
        kv_cache: Optional[object] = None,
        layer_idx: int = 0,
        position_ids: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", Optional[object]]:
        """
        MoE Block forward。输入输出形状均为 (B, T, n_embd)。

        与 Dense Block 的差别只有一处：第二子层走 self.moe 而非 self.mlp，
        并把路由统计挂到 self._last_stats 供 Qwen3MoE.forward 汇总。

        参数 / 返回：与 Qwen3Dense.Block.forward 一致
        """
        attn_out, kv_cache = self.attn(
            self.ln_1(x), kv_cache=kv_cache, layer_idx=layer_idx, position_ids=position_ids
        )
        x = x + attn_out
        moe_out, stats = self.moe._forward_ffn(self.ln_2(x))
        self._last_stats = stats
        x = x + moe_out
        return x, kv_cache


class Qwen3MoE(nn.Module):
    """
    Tiny Qwen3 MoE 主模型。

    主干（embedding / attention / RMSNorm）与 Qwen3Dense **逐行一致**，
    唯一差别是 Block 的 FFN 从 SwiGLUMLP 换成 SparseMoeBlock。
    对应设计文档 §5.8：forward 同样返回恒定 4 元组，
    差别只在 extras 里多了 aux_loss 与 expert_load。
    """

    def __init__(
        self,
        vocab_size: int,
        cfg: Optional[Qwen3Config] = None,
        dropout: Optional[float] = None,
    ) -> None:
        """
        初始化 Qwen3MoE。

        参数：
            vocab_size: 词表大小
            cfg:        配置对象；None 时用 config.MOE_CONFIG
            dropout:    dropout 覆盖值
        """
        super().__init__()
        if cfg is None:
            from config import MOE_CONFIG
            cfg = MOE_CONFIG
        if cfg.n_experts is None:
            raise ValueError("MoE 配置缺少 n_experts，请用 config.MOE_CONFIG")

        self.cfg = cfg
        self.vocab_size = vocab_size
        self.block_size = cfg.block_size
        d = cfg.dropout if dropout is None else dropout
        self.dropout_p = d

        # --- 主干：与 Qwen3Dense 完全相同 ---
        self.wte = nn.Embedding(vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(d)

        # --- N 个 MoE Block ---
        self.blocks = nn.ModuleList([MoEBlock(cfg) for _ in range(cfg.n_layer)])

        self.ln_f = RMSNorm(cfg.n_embd, eps=cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """初始化：Linear / Embedding 的 weight 用 N(0, 0.02)，bias 用 0。"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: "torch.Tensor",
        targets: Optional["torch.Tensor"] = None,
        kv_cache: Optional[object] = None,
        position_ids: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"], Optional[object], dict]:
        """
        主前向。签名与返回结构与 Qwen3Dense.forward **完全一致**。

        【与 dense 的唯一差异】
            extras 里多了两个键：
                "aux_loss":    负载均衡辅助损失（scalar tensor，多层取平均）
                "expert_load": 逐层的专家负载（list of list，每层元素和 = B*T）
                "expert_load_last": 最后一层的专家负载（list，元素和 = B*T）

        参数 / 返回：与 Qwen3Dense.forward 一致
        """
        B, T = idx.size()
        if T > self.block_size:
            raise ValueError(f"输入序列长度 T={T} 超过 block_size={self.block_size}")

        if position_ids is None:
            position_ids = torch.arange(T, dtype=torch.long, device=idx.device)
            position_ids = position_ids.unsqueeze(0).expand(B, T)

        x = self.drop(self.wte(idx))

        # --- N × MoE Block：累积每层的 aux_loss ---
        new_kv = kv_cache
        aux_losses: List[torch.Tensor] = []
        expert_loads: List[List[float]] = []
        for layer_idx, block in enumerate(self.blocks):
            x, new_kv = block(x, kv_cache=new_kv, layer_idx=layer_idx,
                              position_ids=position_ids)
            # 每一层把自己的路由统计挂在 block._last_stats 上（见 MoEBlock.forward）
            st = getattr(block, "_last_stats", None)
            if st is not None:
                aux_losses.append(st["aux_loss"])
                expert_loads.append(st["expert_load"])

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss: Optional[torch.Tensor] = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        extras: dict = {}
        if aux_losses:
            # 多层时取平均：每层的负载均衡同等重要
            extras["aux_loss"] = torch.stack(aux_losses).mean()
            # 逐层负载也一并返回，便于打印诊断
            extras["expert_load"] = expert_loads
            extras["expert_load_last"] = expert_loads[-1]

        return logits, loss, new_kv, extras

    def num_params(self, non_embedding: bool = False) -> int:
        """统计可训练参数量（non_embedding=True 时排除 token embedding）。"""
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            n -= self.wte.weight.numel()
        return n

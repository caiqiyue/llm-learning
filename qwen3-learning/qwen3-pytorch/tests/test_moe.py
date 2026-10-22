"""
Qwen3MoE smoke（对应 feat-021）。

对应笔记：qwen3-learning/Qwen3_MoE_学习笔记.md

运行：python tests/test_moe.py
"""

import os
import sys

import torch

# 直接执行 `python tests/test_x.py` 时 sys.path[0] 是 tests/ 目录，
# 找不到项目根下的 config.py / model 包 —— 这里把项目根补进 sys.path。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DENSE_CONFIG, MOE_CONFIG
from model.qwen3_dense import Qwen3Dense
from model.qwen3_moe import MoEBlock, Qwen3MoE, SparseMoeBlock

torch.manual_seed(DENSE_CONFIG.seed)
VOCAB = 512
B, T = 4, 16

model = Qwen3MoE(vocab_size=VOCAB, cfg=MOE_CONFIG, dropout=0.0)
model.eval()

# =====================================================================
# 1) 接口与 dense 完全一致（这是设计文档 §5.8 的核心约定）
# =====================================================================
idx = torch.randint(0, VOCAB, (B, T))
logits, loss, cache, extras = model(idx)
assert logits.shape == (B, T, VOCAB), f"logits 形状错: {tuple(logits.shape)}"
assert isinstance(extras, dict), f"extras 必须是 dict，实得 {type(extras)}"
assert "aux_loss" in extras, "MoE 的 extras 必须含 aux_loss"
assert "expert_load" in extras, "MoE 的 extras 必须含 expert_load"
assert extras["aux_loss"] is not None, "aux_loss 不应为 None"

# dense 的 extras 是空 dict —— 训练脚本靠 extras.get("aux_loss") 统一处理两种模型
dense_model = Qwen3Dense(vocab_size=VOCAB, cfg=DENSE_CONFIG, dropout=0.0)
dense_model.eval()
_, _, _, dense_extras = dense_model(idx)
assert dense_extras == {}, f"dense extras 应为空，实得 {dense_extras}"
assert dense_extras.get("aux_loss") is None, "dense 用 .get 应拿到 None，训练脚本无需 if 分支"

# =====================================================================
# 2) 主干与 Dense 完全一致：只有 FFN 不同
# =====================================================================
moe_block = model.blocks[0]
# 注意层次：model.blocks[i] 是 MoEBlock（Pre-LN 容器），
# 里面 .attn 是注意力子层，.moe 才是 SparseMoeBlock（专家集合）
assert isinstance(moe_block, MoEBlock), f"blocks[0] 应为 MoEBlock，实得 {type(moe_block)}"
assert isinstance(moe_block.moe, SparseMoeBlock), f"block.moe 应为 SparseMoeBlock，实得 {type(moe_block.moe)}"
assert isinstance(moe_block.ln_1.gain, torch.nn.Parameter), "ln_1 应是 RMSNorm"
assert moe_block.attn.n_head == DENSE_CONFIG.n_head, "MoE 的注意力应与 Dense 相同"
assert moe_block.attn.n_head_kv == DENSE_CONFIG.n_head_kv, "MoE 的 GQA 配置应与 Dense 相同"
assert moe_block.attn.head_dim == DENSE_CONFIG.head_dim, "MoE 的 head_dim 应与 Dense 相同"

# =====================================================================
# 3) 稀疏路由：只过 top-k 个专家
# =====================================================================
ffn = moe_block.moe          # SparseMoeBlock
ffn.eval()
h = torch.randn(B, T, DENSE_CONFIG.n_embd)
with torch.no_grad():
    out, stats = ffn._forward_ffn(h)
assert out.shape == h.shape, f"MoE FFN 输出应回到 n_embd，实得 {tuple(out.shape)}"
assert stats["routed_tokens"] == B * T, f"应路由 B*T={B*T} 个 token，实得 {stats['routed_tokens']}"
# 稀疏性：top-k=2 时实际参与计算的 token-专家对是 B*T*2，而非 B*T*4
assert stats["expert_token_pairs"] == B * T * MOE_CONFIG.experts_per_tok, \
    f"参与计算的 token-专家对应为 {B*T*MOE_CONFIG.experts_per_tok}，实得 {stats['expert_token_pairs']}"
# 负载均衡损失为有限正数
assert torch.isfinite(stats["aux_loss"]), "aux_loss 非有限"
assert stats["aux_loss"].item() > 0, "aux_loss 应为正数"

# =====================================================================
# 4) 专家负载分布：无专家饿死
# =====================================================================
load = stats["expert_load"]
assert len(load) == MOE_CONFIG.n_experts, f"负载列表长度应为 {MOE_CONFIG.n_experts}"
total = sum(load)
assert abs(total - B * T) < 1e-3, f"负载总和应为 token 总数 {B*T}，实得 {total}"
assert min(load) > 0, f"存在专家未被路由到（饿死），负载分布={load}"
ideal = B * T / MOE_CONFIG.n_experts
print(f"[专家负载] {['%.1f' % x for x in load]}  理想={ideal:.1f}  "
      f"最大偏差={max(abs(x-ideal) for x in load)/ideal:.1%}")

# =====================================================================
# 5) norm_topk_prob：top-k 权重归一化后和为 1
# =====================================================================
assert MOE_CONFIG.norm_topk_prob is True, "本配置应开启 norm_topk_prob"
ffn.norm_topk_prob = True
with torch.no_grad():
    _, stats2 = ffn._forward_ffn(h)
w = stats2["topk_weights"]
assert torch.allclose(w.sum(dim=-1), torch.ones_like(w.sum(dim=-1)), atol=1e-5), \
    "norm_topk_prob 打开时 top-k 权重和应为 1"
ffn.norm_topk_prob = False
with torch.no_grad():
    _, stats3 = ffn._forward_ffn(h)
assert not torch.allclose(stats3["topk_weights"].sum(dim=-1), torch.ones_like(w.sum(dim=-1)), atol=1e-3), \
    "norm_topk_prob 关闭时权重和不应为 1"
ffn.norm_topk_prob = MOE_CONFIG.norm_topk_prob

# =====================================================================
# 6) aux_loss 的梯度能回传到每个 router
# =====================================================================
model.train()
logits, loss, _, extras = model(idx, targets=torch.randint(0, VOCAB, (B, T)))
assert "aux_loss" in extras, "给 targets 时 extras 也应含 aux_loss"
model.zero_grad()
extras["aux_loss"].backward(retain_graph=True)
router_grads = [
    blk.moe.router.weight.grad for blk in model.blocks if blk.moe.router.weight.grad is not None
]
assert len(router_grads) == MOE_CONFIG.n_layer, "每个 MoE block 的 router 都应收到 aux_loss 的梯度"
assert all(torch.isfinite(g).all() for g in router_grads), "router 梯度非有限"

# =====================================================================
# 7) 因果性与确定性
# =====================================================================
model.eval()
with torch.no_grad():
    l1 = model(idx)[0]
    l2 = model(idx)[0]
assert torch.equal(l1, l2), "eval 模式下两次前向应完全一致"

idx2 = idx.clone()
idx2[:, T // 2:] = torch.randint(0, VOCAB, (B, T - T // 2))
with torch.no_grad():
    a = model(idx)[0]
    b = model(idx2)[0]
assert torch.allclose(a[:, :T // 2, :], b[:, :T // 2, :], atol=1e-5), "MoE 模型级因果掩码失效"

# =====================================================================
# 8) 专家数量与结构正确
# =====================================================================
assert len(ffn.experts) == MOE_CONFIG.n_experts, f"专家数应为 {MOE_CONFIG.n_experts}"
assert ffn.experts[0].gate_proj.out_features == MOE_CONFIG.moe_intermediate_size
assert ffn.router.out_features == MOE_CONFIG.n_experts, "router 输出维度应为专家数"
assert ffn.router.bias is None, "router 不应有 bias"

# =====================================================================
# 9) 【核心】稀疏是真的：专家实际处理的行数 = N·K，而不是 N·E
# =====================================================================
# 用 forward hook 统计每个专家一次调用处理了多少行。
# 若实现是「算了再丢」（全算 E 个专家再取 top-k），这里会是 N·E。
rows = []
handles = []
for e in ffn.experts:
    def _hook(mod, inp, out, _rows=rows):
        _rows.append(inp[0].shape[0])
    handles.append(e.register_forward_hook(_hook))
with torch.no_grad():
    ffn._forward_ffn(h)
for hd in handles:
    hd.remove()

total_rows = sum(rows)
N = B * T
K = MOE_CONFIG.experts_per_tok
E = MOE_CONFIG.n_experts
assert len(rows) == E, f"每个专家应各被调用一次，实得 {len(rows)} 次"
assert total_rows == N * K, \
    f"专家实际处理总行数应为 N*K={N*K}，实得 {total_rows}（若为 {N*E} 说明没省计算）"
assert total_rows < N * E, "总行数应严格小于「全算」的 N*E"
print(f"[稀疏性] 专家处理总行数={total_rows} (=N*K={N}*{K})，全算需 {N*E}，"
      f"省下 {(1 - total_rows / (N * E)):.0%}")

# =====================================================================
# 10) MoE 与 Dense 的参数量对比
# =====================================================================
n_dense = dense_model.num_params()
n_moe = model.num_params()
print(f"[参数量] dense={n_dense:,}  moe={n_moe:,}  比值={n_moe/n_dense:.2f}x")
assert n_moe > n_dense, "MoE 的 FFN 专家更多，总参数量应大于 Dense"

print(f"test_moe OK  n_experts={MOE_CONFIG.n_experts} top_k={MOE_CONFIG.experts_per_tok} "
      f"moe_intermediate={MOE_CONFIG.moe_intermediate_size} params={n_moe:,}")

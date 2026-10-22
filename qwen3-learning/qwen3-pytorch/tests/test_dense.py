"""
Qwen3Dense 完整前向 smoke（对应 feat-016）。

运行：python tests/test_dense.py
"""

import os
import sys

import torch

# 直接执行 `python tests/test_x.py` 时 sys.path[0] 是 tests/ 目录，
# 找不到项目根下的 config.py / model 包 —— 这里把项目根补进 sys.path。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DENSE_CONFIG as C
from model.common.kv_cache import KVCache
from model.qwen3_dense import Block, Qwen3Dense

torch.manual_seed(C.seed)
VOCAB = 512
B, T = 2, 16

model = Qwen3Dense(vocab_size=VOCAB, cfg=C, dropout=0.0)
model.eval()

# --- 1) 基础前向：形状与返回结构 ---
idx = torch.randint(0, VOCAB, (B, T))
logits, loss, cache, extras = model(idx)
assert logits.shape == (B, T, VOCAB), f"logits 形状错: {tuple(logits.shape)}"
assert loss is None, "未给 targets 时 loss 应为 None"
assert cache is None, "未给 kv_cache 时应返回 None"
assert extras == {}, f"dense 的 extras 应为空 dict，实得 {extras}"

# --- 2) 给 targets 时 loss 有限 ---
targets = torch.randint(0, VOCAB, (B, T))
logits, loss, _, _ = model(idx, targets=targets)
assert loss is not None and torch.isfinite(loss), f"loss 异常: {loss}"
# loss 应接近 ln(vocab)（随机初始化时）
ln_vocab = torch.log(torch.tensor(float(VOCAB))).item()
assert abs(loss.item() - ln_vocab) < 2.0, \
    f"随机初始化时 loss 应接近 ln({VOCAB})={ln_vocab:.2f}，实得 {loss.item():.2f}"

# --- 3) 权重绑定：lm_head.weight 与 wte.weight 是同一个张量 ---
assert model.lm_head.weight is model.wte.weight, "tie_embeddings 应让两者共享同一对象"
# 绑定后参数只算一份：parameters() 对同一张量去重
n_unique = len({id(p) for p in model.parameters()})
assert n_unique == len(list(model.parameters())), "绑定后不应有重复参数对象"

# --- 4) 无 pos_emb：Qwen3 用 RoPE，不需要加法式位置编码 ---
assert not hasattr(model, "pos_emb"), "Qwen3 用 RoPE，不应有 sinusoidal pos_emb buffer"
assert not any("pos_emb" in n for n, _ in model.named_parameters()), "不应有位置嵌入参数"

# --- 5) 层数与结构 ---
assert len(model.blocks) == C.n_layer, f"层数错: {len(model.blocks)}"
blk = model.blocks[0]
assert isinstance(blk.ln_1, type(blk.attn.q_norm)), "ln_1 应是 RMSNorm"
assert isinstance(blk.ln_2, type(blk.attn.q_norm)), "ln_2 应是 RMSNorm"
assert isinstance(blk.mlp, type(model.blocks[0].mlp)), "mlp 应是 SwiGLUMLP"
assert isinstance(model.ln_f, type(blk.ln_1)), "ln_f 应是 RMSNorm"
assert isinstance(blk, Block), "blocks 元素应是 Block"

# --- 6) Pre-LN：ln_1 的输入维度是 n_embd ---
assert blk.ln_1.dim == C.n_embd and blk.ln_2.dim == C.n_embd

# --- 7) 参数量统计自洽 ---
total = sum(p.numel() for p in model.parameters())
assert model.num_params() == total, f"num_params({model.num_params()}) 与实际({total})不符"
# embedding 占比很高（tiny 规模正常）
n_emb_params = model.wte.weight.numel()
print(f"[参数量] total={total:,} embedding={n_emb_params:,} "
      f"({n_emb_params/total:.1%}) 非embedding={model.num_params(True):,}")

# --- 8) 因果性：模型级也不泄漏未来 ---
x2 = idx.clone()
x2[:, T // 2:] = torch.randint(0, VOCAB, (B, T - T // 2))
with torch.no_grad():
    l1 = model(idx)[0]
    l2 = model(x2)[0]
assert torch.allclose(l1[:, :T // 2, :], l2[:, :T // 2, :], atol=1e-5), "模型级因果掩码失效"

# --- 9) 超出 block_size 报错 ---
try:
    model(torch.randint(0, VOCAB, (1, C.block_size + 1)))
    raise SystemExit("超过 block_size 应报错但未报")
except ValueError:
    pass

# --- 10) 确定性 ---
with torch.no_grad():
    a = model(idx)[0]
    b = model(idx)[0]
assert torch.equal(a, b), "eval 模式下两次前向应完全一致"

# --- 11) 模型级 KV Cache：prefill + 逐 token decode == 一次性全量 forward ---
cache = KVCache(n_layer=C.n_layer, batch_size=B, n_head_kv=C.n_head_kv,
                head_dim=C.head_dim, max_len=T + 4)
P = 5
with torch.no_grad():
    l_pre, _, _, _ = model(idx[:, :P], kv_cache=cache)
    assert torch.allclose(l_pre, l1[:, :P, :], atol=1e-5), "prefill 与全量 forward 前段不一致"
    for step in range(P, T):
        pos = torch.full((B, 1), step, dtype=torch.long)
        l_step, _, _, _ = model(idx[:, step:step + 1], kv_cache=cache, position_ids=pos)
        assert torch.allclose(l_step, l1[:, step:step + 1, :], atol=1e-5), \
            f"decode 第 {step} 步与全量 forward 不一致"
assert cache.seq_len(0) == T, f"每层 cache 长度应为 {T}，实得 {cache.seq_len(0)}"

# --- 12) 梯度能从 lm_head 回流到 wte（权重绑定生效）---
model.zero_grad()
_, loss_g, _, _ = model(idx, targets=targets)
loss_g.backward()
assert model.wte.weight.grad is not None, "wte 没有梯度"
assert model.wte.weight.grad.abs().sum().item() > 0, "wte 梯度全零"

# --- 13) dropout 覆盖真的传到每一处（feat-024 修的 bug 的回归断言）---
#     历史 bug：Qwen3Dense 只把覆盖值给 embedding 的 Dropout，
#     Block 里的注意力/MLP 一直用 cfg.dropout ⇒ `dropout=0.0` 关不掉它们。
#     下面两条断言（结构 + 行为）任缺其一都抓不住那个 bug。
model_no_drop = Qwen3Dense(vocab_size=VOCAB, cfg=C, dropout=0.0)
ps = sorted({m.p for m in model_no_drop.modules() if isinstance(m, torch.nn.Dropout)})
assert ps == [0.0], f"dropout=0.0 时所有 Dropout 的 p 都应是 0.0，实得 {ps}"
# 行为断言：dropout=0 ⇒ train 模式下两次前向必须逐元素相同
model_no_drop.train()
with torch.no_grad():
    t1 = model_no_drop(idx)[0]
    t2 = model_no_drop(idx)[0]
assert torch.equal(t1, t2), "dropout=0.0 时 train 模式的两次前向应完全一致"
# 反向：不给覆盖值时应当用 cfg.dropout
model_cfg_drop = Qwen3Dense(vocab_size=VOCAB, cfg=C)
ps_cfg = sorted({m.p for m in model_cfg_drop.modules() if isinstance(m, torch.nn.Dropout)})
assert ps_cfg == [C.dropout], f"不给覆盖值时所有 Dropout 的 p 都应是 cfg.dropout={C.dropout}，实得 {ps_cfg}"

print(f"test_dense OK  params={total:,} layers={C.n_layer} "
      f"n_head={C.n_head} n_head_kv={C.n_head_kv} head_dim={C.head_dim}")

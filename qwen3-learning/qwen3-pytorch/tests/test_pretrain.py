"""
预训练脚本 smoke：模型能建起来、能走一步训练、初始 loss 合理。

对应 feat-020（开发计划 Task 13）。

运行：python tests/test_pretrain.py
"""

import math
import os
import sys

import torch

# 直接执行 `python tests/test_x.py` 时 sys.path[0] 是 tests/ 目录，
# 找不到项目根下的 config.py / model 包 —— 这里把项目根补进 sys.path。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DENSE_CONFIG, MOE_CONFIG
from model.qwen3_dense import Qwen3Dense
from model.qwen3_moe import Qwen3MoE
from train.train_pretrain import build_model, lr_at

torch.manual_seed(DENSE_CONFIG.seed)
V = 256
model = Qwen3Dense(vocab_size=V, cfg=DENSE_CONFIG, dropout=0.0)

x = torch.randint(0, V, (2, 8))
y = torch.randint(0, V, (2, 8))

logits, loss, cache, extras = model(x, targets=y)
assert logits.shape == (2, 8, V), f"logits 形状错: {tuple(logits.shape)}"
assert loss is not None and torch.isfinite(loss), f"loss 异常: {loss}"
assert cache is None, "训练时不应有 kv_cache"
assert extras == {}, f"dense 的 extras 应为空 dict，实得 {extras}"

# loss 应接近 ln(V) —— 随机初始化时模型对每个 token 都是均匀分布
assert abs(loss.item() - math.log(V)) < 1.0, \
    f"初始 loss 应接近 ln(V)={math.log(V):.2f}，实得 {loss.item():.3f}"

# 反向传播能跑通
loss.backward()
grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
assert len(grads) > 0, "没有任何参数收到梯度"

# =====================================================================
# 学习率调度：warmup → cosine → 恒定 min_lr
# =====================================================================
cfg = DENSE_CONFIG
assert lr_at(0, cfg) == cfg.learning_rate * 1 / cfg.warmup_iters, "warmup 第一步不对"
assert lr_at(cfg.warmup_iters - 1, cfg) == cfg.learning_rate, "warmup 结束应等于 learning_rate"
assert abs(lr_at(cfg.warmup_iters, cfg) - cfg.learning_rate) < 1e-9, \
    "cosine 起点应等于 learning_rate"
assert abs(lr_at(cfg.lr_decay_iters, cfg) - cfg.min_lr) < 1e-9, \
    "衰减结束应等于 min_lr"
assert lr_at(cfg.lr_decay_iters + 100, cfg) == cfg.min_lr, "衰减之后应恒定在 min_lr"
# 整体单调性：warmup 段递增，衰减段递减
warm = [lr_at(s, cfg) for s in range(cfg.warmup_iters)]
assert all(warm[i] < warm[i + 1] for i in range(len(warm) - 1)), "warmup 段必须递增"
decay = [lr_at(s, cfg) for s in range(cfg.warmup_iters, cfg.lr_decay_iters)]
assert all(decay[i] >= decay[i + 1] for i in range(len(decay) - 1)), "衰减段必须递减"

# =====================================================================
# build_model：dense / moe 都能建，且 MoE 的 aux_loss 能进总 loss
# =====================================================================
m_dense = build_model("dense", V, dropout=0.0)
assert isinstance(m_dense, Qwen3Dense), "build_model('dense') 应返回 Qwen3Dense"
m_moe = build_model("moe", V, dropout=0.0)
assert isinstance(m_moe, Qwen3MoE), "build_model('moe') 应返回 Qwen3MoE"
try:
    build_model("transformer", V)
    raise AssertionError("未知的 model 名应当抛 ValueError")
except ValueError:
    pass

# MoE：总 loss = lm_loss + aux_loss_coef * aux_loss，梯度能回到 router
m_moe.train()
_, moe_loss, _, moe_extras = m_moe(x, targets=y)
assert moe_extras.get("aux_loss") is not None, "MoE 的 extras 必须含 aux_loss"
total = moe_loss + MOE_CONFIG.aux_loss_coef * moe_extras["aux_loss"]
assert torch.isfinite(total), "MoE 总 loss 非有限"
total.backward()
router_grads = [b.moe.router.weight.grad for b in m_moe.blocks
                if b.moe.router.weight.grad is not None]
assert len(router_grads) == MOE_CONFIG.n_layer, "router 应收到梯度"
# dense 侧没有 aux_loss，训练脚本用 extras.get() 统一处理
assert m_dense(x, targets=y)[3].get("aux_loss") is None, \
    "dense 不应有 aux_loss（训练脚本靠 .get 拿 None）"

print(f"test_pretrain OK  loss={loss.item():.4f} params={model.num_params():,} "
      f"lr[0]={lr_at(0, cfg):.2e} lr[end]={lr_at(cfg.lr_decay_iters, cfg):.2e}")

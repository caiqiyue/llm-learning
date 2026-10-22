"""
RMSNorm smoke（对应 feat-012）。

运行：python tests/test_rmsnorm.py
"""

import os
import sys

import torch

# 直接执行 `python tests/test_x.py` 时 sys.path[0] 是 tests/ 目录，
# 找不到项目根下的 config.py / model 包 —— 这里把项目根补进 sys.path。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DENSE_CONFIG
from model.common.rmsnorm import RMSNorm

torch.manual_seed(DENSE_CONFIG.seed)

# --- 1) 形状：支持任意前置维度 ---
norm = RMSNorm(DENSE_CONFIG.n_embd, eps=DENSE_CONFIG.rms_eps)
x = torch.randn(2, 8, DENSE_CONFIG.n_embd)
out = norm(x)
assert out.shape == x.shape, f"形状被改变: {tuple(x.shape)} -> {tuple(out.shape)}"

# --- 2) 数值：平方均值被归一化到 1 ---
#    注意：RMSNorm 不减均值，所以「均值≈0」是数据本身的性质，不是 RMSNorm 的性质；
#    真正的不变量是「平方均值归一化到 1」。
flat = out.reshape(-1, DENSE_CONFIG.n_embd)
rms = flat.pow(2).mean(dim=-1).sqrt()
assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4), f"RMS 未归一到 1: {rms[:5]}"

# --- 3) gain 为 1 时输出 == 归一化结果；gain 可学习 ---
assert torch.allclose(out, x / x.pow(2).mean(-1, keepdim=True).add(DENSE_CONFIG.rms_eps).sqrt(),
                      atol=1e-5), "gain=1 时应等于纯归一化结果"
assert isinstance(norm.gain, torch.nn.Parameter), "gain 必须是可学习参数"
assert norm.gain.requires_grad, "gain 必须参与训练"
assert norm.bias is None, "RMSNorm 不应有 bias"

# --- 4) 放大 gain 后输出按比例放大 ---
norm.gain.data.fill_(2.0)
out2 = norm(x)
assert torch.allclose(out2, out * 2.0, atol=1e-5), "gain 未正确作用于输出"

# --- 5) 极小输入不产生 NaN（eps 兜底）---
tiny = torch.full((1, 1, DENSE_CONFIG.n_embd), 1e-12)
tiny_out = norm(tiny)
assert torch.isfinite(tiny_out).all(), "eps 未兜住除零，产出 NaN"

# --- 6) 梯度能回传到 gain ---
norm.gain.grad = None
norm.gain.data.fill_(1.0)
norm(x).pow(2).mean().backward()
assert norm.gain.grad is not None and torch.isfinite(norm.gain.grad).all(), "gain 无有效梯度"

print(f"test_rmsnorm OK  shape={tuple(out.shape)} eps={DENSE_CONFIG.rms_eps}")

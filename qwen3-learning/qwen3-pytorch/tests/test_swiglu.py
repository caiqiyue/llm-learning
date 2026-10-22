"""
SwiGLU MLP smoke（对应 feat-015）。

对应笔记：qwen3-learning/Qwen3_SwiGLU_MLP_学习笔记.md

运行：python tests/test_swiglu.py
"""

import os
import sys

import torch
import torch.nn.functional as F

# 直接执行 `python tests/test_x.py` 时 sys.path[0] 是 tests/ 目录，
# 找不到项目根下的 config.py / model 包 —— 这里把项目根补进 sys.path。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DENSE_CONFIG as C
from model.common.mlp import SwiGLUMLP, align_intermediate_size

torch.manual_seed(C.seed)

mlp = SwiGLUMLP(C.n_embd, C.intermediate_size, dropout=0.0, bias=C.attention_bias)
mlp.eval()

# --- 1) 形状：输出必须回到残差流宽度 ---
B, T = 2, 8
x = torch.randn(B, T, C.n_embd)
with torch.no_grad():
    out = mlp(x)
assert out.shape == (B, T, C.n_embd), f"输出形状错: {tuple(out.shape)}"

# --- 2) 三个投影的形状 ---
assert mlp.gate_proj.weight.shape == (C.intermediate_size, C.n_embd)
assert mlp.up_proj.weight.shape == (C.intermediate_size, C.n_embd)
assert mlp.down_proj.weight.shape == (C.n_embd, C.intermediate_size)
# 全部无 bias
for name in ["gate_proj", "up_proj", "down_proj"]:
    assert getattr(mlp, name).bias is None, f"{name} 不应有 bias"

# --- 3) 【核心】与逐元素手写一致 ---
with torch.no_grad():
    gate = F.silu(mlp.gate_proj(x))
    up = mlp.up_proj(x)
    manual = mlp.down_proj(gate * up)
    assert torch.allclose(out, manual, atol=1e-5), "SwiGLU 与逐元素手写结果不一致"
    # 三者中间形状确认
    assert gate.shape == (B, T, C.intermediate_size)
    assert up.shape == (B, T, C.intermediate_size)

# --- 4) 门控确实起作用：把 gate 换成常量后输出应改变 ---
with torch.no_grad():
    flat_gate = mlp.down_proj(F.silu(torch.zeros_like(gate)) * up)
    assert not torch.allclose(out, flat_gate, atol=1e-4), "门控分支未起作用"

# --- 5) 【关键对比】SwiGLU 与 GPT-2 的 4x-GELU 在同 n_embd 下的参数量 ---
n_swiglu = sum(p.numel() for m in [mlp.gate_proj, mlp.up_proj, mlp.down_proj] for p in m.parameters())
n_gelu = 2 * C.n_embd * (4 * C.n_embd)      # GPT-2: c_fc + c_proj，各 128×512
ratio = n_swiglu / n_gelu

# 【为什么是 1.125 而不是 1.0 或 2.25】
#   参数量 = 矩阵个数 × n_embd × intermediate
#   · SwiGLU 3 个矩阵 × 128 × 384 = 147,456
#   · GPT-2  2 个矩阵 × 128 × 512 = 131,072
#   → 比值 = 3×384 / (2×512) = 9/8 = 1.125
#   而如果 intermediate 精确取 8/3·n_embd（不向上对齐），两边都是 8·n_embd²，
#   参数量**完全相等** —— 8/3 规则的设计意图就是「持平」。
#   1.125 这个超出量**全部来自向上对齐到 128 的倍数**（341.33 → 384 = 3×128）。
assert n_swiglu == 3 * C.n_embd * C.intermediate_size, "SwiGLU 参数量公式不对"
assert n_gelu == 8 * C.n_embd ** 2, "GPT-2 4×-GELU 参数量公式不对"
assert abs(ratio - 9 / 8) < 1e-9, f"比值应为 9/8=1.125，实得 {ratio:.6f}"

# 反面对照：若 intermediate 精确取 8/3·n_embd，两边参数量相等
n_swiglu_exact = 3 * C.n_embd * (C.n_embd * 8 / 3)
assert abs(n_swiglu_exact - n_gelu) < 1e-6, "精确 8/3 时 SwiGLU 与 4×-GELU 参数量应相等"
print(f"[参数对比] n_embd={C.n_embd}: SwiGLU(中间维={C.intermediate_size}) {n_swiglu:,} vs "
      f"GPT-2 GELU(中间维={4*C.n_embd}) {n_gelu:,} → 比值 {ratio:.3f}")
print("  └─ 精确 8/3 时两者参数量相等 (8*n_embd^2)；本配置多出的 12.5% 全部来自"
      "向上对齐到 128 的倍数 (341.33 -> 384 = 3*128)")

# --- 6) 中间维对齐规则校验 ---
aligned = align_intermediate_size(C.n_embd)
assert C.intermediate_size == aligned, f"intermediate_size 应为 {aligned}，实得 {C.intermediate_size}"
assert C.intermediate_size % 128 == 0, "intermediate_size 必须是 128 的倍数"
assert C.intermediate_size >= C.n_embd * 8 / 3, "向上对齐后不能小于 8/3·n_embd"

# --- 7) 位置无关性：逐位置独立 ---
with torch.no_grad():
    out_single = mlp(x[:, 3:4, :])
assert torch.allclose(out[:, 3:4, :], out_single, atol=1e-5), "MLP 应是位置独立的逐位置变换"

print(f"test_swiglu OK  n_embd={C.n_embd} intermediate={C.intermediate_size}")

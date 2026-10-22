"""
RoPE smoke（对应 feat-013）。

最关键的一条是「旋转等变性」：**同一个** q / k 向量放在不同绝对位置、
但相对距离相同时，旋转后的内积相等。这是 RoPE 全部优势的来源。

运行：python tests/test_rope.py
"""

import os
import sys

import torch

# 直接执行 `python tests/test_x.py` 时 sys.path[0] 是 tests/ 目录，
# 找不到项目根下的 config.py / model 包 —— 这里把项目根补进 sys.path。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DENSE_CONFIG
from model.common.rope import RoPE

torch.manual_seed(DENSE_CONFIG.seed)

HEAD_DIM = 8  # 必须是偶数（旋转按前后两半配对）
MAX_LEN = 64
rope = RoPE(HEAD_DIM, max_seq_len=MAX_LEN, theta=DENSE_CONFIG.rope_theta)

# --- 1) cos / sin 缓存形状 ---
assert rope.cos.shape == (MAX_LEN, HEAD_DIM), f"cos 形状错: {tuple(rope.cos.shape)}"
assert rope.sin.shape == (MAX_LEN, HEAD_DIM), f"sin 形状错: {tuple(rope.sin.shape)}"

# --- 2) cos² + sin² == 1（旋转保范数的代数前提）---
assert torch.allclose(rope.cos.pow(2) + rope.sin.pow(2),
                      torch.ones(MAX_LEN, HEAD_DIM), atol=1e-5), "cos²+sin² ≠ 1"

# --- 3) 位置 0 处 cos=1, sin=0（不旋转）---
assert torch.allclose(rope.cos[0], torch.ones(HEAD_DIM), atol=1e-6), "cos[0] 应全为 1"
assert torch.allclose(rope.sin[0], torch.zeros(HEAD_DIM), atol=1e-6), "sin[0] 应全为 0"

# --- 4) 前向形状不变 ---
q = torch.randn(2, 4, MAX_LEN, HEAD_DIM)
k = torch.randn(2, 4, MAX_LEN, HEAD_DIM)
qr, kr = rope(q, k)
assert qr.shape == q.shape and kr.shape == k.shape, "RoPE 不应改变形状"


# --- 5) 【核心】旋转等变性：内积只依赖相对距离 ---
#     必须是**同一个** q 向量、同一个 k 向量，只改绝对位置 ——
#     等变性说的是 q_mᵀR(m)ᵀR(n)k 只依赖 n-m，q/k 本身不能换。
#     （开发计划初稿在这里写成「取不同位置的 q / k」，是错的，已修正。）
def dot_at(qv, kv, m, n):
    """
    把同一个 q 向量放在位置 m、同一个 k 向量放在位置 n，旋转后取内积。

    参数：
        qv: (B, H, 1, head_dim) —— 固定不变的 q 向量
        kv: (B, H, 1, head_dim) —— 固定不变的 k 向量
        m:  q 的绝对位置
        n:  k 的绝对位置
    返回：
        (B, H, 1) 内积
    """
    pos = torch.tensor([[[m, n]]])                 # (1,1,2)
    cos, sin = rope.get_cos_sin(pos)               # (1,1,2,HEAD_DIM)
    qm = qv[:, :, :1, :]                           # 同一个 q 向量
    kn = kv[:, :, :1, :]                           # 同一个 k 向量
    qm_r = rope._apply_rotation(qm, cos[:, :, :1], sin[:, :, :1])
    kn_r = rope._apply_rotation(kn, cos[:, :, 1:], sin[:, :, 1:])
    return (qm_r * kn_r).sum(dim=-1)               # (B,H,1)


# 固定 q / k 向量，只改绝对位置：
# (m=0,n=3) 与 (m=10,n=13) 的相对距离都是 3 → 内积应相等
d1 = dot_at(q[:, :, :1, :], k[:, :, :1, :], 0, 3)
d2 = dot_at(q[:, :, :1, :], k[:, :, :1, :], 10, 13)
assert torch.allclose(d1, d2, atol=1e-4), (
    f"旋转等变性被破坏：dist=3 的两组内积应相等，实得 {d1.flatten()[:3]} vs {d2.flatten()[:3]}"
)

# 不同相对距离应给出不同内积（否则 RoPE 白做了）
d3 = dot_at(q[:, :, :1, :], k[:, :, :1, :], 0, 4)   # dist = 4
assert not torch.allclose(d1, d3, atol=1e-3), "不同相对距离内积应不同"

# --- 6) position_ids 支持：同一组 q/k 取不同位置应得不同结果 ---
pos_a = torch.tensor([[[0, 1]]])
pos_b = torch.tensor([[[50, 51]]])
qa, ka = rope(q[:, :, :2, :], k[:, :, :2, :], position_ids=pos_a)
qb, kb = rope(q[:, :, :2, :], k[:, :, :2, :], position_ids=pos_b)
assert not torch.allclose(qa, qb, atol=1e-3), "不同 position_ids 应产生不同旋转结果"

# --- 7) position_ids=None 时默认 0..T-1 ---
q_def, _ = rope(q[:, :, :2, :], k[:, :, :2, :])
assert torch.allclose(q_def, qa, atol=1e-6), "position_ids=None 应等价于 0..T-1"

# --- 8) theta 影响低频维度：theta 越大，低维旋转越慢 ---
#     注意两点（开发计划初稿两处都写错了）：
#       a) 位置必须 < max_seq_len（64 越界了，这里用 63）
#       b) head_dim=8 时 inv_freq = [1, θ^-0.25, θ^-0.5, θ^-0.75]，
#          **只有第 0 维与 theta 无关**（inv_freq[0] = θ^0 = 1），
#          「前两维都几乎一致」不成立 —— 第 1 维是 θ^-0.25，受影响很大。
rope_big = RoPE(HEAD_DIM, max_seq_len=MAX_LEN, theta=1e6)
rope_small = RoPE(HEAD_DIM, max_seq_len=MAX_LEN, theta=1e4)
pos = torch.tensor([[[MAX_LEN - 1]]])          # 63，合法位置
cb_, sb_ = rope_big.get_cos_sin(pos)
cs_, ss_ = rope_small.get_cos_sin(pos)
# 最高频维度（inv_freq = 1）与 theta 无关 → 完全一致
assert torch.allclose(cb_[..., :1], cs_[..., :1], atol=1e-6), "最高频维度不应受 theta 影响"
# 最低频维度（inv_freq = θ^-0.75）差异明显
assert not torch.allclose(cb_[..., -2:], cs_[..., -2:], atol=1e-4), "低频维度 theta 应有影响"

# --- 9) 越界保护 ---
try:
    rope(torch.randn(1, 1, MAX_LEN + 1, HEAD_DIM), torch.randn(1, 1, MAX_LEN + 1, HEAD_DIM),
         position_ids=torch.full((1, MAX_LEN + 1), MAX_LEN, dtype=torch.long))
    raise SystemExit("越界位置应报错但未报")
except IndexError:
    pass

print(f"test_rope OK  head_dim={HEAD_DIM} theta={DENSE_CONFIG.rope_theta}")

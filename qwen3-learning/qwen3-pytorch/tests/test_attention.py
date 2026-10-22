"""
CausalSelfAttention smoke（对应 feat-014 下半）。

覆盖 8 项：
  1. 形状契约：输出 (B, T, n_embd)；四个投影的 in/out 维度
  2. 因果性：改动后面的 token 不影响前面位置的输出
  3. **QK-Norm 在 RoPE 之前**（用非均匀 gain 做区分性数值验证）
  4. GQA 分组：repeat_interleave 语义（不是 repeat）
  5. KV Cache：prefill + 逐步 decode 的结果 == 一次性全量 forward
  6. eval 模式确定性
  7. QK-Norm 作用在 head_dim 而非 n_embd
  8. 位置敏感：同样的 token 放在不同绝对位置 → 输出不同

运行：python tests/test_attention.py
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

# 直接执行 `python tests/test_x.py` 时 sys.path[0] 是 tests/ 目录，
# 找不到项目根下的 config.py / model 包 —— 这里把项目根补进 sys.path。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DENSE_CONFIG
from model.common.attention import CausalSelfAttention
from model.common.kv_cache import KVCache

torch.manual_seed(DENSE_CONFIG.seed)

B, T, N_EMBD = 2, 6, 32
N_HEAD, N_HEAD_KV, HEAD_DIM = 4, 2, 8
GROUP = N_HEAD // N_HEAD_KV
BLOCK = 16


def make_attn(dropout: float = 0.0) -> "CausalSelfAttention":
    """
    造一个测试用的小注意力模块（eval 模式、dropout=0，保证可复现）。

    参数：
        dropout: dropout 比例
    返回：
        CausalSelfAttention 实例
    """
    attn = CausalSelfAttention(
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_head_kv=N_HEAD_KV,
        head_dim=HEAD_DIM,
        block_size=BLOCK,
        dropout=dropout,
        bias=False,
        rope_theta=DENSE_CONFIG.rope_theta,
        rms_eps=DENSE_CONFIG.rms_eps,
    )
    attn.eval()
    return attn


attn = make_attn()
x = torch.randn(B, T, N_EMBD)

# ----------------------------------------------------------------------
# 1) 形状契约
# ----------------------------------------------------------------------
y, _ = attn(x)
assert y.shape == (B, T, N_EMBD), f"输出形状应为 (B,T,n_embd)，实得 {tuple(y.shape)}"

assert attn.q_proj.in_features == N_EMBD and attn.q_proj.out_features == N_HEAD * HEAD_DIM, \
    f"q_proj 维度错: {attn.q_proj.in_features}→{attn.q_proj.out_features}"
assert attn.k_proj.out_features == N_HEAD_KV * HEAD_DIM, "k_proj 输出维度应按 n_head_kv"
assert attn.v_proj.out_features == N_HEAD_KV * HEAD_DIM, "v_proj 输出维度应按 n_head_kv"
assert attn.o_proj.in_features == N_HEAD * HEAD_DIM and attn.o_proj.out_features == N_EMBD, \
    "o_proj 应把 H·D 映射回 n_embd"
assert attn._kv_group_size == GROUP, f"kv 组大小应为 {GROUP}，实得 {attn._kv_group_size}"

# head_dim 与 n_embd **解耦**：H·D 可以不等于 n_embd（GPT-2 不行，因为 Q/K/V 必须等长）
attn_dec = CausalSelfAttention(
    n_embd=N_EMBD, n_head=N_HEAD, n_head_kv=N_HEAD_KV, head_dim=16,
    block_size=BLOCK, dropout=0.0, bias=False,
    rope_theta=DENSE_CONFIG.rope_theta, rms_eps=DENSE_CONFIG.rms_eps,
)
attn_dec.eval()
y_dec, _ = attn_dec(x)
assert N_HEAD * 16 != N_EMBD, "本用例应当是解耦配置（4×16=64 ≠ 32）"
assert attn_dec.o_proj.in_features == N_HEAD * 16, "o_proj 输入维度应为 H·head_dim"
assert y_dec.shape == (B, T, N_EMBD), "解耦配置下输出仍须回到 n_embd"

# ----------------------------------------------------------------------
# 2) 因果性：改后面的 token，前面的输出不变
# ----------------------------------------------------------------------
x2 = x.clone()
x2[:, 4:, :] = torch.randn(B, T - 4, N_EMBD)
y2, _ = attn(x2)
assert torch.allclose(y[:, :4], y2[:, :4], atol=1e-5), \
    f"因果性被破坏：前面位置输出变了，最大差 {(y[:, :4] - y2[:, :4]).abs().max().item():.3e}"
assert not torch.allclose(y[:, 4:], y2[:, 4:], atol=1e-5), "改动后续 token 后当前位置竟无变化"

# ----------------------------------------------------------------------
# 3) 【核心】QK-Norm 在 RoPE 之前 —— 用非均匀 gain 做区分性验证
# ----------------------------------------------------------------------
# 背景：RMSNorm 与 RoPE 在 gain **均匀** 时数值上几乎可交换（实测 max|diff| ≈ 5e-7，
# 纯浮点噪声）；只有 gain **非均匀** 时两者才显著不同（实测 ≈ 14.9）。
# 原因：rotate_half 把特征维前后两半交叉配对（dim i ↔ dim i+half），
# 均匀缩放与旋转可交换，非均匀缩放则不可交换。
# 所以要证明「谁在前」，**必须用非均匀 gain**。

attn3 = make_attn()
with torch.no_grad():
    attn3.q_norm.gain.copy_(torch.arange(1, HEAD_DIM + 1, dtype=torch.float32))
    attn3.k_norm.gain.copy_(torch.arange(1, HEAD_DIM + 1, dtype=torch.float32))

x3 = torch.randn(1, 4, N_EMBD)
q_impl, k_impl = attn3._project_and_rotate(x3)

# 手工重算两种顺序
B3, T3, _ = x3.shape
q_raw = attn3.q_proj(x3).view(B3, T3, N_HEAD, HEAD_DIM).transpose(1, 2)
k_raw = attn3.k_proj(x3).view(B3, T3, N_HEAD_KV, HEAD_DIM).transpose(1, 2)

# 顺序 A（正确）：QK-Norm → RoPE
q_A, k_A = attn3.rope(attn3.q_norm(q_raw), attn3.k_norm(k_raw))
# 顺序 B（错误）：RoPE → QK-Norm
q_B_raw, _ = attn3.rope(q_raw, k_raw)
q_B = attn3.q_norm(q_B_raw)

assert torch.allclose(q_impl, q_A, atol=1e-5), \
    f"实现与「QK-Norm → RoPE」不一致，最大差 {(q_impl - q_A).abs().max().item():.3e}"
assert torch.allclose(k_impl, k_A, atol=1e-5), \
    f"实现与「QK-Norm → RoPE」不一致（K），最大差 {(k_impl - k_A).abs().max().item():.3e}"

diff_ab = (q_impl - q_B).abs().max().item()
assert diff_ab > 1.0, \
    f"实现与「RoPE → QK-Norm」几乎相同（max|diff|={diff_ab:.3e}），说明顺序没被真正区分"

# 反面对照：gain 均匀时两种顺序数值几乎一致（可交换）—— 记录这个事实
attn3u = make_attn()
q_u, _ = attn3u._project_and_rotate(x3)
q_raw_u = attn3u.q_proj(x3).view(B3, T3, N_HEAD, HEAD_DIM).transpose(1, 2)
k_raw_u = attn3u.k_proj(x3).view(B3, T3, N_HEAD_KV, HEAD_DIM).transpose(1, 2)
q_Au, _ = attn3u.rope(attn3u.q_norm(q_raw_u), attn3u.k_norm(k_raw_u))
q_Bu_raw, _ = attn3u.rope(q_raw_u, k_raw_u)
q_Bu = attn3u.q_norm(q_Bu_raw)
diff_uniform = (q_Au - q_Bu).abs().max().item()
assert diff_uniform < 1e-4, f"gain 均匀时两序应几乎相同，实得 {diff_uniform:.3e}"
assert diff_ab > 1e3 * max(diff_uniform, 1e-12), "非均匀 gain 下两序的差异应远大于均匀时"

# ----------------------------------------------------------------------
# 4) GQA 分组语义：必须是 repeat_interleave（分组共享），不是 repeat
# ----------------------------------------------------------------------
# 手工复算整条注意力链路，与实现逐元素比对
attn4 = make_attn()
x4 = torch.randn(B, T, N_EMBD)
y4_impl, _ = attn4(x4)

q4, k4 = attn4._project_and_rotate(x4)
v4 = attn4.v_proj(x4).view(B, T, N_HEAD_KV, HEAD_DIM).transpose(1, 2)


def manual_attention(q, k, v, group, block, use_repeat: bool = False):
    """
    手工复算注意力：缩放点积 → 因果 mask → softmax → 加权求和 → 拼头 → 输出投影。

    参数：
        q:         (B, H, T, D)
        k, v:      (B, Hkv, T, D)
        group:     GQA 组大小
        block:     block_size
        use_repeat: True 时用 repeat（错误语义），False 时用 repeat_interleave
    返回：
        (B, T, n_embd)
    """
    Bx, H, Tq, D = q.shape
    if use_repeat:
        k_e = k.repeat(1, H // group, 1, 1)
        v_e = v.repeat(1, H // group, 1, 1)
    else:
        k_e = k.repeat_interleave(group, dim=1)
        v_e = v.repeat_interleave(group, dim=1)
    att = (q @ k_e.transpose(-2, -1)) / math.sqrt(D)
    m = torch.triu(torch.ones(block, block, dtype=torch.bool), diagonal=1)
    att = att.masked_fill(m[:Tq, :Tq], float("-inf"))
    p = F.softmax(att, dim=-1)
    o = p @ v_e
    o = o.transpose(1, 2).contiguous().view(Bx, Tq, H * D)
    return attn4.o_proj(o)


y4_expected = manual_attention(q4, k4, v4, GROUP, BLOCK, use_repeat=False)
assert torch.allclose(y4_impl, y4_expected, atol=1e-5), \
    f"实现与手工复算不符，最大差 {(y4_impl - y4_expected).abs().max().item():.3e}"

y4_wrong = manual_attention(q4, k4, v4, GROUP, BLOCK, use_repeat=True)
assert not torch.allclose(y4_impl, y4_wrong, atol=1e-4), \
    "实现结果与 repeat（错误分组语义）相同，说明 GQA 用错了 repeat_interleave"

# ----------------------------------------------------------------------
# 5) KV Cache：prefill + 逐 token decode == 一次性全量 forward
# ----------------------------------------------------------------------
attn5 = make_attn()
x5 = torch.randn(B, T, N_EMBD)
y5_full, _ = attn5(x5)

cache = KVCache(n_layer=1, batch_size=B, n_head_kv=N_HEAD_KV,
                head_dim=HEAD_DIM, max_len=T + 4)
prefill_len = 3
y5_pre, _ = attn5(x5[:, :prefill_len], kv_cache=cache, layer_idx=0)
assert torch.allclose(y5_pre, y5_full[:, :prefill_len], atol=1e-5), \
    "prefill 结果与全量 forward 前段不一致"

for step in range(prefill_len, T):
    pos = torch.full((B, 1), step, dtype=torch.long)
    y5_step, _ = attn5(x5[:, step:step + 1], kv_cache=cache, layer_idx=0, position_ids=pos)
    assert torch.allclose(y5_step, y5_full[:, step:step + 1], atol=1e-5), \
        f"decode 第 {step} 步与全量 forward 不一致"
assert cache.seq_len(0) == T, f"cache 长度应为 {T}，实得 {cache.seq_len(0)}"

# ----------------------------------------------------------------------
# 6) eval 模式确定性
# ----------------------------------------------------------------------
attn6 = make_attn()
x6 = torch.randn(B, T, N_EMBD)
y6a, _ = attn6(x6)
y6b, _ = attn6(x6)
assert torch.equal(y6a, y6b), "eval 模式两次 forward 结果应完全一致"

# ----------------------------------------------------------------------
# 7) QK-Norm 作用维度是 head_dim（逐 head 归一化），不是 n_embd
# ----------------------------------------------------------------------
assert attn.q_norm.gain.shape == (HEAD_DIM,), \
    f"q_norm.gain 应为 (head_dim,)，实得 {tuple(attn.q_norm.gain.shape)}"
assert attn.k_norm.gain.shape == (HEAD_DIM,), "k_norm.gain 应为 (head_dim,)"

# ----------------------------------------------------------------------
# 8) 位置行为：RoPE 是**相对**位置编码
#    8a) 整体平移绝对位置 → 输出不变（相对位置编码的标志）
#    8b) 改变**相对**距离 → 输出改变
# ----------------------------------------------------------------------
attn8 = make_attn()
x8 = torch.randn(1, 4, N_EMBD)
y8_p0, _ = attn8(x8)
y8_p5, _ = attn8(x8, position_ids=torch.arange(5, 9).unsqueeze(0))
assert torch.allclose(y8_p0, y8_p5, atol=1e-5), \
    f"整体平移位置后输出应变不变（RoPE 相对位置性质），最大差 " \
    f"{(y8_p0 - y8_p5).abs().max().item():.3e}"

# 8b) 同一个 token 与前文的相对距离变了 → 输出必须变
x8a = torch.randn(1, 4, N_EMBD)
x8b = torch.cat([x8a[:, :1], torch.randn(1, 1, N_EMBD), x8a[:, 1:2]], dim=1)  # 中间插一个 token
y8a, _ = attn8(x8a)
y8b, _ = attn8(x8b)
assert not torch.allclose(y8a[:, 1], y8b[:, 2], atol=1e-4), \
    "相对距离改变后输出竟相同，说明位置信息没进到注意力里"

print(f"test_attention OK  n_head={N_HEAD} n_head_kv={N_HEAD_KV} group={GROUP} "
      f"head_dim={HEAD_DIM}  顺序差(非均匀gain)={diff_ab:.4f} "
      f"顺序差(均匀gain)={diff_uniform:.2e}")

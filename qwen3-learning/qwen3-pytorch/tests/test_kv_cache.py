"""
KVCache smoke（对应 feat-014 上半）。

GQA 下的核心断言：**缓存按 KV head 数分配**，因此占用是同尺寸 MHA 的 1/4。

运行：python tests/test_kv_cache.py
"""

import os
import sys

import torch

# 直接执行 `python tests/test_x.py` 时 sys.path[0] 是 tests/ 目录，
# 找不到项目根下的 config.py / model 包 —— 这里把项目根补进 sys.path。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DENSE_CONFIG
from model.common.kv_cache import KVCache

torch.manual_seed(DENSE_CONFIG.seed)

B, N_LAYER, N_HEAD, N_HEAD_KV, HEAD_DIM, MAX_LEN = 2, 4, 8, 2, 16, 32

# --- 1) 分配后的形状按 n_head_kv 而非 n_head ---
cache = KVCache(n_layer=N_LAYER, batch_size=B, n_head_kv=N_HEAD_KV,
                head_dim=HEAD_DIM, max_len=MAX_LEN)
k = torch.randn(B, N_HEAD_KV, 4, HEAD_DIM)
v = torch.randn(B, N_HEAD_KV, 4, HEAD_DIM)
cache.append(0, k, v)
k_full, v_full = cache.get(0)
assert k_full.shape == (B, N_HEAD_KV, 4, HEAD_DIM), f"K 形状错: {tuple(k_full.shape)}"

# --- 2) 占用是 MHA 的 1/4（N_KV / N_HEAD = 2/8）---
cache_mha = KVCache(n_layer=N_LAYER, batch_size=B, n_head_kv=N_HEAD,
                    head_dim=HEAD_DIM, max_len=MAX_LEN)
moe_size = cache.nbytes()
mha_size = cache_mha.nbytes()
assert mha_size == moe_size * (N_HEAD // N_HEAD_KV), f"占用比不对: {moe_size} vs {mha_size}"
assert mha_size / moe_size == 4.0, f"GQA(8Q/2KV) 应为 MHA 的 1/4，实得 {mha_size/moe_size:.2f}x"

# --- 3) prefill 阶段：一次写入 T 个 token ---
cache.append(1, k, v)
assert cache.seq_len(1) == 4, f"seq_len 应为 4，实得 {cache.seq_len(1)}"

# --- 4) decode 阶段：每次追加 1 个 token ---
for _step in range(3):
    k_new = torch.randn(B, N_HEAD_KV, 1, HEAD_DIM)
    v_new = torch.randn(B, N_HEAD_KV, 1, HEAD_DIM)
    cache.append(1, k_new, v_new)
assert cache.seq_len(1) == 7, f"decode 3 步后应为 7，实得 {cache.seq_len(1)}"
_, v_full = cache.get(1)
assert v_full.shape[2] == 7, f"取出的 V 长度应为 7，实得 {v_full.shape[2]}"

# --- 5) 内容正确：前 4 步的值不被后续追加改动 ---
k_ret, _ = cache.get(1)
assert torch.allclose(k_ret[:, :, :4, :], k, atol=1e-6), "历史 K 被后续追加污染"

# --- 6) 不同 batch 元素互不污染 ---
assert not torch.allclose(k_ret[0], k_ret[1], atol=1e-6), "不同 batch 元素应持有不同 KV"
# 缓存是 batch 同步的：一次 append 对所有 batch 同步增长
cache.append(1, torch.randn(B, N_HEAD_KV, 1, HEAD_DIM), torch.randn(B, N_HEAD_KV, 1, HEAD_DIM))
assert cache.seq_len(1) == 8, "追加应对所有 batch 同步增长"

# --- 7) 不同 layer 独立 ---
assert cache.seq_len(0) == 4 and cache.seq_len(1) == 8, "不同 layer 的 seq_len 应独立"

# --- 8) 溢出保护 ---
cache_small = KVCache(n_layer=1, batch_size=1, n_head_kv=1, head_dim=4, max_len=2)
cache_small.append(0, torch.randn(1, 1, 2, 4), torch.randn(1, 1, 2, 4))
try:
    cache_small.append(0, torch.randn(1, 1, 1, 4), torch.randn(1, 1, 1, 4))
    raise SystemExit("超出 max_len 应报错但未报")
except RuntimeError:
    pass

# --- 9) reset 可复用 ---
cache.reset()
assert cache.seq_len(0) == 0 and cache.seq_len(1) == 0, "reset 后 seq_len 应归零"

print(f"test_kv_cache OK  N_HEAD={N_HEAD} N_HEAD_KV={N_HEAD_KV} "
      f"GQA占用={moe_size}B MHA占用={mha_size}B 比值={mha_size/moe_size:.1f}")

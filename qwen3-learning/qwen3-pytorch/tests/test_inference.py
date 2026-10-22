"""
Qwen3 推理（prefill / decode / 采样 / KV Cache 占用）smoke（对应 feat-018）。

运行：python tests/test_inference.py
"""

import os
import sys

import torch

# 直接执行 `python tests/test_x.py` 时 sys.path[0] 是 tests/ 目录，
# 找不到项目根下的 config.py / model 包 —— 这里把项目根补进 sys.path。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DENSE_CONFIG as CFG
from inference import (
    benchmark_speed,
    generate_naive,
    generate_with_cache,
    kv_cache_bytes,
    kv_cache_report,
    make_kv_cache,
    sample_from_logits,
)
from model.qwen3_dense import Qwen3Dense

torch.manual_seed(CFG.seed)

VOCAB = 64
model = Qwen3Dense(vocab_size=VOCAB, cfg=CFG, dropout=0.0)
model.eval()

PROMPT_LEN = 8
prompt = torch.randint(0, VOCAB, (1, PROMPT_LEN), dtype=torch.long)

# --- 1) greedy（temperature <= 0）就是 argmax ---
logits = torch.tensor([[1.0, 5.0, 2.0, 0.0]])
assert int(sample_from_logits(logits, temperature=0.0).item()) == 1, "greedy 应取最大值"
# greedy 路径**不执行 top-k / top-p**：即使 top_k=1 也应取全局最大而不是第 0 个位置
assert int(sample_from_logits(logits, temperature=0.0, top_k=1).item()) == 1, \
    "greedy 不应受 top_k 影响"

# --- 2) temperature 只改变分布形状，不改变排序 ---
sharp = sample_from_logits(logits, temperature=0.5, top_k=2)
assert sharp.shape == (1, 1), f"采样结果形状应为 (1,1)，实得 {tuple(sharp.shape)}"
assert int(sharp.item()) in (0, 1), "top-k=2 后只可能取前 2 名"

# --- 3) top-k 把第 k 名之后的 logits 置 -inf ---
#    构造一个「只有前 2 个 token 有概率」的分布，top_k=1 时必须只取第 0 个
skew = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
for _ in range(20):
    assert int(sample_from_logits(skew, temperature=1.0, top_k=1).item()) == 0, \
        "top_k=1 时必须恒取第 0 个"

# --- 4) top-p：概率高度集中时，nucleus 只保留 1 个 token ---
#    top_p=0.5 且第 0 个概率 > 0.5 ⇒ 只剩它
for _ in range(20):
    assert int(sample_from_logits(skew, temperature=1.0, top_p=0.5).item()) == 0, \
        "top_p=0.5 在高度集中的分布上应只保留第 0 个"

# --- 5) ★ 有/无 KV Cache 的生成本身一致（greedy）---
seq_naive = generate_naive(model, prompt, max_new_tokens=12, temperature=0.0)
seq_kv = generate_with_cache(model, prompt, max_new_tokens=12, temperature=0.0)
assert seq_naive.shape == seq_kv.shape == (1, PROMPT_LEN + 12), \
    f"两条路径的输出长度应相同，实得 {tuple(seq_naive.shape)} / {tuple(seq_kv.shape)}"
assert torch.equal(seq_naive, seq_kv), (
    f"greedy 下有/无 KV Cache 应生成完全相同的 token 序列\n"
    f"naive={seq_naive.tolist()}\nkv  ={seq_kv.tolist()}"
)

# --- 6) ★ 更强的一致性断言：逐步 logits 对得上（不依赖 argmax 不平局）---
_, l_naive = generate_naive(model, prompt, max_new_tokens=12, temperature=0.0,
                            return_logits=True)
_, l_kv = generate_with_cache(model, prompt, max_new_tokens=12, temperature=0.0,
                              return_logits=True)
assert len(l_naive) == len(l_kv) == 12, "两条路径应各产出 12 步 logits"
max_diff = 0.0
for a, b in zip(l_naive, l_kv):
    max_diff = max(max_diff, float((a - b).abs().max().item()))
assert max_diff < 1e-4, f"两条路径的逐步 logits 应一致，实得 max|diff| = {max_diff:.3e}"

# --- 7) decode 阶段 cache 的长度逐步增长（prefill 一次写 P，之后每步 +1）---
cache = make_kv_cache(model, batch_size=1, max_len=PROMPT_LEN + 12)
pos = torch.arange(PROMPT_LEN, dtype=torch.long).unsqueeze(0)
with torch.no_grad():
    logits, _, cache, _ = model(prompt, kv_cache=cache, position_ids=pos)
for layer_idx in range(CFG.n_layer):
    assert cache.seq_len(layer_idx) == PROMPT_LEN, \
        f"prefill 后第 {layer_idx} 层应缓存 {PROMPT_LEN} 个，实得 {cache.seq_len(layer_idx)}"
nxt = torch.randint(0, VOCAB, (1, 1), dtype=torch.long)
with torch.no_grad():
    _, _, cache, _ = model(nxt, kv_cache=cache,
                           position_ids=torch.full((1, 1), PROMPT_LEN, dtype=torch.long))
for layer_idx in range(CFG.n_layer):
    assert cache.seq_len(layer_idx) == PROMPT_LEN + 1, \
        f"decode 1 步后第 {layer_idx} 层应缓存 {PROMPT_LEN + 1} 个，实得 {cache.seq_len(layer_idx)}"

# --- 8) ★ decode 不施加因果 mask：T_q=1 时应能看到全部历史 ---
#    可控实验：prefill 之后把 cache 里「除最后一个之外」的历史 K/V 清零，
#    再各走一步 decode。
#      · 若 decode 真的读了历史（T_q=1、不 mask）→ 输出**必须改变**
#      · 若 decode 错误地把历史屏蔽掉 → 输出**不会变**
def _decode_one(cache_obj):
    """在给定 cache 上再 decode 一步，返回 (B, V) 的 logits。"""
    with torch.no_grad():
        lg, _, _, _ = model(nxt, kv_cache=cache_obj,
                            position_ids=torch.full((1, 1), PROMPT_LEN, dtype=torch.long))
    return lg[:, -1, :]


clean_cache = make_kv_cache(model, batch_size=1, max_len=PROMPT_LEN + 4)
with torch.no_grad():
    _, _, clean_cache, _ = model(prompt, kv_cache=clean_cache, position_ids=pos)
clean_logits = _decode_one(clean_cache)

dirty_cache = make_kv_cache(model, batch_size=1, max_len=PROMPT_LEN + 4)
with torch.no_grad():
    _, _, dirty_cache, _ = model(prompt, kv_cache=dirty_cache, position_ids=pos)
# 清零「除最后一个 token 外」的历史 K/V（cache 形状：layer, B, n_head_kv, T, head_dim）
dirty_cache.k_buf[:, :, :, : PROMPT_LEN - 1, :] = 0.0
dirty_cache.v_buf[:, :, :, : PROMPT_LEN - 1, :] = 0.0
dirty_logits = _decode_one(dirty_cache)

assert float((dirty_logits - clean_logits).abs().max().item()) > 1e-5, \
    "污染历史 KV 后 decode 输出应发生变化（说明 decode 确实读了历史、不该有因果 mask）"

# --- 9) eos 早停：先贪心跑一步拿到「模型会生成的那个 token」，把它当 eos ---
first = int(generate_with_cache(model, prompt, max_new_tokens=1, temperature=0.0)[0, -1].item())
seq_eos = generate_with_cache(model, prompt, max_new_tokens=10, temperature=0.0,
                              eos_id=first)
assert seq_eos.size(1) == PROMPT_LEN + 1, \
    f"第一个 token 就是 eos，应只生成 1 个，实得 {seq_eos.size(1) - PROMPT_LEN} 个"
seq_naive_eos = generate_naive(model, prompt, max_new_tokens=10, temperature=0.0,
                               eos_id=first)
assert seq_naive_eos.size(1) == PROMPT_LEN + 1, \
    f"naive 路径的 eos 早停应同样只生成 1 个，实得 {seq_naive_eos.size(1) - PROMPT_LEN} 个"

# --- 10) 采样：temperature > 0 时同一 prompt 多次采样应能产生不同结果 ---
outs = set()
for _ in range(20):
    s = generate_with_cache(model, prompt, max_new_tokens=1, temperature=2.0, top_k=None)
    outs.add(int(s[0, -1].item()))
assert len(outs) > 1, f"temperature=2.0 采样 20 次应出现多个不同 token，实得 {outs}"

# --- 11) KV Cache 占用：GQA 是同尺寸 MHA 的 n_head_kv/n_head ---
report = kv_cache_report(model, prompt_len=16, max_new_tokens=48)
expect_gqa = CFG.n_layer * 1 * CFG.n_head_kv * 64 * CFG.head_dim * 2 * 4
assert report["gqa_bytes"] == expect_gqa, \
    f"KV Cache 占用公式不对：{report['gqa_bytes']} != {expect_gqa}"
assert report["ratio"] == CFG.n_head_kv / CFG.n_head, \
    f"GQA/MHA 占用比应为 {CFG.n_head_kv}/{CFG.n_head}，实得 {report['ratio']}"
assert report["mha_bytes"] == report["gqa_bytes"] * (CFG.n_head // CFG.n_head_kv), \
    "MHA 占用应为 GQA 的 n_head/n_head_kv 倍"
assert kv_cache_bytes(1, 1, 1, 1, 1, 1) == 2, "公式自检：n_layer=B=1,T=1,D=1,dtype=1 应为 2 字节"

# --- 12) ★ benchmark_speed 的列名与 GPT-2 线一致，且 speedup > 1 ---
bench = benchmark_speed(model, prompt, n_runs=2, max_new_tokens=16, temperature=0.0)
for key in ("naive_avg_ms", "kv_avg_ms", "speedup", "naive_tokens", "kv_tokens"):
    assert key in bench, f"benchmark_speed 缺少与 GPT-2 线同名的列：{key}"
assert bench["naive_tokens"] == bench["kv_tokens"] == 16, \
    f"两边生成的 token 数应相同，实得 {bench['naive_tokens']} / {bench['kv_tokens']}"
assert bench["speedup"] > 1.0, f"有 KV Cache 应更快，实得 speedup={bench['speedup']:.3f}"

print(f"test_inference OK  greedy一致=逐位相同 logits_max|diff|={max_diff:.2e} "
      f"speedup={bench['speedup']:.2f}x "
      f"KV占用(GQA/MHA)={report['gqa_bytes']}/{report['mha_bytes']}B "
      f"比值=1/{CFG.n_head // CFG.n_head_kv}")

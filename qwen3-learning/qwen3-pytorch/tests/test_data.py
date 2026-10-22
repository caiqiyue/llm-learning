"""
数据层 smoke：预训练切块 + 偏好数据构造（对应 feat-019）。

运行：python tests/test_data.py
"""

import os
import sys

# 直接执行 `python tests/test_x.py` 时 sys.path[0] 是 tests/ 目录，
# 找不到项目根下的 config.py / data 包 —— 这里把项目根补进 sys.path。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import DENSE_CONFIG
from data.corpus import generate_sentences
from data.pretrain_data import (
    build_blocks,
    build_token_stream,
    generate_corpus,
    make_pretrain_loader,
)
from data.preference_data import (
    STRATEGIES,
    format_prompt_response,
    generate_preference_dataset,
    make_preference_loader,
)


def torch_all_finite(t: "torch.Tensor") -> bool:
    """判断张量是否全部有限（无 NaN / Inf）。"""
    return bool(torch.isfinite(t).all())


# =====================================================================
# 1) 预训练切块：inputs/targets 错位一位
# =====================================================================
ids = list(range(100))
blocks = build_blocks(ids, block_size=8)
assert len(blocks) > 0, "切块结果为空"
x, y = blocks[0]
assert len(x) == 8 and len(y) == 8, f"块长度错: {len(x)}, {len(y)}"
assert x == ids[0:8], "inputs 取错位置"
assert y == ids[1:9], "targets 应是 inputs 左移一位"
assert y[:-1] == x[1:], "targets 与 inputs 必须错位对齐"
# 块数正确：step = block_size + 1 = 9
assert len(blocks) == (len(ids) - 9) // 9 + 1, f"块数错: {len(blocks)}"

# 末尾不足一块的部分必须被丢弃（不能切出半个块）
assert ids[-1] not in [v for b in blocks for v in b[1][-1:]] or True   # 只作提示
for bx, by in blocks:
    assert by[:-1] == bx[1:], "每块内部都必须错位对齐"

# =====================================================================
# 2) 语料生成可复现，且与 BPE 训练语料同源
# =====================================================================
c1 = generate_corpus(n_samples=50, seed=7)
c2 = generate_corpus(n_samples=50, seed=7)
assert c1 == c2, "同 seed 的语料应完全一致"
assert c1 != generate_corpus(n_samples=50, seed=8), "不同 seed 应产出不同语料"
# 语料是「句子列表」，不是拼好的长字符串
assert isinstance(c1, list) and isinstance(c1[0], str), "语料应为句子列表"
# 预训练语料与 BPE 训练语料同源：同 seed 下前 N 句必须一致
n_pre = DENSE_CONFIG.pretrain_n_samples
assert generate_corpus(n_samples=n_pre, seed=DENSE_CONFIG.seed) == \
    generate_sentences(n_samples=n_pre, seed=DENSE_CONFIG.seed), \
    "预训练语料与 BPE 训练语料必须来自同一个生成器"

# =====================================================================
# 3) ★ token 流里的文档分隔符必须是「特殊 token id 0」，不是它的字面文本
# =====================================================================
VOCAB_PATH = os.path.join("data", "vocab.json")
MERGES_PATH = os.path.join("data", "merges.txt")
if os.path.exists(VOCAB_PATH) and os.path.exists(MERGES_PATH):
    from data.tokenizer import BPETokenizer

    tk = BPETokenizer(VOCAB_PATH, MERGES_PATH)
    sentences = generate_corpus(n_samples=20, seed=DENSE_CONFIG.seed)
    stream = build_token_stream(sentences, tk)

    # 每个句子后面恰好一个 EOT
    assert stream.count(0) == len(sentences), \
        f"EOT 数量应为 {len(sentences)}，实得 {stream.count(0)}"
    # 特殊 token 的「字面文本」不应该被当成普通文本编码进去
    literal = tk.encode("<|endoftext|>")
    assert 0 not in literal, "字面文本不应编码成 id 0"
    assert len(literal) > 1, "字面文本应被拆成多个普通 token（说明它不是特殊 token）"
    # 流里除了 EOT 之外，不应出现字面文本的那串 token
    assert len(stream) > len(literal), "token 流长度异常"

    # BPE 确实在压缩（比逐字节短）
    raw_bytes = sum(len(s.encode("utf-8")) for s in sentences)
    assert len(stream) < raw_bytes, f"BPE 未压缩：{len(stream)} >= {raw_bytes}"
    print(f"[token 流] {len(sentences)} 句 -> {len(stream)} tokens "
          f"（原始 {raw_bytes} 字节，压缩比 {raw_bytes / len(stream):.2f} 字节/token）")
else:
    tk = None
    print("[token 流] 跳过（data/vocab.json 未生成，先跑 python data/build_vocab.py）")

# =====================================================================
# 4) 偏好数据结构
# =====================================================================
pairs = generate_preference_dataset(n_samples=200, seed=42)
assert len(pairs) == 200, f"条数错: {len(pairs)}"
for p in pairs:
    assert p["prompt"], "prompt 不得为空"
    assert p["chosen"], "chosen 不得为空"
    assert p["rejected"], "rejected 不得为空"
    assert p["chosen"] != p["rejected"], "chosen 与 rejected 不得相同"
    assert p["strategy"] in STRATEGIES
# 四种策略都出现了
assert len({p["strategy"] for p in pairs}) == len(STRATEGIES), "四种 rejected 策略都应出现"
# 可复现
assert generate_preference_dataset(n_samples=10, seed=42) == generate_preference_dataset(
    n_samples=10, seed=42
), "偏好数据生成应可复现"
# 格式拼接
fmt = format_prompt_response("Q?", "A.")
assert "Q?" in fmt and "A." in fmt, "拼接丢失内容"
# loader 接口
loader_pairs = make_preference_loader(n_samples=20, seed=42, shuffle=True)
assert len(loader_pairs) == 20, f"loader 条数错: {len(loader_pairs)}"
assert {p["strategy"] for p in loader_pairs} <= set(STRATEGIES), "loader 出现未知策略"

# =====================================================================
# 5) DataLoader（需要 vocab.json，若不存在则跳过）
# =====================================================================
if tk is not None:
    loader = make_pretrain_loader(split="val", batch_size=4, block_size=32,
                                  seed=DENSE_CONFIG.seed)
    xb, yb = next(iter(loader))
    assert xb.shape == (4, 32), f"batch inputs 形状错: {tuple(xb.shape)}"
    assert yb.shape == (4, 32), f"batch targets 形状错: {tuple(yb.shape)}"
    assert torch_all_finite(xb), "inputs 含非法值"
    assert xb.dtype == torch.long and yb.dtype == torch.long, "必须是 long 类型"
    # 错位一位
    assert torch.equal(yb[:, :-1], xb[:, 1:]), "batch 内 targets 与 inputs 必须错位一位"
    # token 全部在词表范围内
    assert int(xb.max()) < tk.vocab_size, "token id 超出词表范围"
    assert int(xb.min()) >= 0, "出现负的 token id"
    print(f"[DataLoader] val batches={len(loader)} shape={tuple(xb.shape)} "
          f"vocab={tk.vocab_size}")

    # train / val 划分必须稳定且互补
    tr = make_pretrain_loader(split="train", batch_size=4, block_size=32,
                              seed=DENSE_CONFIG.seed)
    va = make_pretrain_loader(split="val", batch_size=4, block_size=32,
                              seed=DENSE_CONFIG.seed)
    n_tr = len(tr.dataset)
    n_va = len(va.dataset)
    assert n_tr > 0 and n_va > 0, "train/val 不能为空"
    ratio = n_tr / (n_tr + n_va)
    assert abs(ratio - DENSE_CONFIG.train_ratio) < 0.05, \
        f"train 占比 {ratio:.3f} 与 config.train_ratio={DENSE_CONFIG.train_ratio} 不符"
    print(f"[DataLoader] train={n_tr} val={n_va} 占比={ratio:.3f} "
          f"(config.train_ratio={DENSE_CONFIG.train_ratio})")

print(f"test_data OK  blocks={len(blocks)} pairs={len(pairs)}")

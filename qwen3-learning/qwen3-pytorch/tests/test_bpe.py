"""
byte-level BPE smoke（对应 feat-022）。

对应笔记：qwen3-learning/Qwen3_BPE_学习笔记.md

运行：python tests/test_bpe.py
"""

import os
import sys
import tempfile

# 直接执行 `python tests/test_x.py` 时 sys.path[0] 是 tests/ 目录，
# 找不到项目根下的 config.py / data 包 —— 这里把项目根补进 sys.path。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.bpe import (
    BASE_VOCAB_SIZE,
    ENDOFTEXT_ID,
    MERGE_OFFSET,
    bytes_to_token_str,
    decode,
    encode,
    load_bpe,
    save_bpe,
    train_bpe,
)
from data.tokenizer import BPETokenizer

# =====================================================================
# 1) 词表结构：id 0 特殊 token / id 1..256 字节 / id 257+ 合并
# =====================================================================
CORPUS = [
    "The student trained a tiny transformer.",
    "The quick brown fox jumps over the lazy dog.",
    "Transformers are built on attention mechanisms.",
    "你好，这是一段中文测试语料。",
    "def train(model, batch):\n    return model(batch)\n",
    "混合 mixed 語言 12345 !!!",
]
VOCAB_SIZE = 300          # 300 - 257 = 43 次合并，足够看到 BPE 的效果又很快

vocab, merges = train_bpe(CORPUS, vocab_size=VOCAB_SIZE)

assert vocab[ENDOFTEXT_ID] == b"<|endoftext|>", \
    f"特殊 token 必须占 id 0，实得 {vocab.get(0)!r}"
assert len(vocab) == BASE_VOCAB_SIZE + len(merges), \
    f"词表大小应为 {BASE_VOCAB_SIZE}+{len(merges)}={BASE_VOCAB_SIZE + len(merges)}，" \
    f"实得 {len(vocab)}"

# 256 个字节 token 必须在，且 id = 字节值 + 1
for byte_val in range(256):
    assert vocab[byte_val + 1] == bytes([byte_val]), \
        f"字节 token 缺失：字节 {byte_val} 应在 id {byte_val + 1}"

# ★ 回归防线：字节 0xFF 必须在 id 256 上，不能被第一个合并 token 顶掉。
#   初稿把合并 token 的起始 id 定成 256，与「字节 0xFF 占 id 256」撞车，
#   结果 0xFF 被静默吞掉（词表少了 1 个 id，0xFF 无法编码）。
assert vocab[256] == bytes([0xFF]), \
    f"id 256 必须是字节 0xFF，实得 {vocab[256]!r}"
assert bytes([0xFF]) in vocab.values(), "字节 0xFF 在词表中消失了"

# 合并 token 必须从 257 开始，且内容 = 两个被合并 token 的字节拼接
for i, (a, b) in enumerate(merges):
    mid = MERGE_OFFSET + i
    assert mid in vocab, f"合并 token id {mid} 不在词表中"
    assert vocab[mid] == vocab[a] + vocab[b], \
        f"合并 token {mid} 的字节内容与 (a={a}, b={b}) 不符"
assert MERGE_OFFSET == 257, f"合并起始 id 应为 257，实得 {MERGE_OFFSET}"
assert all(mid >= MERGE_OFFSET for mid in vocab if mid not in range(0, 257)), \
    "存在 id 落在 [257, ) 之外的合并 token"

# 所有 token 的字节串必须互不相同（id 撞车的另一面）
byte_strings = list(vocab.values())
assert len(set(byte_strings)) == len(byte_strings), "词表中存在内容相同的两个 token"

# =====================================================================
# 2) 编解码往返：任意文本逐字节还原
# =====================================================================
for text in CORPUS:
    ids = encode(text, merges, vocab)
    out = decode(ids, vocab)
    assert out == text, f"往返失败：{text!r} -> {ids} -> {out!r}"

# 语料里没出现过的字符也必须能还原（byte-level 的意义：永不 UNK）
for text in ["€uro", "\U0001f642", "\t\n\r", "Ǆungla", "a b", "  ", "。"]:
    assert decode(encode(text, merges, vocab), vocab) == text, f"特殊字符往返失败: {text!r}"

# =====================================================================
# 3) 压缩：语料内的句子被压短；语料外的文本最多持平（不会变长）
# =====================================================================
# 注意：压缩断言必须用**语料内**的句子。
# 预分词按空白切块，训练学到的合并规则只在语料里出现过的块上有效；
# 用语料外的字符串（如 "The quick brown fox" 里的 " quick"）测压缩是测不出来的。
in_corpus = CORPUS[0]
raw_len = len(in_corpus.encode("utf-8"))
bpe_len = len(encode(in_corpus, merges, vocab))
assert bpe_len <= raw_len, f"BPE 编码不应比字节更长：{bpe_len} > {raw_len}"
assert bpe_len < raw_len, f"训练后的 BPE 应压缩语料内句子（{raw_len} -> {bpe_len}）"
print(f"[压缩] 语料内句子 {raw_len}B -> {bpe_len} tokens "
      f"({raw_len / max(1, bpe_len):.2f} 字节/token)")

out_of_corpus = "The quick brown fox"
o_raw = len(out_of_corpus.encode("utf-8"))
o_bpe = len(encode(out_of_corpus, merges, vocab))
assert o_bpe <= o_raw, f"语料外文本也不应变长：{o_bpe} > {o_raw}"

# =====================================================================
# 4) 确定性：同语料两次训练结果完全一致
# =====================================================================
vocab2, merges2 = train_bpe(CORPUS, vocab_size=VOCAB_SIZE)
assert merges == merges2, "两次训练的 merge 表不同，训练不确定"
assert vocab == vocab2, "两次训练的词表不同"

# =====================================================================
# 5) 语料耗尽时应提前停止，而非硬凑到目标词表
# =====================================================================
small_vocab, small_merges = train_bpe(["ab"], vocab_size=1000)
assert len(small_merges) < 1000 - BASE_VOCAB_SIZE, "语料耗尽时应提前停止，而非死循环"
assert len(small_vocab) == BASE_VOCAB_SIZE + len(small_merges)

# =====================================================================
# 6) 落盘与加载往返
# =====================================================================
with tempfile.TemporaryDirectory() as td:
    vocab_path = os.path.join(td, "vocab.json")
    merges_path = os.path.join(td, "merges.txt")
    save_bpe(vocab, merges, vocab_path, merges_path)
    assert os.path.exists(vocab_path) and os.path.exists(merges_path), "文件未落盘"

    # merges.txt 每行必须是「两个 token」，且 token 内部不能含空白
    # （初稿直接写原始字节，空格/换行本身就是字节 token，行会被切错）
    with open(merges_path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip() and not ln.startswith("#")]
    assert len(lines) == len(merges), f"merges.txt 行数应为 {len(merges)}，实得 {len(lines)}"
    for ln in lines:
        parts = ln.split(" ")
        assert len(parts) == 2, f"merges.txt 某行不是两个 token: {ln!r}"
        for p in parts:
            assert p and not any(c.isspace() for c in p), \
                f"merges.txt 某行含空白字符的 token: {ln!r}"

    loaded_vocab, loaded_merges = load_bpe(vocab_path, merges_path)
    assert loaded_vocab == vocab, "词表加载后不一致"
    assert loaded_merges == merges, "merge 表加载后不一致"
    for text in CORPUS:
        assert decode(encode(text, loaded_merges, loaded_vocab), loaded_vocab) == text

# =====================================================================
# 7) BPETokenizer：对外接口与 CharTokenizer 对齐
# =====================================================================
with tempfile.TemporaryDirectory() as td:
    vocab_path = os.path.join(td, "vocab.json")
    merges_path = os.path.join(td, "merges.txt")
    save_bpe(vocab, merges, vocab_path, merges_path)
    tk = BPETokenizer(vocab_path, merges_path)

    assert tk.vocab_size == len(vocab), f"vocab_size 应为 {len(vocab)}，实得 {tk.vocab_size}"
    assert len(tk.merges) == len(merges), "merges 数量不一致"
    assert len(tk.stoi) == tk.vocab_size, "stoi 与 vocab_size 不等长（存在重复 token 字符串）"
    assert tk.stoi["<|endoftext|>"] == ENDOFTEXT_ID, "特殊 token 应可按名字查到"

    for text in CORPUS:
        ids = tk.encode(text)
        assert tk.decode(ids) == text, f"BPETokenizer 往返失败: {text!r}"

    # itos / stoi 自洽：任意 id 都能查到它的字符串，再查回来还是同一个 id
    for i in range(tk.vocab_size):
        assert tk.stoi[tk.itos[i]] == i, f"itos/stoi 在 id {i} 上不自洽"

    # 空格字节显示成 'Ġ'（GPT-2 约定），便于肉眼读 token
    assert tk.itos[ord(" ") + 1] == "Ġ", "空格字节的显示形式应为 'Ġ'"

# =====================================================================
# 8) 真实落盘词表（data/vocab.json + data/merges.txt）自检
#    训练/推理真正用的是这份产物，它必须与 bpe.py 的编码逻辑完全一致
# =====================================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
real_vocab_path = os.path.join(ROOT, "data", "vocab.json")
real_merges_path = os.path.join(ROOT, "data", "merges.txt")
if os.path.exists(real_vocab_path) and os.path.exists(real_merges_path):
    rvocab, rmerges = load_bpe(real_vocab_path, real_merges_path)
    assert rvocab[ENDOFTEXT_ID] == b"<|endoftext|>", "真实词表的 id 0 不是特殊 token"
    assert len(rvocab) == BASE_VOCAB_SIZE + len(rmerges), "真实词表大小与 merge 数不匹配"
    # 256 个字节 token 一个都不能少（初稿的 id 撞车 bug 就在这里）
    for byte_val in range(256):
        assert rvocab[byte_val + 1] == bytes([byte_val]), \
            f"真实词表缺字节 token {byte_val}（id {byte_val + 1}）"
    rtk = BPETokenizer(real_vocab_path, real_merges_path)
    for text in CORPUS:
        assert rtk.decode(rtk.encode(text)) == text, f"真实词表往返失败: {text!r}"
    # 真实语料里的句子应被明显压缩
    probe = "The student trained a tiny transformer."
    r_raw = len(probe.encode("utf-8"))
    r_bpe = len(rtk.encode(probe))
    assert r_bpe < r_raw, f"真实词表对语料内句子无压缩：{r_bpe} >= {r_raw}"
    print(f"[真实词表] vocab={rtk.vocab_size} merges={len(rtk.merges)} "
          f"'{probe[:20]}' {r_raw}B -> {r_bpe} tokens")
else:
    print("[真实词表] 未找到 data/vocab.json，跳过（先跑 python data/build_vocab.py）")

print(f"test_bpe OK  vocab={len(vocab)} merges={len(merges)} "
      f"'{in_corpus[:20]}' {raw_len}B -> {bpe_len} tokens")

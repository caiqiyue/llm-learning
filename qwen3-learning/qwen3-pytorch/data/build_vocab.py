"""
训练 byte-level BPE 词表并落盘。

用法：python data/build_vocab.py

输出：data/vocab.json、data/merges.txt

【为什么不复用 GPT-2 线的语料】
    AGENTS.md 要求两条线代码完全隔离、不跨目录 import。
    本脚本自行合成语料，思路与 gpt-learning/gpt-learning-pytorch/data/pretrain_data.py 一致：
    用模板 + 随机填充生成可控长度的句子，语料分布稳定、可复现。

【语料从哪来】
    `data/corpus.py` 是合成语料的**唯一真源**：
    预训练语料（data/pretrain_data.py）与 BPE 训练语料（本脚本）用的是同一批句子。
    两边同 seed → 预训练语料是本脚本语料的前缀 → **词表完全覆盖预训练语料**。

【训练要多快】
    预分词后语料只剩几十种唯一种块，743 轮合并只在这几十种块上跑，
    总体是秒级（初稿按整段字节跑要十几分钟）。
"""

import os
import sys
from typing import List

# 保证以 `python data/build_vocab.py` 方式运行时能 import 到 config / data 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

from data.bpe import encode, save_bpe, train_bpe
from data.corpus import generate_sentences

VOCAB_PATH = os.path.join("data", "vocab.json")
MERGES_PATH = os.path.join("data", "merges.txt")

# BPE 训练语料的规模。比预训练语料大一些（config.pretrain_n_samples = 10000），
# 词表由更大的语料训出，对预训练语料的覆盖只会更好。
BPE_CORPUS_SIZE = 20000


def main() -> None:
    """训练并落盘 BPE 词表。"""
    corpus: List[str] = generate_sentences(
        n_samples=BPE_CORPUS_SIZE, seed=config.DENSE_CONFIG.seed
    )
    n_bytes = sum(len(s.encode("utf-8")) for s in corpus)
    print(f"[build_vocab] 语料 {len(corpus)} 句 / {n_bytes} 字节")

    vocab, merges = train_bpe(corpus, vocab_size=config.DENSE_CONFIG.bpe_vocab_size)
    print(f"[build_vocab] 词表 {len(vocab)}（合并 {len(merges)} 次，目标 "
          f"{config.DENSE_CONFIG.bpe_vocab_size}）")

    # 报告压缩比（取前 500 句）
    sample = corpus[:500]
    raw = sum(len(s.encode("utf-8")) for s in sample)
    n_tok = sum(len(encode(s, merges, vocab)) for s in sample)
    print(f"[build_vocab] 压缩比：{raw} 字节 -> {n_tok} tokens "
          f"({raw / max(1, n_tok):.2f} 字节/token)")

    save_bpe(vocab, merges, VOCAB_PATH, MERGES_PATH)


if __name__ == "__main__":
    main()

"""
训练 byte-level BPE 词表并落盘。

用法：python data/build_vocab.py

输出：data/vocab.json、data/merges.txt

【为什么不复用 GPT-2 线的语料】
    AGENTS.md 要求两条线代码完全隔离、不跨目录 import。
    本脚本自行合成语料，思路与 gpt-learning/gpt-learning-pytorch/data/pretrain_data.py 一致：
    用模板 + 随机填充生成可控长度的句子，语料分布稳定、可复现。

【为什么语料是「句子列表」而不是「一整段大字符串」（与初稿的差异）】
    BPE 训练前会做预分词（按空白切块），而预分词的边界依赖**输入单元的形状**：
      - 单句 "The cat sat."  → 块 ["The", " cat", " sat."]
      - 拼进大字符串 "\nThe cat sat." → 块 ["\\nThe", " cat", " sat."]
    换行会被吸附到下一个词上，导致「同一句话」在不同输入单元下切出不同的块，
    训练学到的合并规则也就对不上单独编码这一句时的结果。

    所以语料以**句子列表**返回，训练单元 = 编码单元。
    预训练阶段（Task 12）同样按「一条样本一个字符串」编码，规则才自洽。

【训练要多快】
    预分词后语料只剩几十种唯一种块，743 轮合并只在这几十种块上跑，
    总体是秒级（初稿按整段字节跑要十几分钟）。
"""

import os
import random
import sys
from typing import List

# 保证以 `python data/build_vocab.py` 方式运行时能 import 到 config / data 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

from data.bpe import encode, save_bpe, train_bpe

VOCAB_PATH = os.path.join("data", "vocab.json")
MERGES_PATH = os.path.join("data", "merges.txt")


# ---------------------------------------------------------------------------
# 语料合成
# ---------------------------------------------------------------------------
SUBJECTS = [
    "The student", "A teacher", "The engineer", "The cat", "The robot",
    "我的同学", "那位老师", "The company", "The model", "The system",
]
VERBS = [
    "studied", "built", "debugged", "trained", "evaluated", "wrote",
    "喜欢", "调试了", "训练了", "阅读了",
]
OBJECTS = [
    "a tiny transformer", "the attention mechanism", "a small language model",
    "the reward function", "a preference dataset",
    "微型 GPT-2", "位置编码", "RoPE 旋转编码", "一份偏好数据",
]
CONNECTORS = [".", ".", ".", "!", "?", "。"]


def build_corpus(n_samples: int = 20000, seed: int = 42) -> List[str]:
    """
    合成 BPE 训练语料。

    语料要足够大、重复度足够高，BPE 才会学出有意义的合并规则；
    同时不能太大，否则训练循环（O(merges × 唯一种块数)）会变慢。

    参数：
        n_samples: 句子条数
        seed:      随机种子（保证可复现）
    返回：
        句子列表（每个元素是一条独立样本）
    """
    rng = random.Random(seed)
    lines: List[str] = []
    for _ in range(n_samples):
        parts = [
            rng.choice(SUBJECTS),
            rng.choice(VERBS),
            rng.choice(OBJECTS),
            rng.choice(CONNECTORS),
        ]
        # 英文句子用空格连接；中文主体后面不加空格（更自然，且块数更少）
        if parts[0].isascii():
            lines.append(" ".join(parts))
        else:
            lines.append("".join(parts))
    return lines


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    """训练并落盘 BPE 词表。"""
    corpus = build_corpus(n_samples=20000, seed=config.DENSE_CONFIG.seed)
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

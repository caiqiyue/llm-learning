"""
合成语料的唯一真源（纯 Python，不依赖 torch）。

【为什么单独一个文件】
    BPE 词表的训练语料（data/build_vocab.py）与预训练语料
    （data/pretrain_data.py）**必须是同一批句子**，否则词表对预训练语料的
    覆盖率会打折（合成语料的可合并对有限，语料换了规则就全变了）。

    初稿把模板写在 build_vocab.py 里、由 pretrain_data.py 反向 import，
    结果 pretrain_data.py 被迫依赖 torch（build_vocab → corpus 是纯 Python，
    反向 import 本身没问题，但两边互相 import 就成了环）。
    拆出本文件后：
        corpus.py  ← build_vocab.py（纯 Python，训 BPE）
        corpus.py  ← pretrain_data.py（要 torch，切块/Loader）
    两边只依赖 corpus.py，没有环。

【为什么语料是「句子列表」而不是「一整段大字符串」】
    BPE 预分词按空白切块，块的边界依赖输入单元的形状：
      单句 "The cat sat."          → 块 ["The", " cat", " sat."]
      拼进大字符串 "\\nThe cat sat." → 块 ["\\nThe", " cat", " sat."]
    换行会吸附到下一个词上，导致「同一句话」在不同输入单元下切出不同的块，
    训练学到的合并规则也就对不上单独编码这一句时的结果。

    所以**训练单元 = 编码单元**：BPE 训练、预训练切块、推理编码
    全都以「一句」为单位，句子之间用特殊 token <|endoftext|> 连接。

【可复现】
    全部随机性都来自显式传入的 seed，同样的 (n_samples, seed) 必然产出同样的句子列表。
"""

import random
from typing import List

# ---------------------------------------------------------------------------
# 语料模板
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


def generate_sentences(n_samples: int = 20000, seed: int = 42) -> List[str]:
    """
    合成句子列表（预训练语料 / BPE 训练语料共用）。

    参数：
        n_samples: 句子条数
        seed:      随机种子（保证可复现）
    返回：
        句子列表
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


def corpus_stats(sentences: List[str]) -> dict:
    """
    统计语料规模（写日志 / 断言用）。

    参数：
        sentences: 句子列表
    返回：
        {"n_sentences": 句子数, "n_bytes": UTF-8 字节数, "n_chars": 字符数}
    """
    return {
        "n_sentences": len(sentences),
        "n_bytes": sum(len(s.encode("utf-8")) for s in sentences),
        "n_chars": sum(len(s) for s in sentences),
    }

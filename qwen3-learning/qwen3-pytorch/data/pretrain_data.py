"""
预训练数据构造。

对应主线：gpt-learning/gpt-learning-pytorch/data/pretrain_data.py
接口 `make_pretrain_loader(split, batch_size, block_size, seed)` 与主线**同名同义**，
方便两条线的训练脚本写法对齐。代码独立重写，不跨目录 import（AGENTS.md 硬约束）。

【流程】
    合成句子 → 每句 BPE 编码 → 用 <|endoftext|> 拼成一条 token 长流
             → 按 block_size+1 切块 → (inputs, targets) 错位一位

【为什么要 +1】
    自回归训练需要「用位置 0..T-2 预测位置 1..T-1」。
    切 block_size+1 个 token，前 block_size 个当输入、后 block_size 个当 target，
    刚好错位对齐且不浪费边界。

【★ 与初稿的关键差异：文档分隔符用「特殊 token」而不是「它的字面文本」】
    初稿把句子用字面文本 "<|endoftext|>" 拼起来再整体编码 —— 这是错的：
      - 特殊 token <|endoftext|> 的 id 是 0，它**不是**字节序列，
        把它的字面文本喂进 byte-level BPE，编码出来的是 13 个普通字节 token
        （'<', '|', 'e', 'n', 'd', ...），模型永远学不会「用 id 0 表示文档边界」
      - 更糟的是，字面文本会打断预分词的块边界（"<|endoftext|>\nThe" 里
        换行前面多了 13 个字符），使得 BPE 训练时学到的合并规则对不上
    正确做法：**先编码每句，再在 token 层面用 id 0 连接**
        ids = [tok(s1), EOT, tok(s2), EOT, ...]
    这样既是「训练单元 = 编码单元」（与 BPE 训练一致），
    又让模型真正学到 id 0 是文档边界。

【train/val 划分】
    按「块下标的哈希」稳定划分（不是按前 90%/后 10%），
    避免顺序偏差（合成语料里句子高度重复，按顺序切会让 train/val 分布不一致）。
"""

import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from data.corpus import generate_sentences

ENDOFTEXT = "<|endoftext|>"


# ---------------------------------------------------------------------------
# 语料合成（转发到唯一真源 data/corpus.py）
# ---------------------------------------------------------------------------
def generate_corpus(n_samples: int = 10000, seed: int = 42) -> List[str]:
    """
    合成预训练语料。

    参数：
        n_samples: 句子条数
        seed:      随机种子
    返回：
        句子列表（**不是**拼好的长字符串 —— 见模块 docstring 的「★」段）
    """
    return generate_sentences(n_samples=n_samples, seed=seed)


# ---------------------------------------------------------------------------
# token 流
# ---------------------------------------------------------------------------
def build_token_stream(sentences: Sequence[str], tk) -> List[int]:
    """
    把句子列表编成一条 token 长流，句子之间插特殊 token <|endoftext|>。

    参数：
        sentences: 句子列表
        tk:        BPETokenizer 实例
    返回：
        token id 列表，形如 [tok(s1), EOT, tok(s2), EOT, ...]
    """
    eot = tk.stoi[ENDOFTEXT]
    ids: List[int] = []
    for s in sentences:
        ids.extend(tk.encode(s))
        ids.append(eot)
    return ids


# ---------------------------------------------------------------------------
# 切块
# ---------------------------------------------------------------------------
def build_blocks(token_ids: List[int], block_size: int) -> List[Tuple[List[int], List[int]]]:
    """
    把 token 流切成 (inputs, targets) 块。

    参数：
        token_ids: 完整 token 序列
        block_size: 每块的输入长度
    返回：
        [(inputs, targets), ...]，inputs 与 targets 长度均为 block_size，
        targets = inputs 左移一位（末尾不足一块则丢弃）
    """
    blocks: List[Tuple[List[int], List[int]]] = []
    step = block_size + 1        # 多留 1 个 token 给 label shift
    for i in range(0, len(token_ids) - step + 1, step):
        chunk = token_ids[i:i + step]
        blocks.append((chunk[:-1], chunk[1:]))
    return blocks


class PretrainDataset(Dataset):
    """
    预训练数据集：返回 (inputs, targets) 两个 LongTensor。

    参数：
        blocks: [(inputs_list, targets_list), ...]
    """

    def __init__(self, blocks: List[Tuple[List[int], List[int]]]) -> None:
        """
        参数：
            blocks: 切好的块
        """
        self.blocks = blocks

    def __len__(self) -> int:
        """块的数量。"""
        return len(self.blocks)

    def __getitem__(self, idx: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        取第 idx 块。

        参数：
            idx: 块下标
        返回：
            (inputs, targets)，各 (block_size,) 的 LongTensor
        """
        x, y = self.blocks[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ---------------------------------------------------------------------------
# DataLoader 工厂
# ---------------------------------------------------------------------------
def _stable_train_val_mask(sample_id: int, train_ratio: float = 0.9) -> bool:
    """
    用下标哈希稳定地划分 train/val，保证每次运行划分一致。

    参数：
        sample_id: 块的下标
        train_ratio: 训练集占比
    返回：
        True 表示属于训练集
    """
    # 用乘法散列把下标打散，避免「前 90% 是训练集」的顺序偏差
    h = (sample_id * 2654435761) % 1000 / 1000.0
    return h < train_ratio


def make_pretrain_loader(
    split: str = "train",
    batch_size: int = 32,
    block_size: int = 128,
    seed: int = 42,
    vocab_path: Optional[str] = None,
    merges_path: Optional[str] = None,
) -> DataLoader:
    """
    构造预训练 DataLoader（与 GPT-2 线同名函数语义一致）。

    参数：
        split:      "train" 或 "val"
        batch_size: batch 大小
        block_size: 序列长度
        seed:       随机种子（语料生成用）
        vocab_path: 词表路径，默认 data/vocab.json
        merges_path:merge 表路径，默认 data/merges.txt
    返回：
        DataLoader，产出 (x, y) 各 (B, block_size) 的 LongTensor
    """
    from config import DENSE_CONFIG
    from data.tokenizer import BPETokenizer

    vocab_path = vocab_path or os.path.join("data", "vocab.json")
    merges_path = merges_path or os.path.join("data", "merges.txt")
    tk = BPETokenizer(vocab_path, merges_path)

    sentences = generate_corpus(n_samples=DENSE_CONFIG.pretrain_n_samples, seed=seed)
    ids = build_token_stream(sentences, tk)
    blocks = build_blocks(ids, block_size)

    ratio = DENSE_CONFIG.train_ratio
    if split == "train":
        sel = [b for i, b in enumerate(blocks) if _stable_train_val_mask(i, ratio)]
    else:
        sel = [b for i, b in enumerate(blocks) if not _stable_train_val_mask(i, ratio)]

    return DataLoader(PretrainDataset(sel), batch_size=batch_size, shuffle=(split == "train"))

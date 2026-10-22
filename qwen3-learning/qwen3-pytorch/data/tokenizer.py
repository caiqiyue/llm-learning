"""
BPE tokenizer。

对应主线：gpt-learning/gpt-learning-pytorch/data/tokenizer.py 的 CharTokenizer

接口刻意与 CharTokenizer 对齐（encode / decode / vocab_size / stoi / itos），
这样两条线的训练脚本写法一致，便于横向对照。

【与字符级的对比】
    CharTokenizer: vocab ≈ 64，一个字符一个 token，序列长
    BPETokenizer:  vocab ≈ 1000+，常见子词一个 token，序列短

【itos / stoi 里存的是什么】
    存的是「token 的可打印字符串形式」，不是原始字节。
    字节 0x20（空格）在 GPT-2 体系里显示成 'Ġ'（U+0120），
    字节 0x0A（换行）显示成 'Ċ'（U+010A）。
    这样每个 token 恰好一个字符，merges.txt 才能用空格分隔，
    且与 HuggingFace tokenizers 的格式兼容（见 data/bpe.py 的 bytes_to_unicode）。
"""

from typing import Dict, List, Tuple

from data.bpe import bytes_to_token_str, decode, encode, load_bpe


class BPETokenizer:
    """
    byte-level BPE tokenizer。

    参数：
        vocab_path:  vocab.json 路径
        merges_path: merges.txt 路径
    """

    def __init__(self, vocab_path: str, merges_path: str) -> None:
        """
        参数：
            vocab_path:  词表路径（data/build_vocab.py 生成）
            merges_path: merge 表路径
        """
        self.vocab_path = vocab_path
        self.merges_path = merges_path
        # 内部结构：{id: bytes} 与 [(a_id, b_id), ...]
        self._vocab_bytes, self.merges = load_bpe(vocab_path, merges_path)
        # 外部可读结构：itos 是 {id: 单字符/多字符}，stoi 是它的反查表
        self.itos: Dict[int, str] = {
            i: bytes_to_token_str(b) for i, b in self._vocab_bytes.items()
        }
        self.stoi: Dict[str, int] = {s: i for i, s in self.itos.items()}
        # 「块 → id 列表」缓存：块的编码与上下文无关，同一块只算一次。
        # 10 万块的合成语料从几十秒降到接近线性。
        self._chunk_cache: Dict[str, List[int]] = {}

    # ------------------------------------------------------------------
    # 公开接口（对齐 CharTokenizer）
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        """词表大小。"""
        return len(self._vocab_bytes)

    def encode(self, s: str) -> List[int]:
        """
        把文本编码成 token id 列表（带块级缓存）。

        参数：
            s: 输入文本
        返回：
            token id 列表
        """
        return encode(s, self.merges, self._vocab_bytes, cache=self._chunk_cache)

    def decode(self, ids: List[int]) -> str:
        """
        把 token id 列表解码回文本。

        参数：
            ids: token id 列表
        返回：
            原始文本
        """
        return decode(ids, self._vocab_bytes)

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    def encode_with_tokens(self, s: str) -> List[Tuple[int, str]]:
        """
        编码并同时返回每个 token 的可读形式（调试 / 笔记演示用）。

        参数：
            s: 输入文本
        返回：
            [(id, token 字符串), ...]
        """
        return [(i, self.itos.get(i, "?")) for i in self.encode(s)]

    def __repr__(self) -> str:
        """便于 print 的简短表示。"""
        return f"BPETokenizer(vocab_size={self.vocab_size}, merges={len(self.merges)})"

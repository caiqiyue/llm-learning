"""
字符级 tokenizer。
- 从 vocab.txt 读字符表（每行一个 token，# 开头的行视作注释）
- 提供 encode / decode / vocab_size / stoi / itos
- 不依赖 torch / tensorflow，P0 阶段可独立 import

特殊 token 常量（与 dataset_construction.md §1.3 一致）：
  PAD_ID=1  BOS_ID=2  EOS_ID=3  SEP_ID=4  UNK_ID=0
"""

# 特殊 token id（与 build_vocab.py 写入 vocab.txt 的前 5 行顺序一致）
UNK_ID = 0
PAD_ID = 1
BOS_ID = 2
EOS_ID = 3
SEP_ID = 4


class CharTokenizer:
    """字符级 tokenizer：从 vocab.txt 读 token 表，提供 encode / decode。"""

    def __init__(self, vocab_path: str):
        """参数：
            vocab_path: vocab.txt 文件路径；每行一个 token；# 开头为注释行。
        """
        self.stoi = {}  # string -> int
        self.itos = []  # int -> string
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                tok = line.rstrip("\n")
                if not tok or tok.startswith("#"):
                    continue
                if tok in self.stoi:
                    # 重复 token 防御：跳过
                    continue
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)
        self.vocab_size = len(self.itos)

    def encode(self, s: str):
        """字符串 → id 列表。未知字符映射为 UNK_ID。"""
        return [self.stoi.get(c, UNK_ID) for c in s]

    def decode(self, ids):
        """id 列表 → 字符串。未知 id 跳过。"""
        chars = []
        for i in ids:
            i = int(i)
            if 0 <= i < self.vocab_size:
                chars.append(self.itos[i])
        return "".join(chars)

    def __repr__(self):
        return f"CharTokenizer(vocab_size={self.vocab_size})"

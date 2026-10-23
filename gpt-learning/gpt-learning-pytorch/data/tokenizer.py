"""
字符级 tokenizer。
- 从 vocab.txt 读字符表（每行一个 token，# 开头的行视作注释）
- 提供 encode / decode / vocab_size / stoi / itos
- 不依赖 torch / tensorflow，P0 阶段可独立 import

特殊 token 常量（与 dataset_construction.md §1.3 一致）：
  PAD_ID=1  BOS_ID=2  EOS_ID=3  SEP_ID=4  UNK_ID=0

【feat-027】特殊 token 必须是**整体一个 id**，而不是它的每个字符：
  词表里 "<bos>" / "<sep>" / "<eos>" 是**多字符**的一行，若 encode 仍然逐字符查表，
  "<sep>" 会被拆成 5 个 id（'<' 不在字符表里 → UNK），于是
  BOS_ID / SEP_ID / EOS_ID 三个 id **永远不会被产生**：
  · 模型从没学过吐 EOS ⇒ 生成永远跑满 max_new_tokens，不会自己停；
  · decode 回来还是 "<unk>sep<unk>"，不是 "<sep>"。
  所以 encode 改成「最长匹配优先」：先匹配特殊 token（多字符的词表条目），
  匹配不上再逐字符查表。于是 decode(encode(s)) == s 成为不变量
  （前提是 s 里的字符都在词表里）。
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

        # 【feat-027】特殊 token = 词表里**多字符**的条目（"<bos>" / "<sep>" / "<eos>" …）
        # 按长度降序排列 ⇒ encode 时「最长匹配优先」，"<bos>" 不会被 "<b" 之类截断。
        # 纯单字符的词表条目走逐字符查表那条路，行为与从前完全一致。
        self._specials = sorted(
            ((tok, tid) for tok, tid in self.stoi.items() if len(tok) > 1),
            key=lambda kv: -len(kv[0]),
        )

    def encode(self, s: str):
        """字符串 → id 列表。

        规则（feat-027）：
          1. 先按「最长匹配优先」匹配特殊 token（多字符词表条目）→ 整体变成 1 个 id；
          2. 匹配不上的字符逐个查表，查不到 → UNK_ID。

        参数：
            s: 待编码字符串
        返回：
            List[int]：token id 列表
        """
        if not s:
            return []
        ids = []
        i = 0
        n = len(s)
        while i < n:
            hit = None
            for tok, tid in self._specials:
                if s.startswith(tok, i):
                    hit = (tok, tid)
                    break
            if hit is not None:
                tok, tid = hit
                ids.append(tid)
                i += len(tok)
            else:
                ids.append(self.stoi.get(s[i], UNK_ID))
                i += 1
        return ids

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

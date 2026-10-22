"""
从零手写的 byte-level BPE（Byte Pair Encoding）。

对应笔记：gpt-learning/1.bpe-tokenizer.md
对照主线：gpt-learning/gpt-learning-pytorch/data/tokenizer.py（字符级 CharTokenizer）

【为什么 Qwen3 线用 BPE 而 GPT-2 线用字符级】
    字符级：每个字符一个 token，vocab ≈ 64，但序列很长（128 字符 = 128 token）
    BPE：   常见子词合并成一个 token，同样内容序列更短 → 同样 block_size 能装更多语义
    真实 Qwen3 / GPT-2 用的都是 BPE。本线手写它，**不引入 sentencepiece/tokenizers**
    —— 见 AGENTS.md 的依赖上限约束。

【训练算法（三步循环）】
    1. 文本 → 预分词 → UTF-8 字节 → 初始 token 序列（每个字节一个 token，id = 字节值 + 1）
    2. 统计所有相邻 token 对 (a, b) 的出现频次
    3. 取频次最高的对合并成新 token（id 从 257 递增），从语料中替换所有该对
    重复 2-3，直到达到目标词表大小，或没有可合并的对了

【byte-level 的含义】
    「byte」指基础单元直接是原始字节（0..255），而不是字符。
    这样任何文本（含 emoji、中文、代码）都能被覆盖，不需要 UNK 字符。

【id 分配（改过初稿，这里是修正后的版本）】
    id 0        → <|endoftext|>（特殊 token，不参与任何合并）
    id 1..256   → 单字节 token（字节值 = id - 1）
    id 257+i    → 第 i 次合并产生的 token

    ⚠️ 初稿把合并 token 的起始 id 定成 256，与「字节 0xFF 占 id 256」**撞车**，
    导致 0xFF 这个字节 token 被静默吞掉。修正后合并 token 从 257 开始。

【为什么序列化要用 byte→unicode 映射】
    字节 token 里有空格（0x20）和换行（0x0A）。如果直接把 token 的原始字节
    写进 merges.txt，一行就会变成 " a b" 这种形式，用空格分隔根本解析不了
    （初稿的 `line.split(" ")` 会炸）。

    GPT-2 的做法是加一层 byte→unicode 的**双射**映射：把 0..255 映射到
    32 个可打印 ASCII + 若干 Latin-1 字符 + 256+n 个 U+0100 段的字符，
    映射后的字符**一定不是空格、不是换行**，于是 merges.txt 可以用空格分隔，
    且与 HF tokenizers 的格式兼容。

    内部表示仍然是 {id: bytes}，映射只用于落盘/读盘。

【编码 vs 训练是两回事】
    训练：统计全局频次 → 得到「合并优先级列表」（merges）
    编码：按 merges 的先后顺序把相邻对合并掉，与训练无关，只依赖 merges
"""

import json
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

# 特殊 token 的 id 固定为 0，保证不会被任何字节 token 或合并结果占用
ENDOFTEXT = "<|endoftext|>"
ENDOFTEXT_ID = 0
# 基础词表：1 个特殊 token + 256 个字节 token
BASE_VOCAB_SIZE = 257
# 合并产生的 token 从这个 id 开始（257，避免与字节 token 的 id 1..256 撞车）
MERGE_OFFSET = 257

# 预分词正则：把文本切成「空白 + 非空白」的块，块内不再切分。
# 与 GPT-2 的正则思路一致（词首空白跟词走），但本线不区分词类，
# 目的是**只做加速**：合并规则不跨块，块数远小于字节数。
# 这个切分是无损的：把所有块拼回去 === 原文本。
_PRETOKEN_RE = re.compile(r"\s*\S+|\s+")


def bytes_to_unicode() -> Dict[int, str]:
    """
    构造 byte → unicode 的双射（GPT-2 的做法）。

    规则：
      - 0x21..0x7E（可打印 ASCII）、0xA1..0xAC、0xAE..0xFF 映射到自身
      - 其余字节（控制字符、0x20 空格、0x7F..0xA0）映射到 U+0100 起的连续码位

    这样映射后的字符**一定不是空格也不是换行**，merges.txt 才能用空格分隔。

    返回：
        {字节值: 单字符}
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def unicode_to_bytes() -> Dict[str, int]:
    """
    构造 unicode → byte 的反查表（bytes_to_unicode 的逆）。

    返回：
        {单字符: 字节值}
    """
    return {v: k for k, v in bytes_to_unicode().items()}


_B2U = bytes_to_unicode()
_U2B = unicode_to_bytes()


def bytes_to_token_str(piece: bytes) -> str:
    """
    把一个 token 的字节串转成可落盘的单字符串形式。

    因为 B2U 是逐字节的双射，合并 token 的字符串就是各字节字符串的拼接。

    参数：
        piece: token 的字节串
    返回：
        可落盘的字符串
    """
    return "".join(_B2U[b] for b in piece)


def token_str_to_bytes(s: str) -> bytes:
    """
    把可落盘的字符串形式还原成字节串（bytes_to_token_str 的逆）。

    参数：
        s: 可落盘的字符串
    返回：
        字节串
    """
    return bytes(_U2B[c] for c in s)


def pretokenize(text: str) -> List[str]:
    """
    预分词：把文本切成「空白 + 非空白」的块。

    【为什么需要它】
        BPE 的合并统计是 O(merges × 语料长度) 的纯 Python 循环。
        语料直接按字节处理（1MB）时，743 轮合并要跑十几分钟。
        先切成块并**按块去重计数**后，参与统计的只有「唯一种块」，
        同样的语料规模下合并过程从十几分钟降到几秒。

    【为什么这样切是无损的】
        所有块拼回去 === 原文本，所以 encode→decode 仍能逐字节还原。
        代价是合并规则不跨块边界（`" th"` 和 `"the "` 是两个块），
        这与 GPT-2 的做法同源（GPT-2 的正则也不跨某些边界）。

    参数：
        text: 输入文本
    返回：
        块的列表
    """
    return _PRETOKEN_RE.findall(text)


def _iter_pairs(ids: Sequence[int]) -> Counter:
    """
    统计一个 token 序列里所有相邻对的出现次数。

    参数：
        ids: token id 序列
    返回：
        Counter，键为 (a_id, b_id)，值为出现次数
    """
    return Counter(zip(ids, ids[1:]))


def train_bpe(
    texts: Iterable[str],
    vocab_size: int,
    verbose: bool = False,
) -> Tuple[Dict[int, bytes], List[Tuple[int, int]]]:
    """
    在语料上训练 BPE。

    【训练循环】
        1. 文本 → 预分词 → UTF-8 字节 → 唯一种块（带出现次数）
        2. Counter 统计所有相邻对 (a, b) 的频次（按块的重复次数加权）
        3. 取频次最高的对；并列时取 (a, b) 字典序最小者，保证确定性
        4. 新 id = 257 + i，把语料里所有 (a, b) 原地替换成新 id
        5. 没有频次 ≥ 2 的对就提前退出（合并只出现一次的对没有收益）

    【加速的关键】
        第 2、4 步只处理**唯一种块**，重复的块只算一次频次。
        合成语料（模板 + 随机填充）的唯一种块通常只有几百个，
        743 轮合并的总耗时从十几分钟降到几秒。

    参数：
        texts:      语料
        vocab_size: 目标词表大小（总 id 数，含 1 特殊 + 256 字节 token）
        verbose:    True 时每轮打印
    返回：
        (vocab, merges)
            vocab:  {id: token 的字节串}
            merges: [(a_id, b_id), ...]，按合并先后顺序（编码时按此顺序应用）
    """
    n_target = max(0, vocab_size - BASE_VOCAB_SIZE)   # 合并次数 = 词表 - 257

    # --- 1) 预分词 + 去重计数 ---
    chunk_counts: Counter = Counter()
    for text in texts:
        chunk_counts.update(pretokenize(text))

    # 语料预编码成字节序列：list[list[int]]，weights 是每个块的重复次数
    # ⚠️ 这里的 id 必须是「字节值 + 1」，与 encode / build_vocab 保持一致。
    #    写成原始字节值会整体错位 1（merge 记录的是字节值，词表里却是字节值+1），
    #    结果合并规则全错、编码几乎不压缩。
    chunks: List[List[int]] = [[b + 1 for b in c.encode("utf-8")] for c in chunk_counts]
    weights: List[int] = list(chunk_counts.values())
    merges: List[Tuple[int, int]] = []

    for i in range(n_target):
        # --- 2) 统计全部相邻对频次（按块权重累加）---
        counts: Counter = Counter()
        for ids, w in zip(chunks, weights):
            for pair in zip(ids, ids[1:]):
                counts[pair] += w
        if not counts:
            break   # 所有序列都只剩 1 个 token，无可合并

        # --- 3) 取最高频对；并列按 (a,b) 字典序，保证确定性 ---
        (a, b), freq = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if freq < 2:
            break   # 只出现一次，合并无收益且白占 id

        # --- 4) 原地替换所有 (a, b) 为新 id ---
        new_id = MERGE_OFFSET + i
        for ci, ids in enumerate(chunks):
            out: List[int] = []
            j = 0
            n = len(ids)
            while j < n:
                if j + 1 < n and ids[j] == a and ids[j + 1] == b:
                    out.append(new_id)
                    j += 2          # 跳过两个被合并的 token
                else:
                    out.append(ids[j])
                    j += 1
            chunks[ci] = out

        merges.append((a, b))
        if verbose:
            print(f"[bpe] merge {i}: ({a}, {b}) freq={freq} -> id={new_id}")

    return build_vocab(merges), merges


def build_vocab(merges: Sequence[Tuple[int, int]]) -> Dict[int, bytes]:
    """
    由 merge 表反推完整词表：id → 该 token 对应的字节串。

        id 0         → <|endoftext|>（特殊 token，无字节内容）
        id 1..256    → 对应的单字节（bytes([id-1])）
        id 257+i     → merges[i] 合并出来的字节串（两者字节串拼接）

    参数：
        merges: 合并顺序表
    返回：
        {id: bytes}
    """
    vocab: Dict[int, bytes] = {ENDOFTEXT_ID: ENDOFTEXT.encode("utf-8")}
    for i in range(256):
        vocab[i + 1] = bytes([i])
    for i, (a, b) in enumerate(merges):
        vocab[MERGE_OFFSET + i] = vocab[a] + vocab[b]
    return vocab


def encode(text: str, merges: Sequence[Tuple[int, int]], vocab: Dict[int, bytes]) -> List[int]:
    """
    用训练好的 merge 表把文本编码成 token id 序列。

    【与训练的区别】
        训练：统计全局频次，产出 merges
        编码：只按 merges 的**先后顺序**反复合并，与语料无关
        所以 encode 是纯函数：同样 (text, merges) 必然得到同样结果。

    【编码循环的形状】
        ┌──────────────────────────────────────────────────────────┐
        │ 1. 文本 → 预分词 → 每块 UTF-8 字节 → 初始 id（字节值+1）│
        │ 2. for (a, b) in merges:                                 │
        │      把该块里所有相邻 (a, b) 换成 257+i                   │
        │ 3. 拼接所有块的 id 序列                                   │
        └──────────────────────────────────────────────────────────┘
    为什么按 merges 顺序逐个应用，而不是循环到不动点？
        因为 merges 是**严格按训练时的优先级**排好的：靠前的合并规则优先级更高。
        顺序应用一次即可得到与训练语料一致的结果，且只需 O(len(merges) × T)。

    参数：
        text:   输入文本
        merges: merge 表
        vocab:  词表（encode 实际不需要，但要传以便校验）
    返回：
        token id 列表
    """
    ids: List[int] = []
    for chunk in pretokenize(text):
        chunk_ids: List[int] = [b + 1 for b in chunk.encode("utf-8")]
        for i, (a, b) in enumerate(merges):
            # 快路径：块里根本没有 a 这个 id，这一轮必然无事可做
            if a not in chunk_ids:
                continue
            new_id = MERGE_OFFSET + i
            out: List[int] = []
            changed = False
            j = 0
            n = len(chunk_ids)
            while j < n:
                if j + 1 < n and chunk_ids[j] == a and chunk_ids[j + 1] == b:
                    out.append(new_id)
                    j += 2
                    changed = True
                else:
                    out.append(chunk_ids[j])
                    j += 1
            if not changed:
                continue
            chunk_ids = out
            # 若整个块已合并成一个，提前结束
            if len(chunk_ids) == 1:
                break
        ids.extend(chunk_ids)
    return ids


def decode(ids: Sequence[int], vocab: Dict[int, bytes]) -> str:
    """
    把 token id 序列解码回文本。

    参数：
        ids:  token id 序列
        vocab: {id: bytes}
    返回：
        原始文本（字节非法时用 replacement 字符兜底）
    """
    buf = bytearray()
    for i in ids:
        piece = vocab.get(i)
        if piece is None:
            continue          # 越界 id 跳过（对齐 GPT-2 线 decode 的容错风格）
        buf.extend(piece)
    return buf.decode("utf-8", errors="replace")


def save_bpe(
    vocab: Dict[int, bytes],
    merges: Sequence[Tuple[int, int]],
    vocab_path: str,
    merges_path: str,
) -> None:
    """
    落盘：vocab.json（token 可读形式）+ merges.txt（每行 "a b"）。

    vocab.json 的格式：{"<id>": "<token 的字符串形式>"}
        字节 token 存成 byte→unicode 映射后的单字符
        （保证 JSON 可读、且映射后不含空格/换行，可被空格分隔解析）

    参数：
        vocab:      {id: bytes}
        merges:     merge 表
        vocab_path: 词表输出路径
        merges_path:merge 表输出路径
    """
    os.makedirs(os.path.dirname(os.path.abspath(vocab_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(merges_path)), exist_ok=True)

    # bytes → 可落盘字符串，保证 JSON 里每个 token 恰好一个字符
    vocab_str = {str(k): bytes_to_token_str(v) for k, v in vocab.items()}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_str, f, ensure_ascii=False, indent=1)

    with open(merges_path, "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")     # 与 HF tokenizers 的格式约定保持一致
        for a, b in merges:
            f.write(f"{bytes_to_token_str(vocab[a])} {bytes_to_token_str(vocab[b])}\n")
    print(f"[bpe] saved vocab({len(vocab)}) -> {vocab_path}")
    print(f"[bpe] saved merges({len(merges)}) -> {merges_path}")


def load_bpe(vocab_path: str, merges_path: str) -> Tuple[Dict[int, bytes], List[Tuple[int, int]]]:
    """
    加载 vocab.json + merges.txt，还原成 encode 需要的结构。

    merges.txt 里存的是 token 的字符串形式，需要反查回 id 再组成 (a_id, b_id)。
    因为用了 byte→unicode 映射，字符串里不会出现空格/换行，按空格切分是安全的。

    参数：
        vocab_path:  词表路径
        merges_path: merge 表路径
    返回：
        (vocab, merges)，与 train_bpe 的返回结构一致
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_str = json.load(f)

    # 字符串 → id，反向查表
    str_to_id = {s: int(i) for i, s in vocab_str.items()}
    vocab: Dict[int, bytes] = {
        int(i): (ENDOFTEXT.encode("utf-8") if int(i) == ENDOFTEXT_ID else token_str_to_bytes(s))
        for i, s in vocab_str.items()
    }
    # 字节 → id（merges.txt 里只有普通 token，不会含特殊 token）
    bytes_to_id = {v: k for k, v in vocab.items()}

    merges: List[Tuple[int, int]] = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            a_s, b_s = line.split(" ")
            merges.append((bytes_to_id[token_str_to_bytes(a_s)],
                           bytes_to_id[token_str_to_bytes(b_s)]))

    print(f"[bpe] loaded vocab({len(vocab)}) merges({len(merges)})")
    return vocab, merges

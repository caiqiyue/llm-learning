# Qwen3 学习笔记：byte-level BPE Tokenizer

> **对应 GPT-2 实现位置**：[`gpt-learning/1.bpe-tokenizer.md`](../gpt-learning/1.bpe-tokenizer.md)（原理讲解）｜
> **`gpt-learning/gpt-learning-pytorch/data/tokenizer.py` 的 `CharTokenizer`**
> —— ⚠️ **GPT-2 线的代码里没有 BPE 实现**，它用的是**字符级** `CharTokenizer`
> （`vocab_size = 45`，未知字符一律映射成 `UNK_ID = 0`）。
> 本线是**从零手写** byte-level BPE，**不引入** `sentencepiece` / `tokenizers`（依赖上限约束）。
> **对应 Qwen3 源码位置**：[`qwen3-learning/qwen3-pytorch/data/bpe.py`](qwen3-pytorch/data/bpe.py)
> → `train_bpe` / `encode` / `decode` / `save_bpe` / `load_bpe`；
> [`qwen3-learning/qwen3-pytorch/data/tokenizer.py`](qwen3-pytorch/data/tokenizer.py) → `BPETokenizer`；
> [`qwen3-learning/qwen3-pytorch/data/build_vocab.py`](qwen3-pytorch/data/build_vocab.py) → 训练并落盘词表
> **对应可跑测试**：[`qwen3-learning/qwen3-pytorch/tests/test_bpe.py`](qwen3-pytorch/tests/test_bpe.py)（8 组断言）

---

## 0. 先记住五句话

1. **byte-level** = 基础单元是**原始字节**（0..255），不是字符 ⇒ 任何文本（含 emoji、中文、代码）都能编码，**永不 UNK**。
2. **BPE 训练** = 反复「统计最频繁的相邻 token 对 → 合并成一个新 token」，产出的是一张**合并优先级表**（merges）。
3. **BPE 编码** = 按 merges 的先后顺序把相邻对合并掉，**与语料无关**，是纯函数。
4. **id 分配**：0 = `<|endoftext|>`，1..256 = 单字节，**257+ = 合并产生的 token**。
   ⚠️ 合并 token 必须从 **257** 开始 —— 256 是字节 `0xFF` 的位置，撞车会把它**静默吞掉**。
5. **merges.txt 用空格分隔** ⇒ 落盘前必须做 **byte→unicode 双射**，
   否则「空格」和「换行」本身就是字节 token，一行会被切错。

---

## 1. 为什么 Qwen3 线要手写 BPE，而 GPT-2 线用字符级

两条线的 tokenizer 实测对照（同一句话，同样的 CPU 环境）：

| 维度 | GPT-2 线 `CharTokenizer` | Qwen3 线 `BPETokenizer` |
|---|---|---|
| 基础单元 | 字符 | **UTF-8 字节** |
| 词表大小 | **45** | **778** |
| 未知字符 | **映射成 `UNK_ID = 0`** | **没有 UNK**（byte-level 的直接后果） |
| `"The student trained a tiny transformer."` | **39 token**（= 字符数） | **6 token** |
| `"你好，这是一段中文测试语料。"` | 14 token，**14 个全是 UNK** | 42 字节 → **36 token，0 个 UNK** |

两个后果：

- **序列长度**：字符级下 `block_size = 128` 只能装 128 个字符；
  BPE 下同样 128 能装约 800 个字符的常见英文 ⇒ **同样的上下文预算能装更多语义**。
- **覆盖率**：字符级词表里没有中文字符 ⇒ 中文**全部变 UNK** ⇒ 模型学不到任何中文结构。
  byte-level 把这件事从根上消掉。

> 真实 Qwen3 / GPT-2 用的都是 BPE。GPT-2 线之所以用字符级，是因为那条线的目标是
> **把 RLHF 链路讲清楚**，tokenizer 越简单越好；Qwen3 线要体现「现代 tokenizer」，所以手写一份。
> 两边的 `encode / decode / vocab_size / stoi / itos` **接口刻意对齐**，
> 这样两边的训练脚本写法一致，能横向对照。

---

## 2. byte-level 到底「byte」在哪

`data/bpe.py` 的初始 id 分配：

```text
id 0        → <|endoftext|>   （特殊 token，不参与任何合并）
id 1..256   → 单字节 token     （字节值 = id - 1）
id 257+i    → 第 i 次合并产生的 token
```

「byte-level」的含义是：**基础单元直接是原始字节**。文本先按 UTF-8 编码成字节流，
每个字节占一个初始 token（id = 字节值 + 1）。

直接后果（都能在 `tests/test_bpe.py` 第 2 组里实测）：

```python
for text in ["€uro", "\U0001f642", "\t\n\r", "Ǆungla", "a b", "  ", "。"]:
    assert decode(encode(text, merges, vocab), vocab) == text
```

**语料里没出现过的字符也必须能逐字节还原** —— 这就是 byte-level 的意义。

### 2.1 ★ id 撞车：一个「不报错但静默吞掉 token」的 bug

初稿把合并 token 的起始 id 定成 **256**。而字节 token 占的是 `1..256`，
其中 **`id 256` 就是字节 `0xFF`**。两者撞车 ⇒ `0xFF` 这个字节 token 被**静默吞掉**：
词表少了一个 id，`0xFF` 无法编码，而且**代码不报任何错**。

修正后合并 token 从 **257** 开始，并且 `tests/test_bpe.py` 第 1 组把它钉成回归防线：

```python
assert vocab[256] == bytes([0xFF]), f"id 256 必须是字节 0xFF，实得 {vocab[256]!r}"
assert bytes([0xFF]) in vocab.values(), "字节 0xFF 在词表中消失了"
assert MERGE_OFFSET == 257, f"合并起始 id 应为 257，实得 {MERGE_OFFSET}"
```

**教训**：id 分配不是「随便挑个起点」，每一个 id 都要有明确的归属者。
`test_bpe.py` 里还有一条 `len(set(byte_strings)) == len(byte_strings)`
（词表中不允许出现内容相同的两个 token）—— 这是撞车的**另一面**。

---

## 3. ★ 为什么要 byte→unicode 映射（merges.txt 的空格分隔）

字节 token 里有**空格**（`0x20`）和**换行**（`0x0A`）。
如果直接把 token 的原始字节写进 `merges.txt`，一行会长成 `" a b"` 这种形式，
用空格分隔根本解析不了（`line.split(" ")` 会切出 3 段甚至更多）。

GPT-2 的做法是加一层 **byte→unicode 双射**：

- `0x21..0x7E`（可打印 ASCII）、`0xA1..0xAC`、`0xAE..0xFF` → 映射到自身；
- 其余字节（控制字符、空格、`0x7F..0xA0`）→ 映射到 `U+0100` 起的连续码位。

映射后的字符**一定不是空格、不是换行** ⇒ `merges.txt` 可以用空格分隔，
且与 HuggingFace `tokenizers` 的格式兼容。

```python
bs = (list(range(ord("!"), ord("~") + 1))
      + list(range(ord("\xa1"), ord("\xac") + 1))
      + list(range(ord("\xae"), ord("\xff") + 1)))
cs = bs[:]
n = 0
for b in range(256):
    if b not in bs:
        bs.append(b); cs.append(256 + n); n += 1
return dict(zip(bs, [chr(c) for c in cs]))
```

**内部表示仍然是 `{id: bytes}`，映射只用于落盘 / 读盘。**

肉眼可见的效果（`BPETokenizer.itos` 存的是「可打印字符串形式」）：

| 字节 | 显示成 | 码位 |
|---|---|---|
| 空格 `0x20` | `Ġ` | U+0120 |
| 换行 `0x0A` | `Ċ` | U+010A |

所以本线词表里看到的 `'Ġstudent'` 表示的是**「前面带一个空格的 student」**。
`test_bpe.py` 第 7 组把这条约定也钉住了：

```python
assert tk.itos[ord(" ") + 1] == "Ġ", "空格字节的显示形式应为 'Ġ'"
```

---

## 4. 预分词：为什么必须有，以及它是无损的

`pretokenize()` 用一条正则把文本切成「空白 + 非空白」的块：

```python
_PRETOKEN_RE = re.compile(r"\s*\S+|\s+")
```

| 作用 | 说明 |
|---|---|
| **加速** | BPE 的合并统计是 `O(merges × 语料长度)` 的纯 Python 循环。语料直接按字节处理（785 KB）时，521 轮合并要跑十几分钟。先切成块并**按块去重计数**后，参与统计的只有「唯一种块」（本线只有几百个）⇒ **十几分钟降到几秒** |
| **无损** | 所有块拼回去 === 原文本 ⇒ `encode → decode` 仍能逐字节还原 |
| **代价** | 合并规则**不跨块边界**（`" th"` 和 `"the "` 是两个块） |

「不跨块」不是 bug，是 GPT-2 的正则就有的性质。代价是：**语料外的文本压缩效果明显差于语料内**。
`test_bpe.py` 第 3 组因此把压缩断言分成两条：

```python
in_corpus = CORPUS[0]           # 语料内：必须**严格压缩**
assert bpe_len < raw_len, ...
out_of_corpus = "The quick brown fox"   # 语料外：只保证**不变长**
assert o_bpe <= o_raw, ...
```

> ⚠️ 用语料外的字符串测压缩是**测不出来的** —— 训练学到的合并规则只在语料里出现过的块上有效。
> 这条「断言必须用对样本」和 feat-017 的「全关 < 阈值必须有反证」是同一类教训。

---

## 5. 训练算法：三步循环

```text
1. 文本 → 预分词 → UTF-8 字节 → 初始 token 序列（id = 字节值 + 1）
2. 统计所有相邻 token 对 (a, b) 的出现频次
3. 取频次最高的对合并成新 token（id 从 257 递增），从语料中替换所有该对
   重复 2-3，直到达到目标词表大小，或没有可合并的对了
```

`train_bpe()` 里对应的四行：

```python
# 2) 统计全部相邻对频次（按块权重累加）
counts: Counter = Counter()
for ids, w in zip(chunks, weights):
    for pair in zip(ids, ids[1:]):
        counts[pair] += w
# 3) 取最高频对；并列按 (a,b) 字典序，保证确定性
(a, b), freq = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))
if freq < 2:
    break   # 只出现一次，合并无收益且白占 id
# 4) 原地替换所有 (a, b) 为新 id
new_id = MERGE_OFFSET + i
```

三个细节：

| 细节 | 为什么 |
|---|---|
| **并列取字典序最小** | 保证训练**确定性** —— 同语料两次训练结果必须逐字节一致（`test_bpe.py` 第 4 组） |
| **频次 < 2 就停** | 只出现一次的对，合并后**省不下任何 token**，却白占一个 id |
| **初始 id 必须是「字节值 + 1」** | 初稿写成原始字节值会**整体错位 1**（merge 记录的是字节值，词表里却是字节值+1）⇒ 合并规则全错、编码几乎不压缩 |

**本线实测**：语料 20000 句 / 785065 字节 → 目标词表 1000，实际只训出 **778**
（合并 **521** 次后没有频次 ≥ 2 的对了，提前退出）。

> ⚠️ **778 < 1000 不是 bug**，是**语料多样性问题**：合成语料（模板 + 随机填充）的
> 唯一种块只有几百个，可合并的对用尽了。真实语料不会这样。

---

## 6. 编码 vs 训练：两件不同的事

| | 训练 `train_bpe` | 编码 `encode` |
|---|---|---|
| 输入 | 语料 + 目标词表大小 | 文本 + 已训好的 merges |
| 输出 | `(vocab, merges)` | token id 列表 |
| 依赖语料吗 | **是**（要统计频次） | **否**（只看 merges） |
| 性质 | 慢 | 纯函数：同样 `(text, merges)` 必然同样结果 |

编码循环的形状：

```text
1. 文本 → 预分词 → 每块 UTF-8 字节 → 初始 id（字节值+1）
2. for (a, b) in merges:        ← 按训练时的优先级顺序，逐条应用
      把该块里所有相邻 (a, b) 换成 257+i
3. 拼接所有块的 id 序列
```

**为什么按 merges 顺序应用一次就够，而不是循环到不动点？**
因为 merges 是**严格按训练时的优先级**排好的：靠前的合并规则优先级更高。
顺序应用一次即可得到与训练语料一致的结果，复杂度 `O(len(merges) × T)`。

**块级缓存**：`encode()` 接受一个可选的 `cache: {块文本 → id 列表}`。
因为块的编码与上下文无关（合并不跨块），同一块只算一次。
合成语料的唯一种块只有几十个、**出现次数是几万次** ⇒ 10 万个块从几十秒降到接近线性。
`BPETokenizer` 内部自带这个缓存（`self._chunk_cache`）。

---

## 7. 落盘格式

`save_bpe()` 产出两个文件（都在 `data/` 下，**已入库**）：

| 文件 | 格式 | 例子 |
|---|---|---|
| `data/vocab.json` | `{"<id>": "<token 的字符串形式>"}`，`indent=1` | `{"258": "The", ...}` |
| `data/merges.txt` | 首行 `#version: 0.2`，之后每行 `"a b"` | `Ġstudent Ġtrained` |

`load_bpe()` 是它的逆：字符串 → id（反查 `vocab.json`），
再把 `merges.txt` 里的字符串反查成 `(a_id, b_id)`。

**因为用了 byte→unicode 映射，字符串里不会出现空格 / 换行，按空格切分是安全的。**
`test_bpe.py` 第 6 组把这条当成硬约束：

```python
parts = ln.split(" ")
assert len(parts) == 2, f"merges.txt 某行不是两个 token: {ln!r}"
for p in parts:
    assert p and not any(c.isspace() for c in p), ...
```

**语料同源**：`data/corpus.py` 是合成语料的**唯一真源**。
预训练语料（`data/pretrain_data.py`）与 BPE 训练语料（`data/build_vocab.py`）用同一批句子，
两边同 seed ⇒ 预训练语料是 BPE 语料的**前缀** ⇒ **词表完全覆盖预训练语料**。
重建词表后 `vocab.json` / `merges.txt` **逐字节一致**（本 session 实测：`git status` 干净）。

---

## 8. 实测数字（本仓库代码实跑，CPU，2026-10-23）

```bash
$ python data/build_vocab.py
[build_vocab] 语料 20000 句 / 785065 字节
[build_vocab] 词表 778（合并 521 次，目标 1000）
[build_vocab] 压缩比：19641 字节 -> 2767 tokens (7.10 字节/token)
[bpe] saved vocab(778) -> data\vocab.json
[bpe] saved merges(521) -> data\merges.txt

$ python tests/test_bpe.py
[压缩] 语料内句子 39B -> 15 tokens (2.60 字节/token)
[bpe] loaded vocab(778) merges(521)
[真实词表] vocab=778 merges=521 'The student trained ' 39B -> 6 tokens
test_bpe OK  vocab=293 merges=36 'The student trained ' 39B -> 15 tokens
```

| 指标 | 数值 | 出处 |
|---|---|---|
| 训练语料 | 20000 句 / 785065 字节 | `build_vocab.py` |
| 目标词表 / 实际词表 | 1000 / **778** | `config.DENSE_CONFIG.bpe_vocab_size` |
| 合并次数 | **521** | `train_bpe` 返回的 `len(merges)` |
| 压缩比（语料前 500 句） | 19641 字节 → 2767 tokens = **7.10 字节/token** | `build_vocab.py` |
| 单句压缩（真实词表） | `"The student trained a tiny transformer."` 39 B → **6 tokens** | `test_bpe.py` 第 8 组 |
| 单句压缩（小语料，vocab=293） | 同一句 39 B → **15 tokens**（2.60 字节/token） | `test_bpe.py` 第 3 组 |
| 中文句 | 42 字节 → **36 tokens**，0 个 UNK | 实测 |

**真实词表下的 token 切分**（`Ġ` = 前置空格）：

```text
(258, 'The') (424, 'Ġstudent') (491, 'Ġtrained') (272, 'Ġa') (347, 'Ġtiny') (504, 'Ġtransformer.')
```

**小语料 vs 大语料的差别（6 token vs 15 token）** 说明一件重要的事：
**压缩比是语料规模 + 词表覆盖度的函数，不是 BPE 这个算法的常数。**
本线的 7.10 字节/token 是在「合成语料 + 778 词表」上测的，
换一份语料这个数字就会变 —— 引用它时必须连语料一起说。

---

## 9. 陷阱清单

| 陷阱 | 表现 | 防线 |
|---|---|---|
| **合并 token 从 256 开始** | 字节 `0xFF` 被静默吞掉，**不报错** | `MERGE_OFFSET = 257` + `test_bpe.py` 第 1 组 |
| **初始 id 写成原始字节值** | 整体错位 1，合并规则全错、几乎不压缩 | 注释里写死「字节值 + 1」 |
| **merges.txt 写原始字节** | 空格/换行把行切错，`split(" ")` 炸 | byte→unicode 双射 + 第 6 组断言 |
| **压缩断言用语料外文本** | 测出来的压缩比虚高 / 断言假失败 | 断言分「语料内严格压缩」与「语料外不变长」两条 |
| **目标词表没训满就当 bug** | 本线 778 < 1000 是**语料耗尽**，不是实现问题 | `train_bpe` 的 `freq < 2` 提前退出 + 词表大小自洽断言 |
| **训练不保证确定性** | 两次训练结果不同 ⇒ checkpoint 不可复现 | 并列取字典序最小 + 第 4 组断言 |
| **编码不查 merges 顺序** | 优先级错乱，压缩率下降 | `encode` 按 `enumerate(merges)` 顺序应用 |

---

## 10. 复现命令

```bash
cd qwen3-learning/qwen3-pytorch

python data/build_vocab.py      # 训练并落盘词表（vocab.json / merges.txt），秒级
python tests/test_bpe.py        # 8 组断言：id 分配 / 往返 / 压缩 / 确定性 / 耗尽 / 落盘 / 接口 / 真实词表
```

**什么时候需要重建词表？**
`data/corpus.py`（语料真源）或 `config.DENSE_CONFIG.bpe_vocab_size` 改了才需要。
重建是**幂等**的：同 seed ⇒ 输出逐字节一致。

---

## 11. 自测题（先自己答，再看答案）

**Q1**：为什么合并 token 的 id 必须从 257 开始，而不是 256？
> 字节 token 占 `id 1..256`，其中 `256` 是字节 `0xFF` 的位置。从 256 开始合并会**静默吞掉** `0xFF`，
> 而且**不报任何错** —— 词表少一个 id，`0xFF` 无法编码。

**Q2**：`merges.txt` 里为什么不能直接写 token 的原始字节？
> 因为空格（`0x20`）和换行（`0x0A`）本身就是字节 token。
> 写进去之后一行会变成 `" a b"`，用空格分隔根本切不出两个 token。
> 必须先做 **byte→unicode 双射**，映射后的字符保证不是空白。

**Q3**：BPE 的「训练」和「编码」有什么本质区别？
> 训练**依赖语料**（要统计全局频次），产出 merges；
> 编码**不依赖语料**，只按 merges 的先后顺序把相邻对合并掉，是**纯函数**。

**Q4**：为什么「压缩比 7.10 字节/token」这个数字不能直接拿去跟别的模型比？
> 因为压缩比是**语料规模 + 词表覆盖度**的函数，不是 BPE 算法的常数。
> 本线的 7.10 是在「20000 句合成语料 + 778 词表」上测的；
> 同一份实现换成真实语料、词表训到 5 万，数字会完全不同。

**Q5**：`test_bpe.py` 第 3 组为什么要把压缩断言拆成「语料内严格压缩」和「语料外不变长」两条？
> 因为预分词让合并规则**不跨块**，训练学到的规则只在语料里出现过的块上有效。
> 用语料外的字符串测压缩是测不出来的 —— 用一条错样本去测，等于把「测不准」误判成「算法不行」。

---

## 12. 相关文件速查

| 想做什么 | 看哪个 |
|----------|--------|
| 理解 BPE 原理（主线视角） | [`gpt-learning/1.bpe-tokenizer.md`](../gpt-learning/1.bpe-tokenizer.md) |
| 看本线实现 | [`qwen3-learning/qwen3-pytorch/data/bpe.py`](qwen3-pytorch/data/bpe.py) |
| 看对外接口 | [`qwen3-learning/qwen3-pytorch/data/tokenizer.py`](qwen3-pytorch/data/tokenizer.py) |
| 看训练与落盘 | [`qwen3-learning/qwen3-pytorch/data/build_vocab.py`](qwen3-pytorch/data/build_vocab.py) |
| 看断言 | [`qwen3-learning/qwen3-pytorch/tests/test_bpe.py`](qwen3-pytorch/tests/test_bpe.py) |
| 看数据怎么用 token | [`qwen3-learning/qwen3-pytorch/data/pretrain_data.py`](qwen3-pytorch/data/pretrain_data.py) |

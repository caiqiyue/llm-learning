"""
debug_tokenizer.py — 单步调试 CharTokenizer

跑法（在 gpt-learning-pytorch/ 目录下）：
    python data/debug_tokenizer.py

目的：
  - 可视化 char ↔ id 的对应关系（30 个 token 一行）
  - 检查 5 个 special token id 锁死正确
  - 检查 encode/decode 是否可逆
  - 检查未知字符如何处理

每个 step 之间可加断点，逐步观察数据。
"""

import os
import sys

# 把 gpt-learning-pytorch/ 加进 sys.path，确保 `from data.xxx` 可 import
_HERE = os.path.dirname(os.path.abspath(__file__))
_PJT = os.path.dirname(_HERE)
sys.path.insert(0, _PJT)

from data.tokenizer import CharTokenizer, UNK_ID, PAD_ID, BOS_ID, EOS_ID, SEP_ID  # noqa: E402


def hr(title: str, char: str = "="):
    """打印一个分隔标题。"""
    line = char * 64
    print(f"\n{line}\n {title}\n{line}")


def show_tokens(tk: CharTokenizer, ids, max_show=30):
    """把 id 列表可视化：[id]char 形式。"""
    out = []
    for i, tid in enumerate(ids[:max_show]):
        c = tk.itos[tid] if 0 <= tid < tk.vocab_size else "?"
        c = c if c != "\n" else "\\n"
        out.append(f"[{tid:3}]{c!r}")
    if len(ids) > max_show:
        out.append(f"... (+{len(ids)-max_show} more)")
    print("    " + " ".join(out))


def main():
    vocab_path = os.path.join(_HERE, "vocab.txt")

    hr("Step 1: 检查 vocab.txt 是否存在")
    print(f"路径: {vocab_path}")
    print(f"存在? {os.path.exists(vocab_path)}")
    if not os.path.exists(vocab_path):
        print("\n[!] vocab.txt 不存在，请先跑: python data/build_vocab.py")
        return

    hr("Step 2: 加载 CharTokenizer（关键形状）")
    tk = CharTokenizer(vocab_path)
    # 【形状速查】
    #   stoi: dict[str, int]  长度 = vocab_size
    #   itos: list[str]       长度 = vocab_size
    print(f"vocab_size: {tk.vocab_size}")
    print(f"前 10 个 token (按 id 顺序): {tk.itos[:10]}")
    print(f"\n★ 5 个 special token 必须锁死（与 build_vocab.py 写入顺序一致）:")
    for name, tid in [("UNK_ID", UNK_ID), ("PAD_ID", PAD_ID),
                      ("BOS_ID", BOS_ID), ("EOS_ID", EOS_ID), ("SEP_ID", SEP_ID)]:
        tok = tk.itos[tid]
        mark = "[OK]" if tok in ("<unk>", "<pad>", "<bos>", "<eos>", "<sep>") else "[!]"
        print(f"  {mark} {name:8} = {tid}  → '{tok}'")

    # ===== ★★★ 重要发现（必读）★★★
    print()
    print("#" * 64)
    print("# ★★★ 重要：特殊 token 整体一个 id（feat-027 之后）★★★")
    print("#" * 64)
    print("# CharTokenizer.encode 现在是「最长匹配优先」：")
    print("#     先匹配词表里多字符的条目（<unk> <pad> <bos> <eos> <sep>），")
    print("#     匹配不上再逐字符查表（表外字符 → UNK_ID）")
    print("# 所以 '<bos>' 被编码成 [2]（BOS_ID），而不是 5 个字符")
    print("#")
    print("# ★ 为什么必须这样：")
    print("#    之前 encode 逐字符查表 ⇒ '<sep>' 变成 [0, s_id, e_id, p_id, 0]，")
    print("#    于是 BOS_ID / SEP_ID / EOS_ID 三个 id **永远不会被产生**：")
    print("#      · 模型从没学过吐 EOS ⇒ 生成永远跑满 max_new_tokens，不会自己停；")
    print("#      · decode 回来是 '<unk>sep<unk>' 而不是 '<sep>'。")
    print("#    配套的坑：UNK(id=0) 解码回来是整条 5 字符的 '<unk>'，")
    print("#    所以「回答」不能按**字符串长度**切，必须按 **token id 个数**切。")
    print("#" * 64)
    print()
    # 现场验证
    test_str = "<bos>"
    ids = tk.encode(test_str)
    print(f"\n【现场验证】encode('<bos>') = {ids}")
    print(f"  期望: [{BOS_ID}]  （特殊 token 整体一个 id）")
    print(f"  实际: {ids}")
    assert ids == [BOS_ID], "特殊 token 必须整体一个 id"

    hr("Step 3: 几条典型 SFT 字符串测试 encode/decode")
    test_strs = [
        "<bos> What is a cat? <sep> A cat is a small creature. <eos>",
        "The cat went to the forest.",
        "<unk>",
        "",  # 空字符串
    ]
    for s in test_strs:
        ids = tk.encode(s)
        back = tk.decode(ids)
        reversible = (s == back)
        # ★ 注意：reversible=False 可能正常（未知字符会变成 UNK）
        print(f"\n原: {s!r}")
        print(f"ids: {ids}")
        print(f"decoded: {back!r}")
        print(f"可逆? {reversible}  {'(正常：未知字符 → UNK)' if not reversible and 'unk' not in s else ''}")

    hr("Step 4: 字符 ↔ id 逐字符映射（取前 30 字符）")
    s = "The cat is happy.\n"
    print(f"原文: {s!r}  (len={len(s)})")
    for i, c in enumerate(list(s)):
        cid = tk.stoi.get(c, UNK_ID)
        marker = "  " if cid != UNK_ID else "[?]"  # ? = 未知
        print(f"  [{i:2}] char={c!r:8}  id={cid:4} {marker}")

    hr("Step 5: ★ vocab 里到底有哪些字符？（逐字符验证）")
    has_chars = sorted(set(tk.itos[5:]))  # 5 是 special 后第一个普通字符
    print(f"vocab 普通字符数: {len(has_chars)}")
    print(f"全部字符: {''.join(has_chars)!r}")
    print()
    print("【检查 pretrain_data 期望的特殊字符是否在 vocab】")
    expected_in_pretrain = [" ", "\n", "\t", ".", ",", "?", "!"]
    for c in expected_in_pretrain:
        tid = tk.stoi.get(c, None)
        mark = "[OK]" if tid is not None else "[不在 vocab]"
        disp = repr(c).replace("'", "")
        print(f"  {disp:<10}  {mark}  (id={tid if tid is not None else 'N/A'})")
    print()
    print("★★ 重要发现：")
    print("   vocab.txt 里有 ' ' (空格) 但没有 '\\n' (换行符)")
    print("   pretrain_data.format_pretrain_for_lm 用 '\\n'.join(...) 拼接样本")
    print("   这些 '\\n' 全部 encode 成 UNK_ID=0！")
    print("   这是数据管道的隐性 bug（不致命，因为 128 长的 block 里 \\n 占比小）")
    print("   - 如果你想修：在 build_vocab.py 加 '\\n' 到 SPECIAL_TOKENS 或 char 列表里")
    print("   - 如果你想跳过：debug 完记住这一点即可")

    hr("Step 6: 逐字符可视化（高效查看 tokenization 是否正确）")
    test = "Once upon a time, the cat went to the forest."
    ids = tk.encode(test)
    print(f"原文: {test!r}")
    print(f"ids:  {ids}")
    show_tokens(tk, ids, max_show=len(ids))

    hr("Step 7: 未知字符处理（验证容错）")
    for s in ["Hello, 世界! 🌍", "café résumé 100% pure"]:
        ids = tk.encode(s)
        unk_count = ids.count(UNK_ID)
        decoded = tk.decode(ids)
        print(f"\n原:     {s!r}")
        print(f"ids:    {ids}")
        print(f"decoded:{decoded!r}")
        print(f"UNK 次数: {unk_count}  (字节被 map 到 <unk>={UNK_ID})")

    hr("Step 8: 不同长度字符串 → token 数")
    print(f"{'原字符串':<60}  {'char':>4}  {'tok':>4}  ratio")
    test_strs = [
        "The cat.",
        "<bos> What is a cat? <sep> A cat is small. <eos>",
        "Once upon a time, the cat went to the forest. The cat saw a rock.",
    ]
    for s in test_strs:
        ids = tk.encode(s)
        ratio = len(s) / max(1, len(ids))
        print(f"{s[:58]!s:<60}  {len(s):4}  {len(ids):4}  {ratio:.2f}")

    hr("Step 9: 查看 vocab 末尾（普通字符 + Unicode 范围）")
    print(f"vocab[10:30]: {tk.itos[10:30]}")  # 第 11 个开始是普通字符
    print(f"vocab[-10:]: {tk.itos[-10:]}")
    chars_ord_min = min(ord(c) for c in tk.itos if len(c) == 1)
    chars_ord_max = max(ord(c) for c in tk.itos if len(c) == 1)
    print(f"普通字符的 Unicode 范围: {chars_ord_min}-{chars_ord_max}")

    print("\n" + "=" * 64)
    print(" [OK] tokenizer debug 完成")
    print("=" * 64)


if __name__ == "__main__":
    main()

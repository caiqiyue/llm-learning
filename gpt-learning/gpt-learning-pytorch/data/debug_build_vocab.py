"""
debug_build_vocab.py — 单步调试 vocab 构建流程

跑法（在 gpt-learning-pytorch/ 目录下）：
    python data/debug_build_vocab.py

目的：
  - 看 build_vocab 的内部流程：原始数据 → 字符集合 → vocab.txt
  - 验证 5 个 special token 在最前面
  - 验证字符按 Unicode 排序
  - 验证 vocab.txt 格式符合 tokenizer 预期

注意：本脚本**不会**重复写 vocab.txt，只会读取已有文件 + 模拟 build 流程展示。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PJT = os.path.dirname(_HERE)
sys.path.insert(0, _PJT)

from data.build_vocab import SPECIAL_TOKENS, build_vocab  # noqa: E402


def hr(title: str):
    print(f"\n{'=' * 64}\n {title}\n{'=' * 64}")


def main():
    vocab_path = os.path.join(_HERE, "vocab.txt")

    hr("Step 1: 模拟 build_vocab 的数据收集流程")
    print("调用 generate_*_dataset 各取少量样本，看原始数据长啥样：")
    print()

    from data.pretrain_data import generate_pretrain_dataset
    from data.sft_data import generate_sft_dataset
    from data.preference_data import generate_preference_dataset

    pretrain = generate_pretrain_dataset(n_samples=5, seed=42)
    """
    [{'text': 'Once upon a time, the cat went to the cave. The bear saw a mushroom. The fox said: "goodbye".', 'id': 0}, {'text': 'One day, the dog decided to listen at the village. What a small adventure!', 'id': 1}, {'text': 'Once upon a time, the dog went to the mountain. The bear saw a shell. The frog said: "hello".', 'id': 2}, {'text': 'One day, the bear decided to watch at the village. What a quiet adventure!', 'id': 3}, {'text': 'The loud rabbit found a tree near the river. It was a cool day.', 'id': 4}]
    """
    print("【pretrain 样本】(n=5)")
    for s in pretrain:
        print(f"  [{s['id']}] {s['text'][:80]}{'...' if len(s['text'])>80 else ''}")

    sft = generate_sft_dataset(n_samples=3, seed=42)
    """
    [{'prompt': 'Tell me about a cat.', 'response': 'The rabbit is quiet and lives in the mountain.', 'prompt_template': 'Tell me about a {animal}.', 'response_template': 'The {animal} is {adjective} and lives in the {place}.', 'id': 0}, {'prompt': 'Where does a dog live?', 'response': 'The owl is big and lives in the hill.', 'prompt_template': 'Where does a {animal} live?', 'response_template': 'The {animal} is {adjective} and lives in the {place}.', 'id': 1}, {'prompt': 'Tell me a story about a cat.', 'response': 'The big bear is a windy creature that likes to watch.', 'prompt_template': 'Tell me a story about a {animal}.', 'response_template': 'The {adjective} {animal} is a {adjective2} creature that likes to {action}.', 'id': 2}]
    """
    print("\n【sft 样本】(n=3)")
    for s in sft:
        print(f"  [{s['id']}]")
        print(f"    prompt:   {s['prompt']}")
        print(f"    response: {s['response']}")

    pref = generate_preference_dataset(n_samples=3, seed=42)
    """
    [{'prompt': 'Tell me about a cat.', 'chosen': 'The rabbit is quiet and lives in the mountain.', 'rejected': 'the quiet is mountain. lives in rabbit and The', 'rejected_strategy': 'noisy_keywords', 'id': 0}, {'prompt': 'Where does a dog live?', 'chosen': 'The owl is big and lives in the hill.', 'rejected': 'The rabbit is quiet and lives in the mountain.', 'rejected_strategy': 'unrelated', 'id': 1}, {'prompt': 'Tell me a story about a cat.', 'chosen': 'The big bear is a windy creature that likes to watch.', 'rejected': 'The big rabbit is a windy creature that likes to watch.', 'rejected_strategy': 'wrong_entity', 'id': 2}]
    """
    print("\n【preference 样本】(n=3)")
    for s in pref:
        print(f"  [{s['id']}] strategy={s['rejected_strategy']}")
        print(f"    prompt:   {s['prompt']}")
        print(f"    chosen:   {s['chosen']}")
        print(f"    rejected: {s['rejected'][:80]}{'...' if len(s['rejected'])>80 else ''}")

    hr("Step 2: 收集所有出现过的字符 + 按 Unicode 排序")
    print("(下面我们用与 build_vocab 完全相同的方式收集字符集)")
    texts = []
    for s in generate_pretrain_dataset(n_samples=2000, seed=42):
        texts.append(s["text"])
    for s in generate_sft_dataset(n_samples=1000, seed=42):
        texts.append(s["prompt"])
        texts.append(s["response"])
    for s in generate_preference_dataset(n_samples=1000, seed=42):
        texts.append(s["prompt"])
        texts.append(s["chosen"])
        texts.append(s["rejected"])

    chars = sorted(set("".join(texts)))
    print(f"去重后字符总数: {len(chars)}")
    print(f"前 30 个字符: {''.join(chars[:30])!r}")
    print(f"后 10 个字符: {''.join(chars[-10:])!r}")

    # Unicode 范围
    ords = [ord(c) for c in chars]
    print(f"Unicode 范围: {min(ords)} ({chr(min(ords))!r}) ... {max(ords)} ({chr(max(ords))!r})")

    # 检查特殊字符（先算出来，避开 f-string 里用反斜杠）
    has_n = ("\n" in ''.join(chars))
    has_space = (" " in ''.join(chars))
    print(f"包含换行符 \\n ? {has_n}")
    print(f"包含空格 ' ' ? {has_space}")

    hr("Step 3: vocab.txt 文件结构")
    print(f"路径: {vocab_path}")
    print(f"存在? {os.path.exists(vocab_path)}")
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f if line.strip() and not line.startswith("#")]

        print(f"\n文件总行数（去除空行/注释）: {len(lines)}")
        print(f"\n【前 5 行必须按这个顺序（special tokens）】")
        for i, line in enumerate(lines[:5]):
            expect = SPECIAL_TOKENS[i] if i < len(SPECIAL_TOKENS) else "?"
            ok = "[OK]" if line == expect else "[!]"
            print(f"  [{i}] {ok} '{line}'  (期望: '{expect}')")

        print(f"\n【前 15 行】（5 special + 10 普通字符）")
        for i, line in enumerate(lines[:15]):
            print(f"  [{i:2}] id={i:<3} '{line}'")

        print(f"\n【后 10 行】")
        for i, line in enumerate(lines[-10:]):
            real_id = len(lines) - 10 + i
            print(f"  [{i:2}] id={real_id:<3} '{line}'")

    hr("Step 4: 重新跑一次 build_vocab() 看输出")
    print("（这会覆盖 vocab.txt—— 如果你有自定义 vocab 请先备份）")
    try:
        ans = input("确认重跑？(y/N): ").strip().lower()
    except EOFError:
        print("（无 stdin，自动跳过；想重跑请直接 python data/build_vocab.py）")
        ans = "n"
    if ans == "y":
        total = build_vocab(save_path=vocab_path)
        print(f"\n写入 token 总数: {total}")
    else:
        print("（已跳过）")

    print("\n" + "=" * 64)
    print(" [OK] build_vocab debug 完成")
    print("=" * 64)


if __name__ == "__main__":
    main()

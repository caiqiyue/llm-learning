"""
debug_benchmark_data.py — 单步调试 Benchmark 题库

跑法（在 gpt-learning-pytorch/ 目录下）：
    python data/debug_benchmark_data.py

目的：
  - 看 4 种问题模板生成逻辑
  - 看 choices 的结构 + answer_idx 的位置
  - 看 answer_idx 在 [0, 3] 上是否真均匀
  - 看 benchmark.json 的保存 / 读取
"""

import os
import sys
import json
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_PJT = os.path.dirname(_HERE)
sys.path.insert(0, _PJT)

from data.benchmark_data import (  # noqa: E402
    QUESTION_BUILDERS,
    build_benchmark_questions,
    save_benchmark,
    load_benchmark,
    _build_place_question,
    _build_action_question,
    _build_adjective_question,
    _build_animal_in_place_question,
)


def hr(title: str):
    print(f"\n{'=' * 64}\n {title}\n{'=' * 64}")


def show_question(q, idx=None):
    """打印一道题。"""
    label = f"[{idx}]" if idx is not None else ""
    print(f"{label} Q: {q['question']}")
    for j, c in enumerate(q["choices"]):
        marker = " ◀ ANSWER" if j == q["answer_idx"] else ""
        print(f"      ({j}) {c}{marker}")
    print()


def main():
    benchmark_path = os.path.join(_HERE, "benchmark.json")

    hr("Step 1: 4 种问题模板概览")
    print("build_benchmark_questions 按 i % 4 轮询调用下面 4 个 builder：\n")
    for i, b in enumerate(QUESTION_BUILDERS):
        desc = {
            "place":           "Where does a {animal} live?   → pick correct place",
            "action":          "What does a {animal} like?    → pick correct action",
            "adjective":       "What is a {adj} {animal}?     → pick the adjective",
            "animal_in_place": "Who is in the {place}?        → pick correct animal",
        }
        print(f"  [{i}] {b:<22}  ← {desc[b]}")

    hr("Step 2: 单题示例 —— 每种模板各 1 题")
    import random
    rng = random.Random(7)
    builders = [
        _build_place_question,
        _build_action_question,
        _build_adjective_question,
        _build_animal_in_place_question,
    ]

    print("下面用相同的 rng 顺序构造 4 题:\n")
    for i, b in enumerate(builders):
        q = b(rng)
        show_question(q, idx=i)

    hr("Step 3: answer_idx 的均匀性检验")
    n = 1000
    items = build_benchmark_questions(n=n, seed=1234)
    counter = Counter(q["answer_idx"] for q in items)
    print(f"样本数: {n}\n")
    print(f"{'answer_idx':<12}{'出现次数':<12}{'占比':<10}{'均匀期望 25%?'}")
    print("-" * 64)
    for idx in range(4):
        cnt = counter.get(idx, 0)
        pct = cnt / n * 100
        diff = pct - 25
        print(f"{idx:<12}{cnt:<12}{pct:.1f}%      {diff:+.1f}%")

    hr("Step 4: 题目模板各占 1/4 验证")
    # build_benchmark_questions 按 i % 4 轮询，所以 n=100 时每种 25 题
    print(f"build_benchmark_questions 默认 n=100，4 种 builder 每种 25 题")
    # 我们识别每题属于哪种 builder
    items = build_benchmark_questions(n=8, seed=42)
    print(f"\n前 8 题:")
    for i, q in enumerate(items):
        qtype = (
            "place"          if q["question"].startswith("Where does") else
            "action"         if q["question"].startswith("What does") else
            "adjective"      if q["question"].startswith("What is a") else
            "animal_in_place"
        )
        print(f"  [{i}] {qtype:<18}  Q: {q['question']}")

    hr("Step 5: 干扰项的构造细节")
    print("★ 每题有一个正确答案 + 3 个干扰")
    print("★ 不同模板的干扰类型不同（关键）:\n")
    print("  - place (动物栖息地)    干扰: 其他 place + 1 个 'likes to {action}'（语义不一致）")
    print("  - action (动物行为)      干扰: 其他 action + 1 个 'in the {place}'")
    print("  - adjective (形容词)     干扰: 其他 3 个形容词（同类型）")
    print("  - animal_in_place        干扰: 其他 3 个动物（同类型）")

    print("\n例子（从 100 题里挑）:")
    items100 = build_benchmark_questions(n=100, seed=1234)
    # 把每个 builder 的第一题拿出来
    rng2 = random.Random(99)
    seen_types = set()
    for q in items100:
        if q["question"].startswith("Where does"):
            t = "place"
        elif q["question"].startswith("What does"):
            t = "action"
        elif q["question"].startswith("What is a"):
            t = "adjective"
        else:
            t = "animal_in_place"
        if t not in seen_types:
            seen_types.add(t)
            show_question(q, idx=t)

    hr("Step 6: 保存 / 读取 JSON")
    print(f"benchmark.json 路径: {benchmark_path}")
    print(f"存在? {os.path.exists(benchmark_path)}")

    if os.path.exists(benchmark_path):
        loaded = load_benchmark(benchmark_path)
        print(f"\n从 JSON 读取到 {len(loaded)} 题")
        print(f"\nJSON 第 1 题前 200 字符:")
        first = loaded[0]
        s = json.dumps(first, ensure_ascii=False, indent=2)
        print(s[:300])

    print("\n★ evaluate.py 会用 load_benchmark 读这个文件，然后:")
    print("   - prompt = 'Q: <question>\\nA:'")
    print("   - 让模型对 4 个 choice 各算一次续写 loss")
    print("   - loss 最低的 = 模型认为最可能的答案")

    hr("Step 7: 重新生成保存 (可选)")
    try:
        ans = input(f"重新生成 {benchmark_path}? (y/N): ").strip().lower()
    except EOFError:
        print("（无 stdin，自动跳过；想重跑请直接 python data/benchmark_data.py）")
        ans = "n"
    if ans == "y":
        items = build_benchmark_questions(n=100, seed=1234)
        save_benchmark(items, benchmark_path)
        print(f"已保存 {len(items)} 题到 {benchmark_path}")
    else:
        print("（已跳过）")

    print("\n" + "=" * 64)
    print(" [OK] benchmark_data debug 完成")
    print("=" * 64)


if __name__ == "__main__":
    main()

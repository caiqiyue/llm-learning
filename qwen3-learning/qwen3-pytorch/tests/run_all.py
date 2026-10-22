"""
Qwen3 线全部 smoke 测试的汇总运行器。

用法：python tests/run_all.py

设计约束（见开发计划 §0.2）：
  - 不依赖 pytest（项目依赖上限只有 torch / numpy / nltk）
  - 每个 test_*.py 顶层线性执行 + assert，exit 0 表示通过
  - 本脚本用 subprocess 逐个跑，隔离崩溃与污染，打印 PASS/FAIL 汇总
"""

import os
import subprocess
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)


def discover() -> list:
    """找出全部 tests/test_*.py（按文件名排序，保证执行顺序稳定）。"""
    names = [n for n in os.listdir(TESTS_DIR) if n.startswith("test_") and n.endswith(".py")]
    return sorted(names)


def main() -> int:
    """逐个跑测试并汇总。返回进程退出码。"""
    results = []

    for name in discover():
        proc = subprocess.run(
            [sys.executable, os.path.join(TESTS_DIR, name)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        ok = proc.returncode == 0
        results.append((name, ok, proc))
        status = "PASS" if ok else "FAIL"
        detail = (proc.stdout.strip().splitlines() or [""])[-1]
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            tail = (proc.stderr.strip().splitlines() or [""])[-5:]
            for line in tail:
                print(f"         {line}")

    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_pass
    print(f"\n[run_all] PASS={n_pass}  FAIL={n_fail}  TOTAL={len(results)}")

    if n_fail > 0:
        return 1
    print("[run_all] all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())

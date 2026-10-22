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


def _safe(s: str) -> str:
    """
    把字符串里的「当前控制台编码无法表示」的字符替换掉，避免 print 时抛
    UnicodeEncodeError。

    Windows 默认控制台是 GBK，像 '²'、'Ġ' 这类字符不在 GBK 里，
    直接 print 会崩（之前 test_swiglu 就这么炸过）。
    中文能表示，所以照常输出。
    """
    enc = sys.stdout.encoding or "utf-8"
    return s.encode(enc, errors="replace").decode(enc, errors="replace")


def discover() -> list:
    """找出全部 tests/test_*.py（按文件名排序，保证执行顺序稳定）。"""
    names = [n for n in os.listdir(TESTS_DIR) if n.startswith("test_") and n.endswith(".py")]
    return sorted(names)


def main() -> int:
    """逐个跑测试并汇总。返回进程退出码。"""
    results = []

    # 统一用 UTF-8 输出：Windows 默认 stdout 是 GBK，测试里打印的中文 /
    # 'Ġ' 这类字符在 GBK 下要么被替换要么直接抛 UnicodeEncodeError。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:      # pragma: no cover - 极老的 Python 没有 reconfigure
        pass

    # 子进程以文件路径方式启动，sys.path[0] 是 tests/ —— 额外用 PYTHONPATH
    # 把项目根也塞进去（测试文件自身也会做一次 sys.path 引导，双保险）
    env = dict(os.environ)
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    # 强制子进程用 UTF-8 输出：测试里会打印 'Ġ'、'²' 这类 GBK 编不了的字符，
    # 若让子进程用系统默认编码，它自己 print 时就会崩。父进程这边按 UTF-8 解码。
    env["PYTHONIOENCODING"] = "utf-8"

    for name in discover():
        proc = subprocess.run(
            [sys.executable, os.path.join(TESTS_DIR, name)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            env=env,
        )
        ok = proc.returncode == 0
        results.append((name, ok, proc))
        status = "PASS" if ok else "FAIL"
        out = proc.stdout.decode("utf-8", errors="replace")
        err = proc.stderr.decode("utf-8", errors="replace")
        detail = (out.strip().splitlines() or [""])[-1]
        print(f"  [{status}] {name}" + (f" — {_safe(detail)}" if detail else ""))
        if not ok:
            tail = (err.strip().splitlines() or [""])[-5:]
            for line in tail:
                print(f"         {_safe(line)}")

    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_pass
    print(f"\n[run_all] PASS={n_pass}  FAIL={n_fail}  TOTAL={len(results)}")

    if n_fail > 0:
        return 1
    print("[run_all] all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())

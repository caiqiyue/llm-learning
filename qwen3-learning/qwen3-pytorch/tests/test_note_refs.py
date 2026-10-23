"""
笔记 ↔ 源码交叉引用的可运行校验（对应 feat-008）。

要守的东西：
  1. 每篇 Qwen3 笔记开头都有「对应 GPT-2 实现位置 / 对应 Qwen3 源码位置」两行
  2. 交叉引用块里的**每个 markdown 链接**都能点开、且目标文件真实存在
  3. 笔记正文里出现的**路径型反引号串**都能在仓库里找到对应文件
  4. 根 README 的 Qwen3 索引表覆盖了全部笔记，且每一行都带「Qwen3 源码」列
  5. 唯一缺过的那篇笔记（Qwen3_BPE_学习笔记.md）已补齐
  6. 全仓 markdown 里的 `#L<行号>` 源码锚点：**行号在目标文件的范围内**（R1），
     且**单行锚点必须精确指向笔记引用的那一句**（R2）（feat-030）

运行：python tests/test_note_refs.py
"""

import io
import os
import re
import sys

# tests/ → qwen3-pytorch → qwen3-learning → llm-learning（仓库根）
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)          # qwen3-learning/qwen3-pytorch
NOTES_DIR = os.path.dirname(PROJECT_ROOT)          # qwen3-learning
REPO_ROOT = os.path.dirname(NOTES_DIR)             # llm-learning

# 路径查找的候选根（按顺序找，第一个命中的算「解析成功」）
_SEARCH_ROOTS = [
    REPO_ROOT,
    PROJECT_ROOT,
    os.path.join(REPO_ROOT, "gpt-learning", "gpt-learning-pytorch"),
    NOTES_DIR,
]

# 反引号里「像路径」的串：必须含 / + 有文件扩展名 + 只含路径安全字符
_PATHLIKE_RE = re.compile(
    r"^(?:[A-Za-z0-9_.一-鿿-]+/)+[A-Za-z0-9_.一-鿿-]+\.(?:py|md|json|txt|pt|sh)$"
)

# markdown 链接：[文字](目标)
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

# 交叉引用块的两行标记
_GPT2_LINE = "> **对应 GPT-2 实现位置**"
_QWEN3_LINE = "> **对应 Qwen3 源码位置**"

# markdown 链接带行号锚点：[文字](路径#L12) 或 [文字](路径#L12-L34)
_ANCHOR_RE = re.compile(r"\[([^\]]*)\]\(([^)\s]+)#L(\d+)(?:-L(\d+))?\)")

# 扫全仓 markdown 时要跳过的目录
_SKIP_DIRS = {".git", "checkpoints", "docs", "__pycache__", ".claude", ".vscode"}


def _note_files() -> list:
    """找出全部 Qwen3 学习笔记（不含设计文档与开发计划）。"""
    names = [
        n for n in os.listdir(NOTES_DIR)
        if n.startswith("Qwen3_") and n.endswith(".md")
    ]
    return sorted(names)


def _read(path: str) -> str:
    """读文件（显式 UTF-8）。"""
    with io.open(path, encoding="utf-8") as f:
        return f.read()


def _resolve(rel: str) -> bool:
    """
    判断一个相对路径能否在候选根下找到真实文件。

    参数：
        rel: 相对路径（可能带 `::符号名` 后缀）
    返回：
        True 表示命中
    """
    rel = rel.split("::", 1)[0]          # 去掉 `::function` 这种定位后缀
    rel = rel.lstrip("./")
    for root in _SEARCH_ROOTS:
        if os.path.isfile(os.path.join(root, rel)):
            return True
    return False


def _is_skippable(s: str) -> bool:
    """
    判断一个反引号串是否**不该**被当成仓库路径。

    跳过的三类：
      - 绝对路径（带盘符，如 `E:/tmp/xxx.py`）—— 在仓库外
      - 产物目录（`checkpoints/` / `docs/`）—— 按 AGENTS.md 不入库
      - 含空格 —— 那是命令或自然语言，不是路径
    """
    if " " in s or "\t" in s:
        return True
    if re.match(r"^[A-Za-z]:", s):
        return True
    if s.startswith(("checkpoints/", "docs/")):
        return True
    return False


def _all_markdown_files() -> list:
    """列出全仓所有 markdown 文件（跳过产物目录 / 元数据目录）。"""
    out = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            if name.endswith(".md"):
                out.append(os.path.join(dirpath, name))
    return sorted(out)


def _strip_comment(line: str) -> str:
    """去掉行尾注释并把内部空白压成单空格（比逐字比较稳）。"""
    h = line.find("#")
    body = line if h < 0 else line[:h]
    return " ".join(body.split())


def _first_code_line(lines: list, fence_idx: int):
    """围栏里第一条「非空且非纯注释」的语句（归一化后），没有则返回 None。"""
    for raw in lines[fence_idx:]:
        if raw.strip().startswith("```"):
            break
        s = _strip_comment(raw)
        if s:
            return s
    return None


def _check_anchors() -> int:
    """校验全仓 markdown 里的 `#L<行号>` 源码锚点，返回锚点总数。

    ★ 为什么要有这一条（feat-030）：
      「链接点得开」≠「链接指得准」—— `file.py#L262-L268` 里的**行号**
      会随代码改动漂移，而**没人点就没人发现** ⇒ 静默腐烂。
      本仓真实踩过：`GPT2训练数学公式与超参详解.md` 的 4 个锚点指向
      `train_ppo.py`（已被改造成 **14 行**的 thin wrapper），
      行号 262 / 271 / 303 / 392 / 397 **全部越界**；
      更严重的是被引用的那几段代码（`compute_kl` 等）在重构后**已经不存在**。

    ★ 两条规则：
      **R1（范围）**：目标文件存在，且 `1 <= start <= end <= 文件行数`。
      **R2（单行锚点必须精确）**：`file.py#L123` 这种**单行**锚点，
        它的意思是「就是这一行」⇒ 紧跟其后的 python 代码块**第一条语句**
        必须**就是**源码第 123 行（去注释、去空白后逐字相等）。
        —— 行号在范围内也可能是**指错行**（本仓实测有 3 处）。
      **多行锚点**（`file.py#L33-L57`）只做 R1：
        笔记里的代码**本来就是简化版**（会省 docstring、拆开签名），
        要求逐字落在区间内会误报（实测 10/18 对不上，全是风格差异不是错误）。
    """
    problems = []
    n_anchor = 0
    for path in _all_markdown_files():
        rel_note = os.path.relpath(path, REPO_ROOT).replace("\\", "/")
        text = _read(path)
        lines = text.split("\n")
        for m in _ANCHOR_RE.finditer(text):
            n_anchor += 1
            target = m.group(2)
            a = int(m.group(3))
            b = int(m.group(4) or m.group(3))
            if target.startswith(("http://", "https://", "#")):
                continue
            resolved = os.path.normpath(os.path.join(os.path.dirname(path), target))
            if not os.path.isfile(resolved):
                problems.append(
                    f"{rel_note}: 锚点目标文件不存在 → {target}#L{a}")
                continue
            src_lines = _read(resolved).split("\n")
            n_lines = len(src_lines)
            if a < 1 or b < a or b > n_lines:
                problems.append(
                    f"{rel_note}: 锚点行号越界 → {target}#L{a}"
                    + (f"-L{b}" if b != a else "")
                    + f"（目标文件只有 {n_lines} 行）")
                continue
            if a != b:
                continue                      # 多行锚点只做 R1
            # --- R2：单行锚点必须精确指向被引用的那一句 ---
            lineno = text[:m.start()].count("\n")     # 锚点所在行（0 基）
            fence = None
            j, seen = lineno + 1, 0
            while j < len(lines) and seen < 3:
                if lines[j].strip():
                    seen += 1
                if lines[j].strip() == "```python":
                    fence = j + 1
                    break
                j += 1
            if fence is None:
                continue                  # 后面没有代码块，无从比对
            first = _first_code_line(lines, fence)
            if first is None:
                continue
            src_stmt = _strip_comment(src_lines[a - 1]) if a - 1 < len(src_lines) else ""
            if first != src_stmt:
                problems.append(
                    f"{rel_note}: 单行锚点指错行 → {target}#L{a}\n"
                    f"      笔记里引用的：{first}\n"
                    f"      源码第 {a} 行：{src_stmt or '（空行 / 越界）'}")
    assert not problems, (
        "笔记里的 `#L` 行号锚点失效（feat-030）：\n    "
        + "\n    ".join(problems))
    return n_anchor


def main() -> int:
    """跑全部断言。返回进程退出码。"""
    notes = _note_files()

    # =================================================================
    # 0) 唯一缺过的那篇笔记必须存在
    # =================================================================
    assert "Qwen3_BPE_学习笔记.md" in notes, \
        "Qwen3_BPE_学习笔记.md 仍缺 —— feat-022 的 BPE 实现没有对应的学习笔记"
    print(f"[refs] 笔记 {len(notes)} 篇：{', '.join(notes)}")

    n_links = 0
    n_paths = 0

    for name in notes:
        path = os.path.join(NOTES_DIR, name)
        text = _read(path)
        lines = text.split("\n")

        # =================================================================
        # 1) 开头必须有「对应 GPT-2 / 对应 Qwen3」两行
        #    （允许在标题后若干行内，但不能整篇都没有）
        # =================================================================
        gpt2_idx = next((i for i, l in enumerate(lines) if l.startswith(_GPT2_LINE)), None)
        qwen3_idx = next((i for i, l in enumerate(lines) if l.startswith(_QWEN3_LINE)), None)
        assert gpt2_idx is not None, f"{name}: 缺少「{_GPT2_LINE}」行"
        assert qwen3_idx is not None, f"{name}: 缺少「{_QWEN3_LINE}」行"
        assert gpt2_idx < 12, f"{name}: 交叉引用块不在开头（第 {gpt2_idx + 1} 行）"

        # =================================================================
        # 2) 交叉引用块里的每个 markdown 链接都要能点开
        # =================================================================
        block_lines = [l for l in lines if l.startswith("> ") and "](" in l]
        assert block_lines, f"{name}: 交叉引用块里没有 markdown 链接"
        for line in block_lines:
            for _text, target in _MD_LINK_RE.findall(line):
                if target.startswith(("http://", "https://", "#")):
                    continue
                n_links += 1
                assert not target.startswith("/"), \
                    f"{name}: 链接用了绝对路径「{target}」，应改成相对路径"
                resolved = os.path.normpath(os.path.join(NOTES_DIR, target))
                assert os.path.isfile(resolved), \
                    f"{name}: 链接目标不存在 → {target}（解析为 {resolved}）"

        # =================================================================
        # 3) 正文里路径型的反引号串必须能在仓库里找到
        # =================================================================
        for raw in re.findall(r"`([^`\n]+)`", text):
            if _is_skippable(raw):
                continue
            if not _PATHLIKE_RE.match(raw):
                continue
            n_paths += 1
            assert _resolve(raw), f"{name}: 路径不存在 → `{raw}`"

    # =================================================================
    # 4) 根 README 的 Qwen3 索引表：覆盖全部笔记 + 带「Qwen3 源码」列
    # =================================================================
    readme = _read(os.path.join(REPO_ROOT, "README.md"))
    assert "| Qwen3 源码 |" in readme, \
        "根 README 的 Qwen3 索引表没有「Qwen3 源码」列"
    for name in notes:
        assert f"qwen3-learning/{name}" in readme, \
            f"根 README 的 Qwen3 索引表缺 {name}"
        # 索引表里的笔记链接必须能点开
        target = os.path.join(REPO_ROOT, "qwen3-learning", name)
        assert os.path.isfile(target), f"根 README 指向的 {name} 不存在"

    # 根 README 里 qwen3-learning/ 下的链接也要能点开
    for _text, target in _MD_LINK_RE.findall(readme):
        if not target.startswith("qwen3-learning/"):
            continue
        if target.startswith(("http://", "https://", "#")):
            continue
        n_links += 1
        resolved = os.path.normpath(os.path.join(REPO_ROOT, target))
        assert os.path.exists(resolved), \
            f"根 README: 链接目标不存在 → {target}"

    # =================================================================
    # 5) qwen3-pytorch/README.md 的「概念 → 笔记 → 源码」表要覆盖全部笔记
    # =================================================================
    q_readme = _read(os.path.join(PROJECT_ROOT, "README.md"))
    for name in notes:
        assert name in q_readme, f"qwen3-pytorch/README.md 缺 {name} 的入口"

    # =================================================================
    # 6) 全仓 markdown 的 `#L<行号>` 锚点：目标存在 + 行号在范围内
    #    （feat-030：链接「点得开」不等于「指得准」）
    # =================================================================
    n_anchor = _check_anchors()

    print(f"[refs] 交叉引用链接 {n_links} 条全部可点开；"
          f"正文路径 {n_paths} 处全部存在；"
          f"源码行号锚点 {n_anchor} 处全部在范围内")
    print(f"test_note_refs OK  笔记 {len(notes)} 篇 / 链接 {n_links} 条 / "
          f"路径 {n_paths} 处 / 锚点 {n_anchor} 处")
    return 0


if __name__ == "__main__":
    sys.exit(main())

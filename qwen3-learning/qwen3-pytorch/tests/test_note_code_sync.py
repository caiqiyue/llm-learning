"""feat-029：**笔记代码块行为守卫** —— 笔记里手抄的函数，行为必须与源码一致。

为什么需要它（feat-028 留下的第二个盲点）
-----------------------------------------
feat-028 把「笔记里的**数字**」纳入机器守卫（参数量 / 词表 / 算式 / 比值）。
但笔记里还有一类东西不是数字，是**代码**：笔记会**手抄**一段实现（`get_lr`、
`_build_sinusoidal_pos_emb`、`mark_only_lora_trainable` …），
而**手抄会抄错** —— 本仓库真的发生过：`gpt-learning/0.GPT全流程详解.md` 里
抄的 `get_lr` 写成了 `learning_rate * it / warmup_iters`（少了一次 +1），
是**靠人眼**发现的（见 `claude-progress.md` Session 012）。

★ 为什么不做「笔记代码块 ↔ 源码逐字一致」
----------------------------------------
**量过了，不可行**（feat-028 立项时实测）：**164/164** 个笔记代码块都是
**故意简化**的摘录（教学笔记本来就该简化），
没有一个与源文件逐字相同 ⇒ 逐字守卫上线当天就是 164 条红灯。

★ 那就换一个问法：**不要求「长得一样」，要求「行为一样」**
--------------------------------------------------
本守卫的做法：**把笔记里那段代码真的跑起来，和源码里那段真的跑起来，
在同样输入上比输出**。
  · 笔记的 `_build_sinusoidal_pos_emb` 少了 docstring、少了几行注释 ——
    但算出来的张量**逐位相同**（`max|diff| = 0.000e+00`）⇒ 通过；
  · 笔记的 `get_lr`（数学公式详解版）把 `config.xxx` 改写成局部变量、
    还多了一行 clamp —— 但**输出逐个相同** ⇒ 通过；
  · 真的抄错了（比如那个少一次 +1 的 `get_lr`）⇒ **输出不一样** ⇒ 红灯。

⇒ 教学笔记可以继续简化，但**简化到行为一致**为止。

★ 两条实现上的关键决定
----------------------
  ① **绝不 import 源码模块** —— 源码函数用 `ast` 从文件里**摘出来**再 `exec`。
     原因：`train_pretrain.py` 是**顶层线性脚本**（没有 `__main__` 保护），
     import 它会**把整个预训练重跑一遍**并覆盖 `checkpoints/pretrain.pt`
     （实测踩过）。`ast` 摘取只取那一个函数，零副作用。
  ② **两边用同一套 globals、两个独立 namespace** ——
     否则笔记的 `get_lr` 会被源码的 `get_lr` 覆盖，等于自己跟自己比。

跑法：
    python tests/test_note_code_sync.py
"""

import ast
import io
import math
import os
import re
import sys
import types

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)            # qwen3-learning/qwen3-pytorch
NOTES_DIR = os.path.dirname(PROJECT_ROOT)            # qwen3-learning
REPO_ROOT = os.path.dirname(NOTES_DIR)               # llm-learning
GPT2_DIR = os.path.join(REPO_ROOT, "gpt-learning", "gpt-learning-pytorch")

FENCE_RE = re.compile(r"^```(\w*)\s*$")


# ---------------------------------------------------------------------------
# 1. 从笔记里抽 python 代码块
# ---------------------------------------------------------------------------


def note_python_blocks(rel: str):
    """返回 [(起始行号, 代码文本)]：该笔记里所有 ```python 围栏块。"""
    path = os.path.join(REPO_ROOT, rel)
    with io.open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    out = []
    i = 0
    while i < len(lines):
        m = FENCE_RE.match(lines[i])
        if not m:
            i += 1
            continue
        lang = m.group(1)
        start = i + 1
        j = start
        while j < len(lines) and not lines[j].startswith("```"):
            j += 1
        if lang == "python":
            out.append((start + 1, "\n".join(lines[start:j])))
        i = j + 1
    return out


def find_note_fn(rel: str, fname: str, occurrence: int = 0):
    """找出笔记里第 `occurrence` 个定义了 `fname` 的 python 块。

    返回 (代码文本, 起始行号)；找不到返回 (None, None)。
    """
    hits = 0
    for lineno, code in note_python_blocks(rel):
        if re.search(rf"^def {re.escape(fname)}\b", code, re.M):
            if hits == occurrence:
                return code, lineno
            hits += 1
    return None, None


# ---------------------------------------------------------------------------
# 2. 从源码里摘函数（**不 import 整个模块** —— 见文件头 ①）
# ---------------------------------------------------------------------------


def extract_source_fn(src_rel: str, fname: str, cls_name: str = None):
    """用 ast 从源码文件里摘出某个函数的**源码文本**。

    ★ 为什么不 import：`train_pretrain.py` 是顶层线性脚本，
    import 会把预训练整个重跑一遍（实测踩过，还会覆盖 checkpoints）。
    ast 摘取只取这一个函数，**零副作用**。

    `cls_name` 非空时，取该类里的方法。
    """
    path = os.path.join(REPO_ROOT, src_rel)
    with io.open(path, encoding="utf-8") as f:
        text = f.read()
    tree = ast.parse(text)
    for node in tree.body:
        if cls_name is None:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name == fname:
                return ast.get_source_segment(text, node)
        elif isinstance(node, ast.ClassDef) and node.name == cls_name:
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and sub.name == fname:
                    return ast.get_source_segment(text, sub)
    return None


# ---------------------------------------------------------------------------
# 3. 运行环境（两边共用同一套 globals）
# ---------------------------------------------------------------------------


def _load_gpt2_config():
    """GPT-2 线的 `config` 模块（按文件路径加载，不污染 sys.path）。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ncs_gpt2_config", os.path.join(GPT2_DIR, "config.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ncs_gpt2_config"] = mod
    spec.loader.exec_module(mod)
    return mod


def _qwen3_env():
    """Qwen3 线要放进 globals 的东西：config / torch / LoRA 相关符号。"""
    import torch
    sys.path.insert(0, PROJECT_ROOT)
    from config import DENSE_CONFIG                     # noqa: E402
    from model.common.lora import (                      # noqa: E402
        LoRALinear, lora_parameters, inject_lora,
    )
    return {
        "torch": torch,
        "DENSE_CONFIG": DENSE_CONFIG,
        "LoRALinear": LoRALinear,
        "lora_parameters": lora_parameters,
        "inject_lora": inject_lora,
    }


def build_globals(env: str) -> dict:
    """按环境名组装 globals —— 笔记那边和源码那边用**同一份**。

    ★ 为什么要 `nn` / `typing`：
      源码函数的**类型注解在 def 时就会求值**（`model: nn.Module` 不是字符串），
      所以 `nn` 必须是真正的 `torch.nn`，`List` 必须是真正的 `typing.List`，
      否则 exec 直接炸在 `'NoneType' object has no attribute 'Module'`。
    """
    import typing

    import torch
    base = {
        "math": math,
        "torch": torch,
        "nn": torch.nn,
        "List": typing.List,
        "Tuple": typing.Tuple,
        "Optional": typing.Optional,
        "Dict": typing.Dict,
    }
    if env == "gpt2":
        base["config"] = _load_gpt2_config()
    elif env == "qwen3":
        base.update(_qwen3_env())
    else:
        raise ValueError(f"未知环境 {env}")
    return base


# ---------------------------------------------------------------------------
# 4. 比较
# ---------------------------------------------------------------------------


def compare(a, b, atol: float = 1e-9, rtol: float = 1e-6):
    """比较两边的输出，返回 (是否一致, 说明)。

    ★ 为什么用「绝对 + 相对」双阈值：
      学习率这类量级只有 `1e-4`，纯绝对阈值（1e-6）会让
      「少一次 +1」这种真错误只剩 6e-6 的差 —— 刚好卡在阈值边上，很脆。
      加上相对阈值后，量级越小判得越严。
    """
    import torch
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        ta, tb = torch.as_tensor(a), torch.as_tensor(b)
        if ta.shape != tb.shape:
            return False, f"形状不同：{tuple(ta.shape)} vs {tuple(tb.shape)}"
        d = float((ta.double() - tb.double()).abs().max())
        scale = float(ta.double().abs().max())
        return d <= atol + rtol * scale, f"max|diff| = {d:.3e}"
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            return False, f"长度不同：{len(a)} vs {len(b)}"
        for x, y in zip(a, b):
            ok, why = compare(x, y, atol, rtol)
            if not ok:
                return False, why
        return True, "逐项相等"
    if isinstance(a, float) or isinstance(b, float):
        d = abs(float(a) - float(b))
        return d <= atol + rtol * max(abs(float(a)), abs(float(b))), \
            f"|diff| = {d:.3e}"
    return a == b, f"{a!r} vs {b!r}"


def compare_params(a, b):
    """比较两组可训练参数：个数 + 每一位的 requires_grad。"""
    if len(a) != len(b):
        return False, f"可训练参数个数不同：{len(a)} vs {len(b)}"
    fa = [bool(p.requires_grad) for p in a]
    fb = [bool(p.requires_grad) for p in b]
    if fa != fb:
        return False, f"requires_grad 分布不同：{sum(fa)} 个 True vs {sum(fb)} 个 True"
    return True, f"可训练参数 {len(a)} 个，requires_grad 分布一致"


# ---------------------------------------------------------------------------
# 5. 配对表：**笔记里的手抄代码 ↔ 源码里的实现**
# ---------------------------------------------------------------------------
# 每一条 = 一个「笔记里的函数」和「源码里的函数」必须行为一致。
# `calls` 是喂给两边的同样输入；`prepare` 负责准备需要的环境对象。


def _lora_model():
    """造一个注入了 LoRA 的小模型（给 LoRA 那两条用）。"""
    import torch
    sys.path.insert(0, PROJECT_ROOT)
    from config import DENSE_CONFIG
    from model.qwen3_dense import Qwen3Dense
    from model.common.lora import inject_lora
    m = Qwen3Dense(vocab_size=256, cfg=DENSE_CONFIG, dropout=0.0)
    inject_lora(m, DENSE_CONFIG.lora_target_modules,
                r=DENSE_CONFIG.lora_r, alpha=DENSE_CONFIG.lora_alpha)
    return m


def _kv_stub():
    """造一个 KVCache 形状的替身（给 `append` 那条用）。"""
    import torch
    n_layer, B, n_head, max_seq, head_dim = 2, 1, 2, 8, 4
    st = types.SimpleNamespace()
    st.keys = torch.zeros(n_layer, B, n_head, max_seq, head_dim)
    st.values = torch.zeros_like(st.keys)
    st._seq_lens = [0] * n_layer
    st.max_seq_len = max_seq
    st.n_layer = n_layer
    return st


def _kv_call(layer_idx: int, t_new: int):
    """生成一次 `append(layer_idx, k_new, v_new)` 的调用工厂。

    ★ 必须是**工厂**：`append` 是有状态的（会写 cache），
    笔记那边和源码那边必须各拿一个**全新的** cache，
    否则第二次调用会接着第一次的位置写 ⇒ 必然不一致。

    ★ 但 **K/V 张量必须是同一份**：如果两边各自 random，
    比的就是两份不同的随机数 ⇒ 必然「不一致」，
    那测的就不是「行为是否一致」而是「随机数是否一样」了。
    """
    import torch
    k = torch.randn(1, 2, t_new, 4)
    v = torch.randn(1, 2, t_new, 4)

    def _make():
        return (_kv_stub(), layer_idx, k, v)
    return _make


def _lora_call(*tail):
    """生成一次「拿一个注入了 LoRA 的模型」的调用工厂（每边一个独立模型）。"""
    def _make():
        return (_lora_model(),) + tail
    return _make


PAIRS = [
    # ---------------- GPT-2 线 ----------------
    {
        "id": "gpt2/get_lr（全流程详解）",
        "note": "gpt-learning/0.GPT全流程详解.md",
        "note_fn": "get_lr",
        "src": "gpt-learning/gpt-learning-pytorch/train_pretrain.py",
        "src_fn": "get_lr",
        "env": "gpt2",
        "calls": [(i,) for i in (0, 1, 25, 49, 50, 51, 100, 199, 200, 201, 500)],
    },
    {
        "id": "gpt2/get_lr（数学公式详解）",
        "note": "gpt-learning/GPT2训练数学公式与超参详解.md",
        "note_fn": "get_lr",
        "src": "gpt-learning/gpt-learning-pytorch/train_pretrain.py",
        "src_fn": "get_lr",
        "env": "gpt2",
        "calls": [(i,) for i in (0, 1, 25, 49, 50, 51, 100, 199, 200, 201, 500)],
    },
    {
        "id": "gpt2/_build_sinusoidal_pos_emb（全流程详解）",
        "note": "gpt-learning/0.GPT全流程详解.md",
        "note_fn": "_build_sinusoidal_pos_emb",
        "src": "gpt-learning/gpt-learning-pytorch/model/gpt2.py",
        "src_fn": "_build_sinusoidal_pos_emb",
        "env": "gpt2",
        "calls": [(128, 64), (64, 32), (32, 16)],
    },
    {
        "id": "gpt2/_build_sinusoidal_pos_emb（位置编码）",
        "note": "gpt-learning/2.位置编码.md",
        "note_fn": "_build_sinusoidal_pos_emb",
        "src": "gpt-learning/gpt-learning-pytorch/model/gpt2.py",
        "src_fn": "_build_sinusoidal_pos_emb",
        "env": "gpt2",
        "calls": [(128, 64), (64, 32), (32, 16)],
    },
    {
        "id": "gpt2/_build_sinusoidal_pos_emb（数学公式详解）",
        "note": "gpt-learning/GPT2训练数学公式与超参详解.md",
        "note_fn": "_build_sinusoidal_pos_emb",
        "src": "gpt-learning/gpt-learning-pytorch/model/gpt2.py",
        "src_fn": "_build_sinusoidal_pos_emb",
        "env": "gpt2",
        "calls": [(128, 64), (64, 32), (32, 16)],
    },
    {
        "id": "gpt2/KVCache.append（KV Cache 写入流程）",
        "note": "gpt-learning/0.GPT全流程详解.md",
        "note_fn": "append",
        "src": "gpt-learning/gpt-learning-pytorch/inference.py",
        "src_fn": "append",
        "src_cls": "KVCache",
        "env": "gpt2",
        # calls 是**工厂**：每边各调一次 ⇒ 两边拿到的是各自独立的 cache
        "calls": [_kv_call(0, 1), _kv_call(1, 2), _kv_call(0, 3)],
        "effect": "kv_state",
    },
    # ---------------- Qwen3 线 ----------------
    {
        "id": "qwen3/mark_only_lora_trainable",
        "note": "qwen3-learning/Qwen3_LoRA与DPO_学习笔记.md",
        "note_fn": "mark_only_lora_trainable",
        "src": "qwen3-learning/qwen3-pytorch/model/common/lora.py",
        "src_fn": "mark_only_lora_trainable",
        "env": "qwen3",
        "calls": [_lora_call(), _lora_call()],
        "compare": "params",
    },
    {
        "id": "qwen3/set_lora_enabled",
        "note": "qwen3-learning/Qwen3_LoRA与DPO_学习笔记.md",
        "note_fn": "set_lora_enabled",
        "src": "qwen3-learning/qwen3-pytorch/model/common/lora.py",
        "src_fn": "set_lora_enabled",
        "env": "qwen3",
        "calls": [_lora_call(True), _lora_call(False)],
        "effect": "lora_scaling",
    },
]


# ---------------------------------------------------------------------------
# 6. 主检查
# ---------------------------------------------------------------------------


def _prepare(spec, kind):
    """按 spec 准备调用现场（保留给「不用工厂」的旧式配对）。"""
    if kind == "lora":
        return _lora_model()
    if kind == "kv":
        return _kv_stub()
    return None


def _effect_value(kind, obj):
    """调用之后要比较的「效果」（函数本身返回 None 时用）。"""
    if kind == "lora_scaling":
        from model.common.lora import LoRALinear
        return [round(float(m.scaling), 6)
                for m in obj.modules() if isinstance(m, LoRALinear)]
    if kind == "kv_state":
        return (list(obj._seq_lens), obj.keys.clone(), obj.values.clone())
    raise ValueError(f"未知 effect {kind}")


def run_pair(spec):
    """跑一条配对，返回 (是否通过, 说明, 细节)。"""
    import torch

    # --- 笔记侧：抽块 + exec ---
    code, lineno = find_note_fn(spec["note"], spec["note_fn"])
    if code is None:
        return False, f"笔记里找不到 `def {spec['note_fn']}` 的 python 块", ""
    note_ns = build_globals(spec["env"])
    try:
        exec(compile(code, f"{spec['note']}:{lineno}", "exec"), note_ns)
    except Exception as exc:                      # noqa: BLE001
        return False, f"笔记里的代码块跑不起来：{type(exc).__name__}: {exc}", ""
    note_fn = note_ns.get(spec["note_fn"])
    if note_fn is None:
        return False, f"笔记代码块里没有 `{spec['note_fn']}`", ""

    # --- 源码侧：ast 摘函数 + exec（**不 import 模块**）---
    src_code = extract_source_fn(spec["src"], spec["src_fn"], spec.get("src_cls"))
    if src_code is None:
        return False, f"源码里找不到 `{spec['src_fn']}`", ""
    src_ns = build_globals(spec["env"])
    try:
        exec(compile(src_code, spec["src"], "exec"), src_ns)
    except Exception as exc:                      # noqa: BLE001
        return False, f"源码函数跑不起来：{type(exc).__name__}: {exc}", ""
    src_fn = src_ns.get(spec["src_fn"])
    if src_fn is None:
        return False, f"源码摘出来的代码里没有 `{spec['src_fn']}`", ""

    # --- 同样输入，两边都跑 ---
    cmp_kind = spec.get("compare", "auto")
    details = []
    for call in spec["calls"]:
        # ★ 每边各调一次工厂 ⇒ 两边拿到**各自独立**的对象
        #   （`append` / `set_lora_enabled` 是有状态的，共用对象必然不一致）
        if callable(call):
            args_note = list(call())
            args_src = list(call())
        else:
            args_note = list(call)
            args_src = list(call)
            if spec.get("prepare"):
                args_note.insert(0, _prepare(spec, spec["prepare"]))
                args_src.insert(0, _prepare(spec, spec["prepare"]))
        try:
            out_note = note_fn(*args_note)
        except Exception as exc:                  # noqa: BLE001
            return False, f"笔记里的函数调用失败：{type(exc).__name__}: {exc}", ""
        try:
            out_src = src_fn(*args_src)
        except Exception as exc:                  # noqa: BLE001
            return False, f"源码函数调用失败：{type(exc).__name__}: {exc}", ""

        if spec.get("effect"):
            v_note = _effect_value(spec["effect"], args_note[0])
            v_src = _effect_value(spec["effect"], args_src[0])
            ok, why = compare(v_note, v_src)
        elif cmp_kind == "params":
            ok, why = compare_params(out_note, out_src)
        else:
            ok, why = compare(out_note, out_src)
        details.append(f"→ {why}")
        if not ok:
            return False, f"行为不一致：{why}", "\n      ".join(details)
    return True, "行为一致", "\n      ".join(details)


def main() -> int:
    problems = []
    passed = 0
    for spec in PAIRS:
        ok, why, details = run_pair(spec)
        if ok:
            passed += 1
            print(f"  [PASS] {spec['id']}  —— {why}")
        else:
            problems.append((spec["id"], why, details))
            print(f"  [FAIL] {spec['id']}  —— {why}")

    print()
    if problems:
        print("[note-code] 笔记里的手抄代码与源码行为不一致：\n")
        for pid, why, details in problems:
            print(f"  {pid}\n      {why}")
            if details:
                print(f"      {details}")
        print("\n处理办法：")
        print("  a) 改笔记，让它的行为跟源码一致（教学简化可以，但结果必须一样）；")
        print("  b) 改源码 —— 那笔记也要一起改（期望值跟着代码走）；")
        print("  c) 这确实是「教学版写法」，行为差异是**有意**的 ⇒ 在 PAIRS 里登记理由。")
        print(f"\ntest_note_code_sync FAIL  不一致 {len(problems)} 条 / 共 {len(PAIRS)} 条")
        return 1

    print(f"[note-code] 核对 {len(PAIRS)} 条「笔记手抄代码 ↔ 源码」配对，"
          f"全部行为一致（笔记 {len(set(s['note'] for s in PAIRS))} 篇）")
    print("test_note_code_sync OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

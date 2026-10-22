"""feat-009：笔记里的超参数数字 ↔ 各自 config.py 的自动一致性守卫。

为什么需要它
------------
「笔记和代码必须对得上」是这个仓库的第一优先级，但**靠人读一遍是守不住的**：
`config.py` 改一个数字，几十篇笔记里的引用不会自动跟着变。
本测试把「笔记里写的超参 = 值」当成**可断言的契约**：

  1. 直接 **import** 两个 `config.py`（不是正则猜），拿到真实值；
  2. 扫两条线的 **27 篇**笔记（17 篇 GPT-2 + 9 篇 Qwen3 + `qwen3-pytorch设计.md`），
     跳过围栏代码块，抽出形如 `` `key = value` `` 的写法；
  3. 每一条都必须命中该线 config 里的同名 key 的某个取值；
  4. 命中不了的，必须在 `EXCEPTIONS` 里**显式登记原因**（教学举例 / 官方模型参数 /
     算法定义表 / 退化开关值 / 训练阶段覆盖值 ……）。

所以：**想新增一个「例外」必须写一行理由**，不能悄悄放过。
登记簿里**已经不再命中**的条目（说明笔记被改过了）会被自动统计出来并打印，
提醒把过时的登记删掉 —— 否则例外簿会越积越脏、假阳性越来越多。

★ 已知盲点（不是「把报错关掉」，是「换一种对的核对方式」）
-------------------------------------------------------
**「目录 → config」的规则对「跨线比较的文档」不成立**：
`qwen3-learning/` 里的文档如果引用了 **GPT-2** 的值（例如
「GPT-2 的 `warmup_iters = 200`」），本守卫会拿 Qwen3 的 config 去比 ⇒ 误报。
所以 `../qwen3-pytorch开发计划.md` 被**排除在自动守卫之外**（见 `MANUAL_AUDIT_DOCS`），
改为每次重写该 Task 时人工核对。

跑法：
    python tests/test_config_sync.py
"""

import importlib.util
import io
import os
import re
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)            # qwen3-learning/qwen3-pytorch
NOTES_DIR = os.path.dirname(PROJECT_ROOT)            # qwen3-learning
REPO_ROOT = os.path.dirname(NOTES_DIR)               # llm-learning

GPT2_DIR = os.path.join(REPO_ROOT, "gpt-learning", "gpt-learning-pytorch")

# ---------------------------------------------------------------------------
# 1. 载入两边的 config（按文件路径加载，避开 model 包名冲突）
# ---------------------------------------------------------------------------


def _load_module(name: str, path: str):
    """按**文件路径**加载一个模块（不能用 `import model.xxx`，包名会被占用）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_gpt2_cfg_mod = _load_module("cfg_gpt2_sync", os.path.join(GPT2_DIR, "config.py"))
_qwen3_cfg_mod = _load_module("cfg_qwen3_sync", os.path.join(PROJECT_ROOT, "config.py"))

GPT2_CFG = {k: v for k, v in vars(_gpt2_cfg_mod).items()
            if not k.startswith("__") and isinstance(v, (int, float, str, bool))}

_DENSE = _qwen3_cfg_mod.DENSE_CONFIG
_MOE = _qwen3_cfg_mod.MOE_CONFIG
QWEN3_CFG = {}
for _obj in (_DENSE, _MOE):
    for _k, _v in _obj.to_dict().items():
        if _v is not None:
            QWEN3_CFG[_k] = _v

# ---------------------------------------------------------------------------
# 2. 笔记范围 → 用哪份 config 核对
# ---------------------------------------------------------------------------

GPT2_NOTES = []
for _n in sorted(os.listdir(os.path.join(REPO_ROOT, "gpt-learning"))):
    if _n.endswith(".md"):
        GPT2_NOTES.append(os.path.join("gpt-learning", _n))

# Qwen3 线：**9 篇学习笔记 + 设计.md** 进自动守卫
QWEN3_NOTES = []
for _n in sorted(os.listdir(NOTES_DIR)):
    if _n.endswith(".md") and (_n.startswith("Qwen3_") or _n == "qwen3-pytorch设计.md"):
        QWEN3_NOTES.append(os.path.join("qwen3-learning", _n))

# 开发计划.md **不进自动守卫** ——
# 它是分层工程文档：里面既有「初稿」里的旧数字，也有各 Task 的 Expected 输出、
# 还有对历史错误的复盘引用。这些数字的核对方式是**每次重写该 Task 时人工核对**
# （Task 19 / Task 20 都是这么做的），自动守卫反而会因为「引用一个反例」而误报。
MANUAL_AUDIT_DOCS = ["qwen3-learning/qwen3-pytorch开发计划.md"]

ALL_NOTES = [(p, "gpt2") for p in GPT2_NOTES] + [(p, "qwen3") for p in QWEN3_NOTES]

CFG_OF = {"gpt2": GPT2_CFG, "qwen3": QWEN3_CFG}

# ---------------------------------------------------------------------------
# 3. 例外登记簿：note 相对路径 + key + 值 → 为什么它不是「本仓库当前配置」
# ---------------------------------------------------------------------------
# 每一行都必须写清理由。分类：
#   TEACHING   教学举例用的玩具尺寸，不是本仓库配置
#   UPSTREAM   官方模型（GPT-2 / Qwen3-0.6B）的参数，本仓库是 tiny 版
#   ALGO       算法定义表（不同采样参数的**含义**），不是项目默认值
#   SWITCH     对照实验的退化开关取值
#   PHASE      训练阶段覆盖值（config 里的值只适用于别的阶段）
#   SMOKE      单元 smoke 用的独立小尺寸
EXCEPTIONS = {
    # ---- GPT-2 线 ----
    ("gpt-learning/12.采样方式.md", "temperature", "0"):
        "ALGO: 采样参数定义表（贪心 = temperature 0）",
    ("gpt-learning/12.采样方式.md", "temperature", "1"):
        "ALGO: 采样参数定义表（从完整词表随机采样）",
    ("gpt-learning/12.采样方式.md", "top_k", "None"):
        "ALGO: 采样参数定义表（top_k=None = 不截断）",
    ("gpt-learning/12.采样方式.md", "top_p", "None"):
        "ALGO: 采样参数定义表（top_p=None = 不截断）",
    ("gpt-learning/12.采样方式.md", "top_k", "1"):
        "ALGO: top_k=1 ≈ greedy，是算法定义",
    ("gpt-learning/12.采样方式.md", "top_p", "0.8"):
        "ALGO: 举例用的 top_p 取值，不是项目默认值",
    ("gpt-learning/12.采样方式.md", "top_p", "0.9"):
        "ALGO: 举例用的 top_p 取值，不是项目默认值",
    ("gpt-learning/2.位置编码.md", "n_embd", "4"):
        "TEACHING: 讲位置向量维度用的玩具例子",
    ("gpt-learning/2.位置编码.md", "block_size", "8"):
        "TEACHING: 讲滑窗切分用的玩具例子",
    ("gpt-learning/2.位置编码.md", "n_embd", "768"):
        "UPSTREAM: 官方 GPT-2 (124M) 的 hidden size，本仓库是 tiny 版 128",

    # ---- Qwen3 线 ----
    ("qwen3-learning/Qwen3_GQA与KV_Cache_完整学习笔记.md", "head_dim", "2"):
        "TEACHING: 手算 GQA 的玩具例子（2 个 Q head / 1 个 KV head）",
    ("qwen3-learning/Qwen3_SwiGLU_MLP_学习笔记.md", "intermediate_size", "3072"):
        "UPSTREAM: 官方 Qwen3-0.6B 的 intermediate_size（hidden 1024 → 3072 = 3×）",
    ("qwen3-learning/Qwen3_LoRA与DPO_学习笔记.md", "dropout", "0.0"):
        "PHASE: LoRA/DPO 阶段显式传 dropout=0.0；config.dropout=0.1 只用于预训练",
    ("qwen3-learning/Qwen3_训练推理完整学习笔记.md", "top_p", "0.8"):
        "ALGO: 课后题里的 top_p 取值，不是项目默认值",
    ("qwen3-learning/qwen3-pytorch设计.md", "n_head", "32"):
        "UPSTREAM: 官方 Qwen3-0.6B 的 n_head（hidden 1024 / head_dim 32）",
    ("qwen3-learning/qwen3-pytorch设计.md", "head_dim", "8"):
        "SMOKE: RoPE 旋转等变性单元 smoke 的独立小尺寸",
    ("qwen3-learning/qwen3-pytorch设计.md", "n_head", "8"):
        "SMOKE: KV Cache / Attention 单元 smoke 的独立小尺寸",
    ("qwen3-learning/qwen3-pytorch设计.md", "mlp_type", "gelu"):
        "SWITCH: 对照实验的退化开关取值（主线是 swiglu）",
    ("qwen3-learning/qwen3-pytorch设计.md", "norm_type", "layernorm"):
        "SWITCH: 对照实验的退化开关取值（主线是 rmsnorm）",
    ("qwen3-learning/qwen3-pytorch设计.md", "dropout", "0.0"):
        "SIGNATURE: 这是 `LoRALinear(base, r, alpha, dropout=0.0)` 签名里的默认值，"
        "对应 config.lora_dropout=0.0，不是 cfg.dropout",
    ("qwen3-learning/qwen3-pytorch设计.md", "use_gqa", "False"):
        "SWITCH: 对照实验的退化开关取值（主线是 use_gqa=True）",
    ("qwen3-learning/qwen3-pytorch设计.md", "use_rope", "False"):
        "SWITCH: 对照实验的退化开关取值（主线是 use_rope=True）",
    ("qwen3-learning/qwen3-pytorch设计.md", "use_qk_norm", "False"):
        "SWITCH: 对照实验的退化开关取值（主线是 use_qk_norm=True）",

    # ---- 训练推理笔记里的退化开关 ----
    ("qwen3-learning/Qwen3_训练推理完整学习笔记.md", "use_rope", "False"):
        "SWITCH: 对照实验的退化开关取值（主线是 use_rope=True）",
    ("qwen3-learning/Qwen3_训练推理完整学习笔记.md", "n_head_kv", "n_head"):
        "ALGO: `n_head_kv = n_head` 描述的是**退化开关关闭后**的 MHA 情形，不是当前配置",

    # ---- 开发计划.md 不在自动守卫范围内（见 MANUAL_AUDIT_DOCS）----
}

# ---------------------------------------------------------------------------
# 4. 抽取笔记里的 `` `key = value` `` 写法
# ---------------------------------------------------------------------------

# 只认被反引号包住的 `key = value`；value 允许数字 / 字符串字面量 / None / 表达式
_CLAIM_RE = re.compile(r"`([^`]+)`")
_KV_RE = re.compile(r"([a-z_][a-z0-9_]*)\s*=\s*([^,;/、=\s]+)")
_NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:e[-+]?\d+)?$", re.IGNORECASE)
# 表达式（不是字面量）：无法与 config 比较，遇到就跳过
_EXPR_CHARS = "()[]·×*/+"


def _normalize(value: str):
    """把字面量归一化成可比较的形式；不是字面量（是表达式）返回 None（跳过）。

    会被跳过的：
      * 数字        → float
      * None/True/False / 普通单词（'gelu'、'layernorm'）→ 字符串，**照常比对**
      * 含 `.` 的属性/下标表达式（`cfg.dropout`）→ None
      * 含算术/括号符号的表达式（`4·n_embd`）→ None
    """
    v = value.strip()
    v = v.rstrip("。，,、;；)）】")
    v = v.strip("'\"")
    if _NUM_RE.match(v):
        try:
            return float(v)
        except ValueError:
            return None
    if v in ("None", "True", "False"):
        return v
    if any(ch in _EXPR_CHARS for ch in v):
        return None      # 形如 `4·n_embd` / `(a+1)` 的表达式，不可比较
    if "." in v:
        return None      # 形如 `cfg.dropout` 的属性表达式，不可比较
    return v


def _config_values(cfg: dict, key: str):
    """某个 key 在 config 里的所有可接受取值（归一化后）。"""
    if key not in cfg:
        return None
    n = _normalize(str(cfg[key]))
    return [] if n is None else [n]


def _scan(path: str):
    """扫一篇笔记，返回 [(行号, key, 归一化值, 原始文本)]。

    反引号里可能有多个 kv（`temperature = 1, top_k=None, top_p=None`），
    按 `,` `/` `、` 拆开逐个比对。
    """
    out = []
    with io.open(path, encoding="utf-8") as f:
        in_fence = False
        for i, line in enumerate(f.read().split("\n"), 1):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            for m in _CLAIM_RE.finditer(line):
                chunk = m.group(1)
                for km in _KV_RE.finditer(chunk):
                    key = km.group(1)
                    raw = km.group(2)
                    norm = _normalize(raw)
                    if norm is None:
                        continue
                    out.append((i, key, norm, raw))
    return out


def main() -> int:
    checked = 0
    mismatches = []
    skipped_key = 0
    used = set()

    for rel, scope in ALL_NOTES:
        cfg = CFG_OF[scope]
        path = os.path.join(REPO_ROOT, rel)
        if not os.path.isfile(path):
            continue
        for lineno, key, norm, raw in _scan(path):
            vals = _config_values(cfg, key)
            if vals is None:
                skipped_key += 1          # 该线 config 里没这个 key（跨线/推导量）
                continue
            checked += 1
            if norm in vals:
                continue
            exc_key = (rel.replace("\\", "/"), key,
                       str(raw).strip().strip("'\"").rstrip("。，,、;；)）】"))
            if exc_key in EXCEPTIONS:
                used.add(exc_key)
            else:
                mismatches.append((rel, lineno, key, raw, scope))

    unused = sorted(set(EXCEPTIONS) - used)

    if mismatches:
        print("[sync] 发现未登记的不一致：\n")
        for rel, lineno, key, raw, scope in mismatches:
            cfg = CFG_OF[scope]
            print(f"  {rel}:{lineno}  `{key} = {raw}`  "
                  f"→ config 实际值 = {cfg.get(key, '<无此 key>')}")
        print("\n处理办法（二选一）：")
        print("  a) 改笔记，让它跟 config 一致；")
        print("  b) 改 config，让它跟笔记一致（要同步受影响的其他笔记）；")
        print("  c) 这确实是「不是本仓库配置」的取值 ⇒ 在 EXCEPTIONS 里登记理由。")
        print(f"\ntest_config_sync FAIL  不一致 {len(mismatches)} 处")
        return 1

    if unused:
        print("[sync] ⚠ 以下例外已不再命中（笔记被改过了），请删除这些登记：")
        for rel, key, val in unused:
            print(f"  ({rel}, {key!r}, {val!r})")
        print()

    print(f"[sync] 核对笔记 {len(ALL_NOTES)} 篇；命中超参断言 {checked} 处；"
          f"例外登记 {len(EXCEPTIONS)} 条（命中 {len(used)}，过时 {len(unused)}）；"
          f"跨线/推导量跳过 {skipped_key} 处")
    print(f"[sync] 人工核对文档（不进自动守卫）：{', '.join(MANUAL_AUDIT_DOCS)}")
    print("test_config_sync OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

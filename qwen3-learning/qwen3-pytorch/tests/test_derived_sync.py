"""feat-028：**派生量守卫** —— 笔记里的「参数量 / 词表大小 / 算式 / 比值」必须与代码实测一致。

为什么需要它（feat-009 守卫的第三个盲点）
--------------------------------------------
`test_config_sync.py` 守的是「写成 `键 = 值` 的**超参主张**」，
但笔记里还有一类**派生量**：它们**不是 config 里的值，而是算出来的** ——
参数量（794,496 / 985,856 / 8,704 / 1,150,464 …）、词表大小（45 / 778）、
单层 FFN 参数量（SwiGLU 147,456 vs GPT-2 GELU 131,072）及其比值（1.125 / 1.21x）。
config 守卫**看不见它们**（它们没有 `键 = 值` 的写法），
而它们恰恰是**最容易漂移**的：改任何一个 `n_embd` / `n_layer` / `vocab_size`，
几十个参数量数字会**同时**变成错的，而且没有任何地方会报错。

★ 这个守卫的做法：**不手写期望值，而是现场把模型搭出来数参数。**
------------------------------------------------------------------
期望值全部来自**代码**：
  · GPT-2：按 `config.py` 造一个模型，数 `parameters()`；再按 config 推出
    各个零件的参数量（embedding / 每个 block / final LN / 不共享时的双份 head），
    用来核对笔记 §1.9、§5 那种**逐项拆解**的写法；
  · Qwen3：按 `DENSE_CONFIG` / `MOE_CONFIG` 造 dense / MoE 模型，数参数；
    再注入 LoRA，数「可训练参数 / 总参数」；
    ★ feat-031 追加：**MoE 的两种 LoRA 配方**（只注 attention /
    attention + 每个专家的 FFN）各自的「总参数 / 可训练参数」——
    否则笔记里那张三配方对照表的核心数字（57,856 / 1,242,368）
    就是**没人核对**的。
  · 词表大小：直接走 GPT-2 线自己的 `CharTokenizer` 读 `vocab.txt`、
    读 Qwen3 线的 `vocab.json` —— **不数行**（词表里有「空格」这种行，
    `strip()` 会把它数掉，本仓实测就是数成 44 而不是 45）。
⇒ 期望值**不是抄来的常量**，config 一改，期望值跟着变，笔记不改就红。

★ 六条规则
----------
  R1 **算式自洽**：笔记里「45 × 128 = 5,760」这种算式必须算得对
     （与是不是本仓的配置无关 —— 算错了就是错了）。
  R2 **参数量（两路）**：
     · R2a **紧邻式**：`参数量 = 794,496` / `11520 个参数` —— 关键词紧邻数字；
     · R2b **行级式**：**只要这一行出现参数关键词**
       （`参数 / 可训练 / params`），这一行上**每一个**「参数化数字」
       （带千分位或 ≥5 位）都必须能对上是实测值。
       ★ 为什么必须有 R2b：笔记里最常见的写法是**表格**——
         `| 总可训练参数 | 见 §1.9 | **794,496** |`，
         关键词和数字被 `|` 隔开 ⇒ R2a 抓不到 ⇒ 数字实际上**没被核对**。
         实测（feat-028 立项时）：R2a 只覆盖 21 处主张，
         补上 R2b 后覆盖 **41 处**，新增的 20 处里有 6 处是
         **表格 / 散文里从来没被核对过的参数量**。
  R3 **词表大小**：`vocab / 词表` 附近的那个数必须等于实测词表大小。
  R4 **三元组**：`8,704 / 994,560 (0.88%)` —— 分子分母都要是实测值，
     且百分比必须真的等于分子/分母。
  R5 **比值**：`比值 = 1.21x` 必须真的等于该行两个数字之比。
  R6 **总计**：`可训练总计: A + B + C = N` 里的 N 必须等于**模型实测总参数量**。

★ 归属判定（R2b 用）
------------------
每个数字先按**它前面最近的关键词**判归属（`GPT-2` → gpt2 侧，
`Qwen3 / SwiGLU / dense / moe / LoRA` → qwen3 侧），
判不出来就按**文件所在目录**判（`gpt-learning/` → gpt2 侧，
`qwen3-learning/` → qwen3 侧）。
⇒ 「GPT-2 的 794,496」和「Qwen3 的 794,496」会被分别对待：
后者会被红掉（794,496 不在 Qwen3 侧的实测集合里）。
归属判错但数字本身是实测值时，报**提示**而不是红灯 —— 归属启发式不该拦住 CI。

★ 为什么不用「正则抓散文里的『128 维』『4 层』」那种做法
--------------------------------------------------------
**量过了，不可行**（feat-028 立项时的实测）：
  · 「层」在本仓的笔记里**语义冲突** —— `init.sh` 的 **L4 层**（18 处）、
    「4 层 Transformer」、SwiGLU 的 3 个矩阵，全是「层」；
  · 「维」的例子大多是**教学玩具值**（`n_embd = 4` → 4 维）或**官方模型值**（768 维）；
  · 真正带 key 名的散文主张只有 **12 处**，其中还有一半是跨线引用 / 人工核对文档。
⇒ 「把散文数字也纳入守卫」= **误报机器**，不是守卫。
所以本守卫只做**能算得出真值**的那一类：参数量、词表大小、算式、比值。

已知边界（不是「把报错关掉」，是「换一种对的核对方式」）
--------------------------------------------------------
  ① **跨线比较的文档不进自动守卫**（`qwen3-pytorch开发计划.md`）——
     理由同 `test_config_sync.py`：它是分层工程文档，里面有「初稿旧数字」与
     「历史错误的复盘引用」，自动核对会误报。改为**人工核对**。
  ② **测量值类数字不在范围内**（耗时、加速比、loss、采样类指标）——
     它们**不是 config 的函数**，靠采样/机器负载浮动，锁死小数点只会制造假红灯。
  ③ **不带参数上下文的分数不进守卫** —— 「已完成 0 / 17 (0%)」是进度计数，
     不是参数量比例 ⇒ R4 要求同一行里出现「参数 / params / 可训练」。

跑法：
    python tests/test_derived_sync.py
"""

import importlib.util
import io
import json
import os
import re
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)            # qwen3-learning/qwen3-pytorch
NOTES_DIR = os.path.dirname(PROJECT_ROOT)            # qwen3-learning
REPO_ROOT = os.path.dirname(NOTES_DIR)               # llm-learning
GPT2_DIR = os.path.join(REPO_ROOT, "gpt-learning", "gpt-learning-pytorch")

# 开发计划.md 不进自动守卫（见上面的边界 ①）
MANUAL_AUDIT_DOCS = ["qwen3-learning/qwen3-pytorch开发计划.md"]


# ---------------------------------------------------------------------------
# 1. 按文件路径加载模块（`model` 包名会被两条线占用 ⇒ 不能用 import model.xxx）
# ---------------------------------------------------------------------------


def _load_module(name: str, path: str):
    """按**文件路径**加载一个模块（`model` 包名已被 Qwen3 线占用）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _num(params) -> int:
    """一组参数的总个数。"""
    return sum(p.numel() for p in params)


# ---------------------------------------------------------------------------
# 2. 真值：**现场把模型搭出来数参数**
# ---------------------------------------------------------------------------


def _ground_truth() -> dict:
    """算出所有派生量的真值（全部来自代码，不含手写常量）。"""
    import torch  # noqa: F401  （延迟 import：让 import 失败时报错更直白）

    # ---- GPT-2 线 ----
    g_cfg = _load_module("dg_cfg_gpt2", os.path.join(GPT2_DIR, "config.py"))
    g_tok = _load_module("dg_tok_gpt2", os.path.join(GPT2_DIR, "data", "tokenizer.py"))
    g_mod = _load_module("dg_gpt2", os.path.join(GPT2_DIR, "model", "gpt2.py"))

    g_vocab = g_tok.CharTokenizer(os.path.join(GPT2_DIR, "data", "vocab.txt")).vocab_size
    g_model = g_mod.GPT2(
        vocab_size=g_vocab,
        n_layer=g_cfg.n_layer,
        n_head=g_cfg.n_head,
        n_embd=g_cfg.n_embd,
        n_inner=g_cfg.n_inner,
        block_size=g_cfg.block_size,
        dropout=0.0,
        bias=g_cfg.bias,
    )
    g_total = _num(g_model.parameters())
    g_ln = 2 * g_cfg.n_embd                        # LayerNorm = weight + bias
    g_mlp = 2 * g_cfg.n_embd * g_cfg.n_inner       # c_fc + c_proj
    g_cattn = g_cfg.n_embd * 3 * g_cfg.n_embd      # QKV 合并投影
    g_cproj = g_cfg.n_embd * g_cfg.n_embd          # 输出投影
    g_block = 2 * g_ln + g_cattn + g_cproj + g_mlp  # 一个 block：LN_1 + attn + LN_2 + mlp
    gpt2_vals = {
        g_total,                                   # 可训练总计
        g_vocab * g_cfg.n_embd,                    # tok_emb
        2 * g_vocab * g_cfg.n_embd,                # 不共享 weight tying 时的双份
        g_cfg.block_size * g_cfg.n_embd,           # pos_emb（buffer，不参与训练）
        g_ln,                                      # 单个 LayerNorm / final ln_f
        g_cattn, g_cproj, g_mlp, g_block,
        g_cfg.n_layer * g_block,                   # 全部 block
    }

    # ---- Qwen3 线 ----
    sys.path.insert(0, PROJECT_ROOT)
    from config import DENSE_CONFIG, MOE_CONFIG           # noqa: E402
    from model.qwen3_dense import Qwen3Dense              # noqa: E402
    from model.qwen3_moe import Qwen3MoE                  # noqa: E402
    from model.common.lora import (                       # noqa: E402
        inject_lora, mark_only_lora_trainable,
    )

    def _dense(v: int):
        """造一个 dense 模型，返回 (总参数, 注入 LoRA 后总参数, LoRA 可训练参数)。

        ★ 要分清「注入前」和「注入后」：
          笔记里 dense 的参数量（如 `dense=951,808`）是**注入前**，
          而 LoRA 笔记里的 `994,560` 是**注入后**（951,808 那一档 + 8,704）。
          两个数都在笔记里出现过 ⇒ 守卫必须同时收录。
        """
        m = Qwen3Dense(vocab_size=v, cfg=DENSE_CONFIG, dropout=0.0)
        before = _num(m.parameters())
        inject_lora(m, DENSE_CONFIG.lora_target_modules,
                    r=DENSE_CONFIG.lora_r, alpha=DENSE_CONFIG.lora_alpha)
        tr = mark_only_lora_trainable(m)
        return before, _num(m.parameters()), _num(tr)

    def _moe(v: int) -> int:
        return _num(Qwen3MoE(vocab_size=v, cfg=MOE_CONFIG, dropout=0.0).parameters())

    def _moe_lora(v: int, targets: tuple):
        """MoE 注入 LoRA 后的 (总参数, 可训练参数, 专家侧可训练参数)。

        ★ feat-031 补的：**MoE 的两种 LoRA 配方**（只注 attention /
          attention + 每个专家的 FFN）也必须被守卫覆盖 ——
          否则笔记里那张三配方对照表里的「57,856 / 1,242,368」
          就是**没人核对**的数字。
        ★ 专家侧那一份（49,152）单独拿出来：它是「没被路由到的专家拿不到梯度」
          那个结论的**分母**，也得是实测值。
        """
        m = Qwen3MoE(vocab_size=v, cfg=MOE_CONFIG, dropout=0.0)
        inject_lora(m, targets, r=MOE_CONFIG.lora_r, alpha=MOE_CONFIG.lora_alpha)
        tr = mark_only_lora_trainable(m)
        exp_tr = _num(p for n, p in m.named_parameters()
                      if ".moe.experts." in n and p.requires_grad)
        return _num(m.parameters()), _num(tr), exp_tr

    d778, d778_l, lora_tr = _dense(778)
    d512, d512_l, _ = _dense(512)
    d256, d256_l, _ = _dense(256)
    moe_attn_total, moe_attn_tr, _ = _moe_lora(778, MOE_CONFIG.lora_target_modules)
    moe_exp_total, moe_exp_tr, moe_exp_side = _moe_lora(
        778, MOE_CONFIG.lora_target_modules_expert)
    swiglu_ffn = 3 * DENSE_CONFIG.n_embd * DENSE_CONFIG.intermediate_size
    gelu_ffn = 2 * g_cfg.n_embd * g_cfg.n_inner          # GPT-2 的两矩阵 FFN
    qwen3_vals = {
        d778, d512, d256,                                   # dense 总参数（注入 LoRA 前）
        d778_l, d512_l, d256_l,                            # 注入 LoRA 后的总参数
        lora_tr,                                           # LoRA 可训练参数
        swiglu_ffn,                                        # SwiGLU 单层 FFN
        _moe(778), _moe(512),                              # MoE 总参数
        moe_attn_total, moe_attn_tr,                       # MoE + 只注 attention
        moe_exp_total, moe_exp_tr,                         # MoE + attention + 专家 FFN
        moe_exp_side,                                      # 专家侧 LoRA 参数总数
    }

    # ---- 词表大小（读文件，不数行）----
    with io.open(os.path.join(PROJECT_ROOT, "data", "vocab.json"), encoding="utf-8") as f:
        q_vocab = len(json.load(f))

    return {
        "gpt2": gpt2_vals,
        "qwen3": qwen3_vals,
        "gpt2_total": g_total,
        "gpt2_vocab": g_vocab,
        "qwen3_vocab": q_vocab,
        "qwen3_dense_total": d778,          # dense（注入 LoRA 前）
        "qwen3_dense_lora_total": d778_l,   # dense（注入 LoRA 后）
        "qwen3_lora_trainable": lora_tr,
        "qwen3_moe_total": _moe(778),
        "ratio_ffn": swiglu_ffn / gelu_ffn,
        "ratio_moe": _moe(512) / d512,
    }


# ---------------------------------------------------------------------------
# 3. 笔记扫描：抽出「派生量主张」
# ---------------------------------------------------------------------------

# 说明：这次**不跳过围栏代码块** —— 笔记里贴的「训练日志输出」
# （如 `[可训练参数] 8,704 / 994,560 (0.88%)`）与「参数量拆解表」本身就是主张，必须核对。
# 代价是源码粘贴里偶尔会出现参数化数字（如 `147,456`），那也正好该被核对。

# 「参数量 = N」「params = N」「可训练总计: …」
_P_PARAM = re.compile(
    r"(?:可训练总计|参数量|总参数量|参数(?:量|总数)?|可训练参数|trainable\s*params|params)"
    r"\s*[:：=＝]?\s*([\d][\d,]*)")
# 「N 个参数」（数字在前）
_P_PARAM_PRE = re.compile(r"([\d][\d,]{2,})\s*个(?:可训练)?参数")
# 「N / M (P%)」三元组
_P_FRACTION = re.compile(r"([\d][\d,]*)\s*/\s*([\d][\d,]*)\s*\(\s*([\d.]+)\s*%")
# 词表大小（只认明确的 vocab 关键词；`V = 0.6` 这种采样示例不是词表）
# ★ 要求 ≥ 2 位：「词表大小 5」是假设句（「词表大小 5，就写成：」），不是配置主张
_P_VOCAB = re.compile(r"(?:vocab_size|vocab|词表(?:大小)?)\s*[:：=＝]?\s*(\d[\d,]{2,})")
# 「参数化」数字：带千分位或 ≥5 位。参数量都在这个量级，
#   而「§1.9」「0.8M」这种不是 ⇒ 用它把「参数上下文里的噪声」挡掉。
_P_BIG = re.compile(r"\d{1,3}(?:,\d{3})+|\d{5,}")
# 参数上下文关键词（行级）
_KW_PARAM = re.compile(r"参数量|参数(?:量|总数)?|可训练|trainable\s*params|params")
# 算式：「45 × 128 = 5,760」「256+49,152+… = 197,120」
#   ★ `(?<![\d.])` / `(?!\.)` 是为了**不切开小数**：
#     「surr2=1.2×2=2.4」若不挡，会从「2×2」开始匹配 ⇒ 误报「算错了」。
_P_EXPR = re.compile(
    r"(?<![\d.])([\d][\d,]*(?:\s*[×*+]\s*[\d][\d,]*)+)\s*=\s*([\d][\d,]*)(?!\.)")
# 算式左侧的「尾部数字表达式」（前面可能有 "MLP:" 这种标签）
_P_EXPR_LHS = re.compile(
    r"(?<![\d.])([\d][\d,]*(?:\s*[×*+]\s*[\d][\d,]*)+)\s*$")
# 链式算式中间段必须是纯数字（「= 65,536 + 65,536 =」）
_P_EXPR_MID = re.compile(r"^[\d,\s+×*]*$")
# 链式算式右端（「 5,760」/「 131,072」；后面不能紧跟小数点）
_P_EXPR_RHS = re.compile(r"\s*([\d][\d,]*)(?!\.)")
# 比值：「比值 = 1.21x」「比值 1.125」
_P_RATIO = re.compile(r"比值\s*[:：=＝]?\s*([\d.]+)")
# 参数量上下文（三元组 / 比值规则要求同一行出现）
_CTX_PARAM = re.compile(r"参数|params|可训练|trainable")

# 上下文关键词 → 归属哪条线
_KW_GPT2 = re.compile(r"GPT-?2|gpt2|GELU|CharTokenizer|vocab\.txt")
_KW_QWEN3 = re.compile(r"Qwen3|LoRA|DPO|MoE|Dense|dense|SwiGLU|QK-?Norm|GQA|RMSNorm|RoPE")

# 文件归属：gpt-learning/ 下的笔记讲 GPT-2 线，qwen3-learning/ 下讲 Qwen3 线
_FILE_BUCKET = {"gpt-learning": "gpt2", "qwen3-learning": "qwen3"}

# 数字后面跟着「字节 / bytes」⇒ 那是文件大小，不是参数量 / 词表大小。
# ★ 只把**这个数字**涂掉，不跳过整行 ——
#   「语料 20000 句 / 785065 字节 → 目标词表 1000」这一行里，
#   字节数不是派生量，但同一行的「目标词表 1000」是（登记为 TARGET 例外）。
_BYTES_SUFFIX = re.compile(r"[\d,]*\s*(?:字节|bytes|B\b)")


def _strip_commas(s: str) -> int:
    """'794,496' → 794496。"""
    return int(s.replace(",", ""))


def _eval_expr(lhs: str) -> int:
    """算式的左边（只含 + 与 ×/*）求值。"""
    total = 0
    for term in lhs.replace("*", "×").split("+"):
        prod = 1
        for factor in term.split("×"):
            prod *= _strip_commas(factor.strip())
        total += prod
    return total


def _resolve_chain(line: str):
    """把一行解析成「算式链」，返回 (左侧表达式文本, 右侧整数) 或 None。

    ★ 必须处理**链式**写法：
        「MLP: 128 × 512 + 512 × 128 = 65,536 + 65,536 = 131,072」
    只看第一个等号会得到「131,072 vs 65,536」⇒ 误报「算错了」。
    正确做法：**左边整体求值 == 最后一个等号右边的数**。

    返回 None 的情形（本守卫不管）：
      · 中间段不是纯数字（多半是代码或带小数点的公式，如 `surr2=1.2×2=2.4`）；
      · 左侧没有数字表达式（多半是 `x = 1` 这种赋值）。
    """
    if "=" not in line:
        return None
    parts = line.split("=")
    for mid in parts[1:-1]:
        if not _P_EXPR_MID.match(mid):
            return None
    m_lhs = _P_EXPR_LHS.search(parts[0])
    if not m_lhs:
        return None
    m_rhs = _P_EXPR_RHS.match(parts[-1])
    if not m_rhs:
        return None
    return m_lhs.group(1), _strip_commas(m_rhs.group(1))


def _context_bucket(line: str, pos: int) -> str:
    """看主张**前面**的一段文字，判断它说的是哪条线。

    返回 "gpt2" / "qwen3" / "any"（"any" = 看不出来 ⇒ 只能按并集核对）。
    """
    head = line[max(0, pos - 46):pos]
    g2 = _KW_GPT2.search(head)
    q3 = _KW_QWEN3.search(head)
    if g2 and not q3:
        return "gpt2"
    if q3 and not g2:
        return "qwen3"
    if g2 and q3:
        return "gpt2" if g2.start() > q3.start() else "qwen3"
    return "any"


def _scan_notes():
    """扫全部笔记，返回 (笔记列表, 主张列表, 参数上下文行列表)。"""
    notes = []
    for d in ("gpt-learning", "qwen3-learning"):
        base = os.path.join(REPO_ROOT, d)
        for name in sorted(os.listdir(base)):
            if name.endswith(".md"):
                rel = f"{d}/{name}"
                if rel in MANUAL_AUDIT_DOCS:
                    continue
                notes.append(rel)

    out = []
    param_lines = []
    for rel in notes:
        with io.open(os.path.join(REPO_ROOT, rel), encoding="utf-8") as f:
            for lineno, raw_line in enumerate(f.read().split("\n"), 1):
                # 把「785065 字节」这种文件大小涂掉，**其余内容照常核对**
                line = _BYTES_SUFFIX.sub(" ", raw_line)
                for m in _P_EXPR.finditer(line):
                    out.append((rel, lineno, "expr", m.group(0), m, line.strip()))
                for m in _P_FRACTION.finditer(line):
                    out.append((rel, lineno, "fraction", m.group(0), m, line.strip()))
                for m in _P_RATIO.finditer(line):
                    out.append((rel, lineno, "ratio", m.group(0), m, line.strip()))
                for m in _P_PARAM.finditer(line):
                    out.append((rel, lineno, "param", m.group(0), m, line.strip()))
                for m in _P_PARAM_PRE.finditer(line):
                    out.append((rel, lineno, "param", m.group(0), m, line.strip()))
                for m in _P_VOCAB.finditer(line):
                    out.append((rel, lineno, "vocab", m.group(0), m, line.strip()))
                if _KW_PARAM.search(line):
                    param_lines.append((rel, lineno, line))
    return notes, out, param_lines


# ---------------------------------------------------------------------------
# 4. 例外登记簿：note 相对路径 + 数值 → 为什么它不是「代码实测值」
# ---------------------------------------------------------------------------
# 分类：
#   SMOKE      单元 smoke / 消融用的独立小尺寸模型（不是主线模型）
#   UPSTREAM   官方模型（GPT-2 124M）或另一份实现（TF 端）
#   TARGET     目标值（不是实测值）
#   TEACHING   教学举例用的玩具值
EXCEPTIONS = {
    ("gpt-learning/0.GPT全流程详解.md", 800256):
        "UPSTREAM: TF 端实现的参数量（embedding transpose 差异），本仓只跑 PyTorch",
    ("gpt-learning/1.bpe-tokenizer.md", 50257):
        "UPSTREAM: 官方 GPT-2 的 vocab_size（50257），本仓是字符级 tiny 版 45",
    ("gpt-learning/4.weight-tying.md", 50257):
        "UPSTREAM: 官方 GPT-2 的 vocab_size（50257），本仓是字符级 tiny 版 45",
    ("qwen3-learning/Qwen3_BPE_学习笔记.md", 1000):
        "TARGET: **目标**词表大小（1000），不是实测值（实测 778）",
    ("qwen3-learning/Qwen3_BPE_学习笔记.md", 293):
        "SMOKE: `tests/test_bpe.py` 第 3 组用小语料训出的词表（vocab=293）",
    ("qwen3-learning/Qwen3_LoRA与DPO_学习笔记.md", 256):
        "SMOKE: 单测里的小词表模型（vocab=256）的总参数量",
    ("qwen3-learning/Qwen3_MoE_学习笔记.md", 512):
        "SMOKE: `tests/test_moe.py` 的小词表模型（vocab=512）",
    ("qwen3-learning/Qwen3_训练推理完整学习笔记.md", 293):
        "SMOKE: feat-017 对照实验用的小词表（vocab=293）",
    ("qwen3-learning/Qwen3_LoRA与DPO_学习笔记.md", 6144):
        "RUNTIME: feat-031 的逐专家诊断里「这一步 `grad is None` 的 LoRA 参数个数」—— "
        "它**随这一步的路由结果浮动**（路由器冻结 ⇒ 哪些专家被选中会变），"
        "不是由 config 决定的静态参数量 ⇒ 不进实测集合。"
        "同段落的分母 49,152（专家侧 LoRA 参数总数）是静态的，已由代码实测覆盖。",
}


# ---------------------------------------------------------------------------
# 5. 主检查
# ---------------------------------------------------------------------------


def main() -> int:
    gt = _ground_truth()
    gpt2_vals, qwen3_vals = gt["gpt2"], gt["qwen3"]
    union = gpt2_vals | qwen3_vals
    vocab_vals = {gt["gpt2_vocab"], gt["qwen3_vocab"]}

    notes, claims, param_lines = _scan_notes()
    problems = []
    warnings = []
    checked = 0
    used_exc = set()

    def _exc(rel: str, value: int) -> bool:
        if (rel, value) in EXCEPTIONS:
            used_exc.add((rel, value))
            return True
        return False

    for rel, lineno, kind, raw, m, line in claims:
        # ---- R1 算式自洽 ----
        if kind == "expr":
            chain = _resolve_chain(line)
            if chain is None:
                continue
            lhs, want = chain
            try:
                got = _eval_expr(lhs)
            except ValueError:
                continue
            checked += 1
            if got != want:
                problems.append((rel, lineno, kind, raw,
                                 f"算式算错了：左边 = {got:,}，右边写的是 {want:,}"))
            continue

        # ---- R4 三元组 ----
        if kind == "fraction":
            if not _CTX_PARAM.search(line):
                continue          # 不是参数量比例（如「已完成 0 / 17 (0%)」）
            checked += 1
            num, den = _strip_commas(m.group(1)), _strip_commas(m.group(2))
            pct = float(m.group(3))
            if num not in union or den not in union:
                problems.append((rel, lineno, kind, raw,
                                 f"分子 {num:,} / 分母 {den:,} 不是代码实测出来的参数量"))
            elif den and abs(num / den * 100.0 - pct) > 0.011:
                problems.append((rel, lineno, kind, raw,
                                 f"百分比 {pct}% 与 {num:,} / {den:,} "
                                 f"= {num / den * 100:.2f}% 对不上"))
            continue

        # ---- R5 比值 ----
        if kind == "ratio":
            nums = [_strip_commas(x) for x in
                    re.findall(r"\b\d{1,3}(?:,\d{3})+\b", line)]
            if len(nums) != 2:
                continue
            a, b = max(nums), min(nums)
            if b == 0:
                continue
            claimed = float(m.group(1))
            checked += 1
            if abs(a / b - claimed) > 0.006:
                problems.append((rel, lineno, kind, raw,
                                 f"比值 {claimed} 与 {a:,} / {b:,} = {a / b:.4f} 对不上"))
            continue

        # ---- R2 参数量 ----
        if kind == "param":
            val = _strip_commas(m.group(1))
            if val in union:
                continue
            if _exc(rel, val):
                continue
            bucket = _context_bucket(line, m.start())
            problems.append((rel, lineno, kind, raw,
                             f"参数量 {val:,} 在代码实测里找不到"
                             f"（归属 {bucket}；GPT-2 侧 {sorted(gpt2_vals)} / "
                             f"Qwen3 侧 {sorted(qwen3_vals)}）"))
            continue

        # ---- R3 词表大小 ----
        if kind == "vocab":
            val = _strip_commas(m.group(1))
            if val in vocab_vals:
                continue
            if _exc(rel, val):
                continue
            problems.append((rel, lineno, kind, raw,
                             f"词表大小 {val} 与实测不符"
                             f"（GPT-2 = {gt['gpt2_vocab']} / Qwen3 = {gt['qwen3_vocab']}）"))

    # ---- R2' 行级：参数上下文行上的**每一个**参数化数字都要对得上 ----
    # 为什么要行级：R2 的正则要求「关键词紧邻数字」，
    #   而笔记里最常见的写法是**表格 / 拆解表**：
    #       | 总可训练参数 | 见 §1.9 | **794,496** |
    #   关键词和数字被 `|` 隔开 ⇒ R2 抓不到 ⇒ 数字实际上**没被核对**。
    #   行级规则把它补上：**这一行只要出现参数关键词，
    #   上面每一个「参数化数字」（带千分位或 ≥5 位）都必须能对上是实测值。**
    for rel, lineno, line in param_lines:
        file_bucket = _FILE_BUCKET.get(rel.split("/")[0], "any")
        for m in _P_BIG.finditer(line):
            val = _strip_commas(m.group(0))
            checked += 1
            line_bucket = _context_bucket(line, m.start())
            bucket = line_bucket if line_bucket != "any" else file_bucket
            if bucket == "any":
                allowed = union
            else:
                allowed = gpt2_vals if bucket == "gpt2" else qwen3_vals
            if val in allowed:
                continue
            if val in union:
                # 数字本身是实测值，只是**归属**判错了（跨线对照行）——
                # 报出来当提示，但不让它红（归属判错不该拦住 CI）。
                warnings.append(
                    (rel, lineno, m.group(0),
                     f"归属判成 {bucket}，但它其实是另一条线的实测值（"
                     f"GPT-2 侧 {sorted(gpt2_vals)} / Qwen3 侧 {sorted(qwen3_vals)}）"))
                continue
            if _exc(rel, val):
                continue
            problems.append((rel, lineno, "param-line", m.group(0),
                             f"参数量 {val:,} 在代码实测里找不到"
                             f"（归属 {bucket}）"))

    # ---- R6 可训练总计必须等于模型实测总参数 ----
    for rel, lineno, kind, raw, m, line in claims:
        if kind != "param" or "可训练总计" not in line:
            continue
        tail = line.rsplit("=", 1)
        if len(tail) != 2:
            continue
        nums = re.findall(r"[\d][\d,]*", tail[1])
        if not nums:
            continue
        checked += 1
        total = _strip_commas(nums[0])
        if total != gt["gpt2_total"]:
            problems.append((rel, lineno, "total", raw,
                             f"「可训练总计」写的是 {total:,}，"
                             f"模型实测是 {gt['gpt2_total']:,}"))

    if problems:
        print("[derived] 发现与代码实测不一致的派生量：\n")
        for rel, lineno, kind, raw, why in problems:
            print(f"  {rel}:{lineno}  «{raw}»\n      {why}")
        print("\n处理办法（三选一）：")
        print("  a) 改笔记，让它跟代码实测一致；")
        print("  b) 改代码/配置（那样期望值会跟着变，笔记也要一起改）；")
        print("  c) 这确实不是本仓库的实测值 ⇒ 在 EXCEPTIONS 里登记理由。")
        print(f"\ntest_derived_sync FAIL  不一致 {len(problems)} 处")
        return 1

    unused = sorted(set(EXCEPTIONS) - used_exc)
    if unused:
        print("[derived] ⚠ 以下例外已不再命中（笔记被改过了），请删除这些登记：")
        for rel, val in unused:
            print(f"  ({rel!r}, {val})")
        print()

    if warnings:
        print("[derived] ⚠ 归属提示（不拦 CI）：")
        for rel, lineno, raw, why in warnings:
            print(f"  {rel}:{lineno}  «{raw}»  {why}")
        print()

    print(f"[derived] 真值：GPT-2 总参数 {gt['gpt2_total']:,}"
          f"（vocab {gt['gpt2_vocab']}）"
          f" / Qwen3 dense {gt['qwen3_dense_total']:,}"
          f"（vocab {gt['qwen3_vocab']}，注入 LoRA 后 {gt['qwen3_dense_lora_total']:,}"
          f"，MoE {gt['qwen3_moe_total']:,}）")
    print(f"[derived] 核对笔记 {len(notes)} 篇；派生量主张 {checked} 处；"
          f"例外登记 {len(EXCEPTIONS)} 条（命中 {len(used_exc)}，过时 {len(unused)}）")
    print(f"[derived] 人工核对文档（不进自动守卫）：{', '.join(MANUAL_AUDIT_DOCS)}")
    print("test_derived_sync OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

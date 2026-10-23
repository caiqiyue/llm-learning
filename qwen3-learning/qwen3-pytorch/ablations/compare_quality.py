"""
feat-026：Qwen3 与 GPT-2 的**生成质量并排对照**。

【目的】
    把「两条线的生成质量能不能比」从一句文字断言变成**一次可跑的实验**。
    feat-025 收班时，Qwen3 线的 `diversity` / `response_quality` 两项还是
    「⏳ 未实现」—— 本脚本是那半块拼图。

【三条铁律（缺一，这个对照就没有意义）】
    1. **同一批 prompt**：用 Qwen3 线的 `data/quality_prompts.py` 生成的
       动物探针，两条线喂**完全一样**的问题（各自的 prompt 包装按各自的约定：
       GPT-2 用 `<bos> {prompt} <sep> `，Qwen3 用 `### Question:\\n...\\n\\n### Answer:\\n`）。
    2. **同一份指标代码**：4 个指标一律走 `qwen3-pytorch/quality.py`，
       **不允许「两边各写一份」**（那只是「看起来一样」，不是「同一个指标」）。
    3. **同一把尺子要有证据**：用 **GPT-2 线自己的** `evaluate.py` 在
       **同一批固定回答**上重算一遍，值必须与共享实现**完全相等** ——
       不相等就直接报错退出。这是「同名同口径」的**机器证明**，不是承诺。

【两对模型，各有各的意义】
    · **微调对**：GPT-2 的 `sft.pt`  vs  Qwen3 的 `lora.pt`
      —— 都是「指令微调之后」的模型，是可比的两种模型。
    · **底模对**：GPT-2 的 `pretrain.pt`  vs  Qwen3 的 `pretrain_dense.pt`
      —— 都是「还没微调的语言模型」。加这一对是因为**实测发现 GPT-2 的
      `sft.pt` / `ppo.pt` 会退化成常量输出**（见下面的「实测发现」），
      只比微调对的话，量到的是「谁塌得更狠」，不是「谁生成得更好」。

【★ 实测发现（2026-10-23，脚本跑出来的，不是猜的）】
    GPT-2 线的 `sft.pt` / `ppo.pt` 对**任何** prompt 都输出同一串常量
    （`'? <unk>sep<unk>'` + 一堆空格，48 字符），而 `pretrain.pt` 的输出是**有变化的**。
    ⇒ P1 线的 SFT 之后模型**塌成了常量输出**。这与 handoff 里记着的
      「PPO 相对 SFT 无提升、rm_score=0.0696」是同一个现象。
      **本 feature 不修它**（修它要改 P1 训练代码 + 跑 L3 全量），
      只在这里如实报告，并把结论写成**由数据算出**的。

【★ 表 1 该怎么读（不要误读）】
    · 探针对 **GPT-2 是域内**（它的预训练语料就是动物童话），
      对 **Qwen3 是域外**（Qwen3 训在 transformer 主题的合成语料上）
      ⇒ 表 1 **不是**「哪个架构更好」的证据，只是「同一把尺子下的绝对水平」。
    · `response_quality` 里的**长度规则（10~50 字符）**是按**字符级 tokenizer**
      标定的；Qwen3 用 BPE，回答普遍更长 ⇒ 这一条规则对 Qwen3 系统性不利。
      所以表里同时给 **`response_quality_ex_length`**（扣掉长度规则）
      与**每条规则的命中率** —— 偏差有多大，看得见。
    · `token_entropy` 的**绝对值依赖词表大小**（均匀分布时 H = log V），
      跨线不可比；可比的是 `token_entropy_normalized` = H / log(V)。

【怎么跑】
    cd qwen3-learning/qwen3-pytorch
    python ablations/compare_quality.py
    python ablations/compare_quality.py --n 30 --max-new-tokens 32
    python ablations/compare_quality.py --skip-gpt2     # 只有 Qwen3 侧时
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

import torch

# --- 路径引导 ---------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# GPT-2 线在 llm-learning/gpt-learning/gpt-learning-pytorch/ 下。
# ⚠️ 目录层数：ablations → qwen3-pytorch → qwen3-learning → llm-learning
REPO_ROOT = os.path.dirname(os.path.dirname(PROJECT_ROOT))
GPT2_DIR = os.path.join(REPO_ROOT, "gpt-learning", "gpt-learning-pytorch")

import evaluate  # noqa: E402  （Qwen3 线的评估脚本：模型加载等工具）
import quality  # noqa: E402  （共享指标实现 —— 两条线同一份代码）
from config import DENSE_CONFIG  # noqa: E402
from data.quality_prompts import (  # noqa: E402
    QUALITY_ENTITIES,
    get_corpus_prompts,
    get_quality_prompts,
)
from data.tokenizer import BPETokenizer  # noqa: E402

# 交叉验证允许的误差（两边的浮点运算顺序不同 ⇒ 不要求 bit 级相等；
# 但实测是**完全相等**（差值 0.000e+00），这个容差只是兜底）
CROSS_TOL = 1e-9

# 两对模型。`with_lora` 决定 Qwen3 侧加载 checkpoint 时要不要先注入 LoRA 分支
PAIRS: List[dict] = [
    {
        "name": "微调对（GPT-2 SFT  vs  Qwen3 LoRA）",
        "gpt2": os.path.join(GPT2_DIR, "checkpoints", "sft.pt"),
        "qwen3": os.path.join(PROJECT_ROOT, "checkpoints", "lora.pt"),
        "with_lora": True,
    },
    {
        "name": "底模对（GPT-2 预训练  vs  Qwen3 预训练）",
        "gpt2": os.path.join(GPT2_DIR, "checkpoints", "pretrain.pt"),
        "qwen3": os.path.join(PROJECT_ROOT, "checkpoints", "pretrain_dense.pt"),
        "with_lora": False,
    },
]


# ---------------------------------------------------------------------------
# Qwen3 侧
# ---------------------------------------------------------------------------
def run_qwen3(
    ckpt: str,
    tk: BPETokenizer,
    prompts: List[str],
    max_new_tokens: int,
    device: torch.device,
    with_lora: bool = True,
    seed: int = 42,
) -> Dict[str, object]:
    """
    用 Qwen3 模型跑一遍生成质量。

    参数：
        ckpt:            checkpoint 路径
        tk:              tokenizer
        prompts:         探针
        max_new_tokens:  每条最多生成多少 token
        device:          设备
        with_lora:       True ⇒ checkpoint 里带 LoRA 分支（先注入再加载）
        seed:            随机种子（生成是**采样**，不固定种子数字每次都在动）
    返回：
        {"responses": [...], "metrics": {...}}
    """
    cfg = DENSE_CONFIG
    # ★ 固定种子 ⇒ 可复现（否则两次跑的数字对不上，「对照」就无从谈起）
    torch.manual_seed(seed)
    model = evaluate._load_dense(ckpt, tk, device, with_lora=with_lora)
    responses, logits_groups = quality.generate_responses_with_logits(
        model, tk, prompts,
        max_new_tokens=max_new_tokens,
        temperature=cfg.temperature, top_k=cfg.top_k, top_p=cfg.top_p,
    )
    rq = quality.response_quality_detail(prompts, responses)
    ent = quality.token_entropy_from_logits(logits_groups, temperature=cfg.temperature)
    metrics = {
        "distinct_1": quality.distinct_n(responses, 1),
        "distinct_2": quality.distinct_n(responses, 2),
        "self_bleu": quality.self_bleu(responses),
        "token_entropy": ent,
        "token_entropy_normalized": ent / math.log(tk.vocab_size),
        "response_quality": rq["value"],
        "response_quality_ex_length": rq["value_ex_length"],
        "response_quality_by_rule": rq["by_rule"],
        "mean_response_chars": rq["mean_chars"],
        "n_distinct_responses": len(set(responses)),
    }
    return {"responses": responses, "metrics": metrics}


# ---------------------------------------------------------------------------
# GPT-2 侧（子进程）
# ---------------------------------------------------------------------------
def run_gpt2(
    ckpt: str,
    prompts: List[str],
    max_new_tokens: int,
    n_entropy: int,
    temperature: float,
    seed: int = 42,
) -> Dict[str, object]:
    """
    在 **GPT-2 自己的进程**里跑一遍生成质量（子进程，见 runner 的 docstring）。

    参数：
        ckpt:            GPT-2 线的 checkpoint 路径
        prompts:         探针（与 Qwen3 侧**完全一样**）
        max_new_tokens:  每条最多生成多少 token
        n_entropy:       算熵时用前多少条 prompt
        temperature:     采样温度
        seed:            随机种子（与 Qwen3 侧用同一个）
    返回：
        runner 写出来的 JSON（responses / shared / own / gpt2_entities）
    """
    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "_gpt2_quality_runner.py")
    tmpdir = tempfile.mkdtemp(prefix="gpt2_quality_")
    payload_path = os.path.join(tmpdir, "payload.json")
    out_path = os.path.join(tmpdir, "out.json")
    with open(payload_path, "w", encoding="utf-8") as f:
        json.dump({
            "prompts": prompts,
            "checkpoint": os.path.abspath(ckpt),
            "max_new_tokens": max_new_tokens,
            "n_entropy_samples": n_entropy,
            "temperature": temperature,
            "seed": seed,
        }, f, ensure_ascii=False)

    env = dict(os.environ)
    # ★ PYTHONPATH 指向 GPT-2 线 ⇒ 那个进程里的 `data` / `model` / `evaluate`
    #   全部解析到 GPT-2 的版本（两个线的包名同名，只能这样隔离）
    env["PYTHONPATH"] = GPT2_DIR + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, runner, "--payload", payload_path, "--out", out_path],
        cwd=GPT2_DIR, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "GPT-2 侧的跑分进程失败（退出码 %d）：\n%s\n%s"
            % (proc.returncode, proc.stdout[-2000:], proc.stderr[-2000:])
        )
    with open(out_path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 交叉验证：GPT-2 自己的 evaluate.py  vs  共享实现
# ---------------------------------------------------------------------------
def cross_validate(gpt2: Dict[str, object]) -> List[dict]:
    """
    逐项比对「GPT-2 自己的 evaluate.py」与「共享实现」在**同一批固定回答**上的值。

    参数：
        gpt2: runner 的输出
    返回：
        [{"metric": 名, "gpt2_own": 值, "shared": 值, "diff": 差值}, ...]
    """
    own = gpt2["own"]
    shared = gpt2["shared"]
    rows = []
    for name in ("distinct_1", "distinct_2", "self_bleu",
                 "response_quality", "token_entropy"):
        a = float(own[name])
        b = float(shared[name])
        rows.append({"metric": name, "gpt2_own": a, "shared": b, "diff": abs(a - b)})
    return rows


def check_entities(gpt2: Dict[str, object]) -> None:
    """
    探针集的实体表必须与 GPT-2 线的 `ENTITIES` 一致（否则 response quality
    的「含动物名」规则量的不是同一件事）。

    参数：
        gpt2: runner 的输出
    """
    gpt2_entities = gpt2["gpt2_entities"]
    for key in ("animal", "action"):
        if list(gpt2_entities[key]) != list(QUALITY_ENTITIES[key]):
            raise AssertionError(
                f"探针集的 `{key}` 与 GPT-2 线的 ENTITIES 不一致 —— "
                f"response quality 就不是同一个指标了。\n"
                f"  GPT-2 : {gpt2_entities[key]}\n"
                f"  探针集: {QUALITY_ENTITIES[key]}"
            )


# ---------------------------------------------------------------------------
# 打印
# ---------------------------------------------------------------------------
def _fmt(v: object) -> str:
    """把一个指标值格式化成固定宽度的字符串。"""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def print_table(title: str, header: List[str], rows: List[List[str]]) -> None:
    """
    打印一张简单的对齐表。

    参数：
        title:  表标题
        header: 表头
        rows:   数据行
    """
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
              for i in range(len(header))]
    print("  ".join(header[i].ljust(widths[i]) for i in range(len(header))))
    print("  ".join("-" * widths[i] for i in range(len(header))))
    for r in rows:
        print("  ".join(r[i].ljust(widths[i]) for i in range(len(header))))


def _metric_rows(gm: Dict[str, object], qm: Dict[str, object]) -> List[List[str]]:
    """把两边的指标拼成表格行。"""
    return [
        ["distinct_1", _fmt(gm["distinct_1"]), _fmt(qm["distinct_1"])],
        ["distinct_2", _fmt(gm["distinct_2"]), _fmt(qm["distinct_2"])],
        ["self_bleu", _fmt(gm["self_bleu"]), _fmt(qm["self_bleu"])],
        ["token_entropy", _fmt(gm["token_entropy"]), _fmt(qm["token_entropy"])],
        ["token_entropy_normalized",
         _fmt(gm["token_entropy_normalized"]), _fmt(qm["token_entropy_normalized"])],
        ["response_quality", _fmt(gm["response_quality"]), _fmt(qm["response_quality"])],
        ["response_quality_ex_length",
         _fmt(gm["response_quality_ex_length"]), _fmt(qm["response_quality_ex_length"])],
        ["mean_response_chars",
         _fmt(gm["mean_response_chars"]), _fmt(qm["mean_response_chars"])],
    ]


def _common_suffix_len(responses: List[str]) -> int:
    """
    一批回答的**最长公共后缀**长度。

    【为什么用它判「退化」】
        「不同回答的数量」这个指标**会被骗**：GPT-2 的 `sft.pt` 会把 prompt 的
        最后一个字符（`?` / `.`）一起吐出来，于是 30 条不同问题得到
        30 条**看起来不同**的回答，其实内容完全一样。
        公共后缀长度不受这种「首字符抖动」影响 ⇒ 更能看出「内容是不是常量」。

    参数：
        responses: 一批回答
    返回：
        最长公共后缀的字符数
    """
    if not responses:
        return 0
    suffix = responses[0]
    for r in responses[1:]:
        k = 0
        while (k < len(suffix) and k < len(r)
               and suffix[len(suffix) - 1 - k] == r[len(r) - 1 - k]):
            k += 1
        suffix = suffix[len(suffix) - k:] if k else ""
        if not suffix:
            return 0
    return len(suffix)


def _degeneracy_report(responses: List[str]) -> Tuple[str, float]:
    """
    判断这批回答是不是「退化成常量输出」。

    参数：
        responses: 一批回答
    返回：
        (结论文字, 公共后缀占平均长度的比例)
    """
    if not responses:
        return "（没有回答）", 0.0
    n_suffix = _common_suffix_len(responses)
    mean_chars = sum(len(r) for r in responses) / len(responses)
    ratio = n_suffix / max(mean_chars, 1e-9)
    if ratio >= 0.5:
        return (f"退化成常量输出（公共后缀 {n_suffix} 字符 / 平均 {mean_chars:.1f} 字符 "
                f"= {ratio:.0%}）", ratio)
    return (f"正常（公共后缀 {n_suffix} 字符 / 平均 {mean_chars:.1f} 字符 = {ratio:.0%}）", ratio)


def _rule_rows(gm: Dict[str, object], qm: Dict[str, object]) -> List[List[str]]:
    """把两边的规则命中率拼成表格行。"""
    gr = gm["response_quality_by_rule"]
    qr = qm["response_quality_by_rule"]
    return [[k, _fmt(gr[k]), _fmt(qr[k])] for k in ("length", "entity", "verb", "ending")]


def _higher_lower(a: float, b: float, hi_is_good: bool = True) -> str:
    """
    比两个数，返回一句「谁更高」的话（**结论由数据算出**，不写死）。

    参数：
        a:        左边的数
        b:        右边的数
        hi_is_good: 大的一方是不是好的（self_bleu 是越低越好）
    返回：
        一句话
    """
    if abs(a - b) < 1e-12:
        return "持平"
    higher = "左边" if a > b else "右边"
    lower = "右边" if a > b else "左边"
    good = higher if hi_is_good else lower
    return f"{higher}更高（{good}更好）"


def main() -> int:
    """并排对照的主流程。"""
    ap = argparse.ArgumentParser(description="Qwen3 ↔ GPT-2 生成质量并排对照")
    ap.add_argument("--n", type=int, default=DENSE_CONFIG.quality_n_samples,
                    help="探针条数（两条线一样多）")
    ap.add_argument("--max-new-tokens", type=int,
                    default=DENSE_CONFIG.quality_max_new_tokens,
                    help="每条最多生成多少 token（两条线一样多）")
    ap.add_argument("--skip-gpt2", action="store_true",
                    help="只跑 Qwen3 侧（没有 GPT-2 checkpoint 时用）")
    ap.add_argument("--pairs", default="both", choices=["both", "tuned", "base"],
                    help="比哪一对模型：tuned=微调对 / base=底模对 / both=两对都比")
    ap.add_argument("--seed", type=int, default=DENSE_CONFIG.eval_seed,
                    help="随机种子（生成是采样 ⇒ 不固定种子数字每次都在动）")
    args = ap.parse_args()

    device = torch.device("cpu")
    tk = BPETokenizer(os.path.join(PROJECT_ROOT, "data", "vocab.json"),
                      os.path.join(PROJECT_ROOT, "data", "merges.txt"))
    probes = get_quality_prompts(args.n, DENSE_CONFIG.eval_seed)
    n_entropy = min(20, len(probes))

    if args.pairs == "tuned":
        pairs = PAIRS[:1]
    elif args.pairs == "base":
        pairs = PAIRS[1:]
    else:
        pairs = PAIRS

    qwen3_results: List[Tuple[str, dict]] = []
    gpt2_results: List[Tuple[str, dict]] = []
    for pair in pairs:
        name = pair["name"]
        print(f"[quality] {name}")
        if not os.path.exists(pair["qwen3"]):
            print(f"  跳过：找不到 Qwen3 checkpoint {pair['qwen3']}")
            continue
        q = run_qwen3(pair["qwen3"], tk, probes, args.max_new_tokens, device,
                      with_lora=pair["with_lora"], seed=args.seed)
        qwen3_results.append((name, q))
        print(f"  Qwen3 侧跑完（ckpt={pair['qwen3']}）")

        if args.skip_gpt2:
            continue
        if not os.path.exists(pair["gpt2"]):
            print(f"  跳过 GPT-2 侧：找不到 {pair['gpt2']}")
            continue
        g = run_gpt2(pair["gpt2"], probes, args.max_new_tokens, n_entropy,
                     DENSE_CONFIG.temperature, seed=args.seed)
        gpt2_results.append((name, g))
        print(f"  GPT-2 侧跑完（ckpt={pair['gpt2']}，子进程）")

    # ---------------- 表 1：共同探针并排 ----------------
    if gpt2_results:
        for (name, g), (_, q) in zip(gpt2_results, qwen3_results):
            gm, qm = g["shared"], q["metrics"]
            print_table(
                f"表 1  共同探针（动物类问题；GPT-2 域内 / Qwen3 域外）—— {name}",
                ["指标", "GPT-2", "Qwen3"], _metric_rows(gm, qm),
            )
            print_table(
                f"表 1b 规则命中率 —— {name}",
                ["规则", "GPT-2", "Qwen3"], _rule_rows(gm, qm),
            )
            print("  ⚠️ 这不是「架构谁更好」的证据：探针对 GPT-2 是域内、对 Qwen3 是域外。")
            print("  ⚠️ token_entropy 绝对值依赖词表大小（GPT-2 ~64 字符 vs Qwen3 778 子词），跨线不可比。")

            # 退化检测：内容是不是常量？（公共后缀法，比「不同回答数」靠谱）
            g_deg, _ = _degeneracy_report(g["responses"])
            q_deg, _ = _degeneracy_report(q["responses"])
            print(f"  · GPT-2 这一侧：{g_deg}")
            print(f"  · Qwen3 这一侧：{q_deg}")
            if "退化" in g_deg:
                print("  · ⚠️ GPT-2 这一侧退化成常量输出 —— 这是 **P1 线既有状况**"
                      "（sft.pt / ppo.pt 都一样），不是本脚本造成的；"
                      "修它要改 P1 训练代码并跑 L3 全量，本 feature 不碰。")

            # 结论由数据算出
            for key, hi_is_good in (("distinct_1", True), ("distinct_2", True),
                                    ("self_bleu", False)):
                print(f"  · {key}: {_higher_lower(gm[key], qm[key], hi_is_good)}。")

        # ---------------- 表 2：口径交叉验证 ----------------
        for name, g in gpt2_results:
            cv = cross_validate(g)
            print_table(
                f"表 2  口径交叉验证（GPT-2 自己的 evaluate.py  vs  共享实现 quality.py）—— {name}",
                ["指标", "GPT-2 自己算", "共享实现算", "差值"], [
                    [r["metric"], _fmt(r["gpt2_own"]), _fmt(r["shared"]), f"{r['diff']:.3e}"]
                    for r in cv
                ],
            )
            bad = [r for r in cv if r["diff"] > CROSS_TOL]
            if bad:
                print("\n❌ 口径交叉验证失败：以下指标两边的值不相等 —— "
                      "「同名同口径」不成立，报告里的并排数字不能信：")
                for r in bad:
                    print(f"   {r['metric']}: GPT-2 自己算 {r['gpt2_own']:.10f} vs "
                          f"共享实现 {r['shared']:.10f}（差 {r['diff']:.3e}）")
                return 1
            print(f"  ✅ {len(cv)} 个指标全部逐值相等（容差 {CROSS_TOL:g}）⇒ 两条线量的是同一件事。")
            check_entities(g)
            print("  ✅ 探针集实体表与 GPT-2 线的 ENTITIES 一致（animal / action）。")

    # ---------------- 表 3：Qwen3 域内探针 ----------------
    corpus_prompts = get_corpus_prompts(args.n, DENSE_CONFIG.eval_seed)
    for (name, q), pair in zip(qwen3_results, pairs):
        ckpt = pair["qwen3"]
        with_lora = pair["with_lora"]
        corpus = run_qwen3(ckpt, tk, corpus_prompts, args.max_new_tokens, device,
                           with_lora=with_lora, seed=args.seed)
        cm = corpus["metrics"]
        qm = q["metrics"]
        rows = [
            ["distinct_1", _fmt(qm["distinct_1"]), _fmt(cm["distinct_1"])],
            ["distinct_2", _fmt(qm["distinct_2"]), _fmt(cm["distinct_2"])],
            ["self_bleu", _fmt(qm["self_bleu"]), _fmt(cm["self_bleu"])],
            ["mean_response_chars", _fmt(qm["mean_response_chars"]), _fmt(cm["mean_response_chars"])],
        ]
        print_table(
            f"表 3  Qwen3 自己：域外探针（动物）  vs  域内探针（自己的语料）—— {name}",
            ["指标", "域外（动物）", "域内（语料）"], rows,
        )
        # ★ 结论由**数据**算出，不写死
        d_out, d_in = qm["distinct_1"], cm["distinct_1"]
        s_out, s_in = qm["self_bleu"], cm["self_bleu"]
        print(f"  · 多样性（distinct_1）：域外 {d_out:.4f} vs 域内 {d_in:.4f} —— "
              f"{'域内更高' if d_in > d_out else '域外更高' if d_out > d_in else '持平'}。")
        print(f"  · 重复度（self_bleu，越低越多样）：域外 {s_out:.4f} vs 域内 {s_in:.4f} —— "
              f"{'域内更少重复' if s_in < s_out else '域外更少重复' if s_out < s_in else '持平'}。")
        print("  · ⇒ 域内/域外探针的**多样性**差异（见上两行）说明分布差异有多大；"
              "「含动物名 / 含动词」这类规则命中率（表 1b）则是分布差异最直接的体现。")

    # ---------------- 样例 ----------------
    if gpt2_results:
        print()
        print("=" * 78)
        print("样例回答（前 3 条；两条线面对的是同一批问题）")
        print("=" * 78)
        _, g = gpt2_results[0]
        _, q = qwen3_results[0]
        for i in range(min(3, len(probes))):
            print(f"[Q] {probes[i]}")
            print(f"[GPT-2] {g['responses'][i]!r}")
            print(f"[Qwen3] {q['responses'][i]!r}")
            print("-" * 40)
    return 0


if __name__ == "__main__":
    sys.exit(main())

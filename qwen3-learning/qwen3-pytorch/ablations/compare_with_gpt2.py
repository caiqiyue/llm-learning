"""
feat-017：Qwen3 与 GPT-2 的结构等价性对照实验。

【目的】
    把「Qwen3 相对 GPT-2 改了什么」从**文字断言**变成**一次可跑的实验**。
    做法：把 Qwen3 的 5 个组件开关全部关掉，退化成与 GPT-2 相同的结构，
    再把 GPT-2 的权重逐层映射进退化的 Qwen3，比对同输入下的 logits。

【为什么这个对照有意义】
    7 篇 Qwen3 笔记里满是「Qwen3 用了 RoPE / GQA / QK-Norm / SwiGLU / RMSNorm」
    这样的陈述，但**没有一个数字**能说明「这些改动到底改变了什么」。
    本脚本给出的就是那个数字：
        · 全关 ⇒ 与 GPT-2 的输出差异 < 1e-4（说明退化是**真的**退化成 GPT-2）
        · 只开某一个 ⇒ 差异 > 1e-4（说明那个组件**确实**改变了模型行为）
    两组数字合起来才是完整的论证。

【参数名映射表（GPT-2 → 退化的 Qwen3）】
    ---------------------------------------------------------------------------
    GPT-2                                   →  Qwen3（退化后）
    ---------------------------------------------------------------------------
    tok_emb.weight                          →  wte.weight
    pos_emb（buffer，不在 state_dict 里）    →  wpe.weight      [仅 use_rope=False]
    blocks.{i}.ln_1.weight / .bias          →  blocks.{i}.ln_1.weight / .bias
                                             （norm_type='layernorm' 时同名同义；
                                               norm_type='rmsnorm' 时 → .gain，且丢弃 bias）
    blocks.{i}.attn.c_attn.weight[0:C]      →  blocks.{i}.attn.q_proj.weight
    blocks.{i}.attn.c_attn.weight[C:2C]     →  blocks.{i}.attn.k_proj.weight
    blocks.{i}.attn.c_attn.weight[2C:3C]    →  blocks.{i}.attn.v_proj.weight
        （GPT-2 用**一个** c_attn 一次投出 Q/K/V；Qwen3 因 head_dim 与 n_embd 解耦，
          必须拆成三个投影。GQA 打开时 K/V 只取前 n_head_kv·head_dim 行。）
    blocks.{i}.attn.c_proj.weight           →  blocks.{i}.attn.o_proj.weight
    blocks.{i}.ln_2.weight / .bias          →  blocks.{i}.ln_2.weight / .bias
    blocks.{i}.mlp.c_fc.weight               →  blocks.{i}.mlp.up_proj.weight   [mlp_type='gelu']
    blocks.{i}.mlp.c_proj.weight             →  blocks.{i}.mlp.down_proj.weight [mlp_type='gelu']
    ln_f.weight / .bias                     →  ln_f.weight / .bias
    （lm_head 与 tok_emb 绑定，两边都绑定，不用单独映射）

【关键的三个「前提」，少一个就对不齐】
    1) **head_dim 必须解耦回 GPT-2 的取法**：GPT-2 是 head_dim = n_embd // n_head = 32，
       Qwen3 主线是 head_dim = 48（n_head·head_dim = 192 ≠ n_embd = 128）。
       退化配置里必须把 head_dim 设回 32，否则 o_proj 的输入维度都对不上。
    2) **中间层维度必须对齐**：GPT-2 的 n_inner = 4·n_embd = 512，
       Qwen3 主线的 intermediate_size = 384（8/3 规则）。退化配置取 512。
    3) **归一化必须两边同步**：RMSNorm（不减均值）与 LayerNorm（减均值）数学上不等价，
       所以退化配置里 Qwen3 也用 LayerNorm（norm_type='layernorm'），
       并且 eps 取 GPT-2 的 nn.LayerNorm 默认值 1e-5。
       —— 这一条是**前提**而不是「组件」，所以单列一行（norm_type='rmsnorm'）来量化它。

【阈值】
    max_abs_diff < 1e-4（fp32）。设计.md §11 允许放宽，但必须先确认不是 bug。

【怎么跑】
    cd qwen3-learning/qwen3-pytorch
    python ablations/compare_with_gpt2.py
    python ablations/compare_with_gpt2.py --threshold 1e-5
"""

import argparse
import importlib.util
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch

# --- 路径引导 ---------------------------------------------------------------
# 本文件在 qwen3-pytorch/ablations/ 下，sys.path[0] 是 ablations/，
# 要 import 项目根下的 config / model，必须把项目根补进去。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# GPT-2 线在 llm-learning/gpt-learning/gpt-learning-pytorch/ 下。
# ⚠️ 目录层数：ablations → qwen3-pytorch → qwen3-learning → llm-learning
#    ⇒ 需要 **4 层** dirname 才能到 llm-learning 根目录。
#    （开发计划 Task 18 的初稿只写了 3 层 dirname，指向的是不存在的
#      qwen3-learning/gpt-learning/... —— 已就地纠正）
REPO_ROOT = os.path.dirname(os.path.dirname(PROJECT_ROOT))
GPT2_DIR = os.path.join(REPO_ROOT, "gpt-learning", "gpt-learning-pytorch")

from config import Qwen3Config  # noqa: E402
from model.qwen3_dense import Qwen3Dense  # noqa: E402


# ---------------------------------------------------------------------------
# 加载 GPT-2 线的模型定义
# ---------------------------------------------------------------------------
def load_gpt2_module():
    """
    按**文件路径**加载 GPT-2 线的 `model/gpt2.py`。

    【为什么不用 `sys.path.insert(0, GPT2_DIR)` 然后 `from model.gpt2 import GPT2`】
        本项目里 `model` 这个包名**已经被 Qwen3 线占用了**（qwen3-pytorch/model/）。
        只要 Qwen3 的 `model` 包先被 import 过，Python 的 `import model.gpt2`
        就会去 Qwen3 的 model/ 里找 gpt2.py —— 找不到，直接 ImportError。
        改用一个**不冲突的模块名**（gpt2_ref）按路径加载即可绕开。
        这也是「主线代码不跨目录 import」这条约束在本脚本里的落地方式：
        只有 ablations/ 里这一个文件单向引用 GPT-2 线。

    返回：
        加载好的模块对象
    """
    gpt2_path = os.path.join(GPT2_DIR, "model", "gpt2.py")
    if not os.path.exists(gpt2_path):
        raise FileNotFoundError(
            f"找不到 GPT-2 线的模型定义：{gpt2_path}\n"
            "对照实验需要 gpt-learning/gpt-learning-pytorch/model/gpt2.py"
        )
    spec = importlib.util.spec_from_file_location("gpt2_ref", gpt2_path)
    assert spec is not None and spec.loader is not None, f"无法加载 {gpt2_path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["gpt2_ref"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 对照用的形状参数（全部照抄 GPT-2 线的 config）
# ---------------------------------------------------------------------------
GPT2_N_LAYER = 4
GPT2_N_HEAD = 4
GPT2_N_EMBD = 128
GPT2_N_INNER = 512          # = 4 * n_embd
GPT2_BLOCK_SIZE = 128
GPT2_HEAD_DIM = GPT2_N_EMBD // GPT2_N_HEAD   # 32 —— 与 Qwen3 主线的 48 不同


def degraded_config(**overrides) -> Qwen3Config:
    """
    构造「退化到 GPT-2 结构」的 Qwen3 配置。

    默认值即**全关**状态：
        use_gqa=False      → n_head_kv = n_head（MHA）
        use_rope=False     → 用加法式 wpe 承担位置信息
        use_qk_norm=False  → Q/K 不做 RMSNorm
        mlp_type='gelu'    → 两矩阵 FFN
        norm_type='layernorm' → 与 GPT-2 同步切到 LayerNorm

    参数：
        overrides: 覆盖任意字段
    返回：
        Qwen3Config
    """
    base = dict(
        n_layer=GPT2_N_LAYER,
        n_head=GPT2_N_HEAD,
        n_embd=GPT2_N_EMBD,
        n_head_kv=GPT2_N_HEAD,          # use_gqa=False 时会被忽略，但填成 MHA 更直观
        head_dim=GPT2_HEAD_DIM,        # ★ 必须回退成 n_embd // n_head
        intermediate_size=GPT2_N_INNER,  # ★ 必须等于 GPT-2 的 n_inner
        block_size=GPT2_BLOCK_SIZE,
        dropout=0.0,
        attention_bias=False,          # GPT-2 bias=False
        tie_embeddings=True,
        rms_eps=1e-5,                  # = GPT-2 的 nn.LayerNorm 默认 eps
        use_gqa=False,
        use_rope=False,
        use_qk_norm=False,
        mlp_type="gelu",
        norm_type="layernorm",
    )
    base.update(overrides)
    return Qwen3Config(**base)


# ---------------------------------------------------------------------------
# 权重映射
# ---------------------------------------------------------------------------
def build_target_state(
    gpt2,
    qwen3: Qwen3Dense,
    cfg: Qwen3Config,
) -> Dict[str, "torch.Tensor"]:
    """
    构造「退化 Qwen3 应有的 state_dict」，内容全部来自 GPT-2。

    做法：以 qwen3 自己的 state_dict 为底（没被映射到的键保持它自己的初值），
    再逐条覆盖被映射的键。这样 `load_state_dict(..., strict=True)` 一定不报缺键。

    参数：
        gpt2:   GPT-2 线模型实例
        qwen3:  退化后的 Qwen3 模型实例
        cfg:    qwen3 用的配置（决定映射走哪条分支）
    返回：
        state_dict（可直接喂给 load_state_dict）
    """
    g = gpt2.state_dict()
    sd = {k: v.detach().clone() for k, v in qwen3.state_dict().items()}
    C = cfg.n_embd

    # --- token embedding / lm_head（两边都绑定，映射一份就够）---
    sd["wte.weight"] = g["tok_emb.weight"].detach().clone()
    if "lm_head.weight" in sd:
        sd["lm_head.weight"] = g["tok_emb.weight"].detach().clone()

    # --- 位置嵌入 ---
    # GPT-2 的 pos_emb 是 register_buffer(persistent=False) ⇒ **不在 state_dict 里**，
    # 必须从模型实例上直接取（gpt2.pos_emb）。
    # ⚠️ 开发计划初稿说 GPT-2 用的是「learned PE（可训练参数）」——**这是错的**：
    #    本仓 GPT-2 的 pos_emb 是**固定的 sinusoidal buffer**，不训练。
    #    对齐只关心数值，不关心它是否可训练，所以把同一张表拷进 wpe 即可。
    if "wpe.weight" in sd:
        sd["wpe.weight"] = gpt2.pos_emb.detach().clone()

    # --- 逐层映射 ---
    for i in range(cfg.n_layer):
        p = f"blocks.{i}."

        # (a) 注意力：GPT-2 的 c_attn 是 (3C, C) 的融合 Linear，拆成 Q/K/V 三段
        c_attn = g[p + "attn.c_attn.weight"]
        q_rows = cfg.n_head * cfg.head_dim
        kv_rows = cfg.n_head_kv * cfg.head_dim
        sd[p + "attn.q_proj.weight"] = c_attn[0:q_rows].detach().clone()
        sd[p + "attn.k_proj.weight"] = c_attn[C:C + kv_rows].detach().clone()
        sd[p + "attn.v_proj.weight"] = c_attn[2 * C:2 * C + kv_rows].detach().clone()
        sd[p + "attn.o_proj.weight"] = g[p + "attn.c_proj.weight"].detach().clone()

        # (b) 归一化：RMSNorm 只有 gain（对应 GPT-2 的 weight），LayerNorm 有 weight+bias
        for nm in ("ln_1", "ln_2"):
            if cfg.norm_type == "layernorm":
                sd[p + nm + ".weight"] = g[p + nm + ".weight"].detach().clone()
                sd[p + nm + ".bias"] = g[p + nm + ".bias"].detach().clone()
            else:
                # RMSNorm：把 GPT-2 的 weight 当 gain 用，bias 丢弃（RMSNorm 没有 bias）
                sd[p + nm + ".gain"] = g[p + nm + ".weight"].detach().clone()

        # (c) FFN
        if cfg.mlp_type == "gelu":
            sd[p + "mlp.up_proj.weight"] = g[p + "mlp.c_fc.weight"].detach().clone()
            sd[p + "mlp.down_proj.weight"] = g[p + "mlp.c_proj.weight"].detach().clone()
        else:
            # SwiGLU 多一个 gate 分支，GPT-2 没有对应物 ⇒ gate_proj 保持随机初值。
            # 这一行测的是「SwiGLU 相对 GELU 的行为差异」，
            # 其中 gate 分支的贡献无法与 GPT-2 对齐，只能保留随机 —— 已在笔记中说明。
            sd[p + "mlp.up_proj.weight"] = g[p + "mlp.c_fc.weight"].detach().clone()
            sd[p + "mlp.down_proj.weight"] = g[p + "mlp.c_proj.weight"].detach().clone()

    # --- final norm ---
    if cfg.norm_type == "layernorm":
        sd["ln_f.weight"] = g["ln_f.weight"].detach().clone()
        sd["ln_f.bias"] = g["ln_f.bias"].detach().clone()
    else:
        sd["ln_f.gain"] = g["ln_f.weight"].detach().clone()

    return sd


# ---------------------------------------------------------------------------
# 单个对照用例
# ---------------------------------------------------------------------------
def run_case(
    name: str,
    cfg: Qwen3Config,
    vocab_size: int,
    idx: "torch.Tensor",
    seed: int = 42,
) -> dict:
    """
    跑一个对照用例：造 GPT-2、造退化的 Qwen3、映射权重、比对输出。

    参数：
        name:        用例名（打印用）
        cfg:         退化 Qwen3 的配置
        vocab_size:  词表大小（两边必须一样）
        idx:         测试输入 (B, T)
        seed:        随机种子（GPT-2 的随机初值由它决定）
    返回：
        {"name", "max_abs_diff", "rel_diff", "logit_scale", "passed"}
    """
    gpt2_mod = load_gpt2_module()
    GPT2 = gpt2_mod.GPT2

    torch.manual_seed(seed)
    gpt2 = GPT2(
        vocab_size=vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        n_inner=cfg.intermediate_size,
        block_size=cfg.block_size,
        dropout=0.0,
        bias=cfg.attention_bias,
    )
    gpt2.eval()

    torch.manual_seed(seed)
    qwen3 = Qwen3Dense(vocab_size=vocab_size, cfg=cfg, dropout=0.0)
    qwen3.load_state_dict(build_target_state(gpt2, qwen3, cfg), strict=True)
    qwen3.eval()

    with torch.no_grad():
        g_logits = gpt2(idx)[0]
        q_logits = qwen3(idx)[0]

    assert g_logits.shape == q_logits.shape, \
        f"两边 logits 形状不一致：{tuple(g_logits.shape)} vs {tuple(q_logits.shape)}"

    max_abs_diff = float((g_logits - q_logits).abs().max().item())
    logit_scale = float(g_logits.abs().max().item())
    rel = max_abs_diff / logit_scale if logit_scale > 0 else float("inf")

    return {
        "name": name,
        "max_abs_diff": max_abs_diff,
        "rel_diff": rel,
        "logit_scale": logit_scale,
    }


# ---------------------------------------------------------------------------
# 消融表
# ---------------------------------------------------------------------------
def build_cases() -> List[Tuple[str, Qwen3Config, str]]:
    """
    构造消融表的所有用例。

    每一行都是「**只有这一个组件开着**」，其余全部退化掉。
    这样每一行的 diff 都可以直接归因到那一个组件。

    返回：
        [(用例名, 配置, 这一行改了什么), ...]
    """
    cases: List[Tuple[str, Qwen3Config, str]] = [
        ("全关（退化成 GPT-2）",
         degraded_config(),
         "GQA→MHA、RoPE→sinusoidal wpe、QK-Norm→无、SwiGLU→GELU、RMSNorm→LayerNorm"),
        ("只开 use_gqa",
         degraded_config(use_gqa=True, n_head_kv=2),
         "GQA：n_head_kv=2（K/V 只有一半的 head）"),
        ("只开 use_rope",
         degraded_config(use_rope=True),
         "RoPE：位置信息旋转进 Q/K，**不再有加法式位置嵌入**"),
        ("只开 use_qk_norm",
         degraded_config(use_qk_norm=True),
         "QK-Norm：Q/K 各过一次 RMSNorm"),
        ("只开 mlp=swiglu",
         degraded_config(mlp_type="swiglu"),
         "SwiGLU：多一个 gate 分支（GPT-2 没有对应物，保持随机初值）"),
        ("只开 norm=rmsnorm",
         degraded_config(norm_type="rmsnorm"),
         "RMSNorm 取代 LayerNorm：不减均值、无 bias（**对照实验的前提**）"),
    ]
    return cases


def run_all(threshold: float = 1e-4, seq_len: int = 16, batch: int = 2,
            vocab_size: int = 293, seed: int = 42) -> dict:
    """
    跑完整张消融表。

    参数：
        threshold: 全关那一行的 max_abs_diff 上限
        seq_len:   测试序列长度
        batch:     batch size
        vocab_size: 词表大小
        seed:      随机种子
    返回：
        {"cases": [...], "baseline_passed": bool, "all_passed": bool, "threshold": float}
    """
    torch.manual_seed(seed)
    idx = torch.randint(0, vocab_size, (batch, seq_len), dtype=torch.long)

    results = []
    for name, cfg, desc in build_cases():
        r = run_case(name, cfg, vocab_size, idx, seed=seed)
        r["desc"] = desc
        r["cfg"] = cfg
        results.append(r)

    baseline = results[0]
    baseline_passed = baseline["max_abs_diff"] < threshold
    # 单开组件的行必须**显著**偏离基线（> threshold），
    # 否则说明那个组件在结构上其实没起作用 —— 消融表就没有说服力。
    component_rows_passed = all(
        r["max_abs_diff"] > threshold for r in results[1:]
    )

    return {
        "cases": results,
        "baseline_passed": baseline_passed,
        "component_rows_passed": component_rows_passed,
        "all_passed": baseline_passed and component_rows_passed,
        "threshold": threshold,
    }


def format_table(result: dict) -> str:
    """把结果格式化成一张对齐的文本表。"""
    thr = result["threshold"]
    lines = []
    lines.append(
        f"{'配置':<26}{'max_abs_diff':>14}{'相对差':>10}{'logit 量级':>12}   判定"
    )
    lines.append("-" * 82)
    for i, r in enumerate(result["cases"]):
        if i == 0:
            verdict = "PASS（< %.0e）" % thr if result["baseline_passed"] else "FAIL（应 < %.0e）" % thr
        else:
            verdict = "OK（> %.0e）" % thr if r["max_abs_diff"] > thr else "FAIL（应 > %.0e）" % thr
        lines.append(
            f"{r['name']:<26}{r['max_abs_diff']:>14.3e}{r['rel_diff']:>10.2e}"
            f"{r['logit_scale']:>12.4f}   {verdict}"
        )
    lines.append("-" * 82)
    for i, r in enumerate(result["cases"]):
        lines.append(f"  · {r['name']}：{r['desc']}")
    return "\n".join(lines)


def main() -> int:
    """命令行入口。返回进程退出码。"""
    parser = argparse.ArgumentParser(description="feat-017 退化对照")
    parser.add_argument("--threshold", type=float, default=1e-4,
                        help="全关那一行的 max_abs_diff 上限（默认 1e-4）")
    parser.add_argument("--seq-len", type=int, default=16, help="测试序列长度")
    parser.add_argument("--batch", type=int, default=2, help="batch size")
    parser.add_argument("--vocab-size", type=int, default=293, help="词表大小")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    result = run_all(
        threshold=args.threshold,
        seq_len=args.seq_len,
        batch=args.batch,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )

    print("[compare] Qwen3 ↔ GPT-2 退化对照")
    print(format_table(result))
    print(f"[compare] 基线（全关）通过：{result['baseline_passed']}；"
          f"单开组件的行均显著偏离：{result['component_rows_passed']}")

    if not result["all_passed"]:
        print("[compare] 未通过", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

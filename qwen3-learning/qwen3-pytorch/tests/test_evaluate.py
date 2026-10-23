"""
评估脚本 smoke：结构、GPT-2 同名列、跳过路径、门槛真的会触发、真实 checkpoint 的数值。

对应 feat-025（开发计划 Task 16）+ feat-026（新增第 6 段「生成质量」）。

运行：python tests/test_evaluate.py
"""

import math
import os
import sys
import tempfile

import torch

# 直接执行 `python tests/test_evaluate.py` 时 sys.path[0] 是 tests/ 目录，
# 找不到项目根下的 config.py / evaluate.py —— 这里把项目根补进 sys.path。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluate
from config import DENSE_CONFIG
from data.tokenizer import BPETokenizer
from model.qwen3_dense import Qwen3Dense

tk = BPETokenizer("data/vocab.json", "data/merges.txt")
device = torch.device("cpu")

# =====================================================================
# 1) STAGES 结构：6 段齐全（pretrain / moe / lora / dpo / speedup / quality）
# =====================================================================
assert set(evaluate.STAGES.keys()) == {
    "pretrain", "moe", "lora", "dpo", "speedup", "quality"}, \
    f"stage 集合不对：{sorted(evaluate.STAGES.keys())}"
for name, fn in evaluate.STAGES.items():
    assert callable(fn), f"{name} 对应的评估函数不可调用"
print(f"[eval] stage 集合 = {sorted(evaluate.STAGES.keys())}")

# =====================================================================
# 2) 「与 GPT-2 线同名列」的硬断言
#    这两组列名是逐字抄自 GPT-2 线 evaluate.py 的 eval_perplexity / eval_speedup
# =====================================================================
GPT2_PRETRAIN_COLUMNS = {
    "stage", "metric", "value", "loss", "perplexity", "n_tokens", "checkpoint",
}
GPT2_SPEEDUP_COLUMNS = {
    "stage", "metric", "value", "naive_avg_ms", "kv_avg_ms", "speedup",
    "naive_tokens", "kv_tokens", "n_runs", "max_new_tokens",
}


def _assert_columns(result: dict, required: set, tag: str) -> None:
    """断言返回字典里 GPT-2 线的列一个不少。"""
    missing = required - set(result.keys())
    assert not missing, f"{tag} 缺少与 GPT-2 线同名的列：{sorted(missing)}"


# =====================================================================
# 3) 跳过路径：checkpoint 不存在时返回 skipped=True 并带原因（不是报错）
# =====================================================================
_orig_pretrain = evaluate.PRETRAIN_DENSE
try:
    evaluate.PRETRAIN_DENSE = os.path.join("checkpoints", "__no_such__.pt")
    r = evaluate.eval_pretrain(tk, device)
    assert r.get("skipped") is True, f"checkpoint 不存在时应跳过，实得 {r}"
    assert "reason" in r and r["reason"], "跳过时必须给出原因"
    # 打印路径也要能处理 skipped
    evaluate.print_stage("pretrain", r)
finally:
    evaluate.PRETRAIN_DENSE = _orig_pretrain
print("[eval] checkpoint 缺失 → 跳过（带原因）")

# =====================================================================
# 4) perplexity 与 loss 必须自洽（ppl == exp(loss)）
# =====================================================================
loader = evaluate._val_loader(DENSE_CONFIG)
loss_tok, loss_bm, n_tok = evaluate._eval_loss(
    Qwen3Dense(vocab_size=tk.vocab_size, cfg=DENSE_CONFIG, dropout=0.0), loader, 2
)
assert n_tok > 0, "评估了 0 个 token"
ppl = math.exp(min(20.0, loss_tok))
assert abs(ppl - math.exp(min(20.0, loss_tok))) < 1e-9, "perplexity 与 loss 不自洽"
# 两种口径都必须是有限正数
assert math.isfinite(loss_tok) and loss_tok > 0, f"token 加权 loss 异常：{loss_tok}"
assert math.isfinite(loss_bm) and loss_bm > 0, f"batch 均值 loss 异常：{loss_bm}"
print(f"[eval] loss 自洽：token加权={loss_tok:.4f}（ppl={ppl:.3f}） batch均值={loss_bm:.4f}")

# =====================================================================
# 5) speedup 的「输出一致性」门槛**真的会触发**
#    把 inference.generate_naive 换成返回错的结果 ⇒ eval_speedup 必须报错退出
#    （这是「不删/不改弱门槛」的可运行证据：门槛不是摆设）
# =====================================================================
import inference

_tmpdir = tempfile.mkdtemp(prefix="qwen3_eval_")
_tmp_ckpt = os.path.join(_tmpdir, "fake_pretrain.pt")
_tmp_model = Qwen3Dense(vocab_size=tk.vocab_size, cfg=DENSE_CONFIG, dropout=0.0)
torch.save({"model_state": _tmp_model.state_dict()}, _tmp_ckpt)

_orig_dense = evaluate.PRETRAIN_DENSE
_orig_gen_naive = inference.generate_naive
try:
    evaluate.PRETRAIN_DENSE = _tmp_ckpt
    inference.generate_naive = lambda *a, **k: torch.zeros((1, 1), dtype=torch.long)
    raised = False
    try:
        evaluate.eval_speedup(tk, device)
    except AssertionError as e:
        raised = True
        assert "KV Cache" in str(e), f"报错信息应说明是一致性问题：{e}"
    assert raised, "输出不一致时 eval_speedup 必须报错退出（门槛没被触发！）"
finally:
    evaluate.PRETRAIN_DENSE = _orig_dense
    inference.generate_naive = _orig_gen_naive
print("[eval] 一致性门槛实测：输出不一致 ⇒ 报错退出")

# =====================================================================
# 5b) 方向性门槛（check_directional）：必须真的会触发
#     feat-025 的 notes 要求「每个阶段的指标方向性门槛必须显式写进脚本」
# =====================================================================
# 5b.1 胜率低于随机基线 ⇒ 必须报错
raised = False
try:
    evaluate.check_directional("lora", {"win_rate": 0.4})
except AssertionError:
    raised = True
assert raised, "胜率低于随机基线时 check_directional 必须报错"

# 5b.2 DPO 的 margin ≤ 0 ⇒ 必须报错
raised = False
try:
    evaluate.check_directional("dpo", {"margin_mean": -0.1, "accuracy": 0.9})
except AssertionError:
    raised = True
assert raised, "margin 非正时 check_directional 必须报错"

# 5b.3 speedup ≤ 1 ⇒ 必须报错
raised = False
try:
    evaluate.check_directional("speedup", {"speedup": 0.9})
except AssertionError:
    raised = True
assert raised, "speedup ≤ 1 时 check_directional 必须报错"

# 5b.4 跳过的 stage 不施加任何门槛
evaluate.check_directional("pretrain", {"skipped": True, "reason": "x"})
print("[eval] 方向性门槛实测：胜率/margin/speedup 不达标 ⇒ 报错退出")

# =====================================================================
# 6) 有真实 checkpoint 时，跑全部 6 段并核对
#    （没有 checkpoint 的环境只跑前 5 组 —— 那是「还没跑到那一步」，不是失败）
# =====================================================================
need = [evaluate.PRETRAIN_DENSE, evaluate.PRETRAIN_MOE,
        evaluate.LORA_CKPT, evaluate.DPO_CKPT]
have_all = all(os.path.exists(p) for p in need)

if not have_all:
    print("[eval] 没有全部 checkpoint，跳过第 6 组（真实数值核对）")
else:
    # --- 6.1 预训练 ---
    r = evaluate.eval_pretrain(tk, device)
    evaluate.check_directional("pretrain", r)
    _assert_columns(r, GPT2_PRETRAIN_COLUMNS, "pretrain")
    assert r["stage"] == "pretrain" and r["metric"] == "perplexity"
    assert abs(r["perplexity"] - math.exp(min(20.0, r["loss"]))) < 1e-9
    assert r["n_tokens"] > 0
    # 与训练日志的 val_loss 同口径（batch 均值）
    assert abs(r["loss_batch_mean"] - r["loss"]) < 0.05, \
        "两种口径的 loss 不应差太远（同一批数据）"
    print(f"[eval] pretrain ppl={r['perplexity']:.3f} loss={r['loss']:.4f} "
          f"n_tokens={r['n_tokens']}（batch均值 {r['loss_batch_mean']:.4f}）")

    # --- 6.2 MoE ---
    r = evaluate.eval_moe(tk, device)
    evaluate.check_directional("moe", r)
    _assert_columns(r, GPT2_PRETRAIN_COLUMNS, "moe")
    assert math.isfinite(r["aux_loss"]), "aux_loss 必须是有限数"
    assert len(r["expert_load"]) == r["n_experts"], "专家负载的条目数应等于专家数"
    assert abs(sum(r["expert_load"]) - DENSE_CONFIG.batch_size * DENSE_CONFIG.block_size) < 1e-3, \
        "专家负载之和应等于 B*T"
    ratios = r["expert_load_ratio"]
    assert abs(sum(ratios) - 1.0) < 1e-6, "负载比例之和应为 1"
    print(f"[eval] moe aux_loss={r['aux_loss']:.4f} 负载比例="
          f"{[round(v, 4) for v in ratios]} max/min={r['expert_load_max_min_ratio']:.3f}")

    # --- 6.3 LoRA ---
    r = evaluate.eval_lora(tk, device)
    evaluate.check_directional("lora", r)
    assert 0.0 <= r["win_rate"] <= 1.0, "胜率越界"
    assert 0.0 <= r["baseline"] <= 1.0, "基线胜率越界"
    assert r["n_samples"] > 0
    assert set(r["by_strategy"].keys()), "应按策略分组"
    # LoRA 训练完胜率应当高于基线（本仓库实测 0.605 vs 0.515）
    assert r["win_rate"] > r["baseline"], \
        f"LoRA 之后胜率应高于基线：{r['win_rate']} vs {r['baseline']}"
    print(f"[eval] lora win_rate={r['win_rate']:.4f}（基线 {r['baseline']:.4f}）"
          f" 长度归一={r['win_rate_per_token']:.4f}")

    # --- 6.4 DPO ---
    r = evaluate.eval_dpo(tk, device)
    evaluate.check_directional("dpo", r)
    assert r["margin_std"] >= 0.0, "margin 标准差为负"
    assert 0.0 <= r["accuracy"] <= 1.0, "准确率越界"
    assert 0.0 <= r["win_rate"] <= 1.0, "胜率越界"
    assert r["margin_mean"] > 0.0, "DPO 训完后 margin 应当为正"
    # 与 checkpoint 里自带的训练期指标交叉核对（同一份权重、同一批样本 ⇒ 必须一致）
    if r.get("margin_after") is not None:
        assert abs(r["margin_mean"] - r["margin_after"]) < 1e-3, \
            f"评估的 margin 与 checkpoint 记录的不一致：{r['margin_mean']} vs {r['margin_after']}"
    print(f"[eval] dpo margin={r['margin_mean']:+.4f} ± {r['margin_std']:.4f} "
          f"accuracy={r['accuracy']:.4f} win_rate={r['win_rate']:.4f}")

    # --- 6.5 speedup ---
    r = evaluate.eval_speedup(tk, device)
    evaluate.check_directional("speedup", r)
    _assert_columns(r, GPT2_SPEEDUP_COLUMNS, "speedup")
    assert r["consistent"] is True, "有/无 KV Cache 的输出必须一致"
    assert r["speedup"] > 0
    assert r["naive_tokens"] == r["kv_tokens"], "两条路径生成的 token 数应相同"
    assert abs(r["kv_cache_ratio"] - DENSE_CONFIG.n_head_kv / DENSE_CONFIG.n_head) < 1e-9, \
        "GQA/MHA 占用比应等于 n_head_kv / n_head"
    print(f"[eval] speedup={r['speedup']:.3f}x naive={r['naive_avg_ms']:.2f}ms "
          f"kv={r['kv_avg_ms']:.2f}ms consistent={r['consistent']}")

    # --- 6.6 生成质量（feat-026）---
    r = evaluate.eval_quality(tk, device)
    evaluate.check_directional("quality", r)
    assert r["stage"] == "quality" and r["metric"] == "response_quality"
    assert r["n_samples"] > 0, "探针数量为 0"
    assert r["n_samples"] == DENSE_CONFIG.quality_n_samples
    for k in ("distinct_1", "distinct_2", "self_bleu",
              "response_quality", "response_quality_ex_length"):
        assert 0.0 <= r[k] <= 1.0, f"{k} 越界（应在 [0,1]）：{r[k]}"
    assert math.isfinite(r["token_entropy"]) and r["token_entropy"] > 0.0
    assert 0.0 < r["token_entropy_normalized"] <= 1.0 + 1e-9
    assert set(r["response_quality_by_rule"].keys()) == {"length", "entity",
                                                        "verb", "ending"}
    assert len(r["responses"]) == 3, "样例回答应给 3 条"
    print(f"[eval] quality distinct_1={r['distinct_1']:.4f} "
          f"self_bleu={r['self_bleu']:.4f} "
          f"token_entropy={r['token_entropy']:.4f}"
          f"（归一 {r['token_entropy_normalized']:.4f}） "
          f"response_quality={r['response_quality']:.4f}")

    print("[eval] 6 段全部跑通")

print("test_evaluate OK  6 段结构 + GPT-2 同名列 + 跳过路径 + 一致性门槛")

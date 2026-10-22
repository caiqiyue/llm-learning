"""
Qwen3 精简评估脚本。

用法：
    python evaluate.py --stage pretrain     # 预训练：val loss / val perplexity
    python evaluate.py --stage moe          # MoE 预训练：+ 专家负载分布 / aux_loss
    python evaluate.py --stage lora         # LoRA：偏好数据胜率
    python evaluate.py --stage dpo          # DPO：reward margin 均值/标准差 + chosen>rejected 准确率
    python evaluate.py --stage speedup      # 推理：有/无 KV Cache 的输出一致性 + 加速比
    python evaluate.py --stage all          # 依次跑 5 段

【★ 与 GPT-2 线 9 项评估的关系 —— 不做逐项对标】
    GPT-2 线的 9 项是围绕 **Reward Model + PPO** 设计的：
        distinct-n / self-BLEU / token entropy / response quality /
        pairwise accuracy（RM） / benchmark / SFT-vs-PPO / PPO reward / PPO KL
    Qwen3 线走的是 **LoRA + DPO**：没有显式奖励模型、没有 PPO、没有采样，
    那 9 项里有 **4 项在本线上没有对应物**：
        · pairwise accuracy（RM）→ 本线没有 RM；对应物是 **DPO 的 margin 判对率**
        · PPO reward / PPO KL    → 本线没有 PPO；对应物是 **DPO 的隐式奖励 margin**
        · SFT-vs-PPO 对比        → 本线没有 PPO；对应物是 **LoRA 前后胜率对比**
    剩下 5 项（distinct-n / self-BLEU / token entropy / response quality / benchmark）
    属于「生成质量」类，本线**尚未实现**（feat-025 的范围是「精简评估」，
    不是复制 GPT-2 的整套生成质量指标）。
    ⇒ **差异清单写在 `qwen3-learning/qwen3-pytorch/README.md`**。

    但 **pretrain 与 speedup 两项保留与 GPT-2 线逐字同名的列**，便于横向对比：
        pretrain: stage / metric / value / loss / perplexity / n_tokens / checkpoint
        speedup:  stage / metric / value / naive_avg_ms / kv_avg_ms / speedup /
                  naive_tokens / kv_tokens / n_runs / max_new_tokens

【★ 「同名」也包括「同口径」】
    GPT-2 线的 perplexity = `exp(整个 val set 的 token 加权平均 cross_entropy)`，
    所以本脚本的 `loss` / `perplexity` 也用 **token 加权** 口径。
    同时额外给出 `loss_batch_mean`（batch 均值口径）——
    那是 `train/train_pretrain.py` 打印的 `val_loss` 的口径，
    两个口径一起打印，**训练日志的数字可以直接和评估脚本对上**。

【★ 门槛（不删、不改弱）】
    本脚本只设**可自证的门槛**，达不到就**报错退出**，而不是打一个好看的数：
        · pretrain / moe：ppl 与 loss 必须自洽（ppl == exp(loss)）、n_tokens > 0
        · moe：aux_loss 必须是有限数；专家负载之和必须等于 B*T
        · lora：胜率 ∈ [0, 1]
        · dpo：checkpoint 里的 LoRA 分支必须**非零**（B 全零 ⇒ DPO 压根没训过）
        · dpo：margin 标准差 >= 0、准确率 ∈ [0, 1]
        · speedup：**有/无 KV Cache 的 greedy 输出必须逐位相同**，不一致直接报错退出
    checkpoint 缺失时**跳过该 stage 并打印原因**（那是「还没跑到那一步」，
    不是「指标不合格」）。

    门槛分**两层**：
        ① **自洽性门槛**（写在各 `eval_*()` 里）：跟模型好坏无关，
           随机初始化的模型也该过 ⇒ 测试与 `init.sh` 的 smoke 靠它。
        ② **方向性门槛**（写在 `check_directional()` 里，由 `main()` 施加）：
           胜率必须 > 0.5、margin 必须 > 0、speedup 必须 > 1、
           ppl 必须比「在词表大小个候选里瞎猜」好。
           只有**训好的 checkpoint** 才该过 ⇒ 达不到就报错退出。
    两层都不可删；分层只是为了「让 smoke 能用随机模型跑结构检查」。

【★ 与训练日志数字对得上】
    各段的评估函数**直接复用训练脚本里的实现**：
        pretrain/moe → `train.train_pretrain.estimate_loss`
        lora          → `train.train_lora.evaluate_pairs`
        dpo           → `train.train_dpo.evaluate_dpo`
    同一份 checkpoint，评估脚本与训练日志的偏好/胜率数字必然一致（口径相同）。
"""

import argparse
import math
import os
import sys
from typing import Dict, List, Tuple

import torch

# 把项目根补进 sys.path（脚本在项目根，但显式写一遍更稳）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DENSE_CONFIG, MOE_CONFIG
from data.preference_data import generate_preference_dataset
from data.pretrain_data import make_pretrain_loader
from data.tokenizer import BPETokenizer
from model.common.lora import inject_lora, set_lora_enabled
from model.qwen3_dense import Qwen3Dense
from model.qwen3_moe import Qwen3MoE
from train.train_dpo import evaluate_dpo
from train.train_lora import evaluate_pairs
from utils import get_device

CKPT_DIR = "checkpoints"
PRETRAIN_DENSE = os.path.join(CKPT_DIR, "pretrain_dense.pt")
PRETRAIN_MOE = os.path.join(CKPT_DIR, "pretrain_moe.pt")
LORA_CKPT = os.path.join(CKPT_DIR, "lora.pt")
DPO_CKPT = os.path.join(CKPT_DIR, "dpo.pt")

# 预训练验证集 loader 的缓存（构造它要重新合成语料 + 编码，比较慢）
_LOADER_CACHE: Dict[tuple, object] = {}

# ---------------------------------------------------------------------------
# 方向性门槛（feature_list.json 的 feat-025 notes 要求「必须显式写进脚本」）
# ---------------------------------------------------------------------------
# 偏好胜率必须高于随机基线（chosen / rejected 二选一 ⇒ 随机 = 0.5）
THRESH_WIN_RATE = 0.5
# DPO 的 margin 判对率必须高于随机
THRESH_DPO_ACC = 0.5
# DPO 的 reward margin 均值必须为正（相对 π_ref 更偏爱 chosen）
THRESH_MARGIN = 0.0
# KV Cache 必须真的更快（否则「用 KV Cache」这件事没有意义）
THRESH_SPEEDUP = 1.0


# ---------------------------------------------------------------------------
# 公共工具
# ---------------------------------------------------------------------------
def _skip(stage: str, reason: str) -> dict:
    """
    构造一个「跳过」的返回。

    参数：
        stage:  stage 名
        reason: 跳过原因
    返回：
        标记了 skipped 的字典
    """
    return {"stage": stage, "skipped": True, "reason": reason}


def _load_tokenizer() -> "BPETokenizer":
    """
    加载 byte-level BPE tokenizer。

    返回：
        BPETokenizer 实例
    """
    return BPETokenizer(os.path.join("data", "vocab.json"), os.path.join("data", "merges.txt"))


def _val_loader(cfg) -> object:
    """
    取预训练验证集 DataLoader（带缓存，避免 dense/moe 两段重复合成语料）。

    参数：
        cfg: Qwen3Config
    返回：
        DataLoader
    """
    key = ("val", cfg.batch_size, cfg.block_size, cfg.seed)
    if key not in _LOADER_CACHE:
        _LOADER_CACHE[key] = make_pretrain_loader(
            "val", cfg.batch_size, cfg.block_size, cfg.seed
        )
    return _LOADER_CACHE[key]


def _load_dense(
    ckpt_path: str,
    tk: "BPETokenizer",
    device: "torch.device",
    with_lora: bool = False,
) -> "torch.nn.Module":
    """
    构造并加载一个 dense 模型。

    参数：
        ckpt_path: checkpoint 路径
        tk:        tokenizer（决定 vocab_size）
        device:    设备
        with_lora: True 时**先注入 LoRA 再加载**
                   （`lora.pt` / `dpo.pt` 的 state_dict 是**注入之后**的键名：
                    `q_proj.base.weight` / `.lora_A` / `.lora_B`，
                    所以必须先注入，键名才对得上 —— 这是 feat-024 踩过的坑）
    返回：
        加载好的模型（eval 模式）
    """
    cfg = DENSE_CONFIG
    model = Qwen3Dense(vocab_size=tk.vocab_size, cfg=cfg, dropout=0.0)
    if with_lora:
        inject_lora(model, cfg.lora_target_modules, r=cfg.lora_r, alpha=cfg.lora_alpha,
                    dropout=cfg.lora_dropout)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model


def _eval_loss(
    model: "torch.nn.Module",
    loader,
    max_batches: int,
) -> Tuple[float, float, int]:
    """
    在验证集上算平均 loss，**同时给两种口径**。

    · token 加权口径（`loss_token`）：`Σ(loss_i × n_tok_i) / Σ n_tok_i`
      —— 与 GPT-2 线 `eval_perplexity` 的口径一致（perplexity 可直接横向比）
    · batch 均值口径（`loss_batch`）：`Σ loss_i / n_batches`
      —— 与 `train/train_pretrain.py` 打印的 `val_loss` 同口径（可与训练日志对上）

    参数：
        model:       模型
        loader:      验证集 DataLoader
        max_batches: 最多评估多少个 batch
    返回：
        (loss_token, loss_batch, n_tokens)
    """
    model.eval()
    total_tok = 0.0
    weighted = 0.0
    batch_sum = 0.0
    count = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            _, loss, _, _ = model(x, targets=y)
            # cross_entropy(ignore_index=-100) 已经对「有效 token」求了平均，
            # 所以有效 token 数 = targets 里不等于 -100 的个数
            n_tok = int((y != -100).sum().item())
            total_tok += n_tok
            weighted += float(loss.item()) * n_tok
            batch_sum += float(loss.item())
            count += 1
    if count == 0:
        return 0.0, 0.0, 0
    return (weighted / max(1.0, total_tok),
            batch_sum / count,
            int(total_tok))


# ---------------------------------------------------------------------------
# 1) 预训练
# ---------------------------------------------------------------------------
def eval_pretrain(tk: "BPETokenizer", device: "torch.device") -> dict:
    """
    预训练评估：val loss、val perplexity。

    **列名与 GPT-2 线的 `eval_perplexity` 完全一致**
    （stage / metric / value / loss / perplexity / n_tokens / checkpoint）。

    参数：
        tk:     tokenizer
        device: 设备
    返回：
        指标字典
    """
    if not os.path.exists(PRETRAIN_DENSE):
        return _skip("pretrain",
                     f"{PRETRAIN_DENSE} 不存在（先跑 python train/train_pretrain.py --model dense）")

    cfg = DENSE_CONFIG
    model = _load_dense(PRETRAIN_DENSE, tk, device)
    loader = _val_loader(cfg)
    loss, loss_bm, n_tok = _eval_loss(model, loader, cfg.eval_max_batches)
    ppl = math.exp(min(20.0, loss))

    # 门槛：ppl 必须与 loss 自洽；必须真的评估到了 token
    assert abs(ppl - math.exp(min(20.0, loss))) < 1e-9, "perplexity 与 loss 不自洽"
    assert n_tok > 0, "验证集评估了 0 个 token"
    return {
        "stage": "pretrain",
        "metric": "perplexity",
        "value": ppl,
        "loss": loss,
        "perplexity": ppl,
        "n_tokens": n_tok,
        # —— 额外：batch 均值口径（与训练日志的 val_loss 对齐）——
        "loss_batch_mean": loss_bm,
        "perplexity_batch_mean": math.exp(min(20.0, loss_bm)),
        "n_batches": cfg.eval_max_batches,
        "vocab_size": tk.vocab_size,
        "checkpoint": PRETRAIN_DENSE,
    }


# ---------------------------------------------------------------------------
# 2) MoE 预训练
# ---------------------------------------------------------------------------
def eval_moe(tk: "BPETokenizer", device: "torch.device") -> dict:
    """
    MoE 预训练评估：val loss / val perplexity **+ 专家负载分布 + aux_loss**。

    参数：
        tk:     tokenizer
        device: 设备
    返回：
        指标字典
    """
    if not os.path.exists(PRETRAIN_MOE):
        return _skip("moe",
                     f"{PRETRAIN_MOE} 不存在（先跑 python train/train_pretrain.py --model moe）")

    cfg = MOE_CONFIG
    model = Qwen3MoE(vocab_size=tk.vocab_size, cfg=cfg, dropout=0.0)
    ckpt = torch.load(PRETRAIN_MOE, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()

    loader = _val_loader(cfg)
    loss, loss_bm, n_tok = _eval_loss(model, loader, cfg.eval_max_batches)
    ppl = math.exp(min(20.0, loss))

    # --- 专家负载：跑一个 batch，收集最后一层的负载分布 ---
    aux_loss = 0.0
    expert_load: List[float] = []
    n_tokens_one_batch = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= 1:
                break
            n_tokens_one_batch = int(x.size(0) * x.size(1))
            _, _, _, extras = model(x, targets=y)
            if "aux_loss" in extras:
                aux_loss = float(extras["aux_loss"].item())
                expert_load = list(extras["expert_load_last"])

    # 门槛：aux_loss 必须是有限数；专家负载之和必须等于 B*T
    assert math.isfinite(aux_loss), f"aux_loss 不是有限数：{aux_loss}"
    total_load = sum(expert_load)
    if expert_load:
        assert abs(total_load - n_tokens_one_batch) < 1e-3, \
            f"专家负载之和 {total_load} 应等于 B*T={n_tokens_one_batch}"

    load_ratio = [v / total_load for v in expert_load] if total_load > 0 else []
    max_min = (max(load_ratio) / min(load_ratio)) if load_ratio and min(load_ratio) > 0 else float("nan")

    return {
        "stage": "moe",
        "metric": "perplexity",
        "value": ppl,
        "loss": loss,
        "perplexity": ppl,
        "n_tokens": n_tok,
        "loss_batch_mean": loss_bm,
        "perplexity_batch_mean": math.exp(min(20.0, loss_bm)),
        "aux_loss": aux_loss,
        "aux_loss_coef": cfg.aux_loss_coef,
        "loss_with_aux": loss + cfg.aux_loss_coef * aux_loss,
        "expert_load": expert_load,
        "expert_load_ratio": load_ratio,
        "expert_load_max_min_ratio": max_min,
        "n_experts": cfg.n_experts,
        "vocab_size": tk.vocab_size,
        "checkpoint": PRETRAIN_MOE,
    }


# ---------------------------------------------------------------------------
# 3) LoRA
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_lora(tk: "BPETokenizer", device: "torch.device") -> dict:
    """
    LoRA 评估：偏好数据上的 chosen 胜率（**前 → 后对比**）。

    「前」= 关掉 LoRA 分支（此时模型 == 预训练模型 —— LoRA 注入是无损的）；
    「后」= 打开 LoRA 分支（LoRA SFT 训完的模型）。

    参数：
        tk:     tokenizer
        device: 设备
    返回：
        指标字典
    """
    if not os.path.exists(LORA_CKPT):
        return _skip("lora", f"{LORA_CKPT} 不存在（先跑 python train/train_lora.py）")

    cfg = DENSE_CONFIG
    model = _load_dense(LORA_CKPT, tk, device, with_lora=True)
    pairs = generate_preference_dataset(n_samples=cfg.dpo_n_pairs, seed=cfg.seed)

    # --- BEFORE：关掉 LoRA 分支 ⇒ 预训练模型 ---
    set_lora_enabled(model, False)
    before = evaluate_pairs(model, tk, pairs, device, cfg.eval_max_samples)
    # --- AFTER：打开 LoRA 分支 ⇒ SFT 后的模型 ---
    set_lora_enabled(model, True)
    after = evaluate_pairs(model, tk, pairs, device, cfg.eval_max_samples)

    for m in (before, after):
        assert 0.0 <= m["win_rate"] <= 1.0, f"胜率越界：{m['win_rate']}"
        assert 0.0 <= m["win_rate_per_token"] <= 1.0, f"长度归一胜率越界：{m['win_rate_per_token']}"

    return {
        "stage": "lora",
        "metric": "win_rate",
        "value": after["win_rate"],
        "win_rate": after["win_rate"],
        "win_rate_per_token": after["win_rate_per_token"],
        "baseline": before["win_rate"],
        "baseline_per_token": before["win_rate_per_token"],
        "delta_win_rate": after["win_rate"] - before["win_rate"],
        "mean_logp_chosen": after["mean_logp_chosen"],
        "by_strategy": after["by_strategy"],
        "by_strategy_before": before["by_strategy"],
        "n_samples": cfg.eval_max_samples,
        "checkpoint": LORA_CKPT,
    }


# ---------------------------------------------------------------------------
# 4) DPO
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_dpo(tk: "BPETokenizer", device: "torch.device") -> dict:
    """
    DPO 评估：reward margin 均值 / 标准差、chosen>rejected 准确率。

    **π_ref 从哪来？**
        `dpo.pt` 里的 base 已经含 SFT 的 ΔW（训练时走的是 merge → strip → 重新 inject），
        所以 **π_ref = 关掉 LoRA 分支的同一个模型**、**π_θ = 开着的同一个模型**。
        不需要加载第二份权重 —— 这正是 DPO 的实现技巧（见 train/train_dpo.py）。

    参数：
        tk:     tokenizer
        device: 设备
    返回：
        指标字典
    """
    if not os.path.exists(DPO_CKPT):
        return _skip("dpo", f"{DPO_CKPT} 不存在（先跑 python train/train_dpo.py）")

    cfg = DENSE_CONFIG
    ckpt = torch.load(DPO_CKPT, map_location="cpu", weights_only=False)
    beta = float(ckpt.get("beta", cfg.dpo_beta))
    model = _load_dense(DPO_CKPT, tk, device, with_lora=True)

    # 门槛：LoRA 的 B 必须非零（B 零初始化 ⇒ π_θ == π_ref ⇒ margin 恒 0，
    #        说明这个 checkpoint 压根没训过）
    b_max = 0.0
    for name, p in model.named_parameters():
        if name.endswith("lora_B"):
            b_max = max(b_max, float(p.abs().max().item()))
    assert b_max > 0.0, f"checkpoint 里的 LoRA B 全是 0（max={b_max}）⇒ DPO 没训过"

    pairs = generate_preference_dataset(n_samples=cfg.dpo_n_pairs, seed=cfg.seed)
    m = evaluate_dpo(model, tk, pairs, device, beta, cfg.eval_max_samples)

    # 门槛
    assert m["margin_std"] >= 0.0, "margin 标准差为负"
    assert 0.0 <= m["dpo_acc"] <= 1.0, f"准确率越界：{m['dpo_acc']}"
    assert 0.0 <= m["win_rate"] <= 1.0, f"胜率越界：{m['win_rate']}"

    return {
        "stage": "dpo",
        "metric": "reward_margin",
        "value": m["margin_mean"],
        "margin_mean": m["margin_mean"],
        "margin_std": m["margin_std"],
        "accuracy": m["dpo_acc"],
        "dpo_acc": m["dpo_acc"],
        "win_rate": m["win_rate"],
        "win_rate_per_token": m["win_rate_per_token"],
        "by_strategy": m["by_strategy"],
        "beta": beta,
        "ref_mode": ckpt.get("ref_mode", "?"),
        # checkpoint 里自带的训练期指标，用于交叉核对
        "margin_before": ckpt.get("margin_before"),
        "margin_after": ckpt.get("margin_after"),
        "n_samples": cfg.eval_max_samples,
        "checkpoint": DPO_CKPT,
    }


# ---------------------------------------------------------------------------
# 5) 推理（KV Cache）
# ---------------------------------------------------------------------------
def eval_speedup(tk: "BPETokenizer", device: "torch.device") -> dict:
    """
    推理评估：有/无 KV Cache 的输出一致性与加速比。

    **列名与 GPT-2 线的 `eval_speedup` 完全一致**
    （stage / metric / value / naive_avg_ms / kv_avg_ms / speedup /
    naive_tokens / kv_tokens / n_runs / max_new_tokens）。

    参数：
        tk:     tokenizer
        device: 设备
    返回：
        指标字典
    """
    from inference import benchmark_speed, generate_naive, generate_with_cache, kv_cache_report

    if not os.path.exists(PRETRAIN_DENSE):
        return _skip("speedup",
                     f"{PRETRAIN_DENSE} 不存在（先跑 python train/train_pretrain.py --model dense）")

    cfg = DENSE_CONFIG
    model = _load_dense(PRETRAIN_DENSE, tk, device)

    prompt_ids = tk.encode("Once upon a time, the cat went to the forest. The cat saw a")
    if not prompt_ids:
        return _skip("speedup", "prompt 编码为空")
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # --- 一致性（门槛：不一致直接报错退出）---
    a = generate_naive(model, prompt, max_new_tokens=cfg.max_new_tokens, temperature=0.0)
    b = generate_with_cache(model, prompt, max_new_tokens=cfg.max_new_tokens, temperature=0.0)
    consistent = bool(torch.equal(a, b))
    assert consistent, (
        "有/无 KV Cache 的 greedy 输出不一致 —— 推理路径有问题，拒绝打印好看的数\n"
        f"naive={a.tolist()}\nkv  ={b.tolist()}"
    )

    bench = benchmark_speed(model, prompt, n_runs=3, max_new_tokens=cfg.max_new_tokens,
                            temperature=0.0)
    rep = kv_cache_report(model, prompt_len=prompt.size(1), max_new_tokens=cfg.max_new_tokens)
    return {
        "stage": "speedup",
        "metric": "speedup",
        "value": bench["speedup"],
        "consistent": consistent,
        "naive_avg_ms": bench["naive_avg_ms"],
        "kv_avg_ms": bench["kv_avg_ms"],
        "speedup": bench["speedup"],
        "naive_tokens": bench["naive_tokens"],
        "kv_tokens": bench["kv_tokens"],
        "n_runs": 3,
        "max_new_tokens": cfg.max_new_tokens,
        "prompt_len": int(prompt.size(1)),
        "kv_cache_gqa_bytes": rep["gqa_bytes"],
        "kv_cache_mha_bytes": rep["mha_bytes"],
        "kv_cache_ratio": rep["ratio"],
        "checkpoint": PRETRAIN_DENSE,
    }


STAGES = {
    "pretrain": eval_pretrain,
    "moe": eval_moe,
    "lora": eval_lora,
    "dpo": eval_dpo,
    "speedup": eval_speedup,
}


# ---------------------------------------------------------------------------
# 方向性门槛
# ---------------------------------------------------------------------------
def check_directional(stage: str, r: dict) -> None:
    """
    检查该 stage 的**方向性**门槛（指标该往哪边走）。

    【为什么单独一个函数，不写在 eval_*() 里】
        `eval_*()` 里的是**自洽性门槛**（ppl == exp(loss)、输出一致……），
        跟模型好坏无关 ⇒ 用随机初始化的模型也能过（测试与 init.sh smoke 需要这一点）。
        这里的是**方向性门槛**（胜率必须 > 0.5、margin 必须 > 0、speedup 必须 > 1），
        只有「训好的 checkpoint」才该过 ⇒ 由 `main()` 在评估真实 checkpoint 时统一施加。
        达不到就**报错退出**，不打印一个好看的数。

    参数：
        stage: stage 名
        r:     该 stage 的指标字典
    """
    if r.get("skipped"):
        return

    if stage in ("pretrain", "moe"):
        # perplexity 必须比「在 vocab_size 个候选里瞎猜」好（随机基线 = vocab_size）
        v = r["vocab_size"]
        assert r["perplexity"] < v, \
            f"perplexity={r['perplexity']:.2f} 不小于词表大小 {v} ⇒ 不比随机猜好"
    elif stage == "lora":
        assert r["win_rate"] > THRESH_WIN_RATE, \
            f"偏好胜率 {r['win_rate']:.4f} 未超过随机基线 {THRESH_WIN_RATE}"
    elif stage == "dpo":
        assert r["margin_mean"] > THRESH_MARGIN, \
            f"reward margin 均值 {r['margin_mean']:+.4f} 未大于 {THRESH_MARGIN}"
        assert r["accuracy"] > THRESH_DPO_ACC, \
            f"DPO 判对率 {r['accuracy']:.4f} 未超过随机基线 {THRESH_DPO_ACC}"
    elif stage == "speedup":
        assert r["speedup"] > THRESH_SPEEDUP, \
            f"KV Cache 加速比 {r['speedup']:.3f}x 不大于 {THRESH_SPEEDUP} ⇒ 用 cache 反而更慢"


# ---------------------------------------------------------------------------
# 打印
# ---------------------------------------------------------------------------
def print_stage(name: str, r: dict) -> None:
    """
    按 stage 打印指标。

    参数：
        name: stage 名
        r:    该 stage 的指标字典
    """
    if r.get("skipped"):
        print(f"[eval] stage={name:<9} skip: {r['reason']}")
        return

    if name == "pretrain":
        print(f"[eval] stage=pretrain ppl={r['perplexity']:.3f}  loss={r['loss']:.4f}  "
              f"n_tokens={r['n_tokens']}")
        print(f"[eval]   （batch 均值口径 loss={r['loss_batch_mean']:.4f} "
              f"ppl={r['perplexity_batch_mean']:.3f} —— 与训练日志的 val_loss 同口径）")
    elif name == "moe":
        print(f"[eval] stage=moe     ppl={r['perplexity']:.3f}  loss={r['loss']:.4f}  "
              f"aux_loss={r['aux_loss']:.4f}（coef={r['aux_loss_coef']}）")
        print(f"[eval]   专家负载（{r['n_experts']} 个）= "
              f"{[round(v, 4) for v in r['expert_load_ratio']]}")
        print(f"[eval]   负载 max/min = {r['expert_load_max_min_ratio']:.3f}")
    elif name == "lora":
        print(f"[eval] stage=lora    win_rate={r['win_rate']:.4f}  "
              f"（基线 {r['baseline']:.4f}，Δ={r['delta_win_rate']:+.4f}）")
        print(f"[eval]   长度归一口径 {r['win_rate_per_token']:.4f} "
              f"（基线 {r['baseline_per_token']:.4f}）  n={r['n_samples']}")
        for s, v in sorted(r["by_strategy"].items()):
            b = r["by_strategy_before"].get(s, {})
            print(f"[eval]   {s:<14} n={v['n']:<4} 胜率={v['win_rate']:.3f} "
                  f"（基线 {b.get('win_rate', float('nan')):.3f}）")
    elif name == "dpo":
        print(f"[eval] stage=dpo     margin={r['margin_mean']:+.4f} ± {r['margin_std']:.4f}  "
              f"accuracy={r['accuracy']:.4f}  win_rate={r['win_rate']:.4f}  "
              f"(beta={r['beta']}, ref={r['ref_mode']})")
        print(f"[eval]   长度归一口径 {r['win_rate_per_token']:.4f}  n={r['n_samples']}")
        for s, v in sorted(r["by_strategy"].items()):
            print(f"[eval]   {s:<14} n={v['n']:<4} dpo_acc={v['dpo_acc']:.3f} "
                  f"margin={v['margin_mean']:+.4f} 胜率={v['win_rate']:.3f}")
    elif name == "speedup":
        print(f"[eval] stage=speedup speedup={r['speedup']:.3f}x  "
              f"naive={r['naive_avg_ms']:.2f}ms  kv={r['kv_avg_ms']:.2f}ms  "
              f"consistent={r['consistent']}")
        print(f"[eval]   生成 {r['naive_tokens']} / {r['kv_tokens']} token，"
              f"prompt 长度 {r['prompt_len']}")
        print(f"[eval]   KV Cache 占用 GQA={r['kv_cache_gqa_bytes']:,}B  "
              f"MHA={r['kv_cache_mha_bytes']:,}B  比值={r['kv_cache_ratio']:.3f}")


def main() -> None:
    """评估主流程。"""
    parser = argparse.ArgumentParser(description="Qwen3 精简评估")
    parser.add_argument("--stage", choices=list(STAGES.keys()) + ["all"], default="all",
                        help="要评估哪一段（默认 all）")
    args = parser.parse_args()

    device = get_device()
    tk = _load_tokenizer()
    names = list(STAGES.keys()) if args.stage == "all" else [args.stage]

    print("=" * 72)
    print(f"[eval] Qwen3 精简评估  stage={args.stage}  device={device}")
    print("=" * 72)

    results: Dict[str, dict] = {}
    for name in names:
        r = STAGES[name](tk, device)
        # 方向性门槛：达不到就报错退出（不打印一个好看的数）
        check_directional(name, r)
        results[name] = r
        print_stage(name, r)

    print("=" * 72)
    done = [n for n in names if not results[n].get("skipped")]
    skipped = [n for n in names if results[n].get("skipped")]
    print(f"[eval] 跑了 {len(done)}/{len(names)} 段：{done}")
    if skipped:
        print(f"[eval] 跳过 {len(skipped)} 段：{skipped}")
    print("=" * 72)


if __name__ == "__main__":
    main()

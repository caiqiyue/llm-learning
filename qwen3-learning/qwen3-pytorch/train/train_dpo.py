"""
Qwen3 DPO（Direct Preference Optimization）训练。

用法：
    python train/train_dpo.py                 # 默认：π_ref = SFT 模型（标准 DPO）
    python train/train_dpo.py --ref-mode pretrained   # 消融：π_ref = 预训练模型

【目标函数】
    L_DPO = -E[ log σ( β · ( (logπ_θ(y_w|x) - logπ_ref(y_w|x))
                           - (logπ_θ(y_l|x) - logπ_ref(y_l|x)) ) ) ]

    β 是 KL 惩罚系数（config.dpo_beta）。

    括号里的量就是**隐式奖励的间隔（reward margin）**：
        r_θ(y) = β · ( logπ_θ(y|x) - logπ_ref(y|x) )
        margin = r_θ(y_w) - r_θ(y_l)
    margin > 0 ⇒ 模型相对参考模型更偏爱 chosen。

【★ reference model 的取法 —— 本 Task 最重要的一个技巧】
    **不加载第二份权重**。标准 DPO 里 π_ref 是「SFT 模型」，
    本仓库用三步在同一个模型对象上把它变出来：

        1. 注入 LoRA → 加载 checkpoints/lora.pt（SFT 训好的 base + A/B）
        2. merge_lora：把 SFT 的 ΔW 折进 base  ⇒ 此时模型 == SFT 模型
        3. strip_lora：换回普通 nn.Linear，再 inject 一个**全新的** LoRA
           （B 零初始化 ⇒ π_θ 此刻严格等于 π_ref，margin = 0，loss = log 2）

    之后：
        set_lora_enabled(model, True)   → π_θ（正在训练）
        set_lora_enabled(model, False)  → π_ref（= SFT 模型，base 从没被改过）

    ⚠️ **为什么不能「直接关掉 SFT 的 LoRA」当 π_ref**：
        那样得到的 π_ref 是**预训练模型**，不是 SFT 模型。
        DPO 的 KL 锚点是 SFT 模型；锚到预训练模型会让 DPO 跟 LoRA SFT 互相拉扯
        （SFT 刚把 chosen 的 logprob 抬上去，DPO 的 KL 项又往回压）。
        本仓库把两种做法都实现成 `--ref-mode`，用实测对照来定夺（见笔记 §9）。

【★ 与 GPT-2 线的区别】
    GPT-2 线：SFT → Reward Model → PPO（要显式训一个标量奖励模型 + 采样 + KL 约束）
    Qwen3 线：DPO 直接吃偏好对，**不训 Reward Model**，也不用 PPO、不做采样。

【评估指标】
    margin_mean / margin_std  隐式奖励间隔的均值 / 标准差
    dpo_acc                   margin > 0 的比例（DPO 自己的「判对率」）
    win_rate                  π_θ 下 chosen 总 logprob > rejected 的比例（策略口径）
    win_rate_per_token        长度归一口径（用来监控长度偏置，见笔记 §5）
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

# 把项目根补进 sys.path（脚本在 train/ 子目录里，sys.path[0] 不是项目根）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from config import DENSE_CONFIG
from data.preference_data import generate_preference_dataset
from data.tokenizer import BPETokenizer
from model.common.lora import (
    inject_lora,
    mark_only_lora_trainable,
    merge_lora,
    set_lora_enabled,
    strip_lora,
)
from model.qwen3_dense import Qwen3Dense
from train.train_lora import encode_pair
from utils import get_device, save_checkpoint, set_seed


def dpo_loss(
    model: "torch.nn.Module",
    pair: dict,
    tk: "BPETokenizer",
    beta: float,
    device: torch.device,
) -> Tuple["torch.Tensor", "torch.Tensor", Dict[str, float]]:
    """
    算一对 (chosen, rejected) 的 DPO loss。

    参数：
        model:  已注入 DPO 用 LoRA 的模型（base 已被 merge 成 SFT 模型）
        pair:   {"prompt", "chosen", "rejected", "strategy"}
        tk:     tokenizer
        beta:   KL 惩罚系数
        device: 设备
    返回：
        (loss, margin, stats)
            loss:   scalar tensor，可反向传播
            margin: scalar tensor = β·[(Δw_θ - Δw_ref) - (Δl_θ - Δl_ref)]，
                    **带梯度**（它是 loss 的自变量）
            stats:  标量诊断（lp_w_theta / lp_l_theta / lp_w_ref / lp_l_ref）
    """
    # --- π_θ：开 LoRA，带梯度 ---
    set_lora_enabled(model, True)
    (lp_w_theta, _), (lp_l_theta, _) = encode_pair(model, tk, pair, device)

    # --- π_ref：关 LoRA（= base = SFT 模型），不回传 ---
    set_lora_enabled(model, False)
    with torch.no_grad():
        (lp_w_ref, _), (lp_l_ref, _) = encode_pair(model, tk, pair, device)
    # 还原成 π_θ，准备回传
    set_lora_enabled(model, True)

    # 隐式奖励：r_θ(y) = β · (logπ_θ(y) - logπ_ref(y))
    r_w = beta * (lp_w_theta - lp_w_ref)
    r_l = beta * (lp_l_theta - lp_l_ref)
    margin = r_w - r_l
    loss = -F.logsigmoid(margin).mean()
    stats = {
        "lp_w_theta": float(lp_w_theta.item()),
        "lp_l_theta": float(lp_l_theta.item()),
        "lp_w_ref": float(lp_w_ref.item()),
        "lp_l_ref": float(lp_l_ref.item()),
    }
    return loss, margin.mean(), stats


@torch.no_grad()
def evaluate_dpo(
    model: "torch.nn.Module",
    tk: "BPETokenizer",
    pairs: List[dict],
    device: torch.device,
    beta: float,
    max_samples: int,
) -> dict:
    """
    评估 DPO 训练效果。

    指标：
        margin_mean / margin_std  隐式奖励间隔的均值 / 标准差
        dpo_acc                   margin > 0 的比例
        win_rate                  π_θ 下 chosen 总 logprob > rejected 的比例
        win_rate_per_token        长度归一口径
        by_strategy               按「变差策略」分组的 dpo_acc

    参数：
        model:       模型
        tk:          tokenizer
        pairs:       偏好对列表
        device:      设备
        beta:        KL 惩罚系数
        max_samples: 最多评估多少对
    返回：
        指标字典
    """
    was_training = model.training
    model.eval()
    rows = []
    for pair in pairs[:max_samples]:
        set_lora_enabled(model, True)
        (lp_w, n_w), (lp_l, n_l) = encode_pair(model, tk, pair, device)
        set_lora_enabled(model, False)
        (lp_w_ref, _), (lp_l_ref, _) = encode_pair(model, tk, pair, device)
        set_lora_enabled(model, True)
        margin = float(beta * ((lp_w - lp_w_ref) - (lp_l - lp_l_ref)).item())
        rows.append((margin, float(lp_w.item()), float(lp_l.item()),
                     n_w, n_l, pair.get("strategy", "?")))

    n = max(1, len(rows))
    margins = [r[0] for r in rows]
    mean = sum(margins) / n
    var = sum((m - mean) ** 2 for m in margins) / n
    by_strategy = {}
    for s in sorted({r[5] for r in rows}):
        sub = [r for r in rows if r[5] == s]
        by_strategy[s] = {
            "n": len(sub),
            "dpo_acc": sum(1 for r in sub if r[0] > 0) / max(1, len(sub)),
            "margin_mean": sum(r[0] for r in sub) / max(1, len(sub)),
            "win_rate": sum(1 for r in sub if r[1] > r[2]) / max(1, len(sub)),
        }
    out = {
        "margin_mean": mean,
        "margin_std": var ** 0.5,
        "dpo_acc": sum(1 for m in margins if m > 0) / n,
        "win_rate": sum(1 for r in rows if r[1] > r[2]) / n,
        "win_rate_per_token": sum(
            1 for r in rows if r[1] / max(1, r[3]) > r[2] / max(1, r[4])) / n,
        "by_strategy": by_strategy,
    }
    if was_training:
        model.train()
    return out


def print_eval(tag: str, m: dict) -> None:
    """
    打印一组 DPO 评估指标。

    参数：
        tag: 标签（如 BEFORE / AFTER）
        m:   evaluate_dpo 的返回值
    """
    print(f"[dpo] {tag:<7} margin={m['margin_mean']:+.4f} ± {m['margin_std']:.4f}  "
          f"dpo_acc={m['dpo_acc']:.3f}  win_rate={m['win_rate']:.3f}  "
          f"win_rate_per_token={m['win_rate_per_token']:.3f}")
    for s, v in sorted(m["by_strategy"].items()):
        print(f"[dpo]   {s:<14} n={v['n']:<4} dpo_acc={v['dpo_acc']:.3f}  "
              f"margin={v['margin_mean']:+.4f}  win_rate={v['win_rate']:.3f}")


def build_policy(
    ckpt_path: str,
    tk: "BPETokenizer",
    device: torch.device,
    ref_mode: str,
) -> "torch.nn.Module":
    """
    构造 DPO 的策略模型（**同时也是 π_ref**，靠开关分支切换）。

    ref_mode:
        "sft"         π_ref = LoRA SFT 后的模型（标准 DPO）
        "pretrained"  π_ref = 纯预训练模型（消融：跳过 SFT 直接 DPO）

    返回：
        注入好「全新 LoRA」的模型；此时 B=0 ⇒ π_θ == π_ref，margin = 0
    """
    cfg = DENSE_CONFIG
    model = Qwen3Dense(vocab_size=tk.vocab_size, cfg=cfg, dropout=0.0).to(device)

    if ref_mode == "sft":
        # --- 1) 先注入 LoRA，才能加载 lora.pt（它的 state_dict 是注入后的键名）---
        inject_lora(model, cfg.lora_target_modules, r=cfg.lora_r, alpha=cfg.lora_alpha)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=True)
        # --- 2) 把 SFT 的 ΔW 折进 base ⇒ 模型 == SFT 模型 ---
        merged = merge_lora(model)
        # --- 3) 换回普通 Linear，再注入一个**全新**的 LoRA 给 DPO 用 ---
        stripped = strip_lora(model)
        inject_lora(model, cfg.lora_target_modules, r=cfg.lora_r, alpha=cfg.lora_alpha)
        print(f"[dpo] π_ref = SFT 模型：merge {merged} 个模块 / strip {stripped} 个模块，"
              f"base 已含 SFT 的 ΔW")
    elif ref_mode == "pretrained":
        # π_ref = 纯预训练模型：不加载 lora.pt，直接在预训练权重上注入新 LoRA
        ckpt = torch.load(os.path.join("checkpoints", "pretrain_dense.pt"),
                          map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=True)
        inject_lora(model, cfg.lora_target_modules, r=cfg.lora_r, alpha=cfg.lora_alpha)
        print(f"[dpo] π_ref = 预训练模型（消融：跳过 SFT）")
    else:
        raise ValueError(f"ref_mode 只能是 'sft' 或 'pretrained'，实得 {ref_mode}")

    return model


def main() -> None:
    """DPO 训练主流程。"""
    parser = argparse.ArgumentParser(description="Qwen3 DPO 训练")
    parser.add_argument("--max-iters", type=int, default=None, help="覆盖 config 的 dpo_max_iters")
    parser.add_argument("--ref-mode", type=str, default="sft", choices=["sft", "pretrained"],
                        help="π_ref 用 SFT 模型（默认）还是预训练模型（消融）")
    parser.add_argument("--beta", type=float, default=None, help="覆盖 config 的 dpo_beta")
    args = parser.parse_args()

    cfg = DENSE_CONFIG
    max_iters = args.max_iters if args.max_iters is not None else cfg.dpo_max_iters
    beta = args.beta if args.beta is not None else cfg.dpo_beta

    set_seed(cfg.seed)
    device = get_device()
    print(f"[dpo] device={device} max_iters={max_iters} beta={beta} ref_mode={args.ref_mode}")

    tk = BPETokenizer(os.path.join("data", "vocab.json"), os.path.join("data", "merges.txt"))

    # --- 构造 π_θ / π_ref ---
    ckpt_path = os.path.join("checkpoints", "lora.pt")
    if args.ref_mode == "sft" and not os.path.exists(ckpt_path):
        print(f"[dpo] 找不到 {ckpt_path}，请先跑：python train/train_lora.py")
        return
    model = build_policy(ckpt_path, tk, device, args.ref_mode)

    # --- 只训 LoRA（其余全部冻结）---
    trainable = mark_only_lora_trainable(model)
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[dpo] trainable params = {n_trainable:,} / {n_total:,} ({n_trainable/n_total:.2%})")

    pairs = generate_preference_dataset(n_samples=cfg.dpo_n_pairs, seed=cfg.seed)
    print(f"[dpo] preference pairs = {len(pairs)}")

    # --- 训练前的基线：B=0 ⇒ π_θ == π_ref ⇒ margin 必为 0 ---
    before = evaluate_dpo(model, tk, pairs, device, beta, cfg.eval_max_samples)
    print_eval("BEFORE", before)
    assert abs(before["margin_mean"]) < 1e-4, \
        f"初始 margin 应为 0（B 零初始化 ⇒ π_θ == π_ref），实得 {before['margin_mean']}"

    optimizer = torch.optim.AdamW(trainable, lr=cfg.dpo_lr, weight_decay=cfg.weight_decay)

    # --- 训练 ---
    model.train()
    step = 0
    margins: List[float] = []
    losses: List[float] = []
    while step < max_iters:
        for pair in pairs:
            if step >= max_iters:
                break
            loss, margin, _ = dpo_loss(model, pair, tk, beta, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            optimizer.step()

            margins.append(float(margin.item()))
            losses.append(float(loss.item()))
            if step % cfg.log_every == 0:
                recent = sum(margins[-cfg.log_every:]) / max(1, len(margins[-cfg.log_every:]))
                print(f"[dpo] step {step:>4}/{max_iters} loss {loss.item():.4f} "
                      f"margin {margin.item():+.4f} (avg{cfg.log_every} {recent:+.4f})")
            step += 1

    # --- 评估 ---
    after = evaluate_dpo(model, tk, pairs, device, beta, cfg.eval_max_samples)
    print_eval("AFTER", after)
    print(f"[dpo] margin 首{cfg.log_every}均 {sum(margins[:cfg.log_every])/max(1,len(margins[:cfg.log_every])):+.4f} "
          f"→ 末{cfg.log_every}均 {sum(margins[-cfg.log_every:])/max(1,len(margins[-cfg.log_every:])):+.4f}")

    # --- 落盘 ---
    path = save_checkpoint(
        {"model_state": model.state_dict(),
         "vocab_size": tk.vocab_size,
         "ref_mode": args.ref_mode,
         "beta": beta,
         "metrics": {"before": before, "after": after},
         "margin_before": before["margin_mean"],
         "margin_after": after["margin_mean"],
         "dpo_acc": after["dpo_acc"],
         "config": cfg.to_dict()},
        os.path.join("checkpoints", "dpo.pt"),
    )
    print(f"[dpo] saved -> {path}")


if __name__ == "__main__":
    main()

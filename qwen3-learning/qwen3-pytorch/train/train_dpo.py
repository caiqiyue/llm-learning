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

【★ MoE 版（--model moe，feat-032）】
    feat-031 只把 **LoRA** 侧接到了 MoE，**DPO 侧还没在 MoE 上跑过**。
    这一版做三件事：
      ① 支持 `--model dense|moe`：建 `Qwen3Dense` / `Qwen3MoE`，
         分别加载 `checkpoints/lora.pt`（dense 的 SFT）与 `checkpoints/lora_moe.pt`（MoE 的 SFT），
         分别落到 `checkpoints/dpo.pt` / `checkpoints/dpo_moe.pt`；
      ② **π_ref 的注入集合必须与当初 SFT 时完全一致** ——
         SFT 权重的 state_dict 键名取决于「当初注入了哪些模块」。
         MoE 的 SFT（feat-031）用的是「attention + 专家 FFN」
         （`cfg.lora_target_modules_expert`）⇒ DPO 侧必须注同一组，
         否则 `load_state_dict(strict=True)` 当场红。
         脚本会**读 checkpoint 里记的 `target_modules` 与本地配置对一次**，
         对不上直接报错，而不是让它悄悄错配。
      ③ 验证 DPO 的核心机制在 MoE 上**依然成立**：
         `merge_lora → strip_lora → 重新 inject_lora` 三步在**专家 FFN** 上是否等价？
         （merge 把 ΔW 折进专家的 base 权重，而**路由器是冻结的** ⇒ 路由不变 ⇒ 理论上等价，
           但必须用可运行断言钉住，见 `tests/test_dpo_moe.py`。）

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
from model.qwen3_moe import Qwen3MoE
from train.train_lora import count_expert_lora_params, encode_pair, pick_config
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


def lora_targets_for(model_kind: str, cfg) -> tuple:
    """
    决定「这一条线」该往哪些模块注入 LoRA。

    ★ **必须与当初 SFT 时用的集合完全一致** ——
      SFT 权重的 state_dict 键名是由「注入集合」决定的
      （例如 `blocks.0.attn.q_proj.lora_A` vs `blocks.0.moe.experts.1.gate_proj.lora_A`），
      集合不一致 ⇒ `load_state_dict(strict=True)` 直接报「缺键 / 多键」。

    参数：
        model_kind: "dense" 或 "moe"
        cfg:        配置对象
    返回：
        注入集合元组
    """
    if model_kind == "dense":
        return tuple(cfg.lora_target_modules)
    if model_kind == "moe":
        # ★ MoE 的 SFT（feat-031）用的是「attention + 专家 FFN」
        return tuple(cfg.lora_target_modules_expert)
    raise ValueError(f"未知的 model 类型：{model_kind}")


def check_lora_compat(ckpt: dict, targets: tuple, cfg, ckpt_path: str) -> None:
    """
    核对 checkpoint 里记的 LoRA 配置与本地配置是否一致。

    ★ 为什么要有这一步：DPO 加载的是 **SFT 之后**的 LoRA 权重，
      而 `strict=True` 的 `load_state_dict` 只能告诉你「键对不上」，
      **不会告诉你「你注入的模块集合选错了」**——
      键完全一致但语义错配（例如把「专家 FFN 的 LoRA」当成「attention 的 LoRA」）
      是可能的，因为 LoRA 的键名自带了模块全名。
      这里提前把「注入集合 / r / alpha」三件事核对一遍，对不上就**当场报错**，
      而不是让它悄悄错配。

    参数：
        ckpt:      torch.load 出来的 checkpoint 字典
        targets:   本地准备注入的模块集合
        cfg:       配置对象
        ckpt_path: checkpoint 路径（只用于报错信息）
    返回：
        None。配置对不上时抛 ValueError。
    """
    info = ckpt.get("lora")
    if not isinstance(info, dict):
        return  # 老 checkpoint 没有这串元数据，跳过核对
    saved_targets = tuple(info.get("target_modules", ()))
    if saved_targets and saved_targets != tuple(targets):
        raise ValueError(
            f"LoRA 注入集合不匹配！\n"
            f"  checkpoint {ckpt_path} 记录的是：{list(saved_targets)}\n"
            f"  本地准备注入的是：{list(targets)}\n"
            f"  ⇒ 请用与当初 SFT 相同的 --model / 注入集合重跑"
        )
    for key, local in (("r", cfg.lora_r), ("alpha", cfg.lora_alpha)):
        if key in info and info[key] != local:
            raise ValueError(
                f"LoRA {key} 不匹配：checkpoint 记录 {info[key]}，本地配置 {local}"
            )


def build_policy(
    model_kind: str,
    ckpt_path: str,
    tk: "BPETokenizer",
    device: torch.device,
    ref_mode: str,
) -> "torch.nn.Module":
    """
    构造 DPO 的策略模型（**同时也是 π_ref**，靠开关分支切换）。

    model_kind:
        "dense"  Qwen3Dense，注入 attention 版 LoRA
        "moe"    Qwen3MoE，注入「attention + 专家 FFN」版 LoRA

    ref_mode:
        "sft"         π_ref = LoRA SFT 后的模型（标准 DPO）
        "pretrained"  π_ref = 纯预训练模型（消融：跳过 SFT 直接 DPO）

    返回：
        注入好「全新 LoRA」的模型；此时 B=0 ⇒ π_θ == π_ref，margin = 0

    【★ 为什么 merge → strip 在 MoE 上依然成立】
        merge 把 ΔW 折进 base；strip 换回普通 Linear；重新 inject 一个 B=0 的新 LoRA。
        整个过程**不碰路由器**（router 既不是 LoRA 目标，也不参与 merge），
        而路由只由 router 的输出决定 ⇒ 折进去的 ΔW **不会改变路由** ⇒
        「合并后的模型」与「开着 SFT LoRA 的模型」在数值上等价。
        （这条等价性由 tests/test_dpo_moe.py 第 2 条用可运行断言钉住。）
    """
    cfg = pick_config(model_kind)
    targets = lora_targets_for(model_kind, cfg)

    if model_kind == "dense":
        model = Qwen3Dense(vocab_size=tk.vocab_size, cfg=cfg, dropout=0.0).to(device)
        pretrain_path = os.path.join("checkpoints", "pretrain_dense.pt")
    else:
        model = Qwen3MoE(vocab_size=tk.vocab_size, cfg=cfg, dropout=0.0).to(device)
        pretrain_path = os.path.join("checkpoints", "pretrain_moe.pt")

    if ref_mode == "sft":
        # --- 1) 先注入 LoRA，才能加载 lora*.pt（它的 state_dict 是注入后的键名）---
        inject_lora(model, targets, r=cfg.lora_r, alpha=cfg.lora_alpha)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        check_lora_compat(ckpt, targets, cfg, ckpt_path)
        model.load_state_dict(ckpt["model_state"], strict=True)
        # --- 2) 把 SFT 的 ΔW 折进 base ⇒ 模型 == SFT 模型 ---
        merged = merge_lora(model)
        # --- 3) 换回普通 Linear，再注入一个**全新**的 LoRA 给 DPO 用 ---
        stripped = strip_lora(model)
        inject_lora(model, targets, r=cfg.lora_r, alpha=cfg.lora_alpha)
        print(f"[dpo] π_ref = SFT 模型：merge {merged} 个模块 / strip {stripped} 个模块，"
              f"base 已含 SFT 的 ΔW（注入集合 {list(targets)}）")
    elif ref_mode == "pretrained":
        # π_ref = 纯预训练模型：不加载 lora*.pt，直接在预训练权重上注入新 LoRA
        ckpt = torch.load(pretrain_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=True)
        inject_lora(model, targets, r=cfg.lora_r, alpha=cfg.lora_alpha)
        print(f"[dpo] π_ref = 预训练模型（消融：跳过 SFT，{pretrain_path}）")
    else:
        raise ValueError(f"ref_mode 只能是 'sft' 或 'pretrained'，实得 {ref_mode}")

    return model


def print_expert_dead_report(model: "torch.nn.Module", step: int, tag: str) -> None:
    """
    打印「专家侧 LoRA 参数的梯度状态」诊断（feat-032）。

    ★ 为什么 DPO 里也要看这个（feat-031 的结论在 DPO 里同样成立）：
      路由器是**冻结**的 ⇒ 这一步没被路由到的专家**根本不参与前向**
      ⇒ 它的 LoRA 参数 `grad is None` ⇒ AdamW **整个跳过**
      ⇒ 连 weight decay 都没有 ⇒ 参数**一动不动**。
      ⇒ 往**每个**专家注入 LoRA 时，总有一部分参数是**白注入**的。

    参数：
        model: 模型
        step:  当前步号（区分「第 0 步 A 的零梯度」这个暂时现象）
        tag:   标签
    """
    total, none_p, dead_p = count_expert_lora_params(model, step)
    if not total:
        return
    print(f"[dpo-moe] expert LoRA params ({tag}): total={total:,}; "
          f"grad_none={none_p:,} (unrouted expert -> unchanged); "
          f"grad_zero={dead_p:,} (no loss signal -> weight decay only); "
          f"white_injection={(none_p + dead_p) / total:.1%}")


def main() -> None:
    """DPO 训练主流程。"""
    parser = argparse.ArgumentParser(description="Qwen3 DPO 训练")
    parser.add_argument("--max-iters", type=int, default=None, help="覆盖 config 的 dpo_max_iters")
    parser.add_argument("--ref-mode", type=str, default="sft", choices=["sft", "pretrained"],
                        help="π_ref 用 SFT 模型（默认）还是预训练模型（消融）")
    parser.add_argument("--beta", type=float, default=None, help="覆盖 config 的 dpo_beta")
    parser.add_argument("--model", choices=["dense", "moe"], default="dense",
                        help="在哪个模型上做 DPO（feat-032 新增 moe）")
    args = parser.parse_args()

    model_kind = args.model
    cfg = pick_config(model_kind)
    max_iters = args.max_iters if args.max_iters is not None else cfg.dpo_max_iters
    beta = args.beta if args.beta is not None else cfg.dpo_beta

    # ★ set_seed 必须在**建模型之前**（建模型与 LoRA 的 A 矩阵初始化都会消耗全局 RNG，
    #   种子设置的先后会改变 A 的初值 ⇒ 改变训练轨迹与最终指标，见 train_lora.py 同款注释）
    set_seed(cfg.seed)
    device = get_device()
    print(f"[dpo] model={model_kind} device={device} max_iters={max_iters} "
          f"beta={beta} ref_mode={args.ref_mode}")

    tk = BPETokenizer(os.path.join("data", "vocab.json"), os.path.join("data", "merges.txt"))

    # --- 构造 π_θ / π_ref ---
    if model_kind == "dense":
        ckpt_path = os.path.join("checkpoints", "lora.pt")
        save_path = os.path.join("checkpoints", "dpo.pt")
    else:
        ckpt_path = os.path.join("checkpoints", "lora_moe.pt")
        save_path = os.path.join("checkpoints", "dpo_moe.pt")
    if args.ref_mode == "sft" and not os.path.exists(ckpt_path):
        print(f"[dpo] 找不到 {ckpt_path}，请先跑：python train/train_lora.py --model {model_kind}")
        return
    model = build_policy(model_kind, ckpt_path, tk, device, args.ref_mode)

    # --- 只训 LoRA（其余全部冻结：embedding / Norm / router / 专家 base 都冻）---
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

            # --- MoE 专属：专家侧 LoRA 参数的梯度状态（feat-032）---
            if model_kind == "moe" and (step == 0 or step == max_iters - 1):
                print_expert_dead_report(model, step, f"step {step}")

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
         "model_kind": model_kind,
         "ref_mode": args.ref_mode,
         "beta": beta,
         "lora": {"r": cfg.lora_r, "alpha": cfg.lora_alpha,
                  "target_modules": list(lora_targets_for(model_kind, cfg))},
         "metrics": {"before": before, "after": after},
         "margin_before": before["margin_mean"],
         "margin_after": after["margin_mean"],
         "dpo_acc": after["dpo_acc"],
         "config": cfg.to_dict()},
        save_path,
    )
    print(f"[dpo] saved -> {path}")


if __name__ == "__main__":
    main()

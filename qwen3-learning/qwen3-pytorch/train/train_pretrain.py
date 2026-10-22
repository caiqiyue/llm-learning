"""
Qwen3 预训练脚本（唯一带 --model 开关的脚本）。

用法：
    python train/train_pretrain.py --model dense
    python train/train_pretrain.py --model moe

与 GPT-2 线的三处刻意偏离（见 开发计划.md §0.3）：
    1. 用 argparse 而不是写死参数
    2. 超参全部从 config.py 的 DENSE_CONFIG / MOE_CONFIG 读
    3. 模型 forward 恒定返回 4 元组 (logits, loss, kv_cache, extras)，
       MoE 的 aux_loss 从 extras.get("aux_loss") 取 —— 训练脚本不需要 if is_moe 分支去猜返回结构

【为什么顶部要补 sys.path】
    脚本放在 train/ 子目录里，`python train/train_pretrain.py` 时
    `sys.path[0]` 是 train/，找不到项目根下的 config.py / data 包
    → ModuleNotFoundError。与 GPT-2 线「脚本放在目录顶层」不同，这里必须显式补。
"""

import argparse
import math
import os
import sys
import time

# 把项目根补进 sys.path（见模块 docstring）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import DENSE_CONFIG, MOE_CONFIG
from data.pretrain_data import make_pretrain_loader
from data.tokenizer import BPETokenizer
from model.qwen3_dense import Qwen3Dense
from model.qwen3_moe import Qwen3MoE
from utils import get_device, save_checkpoint, set_seed


def build_model(model_kind: str, vocab_size: int, dropout: float = 0.1):
    """
    按名字构造模型。

    参数：
        model_kind: "dense" 或 "moe"
        vocab_size: 词表大小
        dropout:    dropout 比例
    返回：
        nn.Module
    """
    if model_kind == "dense":
        return Qwen3Dense(vocab_size=vocab_size, cfg=DENSE_CONFIG, dropout=dropout)
    if model_kind == "moe":
        return Qwen3MoE(vocab_size=vocab_size, cfg=MOE_CONFIG, dropout=dropout)
    raise ValueError(f"未知的 --model: {model_kind}（只支持 dense / moe）")


def lr_at(step: int, cfg) -> float:
    """
    warmup + cosine 学习率调度。

        step < warmup_iters        → 线性升温：lr * (step+1) / warmup
        warmup <= step < decay_iters→ cosine 衰减：min_lr + 0.5*(lr-min_lr)*(1+cos(pi*p))
        step >= lr_decay_iters     → 恒定在 min_lr

    参数：
        step: 当前迭代步数（0 起）
        cfg:  Qwen3Config
    返回：
        学习率
    """
    if step < cfg.warmup_iters:
        return cfg.learning_rate * (step + 1) / cfg.warmup_iters
    if step >= cfg.lr_decay_iters:
        return cfg.min_lr
    # cosine 从 learning_rate 降到 min_lr
    progress = (step - cfg.warmup_iters) / max(1, cfg.lr_decay_iters - cfg.warmup_iters)
    return cfg.min_lr + 0.5 * (cfg.learning_rate - cfg.min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, loader, max_batches: int = 20) -> float:
    """
    在验证集上估计平均 loss。

    参数：
        model:       模型
        loader:      验证集 DataLoader
        max_batches: 最多评估多少个 batch（控制 CPU 上的评估耗时）
    返回：
        平均 loss（float）
    """
    model.eval()
    total, count = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        _, loss, _, _ = model(x, targets=y)
        total += float(loss.item())
        count += 1
    model.train()
    return total / max(1, count)


def main() -> None:
    """训练主流程。"""
    parser = argparse.ArgumentParser(description="Qwen3 预训练")
    parser.add_argument("--model", choices=["dense", "moe"], default="dense",
                        help="要训练哪个模型")
    parser.add_argument("--max-iters", type=int, default=None,
                        help="覆盖 config 里的 max_iters（默认用 config 值）")
    args = parser.parse_args()

    cfg = DENSE_CONFIG if args.model == "dense" else MOE_CONFIG
    max_iters = args.max_iters if args.max_iters is not None else cfg.max_iters

    set_seed(cfg.seed)
    device = get_device()
    print(f"[pretrain] model={args.model} device={device} max_iters={max_iters}")

    # --- tokenizer ---
    tk = BPETokenizer(os.path.join("data", "vocab.json"), os.path.join("data", "merges.txt"))
    vocab_size = tk.vocab_size
    print(f"[pretrain] vocab_size = {vocab_size}")

    # --- 数据 ---
    train_loader = make_pretrain_loader("train", cfg.batch_size, cfg.block_size, cfg.seed)
    val_loader = make_pretrain_loader("val", cfg.batch_size, cfg.block_size, cfg.seed)
    print(f"[pretrain] train_batches={len(train_loader)} val_batches={len(val_loader)}")

    # --- 模型 ---
    model = build_model(args.model, vocab_size, cfg.dropout).to(device)
    print(f"[pretrain] params = {model.num_params():,}")

    # --- 优化器：只优化 requires_grad 的参数 ---
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # --- 训练循环 ---
    model.train()
    step = 0
    t0 = time.time()
    first_loss = None
    while step < max_iters:
        for x, y in train_loader:
            if step >= max_iters:
                break
            x, y = x.to(device), y.to(device)

            for g in optimizer.param_groups:
                g["lr"] = lr_at(step, cfg)

            _, loss, _, extras = model(x, targets=y)

            # MoE：总 loss = lm_loss + aux_loss_coef * aux_loss
            # dense 的 extras 是空 dict，extras.get("aux_loss") 返回 None → 不需要 if 分支
            aux = extras.get("aux_loss")
            if aux is not None:
                loss = loss + cfg.aux_loss_coef * aux

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            if first_loss is None:
                first_loss = float(loss.item())
            if step % cfg.log_every == 0:
                print(f"[pretrain] step {step:>4}/{max_iters} loss {loss.item():.4f} "
                      f"lr {optimizer.param_groups[0]['lr']:.2e} "
                      f"elapsed={time.time() - t0:.1f}s")
                if aux is not None:
                    load = extras.get("expert_load_last")
                    print(f"[pretrain]   aux_loss {float(aux.item()):.4f} "
                          f"expert_load {[round(float(v), 2) for v in load]}")
            step += 1

    # --- 收尾评估 ---
    val_loss = estimate_loss(model, val_loader, cfg.eval_max_batches)
    val_ppl = math.exp(min(20.0, val_loss))
    print(f"[pretrain] initial_loss={first_loss:.4f} final val_loss={val_loss:.4f} "
          f"val_ppl={val_ppl:.2f} elapsed={time.time() - t0:.1f}s")

    # --- 落盘 ---
    path = save_checkpoint(
        {"model_state": model.state_dict(), "cfg_name": args.model,
         "vocab_size": vocab_size, "config": cfg.to_dict()},
        os.path.join("checkpoints", f"pretrain_{args.model}.pt"),
    )
    print(f"[pretrain] saved -> {path}")


if __name__ == "__main__":
    main()

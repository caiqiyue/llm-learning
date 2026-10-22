"""
Qwen3 LoRA 微调。

用法：
    python train/train_lora.py

【路线说明】
    路线已定为 LoRA，**不做全参 SFT**（全参 SFT 在 GPT-2 线已有覆盖）。
    目标：在偏好数据的 prompt 上做自监督式微调 —— 给定 prompt，训练模型补全 chosen。

【流程】
    1. 加载 checkpoints/pretrain_dense.pt
    2. inject_lora(model, ["q_proj", "v_proj"], r=4, alpha=8)
    3. 冻结 base（含 embedding / Norm / MLP / lm_head），只训 A / B
    4. 评估：偏好胜率（chosen 的 logprob 高于 rejected 的样本比例）

【★ 为什么回复部分要单独编码、单独计分】
    LoRA 与 DPO 都需要「只对回复部分算 logprob」：
        logπ(y|x) = Σ_{t ∈ 回复位置} log p(y_t | x, y_<t)
    所以 prompt 段与 response 段**分别编码再拼接**，
    这样「回复从哪个 token 开始」是精确已知的，不用在拼接后的字符串里再切一次。
    拼接用的是 data/preference_data.py::format_prompt_response 的同一套分隔符
    （`### Question:\\n...\\n\\n### Answer:\\n`），但**分开编码**，
    避免边界处 BPE 把分隔符和回复第一个字符合并掉。

【★ 关于「注入即无损」】
    LoRA 的 B 矩阵零初始化 ⇒ 刚注入时模型输出与预训练权重**逐元素相等**，
    所以训练起点就是预训练模型本身，不会因为注入而突然变差。
    这一点由 tests/test_lora.py 第 4 条守住。

【★ 评估口径（别过度解读）】
    合成偏好数据只有 5 个 prompt × 5 个 chosen，**训练集与评估集同分布**，
    所以打印出来的胜率是「训练内指标」，**不能当成泛化能力**。
    它的用途是：证明「LoRA 确实让模型更偏向 chosen」。
"""

import argparse
import os
import sys
from typing import List, Optional, Tuple

# 把项目根补进 sys.path（脚本在 train/ 子目录里，sys.path[0] 不是项目根）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from config import DENSE_CONFIG
from data.preference_data import generate_preference_dataset
from data.tokenizer import BPETokenizer
from model.common.lora import inject_lora, mark_only_lora_trainable, set_lora_enabled
from model.qwen3_dense import Qwen3Dense
from utils import get_device, save_checkpoint, set_seed

# prompt 与 response 的分隔模板（与 data/preference_data.py::format_prompt_response 一致）
PROMPT_TEMPLATE = "### Question:\n{prompt}\n\n### Answer:\n"


def encode_prompt_response(
    tk: "BPETokenizer",
    prompt: str,
    response: str,
) -> Tuple[List[int], int]:
    """
    把 prompt 与 response 分别编码后拼接，返回 (token 序列, 回复起始下标)。

    分别编码而不是编码拼接后的字符串：边界处 BPE 可能把分隔符和回复第一个字符合并，
    那时「回复从哪开始」就不再精确了。

    参数：
        tk:       tokenizer
        prompt:   问题
        response: 回答
    返回：
        (ids, n_prompt)：完整 token 列表、回复部分的起始下标
    """
    prefix_ids = tk.encode(PROMPT_TEMPLATE.format(prompt=prompt))
    resp_ids = tk.encode(response)
    return prefix_ids + resp_ids, len(prefix_ids)


def build_training_example(
    tk: "BPETokenizer",
    prompt: str,
    response: str,
    block_size: int,
) -> Optional[Tuple["torch.Tensor", "torch.Tensor", int]]:
    """
    构造一条「只对回复部分计分」的训练样本。

    模型内部的 loss 是 `cross_entropy(logits, targets)`（**不做内部 shift**），
    所以 targets[i] 必须是「位置 i 预测出来的 token」，即 ids[i+1]。
    prompt 段与最后一个位置都用 -100 忽略。

    参数：
        tk:         tokenizer
        prompt:     问题
        response:   回答
        block_size: 最大上下文长度
    返回：
        (inputs, targets, n_prompt)；样本无效（太短）时返回 None
    """
    ids, n_prompt = encode_prompt_response(tk, prompt, response)
    ids = ids[:block_size]
    n_prompt = min(n_prompt, len(ids) - 1)      # 至少留 1 个回复 token
    if len(ids) < 2 or n_prompt >= len(ids) - 1:
        return None
    inputs = torch.tensor(ids, dtype=torch.long).unsqueeze(0)      # (1, L)
    targets = torch.full_like(inputs, -100)
    # 位置 n_prompt-1 .. L-2 的 target 是 ids[n_prompt] .. ids[L-1]
    targets[0, n_prompt - 1: len(ids) - 1] = inputs[0, n_prompt:]
    return inputs, targets, n_prompt


def encode_pair(
    model: "torch.nn.Module",
    tk: "BPETokenizer",
    pair: dict,
    device: torch.device,
    block_size: int = DENSE_CONFIG.block_size,
) -> Tuple[Tuple["torch.Tensor", int], Tuple["torch.Tensor", int]]:
    """
    算一对 (chosen, rejected) 在给定模型下的序列 logprob（**只对回复部分**）。

    参数：
        model:      模型
        tk:         tokenizer
        pair:       {"prompt", "chosen", "rejected"}
        device:     设备
        block_size: 最大上下文长度
    返回：
        ((lp_chosen, n_chosen), (lp_rejected, n_rejected))
        lp 是 (1,) tensor（总 logprob），n 是被计分的 token 数
    """
    out = []
    for key in ("chosen", "rejected"):
        ids, n_prompt = encode_prompt_response(tk, pair["prompt"], pair[key])
        ids = ids[:block_size]
        n_prompt = min(n_prompt, len(ids) - 1)
        if len(ids) < 2 or n_prompt >= len(ids) - 1:
            out.append((torch.zeros(1, device=device), 0))
            continue
        idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(idx)[0]                                    # (1, L, V)
        logp = F.log_softmax(logits.float(), dim=-1)
        # 位置 p 的 logits 预测 ids[p+1] → 回复 token ids[n_prompt:] 由位置 n_prompt-1..L-2 预测
        positions = torch.arange(n_prompt - 1, len(ids) - 1, device=device)
        targets = idx[0, n_prompt:]
        picked = logp[0, positions, :].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        out.append((picked.sum().unsqueeze(0), int(picked.numel())))
    return out[0], out[1]


@torch.no_grad()
def evaluate_pairs(
    model: "torch.nn.Module",
    tk: "BPETokenizer",
    pairs: List[dict],
    device: torch.device,
    max_samples: int,
) -> dict:
    """
    评估「偏好胜率」并拆开看。

    指标：
        win_rate          chosen 的**总** logprob 高于 rejected 的比例（DPO 的口径）
        win_rate_per_token 每 token 平均 logprob 的胜率（**长度归一**后的口径）
        mean_logp_chosen  chosen 的平均总 logprob
        by_strategy       按「变差策略」分组的胜率（长短不同的回复会让两种口径打架）

    【为什么要两种口径】
        总 logprob 与回复长度正相关 —— 回复越短，总 logprob 越接近 0（越大），
        所以「总 logprob 胜率」会**系统性偏袒短回复**（too_short 策略必输）；
        长度归一后又反过来偏袒「可预测」的文本（重复、常见词）。
        **DPO 用 π_ref 相减正好抵消这两种静态偏差** —— 这正是 DPO 减参考模型的原因。

    参数：
        model:       模型
        tk:          tokenizer
        pairs:       偏好对列表
        device:      设备
        max_samples: 最多评估多少对
    返回：
        指标字典
    """
    was_training = model.training
    model.eval()
    rows = []
    for pair in pairs[:max_samples]:
        (lp_w, n_w), (lp_l, n_l) = encode_pair(model, tk, pair, device)
        rows.append((float(lp_w.item()), n_w, float(lp_l.item()), n_l, pair.get("strategy", "?")))

    n = max(1, len(rows))
    wins = sum(1 for lw, _, ll, _, _ in rows if lw > ll)
    wins_norm = sum(1 for lw, nw, ll, nl, _ in rows
                    if lw / max(1, nw) > ll / max(1, nl))
    by_strategy = {}
    for s in sorted({r[4] for r in rows}):
        sub = [r for r in rows if r[4] == s]
        by_strategy[s] = {
            "n": len(sub),
            "win_rate": sum(1 for lw, _, ll, _, _ in sub if lw > ll) / max(1, len(sub)),
            "win_rate_per_token": sum(1 for lw, nw, ll, nl, _ in sub
                                      if lw / max(1, nw) > ll / max(1, nl)) / max(1, len(sub)),
            "len_chosen": sum(r[1] for r in sub) / max(1, len(sub)),
            "len_rejected": sum(r[3] for r in sub) / max(1, len(sub)),
        }
    out = {
        "win_rate": wins / n,
        "win_rate_per_token": wins_norm / n,
        "mean_logp_chosen": sum(r[0] for r in rows) / n,
        "by_strategy": by_strategy,
    }
    if was_training:
        model.train()
    return out


def print_eval(tag: str, m: dict) -> None:
    """
    打印一组评估指标。

    参数：
        tag: 标签（如 BEFORE / AFTER）
        m:   evaluate_pairs 的返回值
    """
    print(f"[lora] {tag:<7} win_rate={m['win_rate']:.3f}  "
          f"win_rate_per_token={m['win_rate_per_token']:.3f}  "
          f"mean_logp(chosen)={m['mean_logp_chosen']:.2f}")
    for s, v in sorted(m["by_strategy"].items()):
        print(f"[lora]   {s:<14} n={v['n']:<4} 总logprob胜率={v['win_rate']:.3f}  "
              f"长度归一胜率={v['win_rate_per_token']:.3f}  "
              f"len(chosen)={v['len_chosen']:.1f} len(rejected)={v['len_rejected']:.1f}")


def main() -> None:
    """LoRA 微调主流程。"""
    parser = argparse.ArgumentParser(description="Qwen3 LoRA 微调")
    parser.add_argument("--max-iters", type=int, default=None,
                        help="覆盖 config 的 lora_max_iters")
    args = parser.parse_args()

    cfg = DENSE_CONFIG
    max_iters = args.max_iters if args.max_iters is not None else cfg.lora_max_iters

    set_seed(cfg.seed)
    device = get_device()
    print(f"[lora] device={device} max_iters={max_iters}")

    tk = BPETokenizer(os.path.join("data", "vocab.json"), os.path.join("data", "merges.txt"))

    # --- 加载预训练权重（必须在注入 LoRA 之前加载：注入后模块名会变）---
    ckpt_path = os.path.join("checkpoints", "pretrain_dense.pt")
    if not os.path.exists(ckpt_path):
        print(f"[lora] 找不到 {ckpt_path}，请先跑：python train/train_pretrain.py --model dense")
        return
    model = Qwen3Dense(vocab_size=tk.vocab_size, cfg=cfg, dropout=0.0).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    print(f"[lora] loaded {ckpt_path} (vocab_size={tk.vocab_size})")

    # --- 注入 LoRA + 只留 A/B 可训 ---
    replaced = inject_lora(model, cfg.lora_target_modules, r=cfg.lora_r,
                           alpha=cfg.lora_alpha, dropout=cfg.lora_dropout)
    trainable = mark_only_lora_trainable(model)
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[lora] injected {len(replaced)} modules: {replaced}")
    print(f"[lora] trainable params = {n_trainable:,} / {n_total:,} ({n_trainable/n_total:.2%})")

    # --- 偏好数据 ---
    pairs = generate_preference_dataset(n_samples=cfg.dpo_n_pairs, seed=cfg.seed)
    print(f"[lora] preference pairs = {len(pairs)}")

    # --- 微调前的基线（注入后 B=0，模型行为 == 预训练模型）---
    before = evaluate_pairs(model, tk, pairs, device, cfg.eval_max_samples)
    print_eval("BEFORE", before)

    optimizer = torch.optim.AdamW(trainable, lr=cfg.lora_lr, weight_decay=cfg.weight_decay)

    # --- 训练：给定 prompt，补全 chosen（loss 只算回复部分）---
    model.train()
    step = 0
    first_loss = None
    while step < max_iters:
        for pair in pairs:
            if step >= max_iters:
                break
            example = build_training_example(tk, pair["prompt"], pair["chosen"], cfg.block_size)
            if example is None:
                continue
            inputs, targets, _ = example
            inputs, targets = inputs.to(device), targets.to(device)

            _, loss, _, _ = model(inputs, targets=targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            optimizer.step()

            if first_loss is None:
                first_loss = float(loss.item())
            if step % cfg.log_every == 0:
                print(f"[lora] step {step:>4}/{max_iters} loss {loss.item():.4f} "
                      f"lr {optimizer.param_groups[0]['lr']:.2e}")
            step += 1

    # --- 评估：偏好胜率 ---
    after = evaluate_pairs(model, tk, pairs, device, cfg.eval_max_samples)
    print_eval("AFTER", after)
    print(f"[lora] initial_loss={first_loss:.4f}")
    print(f"[lora] 胜率 {before['win_rate']:.3f} -> {after['win_rate']:.3f} "
          f"(随机基线 0.5)，长度归一口径 "
          f"{before['win_rate_per_token']:.3f} -> {after['win_rate_per_token']:.3f}")

    # --- 落盘 ---
    path = save_checkpoint(
        {"model_state": model.state_dict(),
         "vocab_size": tk.vocab_size,
         "lora": {"r": cfg.lora_r, "alpha": cfg.lora_alpha,
                  "target_modules": list(cfg.lora_target_modules)},
         "metrics": {"before": before, "after": after},
         "win_rate": after["win_rate"], "base_win_rate": before["win_rate"],
         "config": cfg.to_dict()},
        os.path.join("checkpoints", "lora.pt"),
    )
    print(f"[lora] saved -> {path}")


if __name__ == "__main__":
    main()

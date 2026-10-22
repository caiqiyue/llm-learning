"""
奖励模型 (Reward Model) 训练脚本（PyTorch 端）。
[P5 实现]

目标：
  - 加载 SFT 阶段的 GPT-2 backbone（不带 lm_head 的概率分布预测能力）
  - 在 GPT-2 上**复用并替换** lm_head 那一路 → 改成 scalar value head
    （在 reward 任务中，我们只关心整个序列的"总得分"是几，而不是下一个 token 是什么）
  - 用 Bradley-Terry pairwise ranking loss 训练 RM：
        loss = -logsigmoid(r_chosen - r_rejected).mean()
  - 在 preference 验证集上算 pairwise accuracy
  - 保存 checkpoints/reward.pt（含 SFT backbone + value head）

与 train_sft.py / train_pretrain.py 风格保持一致：
  - 线性脚本（无 if __name__ 包装）
  - 详细中文 docstring
  - 用 `value_head` 字段在 GPT-2 上挂一个 `Linear(n_embd -> 1)`，复用
    `model.forward(... targets=None)` 返回 4 元组里的 `value`

详细 spec：design §4.3 Reward Model、gpt2_training_explained.md Part 4、
implementation-plan.md P5。
"""

# -*- coding: utf-8 -*-

import os
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from utils import set_seed, get_device, load_checkpoint, save_checkpoint
from model.gpt2 import GPT2
from data.preference_data import make_preference_loader
from data.tokenizer import CharTokenizer


# ---------------------------------------------------------------------------
# (1) 固定随机种子 + 选 device
# ---------------------------------------------------------------------------
set_seed(config.seed)
device = get_device()
print(f"[reward] device = {device}")


# ---------------------------------------------------------------------------
# (2) 加载 tokenizer（只为拿到 vocab_size）
# ---------------------------------------------------------------------------
vocab_path = os.path.join("data", "vocab.txt")
v = r'E:\ai-learning\05-llm-fundamentals\gpt-learning\gpt-learning-pytorch\data\vocab.txt'
tokenizer = CharTokenizer(vocab_path=v)
vocab_size = tokenizer.vocab_size
print(f"[reward] vocab_size = {vocab_size}")


# ---------------------------------------------------------------------------
# (3) 实例化 GPT-2（与 SFT 同结构），再挂 scalar value head
# ---------------------------------------------------------------------------
model = GPT2(
    vocab_size=vocab_size,
    n_layer=config.n_layer,
    n_head=config.n_head,
    n_embd=config.n_embd,
    n_inner=config.n_inner,
    block_size=config.block_size,
    dropout=config.dropout,
    bias=config.bias,
)

# 关键：奖励模型不需要 vocab_size 路线的输出头（lm_head），
# 而是要 scalar 输出（每个样本一个 reward 标量）。
# 我们在 GPT-2 上挂一个 `Linear(n_embd -> 1)` 作为 value head，
# forward 路径走的是 GPT-2 内部已经写好的 `value` 字段（取最后 token hidden）。
model.value_head = nn.Linear(model.n_embd, 1, bias=True)
# 显式初始化（与 GPT-2 风格一致：weight ~ N(0, 0.02), bias = 0）
nn.init.normal_(model.value_head.weight, std=0.02)
nn.init.zeros_(model.value_head.bias)
model.to(device)
print(f"[reward] num_params = {model.num_params():,}")


# ---------------------------------------------------------------------------
# (4) 关键：从 SFT checkpoint 加载 backbone 权重
#     - 过滤掉 SFT 没有的 `value_head.*` 字段，让其保持新鲜初始化
#     - SFT 的 lm_head 会被一起加载（lm_head = tok_emb.weight tying），但 RM 不使用它
# ---------------------------------------------------------------------------
# 路径用相对路径（相对本脚本所在目录），不写死绝对路径 —— 换机器/换目录不会失效
sft_ckpt_path = os.path.join("checkpoints", "sft.pt")
print(f"[reward] loading SFT checkpoint from {sft_ckpt_path}")
ckpt = load_checkpoint(sft_ckpt_path)
state = {
    k: v
    for k, v in ckpt["model_state"].items()
    if not k.startswith("value_head.")
}
missing, unexpected = model.load_state_dict(state, strict=False)
print(
    f"[reward] SFT checkpoint loaded "
    f"(missing={len(missing)}, unexpected={len(unexpected)})"
)
# 期望：missing 里就是 'value_head.weight' / 'value_head.bias' 两条
if any("value_head" in k for k in missing):
    print(f"[reward] (expected) value_head.* not in SFT ckpt; will be fresh-initialized")


# ---------------------------------------------------------------------------
# (5) 优化器：AdamW（与 SFT 同款；betas=(0.9, 0.95)）
# ---------------------------------------------------------------------------
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=config.rm_lr,
    weight_decay=config.weight_decay,
    betas=(0.9, 0.95),
)


# ---------------------------------------------------------------------------
# (6) 学习率调度：线性 warmup + cosine decay
# ---------------------------------------------------------------------------
def get_lr(it: int) -> float:
    """
    线性 warmup + cosine decay 到 min_lr。
    - 0 <= it < warmup：lr = rm_lr * (it+1) / warmup
    - warmup <= it <= rm_max_iters：cosine 衰减
    - it > rm_max_iters：min_lr
    """
    base_lr = config.rm_lr
    min_lr = config.min_lr
    warmup = min(config.warmup_iters, max(1, config.rm_max_iters // 5))
    decay_end = config.rm_max_iters
    if it < warmup:
        return base_lr * (it + 1) / max(1, warmup)
    if it > decay_end:
        return min_lr
    decay_ratio = (it - warmup) / max(1, decay_end - warmup)
    decay_ratio = min(1.0, max(0.0, decay_ratio))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


# ---------------------------------------------------------------------------
# (7) DataLoader：preference (chosen_input_ids, rejected_input_ids)
#     - 来自 data/preference_data.py::make_preference_loader
#     - 内部已 padding 到本 batch 最大长度
# ---------------------------------------------------------------------------
train_loader = make_preference_loader(
    split="train",
    batch_size=config.rm_batch_size,
    seed=config.seed,
)
val_loader = make_preference_loader(
    split="val",
    batch_size=config.rm_batch_size,
    seed=config.seed,
)
print(
    f"[reward] train batches = {len(train_loader)}, val batches = {len(val_loader)}"
)


# ---------------------------------------------------------------------------
# (8) 训练主循环
# ---------------------------------------------------------------------------
model.train()
iter_num = 0
best_val_acc = 0.0
best_val_loss = float("inf")
t0 = time.time()

LOG_EVERY = 50
EVAL_EVERY = 200

while iter_num < config.rm_max_iters:
    # (a) 设置 LR
    lr = get_lr(iter_num)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    # (b) 拉一个 batch
    try:
        batch = next(train_iter)
    except (NameError, StopIteration):
        train_iter = iter(train_loader)
        batch = next(train_iter)

    chosen_ids = batch["chosen_input_ids"].to(device)     # (B, L)
    rejected_ids = batch["rejected_input_ids"].to(device) # (B, L)

    # 防御：preference loader 把每个 batch 内的序列 pad 到该 batch 的最大长度，
    # 可能超过 block_size=128。RM 训练时把超过的尾部截断。
    if chosen_ids.size(1) > config.block_size:
        chosen_ids = chosen_ids[:, : config.block_size]
    if rejected_ids.size(1) > config.block_size:
        rejected_ids = rejected_ids[:, : config.block_size]

    # (c) 前向：拿到 chosen / rejected 各自的 scalar reward
    #     GPT-2 在 value_head 不为 None 时返回 4 元组 (logits, loss, kv, value)
    #     RM 不需要算 LM loss，所以传 targets=None
    #     【关键形状变化】
    #       chosen_ids:   (B, L≤128)     整数 token id
    #       r_chosen:     (B,)            每个样本一个标量 reward
    #       r_rejected:   (B,)
    #     前向过程内部：
    #       chosen_ids → tok_emb → ... → ln_f → value_head(last_hidden) → (B,)
    #       last_hidden = x[:, -1, :]   (B, C)  取最后 token 的 hidden
    #       value = value_head(last_hidden).squeeze(-1)  (B,)
    _, _, _, r_chosen = model(chosen_ids, targets=None) #32个分数
    _, _, _, r_rejected = model(rejected_ids, targets=None)

    # (d) Bradley-Terry pairwise ranking loss
    #     loss = -log sigmoid(r_chosen - r_rejected).mean()
    #     r_chosen - r_rejected: (B,)
    #     logsigmoid(...): (B,)
    #     loss: scalar
    #     含义：让 r_chosen > r_rejected；如果搞反，会被大梯度强烈推回
    loss = -F.logsigmoid(r_chosen - r_rejected).mean()
    """
    第一步：计算两个奖励分数的差值
\[
d=r_{chosen}-r_{rejected}=3-1=2
\]

差值越大，意味着奖励模型越倾向于 Chosen。
第二步：通过 Sigmoid 转成偏好概率
\[
P(chosen \succ rejected)=\sigma(d)=\frac{1}{1+e^{-d}}
\]

代入 \(d=2\)：
\[
\sigma(2)\approx0.8808
\]

可以理解为：模型预测 Chosen 优于 Rejected 的概率为 88.08%。
第三步：对概率取负对数
\[
L=-\log(\sigma(d))
\]

\[
L=-\log(0.8808)\approx0.1269
\]
    """

    # (e) 反向 + clip + step
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()

    # (f) 周期性打印
    if iter_num % LOG_EVERY == 0:
        dt = time.time() - t0
        with torch.no_grad():
            acc = (r_chosen > r_rejected).float().mean().item()
        print(
            f"[reward] iter {iter_num:4d} | loss {loss.item():.4f} "
            f"| acc {acc:.3f} | lr {lr:.2e} | {dt:.1f}s"
        )

    # (g) 周期性 eval + 保存 best
    if iter_num > 0 and iter_num % EVAL_EVERY == 0:
        model.eval()
        with torch.no_grad():
            val_correct = 0
            val_total = 0
            val_loss_sum = 0.0
            val_loss_count = 0
            for vb in val_loader:
                vc = vb["chosen_input_ids"].to(device)
                vr = vb["rejected_input_ids"].to(device)
                if vc.size(1) > config.block_size:
                    vc = vc[:, : config.block_size]
                if vr.size(1) > config.block_size:
                    vr = vr[:, : config.block_size]
                _, _, _, r_c = model(vc, targets=None)
                _, _, _, r_r = model(vr, targets=None)
                val_correct += (r_c > r_r).sum().item()
                val_total += vc.size(0)
                bt_loss = -F.logsigmoid(r_c - r_r).mean()
                val_loss_sum += bt_loss.item() * vc.size(0)
                val_loss_count += vc.size(0)
            val_acc = val_correct / max(val_total, 1)
            val_loss = val_loss_sum / max(val_loss_count, 1)
        print(
            f"[reward] === eval === val_loss {val_loss:.4f} | val_acc {val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            save_checkpoint(
                {
                    "model_state": model.state_dict(),  # 包含 value_head
                    "config": {
                        k: v
                        for k, v in vars(config).items()
                        if not k.startswith("_")
                    },
                    "iter_num": iter_num,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                "checkpoints/reward.pt",
            )
            print(
                f"[reward] saved best RM checkpoint (val_acc={val_acc:.4f}, val_loss={val_loss:.4f})"
            )
        model.train()

    iter_num += 1


# ---------------------------------------------------------------------------
# (9) 训练结束：跑最终 eval，并保存 final
# ---------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    val_correct = 0
    val_total = 0
    val_loss_sum = 0.0
    val_loss_count = 0
    for vb in val_loader:
        vc = vb["chosen_input_ids"].to(device)
        vr = vb["rejected_input_ids"].to(device)
        if vc.size(1) > config.block_size:
            vc = vc[:, : config.block_size]
        if vr.size(1) > config.block_size:
            vr = vr[:, : config.block_size]
        _, _, _, r_c = model(vc, targets=None)
        _, _, _, r_r = model(vr, targets=None)
        val_correct += (r_c > r_r).sum().item()
        val_total += vc.size(0)
        bt_loss = -F.logsigmoid(r_c - r_r).mean()
        val_loss_sum += bt_loss.item() * vc.size(0)
        val_loss_count += vc.size(0)
    final_val_acc = val_correct / max(val_total, 1)
    final_val_loss = val_loss_sum / max(val_loss_count, 1)

if final_val_acc > best_val_acc:
    best_val_acc = final_val_acc
    best_val_loss = final_val_loss

# 总是保存 final（覆盖 best）
save_checkpoint(
    {
        "model_state": model.state_dict(),
        "config": {
            k: v for k, v in vars(config).items() if not k.startswith("_")
        },
        "iter_num": iter_num,
        "val_acc": best_val_acc,
        "val_loss": best_val_loss,
    },
    "checkpoints/reward.pt",
)
total_wall = time.time() - t0
print(
    f"[reward] done. iter={iter_num} | val_acc={final_val_acc:.4f} "
    f"| val_loss={final_val_loss:.4f} | best_val_acc={best_val_acc:.4f} "
    f"| best_val_loss={best_val_loss:.4f} | wall={total_wall:.1f}s"
)

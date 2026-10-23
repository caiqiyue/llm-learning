"""
GPT-2 监督微调 (SFT) 脚本。
在预训练权重基础上，用 (prompt, response) 数据微调；prompt 部分 loss mask 掉。

与 train_pretrain.py 的核心差异：
  - 初始化：加载 pretrain checkpoint（而非从零）
  - 数据：SFT 合成数据 (prompt, response)，prompt 部分 label=-100
  - 迭代数：sft_max_iters（通常 1000，比 pretrain 短）
  - 学习率：config.sft_lr（spec 写 3e-4，与 pretrain 同；但 iter 少所以不会灾难性遗忘）
  - 损失：标准 Causal LM 交叉熵，labels 里 -100 已被 cross_entropy(ignore_index=-100) 自动忽略

工作流：
  1) set_seed / device
  2) 加载 tokenizer
  3) 实例化 GPT2（dropout=sft 时也保持小一点）
  4) 加载 pretrain.pt 权重
  5) AdamW + warmup+cosine LR schedule
  6) 训练循环：每个 iter 拉一个 batch（dict）→ 取 input_ids/labels → forward → loss
  7) 每 200 iter 跑 val；变好就保存 best
  8) 训练完保存 final
  9) 打印 5 条 (prompt, response) 采样

详细 spec：design §4.2 SFT、gpt2_training_explained.md Part 3、implementation-plan P4。
"""

# -*- coding: utf-8 -*-

import os                       # 路径
import math                     # cosine 衰减
import time                     # 估时

import torch

import config
from utils import set_seed, get_device, load_checkpoint, save_checkpoint
from model.gpt2 import GPT2
from data.sft_data import make_sft_loader
from data.tokenizer import CharTokenizer
from generate import generate_text, generate_response


# ---------------------------------------------------------------------------
# (1) 固定随机种子 + 选 device
# ---------------------------------------------------------------------------
set_seed(config.seed)
device = get_device()
print(f"[sft] device = {device}")


# ---------------------------------------------------------------------------
# (2) 加载 tokenizer
#     词表路径用相对路径（相对本脚本所在目录），不写死绝对路径 —— 换机器/换目录不会失效
# ---------------------------------------------------------------------------
vocab_path = os.path.join("data", "vocab.txt")
tokenizer = CharTokenizer(vocab_path=vocab_path)
vocab_size = tokenizer.vocab_size
print(f"[sft] vocab_size = {vocab_size}")


# ---------------------------------------------------------------------------
# (3) 实例化 GPT2（与 train_pretrain 同样的超参）
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
model.to(device)
print(f"[sft] num_params = {model.num_params():,}")


# ---------------------------------------------------------------------------
# (4) 关键：从 pretrain 加载权重
#     - SFT 是预训练的延续，不是从零开始
#     - 加载后用同一 optimizer 继续 fine-tune
# ---------------------------------------------------------------------------
pretrain_ckpt_path = "checkpoints/pretrain.pt"
print(f"[sft] loading pretrain checkpoint from {pretrain_ckpt_path}")
ckpt = load_checkpoint(pretrain_ckpt_path)
model.load_state_dict(ckpt["model_state"])
pretrain_val_loss = ckpt.get("val_loss", None)
if pretrain_val_loss is not None:
    print(f"[sft] pretrain val_loss = {pretrain_val_loss:.4f}")
else:
    print(f"[sft] pretrain checkpoint loaded (no val_loss in ckpt)")


# ---------------------------------------------------------------------------
# (5) 优化器：AdamW（与 pretrain 同款；betas=(0.9, 0.95)）
# ---------------------------------------------------------------------------
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.sft_lr,
    weight_decay=config.weight_decay,
    betas=(0.9, 0.95),
)


# ---------------------------------------------------------------------------
# (6) 学习率调度：warmup + cosine（与 pretrain 同结构，但 max_iters 是 sft_max_iters）
# ---------------------------------------------------------------------------
def get_lr(it: int) -> float:
    """
    线性 warmup + cosine decay 到 min_lr。
    - 0 <= it < warmup_iters：lr = sft_lr * (it+1) / warmup_iters
    - warmup_iters <= it <= sft_max_iters：cosine 衰减
    - it > sft_max_iters：min_lr
    """
    base_lr = config.sft_lr
    min_lr = config.min_lr
    # SFT 短训，warmup 比例压低（不超过 1/5 总步数）
    warmup = min(config.warmup_iters, max(1, config.sft_max_iters // 5))
    decay_end = config.sft_max_iters
    if it < warmup:
        return base_lr * (it + 1) / max(1, warmup)
    if it > decay_end:
        return min_lr
    decay_ratio = (it - warmup) / max(1, decay_end - warmup)
    decay_ratio = min(1.0, max(0.0, decay_ratio))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


# ---------------------------------------------------------------------------
# (7) DataLoader
#     - make_sft_loader 是 dict-of-tensors 形式（PyTorch SFTDataset）
#     - 每个 batch 形如 {input_ids: (B, L), labels: (B, L), prompt_len: (B,)}
# ---------------------------------------------------------------------------
train_loader = make_sft_loader(
    split="train",
    batch_size=config.batch_size,
    seed=config.seed,
)
val_loader = make_sft_loader(
    split="val",
    batch_size=config.batch_size,
    seed=config.seed,
)#除了数据集结构不一致，其余的就和预训练差不多了
print(f"[sft] train batches = {len(train_loader)}, val batches = {len(val_loader)}")


# ---------------------------------------------------------------------------
# (8) 训练主循环
#     - iter_num 是全局步数（0-indexed）
#     - train_iter 用 try/except 实现"epoch 内自循环"
# ---------------------------------------------------------------------------
model.train()
iter_num = 0
best_val_loss = float("inf")
t0 = time.time()

LOG_EVERY = 50
EVAL_EVERY = 200

while iter_num < config.sft_max_iters:
    # (a) 设置当前 lr
    lr = get_lr(iter_num)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    # (b) 拿一个 batch
    try:
        batch = next(train_iter)
    except (NameError, StopIteration):
        train_iter = iter(train_loader)
        batch = next(train_iter)

    x = batch["input_ids"].to(device) # 32 108
    y = batch["labels"].to(device).to(device)  # prompt 部分 label=-100

    # (c) forward + loss
    #     - GPT2.forward 在 value_head=None 时返回 3 元组 (logits, loss, kv_cache)
    #     - cross_entropy(ignore_index=-100) 自动跳过 prompt 位置
    #     【关键形状变化 + 关键创新：Prompt-Masked Loss】
    #       x:  (B, L)        L = 该 batch 的最大序列长度（不是固定的 128！SFT 数据是变长）
    #       y:  (B, L)        L 同上；prompt 部分 = -100，response 部分 = 真实 token id
    #       logits: (B, L, V) 每个位置对词表的分数
    #       loss:   scalar     cross_entropy 平均值（只算 label != -100 的位置）
    #     SFT 与 Pretrain 的关键区别：
    #       - 序列长度可变（response 长度不一）
    #       - y 里 prompt 部分是 -100，cross_entropy 自动忽略
    #       - 起始权重是 pretrain.pt，不是随机
    logits, loss, _ = model(x, targets=y)

    # (d) backward + clip + step
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()

    # (e) 周期性打印
    if iter_num % LOG_EVERY == 0:
        dt = time.time() - t0
        print(
            f"[sft] iter {iter_num:4d} | loss {loss.item():.4f} | lr {lr:.2e} | {dt:.1f}s"
        )

    # (f) 周期性 eval + checkpoint
    if iter_num > 0 and iter_num % EVAL_EVERY == 0:
        model.eval()
        with torch.no_grad():
            val_losses = []
            for vb in val_loader:
                vx = vb["input_ids"].to(device)
                vy = vb["labels"].to(device)
                _, vl, _ = model(vx, targets=vy)
                val_losses.append(vl.item())
            val_loss = sum(val_losses) / max(1, len(val_losses))
        val_ppl = math.exp(val_loss)
        print(
            f"[sft] === eval === val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                {
                    "model_state": model.state_dict(),
                    "config": {k: v for k, v in vars(config).items() if not k.startswith("_")},
                    "iter_num": iter_num,
                    "val_loss": val_loss,
                },
                "checkpoints/sft.pt",
            )
            print(f"[sft] saved best checkpoint (val_loss={val_loss:.4f})")
        model.train()

    iter_num += 1


# ---------------------------------------------------------------------------
# (9) 训练完跑最终 eval 并保存
# ---------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    val_losses = []
    for vb in val_loader:
        vx = vb["input_ids"].to(device)
        vy = vb["labels"].to(device)
        _, vl, _ = model(vx, targets=vy)
        val_losses.append(vl.item())
    final_val_loss = sum(val_losses) / max(1, len(val_losses))
    final_val_ppl = math.exp(final_val_loss)

if final_val_loss < best_val_loss:
    best_val_loss = final_val_loss

# 始终保存 final SFT checkpoint
save_checkpoint(
    {
        "model_state": model.state_dict(),
        "config": {k: v for k, v in vars(config).items() if not k.startswith("_")},
        "iter_num": iter_num,
        "val_loss": best_val_loss,
    },
    "checkpoints/sft.pt",
)
total_wall = time.time() - t0
print(
    f"[sft] done. iter={iter_num} | val_loss={final_val_loss:.4f} "
    f"| val_ppl={final_val_ppl:.2f} | best_val_loss={best_val_loss:.4f} "
    f"| wall={total_wall:.1f}s"
)


# ---------------------------------------------------------------------------
# (10) 用训练好的 SFT 模型采 5 条 (prompt, response) 样本，肉眼检查质量
#     - prompt 形如 "<bos> What is a cat? <sep>"
#     - 生成 response 直至 <eos> 或 max_new_tokens
# ---------------------------------------------------------------------------
try:
    model.eval()
    print("[sft] === samples (prompt -> response) ===")
    # 用 SFT 数据集的前 5 条 prompt
    from data.sft_data import generate_sft_dataset
    samples = generate_sft_dataset(n_samples=5, seed=config.eval_seed)
    # 防御：top_k 不能超过 vocab_size（否则 multinomial 报错）
    safe_top_k = min(config.top_k, tokenizer.vocab_size - 1)
    for s in samples:
        # 拼成 SFT 训练时同样的 prompt 格式
        prompt_text = f"<bos> {s['prompt']} <sep> "
        try:
            # 【feat-027】用 generate_response：按 token id 个数切出回答，
            #   不用 len(prompt_text) 切解码后的字符串（UNK 解码成 5 字符 "<unk>" 会错位）。
            resp = generate_response(
                model,
                tokenizer,
                prompt_text,
                max_new_tokens=64,
                mode="naive",  # 不依赖 KV cache 也能跑
                temperature=config.temperature,
                top_k=safe_top_k,
                top_p=config.top_p,
            )
            print(f"  prompt:   {s['prompt']!r}")
            print(f"  response: {resp!r}")
            print()
        except Exception as e:
            print(f"  [sample skipped] {type(e).__name__}: {e}")
            break
except Exception as e:
    print(f"[sft] sample step skipped: {type(e).__name__}: {e}")

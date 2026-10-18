"""
GPT-2 预训练脚本。
线性执行：load_data → build_model → loop → save。
无 if __name__ 包装，方便逐行 debugger。

工作流：
    1) 固定随机种子（set_seed）
    2) 选 device（CPU 跑得动）
    3) 加载 tokenizer（仅用于拿 vocab_size）
    4) 实例化 GPT2（n_layer/n_head/n_embd/n_inner/block_size 全部来自 config）
    5) 实例化 AdamW（lr / weight_decay / betas 来自 config）
    6) 构造带 warmup+cosine 的学习率调度器
    7) 准备 train / val DataLoader
    8) 训练主循环：每个 iter 拉一个 batch → forward → loss → backward → clip → step
    9) 每 200 iter 跑一次 val，算 PPL；变好就保存 best checkpoint
   10) 训练完再保存一次 final checkpoint

详细 spec：design §4.1 预训练、gpt2_training_explained.md Part 2、implementation-plan P3。
"""

# -*- coding: utf-8 -*-

# (1) import：标准库 / 第三方 / 内部
import math                       # cosine 衰减需要
import time                       # 估时
import os                         # 路径处理
import torch                      # 主框架

import config                     # 超参集中地
from utils import set_seed, get_device, save_checkpoint
from model.gpt2 import GPT2
from data.pretrain_data import make_pretrain_loader
from data.tokenizer import CharTokenizer
from generate import generate_text  # 周期采样用


# (2) 固定随机种子：保证可复现
#     - Python random / NumPy / PyTorch CPU 三处都要 seed
#     - CUDA 不可用时不影响（utils.set_seed 内部有 try/except 风格的 if 守卫）
set_seed(config.seed)


# (3) 选 device，cpu
device = get_device()
print(f"[pretrain] device = {device}")


# (4) 加载 tokenizer：仅用来确认 vocab_size
#     - vocab.txt 由 data/build_vocab.py 提前生成
#     - 注：CharTokenizer(vocab_path) 是唯一支持的形式（与 design §6.2 / dataset_construction §6 对齐）

# 与 data/ 同级目录的 vocab.txt；保持相对路径方便项目搬迁  'data\\vocab.txt' 
vocab_path = os.path.join("data", "vocab.txt")
v = r'E:\ai-learning\05-llm-fundamentals\gpt-learning\gpt-learning-pytorch\data\vocab.txt'
tokenizer = CharTokenizer(vocab_path=v)
vocab_size = tokenizer.vocab_size  # 45 个词
print(f"[pretrain] vocab_size = {vocab_size}")


# (5) 实例化 GPT2：所有超参都从 config 读
#     - 参数量约 0.8M（vocab=45 时 794,496；vocab=64 时 ~0.83M）
model = GPT2(
    vocab_size=vocab_size,#45
    n_layer=config.n_layer,#4
    n_head=config.n_head,#4
    n_embd=config.n_embd,#128
    n_inner=config.n_inner,#512
    block_size=config.block_size,#128
    dropout=config.dropout,#0.1
    bias=config.bias,#False
)
model.to(device)
print(f"[pretrain] num_params = {model.num_params():,}")


# (6) 优化器：AdamW
#     - betas=(0.9, 0.95) 是 nanoGPT 风格（GPT-3 原文用 0.9/0.95，PyTorch 默认 0.9/0.999 偏保守）
#     - weight_decay=0.1 把 L2 正则从梯度里解耦出来（更稳）
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
    betas=(0.9, 0.95),
)


# (7) 学习率调度：线性 warmup + cosine decay 到 min_lr
#     - 阶段 1：前 warmup_iters 步，lr 从 0 线性升到 learning_rate（防止训练初梯度爆炸）
#     - 阶段 2：warmup 之后到 lr_decay_iters 之间，cosine 平滑降到 min_lr
#     - 阶段 3：超过 lr_decay_iters 之后，固定为 min_lr
def get_lr(it: int) -> float:
    """
    给定 iter 编号，返回目标学习率。
    1) 线性 warmup
    2) cosine decay
    3) 收尾 min_lr
    """
    # 1) linear warmup
    if it < config.warmup_iters:
        return config.learning_rate * (it + 1) / max(1, config.warmup_iters)
    # 2) 超过 decay 总长则直接给 min_lr
    if it > config.lr_decay_iters:
        return config.min_lr
    # 3) cosine decay
    decay_ratio = (it - config.warmup_iters) / max(1, config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


# (8) DataLoader：复用 data/pretrain_data.py 的工厂函数
#生辰 8993个样本，然后直接合成一个大段，拼接在一起，然后全部tokenizer话，然后按照blocksize 切分，这样可能语义可能别切掉了，但是样本量多了很多
#     - make_pretrain_loader 会按 seed 稳定切 90/10 训练/验证
#     - 返回 DataLoader，每个 batch 是 (x, y) tensor 对，y = x 整体左移一位
train_loader = make_pretrain_loader(
    split="train",
    batch_size=config.batch_size,
    block_size=config.block_size,
    seed=config.seed,
)
val_loader = make_pretrain_loader(
    split="val",
    batch_size=config.batch_size,
    block_size=config.block_size,
    seed=config.seed,
)
print(f"[pretrain] train batches = {len(train_loader)}, val batches = {len(val_loader)}")


# (9) 训练主循环
#     - iter_num 是全局步数（0-indexed）
#     - train_iter 用 try/except 实现"epoch 内自循环"：到头后重新 iter
model.train()
iter_num = 0
best_val_loss = float("inf")
t0 = time.time()

while iter_num < config.max_iters:
    # (a) 根据 iter_num 设置当前 lr（写入 optimizer 的 param_groups）
    lr = get_lr(iter_num)#5.9999e-06
    for pg in optimizer.param_groups:#[{'params': [...], 'lr': 0.0003, 'betas': (...), 'eps': 1e-08, 'weight_decay': 0.1, 'amsgrad': False, 'maximize': False, 'foreach': None, 'capturable': False, 'differentiable': False, 'fused': None, 'decoupled_weight_decay': True}]
        pg["lr"] = lr

    # (b) 拿一个 batch：第一次时 train_iter 还没建，触发 NameError → 进入 except 分支建迭代器
    try:
        x, y = next(train_iter)
    except (NameError, StopIteration):
        train_iter = iter(train_loader)
        x, y = next(train_iter)#x 和 y 都是 32,128

    x = x.to(device)
    y = y.to(device)
    # 此时 x, y 都是 (B, T) 的 LongTensor
    # B = batch_size = 32
    # T = block_size = 128

    # (c) forward + loss
    #     - GPT2.forward 在 value_head=None 时返回 3 元组 (logits, loss, kv_cache)
    #     - 训练时 kv_cache=None，所以第三项是 None
    #     【关键形状变化】
    #       x:  (B, T)        = (32, 128)    token ids
    #       y:  (B, T)        = (32, 128)    target ids (错位 1 位)
    #       logits: (B, T, V)  = (32, 128, 45) 每个位置对词表的分数
    #       loss:   scalar (1,)                cross_entropy 平均值
    logits, loss, _ = model(x, targets=y)#logits 32,128,45   loss 3.8284   

    # (d) 反向传播 + 优化器 step
    optimizer.zero_grad(set_to_none=True)  # set_to_none 省显存
    loss.backward()                        # 反向传播算梯度
    # 梯度裁剪：防止 Transformer 训练中偶发的梯度尖峰
    # 公式: if ||g|| > max_norm: g = g * max_norm / ||g||
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()                       # AdamW 应用梯度

    # (e) 周期性打印训练 loss
    if iter_num % 50 == 0:
        dt = time.time() - t0
        print(
            f"[pretrain] iter {iter_num:4d} | loss {loss.item():.4f} | lr {lr:.2e} | {dt:.1f}s"
        )

    # (f) 周期性 eval：跑一遍 val_loader 算平均 loss + PPL；变好就保存 best checkpoint
    if iter_num > 0 and iter_num % 200 == 0:
        model.eval()
        with torch.no_grad():
            val_losses = []
            for vx, vy in val_loader:
                vx = vx.to(device)
                vy = vy.to(device)
                _, vl, _ = model(vx, targets=vy)
                val_losses.append(vl.item())
            val_loss = sum(val_losses) / max(1, len(val_losses))
            val_ppl = math.exp(val_loss)
        print(
            f"[pretrain] === eval === val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 保存 best checkpoint：含 model_state / config 快照 / iter_num / val_loss
            save_checkpoint(
                {
                    "model_state": model.state_dict(),
                    "config": {k: v for k, v in vars(config).items() if not k.startswith("_")},
                    "iter_num": iter_num,
                    "val_loss": val_loss,
                },
                "checkpoints/pretrain.pt",
            )
            print(f"[pretrain] saved best checkpoint (val_loss={val_loss:.4f})")
        model.train()  # eval 完回到 train 模式（重新打开 dropout）

    iter_num += 1


# (10) 训练循环结束后再跑一次 eval 并保存 final/best checkpoint
#     - 这一步确保最终打印一个 val_ppl（spec 要求）
#     - 若中途 eval 触发过 best save，这里会以"已是最新"理由再覆盖一次（无害）
model.eval()
with torch.no_grad():
    val_losses = []
    for vx, vy in val_loader:
        vx = vx.to(device)
        vy = vy.to(device)
        _, vl, _ = model(vx, targets=vy)
        val_losses.append(vl.item())
    final_val_loss = sum(val_losses) / max(1, len(val_losses))
    final_val_ppl = math.exp(final_val_loss)
print(
    f"[pretrain] === final eval === val_loss {final_val_loss:.4f} | val_ppl {final_val_ppl:.2f}"
)
if final_val_loss < best_val_loss:
    best_val_loss = final_val_loss
    save_checkpoint(
        {
            "model_state": model.state_dict(),
            "config": {k: v for k, v in vars(config).items() if not k.startswith("_")},
            "iter_num": iter_num,
            "val_loss": final_val_loss,
        },
        "checkpoints/pretrain.pt",
    )
    print(f"[pretrain] saved best checkpoint (val_loss={final_val_loss:.4f})")


# (11) 最终再存一次 final checkpoint（即便 best_val_loss 已是最新也确保落盘）
#     注：上面 eval 之后如果触发了 best save，best_val_loss 已经更新；
#     这里无脑再 save 一次是冗余但幂等的（同 dict 落盘）。
print(f"[pretrain] done. best val_loss = {best_val_loss:.4f}")
total_wall = time.time() - t0
print(f"[pretrain] total wall time = {total_wall:.1f}s")
save_checkpoint(
    {
        "model_state": model.state_dict(),
        "config": {k: v for k, v in vars(config).items() if not k.startswith("_")},
        "iter_num": iter_num,
        "val_loss": best_val_loss,
    },
    "checkpoints/pretrain.pt",
)
print(f"[pretrain] final checkpoint saved to checkpoints/pretrain.pt")


# (11) 用训练好的模型采几条样例，肉眼检查是否学到了句式
#     - 这里的 generate_text 走 KV-cache 路径（generate.py::generate_kv）
#     - 如果 inference.py 还没实现（早期 P3 阶段），可以临时切到 model.generate()（朴素版）
try:
    model.eval()
    print("[pretrain] === samples ===")
    for prompt in ["Once upon a time, the ", "In the forest, there lived a "]:
        try:
            text = generate_text(
                model, #gpt2
                tokenizer, 
                prompt, # Once upon a time, the 
                max_new_tokens=64,
                mode="kv",  # 朴素版最稳，不依赖 inference.py
                temperature=1.0,
                top_k=20,
            )
            print(f"  prompt: {prompt!r}\n  text:   {text!r}")
        except Exception as e:
            # 采样失败不影响主流程：pretrain 已经完成，存盘了
            print(f"  [sample skipped] {type(e).__name__}: {e}")
            break
except Exception as e:
    print(f"[pretrain] sample step skipped: {type(e).__name__}: {e}")

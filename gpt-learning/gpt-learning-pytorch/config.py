"""
集中所有超参数。
两端（PyTorch / TensorFlow）保持完全一致的 Python literal，方便跨框架对照。
每个常量附中文 inline 注释，说明用途和典型取值。
"""

# ---------------- 模型架构 ----------------
# Transformer block 层数（GPT-2 原本 12/24/36/48，tiny 场景 4 层足够）
n_layer = 4
# 多头注意力的头数（必须能整除 n_embd，4 × 32 = 128）
n_head = 4
# 隐藏维度 / token embedding 维度
n_embd = 128
# FFN 中间层维度（4 × n_embd 是惯例）
n_inner = 512
# 最大上下文长度（位置编码的最大位置数）
block_size = 128
# dropout 比例
dropout = 0.1
# 是否在 Linear/Conv1D 中使用 bias：True 还原 GPT-2 原版；False 对齐 nanoGPT 风格
bias = False

# 词表大小：运行时由 tokenizer 加载 vocab.txt 后确认，不在 config 中硬编码

# ---------------- 训练通用 ----------------
batch_size = 32            # mini-batch 大小
learning_rate = 3e-4       # 预训练阶段学习率
weight_decay = 0.1         # AdamW 权重衰减
grad_clip = 1.0            # 梯度裁剪阈值（防止梯度爆炸）
max_iters = 200            # 预训练总迭代步数（P3 smoke test：CPU 跑 200 步 ~1-3 分钟）
warmup_iters = 50          # 学习率 warmup 步数
lr_decay_iters = 200       # cosine decay 步数（与 max_iters 同步）
min_lr = 3e-5              # cosine decay 终点学习率

# ---------------- SFT 监督微调 ----------------
sft_max_iters = 1000       # SFT 迭代步数
sft_lr = 3e-4              # SFT 学习率（通常比预训练小 1-2 个数量级）

# ---------------- Reward Model ----------------
rm_max_iters = 1000        # RM 训练迭代步数
rm_lr = 3e-4               # RM 学习率
rm_batch_size = 32         # RM batch size

# ---------------- PPO 强化学习 ----------------
ppo_batch_size = 8         # 一次 rollout 的 prompt 数
ppo_minibatch_size = 4     # 一次 PPO 更新的 minibatch 大小
ppo_max_iters = 100        # 总 PPO 步数（rollout 次数）
ppo_epochs = 4             # 同一批数据连续更新的 epoch 数
ppo_lr_policy = 1e-5       # policy 学习率
ppo_lr_value = 1e-5        # value head 学习率
ppo_gamma = 1.0            # 折扣因子
ppo_lam = 0.95             # GAE λ
ppo_cliprange = 0.2        # PPO clip 范围 ε
ppo_cliprange_value = 0.2  # value clip 范围
ppo_vf_coef = 0.5          # value loss 系数 c1
ppo_kl_coef = 0.1          # KL 惩罚系数 β

# ---------------- 推理 / 采样 ----------------
max_new_tokens = 64        # 默认生成 token 数
temperature = 0.8          # 采样温度
top_k = 50                 # top-k 截断
top_p = 0.95               # nucleus 截断累计概率阈值

# ---------------- 评估 ----------------
eval_max_samples = 200     # 评估集最大样本数（CPU 跑保持小规模）
eval_seed = 1234           # 评估集随机种子

# ---------------- 系统 ----------------
seed = 42                  # 全局随机种子
device = "cpu"             # 默认设备（学习项目无需 GPU）

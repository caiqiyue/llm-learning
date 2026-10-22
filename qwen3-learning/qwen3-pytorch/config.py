"""
Qwen3 线集中超参数（唯一真值）。

与 GPT-2 线 `gpt-learning/gpt-learning-pytorch/config.py` 的差异：
  - GPT-2 线用模块级裸变量；本线用 dataclass，因为要同时支持 dense / MoE 两套配置
  - Qwen3 的 `head_dim` 与 `n_embd` **解耦**：不强制 `n_embd % n_head == 0`
    真实 Qwen3-0.6B 就是这样（hidden=1024, head_dim=128, n_head=16）
  - 两套配置共享字段，MoE 额外多 `n_experts` / `experts_per_tok` /
    `moe_intermediate_size` / `norm_topk_prob` / `aux_loss_coef`

改本文件必须同步更新 `qwen3-learning/` 下引用了对应数字的笔记（见 AGENTS.md §3）。
"""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Qwen3Config:
    """Qwen3 模型结构 + 训练超参的统一配置对象。"""

    # ---------------- 架构 ----------------
    # Transformer block 层数
    n_layer: int = 4
    # 隐藏维度 / token embedding 维度（残差流宽度）
    n_embd: int = 128
    # Q head 数
    n_head: int = 4
    # KV head 数（GQA）。必须整除 n_head；等于 n_head 时退化为 MHA
    n_head_kv: int = 2
    # 单个 head 的特征维度。**与 n_embd 解耦**：Qwen3 里 n_head * head_dim ≠ n_embd
    # 本项目取 48，4 * 48 = 192 ≠ 128，用来在代码里真实体现解耦
    head_dim: int = 48
    # SwiGLU 中间层维度。取整规则：round(n_embd * 8/3) 再向上对齐到 128 的倍数
    # 128 * 8/3 = 341.33 → 3 * 128 = 384
    intermediate_size: int = 384
    # RoPE 频率基数。Qwen3 默认 1e6（比 GPT-2 的 1e4 大 100 倍，为长上下文服务）
    rope_theta: float = 1e6
    # RMSNorm 的 epsilon（Qwen3 用 1e-6，GPT-2 的 LayerNorm 常用 1e-5）
    rms_eps: float = 1e-6
    # 最大上下文长度
    block_size: int = 128
    # dropout 比例
    dropout: float = 0.1
    # Qwen3 全线 Linear 均无 bias
    attention_bias: bool = False
    # 权重绑定：lm_head.weight 与 wte.weight 共享同一份参数
    tie_embeddings: bool = True

    # ---------------- MoE 专属（dense 配置里为 None）----------------
    # 专家数量
    n_experts: Optional[int] = None
    # 每个 token 路由到几个专家（top-k）
    experts_per_tok: Optional[int] = None
    # 每个专家的 FFN 中间层维度
    moe_intermediate_size: Optional[int] = None
    # 是否对 top-k 的路由权重重新归一化（匹配真实 Qwen3-MoE）
    norm_topk_prob: bool = True
    # 负载均衡辅助损失的系数（总 loss = lm_loss + aux_loss_coef * aux_loss）
    aux_loss_coef: float = 0.01

    # ---------------- 训练通用 ----------------
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    max_iters: int = 300
    warmup_iters: int = 50
    lr_decay_iters: int = 300
    min_lr: float = 3e-5
    # 训练日志打印间隔（步）
    log_every: int = 50
    # 验证时最多评估多少个 batch（控制 CPU 上的评估耗时）
    eval_max_batches: int = 20

    # ---------------- tokenizer / 数据 ----------------
    # byte-level BPE 目标词表大小。实际大小由训练语料决定，可能略小于该值
    bpe_vocab_size: int = 1000
    # 预训练训练集占比，其余作为验证集
    train_ratio: float = 0.9
    # 合成语料规模（预训练样本条数）
    pretrain_n_samples: int = 10000

    # ---------------- LoRA ----------------
    # LoRA 低秩矩阵的秩
    lora_r: int = 4
    # LoRA 缩放因子，实际缩放为 alpha / r
    lora_alpha: int = 8
    # LoRA dropout
    lora_dropout: float = 0.0
    # 注入 LoRA 的模块名后缀
    lora_target_modules: tuple = ("q_proj", "v_proj")
    # LoRA 微调学习率
    lora_lr: float = 3e-4
    lora_max_iters: int = 300

    # ---------------- DPO ----------------
    # DPO 的 KL 惩罚系数 β
    dpo_beta: float = 0.1
    dpo_lr: float = 1e-4
    dpo_max_iters: int = 300
    # 偏好数据集规模
    dpo_n_pairs: int = 2000

    # ---------------- 推理 / 采样 ----------------
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95

    # ---------------- 评估 ----------------
    eval_max_samples: int = 200
    eval_seed: int = 1234

    # ---------------- 系统 ----------------
    seed: int = 42
    device: str = "cpu"

    def to_dict(self) -> dict:
        """转成普通 dict（存 checkpoint 用）。"""
        return asdict(self)


# ---------------- Dense 配置 ----------------
DENSE_CONFIG = Qwen3Config()

# ---------------- MoE 配置 ----------------
# 主干与 Dense 完全一致，只有 FFN 换成 Router + 专家
MOE_CONFIG = Qwen3Config(
    n_experts=4,
    experts_per_tok=2,
    moe_intermediate_size=128,
)

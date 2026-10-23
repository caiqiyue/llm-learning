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
    # dropout 比例。
    # **只用于预训练**（train_pretrain.py 显式传 cfg.dropout）。
    # SFT / LoRA / DPO 阶段传 0.0（数据少、要求确定性）—— 见笔记 §6.4。
    # ⚠️ 模型侧的 `dropout` 覆盖参数必须真的传到每一处（embedding + 注意力 + MLP），
    #    否则「关掉 dropout」只是关了一半，train 模式的前向会继续推进全局 RNG，
    #    实验就不可复现。回归断言在 tests/test_dense.py 第 13 条。
    dropout: float = 0.1
    # Qwen3 全线 Linear 均无 bias
    attention_bias: bool = False
    # 权重绑定：lm_head.weight 与 wte.weight 共享同一份参数
    tie_embeddings: bool = True

    # ---------------- 消融开关（feat-017：与 GPT-2 的退化对照）----------------
    # ★ 这 5 个开关**只服务于对照实验** `ablations/compare_with_gpt2.py`。
    #   默认值 = 主线的正常行为 ⇒ **不改变任何既有训练 / 评估的数字**，
    #   所以本文件新增字段对既有笔记里的数字**零影响**（feat-009 核对时不必重算）。
    #   把它们关掉，Qwen3 就会退化成 GPT-2 的结构，用来量化「每个组件到底改了什么」。
    #
    #   开关名            关掉后的行为              对应 GPT-2 的什么
    #   ------------------------------------------------------------------
    #   use_gqa           n_head_kv = n_head（MHA）    GPT-2 就是 MHA
    #   use_rope          不用 RoPE，改用加法式 wpe    GPT-2 的 sinusoidal PE
    #   use_qk_norm       Q/K 不做 RMSNorm            GPT-2 没有 QK-Norm
    #   mlp_type="gelu"   两矩阵 FFN                  GPT-2 的 c_fc → GELU → c_proj
    #   norm_type="layernorm" 归一化换成 LayerNorm     GPT-2 用 nn.LayerNorm
    #
    # ⚠️ `norm_type` 不是「可选的优化」，而是**对照实验的前提**：
    #    RMSNorm（不减均值）与 LayerNorm（减均值）在数学上不等价，
    #    两边不同步切换就永远压不到 max_abs_diff < 1e-4。详见笔记 §12。
    use_gqa: bool = True
    use_rope: bool = True
    use_qk_norm: bool = True
    mlp_type: str = "swiglu"      # "swiglu" | "gelu"
    norm_type: str = "rmsnorm"    # "rmsnorm" | "layernorm"

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
    # （实测 r=4/attn/1500 步：lr 3e-4 → loss 5.743 / 总 logprob 胜率 0.590；
    #   lr 1e-3 → loss 5.517 / 胜率 0.605。取 1e-3。
    #   注：长度归一口径上 300 步的配方反而更高（0.825 vs 0.745）—— 两个口径排序不同，
    #   见 Qwen3_LoRA与DPO_学习笔记.md §7.4）
    # ⚠️ 这组数字是**修完 dropout 覆盖 bug 之后**重测的（笔记 §6.4），
    #    修复前同一配方是 5.926 / 0.590 与 5.701 / 0.595。
    lora_lr: float = 1e-3
    # LoRA 微调步数
    # （实测 r=4/attn/lr=1e-3：300 步 loss 6.073 / 胜率 0.575（还在平台期），
    #   1500 步 loss 5.517 / 胜率 0.605；基线 0.515）
    lora_max_iters: int = 1500

    # ---------------- DPO ----------------
    # DPO 的 KL 惩罚系数 β
    # （消融实测：β=0.1 → margin +4.65 / dpo_acc 0.78 / 绝对胜率 0.605→0.665；
    #   β=1.0  → margin +8.44 / dpo_acc 0.95 / 绝对胜率 0.605→0.625（只涨 margin 不涨胜率）；
    #   β=0.01 → margin +1.01 / dpo_acc 0.71 / 绝对胜率 0.605→0.715（KL 太松，间隔被摊薄）。
    #   取标准做法 0.1，详见 Qwen3_LoRA与DPO_学习笔记.md §8.8）
    dpo_beta: float = 0.1
    # DPO 学习率
    # （消融实测 1500 步 / β=0.1：lr 1e-4 → margin +1.63 / 绝对胜率 0.630；
    #   lr 1e-3 → margin +4.65 / 绝对胜率 0.665。取 1e-3）
    dpo_lr: float = 1e-3
    # DPO 训练步数
    # （消融实测 lr=1e-4：300 步 → margin +0.28 / dpo_acc 0.945 但绝对胜率 0.605→0.605
    #   （一点没动）；1500 步 → margin +1.63 / 绝对胜率 0.630。取 1500）
    dpo_max_iters: int = 1500
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
    # 生成质量评估（feat-026）：探针条数与每条最多生成多少 token
    # （探针 = 动物类问题，GPT-2 线域内 / Qwen3 线域外，见 data/quality_prompts.py）
    quality_n_samples: int = 30
    quality_max_new_tokens: int = 32

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

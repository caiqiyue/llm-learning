# -*- coding: utf-8 -*-
"""
GPT-2 RLHF PPO 训练脚本（标准化改造版，基于当前项目 train_ppo.py）
================================================================

这份代码不是另起一套工程，而是按照当前项目已有接口改造：

    config.py
    utils.py
    model/gpt2.py::GPT2
    data/sft_data.py::get_ppo_prompts
    data/tokenizer.py::CharTokenizer / EOS_ID
    inference.py::KVCache / GPT2WithKVCache

相较原教学版 PPO，核心修正如下：

1. 明确 4 个角色，并真正分成 4 个模型角色：
   - policy_model：Actor，SFT 初始化，负责生成并训练
   - ref_model：SFT 冻结副本，只用于 KL 约束
   - reward_model：RM，冻结，只负责给整段回答打分
   - value_model：Critic，SFT backbone 初始化 + 独立 value_head，负责 token-level V(s_t)

2. Value 改成真正的 token-level：

       hidden_states: (B, T, C)
       value_head(hidden_states): (B, T, 1)
       values: (B, T)

   对第 j 个 response token（action）而言，使用第 j-1 个位置的 hidden state
   预测“生成该 token 之前的状态价值”，从而与 logp 对齐：

       action_logp: (B, T-1)
       action_value: (B, T-1)

3. KL 改成 token-level reward shaping：

       kl_t = log pi_old(a_t|s_t) - log pi_ref(a_t|s_t)
       r_t^KL = - beta * kl_t

   Reward Model 的 sequence-level 分数只加在最后一个有效 response token：

       r_last += R_RM

4. GAE 使用正确的逐 token Value：

       delta_t  = r_t + gamma * V_{t+1} - V_t
       A_t      = delta_t + gamma * lambda * A_{t+1}
       Return_t = A_t + V_t

   terminal 后 V_{t+1}=0；prompt / padding 不参与 GAE。

5. Policy 使用 PPO clipped objective：

       ratio = exp(new_logp - old_logp)
       L_pi  = -min(ratio*A, clip(ratio, 1-eps, 1+eps)*A)

6. Critic 真正拟合 GAE 得到的 Return，而不是拟合旧 value：

       L_V = max(
           (V_new - Return)^2,
           (V_clip - Return)^2,
       ) / 2

7. rollout 生成不再对不同长度 prompt 做 left-padding。
   原教学版把不同长度 prompt left-pad 后直接送进无 attention-mask 的 GPT-2，
   padding 会被模型当作真实 token。这里改为逐样本 KV-cache 生成，最后再 right-pad
   形成 batch；虽然 rollout 慢一点，但语义更正确，也更方便学习/调试。

8. best checkpoint 按 raw RM reward 选择。
   原代码先把 r_total 做 batch 标准化，再拿标准化后的均值作为 best_reward，
   理论上均值会非常接近 0，不适合用于 best 模型判断。

--------------------------------------------------------------------------
重要兼容假设
--------------------------------------------------------------------------
为了得到 token-level hidden state，本文件会直接走 GPT2 backbone。
根据当前项目之前的 GPT2.forward 写法，假设 GPT2 至少有：

    tok_emb
    pos_emb
    drop
    blocks
    ln_f
    n_embd
    n_layer
    n_head
    block_size
    vocab_size

并且 Block 支持：

    block(
        x,
        kv_cache=None,
        layer_idx=layer_idx,
        position_ids=position_ids,
    )

如果你的 model/gpt2.py 后续改过，通常只需要调整 `forward_hidden_states()`。

--------------------------------------------------------------------------
形状约定
--------------------------------------------------------------------------
full_ids:       (B, T)      prompt + response + 右侧 padding
response_mask:  (B, T)      response token=1，其余=0
action_mask:    (B, T-1)    response_mask[:, 1:]
old_logp:       (B, T-1)    rollout policy 对实际 action token 的 logp
ref_logp:       (B, T-1)
old_values:     (B, T-1)    Critic 在每个 action 前状态的 V(s_t)
rewards:        (B, T-1)    token-level KL reward + terminal RM reward
advantages:     (B, T-1)
returns:        (B, T-1)
"""

import os
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from utils import set_seed, get_device, save_checkpoint, load_checkpoint
from model.gpt2 import GPT2
from data.sft_data import get_ppo_prompts
from data.tokenizer import CharTokenizer, EOS_ID
from inference import KVCache, GPT2WithKVCache


# ===========================================================================
# (0) 配置辅助
# ===========================================================================

def cfg(name: str, default):
    """config.py 缺少某个 PPO 参数时使用默认值，避免为了学习代码先改很多配置。"""
    return getattr(config, name, default)


set_seed(config.seed)
device = get_device()             #运行设备
print(f"[ppo] device = {device}")

PPO_BATCH_SIZE = cfg("ppo_batch_size", 8) # 强化学习时，一批八个样本
PPO_MINIBATCH_SIZE = cfg("ppo_minibatch_size", 4) #使用minibatch 来迭代更新
PPO_EPOCHS = cfg("ppo_epochs", 4) #总共4个 epoch
PPO_MAX_ITERS = cfg("ppo_max_iters", 200) #4个epoch * batch * minibatch 总共不能超过200轮迭代

POLICY_LR = cfg("ppo_lr_policy", 1e-6)
VALUE_LR = cfg("ppo_lr_value", POLICY_LR)

GAMMA = cfg("ppo_gamma", 1.0)
GAE_LAMBDA = cfg("ppo_lam", 0.95)
KL_COEF = cfg("ppo_kl_coef", 0.05)

CLIP_RANGE = cfg("ppo_cliprange", 0.2)
VALUE_CLIP_RANGE = cfg("ppo_cliprange_value", CLIP_RANGE)
VALUE_LOSS_COEF = cfg("ppo_vf_coef", 0.5)
ENTROPY_COEF = cfg("ppo_entropy_coef", 0.0)
TARGET_KL = cfg("ppo_target_kl", None)

MAX_NEW_TOKENS = cfg("max_new_tokens", 32)
TEMPERATURE = cfg("temperature", 1.0)
TOP_K = cfg("top_k", 0)
TOP_P = cfg("top_p", 0.95)

GRAD_CLIP = cfg("grad_clip", 1.0)
WEIGHT_DECAY = cfg("weight_decay", 0.0)

LOG_EVERY = cfg("ppo_log_every", 5)
SAVE_EVERY = cfg("ppo_save_every", 25)

# 当前项目原代码用 0 做 left-pad / finished token，占位上基本等价于 unk。
# 这里生成结束后只在 batch 最右侧 padding，且所有 PPO loss 都会 mask 掉它。
PAD_ID = cfg("pad_id", 0)

# reward whitening 可选。默认关闭，避免 batch 很小时把 RM 绝对尺度完全抹掉。
REWARD_WHITEN = cfg("ppo_reward_whiten", False)


# ===========================================================================
# (1) Tokenizer
# ===========================================================================
vocab_path = os.path.join("data", "vocab.txt")
v = r'E:\ai-learning\05-llm-fundamentals\llm-learning\gpt-learning\gpt-learning-pytorch\data\vocab.txt'
tokenizer = CharTokenizer(vocab_path=v)
vocab_size = tokenizer.vocab_size
print(f"[ppo] vocab_size = {vocab_size}")


# ===========================================================================
# (2) 模型构造
# ===========================================================================

def build_gpt2(dropout: float = 0.0) -> GPT2:
    """构造与 pretrain / SFT / RM 阶段相同结构的 GPT-2。"""
    return GPT2(
        vocab_size=vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        n_inner=config.n_inner,
        block_size=config.block_size,
        dropout=dropout,
        bias=config.bias,
    )


# ---------------------------------------------------------------------------
# 2.1 Policy / Actor：SFT 初始化，参与 PPO 更新
# ---------------------------------------------------------------------------

policy_model = build_gpt2(dropout=0.0).to(device) #策略模型，主要被训练的对象

sft_ckpt_path = r"E:\ai-learning\05-llm-fundamentals\llm-learning\gpt-learning\gpt-learning-pytorch\checkpoints\sft.pt"
print(f"[ppo] loading SFT checkpoint from {sft_ckpt_path}")
sft_ckpt = load_checkpoint(sft_ckpt_path)

# 只加载 SFT backbone / lm head；即使 ckpt 偶然带 value_head，也不加载。
sft_state = {
    k: v
    for k, v in sft_ckpt["model_state"].items()
    if not k.startswith("value_head.")
}
missing, unexpected = policy_model.load_state_dict(sft_state, strict=False) #策略模型以微调后的模型为初始出发点
print(
    f"[ppo] policy loaded from SFT "
    f"(missing={len(missing)}, unexpected={len(unexpected)})"
)


# ---------------------------------------------------------------------------
# 2.2 Reference Model：SFT 冻结副本，只计算 KL
# ---------------------------------------------------------------------------

ref_model = build_gpt2(dropout=0.0).to(device) #参考模型，不训练，以微调后的模型作为出发点
ref_model.load_state_dict(sft_state, strict=False)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad_(False)
print("[ppo] ref_model loaded (frozen)")


# ---------------------------------------------------------------------------
# 2.3 Reward Model：加载 reward.pt，冻结
# ---------------------------------------------------------------------------

reward_model = build_gpt2(dropout=0.0) # 奖励模型，不训练
reward_model.value_head = nn.Linear(reward_model.n_embd, 1, bias=True)
reward_model.to(device)

reward_ckpt = load_checkpoint(r"E:\ai-learning\05-llm-fundamentals\llm-learning\gpt-learning\gpt-learning-pytorch\checkpoints\reward.pt")
reward_model.load_state_dict(reward_ckpt["model_state"], strict=False)
reward_model.eval()
for p in reward_model.parameters():
    p.requires_grad_(False)
print("[ppo] reward_model loaded (frozen)")


# ---------------------------------------------------------------------------
# 2.4 Value Model / Critic：独立 GPT-2 backbone + token-level value head
# ---------------------------------------------------------------------------
"""
Reward Model： 这段已经生成的回答，质量好不好？



Value Model： 当前已经生成到这里了，如果接着生成下去，预计最后能获得多少奖励？

1. Reward Model 做什么？
它读取完整的回答，并给出一个奖励：
\[
R=3.5
\]

意思是：根据之前学到的人类偏好，它认为这个回答质量比较好。
在常见的 PPO 实现中，我们主要取回答末尾的一个 Reward 分数作为整体评价。
2. Value Model 做什么？
它不是等回答全部生成后才打一个分数，而是针对每个生成位置预测后续能够获得多少累计奖励。
例如：
已生成内容	Value 预测
深度	1.2
深度学习	1.8
深度学习是	2.1
深度学习是机器	2.7
深度学习是机器学习的分支	3.2

"""

value_model = build_gpt2(dropout=0.0) # value 模型和 奖励模型的架构一样，训练
value_model.value_head = nn.Linear(value_model.n_embd, 1, bias=True)
value_model.to(device)

# backbone 从 SFT 初始化；value_head 是新的，所以 strict=False。
missing, unexpected = value_model.load_state_dict(sft_state, strict=False)
nn.init.normal_(value_model.value_head.weight, mean=0.0, std=0.02)
nn.init.zeros_(value_model.value_head.bias)
print(
    f"[ppo] value_model initialized from SFT backbone "
    f"(missing={len(missing)}, unexpected={len(unexpected)})"
)


# ===========================================================================
# (3) Optimizer
# ===========================================================================

policy_optimizer = torch.optim.AdamW(
    [p for p in policy_model.parameters() if p.requires_grad],
    lr=POLICY_LR,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.95),
)

value_optimizer = torch.optim.AdamW(
    [p for p in value_model.parameters() if p.requires_grad],
    lr=VALUE_LR,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.95),
)


# ===========================================================================
# (4) 模型前向工具
# ===========================================================================

def unpack_logits(model_output):
    """兼容 GPT2.forward 返回 3 元组 / 4 元组。"""
    if len(model_output) == 4:
        logits, _, _, _ = model_output
    else:
        logits, _, _ = model_output
    return logits


def forward_hidden_states(model: GPT2, input_ids: torch.Tensor) -> torch.Tensor:
    """
    直接走 GPT-2 backbone，得到每个 token 的最后层 hidden state。

    输入：
        input_ids: (B, T)

    输出：
        x: (B, T, C)

    为什么不能继续使用当前 GPT2.forward 返回的 value？
    -------------------------------------------------
    当前项目里的 value_head 逻辑是：

        last_hidden = x[:, -1, :]       # (B, C)
        value = value_head(last_hidden) # (B,)

    这只得到 sequence-level scalar。
    PPO Critic 需要每一个 action 前状态的 V(s_t)，因此必须保留全部 hidden state。
    """
    B, T = input_ids.shape #8,106
    if T > model.block_size:
        raise ValueError(
            f"sequence length {T} exceeds block_size={model.block_size}"
        )

    position_ids = torch.arange(T, device=input_ids.device, dtype=torch.long)
    position_ids = position_ids.unsqueeze(0).expand(B, T)  # (B, T)

    tok = model.tok_emb(input_ids)  # (B, T, C)

    # 兼容 pos_emb 是 nn.Embedding 或可索引参数/张量两种写法。
    if isinstance(model.pos_emb, nn.Embedding):
        pos = model.pos_emb(position_ids)
    else:
        pos = model.pos_emb[position_ids]

    x = model.drop(tok + pos)  # (B, T, C) 8,106,128

    new_kv = None
    for layer_idx, block in enumerate(model.blocks):
        block_out = block(
            x,
            kv_cache=new_kv,
            layer_idx=layer_idx,
            position_ids=position_ids,
        )
        if isinstance(block_out, tuple):
            x, new_kv = block_out
        else:
            # 防御：如果未来 Block.forward 只返回 x，也能继续运行。
            x = block_out

    x = model.ln_f(x)  # (B, T, C)  8,106,128
    return x


def token_values(model: GPT2, full_ids: torch.Tensor) -> torch.Tensor:
    """
    计算 token-level Critic Value。

    hidden: (B, T, C)
      ↓ value_head(C -> 1)
    values: (B, T)

    注意：后续与 action/logp 对齐时使用 values[:, :-1]。
    因为位置 t 的 hidden state 表示“已经看到 token[0:t] 的状态”，
    它对应位置 t 的 logits 去预测 token[t+1] 这个 action。
    """
    hidden = forward_hidden_states(model, full_ids)        # (B, T, C) 8,106,128 中间特征
    values = model.value_head(hidden).squeeze(-1)          # (B, T) 8,106
    return values


def action_logp_and_entropy(
    model: GPT2,
    full_ids: torch.Tensor,          # (B, T)
    response_mask: torch.Tensor,     # (B, T)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算实际生成 token 的 log probability 与策略 entropy。

    对 GPT 自回归模型：

        logits[:, t] 预测 full_ids[:, t+1]

    因此：

        selected_logp: (B, T-1)
        action_mask:   response_mask[:, 1:] = (B, T-1)

    prompt / padding 位置会被 mask 为 0。
    """
    logits = unpack_logits(model(full_ids))                # (B, T, V)  8,106,45
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (B, T-1, V) 8,105,45 不看生成的最后一个token算出的概率
    probs = log_probs.exp()      #8,105,45

    next_ids = full_ids[:, 1:]                             # (B, T-1) 8,105 从prompt 第一个词往后移（就是targets的做法）
    selected_logp = log_probs.gather(
        dim=-1,
        index=next_ids.unsqueeze(-1),       
    ).squeeze(-1)                                          # (B, T-1)  8,105 每个样本中每个token预测并选择下一个token时的概率

    entropy = -(probs * log_probs).sum(dim=-1)            # (B, T-1)   代表token预测下一个token时的不确定性，熵越低  不确定性越低，模型回复多样性越低

    action_mask = response_mask[:, 1:].float()            # (B, T-1)
    return selected_logp * action_mask, entropy * action_mask


# ===========================================================================
# (5) Rollout 采样工具：逐样本 KV-cache 生成，避免 left-padding 污染
# ===========================================================================

def _sample_from_logits(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    vocab_size: int,
) -> torch.Tensor:
    """与当前项目 generate/inference 逻辑一致的 temperature + top-k + top-p 采样。"""
    if temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature #温度调整概率分布

    if top_k is not None and top_k > 0: #前top—k
        k = min(int(top_k), vocab_size)
        kth = logits.topk(k, dim=-1).values[:, -1:]
        logits = torch.where(
            logits < kth,
            torch.full_like(logits, float("-inf")),
            logits,
        )

    if top_p is not None and 0.0 < top_p < 1.0: #概率累加超过 top-p
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        sorted_probs = sorted_logits.softmax(dim=-1)
        cumsum = sorted_probs.cumsum(dim=-1)

        # 保留第一个使累计概率跨过 top_p 的 token。
        remove_mask = cumsum - sorted_probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))

        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(1, sorted_idx, sorted_logits)
        logits = filtered

    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_one_with_kv(
    model: GPT2,
    prompt_ids: torch.Tensor,   # (1, T_prompt)
    max_new_tokens: int,
) -> Tuple[torch.Tensor, int, int]:
    """
    单条 prompt 的 KV-cache 自回归生成。

    返回：
        full_ids:      (T_full,) = prompt + response
        prompt_len:    int
        response_len:  int

    这里故意逐样本生成：
    当前 GPT2 没有 attention_mask，直接把不同长度 prompt left-pad 后批量 prefill
    会把 PAD/UNK 当成真实历史 token。逐样本生成更适合当前教学项目。
    """
    model.eval()

    # 至少为 response 留 1 个位置。
    max_prompt_len = max(1, model.block_size - 1)
    if prompt_ids.size(1) > max_prompt_len:
        # 保留 prompt 尾部，确保 <sep> 和最近上下文还在。
        prompt_ids = prompt_ids[:, -max_prompt_len:]

    prompt_len = prompt_ids.size(1) #prompt token 32
    allowed_new = min(max_new_tokens, model.block_size - prompt_len) # 算出response 最多是多少token
    if allowed_new <= 0:
        return prompt_ids.squeeze(0), prompt_len, 0

    wrapped = GPT2WithKVCache(model)
    head_dim = model.n_embd // model.n_head  #每个头32维
    cache = KVCache(
        n_layer=model.n_layer, #4
        n_head=model.n_head, # 4
        head_dim=head_dim, #32
        max_seq_len=model.block_size, #128
        batch_size=1,
    )

    # ---------- Prefill prompt ----------
    pos = torch.arange(prompt_len, device=prompt_ids.device, dtype=torch.long)
    pos = pos.unsqueeze(0)                               # (1, T_prompt)  1,32
    logits, cache = wrapped(prompt_ids, cache, pos) #logits 1,32,45  kv-cache
    next_logits = logits[:, -1, :]                      # (1, V) 只选择最后一个token的logits 1,45

    generated: List[torch.Tensor] = []

    # ---------- Decode ----------
    for step in range(allowed_new): #从1,45 也就是45中选用不同的策略去找出一个
        next_token = _sample_from_logits(
            next_logits,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            vocab_size=model.vocab_size,
        )                                               # (1, 1)
        generated.append(next_token)

        if EOS_ID is not None and int(next_token.item()) == int(EOS_ID):
            break

        # 当前生成 token 位于 prompt_len + step；把它喂进去后预测下一 token。
        cur_pos = torch.tensor(
            [[prompt_len + step]],
            dtype=torch.long,
            device=prompt_ids.device,
        )
        logits, cache = wrapped(next_token, cache, cur_pos)
        next_logits = logits[:, -1, :]

    if generated:
        response = torch.cat(generated, dim=1)          # (1, T_resp)
        full = torch.cat([prompt_ids, response], dim=1) # (1, T_full)
        response_len = response.size(1)
    else:
        full = prompt_ids
        response_len = 0

    return full.squeeze(0), prompt_len, response_len


@torch.no_grad()
def generate_batch(
    model: GPT2,  #gpt-2
    prompts: List[torch.Tensor], #8个样本的token
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int], List[int]]:
    """
    逐样本 rollout，最后 right-pad 成 batch。

    返回：
        full_ids:       (B, T_max)
        response_mask:  (B, T_max)
        prompt_lens:    list[int]
        response_lens:  list[int]
        seq_lens:       list[int]
    """
    sequences = []
    prompt_lens = []
    response_lens = []
    seq_lens = []

    for prompt_ids in prompts: #为每个prompt 生成response
        seq, p_len, r_len = generate_one_with_kv(
            model=model,
            prompt_ids=prompt_ids,  #token
            max_new_tokens=MAX_NEW_TOKENS, #最多生成数
        )
        sequences.append(seq)
        prompt_lens.append(p_len)
        response_lens.append(r_len)
        seq_lens.append(seq.numel())

    B = len(sequences)
    T_max = max(seq_lens)

    full_ids = torch.full( #把最长的作为标杆， prompt + response 最长106，那么都对齐到106
        (B, T_max),
        fill_value=PAD_ID,
        dtype=torch.long,
        device=device,
    )
    response_mask = torch.zeros(  #需要掩盖prompt 部分
        (B, T_max),
        dtype=torch.float32,
        device=device,
    )

    for i, seq in enumerate(sequences):
        seq_len = seq.numel()
        p_len = prompt_lens[i]
        r_len = response_lens[i]

        full_ids[i, :seq_len] = seq
        if r_len > 0:
            response_mask[i, p_len:p_len + r_len] = 1.0

    return full_ids, response_mask, prompt_lens, response_lens, seq_lens


# ===========================================================================
# (6) Reward / GAE
# ===========================================================================

@torch.no_grad()
def score_reward_model(
    full_ids: torch.Tensor,
    seq_lens: List[int],
) -> torch.Tensor:
    """
    Reward Model 给每条完整序列打一个 scalar 分数。

    为什么逐样本算？
    当前 RM 的 value_head 只取 x[:, -1, :]。
    batch right-padding 后若直接整体 forward，短样本最后一个位置会是 PAD，
    RM 就会错误地从 PAD hidden 上打分，所以这里按真实 seq_len 切片。
    """
    scores = []
    for i, seq_len in enumerate(seq_lens):
        ids = full_ids[i:i + 1, :seq_len]               # (1, T_i) 1,96 可以理解为去掉padding
        out = reward_model(ids, targets=None)  #out[3] 是本次response的 reward分数
        if len(out) != 4:
            raise RuntimeError(
                "reward_model.forward 必须在挂 value_head 后返回 4 元组 "
                "(logits, loss, kv, value)"
            )
        _, _, _, score = out
        scores.append(score.squeeze(0))

    return torch.stack(scores, dim=0)                   # (B,)


def whiten_masked(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """只在 mask=1 的有效位置做 whitening，其余位置保持 0。"""
    valid = mask.bool()
    out = torch.zeros_like(x)
    vals = x[valid]
    if vals.numel() == 0:
        return out

    mean = vals.mean()
    # unbiased=False：即使只有一个有效元素也不会产生 NaN。
    std = vals.std(unbiased=False)
    out[valid] = (vals - mean) / (std + 1e-8)
    return out


def compute_gae(
    rewards: torch.Tensor,      # (B, T-1)
    values: torch.Tensor,       # (B, T-1)
    mask: torch.Tensor,         # (B, T-1)
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    标准 token-level GAE。

    对每个有效 action：

        delta_t = r_t + gamma * V_{t+1} * nonterminal - V_t
        A_t     = delta_t + gamma * lambda * nonterminal * A_{t+1}
        Return_t = A_t + V_t

    其中：
        nonterminal = 1  当下一个位置仍是 response action
                    = 0  当前 action 已经是 response 最后一位

    这样最后一个 response token 后不会 bootstrap 到 padding 的 Value。
    """
    B, T = rewards.shape  #8,105
    advantages = torch.zeros_like(rewards) #8,105
    last_gae = torch.zeros(B, device=rewards.device, dtype=rewards.dtype) #8

    for t in reversed(range(T)):
        mask_t = mask[:, t]                             # (B,) mask 8,105  mask_t 8

        if t < T - 1:
            next_mask = mask[:, t + 1]                  # (B,)
            next_value = values[:, t + 1]               # (B,)
        else:
            next_mask = torch.zeros_like(mask_t)
            next_value = torch.zeros_like(mask_t)

        delta = (
            rewards[:, t]  #8,105
            + gamma * next_value * next_mask
            - values[:, t]
        )                                               # (B,)

        last_gae = (
            delta
            + gamma * lam * next_mask * last_gae
        )

        # prompt / padding 位置完全清零，同时切断递归传播。
        last_gae = last_gae * mask_t
        advantages[:, t] = last_gae

    returns = advantages + values                       # (B, T-1)
    return advantages, returns


# ===========================================================================
# (7) Prompt 池
# ===========================================================================

prompts_data = get_ppo_prompts(n=1000, seed=config.seed) #1000 个 prompts
prompt_texts = [p["prompt"] for p in prompts_data] #1000个 prompt
print(f"[ppo] prompt pool size = {len(prompt_texts)}")


def encode_prompt(prompt_str: str) -> List[int]:
    """与 SFT 阶段保持一致：<bos> prompt <sep>，然后让 policy 接 response。"""
    return tokenizer.encode(f"<bos> {prompt_str} <sep>")


# ===========================================================================
# (8) 训练主循环
# ===========================================================================

iter_num = 0
best_raw_rm_reward = -float("inf")
t0 = time.time()

prev_var = None
collapse_streak = 0
stopped_early = False

raw_reward_history = []
kl_history = []
pg_loss_history = []
v_loss_history = []

print(
    f"[ppo] start training. total_iters={PPO_MAX_ITERS}, "
    f"batch={PPO_BATCH_SIZE}, minibatch={PPO_MINIBATCH_SIZE}, "
    f"epochs={PPO_EPOCHS}, max_new_tokens={MAX_NEW_TOKENS}"
)

while iter_num < PPO_MAX_ITERS:#最外层循环，ppo总共迭代 100次
    # -----------------------------------------------------------------------
    # (a) 取一批 prompt
    # -----------------------------------------------------------------------
    bs = PPO_BATCH_SIZE #一批 8个 prompt
    start = (iter_num * bs) % len(prompt_texts)
    batch_prompts = prompt_texts[start:start + bs]
    if len(batch_prompts) < bs:
        batch_prompts += prompt_texts[:bs - len(batch_prompts)]

    prompts_ids_list = [ #8 个 prompt ，每个都tokenizer  
        torch.tensor(
            encode_prompt(p),
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        for p in batch_prompts
    ]

    # -----------------------------------------------------------------------
    # (b) ROLLOUT：当前 policy 生成 response，8个prompt ，每个prompt 都生成response
    # -----------------------------------------------------------------------
    policy_model.eval()
    with torch.no_grad():
        (
            full_ids,           # (B, T)  8,106
            response_mask,      # (B, T)  8,106
            prompt_lens, #记录每个 promtp的 长度
            response_lens, #策略模型回复的 长度
            seq_lens, #prompt+回复的 总长度
        ) = generate_batch(policy_model, prompts_ids_list) #为每个prompt 生成response

    B, T_full = full_ids.shape #padding 到 106
    action_mask = response_mask[:, 1:].float()          # (B, T-1)  8,105  prompt 处为0，response 处为1

    if action_mask.sum().item() == 0:
        print(f"[ppo] iter {iter_num}: no response tokens generated, skip")
        iter_num += 1
        continue

    # -----------------------------------------------------------------------
    # (c) 固定 rollout 数据：old_logp / ref_logp / old_value / RM reward
    # -----------------------------------------------------------------------
    policy_model.eval()
    ref_model.eval()
    reward_model.eval()
    value_model.eval()

    with torch.no_grad():
        old_logp, _ = action_logp_and_entropy( #8,105   旧policy 此时 和 refenrence model 一致
            policy_model,
            full_ids,
            response_mask,
        )                                               # (B, T-1)

        ref_logp, _ = action_logp_and_entropy( #8,105
            ref_model,
            full_ids,
            response_mask,
        )                                               # (B, T-1)

        """
        使用两种奖励信号：
        1. Reward Model 奖励：评价整个回答的质量，只在最后一个有效生成 Token 的位置添加。
        2. KL 惩罚：比较 Policy Model 和 Reference Model 对每个已生成 Token 的概率，避免 Policy 偏离参考模型太远
        """
        # value_model 对每个 token 位置都输出 Value；
        # [:, :-1] 与“由位置 t 预测 token t+1 这个 action”对齐。
        old_values_full = token_values(value_model, full_ids)  # (B, T) 8,106 每个token位置产生的value值
        old_values = old_values_full[:, :-1] * action_mask     # (B, T-1)  8,105   除去最后一个token 的value值，action mask 掩盖prompt部分的value   Value Model 对每个位置预测未来回报

        # RM：每个完整 response 一个 sequence-level scalar。
        rm_scores = score_reward_model(full_ids, seq_lens)      # (B,)  8个  一个response 就一个 reward，但是value 是一个token就一个value

        # sampled token-level KL estimator：
        #     kl_t = log pi_old(a_t|s_t) - log pi_ref(a_t|s_t),计算kl散度，防止策略模型更新离原模型太远
        kl_token = (old_logp - ref_logp) * action_mask          # (B, T-1) 8,105 计算kl散度

        # 每一步先拿 KL shaping reward：-beta * KL_t
        rewards = -KL_COEF * kl_token                           # (B, T-1)

        # RM 的总分只加到每条 response 的最后一个 action 上。
        time_idx = torch.arange(  #8,105  0,1,2,3,...104
            T_full - 1,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0).expand(B, -1)                            # (B, T-1)

        last_action_idx = torch.where(    #8  找出生成的最后一个token的位置，因为都padding到了106，所以最后一个有效token的位置不一定是105
            action_mask.bool(),           # 94,  96, 102, 100, 102, 104,  94,  94   每一个Prompt 生成的response 的最后一个有效token位置
            time_idx,
            torch.full_like(time_idx, -1),
        ).max(dim=1).values                                     # (B,)

        valid_sequence = last_action_idx >= 0
        batch_idx = torch.arange(B, device=device)[valid_sequence] #8
        rewards[                                                   #8,105  8个response的 105 的 kl散度值，奖励制加在最后一个上
            batch_idx,
            last_action_idx[valid_sequence],
        ] += rm_scores[valid_sequence]                              #8  只加在有效位置上

        """
        Whiten 在这里通常就是类似 Z-score 的标准化：
        \[
        \boxed{x'=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}}
        \]

        Reward Whitening	Advantage Whitening
        处理对象	即时 Reward	GAE 计算出的 Advantage
        发生阶段	GAE 之前	GAE 之后
        主要作用	调整奖励数值尺度	调整 Policy Loss 的优势值尺度
        是否改变 Return	会间接改变	通常不会
        特别要注意，Advantage Whitening 后，原本正的 Advantage 也可能变为负数，因为它现在表示相对于当前 Batch 平均 Advantage 的高低。
        """


        if REWARD_WHITEN:
            rewards = whiten_masked(rewards, action_mask)

        # ---------------------------------------------------------------
        # GAE：从 terminal 向前做 credit assignment
        """
                    \[
            \boxed{\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)}
            \]

            我们逐个拆开：
            符号	含义
            \(r_t\)	当前 Token 对应的即时 Reward
            \(V(s_t)\)	执行当前 Action 前，Value Model 预测的累计回报
            \(V(s_{t+1})\)	执行 Action 后，对后续累计回报的预测
            \(\gamma\)	折扣因子，例如 0.99
            \(\delta_t\)	当前一步的 TD 误差


            GAE 递推
                \[
                \boxed{A_t=\delta_t+\gamma\lambda A_{t+1}}
                \]

                其中：
                - \(A_t\)：当前 Token 的 Advantage。
                - \(\delta_t\)：当前的 TD Error。
                - \(A_{t+1}\)：下一个 Token 的 Advantage。
                - \(\lambda\)：控制未来 TD Error 向前传播的程度，常取 0.95。
                最后一个有效 Action 如果是终止位置，那么：
                \[
                V(s_{T+1})=0,\qquad A_{T+1}=0
                \]
        
        """
        # ---------------------------------------------------------------
        advantages, returns = compute_gae(  #advantages 8,105  returns 8,105
            rewards=rewards,
            values=old_values,
            mask=action_mask,
            gamma=GAMMA,
            lam=GAE_LAMBDA,
        )                                                       # 各 (B, T-1)

        # 只对有效 response action 标准化 Advantage。
        advantages = whiten_masked(advantages, action_mask)

    old_logp = old_logp.detach()
    old_values = old_values.detach()
    advantages = advantages.detach()
    returns = returns.detach()

    # 日志统计：KL 用 response token 平均值；RM 使用未标准化原始分数。 一些统计数据
    valid_count = action_mask.sum().clamp(min=1.0)
    mean_kl_iter = float((kl_token.sum() / valid_count).item())
    mean_raw_rm_iter = float(rm_scores.mean().item())

    raw_reward_history.append(mean_raw_rm_iter)
    kl_history.append(mean_kl_iter)

    # -----------------------------------------------------------------------
    # (d) PPO UPDATE：同一批 rollout 数据重复训练 K epoch
    # -----------------------------------------------------------------------
    policy_model.train()
    value_model.train()

    pg_losses_epoch = []
    v_losses_epoch = []
    entropy_epoch = []
    approx_kl_epoch = []

    early_stop_epoch = False

    for ppo_epoch in range(PPO_EPOCHS): #4 使用的是原来的 8条数据  现在属于第二层循环，
        indices = np.random.permutation(B)

        for mb_start in range(0, B, PPO_MINIBATCH_SIZE):  #使用minibatch
            mb_idx = indices[mb_start:mb_start + PPO_MINIBATCH_SIZE]  #本来一个batch 8个样本，现在使用minibatch，变成4个
            if len(mb_idx) == 0:
                continue

            mb_idx_t = torch.tensor(                           #4  minibatch 打乱顺序
                mb_idx,
                dtype=torch.long,
                device=device,
            )

            mb_full_ids = full_ids[mb_idx_t]                    # (mB, T)   
            mb_response_mask = response_mask[mb_idx_t]          # (mB, T)
            mb_mask = action_mask[mb_idx_t]                     # (mB, T-1)
            mb_old_logp = old_logp[mb_idx_t]                    # (mB, T-1)
            mb_old_value = old_values[mb_idx_t]                 # (mB, T-1)
            mb_adv = advantages[mb_idx_t]                       # (mB, T-1)
            mb_ret = returns[mb_idx_t]                          # (mB, T-1)

            n_valid = mb_mask.sum().clamp(min=1.0)

            # -----------------------------------------------------------
            # Actor：new logp + PPO clipped policy loss 使用advantage 更新 策略模型，使用 return 更新 value model

            """
            - old_logp：固定不变，记录生成这批 Response 时的策略概率。
            - new_logp：每次 Mini-batch 更新时，通过当前 Policy Model 重新计算。
            假设 Rollout 时生成了一个 Token D：
            \[
            P_{\text{old}}(D)=0.4
            \]

            第一次 PPO Update 时，假设 Policy 尚未发生更新：
            \[
            P_{\text{new}}(D)=0.4
            \]

            当执行一次参数更新后，可能变成：
            \[
            P_{\text{new}}(D)=0.5
            \]

            此时第二次再计算 new_logp，得到的值就可能不同了。
            
            """
            # -----------------------------------------------------------
            new_logp, entropy = action_logp_and_entropy(         #torch.Size([4, 105])  entropy 4,105  new policy，但是第一次的时候，应该和 old policy， ref policy 一样
                policy_model,
                mb_full_ids,
                mb_response_mask,
            )                                                   # (mB, T-1)

            log_ratio = new_logp - mb_old_logp   #计算ppo ratio  4,105
            ratio = torch.exp(log_ratio)                        # (mB, T-1)
            """
            \(A_t>0\)：当前 Action 的结果比 Critic 原先预期更好，倾向于提高该 Token 的生成概率。



                \(A_t<0\)：当前 Action 的结果比预期更差，倾向于降低该 Token 的生成概率。
            """
            surr1 = ratio * mb_adv   #advantage  乘以 ration
            surr2 = torch.clamp(    # 对ration 进行 控制，防止一下子变化太大
                ratio,
                1.0 - CLIP_RANGE, #0.2
                1.0 + CLIP_RANGE,
            ) * mb_adv

            """
            利用 Advantage 调整 Token 的生成概率，同时通过 PPO Clip 限制过度更新的激励，通过 Entropy Bonus 鼓励一定的探索。
            低熵：模型的概率非常集中，生成结果比较确定。



            高熵：模型的概率比较分散，生成结果有更多不确定性。Entropy 衡量的是整个词表上的概率分布，而不是某一个已采样 Token 的概率


            根据 Advantage，提高好 Action 的概率，降低差 Action 的概率。



            保持适当的概率分布多样性，而不是一味地把某个 Token 的概率推向 100%。            
            """

            pg_loss_per_token = -torch.minimum(surr1, surr2) #4,105  每个token 的 advantage
            pg_loss = (pg_loss_per_token * mb_mask).sum() / n_valid  #总和 advantage

            entropy_mean = (entropy * mb_mask).sum() / n_valid  # 每个 位置的 熵的 总和
            actor_loss = pg_loss - ENTROPY_COEF * entropy_mean  # 去训练 policy model 熵越大，loss 越小（模型更新方向），模型的回复更加多样性

            # -----------------------------------------------------------
            # Critic：token-level Value + clipped value loss
            # -----------------------------------------------------------
            new_values_full = token_values(value_model, mb_full_ids)  # (mB, T)
            new_values = new_values_full[:, :-1]                      # (mB, T-1)

            value_pred_clipped = mb_old_value + torch.clamp(       #裁剪 New Value 相对于 Old Value 的变化量。 如果不允许 Value 的预测变化超过指定范围，那么对应的预测误差是多少？
                new_values - mb_old_value,
                -VALUE_CLIP_RANGE,
                VALUE_CLIP_RANGE,
            )
            """
            普通的平方误差：
                    \[
                    L_{\text{unclipped}}=(V_{\text{new}}-Return)^2
                    \]

                    代入：
                    \[
                    L_{\text{unclipped}}=(1.5-2.0)^2=0.25
                    \]

                    它回答的问题是：
                    当前 Value Model 的预测距离 Return 还有多远？
                    这就是 Critic 最基本的训练目标。
            
            """
            value_loss_unclipped = (new_values - mb_ret) ** 2   # 直接算 value model 预测出的value 和 return 差多远     普通mse        mb_ret gae 算出来的 return，希望以后value model预测的value 越来越接近return
            value_loss_clipped = (value_pred_clipped - mb_ret) ** 2  # 再加上一个 不允许 value 的变化 太大
            value_loss_per_token = 0.5 * torch.maximum(    #普通 MSE 更关心“预测得准不准”，Clipped Value Loss 还试图兼顾“同一轮更新是否过于激进”
                value_loss_unclipped,
                value_loss_clipped,
            )

            """
             选2个 value loss 的原因
            
             无关return 比old value 大还是小， new value 的clip 都是希望 new value 值不要一下子就与return 太近，导致loss更新平缓
             Value Clip 不是为了防止 New Value 太接近 Return，也不是为了避免 Loss 更新平缓，而是为了防止 New Value 相对于 Old Value 变化过大时，模型还能从这种大幅变化中获得额外的 Loss 改善
            裁剪限制了 Value 过度变化所能带来的 Loss 改善。当 New Value 已经朝正确方向变化并超过裁剪范围时，Loss 可能进入平坦区域，使这个 Token 不再提供进一步增大变化的梯度。
            
            """
            value_loss = (value_loss_per_token * mb_mask).sum() / n_valid
            critic_loss = VALUE_LOSS_COEF * value_loss   #去训练 value model

            # -----------------------------------------------------------
            # 分别更新 Actor / Critic
            # -----------------------------------------------------------
            policy_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy_model.parameters(),
                GRAD_CLIP,
            )
            policy_optimizer.step()

            value_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                value_model.parameters(),
                GRAD_CLIP,
            )
            value_optimizer.step()

            # -----------------------------------------------------------
            # PPO new-vs-old approximate KL，用于监控单轮更新是否过猛
            # Schulman 常见近似：E[(ratio - 1) - log_ratio]
            """
            之前的ration 比的是 old policy 和 ref policy
            现在 比的是 new policy 和 old 
            ration 已经 是 new 和 old 概率比值

            PPO 训练时间线
                1. 初始化（尚未训练 PPO）
                Reference = SFT 模型
                Policy = SFT 模型
                如果两者参数完全一致，初始概率也一样。
                2. PPO Iteration 1：Rollout
                Policy 生成 8 条 Response，并保存当前的 old_logp。
                Old Policy = 初始 SFT Policy
                Reference = 初始 SFT 模型
                所以这时候 ref_KL 可能约为 0。
                3. PPO Iteration 1：Update
                使用固定的 old_logp，多次更新当前 policy_model。
                Old LogP = 不变
                New LogP = 随着参数更新而变化
                Reference = 不变
                4. PPO Iteration 2：重新 Rollout
                上一轮更新后的 Policy 现在用于生成新 Response。
                新的 Old Policy = 上轮更新后的 Policy
                新的 Old LogP = 重新计算并保存
                Reference = 仍然是最初的 SFT 模型



            """
            # -----------------------------------------------------------
            """
            为什么需要减 1？
                    先考虑 Old Policy 和 New Policy 完全相同的情况：
                    \[
                    P_{\text{new}}=P_{\text{old}}
                    \]

                    此时：
                    \[
                    r=1
                    \]

                    我们知道两个完全相同的概率分布，KL 散度应该是 0。
                    如果没有 -1：
                    \[
                    r-\ln r=1-\ln1=1
                    \]

                    显然不对。
                    加上 -1：
                    \[
                    (r-1)-\ln r=(1-1)-\ln1=0
                    \]

                    这就符合 KL 的性质了。
                    但这只是表面原因，下面才是它真正的数学依据。
            
            """
            with torch.no_grad():
                approx_kl = (                                    # 在minibatch中  判定 new policy 不要离 old policy 更新幅度太大 ，old policy 是在 batch的时候固定，batch换，old policy也会换，new policy是在 minibatch的时候算的
                    (((ratio - 1.0) - log_ratio) * mb_mask).sum()
                    / n_valid
                )

            #记录minibatch 的训练指标
            pg_losses_epoch.append(float(pg_loss.item()))
            v_losses_epoch.append(float(value_loss.item()))
            entropy_epoch.append(float(entropy_mean.item()))
            approx_kl_epoch.append(float(approx_kl.item()))

            if TARGET_KL is not None and approx_kl.item() > TARGET_KL:
                print(
                    f"[ppo] early stop PPO epoch: approx_kl={approx_kl.item():.4f} "
                    f"> target_kl={TARGET_KL:.4f}"
                )
                early_stop_epoch = True
                break

        if early_stop_epoch:
            break

    mean_pg = float(np.mean(pg_losses_epoch)) if pg_losses_epoch else 0.0
    mean_vl = float(np.mean(v_losses_epoch)) if v_losses_epoch else 0.0
    mean_ent = float(np.mean(entropy_epoch)) if entropy_epoch else 0.0
    mean_update_kl = float(np.mean(approx_kl_epoch)) if approx_kl_epoch else 0.0

    pg_loss_history.append(mean_pg)
    v_loss_history.append(mean_vl)

    # -----------------------------------------------------------------------
    # (e) 日志
    # -----------------------------------------------------------------------
    if iter_num % LOG_EVERY == 0 or iter_num == PPO_MAX_ITERS - 1:
        dt = time.time() - t0
        mean_resp_len = float(np.mean(response_lens))
        var_resp_len = float(np.var(response_lens))

        print(
            f"[ppo] iter {iter_num:4d} "
            f"| RM {mean_raw_rm_iter:+.4f} "
            f"| ref_KL/token {mean_kl_iter:+.4f} "
            f"| update_KL {mean_update_kl:.4f} "
            f"| pg {mean_pg:+.4f} "
            f"| v {mean_vl:.4f} "
            f"| ent {mean_ent:.4f} "
            f"| resp_len mean={mean_resp_len:.1f} var={var_resp_len:.2f} "
            f"| {dt:.1f}s"
        )

    # -----------------------------------------------------------------------
    # (f) 非严格的 mode collapse 监控：仅打印警告，不停训
    #     教学项目目标是“端到端跑通”，因此关闭早停；指标仍可见，便于人工 debug。
    # -----------------------------------------------------------------------
    cur_var = float(np.var(response_lens))
    #模型生成的回答越来越缺少多样性，甚至对不同 Prompt 都倾向于输出相似的内容。
    if prev_var is not None:
        # 仅在训练中段（iter>=10）做一次警告，避免前几轮噪声干扰日志。
        if cur_var < 0.5 and prev_var < 0.5 and iter_num == 10:
            print(
                f"[ppo] NOTE: response-length variance stays low (var={cur_var:.2f}). "
                f"This usually means the policy rarely emits <eos>; "
                f"training continues for the remaining {PPO_MAX_ITERS - iter_num - 1} iters."
            )
    prev_var = cur_var

    # -----------------------------------------------------------------------
    # (g) checkpoint
    # -----------------------------------------------------------------------
    checkpoint = {
        # 保留 model_state 兼容“ppo.pt 直接拿 policy 做推理”的旧代码。
        "model_state": policy_model.state_dict(),
        "policy_state": policy_model.state_dict(),
        "value_state": value_model.state_dict(),
        "config": {
            k: v
            for k, v in vars(config).items()
            if not k.startswith("_")
        },
        "iter_num": iter_num,
        "mean_raw_rm_reward": mean_raw_rm_iter,
        "mean_ref_kl_per_token": mean_kl_iter,
        "mean_update_kl": mean_update_kl,
    }

    if mean_raw_rm_iter > best_raw_rm_reward:
        best_raw_rm_reward = mean_raw_rm_iter
        save_checkpoint(checkpoint, "checkpoints/ppo.pt")
        print(
            f"[ppo] saved best checkpoint "
            f"(raw_rm_reward={best_raw_rm_reward:+.4f})"
        )
    elif iter_num > 0 and iter_num % SAVE_EVERY == 0:
        save_checkpoint(checkpoint, "checkpoints/ppo_latest.pt")

    iter_num += 1


# ===========================================================================
# (9) 训练结束：最终保存
# ===========================================================================

total_wall = time.time() - t0

final_mean_reward = (
    float(np.mean(raw_reward_history[-10:]))
    if raw_reward_history else 0.0
)
final_mean_kl = (
    float(np.mean(kl_history[-10:]))
    if kl_history else 0.0
)
final_mean_pg = (
    float(np.mean(pg_loss_history[-10:]))
    if pg_loss_history else 0.0
)
final_mean_vl = (
    float(np.mean(v_loss_history[-10:]))
    if v_loss_history else 0.0
)

final_checkpoint = {
    "model_state": policy_model.state_dict(),
    "policy_state": policy_model.state_dict(),
    "value_state": value_model.state_dict(),
    "config": {
        k: v
        for k, v in vars(config).items()
        if not k.startswith("_")
    },
    "iter_num": iter_num,
    "mean_raw_rm_reward": final_mean_reward,
    "mean_ref_kl_per_token": final_mean_kl,
    "best_raw_rm_reward": best_raw_rm_reward,
}

# final 单独保存，避免覆盖 best ppo.pt。
save_checkpoint(final_checkpoint, "checkpoints/ppo_final.pt")

print(
    f"[ppo] === done === iter={iter_num} "
    f"| stopped_early={stopped_early} "
    f"| final RM(last10)={final_mean_reward:+.4f} "
    f"| final ref_KL/token={final_mean_kl:+.4f} "
    f"| final pg={final_mean_pg:+.4f} "
    f"| final value={final_mean_vl:.4f} "
    f"| best RM={best_raw_rm_reward:+.4f} "
    f"| wall={total_wall:.1f}s"
)

if len(raw_reward_history) >= 3:
    n = len(raw_reward_history)
    print(
        f"[ppo] RM trend: first 5 = {[f'{x:+.3f}' for x in raw_reward_history[:5]]} "
        f"| mid = {[f'{x:+.3f}' for x in raw_reward_history[max(0, n//2-2):n//2+3]]} "
        f"| last 5 = {[f'{x:+.3f}' for x in raw_reward_history[-5:]]}"
    )
    print(
        f"[ppo] ref_KL/token trend: first 5 = {[f'{x:+.3f}' for x in kl_history[:5]]} "
        f"| mid = {[f'{x:+.3f}' for x in kl_history[max(0, n//2-2):n//2+3]]} "
        f"| last 5 = {[f'{x:+.3f}' for x in kl_history[-5:]]}"
    )

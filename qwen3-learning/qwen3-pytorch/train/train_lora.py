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

【★ MoE 版（--model moe，feat-031）】
    MoE 之前**只验证到预训练**，微调链路一次没跑过。
    这一版做三件事：
      ① 支持 `--model dense|moe`：加载 `checkpoints/pretrain_{dense,moe}.pt`，
         分别建 `Qwen3Dense` / `Qwen3MoE`；
      ② 支持 `--lora-targets attn|attn+expert`：
         `attn+expert` 会把 LoRA **同时注入每个专家的 FFN**
         （gate_proj / up_proj / down_proj）——
         `inject_lora` 是按**子模块名**匹配的，所以不用改 LoRA 实现；
      ③ 打印**逐专家诊断**：每个专家这一步分到多少 token、
         路由权重多大、它的 LoRA 参数梯度范数多大、B 矩阵长到多大。
         ★ 这里有个**真问题**：路由器是**冻结**的，它决定哪个专家被用上 ⇒
         专家的 LoRA 参数有**三种**状态（本脚本会把它们分开计数）：
           ① `grad is None` —— 这一步**专家根本没参与前向**（没被路由到）
              ⇒ AdamW 整个跳过 ⇒ 参数**一动不动**（连 weight decay 都没有）
           ② `grad == 0.0` —— 参与了前向，但它处理的 token 不影响 loss
              （如序列最后一个位置，target 被 -100 忽略）
              ⇒ **没有学习信号**，只有 weight decay
           ③ `grad > 0` —— 真的学到了东西
         ⇒ 往全部专家注入 LoRA 时，有一部分参数是**白注入**的
         （拿不到任何学习信号，但显存与参数量照算）。
         实测：本仓 MoE（4 专家 / top-2）单步里，
         「完全不动」的比例在 6.2% ~ 12.5% 之间浮动。

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
import re
import sys
from typing import Dict, List, Optional, Tuple

# 把项目根补进 sys.path（脚本在 train/ 子目录里，sys.path[0] 不是项目根）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from config import DENSE_CONFIG, MOE_CONFIG
from data.preference_data import generate_preference_dataset
from data.tokenizer import BPETokenizer
from model.common.lora import (LoRALinear, inject_lora, mark_only_lora_trainable,
                               set_lora_enabled)
from model.qwen3_dense import Qwen3Dense
from model.qwen3_moe import Qwen3MoE
from utils import get_device, save_checkpoint, set_seed

# prompt 与 response 的分隔模板（与 data/preference_data.py::format_prompt_response 一致）
PROMPT_TEMPLATE = "### Question:\n{prompt}\n\n### Answer:\n"

# LoRA 目标模块集合**不在这里硬编码**，统一从 config.py 读（见 Qwen3Config）：
#   cfg.lora_target_modules        —— attention 版（dense 默认）
#   cfg.lora_target_modules_expert —— attention + 专家 FFN 版（MoE 默认）
# ★ `inject_lora` 是按**子模块名**（child_name）匹配的，不是全名 ⇒
#   `gate_proj` / `up_proj` / `down_proj` 会命中**每一个专家的 FFN**。


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


def pick_config(model_kind: str):
    """
    按名字选配置对象（dense / moe）。

    单独拆出来是为了让 `set_seed` 能**在建模型之前**调用 ——
    建模型与 LoRA 的 A 矩阵初始化都会消耗全局 RNG，
    种子设置的先后会改变 A 的初值，从而改变训练轨迹与最终指标。

    参数：
        model_kind: "dense" 或 "moe"
    返回：
        配置对象
    """
    if model_kind == "dense":
        return DENSE_CONFIG
    if model_kind == "moe":
        return MOE_CONFIG
    raise ValueError(f"未知的 model 类型：{model_kind}")


def build_model(model_kind: str, vocab_size: int, device: torch.device):
    """
    按名字建模型（dense / moe）。

    参数：
        model_kind: "dense" 或 "moe"
        vocab_size: 词表大小
        device:      设备
    返回：
        (model, cfg, ckpt_path, save_path)
        ckpt_path 是要加载的预训练权重，save_path 是 LoRA 权重的落盘位置。
    """
    cfg = pick_config(model_kind)
    if model_kind == "dense":
        model = Qwen3Dense(vocab_size=vocab_size, cfg=cfg, dropout=0.0)
        ckpt_path = os.path.join("checkpoints", "pretrain_dense.pt")
        save_path = os.path.join("checkpoints", "lora.pt")
    else:
        model = Qwen3MoE(vocab_size=vocab_size, cfg=cfg, dropout=0.0)
        ckpt_path = os.path.join("checkpoints", "pretrain_moe.pt")
        save_path = os.path.join("checkpoints", "lora_moe.pt")
    return model.to(device), cfg, ckpt_path, save_path


# 专家 LoRA 模块名形如 blocks.0.moe.experts.2.gate_proj
_EXPERT_NAME_RE = re.compile(r"blocks\.(\d+)\.moe\.experts\.(\d+)\.")


def collect_expert_stats(model: "torch.nn.Module") -> Dict[Tuple[int, int], dict]:
    """
    按 (层号, 专家号) 汇总专家侧 LoRA 分支的状态。

    返回：
        {(层, 专家): {"grad_a": A 的梯度 L2 范数,
                      "grad_b": B 的梯度 L2 范数,
                      "b":      B 矩阵自身的 L2 范数,
                      "n_none": 这一步 `grad is None` 的参数个数,
                      "n_zero_a": lora_A 梯度**恰好为 0** 的个数,
                      "n_zero_b": lora_B 梯度**恰好为 0** 的个数}}

    【★ 专家的 LoRA 参数其实有**三种**状态，必须分开看（feat-031 实测）】
        ① `grad is None`   —— 这一步**专家根本没参与前向**（没被路由到）。
                              路由器是冻结的 ⇒ 没选中的专家这一轮不进计算图。
                              ⇒ AdamW 会**整个跳过**这个参数：
                                 **连 weight decay 都不会发生**，参数**一动不动**。
        ② `grad == 0.0`    —— 专家参与了前向，但它处理的那几个 token
                              **不影响 loss**（例如序列**最后一个位置**：
                              它的 target 被 -100 忽略，没人预测它）。
                              ⇒ 梯度张量存在但**恰好是 0**：
                                 没有学习信号，但 AdamW 仍会对它做
                                 **weight decay**（`p *= (1 - lr·wd)`）。
        ③ `grad > 0`       —— 真的学到了东西。
        ⇒ 「白注入」= ① + ②：占着参数量与显存，却**拿不到任何学习信号**。

    【为什么要把 A / B 分开看】
        LoRA 分支的输出是 `(alpha/r) · x @ A.T @ B.T`：
            ∂L/∂B = (alpha/r) · (∂L/∂out)ᵀ · (x @ A.T)   ← B=0 时**非零**
            ∂L/∂A = (alpha/r) · (∂L/∂out)ᵀ · B · x        ← B=0 时**恰好是 0**
        ⇒ **第 0 步所有专家的 A 梯度都是 0**（不是 bug，是 B 零初始化的直接后果；
          第 0 步一更新，B≠0，A 就有真梯度了）。
        所以「梯度是不是 0」这个判断主要看 B，
        而「第 0 步 A 的零梯度」**不算**白注入。
    """
    out: Dict[Tuple[int, int], dict] = {}
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        m = _EXPERT_NAME_RE.search(name)
        if m is None:
            continue
        key = (int(m.group(1)), int(m.group(2)))
        st = out.setdefault(key, {"grad_a": 0.0, "grad_b": 0.0, "b": 0.0,
                                  "n_none": 0, "n_zero_a": 0, "n_zero_b": 0})
        for pname, p in module.named_parameters():
            if pname == "lora_A":
                if p.grad is None:
                    st["n_none"] += 1
                else:
                    st["grad_a"] += float(p.grad.detach().norm() ** 2)
                    if float(p.grad.detach().norm()) == 0.0:
                        st["n_zero_a"] += 1
            elif pname == "lora_B":
                if p.grad is None:
                    st["n_none"] += 1
                else:
                    st["grad_b"] += float(p.grad.detach().norm() ** 2)
                    if float(p.grad.detach().norm()) == 0.0:
                        st["n_zero_b"] += 1
                st["b"] += float(p.detach().norm() ** 2)
        st["grad_a"] = st["grad_a"] ** 0.5
        st["grad_b"] = st["grad_b"] ** 0.5
        st["b"] = st["b"] ** 0.5
    return out


def count_expert_lora_params(model: "torch.nn.Module", step: int) -> Tuple[int, int, int]:
    """
    统计专家侧 LoRA 参数的总量，以及两种「没学到东西」的参数量。

    参数：
        model: 模型
        step:  当前步号（用于区分「第 0 步 A 的零梯度」这个**暂时**现象）
    返回：
        (total_params, none_params, dead_params)
        - none_params：这一步 `grad is None`（专家没参与前向）⇒ **完全不动**
                       （连 weight decay 都不会发生）
        - dead_params：这一步梯度**恰好为 0**（参与了前向但不影响 loss）
                        ⇒ 没有学习信号，只有 weight decay。
                        ⚠️ 第 0 步的 lora_A 零梯度**不算**在内
                           （B 零初始化导致，下一步就有真梯度了）。
        两者相加就是「白注入」的参数量。
    """
    total = 0
    n_none = 0
    n_dead = 0
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        if _EXPERT_NAME_RE.search(name) is None:
            continue
        for pname, p in module.named_parameters():
            if pname not in ("lora_A", "lora_B"):
                continue
            total += p.numel()
            if p.grad is None:
                n_none += p.numel()
            elif float(p.grad.detach().norm()) == 0.0:
                if pname == "lora_B" or step > 0:
                    n_dead += p.numel()
    return total, n_none, n_dead


def collect_expert_weights(
    model: "torch.nn.Module",
    layer: int,
    n_experts: int,
) -> Dict[int, dict]:
    """
    从该层 MoE Block 的 `_last_stats` 还原「每个专家被选中时拿了多大的路由权重」。

    ★ 为什么需要它：专家 LoRA 的**梯度大小 ∝ 路由权重**。
      top-k 之后 `norm_topk_prob` 会把权重**重新归一化**到和为 1，
      所以「第二顺位」的专家拿到的权重可能极小（路由器很确信时能到 1e-6 量级），
      它的梯度就**小到打印不出来** ——
      这跟「根本没被路由到」（`grad is None`）是**两回事**，必须分开看。
      没有这一列，就会把「梯度小」误读成「没被路由到」。

    参数：
        model:      模型
        layer:      层号
        n_experts:  专家总数
    返回：
        {专家号: {"n": 被选中的 slot 数, "w_max": 最大路由权重, "w_mean": 平均路由权重}}
    """
    block = model.blocks[layer]
    st = getattr(block, "_last_stats", None)
    if st is None or "topk_indices" not in st:
        return {}
    w = st["topk_weights"]          # (N, K)
    idx = st["topk_indices"]        # (N, K)
    out: Dict[int, dict] = {}
    for e in range(n_experts):
        mask = (idx == e)
        n = int(mask.sum().item())
        if n == 0:
            out[e] = {"n": 0, "w_max": 0.0, "w_mean": 0.0}
        else:
            sel = w[mask]
            out[e] = {"n": n, "w_max": float(sel.max().item()),
                      "w_mean": float(sel.mean().item())}
    return out


def print_expert_report(
    tag: str,
    model: "torch.nn.Module",
    expert_load: Optional[List[float]],
    stats: Dict[Tuple[int, int], dict],
    n_layers: int,
    top_k: int,
) -> None:
    """
    打印逐专家诊断表（只打最后一层，最有代表性）。

    ★ `expert_load` 是**归一化负载**（`slot_counts / K`），**可能是小数**：
      e.g. 某个专家只拿到 1 个 slot、而 K=2 ⇒ 它的 load 是 0.5。
      所以**不能**用 `int(load) == 0` 判断「没被路由到」——
      那样会把 0.5 误判成 0（本仓真踩过，见 feat-031 的返工记录）。
      「有没有被路由到」的**精确**判据是 **LoRA 参数的 `grad is None`**。

    参数：
        tag:         标签（如 "step 0"）
        model:       模型（用来读该层的路由权重）
        expert_load: 最后一层每个专家的归一化负载（来自 forward 的 extras）
        stats:       collect_expert_stats 的返回值
        n_layers:    层数
        top_k:       每个 token 路由到几个专家（用来还原真实 token 数）
    """
    if expert_load is None:
        return
    layer = n_layers - 1
    if not any(k[0] == layer for k in stats):
        # 本次没往专家 FFN 注入 LoRA（`--lora-targets attn`）⇒ 表里必然全是 0
        print(f"[lora-moe] 逐专家诊断（{tag}）：本次**没往专家 FFN 注入 LoRA**"
              f"（--lora-targets attn）⇒ 专家侧没有 LoRA 参数，下表无意义")
        return
    weights = collect_expert_weights(model, layer, len(expert_load))
    print(f"[lora-moe] 逐专家诊断（{tag}，最后一层）")
    print(f"  {'专家':<5}{'路由token':>11}{'路由权重max':>13}{'grad(A)':>13}{'grad(B)':>13}"
          f"{'B范数':>12}{'无梯度(None)':>14}{'零梯度(A)':>11}{'零梯度(B)':>11}")
    unrouted = []
    zero_b = []
    for e, load in enumerate(expert_load):
        st = stats.get((layer, e), {"grad_a": 0.0, "grad_b": 0.0, "b": 0.0,
                                    "n_none": 0, "n_zero_a": 0, "n_zero_b": 0})
        wt = weights.get(e, {"w_max": 0.0})
        n_tok = load * top_k
        print(f"  {e:<5}{n_tok:>11.1f}{wt['w_max']:>13.3e}{st['grad_a']:>13.6f}"
              f"{st['grad_b']:>13.6f}{st['b']:>12.6f}{st['n_none']:>14}"
              f"{st['n_zero_a']:>11}{st['n_zero_b']:>11}")
        if st["n_none"] > 0:
            unrouted.append(e)
        elif st["n_zero_b"] > 0:
            zero_b.append(e)
    if unrouted:
        print(f"  ⚠️ **没参与前向**的专家：{unrouted} ⇒ 它们的 LoRA 参数 `grad is None`"
              f"（**连 weight decay 都不会有**，参数一动不动）——"
              f"路由器是冻结的，没选中的专家这一轮完全不参与前向")
    if zero_b:
        print(f"  ⚠️ **梯度恰好为 0** 的专家：{zero_b} ⇒ 它们参与了前向，"
              f"但处理的 token 不影响 loss（如序列最后一个位置，target 被 -100 忽略）"
              f"⇒ 没有学习信号，只有 weight decay")
    print(f"  注 1：第 0 步 A 的梯度**恒为 0**（B 零初始化 ⇒ ∂L/∂A 含因子 B），"
          f"那是**暂时**现象（下一步 B≠0 就有真梯度），所以「有没有梯度」要看 B 那一列。")
    print(f"  注 2：梯度大小 ∝ **路由权重** —— 权重被 top-k 重新归一化后，"
          f"第二顺位的专家可能只拿到 1e-6 量级的权重 ⇒ 梯度极小但**不为 None**。")


def main() -> None:
    """LoRA 微调主流程。"""
    parser = argparse.ArgumentParser(description="Qwen3 LoRA 微调")
    parser.add_argument("--max-iters", type=int, default=None,
                        help="覆盖 config 的 lora_max_iters")
    parser.add_argument("--model", choices=["dense", "moe"], default="dense",
                        help="用哪个预训练权重做微调（feat-031 新增 moe）")
    parser.add_argument("--lora-targets", choices=["attn", "attn+expert"], default=None,
                        help="LoRA 注入哪些模块；默认 dense=attn、moe=attn+expert")
    args = parser.parse_args()

    model_kind = args.model
    if args.lora_targets is None:
        lora_targets = (DENSE_CONFIG.lora_target_modules if model_kind == "dense"
                        else DENSE_CONFIG.lora_target_modules_expert)
    elif args.lora_targets == "attn":
        lora_targets = DENSE_CONFIG.lora_target_modules
    else:
        lora_targets = DENSE_CONFIG.lora_target_modules_expert

    tk = BPETokenizer(os.path.join("data", "vocab.json"), os.path.join("data", "merges.txt"))

    # --- 随机种子：必须在**建模型之前**设 ---
    #   建模型与 LoRA 的 A 矩阵（kaiming_uniform_）都会消耗全局 RNG；
    #   种子设置的先后会改变 A 的初值 ⇒ 改变训练轨迹与最终指标。
    #   （feat-031 实测：把 set_seed 挪到建模型之后，dense 的胜率 0.605 → 0.600。）
    cfg = pick_config(model_kind)
    set_seed(cfg.seed)
    device = get_device()
    max_iters = args.max_iters if args.max_iters is not None else cfg.lora_max_iters
    print(f"[lora] model={model_kind} device={device} max_iters={max_iters}")

    # --- 建模型 + 加载预训练权重（必须在注入 LoRA 之前加载：注入后模块名会变）---
    model, cfg, ckpt_path, save_path = build_model(model_kind, tk.vocab_size, device)

    if not os.path.exists(ckpt_path):
        print(f"[lora] 找不到 {ckpt_path}，请先跑：python train/train_pretrain.py --model {model_kind}")
        return
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    print(f"[lora] loaded {ckpt_path} (vocab_size={tk.vocab_size})")

    # --- 注入 LoRA + 只留 A/B 可训 ---
    replaced = inject_lora(model, lora_targets, r=cfg.lora_r,
                           alpha=cfg.lora_alpha, dropout=cfg.lora_dropout)
    trainable = mark_only_lora_trainable(model)
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[lora] injected {len(replaced)} modules "
          f"(targets={list(lora_targets)})")
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

            _, loss, _, extras = model(inputs, targets=targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # --- MoE 专属：逐专家诊断（梯度此时已回传但尚未 step）---
            if model_kind == "moe" and (step == 0 or step == max_iters - 1):
                print_expert_report(
                    f"step {step}",
                    model,
                    extras.get("expert_load_last"),
                    collect_expert_stats(model),
                    cfg.n_layer,
                    cfg.experts_per_tok,
                )
                total_p, none_p, dead_p = count_expert_lora_params(model, step)
                if total_p:
                    print(f"  ⇒ 专家侧的 LoRA 参数共 {total_p:,} 个："
                          f"{none_p:,} 个这一步 `grad is None`（专家没参与前向 ⇒ "
                          f"**完全不动**）+ {dead_p:,} 个梯度**恰好为 0**"
                          f"（参与了前向但不影响 loss ⇒ 只有 weight decay，没有学习信号）"
                          f" ⇒ 合计 {(none_p+dead_p)/total_p:.1%} 是**白注入**的")

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
         "model_kind": model_kind,
         "lora": {"r": cfg.lora_r, "alpha": cfg.lora_alpha,
                  "target_modules": list(lora_targets)},
         "metrics": {"before": before, "after": after},
         "win_rate": after["win_rate"], "base_win_rate": before["win_rate"],
         "config": cfg.to_dict()},
        save_path,
    )
    print(f"[lora] saved -> {path}")


if __name__ == "__main__":
    main()

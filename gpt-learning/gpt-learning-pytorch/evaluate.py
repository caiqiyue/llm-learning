"""
9 个评估函数的统一入口（PyTorch 端）。

实现一览：
  P3: eval_perplexity
  P4: eval_distinct_n, eval_self_bleu
  P5: eval_pairwise_accuracy
  P7:
    - eval_token_entropy   （P4 stub 的完整实现：每步生成时记 logits 算熵）
    - eval_response_quality（rule-based：长度、关键词、动词、标点）
    - eval_sft_vs_ppo      （同一 prompt 下 SFT vs PPO 全面对比：diversity + quality + RM 分）
    - eval_benchmark       （Toy 4 选 1 多选准确率，likelihood-based 评分）
    - eval_speedup         （naive vs KV-cache 速度对比）

统一入口 argparse（plan P7.1.3 / design §6）：
    --stage {pretrain, sft, reward, ppo, compare, speedup, all}
    --checkpoint <ckpt>  --sft <sft.pt> --ppo <ppo.pt> --rm <rm.pt>

详细 spec：design §6、gpt2_training_explained.md Part 8、implementation-plan P7.1.3。
"""

# -*- coding: utf-8 -*-

import argparse          # 唯一允许的入口抽象（plan P7.1.3 明确允许）
import math              # exp(loss) 算 PPL
import os                # 路径、文件存在性
import time              # speedup 测时
import re                # tokenize 文本（防御）

import numpy as np       # mean / var

import torch             # 主框架
import torch.nn as nn    # RM 的 value_head 是 nn.Linear
import torch.nn.functional as F  # softmax / log_softmax

import config            # 超参
from utils import load_checkpoint
from model.gpt2 import GPT2
from data.pretrain_data import make_pretrain_loader, ENTITIES
from data.sft_data import make_sft_loader, get_ppo_prompts
from data.preference_data import make_preference_loader
from data.benchmark_data import load_benchmark
from data.tokenizer import CharTokenizer, EOS_ID


# ===========================================================================
# 通用：模型加载器（P3 / P4 / P5 / P7 共用）
# ===========================================================================
def _load_model(checkpoint_path: str, with_value_head: bool = False, device: str = "cpu"):
    """
    统一模型加载器：返回 (model, tokenizer, vocab_size)。

    参数：
        checkpoint_path: 模型 checkpoint 路径
        with_value_head: True → 加载带 value_head 的 RM/PPO 权重
        device:          'cpu' / 'cuda'
    """
    tokenizer = CharTokenizer(vocab_path=os.path.join("data", "vocab.txt"))
    vocab_size = tokenizer.vocab_size

    model = GPT2(
        vocab_size=vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        n_inner=config.n_inner,
        block_size=config.block_size,
        dropout=0.0,
        bias=config.bias,
    )

    if with_value_head:
        # RM / PPO checkpoint 都有 value_head 字段
        model.value_head = nn.Linear(model.n_embd, 1, bias=True)
        ckpt = load_checkpoint(checkpoint_path)
        model.load_state_dict(ckpt["model_state"])
    else:
        # SFT / pretrain checkpoint
        ckpt = load_checkpoint(checkpoint_path)
        # 防御性：过滤掉 value_head 字段（PPO ckpt 混着也不会崩）
        state = {k: v for k, v in ckpt["model_state"].items() if not k.startswith("value_head.")}
        model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()
    return model, tokenizer, vocab_size


# ===========================================================================
# 1. eval_perplexity（P3）
# ===========================================================================
def eval_perplexity(checkpoint_path: str, device: str = "cpu") -> dict:
    """
    在 pretrain 验证集上计算 perplexity。
    PPL = exp(mean cross_entropy_loss)

    【关键形状变化】
    ┌──────────────────────────────────────────────────────────┐
    │ vx, vy:    (B, T)            输入 token id + target      │
    │ logits:    (B, T, V)         模型输出                     │
    │ loss:      scalar             per-token cross_entropy 平均│
    │ avg_loss:  scalar             整个 val set 加权平均       │
    │ ppl:       scalar             exp(avg_loss)               │
    └──────────────────────────────────────────────────────────┘
    解读：PPL=20 表示模型"相当于从 20 个候选里挑"；random baseline = V = 45

    参数：
        checkpoint_path: 预训练 checkpoint 路径
        device:          'cpu' / 'cuda'
    返回：
        dict
    """
    model, _, _ = _load_model(checkpoint_path, with_value_head=False, device=device)

    val_loader = make_pretrain_loader(
        split="val", batch_size=config.batch_size, block_size=config.block_size, seed=config.seed,
    )
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for vx, vy in val_loader:
            # vx, vy: (B, T) LongTensor，B=batch_size，T=block_size
            vx, vy = vx.to(device), vy.to(device)
            # logits: (B, T, V)  loss: scalar
            _, loss, _ = model(vx, targets=vy)
            # 用真实 token 数加权（pretrain 没 padding，但保持一致）
            n = (vy != -100).sum().item()   # (B, T) → scalar
            total_loss += loss.item() * n   # 累计加权 loss
            total_tokens += n               # 累计 token 数
    # 加权平均 loss
    avg_loss = total_loss / max(total_tokens, 1)   # scalar
    return {
        "stage": "pretrain",
        "metric": "perplexity",
        "value": math.exp(avg_loss),     # exp(loss) = PPL
        "loss": avg_loss,
        "perplexity": math.exp(avg_loss),
        "n_tokens": total_tokens,
        "checkpoint": checkpoint_path,
    }


# ===========================================================================
# 2. 响应生成辅助（P4 起复用）
# ===========================================================================
def _generate_responses(
    model: GPT2,
    tokenizer: CharTokenizer,
    prompts,
    max_new_tokens: int = 64,
    device: str = "cpu",
) -> list:
    """
    对一组 prompt 字符串生成 response（不含 prompt），返回 list[str]。
    走 GPT2.generate() 朴素路径（不依赖 KV cache，CPU 友好）。

    【feat-027】切 response 的规则：**按 token id 个数切**，不按字符串长度切
    （UNK 解码回来是 5 字符的 "<unk>"，按长度切会错位）。
    """
    responses = []
    safe_top_k = min(config.top_k, tokenizer.vocab_size - 1)
    for prompt_str in prompts:
        full_prompt = f"<bos> {prompt_str} <sep> "
        ids = tokenizer.encode(full_prompt)
        if not ids:
            responses.append("")
            continue
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(
                idx, max_new_tokens=max_new_tokens,
                temperature=config.temperature, top_k=safe_top_k, top_p=config.top_p,
                eos_id=EOS_ID,
            )
        # 【feat-027】按 **token id 个数**切出回答，不按字符串长度切。
        #   字符级 tokenizer 里 UNK（id=0）解码回来是整条 "<unk>"（5 字符），
        #   用 len(full_prompt) 切会错位 ⇒ 每条 response 都被塞上 "? <unk>sep<unk> " 前缀。
        new_ids = out[0].tolist()[len(ids):]
        if EOS_ID in new_ids:            # 双保险：generate 一般已在 EOS 处停住
            new_ids = new_ids[: new_ids.index(EOS_ID)]
        responses.append(tokenizer.decode(new_ids))
    return responses


# ===========================================================================
# 3. eval_distinct_n（P4）
# ===========================================================================
def eval_distinct_n(checkpoint_path: str, n_gram: int = 1, n_samples: int = 50, device: str = "cpu") -> dict:
    """
    char-level n-gram 多样性 = unique ngram / total ngram。
    越接近 1 越多样，越接近 0 越单调。
    """
    model, tokenizer, _ = _load_model(checkpoint_path, with_value_head=False, device=device)
    prompts = [p["prompt"] for p in get_ppo_prompts(n=n_samples, seed=config.eval_seed)]
    responses = _generate_responses(model, tokenizer, prompts, max_new_tokens=config.max_new_tokens, device=device)

    all_ngrams = []
    for r in responses:
        if len(r) < n_gram:
            continue
        ngrams = [r[i:i + n_gram] for i in range(len(r) - n_gram + 1)]
        all_ngrams.extend(ngrams)
    value = len(set(all_ngrams)) / max(len(all_ngrams), 1)
    return {
        "stage": "sft", "metric": f"distinct_{n_gram}", "value": value,
        "n_samples": n_samples, "n_total_ngrams": len(all_ngrams),
    }


# ===========================================================================
# 4. eval_self_bleu（P4）
# ===========================================================================
def eval_self_bleu(checkpoint_path: str, n_samples: int = 50, device: str = "cpu") -> dict:
    """
    Self-BLEU：每条 response 对其他 N-1 条的 BLEU 平均。越低越多样。
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except Exception as e:
        return {"stage": "sft", "metric": "self_bleu", "value": 0.0, "n_samples": n_samples, "error": str(e)}

    model, tokenizer, _ = _load_model(checkpoint_path, with_value_head=False, device=device)
    prompts = [p["prompt"] for p in get_ppo_prompts(n=n_samples, seed=config.eval_seed)]
    responses = _generate_responses(model, tokenizer, prompts, max_new_tokens=config.max_new_tokens, device=device)
    tokenized = [list(r) for r in responses]
    smoothing = SmoothingFunction().method1
    bleu_scores = []
    for i, hyp in enumerate(tokenized):
        if not hyp:
            continue
        refs = [tokenized[j] for j in range(len(tokenized)) if j != i and tokenized[j]]
        if not refs:
            continue
        try:
            bleu_scores.append(sentence_bleu(refs, hyp, smoothing_function=smoothing))
        except Exception:
            bleu_scores.append(0.0)
    value = sum(bleu_scores) / max(len(bleu_scores), 1)
    return {"stage": "sft", "metric": "self_bleu", "value": value, "n_samples": n_samples}


# ===========================================================================
# 5. eval_token_entropy（P7：完整实现）
# ===========================================================================
def eval_token_entropy(checkpoint_path: str, n_samples: int = 20, max_new_tokens: int = 64, device: str = "cpu") -> dict:
    """
    在生成过程中记录每步的 next-token 分布熵并求平均。
    H(p) = -Σ p_i log p_i

    高熵 = 模型不确定 = 多样但可能不可靠
    低熵 = 模型确定 = 一致但可能呆板

    参数：
        checkpoint_path: 模型权重
        n_samples:       取多少条 prompt
        max_new_tokens:  每条最多生成 token 数
    返回：
        dict
    """
    model, tokenizer, _ = _load_model(checkpoint_path, with_value_head=False, device=device)
    prompts = [p["prompt"] for p in get_ppo_prompts(n=n_samples, seed=config.eval_seed)]

    per_sample_entropies = []
    for prompt_str in prompts:
        full_prompt = f"<bos> {prompt_str} <sep> "
        prompt_ids = tokenizer.encode(full_prompt)
        if not prompt_ids:
            continue
        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        ent_sum = 0.0
        ent_n = 0
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 截断到 block_size
                if x.size(1) > model.block_size:
                    x = x[:, -model.block_size:]
                out = model(x)
                if len(out) == 4:
                    logits = out[0]
                else:
                    logits = out[0]
                last_logits = logits[:, -1, :] / max(config.temperature, 1e-6)
                p = F.softmax(last_logits, dim=-1)
                # 数值稳定：把 p=0 的位置 log(0) 替换为 log(eps) 后再相加
                H = -(p * (p + 1e-10).log()).sum(dim=-1).item()
                ent_sum += H
                ent_n += 1
                # 采样下一个 token
                next_id = torch.multinomial(p, num_samples=1)
                if next_id.item() == EOS_ID:
                    break
                x = torch.cat([x, next_id], dim=1)
        if ent_n > 0:
            per_sample_entropies.append(ent_sum / ent_n)

    value = sum(per_sample_entropies) / max(len(per_sample_entropies), 1)
    return {
        "stage": "sft",
        "metric": "token_entropy",
        "value": value,
        "n_samples": len(per_sample_entropies),
        "max_new_tokens": max_new_tokens,
    }


# ===========================================================================
# 6. eval_response_quality（P7：rule-based）
# ===========================================================================
def eval_response_quality(checkpoint_path: str, n_samples: int = 50, device: str = "cpu") -> dict:
    """
    Rule-based 质量评分（满分 1.0）：
      1) 长度在 10-50 字符（+0.25）
      2) 包含 prompt 中提到的动物名（+0.25）
      3) 包含至少一个常见动词（+0.25）
      4) 以 . ! ? 之一结尾（+0.25）
    """
    model, tokenizer, _ = _load_model(checkpoint_path, with_value_head=False, device=device)
    prompts = [p["prompt"] for p in get_ppo_prompts(n=n_samples, seed=config.eval_seed)]
    responses = _generate_responses(model, tokenizer, prompts, max_new_tokens=config.max_new_tokens, device=device)

    common_verbs = {"play", "run", "jump", "sleep", "sing", "dance", "explore", "rest", "watch", "listen"}
    animals = set(ENTITIES["animal"])

    scores = []
    for prompt_str, resp in zip(prompts, responses):
        s = 0.0
        if 10 <= len(resp) <= 50:
            s += 0.25
        if any(a in resp for a in animals):
            s += 0.25
        if any(v in resp for v in common_verbs):
            s += 0.25
        if resp.rstrip().endswith((".", "!", "?")):
            s += 0.25
        scores.append(s)
    value = sum(scores) / max(len(scores), 1)
    return {"stage": "sft", "metric": "response_quality", "value": value, "n_samples": n_samples}


# ===========================================================================
# 7. eval_pairwise_accuracy（P5 RM 评估）
# ===========================================================================
def eval_pairwise_accuracy(checkpoint_path: str, n_samples: int = 200, device: str = "cpu") -> dict:
    """
    在 preference 验证集上计算 RM 的成对准确率。

    【关键形状变化】
    ┌──────────────────────────────────────────────────────────────┐
    │ vc, vr:   (B, L)              chosen / rejected token id 序列 │
    │ r_c, r_r: (B,)                RM 给每个样本打的标量 reward    │
    │ (r_c > r_r): (B,) bool        RM 是否答对了                  │
    │ correct:   int                答对数量（sum）                │
    │ pair_acc:  float              correct / total               │
    └──────────────────────────────────────────────────────────────┘
    解读：pair_acc = 0.5 是猜，0.7+ 说明 RM 学到了偏好
    """
    model, _, _ = _load_model(checkpoint_path, with_value_head=True, device=device)
    val_loader = make_preference_loader(
        split="val", batch_size=config.rm_batch_size, seed=config.seed,
    )

    correct, total = 0, 0
    loss_sum, loss_count = 0.0, 0
    r_c_sum, r_r_sum = 0.0, 0.0

    with torch.no_grad():
        for i, vb in enumerate(val_loader):
            if i >= n_samples:
                break
            # vc, vr: (B, L)  L = 该 batch 的最大长度（可能超过 block_size）
            vc = vb["chosen_input_ids"].to(device)
            vr = vb["rejected_input_ids"].to(device)
            # 防御：超过 block_size 的尾部截断
            if vc.size(1) > config.block_size:
                vc = vc[:, : config.block_size]
            if vr.size(1) > config.block_size:
                vr = vr[:, : config.block_size]
            # 模型 forward（带 value_head，不算 LM loss）
            # r_c, r_r: (B,) 标量 reward
            _, _, _, r_c = model(vc, targets=None)
            _, _, _, r_r = model(vr, targets=None)
            # 答对的样本数（r_c > r_r 是 bool (B,)）
            correct += (r_c > r_r).sum().item()   # int
            total += vc.size(0)                    # int
            # Bradley-Terry loss（同训练时）
            # bt: scalar
            bt = -F.logsigmoid(r_c - r_r).mean()
            loss_sum += bt.item() * vc.size(0)    # 加权累计
            loss_count += vc.size(0)
            r_c_sum += r_c.sum().item()           # 累计 chosen reward
            r_r_sum += r_r.sum().item()           # 累计 rejected reward

    pair_acc = correct / max(total, 1)
    val_loss = loss_sum / max(loss_count, 1)
    return {
        "stage": "reward", "metric": "pairwise_accuracy",
        "value": pair_acc, "val_loss": val_loss,
        "n_samples": total,
        "mean_r_chosen": r_c_sum / max(total, 1),
        "mean_r_rejected": r_r_sum / max(total, 1),
        "margin": (r_c_sum - r_r_sum) / max(total, 1),
        "checkpoint": checkpoint_path,
    }


# ===========================================================================
# 8. eval_benchmark（P7：Toy 4 选 1）
# ===========================================================================
def eval_benchmark(checkpoint_path: str, n_questions: int = 100, device: str = "cpu") -> dict:
    """
    Toy 4 选 1 多选准确率。
    对每个 question 的每个 choice，算 choice 在 prompt 下的对数似然，
    选 argmax 作为预测，与 answer_idx 对比。

    【关键形状变化】（理解 Likelihood 评分法的核心）
    ┌────────────────────────────────────────────────────────────────┐
    │ 输入: question + 4 choices                                       │
    │ 对每个 choice:                                                  │
    │   text = "<bos> {question} <sep> {choice} <eos>"                │
    │   ids: (L,)                                                    │
    │   x: (1, L)                                                    │
    │   logits: (1, L, V)                                            │
    │   log_probs = log_softmax(logits[0, :-1]): (L-1, V)            │
    │   target_ids = x[0, 1:]: (L-1,)                                │
    │   token_logps = gather: (L-1,)   每个位置"预测下一 token" 的对数概率│
    │   score = sum(token_logps[response_start-1:])                  │
    │           = response 部分的累计 log P                           │
    │ 4 个 choice 各得一个 score，argmax 选最大                       │
    └────────────────────────────────────────────────────────────────┘
    """
    model, tokenizer, _ = _load_model(checkpoint_path, with_value_head=False, device=device)
    questions = load_benchmark(os.path.join("data", "benchmark.json"))[:n_questions]

    correct = 0
    detail_log = []
    for q in questions:
        question = q["question"]
        choices = q["choices"]
        answer_idx = q["answer_idx"]
        scores = []
        for choice in choices:
            # 拼成完整文本：<bos> {question} <sep> {choice} <eos>
            text = f"<bos> {question} <sep> {choice} <eos>"
            ids = tokenizer.encode(text)
            if not ids or len(ids) < 2:
                scores.append(-1e9)
                continue
            # 截断到 block_size
            if len(ids) > config.block_size:
                ids = ids[: config.block_size]
            # x: (1, L)  L ≤ block_size
            x = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(x)
                logits = out[0] if len(out) == 3 else out[0]
            # logits: (1, L, V)
            # === 关键：response 起点 = "<bos> {question} <sep> " 的 token 数 ===
            #    只对 response 部分（不含 prompt）算 log P
            prompt_part = f"<bos> {question} <sep> "
            response_start = len(tokenizer.encode(prompt_part))
            response_start = min(response_start, len(ids) - 1)
            # === log_softmax + 错位 1 位 gather ===
            #    log_probs: (L-1, V)  去掉最后 1 位
            log_probs = F.log_softmax(logits[0, :-1], dim=-1)
            #    target_ids: (L-1,)  左移 1 位（"被预测的目标"）
            target_ids = x[0, 1:]
            #    token_logps[b, t] = log_probs[b, t, target_ids[b, t]]
            #                       = log P(target_ids[b, t] | 前文)
            #    形状: (L-1,) → squeeze → (L-1,)
            token_logps = log_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
            # === 只对 response 部分求和 → 该 choice 的总 log P ===
            #    response_start - 1 是因为 token_logps 索引比原始 ids 少 1
            score = token_logps[max(response_start - 1, 0):].sum().item()
            scores.append(score)
        # 选 score 最大的 choice 作为预测
        pred = int(np.argmax(scores))
        if pred == answer_idx:
            correct += 1
        if len(detail_log) < 3:
            detail_log.append({
                "question": question, "choices": choices,
                "scores": [round(s, 2) for s in scores],
                "pred": pred, "answer_idx": answer_idx,
            })

    acc = correct / max(len(questions), 1)
    return {
        "stage": "benchmark",
        "metric": "benchmark_accuracy",
        "value": acc,
        "n_questions": len(questions),
        "n_correct": correct,
        "checkpoint": checkpoint_path,
    }


# ===========================================================================
# 9. eval_speedup（P7：naive vs KV cache）
# ===========================================================================
def eval_speedup(checkpoint_path: str, n_runs: int = 5, max_new_tokens: int = 64, device: str = "cpu") -> dict:
    """
    对比 generate_naive vs generate_kv 的耗时。
    使用 generate.benchmark_speed 接口。
    """
    from generate import benchmark_speed, generate_text

    model, tokenizer, _ = _load_model(checkpoint_path, with_value_head=False, device=device)
    prompt_str = "Once upon a time, the cat went to the forest. The cat saw a"
    prompt_ids = tokenizer.encode(prompt_str)
    if not prompt_ids:
        return {"metric": "speedup", "value": 0.0, "error": "empty prompt"}
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # benchmark_speed 内部会自动 build GPT2WithKVCache
    res = benchmark_speed(
        model, prompt_tensor, n_runs=n_runs, max_new_tokens=max_new_tokens,
        temperature=config.temperature, eos_id=EOS_ID,
    )
    return {
        "stage": "speedup", "metric": "speedup",
        "value": res["speedup"],
        "naive_avg_ms": res["naive_avg_ms"],
        "kv_avg_ms": res["kv_avg_ms"],
        "naive_tokens": res["naive_tokens"],
        "kv_tokens": res["kv_tokens"],
        "n_runs": n_runs,
        "max_new_tokens": max_new_tokens,
    }


# ===========================================================================
# 10. eval_sft_vs_ppo（P7：完整版）
# ===========================================================================
def eval_sft_vs_ppo(
    sft_ckpt: str,
    ppo_ckpt: str,
    rm_ckpt: str = None,
    n_samples: int = 30,
    device: str = "cpu",
) -> dict:
    """
    同一 prompt 下 SFT vs PPO 全面对比：
      - distinct_1/2 (多样性)
      - response_quality (rule-based)
      - mean RM score (若有 RM)

    参数：
        sft_ckpt: SFT checkpoint 路径
        ppo_ckpt: PPO checkpoint 路径
        rm_ckpt:  RM checkpoint 路径；None 时跳过 RM 评分
        n_samples: 多少条 prompt
    """
    sft_model, tokenizer, _ = _load_model(sft_ckpt, with_value_head=False, device=device)
    ppo_model, _, _ = _load_model(ppo_ckpt, with_value_head=False, device=device)
    rm_model = None
    if rm_ckpt and os.path.exists(rm_ckpt):
        rm_model, _, _ = _load_model(rm_ckpt, with_value_head=True, device=device)

    prompts = [p["prompt"] for p in get_ppo_prompts(n=n_samples, seed=config.eval_seed)]
    sft_resps = _generate_responses(sft_model, tokenizer, prompts, max_new_tokens=config.max_new_tokens, device=device)
    ppo_resps = _generate_responses(ppo_model, tokenizer, prompts, max_new_tokens=config.max_new_tokens, device=device)

    def _distinct(responses, n_gram):
        all_ng = []
        for r in responses:
            if len(r) < n_gram:
                continue
            all_ng.extend([r[i:i + n_gram] for i in range(len(r) - n_gram + 1)])
        return len(set(all_ng)) / max(len(all_ng), 1)

    sft_d1 = _distinct(sft_resps, 1)
    sft_d2 = _distinct(sft_resps, 2)
    ppo_d1 = _distinct(ppo_resps, 1)
    ppo_d2 = _distinct(ppo_resps, 2)

    # response_quality 直接复用上面的逻辑
    def _quality(responses, prompts):
        common_verbs = {"play", "run", "jump", "sleep", "sing", "dance", "explore", "rest", "watch", "listen"}
        animals = set(ENTITIES["animal"])
        scores = []
        for prompt_str, resp in zip(prompts, responses):
            s = 0.0
            if 10 <= len(resp) <= 50:
                s += 0.25
            if any(a in resp for a in animals):
                s += 0.25
            if any(v in resp for v in common_verbs):
                s += 0.25
            if resp.rstrip().endswith((".", "!", "?")):
                s += 0.25
            scores.append(s)
        return sum(scores) / max(len(scores), 1)

    sft_q = _quality(sft_resps, prompts)
    ppo_q = _quality(ppo_resps, prompts)

    sft_rm = ppo_rm = None
    if rm_model is not None:
        def _score(responses):
            ss = []
            for p, r in zip(prompts, responses):
                full = f"<bos> {p} <sep> {r} <eos>"
                ids = tokenizer.encode(full)
                if not ids:
                    continue
                if len(ids) > config.block_size:
                    ids = ids[: config.block_size]
                with torch.no_grad():
                    _, _, _, rr = rm_model(torch.tensor([ids], dtype=torch.long, device=device), targets=None)
                ss.append(float(rr.item()))
            return float(np.mean(ss)) if ss else 0.0
        sft_rm = _score(sft_resps)
        ppo_rm = _score(ppo_resps)

    return {
        "stage": "compare", "metric": "sft_vs_ppo",
        "n_samples": n_samples,
        "sft_distinct_1": sft_d1, "sft_distinct_2": sft_d2,
        "ppo_distinct_1": ppo_d1, "ppo_distinct_2": ppo_d2,
        "sft_response_quality": sft_q,
        "ppo_response_quality": ppo_q,
        "sft_rm_score": sft_rm, "ppo_rm_score": ppo_rm,
        "improvement": (ppo_rm - sft_rm) if (sft_rm is not None and ppo_rm is not None) else None,
    }


# ===========================================================================
# 11. eval_ppo（P6/P7：PPO 阶段评估汇总）
# ===========================================================================
def eval_ppo(checkpoint_path: str, n_samples: int = 50, device: str = "cpu") -> dict:
    """PPO 阶段评估：distinct_1, response_quality, response length 统计。"""
    model, tokenizer, _ = _load_model(checkpoint_path, with_value_head=False, device=device)
    prompts = [p["prompt"] for p in get_ppo_prompts(n=n_samples, seed=config.eval_seed)]
    responses = _generate_responses(model, tokenizer, prompts, max_new_tokens=config.max_new_tokens, device=device)

    # diversity
    all_ng = [c for r in responses for c in r]
    d1 = len(set(all_ng)) / max(len(all_ng), 1)
    lens = [len(r) for r in responses]
    mean_len = float(np.mean(lens))
    var_len = float(np.var(lens))

    # quality
    q = eval_response_quality(checkpoint_path, n_samples=n_samples, device=device)["value"]

    return {
        "stage": "ppo", "metric": "diversity",
        "distinct_1": d1, "mean_response_length": mean_len, "var_response_length": var_len,
        "response_quality": q, "n_samples": n_samples,
    }


# ===========================================================================
# 12. eval_sft（P4/P7：SFT 阶段评估汇总）
# ===========================================================================
def eval_sft(checkpoint_path: str, device: str = "cpu") -> dict:
    """SFT 阶段：distinct_1/2/3 + self_bleu + token_entropy + response_quality + benchmark。"""
    print(f"[eval] stage=sft checkpoint={checkpoint_path}")
    results = {}
    for n in (1, 2, 3):
        r = eval_distinct_n(checkpoint_path, n_gram=n, n_samples=50, device=device)
        results[f"distinct_{n}"] = r["value"]
        print(f"[eval]   distinct_{n}   = {r['value']:.4f}")

    rsb = eval_self_bleu(checkpoint_path, n_samples=50, device=device)
    results["self_bleu"] = rsb["value"]
    print(f"[eval]   self_bleu    = {rsb['value']:.4f}")

    rte = eval_token_entropy(checkpoint_path, n_samples=20, device=device)
    results["token_entropy"] = rte["value"]
    print(f"[eval]   token_entropy= {rte['value']:.4f}")

    rq = eval_response_quality(checkpoint_path, n_samples=50, device=device)
    results["response_quality"] = rq["value"]
    print(f"[eval]   response_quality= {rq['value']:.4f}")

    rb = eval_benchmark(checkpoint_path, n_questions=100, device=device)
    results["benchmark_accuracy"] = rb["value"]
    print(f"[eval]   benchmark_accuracy= {rb['value']:.4f}")

    return results


# ===========================================================================
# 入口：argparse（plan P7.1.3 明确允许 evaluate.py 使用 argparse）
# ===========================================================================
parser = argparse.ArgumentParser(description="GPT-2 RLHF 9 个评估函数的统一入口（PyTorch 端）")
parser.add_argument(
    "--stage", required=True,
    choices=["pretrain", "sft", "reward", "ppo", "compare", "speedup", "benchmark", "diversity", "all"],
    help="评估阶段",
)
parser.add_argument("--checkpoint", default="checkpoints/pretrain.pt", help="主 checkpoint 路径")
parser.add_argument("--sft",  default="checkpoints/sft.pt",    help="SFT checkpoint（compare）")
parser.add_argument("--ppo",  default="checkpoints/ppo.pt",    help="PPO checkpoint（compare）")
parser.add_argument("--rm",   default="checkpoints/reward.pt", help="RM  checkpoint（compare）")
parser.add_argument("--device", default="cpu", help="cpu / cuda")
args = parser.parse_args()


def _print_kv(d, keys=None):
    """格式化打印字典里的指定 key（避免一坨 print）。"""
    if keys is None:
        keys = list(d.keys())
    for k in keys:
        if k in d:
            v = d[k]
            if isinstance(v, float):
                print(f"[eval]   {k:>20s} = {v:.4f}")
            else:
                print(f"[eval]   {k:>20s} = {v}")


if args.stage == "pretrain":
    r = eval_perplexity(args.checkpoint, device=args.device)
    print(f"[eval] stage=pretrain ppl={r['perplexity']:.2f}  loss={r['loss']:.4f}  n_tokens={r['n_tokens']}")

elif args.stage == "sft":
    eval_sft(args.checkpoint, device=args.device)

elif args.stage == "reward":
    r = eval_pairwise_accuracy(args.checkpoint, n_samples=200, device=args.device)
    print(f"[eval] stage=reward  pair_acc={r['value']:.4f}  margin={r['margin']:+.3f}")

elif args.stage == "ppo":
    r = eval_ppo(args.checkpoint, n_samples=50, device=args.device)
    _print_kv(r)

elif args.stage == "compare":
    r = eval_sft_vs_ppo(
        sft_ckpt=args.sft, ppo_ckpt=args.ppo, rm_ckpt=args.rm,
        n_samples=30, device=args.device,
    )
    _print_kv(r)

elif args.stage == "speedup":
    r = eval_speedup(args.checkpoint, n_runs=5, max_new_tokens=64, device=args.device)
    _print_kv(r)

elif args.stage == "benchmark":
    r = eval_benchmark(args.checkpoint, n_questions=100, device=args.device)
    print(f"[eval] stage=benchmark  acc={r['value']:.4f}  ({r['n_correct']}/{r['n_questions']})")

elif args.stage == "diversity":
    # 兼容旧 stage 名：跑 distinct + self_bleu + token_entropy
    r1 = eval_distinct_n(args.checkpoint, n_gram=1, n_samples=50, device=args.device)
    r2 = eval_distinct_n(args.checkpoint, n_gram=2, n_samples=50, device=device)
    rsb = eval_self_bleu(args.checkpoint, n_samples=50, device=args.device)
    rte = eval_token_entropy(args.checkpoint, n_samples=20, device=args.device)
    print(f"[eval] distinct_1={r1['value']:.4f}  distinct_2={r2['value']:.4f}  "
          f"self_bleu={rsb['value']:.4f}  token_entropy={rte['value']:.4f}")

elif args.stage == "all":
    # 跑完整 9 个评估函数
    print("\n" + "=" * 70)
    print("[eval] stage=pretrain → eval_perplexity")
    print("=" * 70)
    if os.path.exists(args.checkpoint):
        r = eval_perplexity(args.checkpoint, device=args.device)
        print(f"[eval]   perplexity = {r['perplexity']:.2f}  loss = {r['loss']:.4f}  n_tokens = {r['n_tokens']}")
    else:
        print(f"[eval]   skip: {args.checkpoint} 不存在")

    sft_ckpt = args.checkpoint.replace("pretrain.pt", "sft.pt") if "pretrain" in args.checkpoint else "checkpoints/sft.pt"
    if os.path.exists(sft_ckpt):
        print("\n" + "=" * 70)
        print(f"[eval] stage=sft → 6 个 SFT 评估函数 (ckpt={sft_ckpt})")
        print("=" * 70)
        eval_sft(sft_ckpt, device=args.device)
    else:
        print(f"[eval]   skip: {sft_ckpt} 不存在")

    rm_ckpt = "checkpoints/reward.pt"
    if os.path.exists(rm_ckpt):
        print("\n" + "=" * 70)
        print(f"[eval] stage=reward → eval_pairwise_accuracy (ckpt={rm_ckpt})")
        print("=" * 70)
        r = eval_pairwise_accuracy(rm_ckpt, n_samples=200, device=args.device)
        _print_kv(r, ["value", "val_loss", "margin", "mean_r_chosen", "mean_r_rejected"])
    else:
        print(f"[eval]   skip: {rm_ckpt} 不存在")

    ppo_ckpt = "checkpoints/ppo.pt"
    if os.path.exists(ppo_ckpt):
        print("\n" + "=" * 70)
        print(f"[eval] stage=ppo → eval_ppo (ckpt={ppo_ckpt})")
        print("=" * 70)
        r = eval_ppo(ppo_ckpt, n_samples=50, device=args.device)
        _print_kv(r)

        # compare
        if os.path.exists(sft_ckpt) and os.path.exists(rm_ckpt):
            print("\n" + "=" * 70)
            print("[eval] stage=compare → eval_sft_vs_ppo")
            print("=" * 70)
            r = eval_sft_vs_ppo(sft_ckpt=sft_ckpt, ppo_ckpt=ppo_ckpt, rm_ckpt=rm_ckpt, n_samples=30, device=args.device)
            _print_kv(r)

    else:
        print(f"[eval]   skip: {ppo_ckpt} 不存在")

    # speedup
    speed_ckpt = sft_ckpt if os.path.exists(sft_ckpt) else args.checkpoint
    if os.path.exists(speed_ckpt):
        print("\n" + "=" * 70)
        print(f"[eval] stage=speedup → eval_speedup (ckpt={speed_ckpt})")
        print("=" * 70)
        r = eval_speedup(speed_ckpt, n_runs=5, max_new_tokens=64, device=args.device)
        _print_kv(r)
    else:
        print(f"[eval]   skip: {speed_ckpt} 不存在")

    print("\n" + "=" * 70)
    print("[eval] 9 个评估函数全部跑完")
    print("=" * 70)

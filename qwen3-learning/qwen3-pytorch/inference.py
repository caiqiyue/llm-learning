"""
Qwen3 推理：prefill / decode 两阶段 + temperature / top-k / top-p 采样。

对应笔记：qwen3-learning/Qwen3_GQA与KV_Cache_完整学习笔记.md
对照主线：gpt-learning/gpt-learning-pytorch/generate.py（GPT-2 线）

【与 GPT-2 线的关系】
    **组织方式一致，代码独立重写**（AGENTS.md 硬约束：Qwen3 线不跨目录 import）。
    接口刻意保持同名同义，好让两条线横向对比：
        generate_naive / generate_with_cache / benchmark_speed
    benchmark_speed 返回的列名与 GPT-2 线完全一致：
        naive_avg_ms / kv_avg_ms / speedup / naive_tokens / kv_tokens

【Qwen3 特有的两点】
    1. **decode 阶段不施加因果 mask**：T_q = 1、T_kv 随步数增长，
       cache 天然只含历史 token（见 model/common/attention.py 第 5 步）。
    2. **KV Cache 按 n_head_kv 分配**：GQA 下占用是同尺寸 MHA 的 n_head_kv/n_head。

【GPT-2 线笔记（gpt-learning/12.采样方式.md）指出的 4 个问题，本线的处理】
    ① 左 padding 没有 attention mask → **本线只做 batch=1 的单条生成**，
       不引入 padding，问题自然不存在（要批处理需另加 mask）。
    ② finished 样本没有真的冻结输出 → 本线在 eos 出现时**直接 break**，
       不再采样后续 token（batch=1 下等价且更干净）。
    ③ KVCache 的 device/dtype 可能不匹配 → 本线**显式传 model 的 dtype**，
       并在 append 时做 `.to(dtype)` 转换。
    ④ cache.get() 每次 clone 整段缓存 → 本线 `KVCache.get` 直接返回**切片 view**
       （见 model/common/kv_cache.py），不复制。

【采样链（与 GPT-2 线一致的执行顺序）】
    logits → temperature → top-k → top-p → softmax → multinomial
"""

import time
from typing import List, Optional, Tuple

import torch

from model.common.kv_cache import KVCache


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------
def sample_from_logits(
    logits: "torch.Tensor",
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> "torch.Tensor":
    """
    从 logits 采样一个 token。

    【执行顺序（与 GPT-2 线一致）】
        1. temperature <= 0 → 直接 greedy（argmax），**不再执行 top-k / top-p**
        2. temperature：logits / T（T<1 更尖锐，T>1 更平坦）
        3. top-k：把第 k 名之后的 logits 置 -inf（**固定候选数量**）
        4. top-p：把累计概率越界之后的 logits 置 -inf（**候选数量动态**）
        5. softmax → multinomial

    【top-p 的 mask 写法】
        `mask = cumsum - probs > top_p`
        「cumsum - probs」= 加入当前 token **之前**的累计概率。
        这样保证「让累计概率第一次超过 top_p 的那个 token 仍然被保留」。

    参数：
        logits:     (B, V)
        temperature: 温度；<= 0 走 greedy
        top_k:      保留概率最高的 k 个；None / <=0 表示不启用
        top_p:      nucleus 阈值；None 或不在 (0,1) 表示不启用
    返回：
        (B, 1) 采样出的 token id
    """
    if logits.dim() != 2:
        raise ValueError(f"logits 应为 (B, V)，实得 {tuple(logits.shape)}")

    # --- 1. greedy ---
    if temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    # --- 2. temperature ---
    logits = logits / temperature

    # --- 3. top-k ---
    if top_k is not None and top_k > 0:
        k = min(int(top_k), logits.size(-1))
        kth = logits.topk(k, dim=-1).values[:, -1:]          # (B, 1)
        logits = torch.where(
            logits < kth,
            torch.full_like(logits, float("-inf")),
            logits,
        )

    # --- 4. top-p ---
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = sorted_logits.softmax(dim=-1)
        cum = sorted_probs.cumsum(dim=-1)
        remove = (cum - sorted_probs) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        # 排序 → 原位还原
        logits = torch.empty_like(logits).scatter(-1, sorted_idx, sorted_logits)

    # --- 5. multinomial ---
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ---------------------------------------------------------------------------
# 缓存构造
# ---------------------------------------------------------------------------
def make_kv_cache(
    model: "torch.nn.Module",
    batch_size: int,
    max_len: int,
    device: Optional["torch.device"] = None,
) -> KVCache:
    """
    按模型配置构造一个 KVCache。

    参数：
        model:      Qwen3 模型（需带 .cfg）
        batch_size: batch 大小
        max_len:    最多缓存多少 token
        device:     设备；None 时从模型参数推断
    返回：
        KVCache 实例
    """
    cfg = model.cfg
    dtype = next(model.parameters()).dtype
    cache = KVCache(
        n_layer=cfg.n_layer,
        batch_size=batch_size,
        n_head_kv=cfg.n_head_kv,
        head_dim=cfg.head_dim,
        max_len=max_len,
        dtype=dtype,
    )
    if device is not None:
        cache.k_buf = cache.k_buf.to(device)
        cache.v_buf = cache.v_buf.to(device)
    return cache


# ---------------------------------------------------------------------------
# 生成：naive（无 cache）
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_naive(
    model: "torch.nn.Module",
    prompt: "torch.Tensor",
    max_new_tokens: int = 16,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_id: Optional[int] = None,
    return_logits: bool = False,
) -> "torch.Tensor":
    """
    **不用 KV Cache** 的生成：每一步都把整条序列重新前向一遍。

    【为什么慢】
        生成第 s 个 token 时要重算前 s+P 个 token 的 Q/K/V/MLP，
        而这些位置的结果在之前的步骤里已经算过 —— 全部白算。
        总计算量随生成长度**平方**增长。

    参数：
        model:          模型（必须是 eval 模式）
        prompt:         (B, P) LongTensor
        max_new_tokens: 最多生成多少 token
        temperature:    采样温度（<= 0 = greedy）
        top_k:          top-k
        top_p:          top-p
        eos_id:         遇到这个 token 就停
        return_logits:  True 时额外返回每步的 logits（(B, V) 的 list）
    返回：
        (B, P + n) LongTensor；return_logits=True 时返回 (tensor, list)
    """
    if prompt.dim() != 2:
        raise ValueError(f"prompt 应为 (B, P)，实得 {tuple(prompt.shape)}")
    model.eval()
    device = prompt.device
    B, P = prompt.shape
    block_size = model.cfg.block_size

    # 整条序列必须放得进 block_size
    room = block_size - P
    if room <= 0:
        raise ValueError(f"prompt 长度 {P} 已达 block_size={block_size}，无法生成")
    n_max = min(max_new_tokens, room)

    seq = prompt
    out_logits: List["torch.Tensor"] = []
    for _ in range(n_max):
        logits, _, _, _ = model(seq)                     # (B, L, V)
        next_logits = logits[:, -1, :]                   # (B, V)
        if return_logits:
            out_logits.append(next_logits)
        nxt = sample_from_logits(next_logits, temperature, top_k, top_p)
        seq = torch.cat([seq, nxt], dim=1)
        if eos_id is not None and int(nxt.item()) == eos_id:
            break
    if return_logits:
        return seq, out_logits
    return seq


# ---------------------------------------------------------------------------
# 生成：with cache（prefill + decode）
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_with_cache(
    model: "torch.nn.Module",
    prompt: "torch.Tensor",
    max_new_tokens: int = 16,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_id: Optional[int] = None,
    return_logits: bool = False,
) -> "torch.Tensor":
    """
    **用 KV Cache** 的生成：prefill 一次 + decode 每步 1 个 token。

    【prefill】
        整条 prompt 一次前向，K/V 写进 cache（每层各写 P 个 token）。
        返回的 logits 里只取最后一个位置来采样第一个新 token。

    【decode】
        每步只喂 **1 个** 新 token：
            T_q = 1，T_kv = 已缓存的历史长度（随步数增长）
        **不再施加因果 mask** —— cache 天然只含历史与当前 token，不含未来。

    参数：
        model:          模型（必须是 eval 模式）
        prompt:         (B, P) LongTensor
        max_new_tokens: 最多生成多少 token
        temperature:    采样温度（<= 0 = greedy）
        top_k:          top-k
        top_p:          top-p
        eos_id:         遇到这个 token 就停
        return_logits:  True 时额外返回每步的 logits（(B, V) 的 list）
    返回：
        (B, P + n) LongTensor；return_logits=True 时返回 (tensor, list)
    """
    if prompt.dim() != 2:
        raise ValueError(f"prompt 应为 (B, P)，实得 {tuple(prompt.shape)}")
    model.eval()
    device = prompt.device
    B, P = prompt.shape
    block_size = model.cfg.block_size

    room = block_size - P
    if room <= 0:
        raise ValueError(f"prompt 长度 {P} 已达 block_size={block_size}，无法生成")
    n_max = min(max_new_tokens, room)

    cache = make_kv_cache(model, batch_size=B, max_len=P + n_max, device=device)

    # --- prefill：整条 prompt 一次前向 ---
    pos = torch.arange(P, dtype=torch.long, device=device).unsqueeze(0).expand(B, P)
    logits, _, cache, _ = model(prompt, kv_cache=cache, position_ids=pos)
    next_logits = logits[:, -1, :]                      # (B, V)
    out_logits: List["torch.Tensor"] = []
    if return_logits:
        out_logits.append(next_logits)
    nxt = sample_from_logits(next_logits, temperature, top_k, top_p)
    generated = [nxt]
    if eos_id is not None and int(nxt.item()) == eos_id:
        # ★ prefill 后采出来的**第一个** token 就是 eos ⇒ 立刻停
        #   （漏掉这一检查会让 eos 多生成出一串无意义 token）
        seq = torch.cat([prompt] + generated, dim=1)
        if return_logits:
            return seq, out_logits
        return seq

    # --- decode：每步 1 个 token ---
    for step in range(n_max - 1):
        pos = torch.full((B, 1), P + step, dtype=torch.long, device=device)
        logits, _, cache, _ = model(nxt, kv_cache=cache, position_ids=pos)
        next_logits = logits[:, -1, :]                  # (B, 1, V) → (B, V)
        if return_logits:
            out_logits.append(next_logits)
        nxt = sample_from_logits(next_logits, temperature, top_k, top_p)
        generated.append(nxt)
        if eos_id is not None and int(nxt.item()) == eos_id:
            break

    seq = torch.cat([prompt] + generated, dim=1)
    if return_logits:
        return seq, out_logits
    return seq


# ---------------------------------------------------------------------------
# 加速比 benchmark
# ---------------------------------------------------------------------------
def benchmark_speed(
    model: "torch.nn.Module",
    prompt: "torch.Tensor",
    n_runs: int = 3,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    eos_id: Optional[int] = None,
) -> dict:
    """
    对比 `generate_naive` 与 `generate_with_cache` 的耗时。

    返回 dict（**列名与 GPT-2 线完全一致**，便于横向对比）：
        {
            "naive_avg_ms":  float,  # naive 平均耗时（毫秒）
            "kv_avg_ms":     float,  # kv 平均耗时（毫秒）
            "speedup":       float,  # 加速比（naive / kv）
            "naive_tokens":  int,    # naive 实际生成的 token 数
            "kv_tokens":     int,    # kv 实际生成的 token 数
        }

    参数：
        model:          模型
        prompt:         (1, P) LongTensor
        n_runs:         每个方法跑几次取平均
        max_new_tokens: 最大生成 token 数
        temperature:    采样温度（默认 greedy，保证两条路径可比）
        eos_id:         早停 token id
    """
    model.eval()

    # warmup：把第一次的 lazy init 等开销跑掉
    _ = generate_naive(model, prompt.clone(), max_new_tokens=4, temperature=temperature,
                       eos_id=eos_id)
    _ = generate_with_cache(model, prompt.clone(), max_new_tokens=4, temperature=temperature,
                             eos_id=eos_id)

    naive_times: List[float] = []
    naive_tokens = 0
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = generate_naive(model, prompt.clone(), max_new_tokens=max_new_tokens,
                             temperature=temperature, eos_id=eos_id)
        naive_times.append(time.perf_counter() - t0)
        naive_tokens = out.size(1) - prompt.size(1)

    kv_times: List[float] = []
    kv_tokens = 0
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = generate_with_cache(model, prompt.clone(), max_new_tokens=max_new_tokens,
                                  temperature=temperature, eos_id=eos_id)
        kv_times.append(time.perf_counter() - t0)
        kv_tokens = out.size(1) - prompt.size(1)

    naive_avg = sum(naive_times) / n_runs
    kv_avg = sum(kv_times) / n_runs
    speedup = naive_avg / kv_avg if kv_avg > 0 else float("inf")
    return {
        "naive_avg_ms": naive_avg * 1000.0,
        "kv_avg_ms": kv_avg * 1000.0,
        "speedup": speedup,
        "naive_tokens": naive_tokens,
        "kv_tokens": kv_tokens,
    }


# ---------------------------------------------------------------------------
# KV Cache 占用统计（GQA vs MHA）
# ---------------------------------------------------------------------------
def kv_cache_bytes(
    n_layer: int,
    batch_size: int,
    n_head_kv: int,
    seq_len: int,
    head_dim: int,
    dtype_size: int = 4,
) -> int:
    """
    KV Cache 占用字节数。

    公式：`n_layer × B × n_head_kv × T × head_dim × 2(K和V) × dtype_size`

    参数：
        n_layer:     层数
        batch_size:  batch 大小
        n_head_kv:   KV head 数
        seq_len:     序列长度 T
        head_dim:    head 特征维度
        dtype_size:  每个元素的字节数（fp32=4, fp16=2）
    返回：
        字节数（int）
    """
    return n_layer * batch_size * n_head_kv * seq_len * head_dim * 2 * dtype_size


def kv_cache_report(
    model: "torch.nn.Module",
    prompt_len: int,
    max_new_tokens: int,
    batch_size: int = 1,
    dtype_size: Optional[int] = None,
) -> dict:
    """
    打印/汇报 GQA 下的 KV Cache 实际占用，并与同尺寸 MHA 对比。

    参数：
        model:          模型
        prompt_len:     prompt 长度
        max_new_tokens: 计划生成的 token 数
        batch_size:     batch 大小
        dtype_size:     元素字节数；None 时从模型参数推断
    返回：
        {
            "seq_len": int,          # 缓存的总长度（prompt + 生成）
            "gqa_bytes": int,        # GQA 实际占用
            "mha_bytes": int,        # 同尺寸 MHA（n_head_kv = n_head）会占多少
            "ratio": float,          # gqa / mha
            "gqa_mib": float,
            "mha_mib": float,
        }
    """
    cfg = model.cfg
    if dtype_size is None:
        dtype_size = next(model.parameters()).element_size()
    T = prompt_len + max_new_tokens
    gqa = kv_cache_bytes(cfg.n_layer, batch_size, cfg.n_head_kv, T, cfg.head_dim, dtype_size)
    mha = kv_cache_bytes(cfg.n_layer, batch_size, cfg.n_head, T, cfg.head_dim, dtype_size)
    return {
        "seq_len": T,
        "gqa_bytes": gqa,
        "mha_bytes": mha,
        "ratio": gqa / mha if mha else float("inf"),
        "gqa_mib": gqa / (1024 * 1024),
        "mha_mib": mha / (1024 * 1024),
    }

"""
高层生成：naive / kv 两种生成函数 + benchmark_speed + 字符串接口。

文件职责：
    - `generate_naive`: 不带 cache，每步重新 forward 整个 prefix（**教学用**）
    - `generate_kv`:    带 KV cache，每步只 forward 新 token（**生产用**）
    - `batched_generate`: 多 prompt 并行采样（inference.py 提供；本文件 re-export）
    - `generate_text`:  字符串级别的高层接口
    - `benchmark_speed`: 对比 naive vs kv 耗时

详细 spec 见 design §5、gpt2_training_explained.md Part 7、implementation-plan.md P7.1.2。

代码风格：详细中文 docstring + 完整 type hint；无 `if __name__`；延迟 import torch。
"""

import time
from typing import List, Optional

import torch
import torch.nn as nn

# 从同包 inference 引入
from inference import KVCache, GPT2WithKVCache, batched_generate as _batched_generate


# ===========================================================================
# 1. generate_naive（教学用，不带 KV cache）
# ===========================================================================
@torch.no_grad()
def generate_naive(
    model: nn.Module,
    idx: "torch.Tensor",
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_id: Optional[int] = None,
) -> "torch.Tensor":
    """
    教学版朴素生成：每次 forward 整段重新算。

    【复杂度】O(L²)：每生成第 t 个 token，要重新 forward 长度为 t 的 prefix。

    【每步形状变化】
    ┌────────────────────────────────────────────────────────┐
    │ 步骤                │ 输入形状        │ 输出形状         │
    ├────────────────────────────────────────────────────────┤
    │ idx                 │ -              │ (B, T)          │
    │ idx_cond (截断)      │ (B, T)         │ (B, min(T,128)) │
    │ forward(idx_cond)   │ (B, T_cond)    │ logits: (B,T,V) │
    │ logits[:, -1, :]    │ (B, T, V)      │ (B, V)          │
    │ sampling            │ (B, V)         │ next_id: (B, 1) │
    │ cat                 │ (B, T) + (B,1) │ (B, T+1)        │
    └────────────────────────────────────────────────────────┘

    用法：
        - 单元测试：与 generate_kv 对比，验证逻辑一致性
        - 教学演示：看清楚每步都做了什么
        - benchmark 对比的"基线"

    参数：
        model:         GPT2 实例（必须 `eval()`）
        idx:           (B, T) 起始 token id（prompt）
        max_new_tokens: 最大生成 token 数
        temperature:   采样温度（T→0 趋近 greedy）
        top_k:         top-k 截断
        top_p:         nucleus 截断
        eos_id:        早停 token id
    返回：
        (B, T + 实际生成长度) LongTensor
    """
    model.eval()
    for _ in range(max_new_tokens):
        # === 0) 截断到 block_size ===
        #    如果 idx 长度超过训练时的 max_len（128），只取最后 128 个
        #    形状: (B, min(T, block_size))
        idx_cond = (  # 1,22
            idx if idx.size(1) <= model.block_size else idx[:, -model.block_size:]
        )

        # === 1) forward（不带 cache，跑整段）===
        #    out 是 3 元组（value_head=None）或 4 元组（已挂 value_head）
        out = model(idx_cond)
        if len(out) == 4:
            logits, _, _, _ = out
        else:
            logits, _, _ = out
        # === 2) 取最后位置的 logits（用于预测下一个 token）===
        #    logits[:, -1, :]: (B, V)
        logits = logits[:, -1, :]  #1,45 只选择最后一个词的logit 预测要生成的词

        # === 3) 采样：temperature + top_k + top_p + multinomial ===
        #    next_id: (B, 1)
        next_id = _sample_from_logits(logits, temperature, top_k, top_p) #next_id : 5 

        # === 4) 拼接到末尾 ===
        #    idx: (B, T) → cat → (B, T+1)
        idx = torch.cat([idx, next_id], dim=1)
        """
        tensor([[17, 33, 22, 24,  5, 40, 35, 34, 33,  5, 20,  5, 39, 28, 32, 24,  9,  5,
         39, 27, 24,  5]])
        idx
        tensor([[17, 33, 22, 24,  5, 40, 35, 34, 33,  5, 20,  5, 39, 28, 32, 24,  9,  5,
                39, 27, 24,  5,  5]])
        """
        # === 5) 早停（教学版简化：所有 batch 都 eos 才停）===
        if eos_id is not None and (next_id == eos_id).all():
            break

    return idx


# ===========================================================================
# 2. generate_kv（生产用，带 KV cache）
# ===========================================================================
@torch.no_grad()
def generate_kv(
    model_with_kv: GPT2WithKVCache,
    idx: "torch.Tensor",
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_id: Optional[int] = None,
) -> "torch.Tensor":
    """
    生产版生成：带 KV cache，每步只 forward 一个新 token。

    【复杂度】O(L) per step：每步只算新 token 的 Q、K、V，K/V 历史部分从 cache 取。

    【单步形状变化】（关键！与 generate_naive 的核心区别）
    ┌────────────────────────────────────────────────────────────┐
    │ 步骤                │ 输入形状        │ 输出形状             │
    ├────────────────────────────────────────────────────────────┤
    │ idx (prompt)        │ -              │ (B, T_prompt)        │
    │ PREFILL             │ (B, T_prompt)  │ logits: (B,T,V)      │
    │   position_ids: (B, T_prompt)  cache: 全部历史 K/V        │
    │ logits[:, -1, :]    │ (B, T, V)      │ (B, V)               │
    │ DECODE 步:          │ (B, 1)         │ logits: (B, 1, V)    │
    │   cur_token: (B, 1) cache 自动追加 K、V                     │
    │ cat                 │ (B, T) + (B,1) │ (B, T+1)             │
    └────────────────────────────────────────────────────────────┘

    【与 generate_naive 的关键区别】
      - naive: 每步 forward 整段 prefix（O(L²) 总复杂度）
      - kv:    每步只 forward 1 个新 token，K/V 从 cache 读（O(L) 总复杂度）
      - 加速比 ~L/3（理论）；本项目实测 PT 1.5-1.8x

    流程：
        1) Prefill：整段 prompt 一次 forward（构造 K、V cache）
        2) Sample 第一个 token
        3) Decode loop：每步 forward 一个 token
        4) 早停：所有样本都 eos 时退出

    参数：
        model_with_kv: GPT2WithKVCache 实例（**eval**）
        idx:           (B, T) 起始 token id
        max_new_tokens: 最大生成 token 数
        temperature:   采样温度
        top_k:         top-k 截断
        top_p:         nucleus 截断
        eos_id:        早停 token id
    返回：
        (B, T + 实际生成长度) LongTensor
    """
    model_with_kv.eval()
    gpt2 = model_with_kv.gpt2
    B, T_prompt = idx.size() #1,22

    # === 1) 构造 batched KV cache ===
    #    cache.keys: (n_layer, B, n_head, max_seq_len, head_dim)  ← 预分配全 0
    head_dim = gpt2.n_embd // gpt2.n_head #4个头，每个头 32 维，
    cache = KVCache( #预先为每层attention 的 k 和 v 分配一块大内存，默认全为 0
        n_layer=gpt2.n_layer, #4层
        n_head=gpt2.n_head, #4
        head_dim=head_dim, #32
        max_seq_len=gpt2.block_size, # 可以理解为最长 字数限制 128
        batch_size=B, # 1
    )

    # === 2) PREFILL：整段 prompt 一次塞进模型 ===
    #    position_ids: (B, T_prompt)  = [0, 1, 2, ..., T_prompt-1]
    position_ids = torch.arange(T_prompt, device=idx.device).unsqueeze(0).expand(B, -1) #1,22
    #    logits: (B, T_prompt, V)
    #    cache 内部：每层 seq_len = T_prompt（K、V 已 append 完毕） prefill 阶段
    logits, cache = model_with_kv(idx, cache, position_ids) # logits 1,22,45  

    # === 3) Sample 第一个 token ===
    #    取 prompt 最后一个位置的 logits
    next_logits = logits[:, -1, :]   # (B, V)
    next_token = _sample_from_logits(next_logits, temperature, top_k, top_p)  # (B, 1)

    # === 4) 跟踪每个样本是否已 finished ===
    if eos_id is None:
        finished = torch.zeros(B, dtype=torch.bool, device=idx.device)
    else:
        finished = (next_token.squeeze(-1) == eos_id)   # (B,) bool

    generated = [next_token]   # list of (B, 1)  prefill 阶段完成后，生成的第一个 token

    # === 5) DECODE LOOP：每步只 forward 1 个新 token ===
    for step in range(max_new_tokens - 1): #最多运行生成64个token
        if finished.all():
            break

        # === 已 finished 的样本强制为 <unk>，避免污染采样 ===
        #    cur_token: (B, 1)
        cur_token = next_token.clone()
        cur_token[finished] = 0  # 用 <unk> id 0 占位

        # === 绝对位置 = T_prompt + step ===
        #    pos: (B, 1)
        """
        torch.full() 的作用是：

创建一个指定形状的张量，并把其中所有元素都填成同一个值。

基本形式：

torch.full(
    size,
    fill_value,
    dtype=None,
    device=None,
)
        """
        pos = torch.full( #1,1  值是 22 ，因为prompt长度是22，即0-21，所以新生成的词 就是从 22开始,23,24。。。。。。
            (B, 1), fill_value=T_prompt + step, dtype=torch.long, device=idx.device
        )
        # === 单步 forward（关键：cur_token 只有 (B, 1)）===
        #    logits: (B, 1, V)   cache 内部 seq_len += 1     decode阶段 一次 就送进去一个 token，pos 也就是 一个位置
        logits, cache = model_with_kv(cur_token, cache, pos)
        next_logits = logits[:, -1, :]   # (B, V)
        next_token = _sample_from_logits(next_logits, temperature, top_k, top_p)  # (B, 1)

        # === 更新 finished mask ===
        if eos_id is not None:
            finished = finished | (next_token.squeeze(-1) == eos_id)

        generated.append(next_token)

    # === 6) 拼成完整序列 ===
    #    gen: (B, actual_gen_len)
    gen = torch.cat(generated, dim=1)
    return torch.cat([idx, gen], dim=1)


# ===========================================================================
# 3. batched_generate（re-export）
# ===========================================================================
def batched_generate(
    model_with_kv: GPT2WithKVCache,
    prompts: List["torch.Tensor"],
    max_new_tokens: int,
    eos_id: Optional[int] = None,
    pad_id: int = 1,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> "torch.Tensor":
    """
    多 prompt 并行采样（inference.batched_generate 的 re-export，方便 generate.py 一站式调用）。

    完整 docstring 见 `inference.batched_generate`。
    """
    return _batched_generate(
        model_with_kv,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        eos_id=eos_id,
        pad_id=pad_id,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )


# ===========================================================================
# 4. generate_text（字符串级高层接口）
# ===========================================================================
@torch.no_grad()
def generate_text(
    model_or_wrapped: nn.Module,
    tokenizer,
    prompt_str: str,
    max_new_tokens: int = 64,
    mode: str = "kv",
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_id: Optional[int] = None,
    device: str = "cpu",
) -> str:
    """
    字符串级的高层接口：prompt 字符串 → 生成字符串。

    参数：
        model_or_wrapped: GPT2（mode='naive'）或 GPT2WithKVCache（mode='kv'）
        tokenizer:        CharTokenizer 实例
        prompt_str:      起始字符串
        max_new_tokens:  最大生成 token 数
        mode:            'naive' 或 'kv'
        temperature:     采样温度
        top_k:           top-k 截断
        top_p:           nucleus 截断
        eos_id:          早停 token id
        device:          'cpu' / 'cuda'
    返回：
        生成的完整字符串（prompt + 生成部分）
    """
    if eos_id is None:
        eos_id = getattr(tokenizer, "EOS_ID", 3) #结束标志位

    model_or_wrapped.eval()
    # encode
    ids = tokenizer.encode(prompt_str) #token 列表
    if not ids:
        return prompt_str
    idx = torch.tensor([ids], dtype=torch.long, device=device) #1,22  从列表转为矩阵

    # generate
    if mode == "naive":
        out = generate_naive(
            model_or_wrapped, idx, max_new_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p, eos_id=eos_id,
        )
    elif mode == "kv":
        # 若传的是原生 GPT2，自动包成 GPT2WithKVCache
        # 这样 API 与 mode='naive' 对称——用户不用关心内部包装
        # （与 benchmark_speed 里的 wrapped = GPT2WithKVCache(model) 一致）
        if not isinstance(model_or_wrapped, GPT2WithKVCache):
            model_or_wrapped = GPT2WithKVCache(model_or_wrapped)
        out = generate_kv(
            model_or_wrapped, idx, max_new_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p, eos_id=eos_id,
        )
    else:
        raise ValueError(f"mode 必须是 'naive' 或 'kv'，收到 {mode!r}")

    # decode
    return tokenizer.decode(out[0].tolist())


# ===========================================================================
# 5. benchmark_speed：naive vs kv 速度对比
# ===========================================================================
def benchmark_speed(
    model: nn.Module,
    prompt: "torch.Tensor",
    n_runs: int = 5,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    eos_id: Optional[int] = None,
) -> dict:
    """
    对比 `generate_naive` 与 `generate_kv` 的耗时。

    返回 dict：
        {
            "naive_avg_ms":  float,  # naive 平均耗时（毫秒）
            "kv_avg_ms":     float,  # kv 平均耗时（毫秒）
            "speedup":       float,  # 加速比（naive / kv）
            "naive_tokens":  int,    # naive 实际生成的 token 数
            "kv_tokens":     int,    # kv 实际生成的 token 数
        }

    参数：
        model:          GPT2 实例
        prompt:         (1, T) LongTensor 的 prompt
        n_runs:         每个方法跑几次取平均
        max_new_tokens: 最大生成 token 数
        temperature:    采样温度
        eos_id:         早停 token id
    """
    # 1) 准备 wrapped 包装（KV 版）
    model.eval()
    wrapped = GPT2WithKVCache(model)

    # 2) warmup：把第一次的 lazy init / CUDA 编译都跑掉
    _ = generate_naive(model, prompt.clone(), max_new_tokens=8, temperature=temperature, eos_id=eos_id)
    _ = generate_kv(wrapped, prompt.clone(), max_new_tokens=8, temperature=temperature, eos_id=eos_id)

    # 3) naive 测时
    naive_times = []
    naive_tokens = 0
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = generate_naive(model, prompt.clone(), max_new_tokens=max_new_tokens,
                             temperature=temperature, eos_id=eos_id)
        naive_times.append(time.perf_counter() - t0)
        naive_tokens = out.size(1) - prompt.size(1)

    # 4) kv 测时
    kv_times = []
    kv_tokens = 0
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = generate_kv(wrapped, prompt.clone(), max_new_tokens=max_new_tokens,
                          temperature=temperature, eos_id=eos_id)
        kv_times.append(time.perf_counter() - t0)
        kv_tokens = out.size(1) - prompt.size(1)

    # 5) 统计
    naive_avg = sum(naive_times) / n_runs
    kv_avg = sum(kv_times) / n_runs
    speedup = naive_avg / kv_avg if kv_avg > 0 else float("inf")

    return {
        "naive_avg_ms": naive_avg * 1000,
        "kv_avg_ms": kv_avg * 1000,
        "speedup": speedup,
        "naive_tokens": naive_tokens,
        "kv_tokens": kv_tokens,
    }


# ===========================================================================
# 6. 内部工具：从 logits 采样
# ===========================================================================
def _sample_from_logits(
    logits: "torch.Tensor",
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
) -> "torch.Tensor":
    """
    从 (B, V) 的 logits 采样 (B, 1) 的 token id。

    实现：temperature 缩放 → top-k 截断 → top-p 截断 → multinomial 采样。
    """
    if temperature <= 0.0:
        # greedy
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature #1,45

    # top-k
    if top_k is not None and top_k > 0:
        kth = logits.topk(top_k, dim=-1).values[:, -1:] #kth 排名 第 20 位 词的 logit
        logits = torch.where(   #只要比 kth 小 都淘汰
            logits < kth, torch.full_like(logits, float("-inf")), logits
        )

    # top-p
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        sorted_probs = sorted_logits.softmax(dim=-1)
        cumsum = sorted_probs.cumsum(dim=-1)
        # 把"累计超过 p 的"mask 掉（保留刚好越过 p 的那个）
        mask = cumsum - sorted_probs > top_p
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

    probs = logits.softmax(dim=-1)#1,45
    """
    
    tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2170, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0283, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0804, 0.0000, 0.0202, 0.0512, 0.1393, 0.0179, 0.0161,
         0.0498, 0.0366, 0.0000, 0.0000, 0.0332, 0.0178, 0.0435, 0.0463, 0.0000,
         0.0000, 0.0460, 0.0270, 0.0736, 0.0202, 0.0000, 0.0159, 0.0000, 0.0198]])
    """
    return torch.multinomial(probs, num_samples=1)

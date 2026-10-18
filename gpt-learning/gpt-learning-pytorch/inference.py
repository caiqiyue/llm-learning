"""
推理底层：KV Cache + GPT2WithKVCache + batched_generate。

文件职责：
    把 KV-cache 加速、增量 forward、多 prompt 并行采样这些"基础设施"集中到一个文件。
    高层 API（带 sampling kwargs 的 generate）放在 generate.py。

详细 spec 见 design §5（推理与 KV Cache）与 gpt2_training_explained.md Part 7（KV Cache 详解）。

代码风格：详细中文 docstring + 完整 type hint；无 `if __name__`；延迟 import torch。
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# ===========================================================================
# 1. KVCache：预分配张量版的 K/V 缓存
# ===========================================================================
class KVCache:
    """
    每层 K、V 张量的预分配缓存。

    设计要点：
      - **预分配** `max_seq_len` 长度的张量，避免 dynamic grow 的开销
      - 每层一个 K、一个 V
      - 用 `seq_len` 字段记录"已经写入的 token 数"
      - `append(layer_idx, k_new, v_new)`：把新 K、V 写到下一个位置
      - `get(layer_idx)`：返回当前累计的 K、V（用切片而不是 view，避免被外部 in-place 改写底层）

    形状约定（与 CausalSelfAttention 对齐）：
        K: (B, n_head, T_cached, head_dim)
        V: (B, n_head, T_cached, head_dim)
    """

    def __init__(
        self,
        n_layer: int,
        n_head: int,
        head_dim: int,
        max_seq_len: int,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        构造 KVCache（预分配张量）。

        【预分配张量形状】
          keys:   (n_layer, B, n_head, max_seq_len, head_dim)
          values: (n_layer, B, n_head, max_seq_len, head_dim)
        每一层独立一个 K 和一个 V；用 zeros 不用 empty（避免未写入位置 NaN）

        【seq_len 维护】
          _seq_lens: List[int] 长度 n_layer
          关键：每层独立 seq_len 而非全局
          原因：同一 forward 内 n_layer 个 attention 层都会调一次 append，
               如果用全局 seq_len 会算错（"跳了"）
        """
        # === 1) 预分配 keys 张量 ===
        #    形状: (n_layer, B, n_head, max_seq_len, head_dim)
        #    内存量: n_layer * B * n_head * max_seq_len * head_dim * 4 bytes
        #    项目典型值: 4 * 1 * 4 * 128 * 32 * 4 = 256KB
        self.keys = torch.zeros(
            n_layer, batch_size, n_head, max_seq_len, head_dim, dtype=dtype
        )
        # === 2) values 同形状（独立张量） ===
        self.values = torch.zeros_like(self.keys)

        # 元信息
        self.n_layer = n_layer
        self.batch_size = batch_size
        self.n_head = n_head
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        # === 3) 每层独立 seq_len（关键！）===
        #    _seq_lens[i] 表示第 i 层当前已写入的 token 数
        self._seq_lens = [0] * n_layer

    def append(
        self,
        layer_idx: int,
        k_new: "torch.Tensor",
        v_new: "torch.Tensor",
    ) -> None:
        """
        把新算出的 K、V 追加到指定层的缓存末尾。

        【形状变化】
        ┌────────────────────────────────────────────────────┐
        │ k_new:        (B, n_head, T_new, head_dim)        │
        │ 写入位置:     keys[layer_idx, :, :, cur:cur+T_new, :] │
        │ 写入后:       self.keys[layer_idx]:               │
        │                (B, n_head, cur+T_new, head_dim)    │
        │ seq_len 更新: cur → cur + T_new                   │
        └────────────────────────────────────────────────────┘

        参数：
            layer_idx: 当前层索引（0..n_layer-1）
            k_new:     (B, n_head, T_new, head_dim)  这次新算的 K
            v_new:     (B, n_head, T_new, head_dim)  这次新算的 V
        """
        T_new = k_new.size(2)
        cur = self._seq_lens[layer_idx]
        # === 防御：检查是否超过 max_seq_len ===
        if cur + T_new > self.max_seq_len:
            raise ValueError(
                f"KVCache 容量超限：layer={layer_idx}, seq_len={cur}, T_new={T_new}, "
                f"max_seq_len={self.max_seq_len}"
            )
        # === 原地写入 ===
        #    keys[layer_idx, :, :, cur:cur+T_new, :]: 预分配张量的某个切片
        #    赋值 k_new 后，那段切片被覆盖
        #    注意：PyTorch 支持 slice assignment（TF 不支持，所以 TF 端要用 .assign()）
        self.keys[layer_idx, :, :, cur : cur + T_new, :] = k_new
        self.values[layer_idx, :, :, cur : cur + T_new, :] = v_new
        # === 更新该层计数 ===
        self._seq_lens[layer_idx] = cur + T_new

    def get(self, layer_idx: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        返回该层当前累计的 K、V。

        【关键】
        - 用 clone 而非 view：防御外部 in-place 修改污染 cache
        - 只返回到当前 seq_len 为止的有效部分（不是整个 max_seq_len）

        返回：
            (K_cached, V_cached)：
                K_cached: (B, n_head, seq_len, head_dim)
                V_cached: (B, n_head, seq_len, head_dim)
        """
        end = self._seq_lens[layer_idx]
        # === 切片到 seq_len 位置 + clone ===
        #    self.keys[layer_idx, :, :, :end, :]: (B, n_head, end, head_dim)
        #    .clone(): 深拷贝（避免外部修改底层）
        return (
            self.keys[layer_idx, :, :, :end, :].clone(),
            self.values[layer_idx, :, :, :end, :].clone(),
        )

    def reset(self) -> None:
        """
        清空缓存（把所有层的 `seq_len` 设为 0）。
        注意：**不**清零底层张量——下次写入会直接覆盖旧数据，省一次 memset。
        """
        for i in range(self.n_layer):
            self._seq_lens[i] = 0

    @property
    def seq_len(self) -> int:
        """
        兼容 spec 中的 `cache.seq_len` 属性访问：
        返回**所有层**的 seq_len（应一致，因为同一序列的 K、V 同步增长）。
        """
        # 防御：所有层 seq_len 应一致；不一致时返回 max
        return max(self._seq_lens)

    @property
    def current_length(self) -> int:
        """
        当前已缓存的 token 数（property 形式，兼容 spec 中 `current_length` 的命名）。
        """
        return self.seq_len

    def __repr__(self) -> str:
        return (
            f"KVCache(n_layer={self.n_layer}, B={self.batch_size}, "
            f"n_head={self.n_head}, head_dim={self.head_dim}, "
            f"max_seq_len={self.max_seq_len}, seq_len={self.seq_len})"
        )


# ===========================================================================
# 2. GPT2WithKVCache：包装 GPT-2，支持增量 forward
# ===========================================================================
class GPT2WithKVCache(nn.Module):
    """
    把一个训练好的 GPT2 包成"增量 forward"接口：

        logits, new_cache = wrapped(idx, cache, position_ids)

    设计要点：
      - 不复制参数（self.gpt2 = gpt2，与 GPT2 共享 state_dict）
      - 调用 `gpt2.forward_with_kv`（这是 GPT2 专门为 KV cache 设计的入口）
      - 用户自己负责管理 KVCache 的生命周期（每次 generate 新建一个）

    用法：
        ```python
        gpt2 = GPT2(...).eval()
        wrapped = GPT2WithKVCache(gpt2)
        cache = KVCache(n_layer, n_head, head_dim, max_seq_len, batch_size)

        # 1) Prefill：整段 prompt 一次 forward
        pos = torch.arange(prompt_len).unsqueeze(0)
        logits, cache = wrapped(prompt_ids, cache, pos)

        # 2) Decode：一次一个 token
        for t in range(max_new_tokens):
            pos_t = torch.tensor([[prompt_len + t]])
            logits, cache = wrapped(next_token, cache, pos_t)
        ```
    """

    def __init__(self, gpt2: nn.Module) -> None:
        """
        参数：
            gpt2: 已构造好的 GPT2 实例（必须已 `eval()`）
        """
        super().__init__()
        self.gpt2 = gpt2

    def forward(
        self,
        idx: "torch.Tensor",
        cache: Optional[KVCache] = None,
        position_ids: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", Optional[KVCache]]:
        """
        增量 forward（详细 docstring 见 `GPT2.forward_with_kv`）。

        参数：
            idx:         (B, T_q) LongTensor；decode 步通常 T_q=1
            cache:       KVCache 或 None
            position_ids:(B, T_q) 绝对位置 id
        返回：
            (logits, new_cache)
        """
        return self.gpt2.forward_with_kv(idx, kv_cache=cache, position_ids=position_ids)

    def num_params(self) -> int:
        """
        透传 GPT2 的参数量。
        """
        return self.gpt2.num_params()


# ===========================================================================
# 3. batched_generate：多 prompt 并行采样（PPO rollout 基础设施）
# ===========================================================================
@torch.no_grad()
def batched_generate(
    wrapped_model: GPT2WithKVCache,
    prompts: List["torch.Tensor"],
    max_new_tokens: int,
    eos_id: Optional[int] = None,
    pad_id: int = 1,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> "torch.Tensor":
    """
    多 prompt 并行采样（带 KV cache）。

    流程：
        1) 把所有 prompt left-pad 到相同长度
        2) 构造一个 batch_size = len(prompts) 的 KVCache
        3) Prefill 一次：把整段 (left-padded) prompt 一起 forward
        4) Decode 循环：每个 batch 一次 forward 一个 token
        5) 早停：一旦某个样本生成 eos，把它的 batch 维度"冻住"，后续 step 不再更新
        6) 最后把 left-padding 裁掉

    参数：
        wrapped_model: GPT2WithKVCache 实例（**eval 模式**）
        prompts:       list of (1, T_i) LongTensor；每个 prompt 一条样本
        max_new_tokens: 最大生成 token 数
        eos_id:        早停 token id（None 则不早停）
        pad_id:        padding token id（默认 1 = <pad>）
        temperature:   采样温度
        top_k:         top-k 截断
        top_p:         nucleus 截断
    返回：
        (B, max_prompt_len + actual_gen_len) LongTensor
    """
    wrapped_model.eval()
    device = next(wrapped_model.parameters()).device
    B = len(prompts)
    if B == 0:
        return torch.zeros(0, 0, dtype=torch.long, device=device)

    # 1) left-pad prompts（统一到一个 batch 维度）
    #    【形状变化】
    #      prompts: list of (1, T_i)        每条样本长度可能不同
    #      max_prompt_len: int              = max(T_i)
    #      padded: (B, max_prompt_len)     left-pad 后所有样本长度一致
    #      为什么要 left-pad？
    #        让 max_prompt_len - 1 位置是每条样本的"真实最后一个 prompt token"
    #        这样 logits[:, -1, :] 就是统一的"每个样本的下一个 token 预测"
    max_prompt_len = max(p.size(1) for p in prompts)
    prompt_lens = [p.size(1) for p in prompts]
    # 形状 (B, max_prompt_len)
    padded = torch.full(
        (B, max_prompt_len), fill_value=pad_id, dtype=torch.long, device=device
    )
    for i, p in enumerate(prompts):
        # left pad: 把 p 放到 padded[i] 的"右侧"（offset 之前填 pad）
        offset = max_prompt_len - p.size(1)
        padded[i, offset:] = p.squeeze(0) if p.dim() == 2 else p

    # 2) 构造 batched KV cache
    #    cache.keys: (n_layer, B, n_head, max_seq_len, head_dim)
    gpt2 = wrapped_model.gpt2
    head_dim = gpt2.n_embd // gpt2.n_head
    cache = KVCache(
        n_layer=gpt2.n_layer,
        n_head=gpt2.n_head,
        head_dim=head_dim,
        max_seq_len=gpt2.block_size,
        batch_size=B,
    )

    # 3) PREFILL：整段 padded 一次塞进模型
    #    【形状变化】
    #      padded:       (B, max_prompt_len)
    #      position_ids: (B, max_prompt_len)  = [0, 1, 2, ..., max_prompt_len-1]
    #      logits:       (B, max_prompt_len, V)
    #      cache:        每层 seq_len = max_prompt_len（已 append 完毕）
    position_ids = torch.arange(max_prompt_len, device=device).unsqueeze(0).expand(B, -1)
    logits, cache = wrapped_model(padded, cache, position_ids)
    # 第一个生成的 token：取每个样本"真实最后一个 prompt 位置"的 logits
    # 因为 left-pad，每个样本的"实际最后一个 prompt token"就是 max_prompt_len - 1
    # 【形状】logits[:, -1, :]: (B, V)  ←  V = vocab_size
    next_logits = logits[:, -1, :]
    next_tokens = _sample_from_logits(next_logits, temperature, top_k, top_p)
    # next_tokens: (B, 1)

    # 跟踪每个样本是否已结束（生成 eos 的不再更新）
    if eos_id is None:
        finished = torch.zeros(B, dtype=torch.bool, device=device)   # (B,) 全 False
    else:
        finished = (next_tokens.squeeze(-1) == eos_id)               # (B,) bool

    # 收集生成结果（不含 prompt）
    generated = [next_tokens]  # list of (B, 1)

    # 4) DECODE LOOP：每步只 forward 1 个新 token
    #    【单步形状变化】
    #      cur_token: (B, 1)        ← 上一步的 next_tokens
    #      pos:       (B, 1)        ← 绝对位置 = max_prompt_len + step
    #      logits:    (B, 1, V)
    #      next_tokens: (B, 1)
    #      cache:     每层 seq_len += 1
    for step in range(max_new_tokens - 1):
        # 所有样本都 finished → 提前结束
        if finished.all():
            break

        # 把已经 finished 的样本的 token 强制为 pad
        # （仍把 token 写进 cache，但 sample 时只看 active 样本）
        # cur_token: (B, 1)
        cur_token = next_tokens.clone()
        cur_token[finished] = pad_id

        # 绝对位置：第 step 个新生成的 token，位置 = max_prompt_len + step
        pos = torch.full(
            (B, 1), fill_value=max_prompt_len + step, dtype=torch.long, device=device
        )
        # 单步 forward
        #   logits: (B, 1, V)   cache 内部 seq_len += 1
        logits, cache = wrapped_model(cur_token, cache, pos)
        # 取最后位置（即新生成那一帧）的 logits
        next_logits = logits[:, -1, :]   # (B, V)
        next_tokens = _sample_from_logits(next_logits, temperature, top_k, top_p)  # (B, 1)

        # 更新 finished：当前 batch 中任何样本生成 eos 则标记
        if eos_id is not None:
            finished = finished | (next_tokens.squeeze(-1) == eos_id)

        generated.append(next_tokens)

    # 5) 拼成 (B, actual_gen_len)
    #    generated: list of (B, 1)，长度 = 实际生成 token 数
    #    gen: (B, actual_gen_len)
    gen = torch.cat(generated, dim=1)
    # 6) 与 prompt 拼起来：(B, max_prompt_len + actual_gen_len)
    #    full: (B, T_full)  ←  T_full = T_prompt + T_resp
    full = torch.cat([padded, gen], dim=1)
    return full


# ===========================================================================
# 4. 内部工具：从 logits 采样（top-k + top-p + temperature）
# ===========================================================================
def _sample_from_logits(
    logits: "torch.Tensor",
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
) -> "torch.Tensor":
    """
    从 (B, V) 的 logits 采样 (B, 1) 的 token id。

    实现逻辑与 `GPT2.generate` 一致，但用 detach + no_grad 跑。
    """
    if temperature <= 0.0:
        # greedy
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        kth = logits.topk(top_k, dim=-1).values[:, -1:]
        logits = torch.where(
            logits < kth, torch.full_like(logits, float("-inf")), logits
        )

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        sorted_probs = sorted_logits.softmax(dim=-1)
        cumsum = sorted_probs.cumsum(dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1)

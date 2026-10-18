"""
GPT2WithValueHead：PPO 用 critic 网络的薄包装。

设计选择（依据 spec）：
    主模型 `GPT2` 在加载 SFT 权重后，会调用 `gpt2.attach_value_head()` 挂上一个
    `nn.Linear(n_embd, 1)` 的 scalar head。`GPT2.forward()` 在 value_head 不为 None 时
    会同时返回 logits 与 value。

    本文件提供 `GPT2WithValueHead` 包装类，方便 PPO 训练脚本以"一个模型 = 一个对象"
    的方式使用，**与 GPT2 共享参数**（不复制 backbone）。这避免了"双份 backbone"
    的内存开销。

用法：
    ```python
    gpt2 = GPT2(...)
    gpt2.load_state_dict(checkpoint["model"])
    gpt2.attach_value_head()                       # 挂 scalar head
    wrapped = GPT2WithValueHead(gpt2)              # 包一层（共享参数）
    logits, loss, kv, value = wrapped(idx, targets)  # 4 元组
    ```

代码风格：与 model/gpt2.py 一致——详细中文 docstring + 完整 type hint，
顶层无 `if __name__`，延迟 import torch。
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class GPT2WithValueHead(nn.Module):
    """
    把一个已挂上 value_head 的 `GPT2` 包成 PPO 训练用的 actor-critic。

    关键设计：
      - **不复制** backbone 参数；内部就是 `self.gpt2 = gpt2`
      - 直接调用 `gpt2.forward`，让它走 value_head 分支
      - PPO 训练时，把 policy、value 一起放进同一个对象，optimizer.step() 时
        AdamW 会按参数归属自然分组（tok_emb / blocks / ln_f / value_head）
    """

    def __init__(self, gpt2: nn.Module) -> None:
        """
        参数：
            gpt2: 已经 `attach_value_head()` 的 `GPT2` 实例
        """
        super().__init__()
        self.gpt2 = gpt2
        # 防御性检查：确保 value_head 已挂
        if getattr(gpt2, "value_head", None) is None:
            raise ValueError(
                "传入的 GPT2 没有 value_head；请先调用 gpt2.attach_value_head()"
            )

    def forward(
        self,
        idx: "torch.Tensor",
        targets: Optional["torch.Tensor"] = None,
        kv_cache: Optional[object] = None,
        position_ids: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"], Optional[object], "torch.Tensor"]:
        """
        与 `GPT2.forward` 同签名、同返回（4 元组）。

        返回：
            (logits, loss, kv_cache, value)
        """
        return self.gpt2(idx, targets=targets, kv_cache=kv_cache, position_ids=position_ids)

    def num_params(self) -> int:
        """
        总可训练参数量（含 value_head）。
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

"""
从零手写 LoRA（Low-Rank Adaptation）。

不引入 peft —— 项目依赖上限是 torch / numpy / nltk。

【核心公式】
    out = base(x) + (alpha / r) * dropout(x) @ A.T @ B.T

    A: (r, in_features)      Kaiming 初始化
    B: (out_features, r)     **零初始化** —— 保证初始时 LoRA 分支输出恒为 0，
                              所以注入 LoRA 前后模型输出逐元素相等

【为什么这样设计】
    - B 零初始化不是为了省事，而是为了「注入即无损」：训练开始时模型行为
      与预训练权重完全一致，LoRA 是在这个基础上慢慢长出来的
    - base.weight / base.bias 全部 requires_grad=False，只训 A / B
    - (alpha / r) 是缩放因子：alpha 通常取 2r，让有效学习率与全参微调可比

【三件套 + 两个补丁】
    三个基本动作：
        inject_lora        按模块名注入
        set_lora_enabled   一键开关 LoRA 分支（DPO 的 π_θ / π_ref 就靠它）
        merge_lora         把 ΔW 折回 base 权重（导出成普通 Linear）
    两个补丁（本仓库实测踩坑后补的，**初稿没有**）：
    1. `mark_only_lora_trainable`：注入 LoRA **只冻结被包装的那个 Linear**，
       embedding / RMSNorm / MLP / lm_head 仍然是 requires_grad=True ——
       那样「LoRA 微调」悄悄变成「大部分参数都在训」，与 LoRA 的前提（基座冻结）
       不符。本函数把其余参数全部冻结，只留 A / B 可训。
    2. `merge_lora` 合并完会把分支的 scaling 置 0：ΔW 折进 base 之后，
       前向再算一次 LoRA 分支就变成 **(base + ΔW)x + ΔWx = 被算两遍**。
       合并的语义是「把旁路收进主干」，所以合并完必须关掉分支。

【第四个动作：strip_lora（feat-024 用）】
    把 LoRALinear 换回普通 nn.Linear（保留 base 权重）。
    DPO 的标准做法是「π_ref = SFT 模型」；本仓库不加载第二份权重，
    而是把 SFT 训好的 LoRA **merge 进 base**（base 就成了 SFT 模型），
    再 strip 成普通 Linear，最后重新 inject 一个**新的** LoRA 给 DPO 训练。
    于是「关掉新 LoRA 分支 = π_ref = SFT 模型」，同一个模型对象搞定两套策略。

【与 DPO 的关系】
    DPO 需要两个模型：π_θ（当前策略）与 π_ref（参考策略）。
    本实现**不加载第二份权重**：同一个模型对象，开着 LoRA 分支是 π_θ，
    关掉（set_lora_enabled(model, False)）就是 π_ref —— 省一份显存与一份 checkpoint。
"""

import math
from typing import List, Sequence

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    包装一个 nn.Linear，加上低秩旁路。

    参数：
        base:     被包装的原始 Linear（会被冻结）
        r:        低秩矩阵的秩
        alpha:    缩放因子的分子，实际缩放 = alpha / r
        dropout:  LoRA 分支上的 dropout
    """

    def __init__(self, base: nn.Linear, r: int = 4, alpha: int = 8, dropout: float = 0.0) -> None:
        """
        初始化 LoRALinear。

        参数：
            base:    被包装的原始 Linear
            r:       秩
            alpha:   缩放分子
            dropout: LoRA 分支 dropout 概率
        """
        super().__init__()
        if r <= 0:
            raise ValueError(f"r 必须为正，实得 {r}")
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = base.in_features
        out_features = base.out_features

        # A: (r, in_features) —— Kaiming 初始化
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B: (out_features, r) —— **零初始化**（注入即无损的关键）
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 冻结 base：LoRA 微调只训 A / B
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        前向：base 分支 + LoRA 分支。

        参数：
            x: (..., in_features)
        返回：
            (..., out_features)
        """
        out = self.base(x)
        delta = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return out + self.scaling * delta

    def merged_weight(self) -> "torch.Tensor":
        """
        返回合并后的等价权重 (out_features, in_features)。

        merge 后 LoRA 分支被折进 base，可用于导出成普通 Linear。
        """
        return self.base.weight.data + self.scaling * (self.lora_B @ self.lora_A)

    def extra_repr(self) -> str:
        """打印时显示 r / alpha / scaling。"""
        return f"r={self.r}, alpha={self.alpha}, scaling={self.scaling}"


def inject_lora(
    model: nn.Module,
    target_modules: Sequence[str] = ("q_proj", "v_proj"),
    r: int = 4,
    alpha: int = 8,
    dropout: float = 0.0,
) -> List[str]:
    """
    按模块名后缀递归替换 Linear 为 LoRALinear。

    参数：
        model:          目标模型
        target_modules: 要注入的模块名后缀元组，例如 ("q_proj", "v_proj")
        r:              秩
        alpha:          缩放分子
        dropout:        LoRA 分支 dropout
    返回：
        被替换的模块全名列表（便于打印与写笔记）
    """
    replaced: List[str] = []
    # 先把 named_modules() 物化成 list —— 边遍历边替换会出问题
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            if child_name not in target_modules:
                continue
            setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            replaced.append(f"{name}.{child_name}" if name else child_name)
    return replaced


def merge_lora(model: nn.Module) -> int:
    """
    把 LoRA 分支合并回 base 权重，**并关掉分支**。

    ⚠️ 关分支是必须的：ΔW 折进 base 之后，如果分支还开着，
    输出就变成 (base + ΔW)x + ΔWx —— 被算两遍。

    参数：
        model: 已注入 LoRA 的模型
    返回：
        成功合并的模块数量
    """
    count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.base.weight.data.copy_(module.merged_weight())
            module.scaling = 0.0        # 合并即收编，分支不再参与前向
            count += 1
    return count


def set_lora_enabled(model: nn.Module, enabled: bool) -> None:
    """
    一键开关整个模型的 LoRA 分支。

    **这是 DPO 的关键技巧**：不单独加载一份冻结的 reference model，
    而是用同一个模型对象 —— 开启得到 π_θ，关闭得到 π_ref。
    省一份显存与一份 checkpoint。

    参数：
        model:   模型
        enabled: True 开启 LoRA 分支，False 关闭
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.scaling = module.alpha / module.r if enabled else 0.0


def strip_lora(model: nn.Module) -> int:
    """
    把 LoRALinear 换回普通 nn.Linear，**保留当前 base 权重**。

    用途：
        1. 导出成普通模型（推理时不想再走一遍旁路）
        2. **DPO 的关键一步**（见 train/train_dpo.py）：
           先把 SFT 训好的 LoRA 用 merge_lora 折进 base，
           再 strip 成普通 Linear，最后重新 inject 一个新的 LoRA。
           这样 base 就是「SFT 模型」，关掉新 LoRA 分支即得 π_ref = SFT 模型。

    ⚠️ strip 之前如果不 merge，base 里就**不含** LoRA 学到的 ΔW，
       等于把 LoRA 训出来的东西丢掉了 —— 顺序必须是 merge → strip。

    参数：
        model: 已注入 LoRA 的模型
    返回：
        被替换的模块数量
    """
    # 先收集再替换：边遍历边 setattr 会破坏迭代
    targets: List[tuple] = []
    for module in model.modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                targets.append((module, child_name, child))

    for parent, child_name, child in targets:
        in_features = child.base.in_features
        out_features = child.base.out_features
        has_bias = child.base.bias is not None
        new_linear = nn.Linear(in_features, out_features, bias=has_bias)
        with torch.no_grad():
            new_linear.weight.copy_(child.base.weight)
            if has_bias:
                new_linear.bias.copy_(child.base.bias)
        setattr(parent, child_name, new_linear)
    return len(targets)


def lora_parameters(model: nn.Module) -> List["torch.nn.Parameter"]:
    """
    取出模型里所有 LoRA 的 A / B 参数。

    参数：
        model: 模型
    返回：
        参数列表（顺序稳定：按模块遍历顺序）
    """
    params: List["torch.nn.Parameter"] = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.append(module.lora_A)
            params.append(module.lora_B)
    return params


def mark_only_lora_trainable(model: nn.Module) -> List["torch.nn.Parameter"]:
    """
    冻结模型里**除 LoRA A/B 以外的全部参数**，只留 A / B 可训。

    为什么需要：inject_lora 只冻结了被包装的那个 Linear，
    embedding / RMSNorm / MLP / lm_head 仍然是可训的 ——
    那样就不是 LoRA 微调，而是「大部分参数都在训」。

    参数：
        model: 已注入 LoRA 的模型
    返回：
        可训练的 LoRA 参数列表
    """
    lora_ids = {id(p) for p in lora_parameters(model)}
    for p in model.parameters():
        p.requires_grad = id(p) in lora_ids
    return [p for p in model.parameters() if p.requires_grad]

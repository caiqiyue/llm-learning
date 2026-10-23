"""
MoE + LoRA smoke（对应 feat-031）。

要守的 6 件事：

  1. 把 LoRA 注进 MoE 的专家 FFN 是**无损**的 —— 注入前后 logits 逐元素相等
  2. `inject_lora` 按**子模块名**匹配 ⇒ 每个专家的
     gate_proj / up_proj / down_proj 都被注到（不用改 LoRA 实现）
  3. 冻结基座后**只有 A/B 可训**（专家的 base 权重也被冻住）
  4. ★ **没被路由到的专家，它的 LoRA 参数 `grad is None`**（不是 0.0）；
     被路由到的专家 `grad is not None`
  5. ★ `grad is None` 的参数在 `AdamW.step()` 之后**一动不动**
     （连 weight decay 都不会发生）⇒ 这部分参数是**白注入**的
  6. ★ MoE 的 `loss` 是**纯 cross_entropy** ——
     负载均衡辅助损失只放在 `extras["aux_loss"]` 里，**没有混进 loss**

运行：python tests/test_lora_moe.py
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# 直接执行 `python tests/test_x.py` 时 sys.path[0] 是 tests/ 目录，
# 找不到项目根下的 config.py / model 包 —— 这里把项目根补进 sys.path。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DENSE_CONFIG, MOE_CONFIG
from model.common.lora import (
    LoRALinear,
    inject_lora,
    mark_only_lora_trainable,
)
from model.qwen3_moe import Qwen3MoE, SparseMoeBlock

torch.manual_seed(MOE_CONFIG.seed)

R = MOE_CONFIG.lora_r
ALPHA = MOE_CONFIG.lora_alpha
V = 64

# =====================================================================
# 0) 准备：建一个小的 MoE
# =====================================================================
model = Qwen3MoE(vocab_size=V, cfg=MOE_CONFIG, dropout=0.0)
model.eval()
idx = torch.randint(0, V, (2, 16))

with torch.no_grad():
    logits_before, _, _, _ = model(idx)

# =====================================================================
# 1) 注入 LoRA 到 MoE（attention + 专家 FFN）是**无损**的
# =====================================================================
replaced = inject_lora(model, MOE_CONFIG.lora_target_modules_expert,
                       r=R, alpha=ALPHA, dropout=0.0)
with torch.no_grad():
    logits_after, _, _, _ = model(idx)

assert torch.equal(logits_before, logits_after), (
    "注入 LoRA 后 logits 变了 —— B 零初始化应当保证「注入即无损」"
)

# =====================================================================
# 2) 注入点数量：attention 的 q/v + 每个专家的 gate/up/down
# =====================================================================
n_layer = MOE_CONFIG.n_layer
n_experts = MOE_CONFIG.n_experts
n_attn = len(MOE_CONFIG.lora_target_modules)
n_ffn = len(MOE_CONFIG.lora_target_modules_expert) - n_attn
expect = n_layer * (n_attn + n_experts * n_ffn)
assert len(replaced) == expect, (
    f"应注入 {expect} 个模块（每层 {n_attn} 个 attention + "
    f"{n_experts}×{n_ffn} 个专家 FFN），实得 {len(replaced)}"
)
n_expert_replaced = sum(1 for n in replaced if ".moe.experts." in n)
assert n_expert_replaced == n_layer * n_experts * n_ffn, (
    f"专家侧应注入 {n_layer * n_experts * n_ffn} 个，实得 {n_expert_replaced}"
)

# =====================================================================
# 3) 冻结基座：只有 A/B 可训
# =====================================================================
trainable = mark_only_lora_trainable(model)
trainable_ids = {id(p) for p in trainable}
for name, p in model.named_parameters():
    if id(p) in trainable_ids:
        continue
    assert not p.requires_grad, f"{name} 不该可训（LoRA 的前提是基座冻结）"
for name, m in model.named_modules():
    if isinstance(m, LoRALinear):
        assert m.base.weight.requires_grad is False, f"{name}.base 应被冻结"
assert len(trainable) > 0, "应该有可训的 LoRA 参数"

# =====================================================================
# 4) ★ 真实路由下：没被路由到的专家 grad is None，被路由到的不是 None
#    ★ 用**只有 1 个 token** 的输入：K=2 / E=4 ⇒ 每个专家只可能被选或不选，
#      必然出现「没被选中」的专家（用 16 个 token 时 4 个专家全都会被选上，
#      这条断言就空转了 —— 本仓真踩过，见 feat-031 的返工记录）。
# =====================================================================
model.train()
targets = torch.randint(0, V, (2, 16))
idx_s = torch.randint(0, V, (1, 1))
targets_s = torch.randint(0, V, (1, 1))
_, loss, _, extras = model(idx_s, targets=targets_s)
loss.backward()

assert "expert_load" in extras, "MoE 的 extras 应带 expert_load"
n_unrouted = 0
n_routed = 0
for layer, load in enumerate(extras["expert_load"]):
    for e, cnt in enumerate(load):
        n_slots = cnt * MOE_CONFIG.experts_per_tok
        mods = [m for name, m in model.named_modules()
                if isinstance(m, LoRALinear)
                and f"blocks.{layer}.moe.experts.{e}." in name]
        assert mods, f"blocks.{layer}.moe.experts.{e} 里没有 LoRALinear"
        for m in mods:
            if n_slots == 0:
                n_unrouted += 1
                assert m.lora_B.grad is None, (
                    f"blocks.{layer}.moe.experts.{e} 一步都没被路由到，"
                    f"它的 lora_B.grad 应当是 None（不是 0.0）"
                )
            else:
                n_routed += 1
                assert m.lora_B.grad is not None, (
                    f"blocks.{layer}.moe.experts.{e} 拿到了 {n_slots} 个 slot，"
                    f"它的 lora_B.grad 应当不是 None"
                )
# 防止「断言空转」：两类都必须真的被检查到
assert n_unrouted > 0, "这一步一个「没被路由到」的专家都没有 ⇒ 第 4 条断言空转了"
assert n_routed > 0, "这一步一个「被路由到」的专家都没有 ⇒ 第 4 条断言空转了"

# =====================================================================
# 5) ★ 强制路由：专家 0 永远被选中，专家 1 永远不被选中
#    ⇒ 专家 1 的 LoRA 梯度是 None，且 AdamW 之后**一动不动**
# =====================================================================
class FixedRouter(nn.Module):
    """固定 logits 的路由器（用来**确定性地**控制路由结果）。"""

    def __init__(self, logits: "torch.Tensor") -> None:
        """
        初始化 FixedRouter。

        参数：
            logits: (E,) 每个专家的固定 logit
        """
        super().__init__()
        self.register_buffer("_logits", logits)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        返回与 batch 大小匹配的固定 logits。

        参数：
            x: (N, C)
        返回：
            (N, E)
        """
        return self._logits.unsqueeze(0).expand(x.size(0), -1)


block = SparseMoeBlock(n_embd=16, n_experts=2, experts_per_tok=1,
                       moe_intermediate_size=8, norm_topk_prob=True, dropout=0.0)
inject_lora(block, ("gate_proj", "up_proj", "down_proj"), r=2, alpha=4, dropout=0.0)
block.router = FixedRouter(torch.tensor([10.0, -10.0]))

x = torch.randn(3, 5, 16)
out, stats = block._forward_ffn(x)
assert stats["expert_load"][0] > 0, "专家 0 应当拿到所有 slot"
assert stats["expert_load"][1] == 0, "专家 1 一个 slot 都不该拿到"

out.sum().backward()

e0 = [m for n, m in block.named_modules()
      if isinstance(m, LoRALinear) and n.startswith("experts.0.")]
e1 = [m for n, m in block.named_modules()
      if isinstance(m, LoRALinear) and n.startswith("experts.1.")]
assert len(e0) == 3 and len(e1) == 3, f"每个专家应有 3 个 LoRALinear，实得 {len(e0)}/{len(e1)}"
for m in e0:
    assert m.lora_B.grad is not None, "专家 0 被路由到，lora_B 应有梯度"
    assert float(m.lora_B.grad.norm()) > 0, "专家 0 的梯度不应为 0"
for m in e1:
    assert m.lora_B.grad is None, "专家 1 没被路由到，lora_B.grad 应是 None"
    assert m.lora_A.grad is None, "专家 1 没被路由到，lora_A.grad 应是 None"

# --- AdamW 之后：grad=None 的参数一动不动，grad=0 的参数只被 weight decay 动 ---
params = [p for m in block.modules() if isinstance(m, LoRALinear)
          for p in (m.lora_A, m.lora_B)]
opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.1)
snapshot = {id(p): p.detach().clone() for p in params}
opt.step()
for name, m in block.named_modules():
    if not isinstance(m, LoRALinear):
        continue
    for pname, p in (("lora_A", m.lora_A), ("lora_B", m.lora_B)):
        if p.grad is None:
            assert torch.equal(p.detach(), snapshot[id(p)]), (
                f"{name}.{pname} 的 grad 是 None，AdamW 之后应当**一动不动**"
            )
        elif float(p.grad.norm()) == 0.0:
            # 梯度恰好为 0 ⇒ 只有 weight decay：p ← p · (1 - lr·wd)
            expect = snapshot[id(p)] * (1 - 1e-3 * 0.1)
            assert torch.allclose(p.detach(), expect, atol=1e-9), (
                f"{name}.{pname} 梯度为 0 时，AdamW 应当只做 weight decay"
            )

# =====================================================================
# 6) ★ MoE 的 loss 是**纯 cross_entropy**（aux_loss 没混进去）
# =====================================================================
model.zero_grad(set_to_none=True)
logits, loss2, _, extras2 = model(idx, targets=targets)
manual = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=-100)
assert torch.allclose(loss2, manual, atol=1e-6), (
    f"MoE 的 loss 应等于纯 cross_entropy，实得 {float(loss2)} vs {float(manual)}"
)
assert "aux_loss" in extras2, "extras 应带 aux_loss"
aux = float(extras2["aux_loss"].item())
assert abs(aux) > 0.0, "aux_loss 应当是个非零的有限数"
assert abs(float(loss2.item()) - (float(loss2.item()) + MOE_CONFIG.aux_loss_coef * aux)) > 1e-9, (
    "aux_loss **没有**被折进 loss（否则两者会相等）"
)

# =====================================================================
# 7) 对照：dense 的 extras 是空的，且 loss 同样是纯 CE
# =====================================================================
from model.qwen3_dense import Qwen3Dense          # noqa: E402  （放在后面避免打断上文）

dense = Qwen3Dense(vocab_size=V, cfg=DENSE_CONFIG, dropout=0.0)
dense.eval()
with torch.no_grad():
    d_logits, d_loss, _, d_extras = dense(idx, targets=targets)
d_manual = F.cross_entropy(d_logits.reshape(-1, V), targets.reshape(-1), ignore_index=-100)
assert torch.allclose(d_loss, d_manual, atol=1e-6), "dense 的 loss 应等于纯 cross_entropy"
assert d_extras == {}, f"dense 的 extras 应为空，实得 {d_extras}"

print("test_lora_moe OK  注入无损 / 专家 FFN 全部注入 / 只有 A/B 可训 / "
      "未路由专家 grad=None / AdamW 不动它 / MoE loss 是纯 CE")

"""
LoRA smoke：注入前后输出一致、只有 A/B 有梯度、merge 后等价。

对应 feat-023（开发计划 Task 14）。

运行：python tests/test_lora.py
"""

import os
import sys

import torch
import torch.nn as nn

# 直接执行 `python tests/test_x.py` 时 sys.path[0] 是 tests/ 目录，
# 找不到项目根下的 config.py / model 包 —— 这里把项目根补进 sys.path。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DENSE_CONFIG
from model.common.lora import (
    LoRALinear,
    inject_lora,
    lora_parameters,
    mark_only_lora_trainable,
    merge_lora,
    set_lora_enabled,
)
from model.qwen3_dense import Qwen3Dense

torch.manual_seed(DENSE_CONFIG.seed)

R = DENSE_CONFIG.lora_r
ALPHA = DENSE_CONFIG.lora_alpha

# =====================================================================
# 1) 单个 LoRALinear：B 零初始化 → 输出与 base 完全一致
# =====================================================================
base = nn.Linear(16, 32, bias=False)
x = torch.randn(4, 16)
y0 = base(x)
lora = LoRALinear(base, r=R, alpha=ALPHA)
y1 = lora(x)
assert torch.allclose(y0, y1, atol=1e-6), "B 零初始化时 LoRA 分支应无输出"
assert lora.scaling == ALPHA / R, f"scaling 应为 alpha/r={ALPHA/R}，实得 {lora.scaling}"

# =====================================================================
# 2) base 被冻结
# =====================================================================
assert base.weight.requires_grad is False, "base.weight 应被冻结"
assert lora.lora_A.requires_grad and lora.lora_B.requires_grad, "A/B 应可训练"

# =====================================================================
# 3) 只有 A/B 收到梯度
# =====================================================================
loss = lora(x).sum()
loss.backward()
assert lora.lora_A.grad is not None and lora.lora_B.grad is not None, "A/B 应有梯度"
assert base.weight.grad is None, "base 不应有梯度"

# 带 bias 的 Linear：bias 也要被冻结
base_b = nn.Linear(16, 32, bias=True)
lora_b = LoRALinear(base_b, r=R, alpha=ALPHA)
assert base_b.bias.requires_grad is False, "base.bias 也应被冻结"

# =====================================================================
# 4) 注入整模型后输出逐元素相等（B 零初始化 ⇒ 注入即无损）
# =====================================================================
V = 256
model = Qwen3Dense(vocab_size=V, cfg=DENSE_CONFIG, dropout=0.0)
model.eval()
idx = torch.randint(0, V, (2, 8))
with torch.no_grad():
    logits_before = model(idx)[0]

replaced = inject_lora(model, DENSE_CONFIG.lora_target_modules, r=R, alpha=ALPHA)
assert len(replaced) > 0, "没有注入任何 LoRA 模块"
# 每个 block 的 q_proj / v_proj 各一个 → n_layer * 2
assert len(replaced) == DENSE_CONFIG.n_layer * len(DENSE_CONFIG.lora_target_modules), \
    f"应注入 {DENSE_CONFIG.n_layer * len(DENSE_CONFIG.lora_target_modules)} 个模块，实得 {len(replaced)}"
with torch.no_grad():
    logits_after = model(idx)[0]
assert torch.allclose(logits_before, logits_after, atol=1e-6), \
    f"注入前后输出应逐元素相等，最大差异 {float((logits_before - logits_after).abs().max())}"

# =====================================================================
# 5) set_lora_enabled(False) 等价于关掉 LoRA 分支（= 注入前的 base 行为）
# =====================================================================
set_lora_enabled(model, False)
with torch.no_grad():
    logits_off = model(idx)[0]
assert torch.allclose(logits_before, logits_off, atol=1e-6), "关闭 LoRA 分支后应回到注入前"

# =====================================================================
# 6) merge_lora 后与「开着 LoRA」等价
#    ⚠️ 回归点：merge 把 ΔW 折进 base 之后，必须把分支关掉，
#    否则输出 = (base + ΔW)x + ΔWx = 被算两遍。
# =====================================================================
# 先把 B 训练成非零（否则 ΔW = 0，测不出双算）
with torch.no_grad():
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_B.normal_(0, 0.02)
set_lora_enabled(model, True)
with torch.no_grad():
    logits_on = model(idx)[0]
merged = merge_lora(model)
assert merged > 0, "merge_lora 应至少合并一个模块"
with torch.no_grad():
    logits_merged = model(idx)[0]
assert torch.allclose(logits_on, logits_merged, atol=1e-4), \
    f"merge 后应与未 merge 等价，最大差异 {float((logits_on - logits_merged).abs().max())}"

# =====================================================================
# 7) ΔW 的秩 ≤ r（这才是 LoRA 的「低秩」约束）
# =====================================================================
sub = LoRALinear(nn.Linear(16, 32, bias=False), r=R, alpha=ALPHA).double()
with torch.no_grad():
    sub.lora_B.normal_(0, 0.5)
delta = sub.scaling * (sub.lora_B @ sub.lora_A)
rank = int(torch.linalg.matrix_rank(delta).item())
assert rank <= R, f"ΔW 的秩应 ≤ r={R}，实得 {rank}"
assert sub.merged_weight().shape == (32, 16), "合并权重形状应与 base 一致"

# =====================================================================
# 8) mark_only_lora_trainable：注入后其余参数必须全部冻结
#    （否则「LoRA 微调」悄悄变成了「大部分参数都在训」）
# =====================================================================
model2 = Qwen3Dense(vocab_size=V, cfg=DENSE_CONFIG, dropout=0.0)
inject_lora(model2, DENSE_CONFIG.lora_target_modules, r=R, alpha=ALPHA)
trainable = mark_only_lora_trainable(model2)
assert len(trainable) > 0, "应至少有一组可训练参数"
for p in model2.parameters():
    is_lora = any(p is q for q in trainable)
    assert p.requires_grad == is_lora, \
        "非 LoRA 参数必须冻结，LoRA 参数必须可训"
# 可训练参数量 = 每个注入模块的 (r*in + out*r)
expected = 0
for m in model2.modules():
    if isinstance(m, LoRALinear):
        expected += m.lora_A.numel() + m.lora_B.numel()
assert sum(p.numel() for p in trainable) == expected, "可训练参数量与 LoRA 参数量不符"
assert sum(p.numel() for p in model2.parameters() if p.requires_grad) < \
    sum(p.numel() for p in model2.parameters()) * 0.2, "可训练参数占比应远小于 20%"
print(f"[可训练参数] {sum(p.numel() for p in trainable):,} / "
      f"{sum(p.numel() for p in model2.parameters()):,} "
      f"({sum(p.numel() for p in trainable)/sum(p.numel() for p in model2.parameters()):.2%})")

# lora_parameters 与 mark_only_lora_trainable 应给同一批参数
assert len(lora_parameters(model2)) == len(trainable), "两个 helper 应返回同一批参数"

# =====================================================================
# 9) 边界情况：没有可注入的模块 / 没有 LoRA 的模型
# =====================================================================
assert inject_lora(model2, ("no_such_module",), r=R, alpha=ALPHA) == [], \
    "不存在的目标模块应返回空列表"
set_lora_enabled(model2, False)   # 没有 LoRA 时不应报错
assert merge_lora(nn.Linear(4, 4)) == 0, "没有 LoRA 时 merge 应返回 0"

print(f"test_lora OK  replaced={len(replaced)} r={R} alpha={ALPHA} "
      f"scaling={ALPHA/R}")

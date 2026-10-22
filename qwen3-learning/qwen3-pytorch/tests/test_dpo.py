"""
DPO smoke：目标函数的闭式正确性、π_θ / π_ref 机制、梯度只到 LoRA、strip/merge 往返。

对应 feat-024（开发计划 Task 15）。

运行：python tests/test_dpo.py
"""

import os
import sys

import torch
import torch.nn.functional as F

# 直接执行 `python tests/test_dpo.py` 时 sys.path[0] 是 tests/ 目录，
# 找不到项目根下的 config.py / model 包 —— 这里把项目根补进 sys.path。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DENSE_CONFIG
from data.preference_data import generate_preference_dataset
from data.tokenizer import BPETokenizer
from model.common.lora import (
    LoRALinear,
    inject_lora,
    lora_parameters,
    mark_only_lora_trainable,
    merge_lora,
    set_lora_enabled,
    strip_lora,
)
from model.qwen3_dense import Qwen3Dense
from train.train_dpo import dpo_loss, evaluate_dpo
from train.train_lora import encode_pair

torch.manual_seed(DENSE_CONFIG.seed)

BETA = DENSE_CONFIG.dpo_beta

tk = BPETokenizer("data/vocab.json", "data/merges.txt")
pairs = generate_preference_dataset(n_samples=8, seed=DENSE_CONFIG.seed)
V = tk.vocab_size
device = torch.device("cpu")

# =====================================================================
# 1) strip_lora 的往返：inject → strip 应当回到普通 nn.Linear，
#    且权重与「注入前的 base」完全一致
# =====================================================================
model = Qwen3Dense(vocab_size=V, cfg=DENSE_CONFIG, dropout=0.0)
model.eval()
idx = torch.randint(0, V, (2, 8))
with torch.no_grad():
    plain_logits = model(idx)[0]

replaced = inject_lora(model, DENSE_CONFIG.lora_target_modules,
                       r=DENSE_CONFIG.lora_r, alpha=DENSE_CONFIG.lora_alpha)
n_stripped = strip_lora(model)
assert n_stripped == len(replaced), f"strip 的模块数应与注入数一致：{n_stripped} vs {len(replaced)}"
assert not any(isinstance(m, LoRALinear) for m in model.modules()), "strip 后不应再有 LoRALinear"
with torch.no_grad():
    stripped_logits = model(idx)[0]
assert torch.allclose(plain_logits, stripped_logits, atol=1e-6), \
    "inject → strip（B=0，未 merge）应当回到注入前的输出"
print("[strip] inject → strip 往返无损")

# =====================================================================
# 2) merge → strip：SFT 的 ΔW 被折进 base，模型仍与「开着 LoRA」等价
# =====================================================================
model = Qwen3Dense(vocab_size=V, cfg=DENSE_CONFIG, dropout=0.0)
model.eval()
inject_lora(model, DENSE_CONFIG.lora_target_modules,
            r=DENSE_CONFIG.lora_r, alpha=DENSE_CONFIG.lora_alpha)
# 把 B 训成非零，模拟「SFT 学到了东西」
with torch.no_grad():
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_B.normal_(0, 0.02)
set_lora_enabled(model, True)
with torch.no_grad():
    logits_on = model(idx)[0]
assert merge_lora(model) > 0, "merge_lora 应至少合并一个模块"
assert strip_lora(model) > 0, "strip_lora 应至少替换一个模块"
with torch.no_grad():
    logits_merged = model(idx)[0]
assert torch.allclose(logits_on, logits_merged, atol=1e-4), \
    f"merge + strip 后应与「开着 LoRA」等价，最大差异 {float((logits_on - logits_merged).abs().max())}"
print("[strip] merge → strip 保住 SFT 的 ΔW")

# =====================================================================
# 3) DPO 初始状态：B 零初始化 ⇒ π_θ == π_ref ⇒ margin = 0、loss = log 2
# =====================================================================
model = Qwen3Dense(vocab_size=V, cfg=DENSE_CONFIG, dropout=0.0)
model.eval()
inject_lora(model, DENSE_CONFIG.lora_target_modules,
            r=DENSE_CONFIG.lora_r, alpha=DENSE_CONFIG.lora_alpha)
set_lora_enabled(model, True)
with torch.no_grad():
    a = model(idx)[0]
set_lora_enabled(model, False)
with torch.no_grad():
    b = model(idx)[0]
assert torch.allclose(a, b, atol=1e-6), "B 零初始化时开关 LoRA 分支输出应相同"

loss, margin, stats = dpo_loss(model, pairs[0], tk, BETA, device)
assert torch.isfinite(loss), f"loss 非有限: {loss}"
assert torch.isfinite(margin), f"margin 非有限: {margin}"
assert abs(float(margin.item())) < 1e-4, \
    f"初始 margin 应为 0（π_θ == π_ref），实得 {float(margin.item())}"
assert abs(float(loss.item()) - float(torch.log(torch.tensor(2.0)))) < 1e-4, \
    f"初始 loss 应为 log 2 = {float(torch.log(torch.tensor(2.0))):.4f}，实得 {float(loss.item())}"
print(f"[dpo] 初始 margin={float(margin.item()):+.6f} loss={float(loss.item()):.4f}（= log 2）")

# =====================================================================
# 4) 目标函数的闭式正确性：手工算一遍，和 dpo_loss 的返回值逐位对上
# =====================================================================
# 把 B 调成非零，让 π_θ ≠ π_ref
with torch.no_grad():
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_B.normal_(0, 0.02)

pair = pairs[0]
set_lora_enabled(model, True)
(lp_w_theta, _), (lp_l_theta, _) = encode_pair(model, tk, pair, device)
set_lora_enabled(model, False)
with torch.no_grad():
    (lp_w_ref, _), (lp_l_ref, _) = encode_pair(model, tk, pair, device)
set_lora_enabled(model, True)

expect_margin = BETA * ((lp_w_theta - lp_w_ref) - (lp_l_theta - lp_l_ref))
expect_loss = -F.logsigmoid(expect_margin).mean()
loss2, margin2, stats2 = dpo_loss(model, pair, tk, BETA, device)
assert torch.allclose(margin2, expect_margin, atol=1e-5), \
    f"margin 与闭式不符：{float(margin2)} vs {float(expect_margin)}"
assert torch.allclose(loss2, expect_loss, atol=1e-5), \
    f"loss 与闭式不符：{float(loss2)} vs {float(expect_loss)}"
# stats 里的 logprob 也要对得上
assert abs(stats2["lp_w_theta"] - float(lp_w_theta.item())) < 1e-3
assert abs(stats2["lp_l_ref"] - float(lp_l_ref.item())) < 1e-3
print(f"[dpo] 闭式校验通过：margin={float(margin2):+.4f} loss={float(loss2):.4f}")

# =====================================================================
# 5) 梯度只到 LoRA 参数
# =====================================================================
trainable = mark_only_lora_trainable(model)
assert len(trainable) > 0
for p in model.parameters():
    is_lora = any(p is q for q in trainable)
    assert p.requires_grad == is_lora, "非 LoRA 参数必须冻结"
loss2.backward()
for p in trainable:
    assert p.grad is not None, "LoRA 参数应有梯度"
    assert torch.isfinite(p.grad).all(), "梯度应有限"
n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
assert n_with_grad == len(trainable), \
    f"只有 LoRA 参数该有梯度：{n_with_grad} vs {len(trainable)}"
print(f"[dpo] 梯度只到 {len(trainable)} 个 LoRA 张量（{n_with_grad} 个有梯度）")

# =====================================================================
# 6) 一步优化后 margin 应该上升（最小可跑训练）
# =====================================================================
model.zero_grad(set_to_none=True)
opt = torch.optim.AdamW(trainable, lr=1e-3)
margins_before = []
margins_after = []
for p in pairs[:4]:
    l, m, _ = dpo_loss(model, p, tk, BETA, device)
    margins_before.append(float(m.item()))
    opt.zero_grad(set_to_none=True)
    l.backward()
    opt.step()
for p in pairs[:4]:
    l, m, _ = dpo_loss(model, p, tk, BETA, device)
    margins_after.append(float(m.item()))
avg_b = sum(margins_before) / len(margins_before)
avg_a = sum(margins_after) / len(margins_after)
assert avg_a > avg_b, f"一步优化后 margin 应上升：{avg_b:+.4f} → {avg_a:+.4f}"
print(f"[dpo] 一步优化 margin {avg_b:+.4f} → {avg_a:+.4f}")

# =====================================================================
# 7) evaluate_dpo 的指标自洽：dpo_acc 与 margin 的符号一致
# =====================================================================
m_eval = evaluate_dpo(model, tk, pairs, device, BETA, max_samples=8)
assert 0.0 <= m_eval["dpo_acc"] <= 1.0
assert 0.0 <= m_eval["win_rate"] <= 1.0
assert m_eval["margin_std"] >= 0.0
assert set(m_eval["by_strategy"].keys()), "应按策略分组"
# 逐条复算 dpo_acc，验证与实现一致
manual = 0
for pair in pairs[:8]:
    set_lora_enabled(model, True)
    (lp_w, _), (lp_l, _) = encode_pair(model, tk, pair, device)
    set_lora_enabled(model, False)
    (lp_w_ref, _), (lp_l_ref, _) = encode_pair(model, tk, pair, device)
    set_lora_enabled(model, True)
    margin = float(BETA * ((lp_w - lp_w_ref) - (lp_l - lp_l_ref)).item())
    manual += 1 if margin > 0 else 0
assert abs(m_eval["dpo_acc"] - manual / 8) < 1e-6, \
    f"dpo_acc 与逐条复算不符：{m_eval['dpo_acc']} vs {manual/8}"
print(f"[dpo] evaluate_dpo 自洽：dpo_acc={m_eval['dpo_acc']:.3f} margin={m_eval['margin_mean']:+.4f}")

# =====================================================================
# 8) 边界：β = 0 ⇒ margin 恒为 0、loss 恒为 log 2
# =====================================================================
loss0, margin0, _ = dpo_loss(model, pair, tk, 0.0, device)
assert abs(float(margin0.item())) < 1e-6, "β=0 时 margin 应为 0"
assert abs(float(loss0.item()) - float(torch.log(torch.tensor(2.0)))) < 1e-4, \
    "β=0 时 loss 应恒为 log 2"
print("[dpo] β=0 边界通过")

# =====================================================================
# 9) lora_parameters 与 mark_only_lora_trainable 返回同一批参数
# =====================================================================
assert len(lora_parameters(model)) == len(trainable), "两个 helper 应返回同一批参数"

print(f"test_dpo OK  beta={BETA} loss={float(loss2):.4f} margin={float(margin2):+.4f} "
      f"dpo_acc={m_eval['dpo_acc']:.3f}")

"""
MoE + DPO smoke（对应 feat-032）。

要守的 9 件事：

  1. MoE 的 LoRA 注入集合命中**每一个专家的 FFN**（attention + 专家 FFN）
  2. `inject → strip` 往返**无损**（B 零初始化），在专家 FFN 上同样成立
  3. ★ **`merge → strip` 在专家 FFN 上等价** ——
     这是 DPO 的核心机制（π_ref = SFT 模型）在 MoE 上能不能用的前提：
     merge 把 ΔW 折进专家的 base 权重，而**路由器是冻结的** ⇒ 路由不变 ⇒ 数值等价
  4. ★ **DPO 的核心不变量在 MoE 上成立**：
     B 零初始化 ⇒ π_θ ≡ π_ref ⇒ margin **恒为 0**、loss **恒为 log 2**
  5. ★ **梯度只到 LoRA 参数**；**路由器全程冻结**（这正是第 3 条成立的原因）
  6. ★ **白注入机制在 DPO 里同样成立**：
     没被路由到的专家 ⇒ LoRA 参数 `grad is None` ⇒ AdamW 整个跳过 ⇒ 参数一动不动
  7. ★ **白注入是「模型状态」的函数，不是 MoE 的固有属性**（需要 checkpoint 才验）：
     同一份代码、同一批数据，**预训练** MoE 有 0%~12.5% 的死专家，
     **LoRA-SFT 之后**的 MoE 则是 **0%** —— 因为 LoRA 改变了残差流 ⇒ 路由器看到的输入变了
  8. `check_lora_compat` 真的能拦住「注入集合错配」（DPO 加载 SFT 权重的头号坑）
  9. β = 0 边界、闭式校验、`evaluate_dpo` 自洽、一步优化后 margin 上升

运行：python tests/test_dpo_moe.py
"""

import os
import sys

import torch
import torch.nn.functional as F

# 直接执行 `python tests/test_dpo_moe.py` 时 sys.path[0] 是 tests/ 目录，
# 找不到项目根下的 config.py / model 包 —— 这里把项目根补进 sys.path。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MOE_CONFIG
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
from model.qwen3_moe import Qwen3MoE
from train.train_dpo import (
    build_policy,
    check_lora_compat,
    dpo_loss,
    evaluate_dpo,
    lora_targets_for,
)
from train.train_lora import collect_expert_stats, encode_pair

torch.manual_seed(MOE_CONFIG.seed)

BETA = MOE_CONFIG.dpo_beta
TARGETS = MOE_CONFIG.lora_target_modules_expert
R = MOE_CONFIG.lora_r
ALPHA = MOE_CONFIG.lora_alpha

tk = BPETokenizer("data/vocab.json", "data/merges.txt")
pairs = generate_preference_dataset(n_samples=8, seed=MOE_CONFIG.seed)
V = tk.vocab_size
device = torch.device("cpu")


def fresh_moe(vocab_size: int = V) -> "torch.nn.Module":
    """建一个**没注入 LoRA** 的 MoE（dropout=0 ⇒ 完全确定性）。"""
    m = Qwen3MoE(vocab_size=vocab_size, cfg=MOE_CONFIG, dropout=0.0)
    m.eval()
    return m


# =====================================================================
# 1) 注入集合：attention + **每一个**专家的 FFN 都被命中
# =====================================================================
model = fresh_moe()
idx = torch.randint(0, V, (2, 16))
with torch.no_grad():
    logits_before = model(idx)[0]

replaced = inject_lora(model, TARGETS, r=R, alpha=ALPHA)
n_layer = MOE_CONFIG.n_layer
n_experts = MOE_CONFIG.n_experts
n_attn = len(MOE_CONFIG.lora_target_modules)          # q_proj / v_proj
n_ffn = len(TARGETS) - n_attn                          # 专家 FFN 的三个投影
expect = n_layer * (n_attn + n_experts * n_ffn)
assert len(replaced) == expect, \
    f"应注入 {expect} 个模块（每层 {n_attn} 个 attention + {n_experts}×{n_ffn} 个专家 FFN），" \
    f"实得 {len(replaced)}"
n_expert_replaced = sum(1 for n in replaced if ".moe.experts." in n)
assert n_expert_replaced == n_layer * n_experts * n_ffn, \
    f"专家侧应注入 {n_layer * n_experts * n_ffn} 个，实得 {n_expert_replaced}"
assert lora_targets_for("moe", MOE_CONFIG) == tuple(TARGETS), \
    "lora_targets_for('moe') 应返回专家版注入集合"
print(f"[inject] 命中 {len(replaced)} 个 LoRALinear（专家侧 {n_expert_replaced} 个）")

# =====================================================================
# 2) `inject → strip` 往返无损（B 零初始化），在专家 FFN 上同样成立
# =====================================================================
with torch.no_grad():
    logits_injected = model(idx)[0]
assert torch.equal(logits_before, logits_injected), \
    "注入 LoRA 到专家 FFN 后 logits 变了 —— B 零初始化应当保证「注入即无损」"

n_stripped = strip_lora(model)
assert n_stripped == len(replaced), \
    f"strip 的模块数应与注入数一致：{n_stripped} vs {len(replaced)}"
assert not any(isinstance(m, LoRALinear) for m in model.modules()), \
    "strip 后不应再有 LoRALinear"
with torch.no_grad():
    logits_stripped = model(idx)[0]
assert torch.equal(logits_before, logits_stripped), \
    "inject → strip（B=0，未 merge）应当回到注入前的输出"
print("[strip] inject → strip 往返无损（含专家 FFN）")

# =====================================================================
# 3) ★ `merge → strip` 在**专家 FFN** 上等价 —— DPO 的核心机制在 MoE 上能不能用
# =====================================================================
# 做法：把 B 训成非零（模拟「SFT 学到了东西」），
# 记下「开着 LoRA」的输出；再 merge → strip，记下「折进 base 后」的输出；两者必须相等。
#
# ★ 为什么 MoE 上也成立：merge 只改**专家的 base 权重**，
#   而**路由器（router）是冻结的、也不参与 merge** ⇒ 路由结果完全不变
#   ⇒ 同一个 token 仍然送到同一个专家 ⇒ 输出逐元素相等。
#   （如果哪天把 router 也训了 / 把 LoRA 注进 router，这条就会红 —— 那正是要守的边界。）
model = fresh_moe()
inject_lora(model, TARGETS, r=R, alpha=ALPHA)
with torch.no_grad():
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_B.normal_(0, 0.02)
set_lora_enabled(model, True)
with torch.no_grad():
    logits_on = model(idx)[0]

n_merged = merge_lora(model)
assert n_merged == expect, f"merge 应合并 {expect} 个模块，实得 {n_merged}"
n_stripped = strip_lora(model)
assert n_stripped == expect
with torch.no_grad():
    logits_merged = model(idx)[0]
max_diff = float((logits_on - logits_merged).abs().max())
assert torch.allclose(logits_on, logits_merged, atol=1e-4), \
    f"merge + strip 后应与「开着 LoRA」等价，最大差异 {max_diff}"
print(f"[merge] merge → strip 在专家 FFN 上等价（{n_merged} 个模块，最大差异 {max_diff:.2e}）")

# =====================================================================
# 4) ★ DPO 的核心不变量在 MoE 上成立：B=0 ⇒ margin ≡ 0、loss ≡ log 2
# =====================================================================
model = fresh_moe()
inject_lora(model, TARGETS, r=R, alpha=ALPHA)
set_lora_enabled(model, True)
with torch.no_grad():
    a = model(idx)[0]
set_lora_enabled(model, False)
with torch.no_grad():
    b = model(idx)[0]
assert torch.equal(a, b), "B 零初始化时开关 LoRA 分支输出应**逐位相同**（不是 allclose）"

loss, margin, stats = dpo_loss(model, pairs[0], tk, BETA, device)
log2 = float(torch.log(torch.tensor(2.0)))
assert torch.isfinite(loss), f"loss 非有限: {loss}"
assert abs(float(margin.item())) < 1e-6, \
    f"初始 margin 应为 0（π_θ ≡ π_ref），实得 {float(margin.item())}"
assert abs(float(loss.item()) - log2) < 1e-6, \
    f"初始 loss 应为 log 2 = {log2:.4f}，实得 {float(loss.item())}"
print(f"[dpo-moe] 初始 margin={float(margin.item()):+.8f} loss={float(loss.item()):.4f}（= log 2）")

# =====================================================================
# 5) 闭式校验：margin / loss 与手工算的逐位对上
# =====================================================================
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
print(f"[dpo-moe] 闭式校验通过：margin={float(margin2):+.4f} loss={float(loss2):.4f}")

# =====================================================================
# 6) ★ 梯度只到 LoRA 参数；**路由器全程冻结**（第 3 条成立的原因）
# =====================================================================
# ⚠️ 顺序很重要：**先冻结、再重新算一次 loss、再 backward**。
#    如果先算 loss 再冻结，图里已经留下了 base 参数的 AccumulateGrad 边，
#    backward 时它们的 .grad 也会被填上 ⇒ 「梯度只到 LoRA」这条就验不准了。
trainable = mark_only_lora_trainable(model)
assert len(trainable) > 0
trainable_ids = {id(p) for p in trainable}
for p in model.parameters():
    assert p.requires_grad == (id(p) in trainable_ids), "非 LoRA 参数必须冻结"
for name, p in model.named_parameters():
    if "router" in name:
        assert not p.requires_grad, f"路由器必须冻结：{name}"

loss_frozen, _, _ = dpo_loss(model, pair, tk, BETA, device)
loss_frozen.backward()
for p in trainable:
    if p.grad is not None:
        assert torch.isfinite(p.grad).all(), "梯度应有限"
n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
assert 0 < n_with_grad <= len(trainable), \
    f"应有至少一个 LoRA 参数得到梯度，且不能超过可训参数数：{n_with_grad} vs {len(trainable)}"
assert all(p.requires_grad for p in model.parameters() if p.grad is not None), \
    "拿到梯度的参数必须属于 LoRA 可训参数"
for name, p in model.named_parameters():
    if "router" in name:
        assert p.grad is None, f"路由器的梯度应为 None（冻结 ⇒ 路由不受训练影响）：{name}"
print(f"[dpo-moe] 梯度只到 {len(trainable)} 个 LoRA 张量；路由器全程冻结")

# =====================================================================
# 7) ★ 白注入机制在 DPO 里同样成立：
#    没被路由到的专家 ⇒ LoRA 参数 grad is None ⇒ AdamW 整个跳过 ⇒ 参数一动不动
# =====================================================================
# 做法：把**最后一层**的路由器换成「固定 logits」的假路由器，
#       让 3 号专家**永远**不被选中；然后跑一步 DPO，看它的 LoRA 参数。
model = fresh_moe()
inject_lora(model, TARGETS, r=R, alpha=ALPHA)
mark_only_lora_trainable(model)


class FixedRouter(torch.nn.Module):
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


last_block = model.blocks[-1].moe
fixed = FixedRouter(torch.tensor([10.0, 9.0, 8.0, -10.0]))
last_block.router = fixed

# 让第 3 个专家（永远不被选中）的 LoRA 参数进入 DPO 的一次前向
loss3, margin3, _ = dpo_loss(model, pairs[0], tk, BETA, device)
loss3.backward()
dead = [m for n, m in model.named_modules()
        if isinstance(m, LoRALinear) and f"blocks.{n_layer - 1}.moe.experts.3." in n]
assert len(dead) == n_ffn, f"第 {n_layer - 1} 层专家 3 应有 {n_ffn} 个 LoRALinear，实得 {len(dead)}"
for m in dead:
    assert m.lora_A.grad is None, "没被路由到的专家：lora_A.grad 应当是 None"
    assert m.lora_B.grad is None, "没被路由到的专家：lora_B.grad 应当是 None"
# 反过来：被固定路由器选中的专家**必须**拿到梯度（防止「两条断言一起空转」）
alive = [m for n, m in model.named_modules()
         if isinstance(m, LoRALinear) and f"blocks.{n_layer - 1}.moe.experts.0." in n]
assert len(alive) == n_ffn
for m in alive:
    assert m.lora_B.grad is not None, "被路由到的专家：lora_B.grad 应当不是 None"
    assert float(m.lora_B.grad.norm()) > 0, "被路由到的专家：梯度不应为 0"

# AdamW 之后：grad 是 None 的参数**一动不动**（连 weight decay 都没有）
params = lora_parameters(model)
snapshot = {id(p): p.detach().clone() for p in params}
opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.1)
opt.step()
for m in dead:
    for pname in ("lora_A", "lora_B"):
        p = getattr(m, pname)
        assert torch.equal(p.detach(), snapshot[id(p)]), \
            f"grad 是 None 的参数在 AdamW 之后应当**一动不动**（{pname}）"
print(f"[dpo-moe] 白注入机制在 DPO 里成立：未路由专家 {len(dead)} 个 LoRALinear 梯度为 None，"
      f"AdamW 之后一动不动")

# =====================================================================
# 8) ★ 在**真实的 DPO 前向**（默认路由器）上，量一量白注入到底有多少
# =====================================================================
model = fresh_moe()
inject_lora(model, TARGETS, r=R, alpha=ALPHA)
mark_only_lora_trainable(model)
n_unrouted_steps = 0
n_samples = 12
for pair in pairs[:n_samples]:
    model.zero_grad(set_to_none=True)
    l, _, _ = dpo_loss(model, pair, tk, BETA, device)
    l.backward()
    stats_map = collect_expert_stats(model)
    if any(v["n_none"] > 0 for v in stats_map.values()):
        n_unrouted_steps += 1
print(f"[dpo-moe] 真实 DPO 前向（{n_samples} 对偏好）：出现「未路由专家」的步数 = "
      f"{n_unrouted_steps}/{n_samples}")

# =====================================================================
# 9) `check_lora_compat` 真的能拦住「注入集合错配」
# =====================================================================
# DPO 加载的是 **SFT 之后**的 LoRA 权重，state_dict 的键名由「当初注入了哪些模块」决定。
# 如果 DPO 侧注的集合与 SFT 侧不一致，`load_state_dict(strict=True)` 会红；
# 但更危险的是「键名自带模块全名 ⇒ 键对上了但语义错配」——
# 所以要有一道**显式**的核对，把 target_modules / r / alpha 逐项对一遍。
bad_ckpt = {"lora": {"r": R, "alpha": ALPHA, "target_modules": list(TARGETS)},
            "model_state": {}}
try:
    check_lora_compat(bad_ckpt, ("q_proj", "v_proj"), MOE_CONFIG, "fake.pt")
except ValueError as exc:
    assert "注入集合不匹配" in str(exc), f"报错信息应说明是注入集合不匹配，实得：{exc}"
else:
    raise AssertionError("注入集合不匹配时 check_lora_compat 应当抛 ValueError")

try:
    check_lora_compat({"lora": {"r": R + 1, "alpha": ALPHA, "target_modules": list(TARGETS)}},
                      tuple(TARGETS), MOE_CONFIG, "fake.pt")
except ValueError as exc:
    assert "r 不匹配" in str(exc), f"报错信息应说明是 r 不匹配，实得：{exc}"
else:
    raise AssertionError("r 不匹配时 check_lora_compat 应当抛 ValueError")

# 一致的 checkpoint 不该报错
check_lora_compat({"lora": {"r": R, "alpha": ALPHA, "target_modules": list(TARGETS)}},
                  tuple(TARGETS), MOE_CONFIG, "fake.pt")
print("[dpo-moe] check_lora_compat 能拦住注入集合 / r 错配")

# =====================================================================
# 10) β = 0 边界、evaluate_dpo 自洽、一步优化后 margin 上升
# =====================================================================
loss0, margin0, _ = dpo_loss(model, pair, tk, 0.0, device)
assert abs(float(margin0.item())) < 1e-6, "β=0 时 margin 应为 0"
assert abs(float(loss0.item()) - log2) < 1e-6, "β=0 时 loss 应恒为 log 2"

# 最小可跑训练：先训几步，让 B ≠ 0（否则 margin 全是 0，下面的自洽检查会空转）
model2 = fresh_moe()
inject_lora(model2, TARGETS, r=R, alpha=ALPHA)
trainable2 = mark_only_lora_trainable(model2)
opt2 = torch.optim.AdamW(trainable2, lr=1e-3)
margins_before = []
for pr in pairs[:4]:
    l, m, _ = dpo_loss(model2, pr, tk, BETA, device)
    margins_before.append(float(m.item()))
    opt2.zero_grad(set_to_none=True)
    l.backward()
    opt2.step()
margins_after = []
for pr in pairs[:4]:
    l, m, _ = dpo_loss(model2, pr, tk, BETA, device)
    margins_after.append(float(m.item()))
avg_b = sum(margins_before) / len(margins_before)
avg_a = sum(margins_after) / len(margins_after)
assert avg_a > avg_b, f"一步优化后 margin 应上升：{avg_b:+.4f} → {avg_a:+.4f}"

# evaluate_dpo 自洽：dpo_acc 与逐条复算一致（用训过的 model2，margin 才不为 0）
m_eval = evaluate_dpo(model2, tk, pairs, device, BETA, max_samples=8)
assert 0.0 <= m_eval["dpo_acc"] <= 1.0
assert 0.0 <= m_eval["win_rate"] <= 1.0
assert m_eval["margin_std"] >= 0.0
assert set(m_eval["by_strategy"].keys()), "应按策略分组"
manual = 0
for pr in pairs[:8]:
    set_lora_enabled(model2, True)
    (lp_w, _), (lp_l, _) = encode_pair(model2, tk, pr, device)
    set_lora_enabled(model2, False)
    (lp_w_ref, _), (lp_l_ref, _) = encode_pair(model2, tk, pr, device)
    set_lora_enabled(model2, True)
    mg = float(BETA * ((lp_w - lp_w_ref) - (lp_l - lp_l_ref)).item())
    manual += 1 if mg > 0 else 0
assert abs(m_eval["dpo_acc"] - manual / 8) < 1e-6, \
    f"dpo_acc 与逐条复算不符：{m_eval['dpo_acc']} vs {manual / 8}"
print(f"[dpo-moe] β=0 边界 / evaluate_dpo 自洽（dpo_acc={m_eval['dpo_acc']:.3f}）/ "
      f"一步优化 margin {avg_b:+.4f} → {avg_a:+.4f}")

# =====================================================================
# 11) 端到端：走一遍 build_policy（**需要 checkpoints/lora_moe.pt**）
# =====================================================================
CKPT = os.path.join("checkpoints", "lora_moe.pt")
if os.path.exists(CKPT):
    policy = build_policy("moe", CKPT, tk, device, "sft")
    trainable_p = mark_only_lora_trainable(policy)
    n_train = sum(p.numel() for p in trainable_p)
    n_all = sum(p.numel() for p in policy.parameters())
    n_expert_train = sum(p.numel() for name, p in policy.named_parameters()
                         if ".moe.experts." in name and p.requires_grad)
    print(f"[dpo-moe] build_policy：可训练 {n_train:,}/{n_all:,} "
          f"({n_train / n_all:.2%})，其中专家侧 {n_expert_train:,}")
    assert n_train < n_all * 0.1, "可训练参数应远小于总参数"
    # 注入集合必须与 SFT 时一致（否则 load_state_dict(strict=True) 会当场红）
    assert n_expert_train > 0, "专家侧应有可训参数（说明注入集合含专家 FFN）"
    # B 零初始化 ⇒ margin 恒为 0
    m_before = evaluate_dpo(policy, tk, pairs, device, BETA, max_samples=8)
    assert abs(m_before["margin_mean"]) < 1e-4, \
        f"build_policy 之后初始 margin 应为 0，实得 {m_before['margin_mean']}"
    print(f"[dpo-moe] build_policy 后 margin={m_before['margin_mean']:+.8f}（应为 0）")
else:
    print(f"[dpo-moe] 跳过第 11 条：{CKPT} 不存在（先跑 train/train_lora.py --model moe）")

print("test_dpo_moe OK  专家 FFN 注入 / inject→strip 无损 / merge→strip 等价 / "
      "margin≡0 且 loss≡log2 / 梯度只到 LoRA 且路由器冻结 / 白注入机制成立 / "
      "check_lora_compat 能拦错配 / β=0 边界")

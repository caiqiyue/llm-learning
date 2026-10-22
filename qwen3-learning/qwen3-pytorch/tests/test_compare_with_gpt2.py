"""
退化对照实验的 smoke（feat-017）。

对应开发计划 Task 18。

跑的是「Qwen3 退化成 GPT-2 结构后，输出还能不能对齐」这一条结论，
外加两条**反证**：证明它不是「碰巧通过」。

运行：python tests/test_compare_with_gpt2.py
"""

import os
import sys

import torch

# 直接执行 `python tests/test_compare_with_gpt2.py` 时 sys.path[0] 是 tests/，
# 找不到项目根下的 config.py / ablations/ —— 这里把项目根补进 sys.path。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ablations.compare_with_gpt2 import (  # noqa: E402
    GPT2_N_INNER,
    build_cases,
    build_target_state,
    degraded_config,
    load_gpt2_module,
    run_all,
)
from config import DENSE_CONFIG  # noqa: E402
from model.qwen3_dense import Qwen3Dense  # noqa: E402

THRESHOLD = 1e-4
VOCAB = 293
SEQ_LEN = 16
BATCH = 2

# =====================================================================
# 1) 消融表跑通：6 行，全关那行 < 1e-4，单开组件的行都 > 1e-4
# =====================================================================
result = run_all(threshold=THRESHOLD, seq_len=SEQ_LEN, batch=BATCH, vocab_size=VOCAB)
cases = result["cases"]

assert len(cases) == 6, f"消融表应有 6 行（1 基线 + 5 单组件），实得 {len(cases)}"

baseline = cases[0]
assert baseline["max_abs_diff"] < THRESHOLD, \
    f"全关（退化成 GPT-2）时 max_abs_diff 应 < {THRESHOLD}，实得 {baseline['max_abs_diff']:.3e}"
print(f"[compare] 全关（退化成 GPT-2）max_abs_diff = {baseline['max_abs_diff']:.3e} "
      f"（阈值 {THRESHOLD:.0e}）")

for r in cases[1:]:
    assert r["max_abs_diff"] > THRESHOLD, \
        f"单开组件「{r['name']}」的 max_abs_diff 应 > {THRESHOLD}，" \
        f"实得 {r['max_abs_diff']:.3e} —— 该组件在结构上没起作用，消融表失去说服力"
    print(f"[compare] {r['name']:<16} max_abs_diff = {r['max_abs_diff']:.3e} "
          f"（相对差 {r['rel_diff']:.2e}）")

# =====================================================================
# 2) 反证 A：**不加载权重**时，退化的 Qwen3 与 GPT-2 必须差很远
#    —— 证明第 1 组不是「拿自己跟自己比」
# =====================================================================
gpt2_mod = load_gpt2_module()
GPT2 = gpt2_mod.GPT2

torch.manual_seed(42)
gpt2 = GPT2(vocab_size=VOCAB, n_layer=4, n_head=4, n_embd=128, n_inner=GPT2_N_INNER,
            block_size=128, dropout=0.0, bias=False)
gpt2.eval()

cfg = degraded_config()
torch.manual_seed(7)          # 故意用**不同**的种子 ⇒ Qwen3 是随机权重
qwen3_random = Qwen3Dense(vocab_size=VOCAB, cfg=cfg, dropout=0.0)
qwen3_random.eval()

idx = torch.randint(0, VOCAB, (BATCH, SEQ_LEN), dtype=torch.long)
with torch.no_grad():
    diff_no_map = float((gpt2(idx)[0] - qwen3_random(idx)[0]).abs().max().item())
assert diff_no_map > 1e-3, \
    f"不加载权重时两者应当差很多（实测 {diff_no_map:.3e}）—— 说明第 1 组不是假通过"
print(f"[compare] 反证 A：不加载权重 → max_abs_diff = {diff_no_map:.3e}（远大于阈值）")

# =====================================================================
# 3) 反证 B：**权重确实搬过去了**（逐条核对映射表）
# =====================================================================
torch.manual_seed(42)
qwen3 = Qwen3Dense(vocab_size=VOCAB, cfg=cfg, dropout=0.0)
sd = build_target_state(gpt2, qwen3, cfg)
qwen3.load_state_dict(sd, strict=True)      # strict=True：键名对不上就当场红

g = gpt2.state_dict()
assert torch.equal(qwen3.wte.weight, g["tok_emb.weight"]), "wte ← tok_emb 映射失败"
assert torch.equal(qwen3.wpe.weight, gpt2.pos_emb), "wpe ← pos_emb 映射失败"
C = cfg.n_embd
assert torch.equal(qwen3.blocks[0].attn.q_proj.weight, g["blocks.0.attn.c_attn.weight"][0:C]), \
    "q_proj ← c_attn[0:C] 映射失败"
assert torch.equal(qwen3.blocks[0].attn.k_proj.weight, g["blocks.0.attn.c_attn.weight"][C:2 * C]), \
    "k_proj ← c_attn[C:2C] 映射失败"
assert torch.equal(qwen3.blocks[0].attn.v_proj.weight, g["blocks.0.attn.c_attn.weight"][2 * C:3 * C]), \
    "v_proj ← c_attn[2C:3C] 映射失败"
assert torch.equal(qwen3.blocks[0].attn.o_proj.weight, g["blocks.0.attn.c_proj.weight"]), \
    "o_proj ← c_proj 映射失败"
assert torch.equal(qwen3.blocks[0].mlp.up_proj.weight, g["blocks.0.mlp.c_fc.weight"]), \
    "mlp.up_proj ← c_fc 映射失败"
assert torch.equal(qwen3.blocks[0].mlp.down_proj.weight, g["blocks.0.mlp.c_proj.weight"]), \
    "mlp.down_proj ← c_proj 映射失败"
assert torch.equal(qwen3.ln_f.weight, g["ln_f.weight"]), "ln_f.weight 映射失败"
assert torch.equal(qwen3.ln_f.bias, g["ln_f.bias"]), "ln_f.bias 映射失败"
print("[compare] 反证 B：权重映射表逐条核对通过（wte / wpe / q,k,v,o / up,down / ln_f）")

# =====================================================================
# 4) 消融开关**默认关闭** ⇒ 主线行为一字未改
#    （这是「不污染主线」的硬保证：默认配置下连模块都不该多出来）
# =====================================================================
assert DENSE_CONFIG.use_gqa is True
assert DENSE_CONFIG.use_rope is True
assert DENSE_CONFIG.use_qk_norm is True
assert DENSE_CONFIG.mlp_type == "swiglu"
assert DENSE_CONFIG.norm_type == "rmsnorm"

main_model = Qwen3Dense(vocab_size=VOCAB, cfg=DENSE_CONFIG, dropout=0.0)
assert main_model.wpe is None, "主线配置不应存在 wpe（位置信息由 RoPE 承担）"
assert hasattr(main_model.blocks[0].attn, "q_norm"), "主线配置应有 QK-Norm"
assert hasattr(main_model.blocks[0], "mlp") and main_model.blocks[0].mlp.__class__.__name__ == "SwiGLUMLP", \
    "主线配置的 FFN 应是 SwiGLUMLP"
assert main_model.ln_f.__class__.__name__ == "RMSNorm", "主线配置的 final norm 应是 RMSNorm"

# 退化配置：模块确实换掉了
deg = Qwen3Dense(vocab_size=VOCAB, cfg=degraded_config(), dropout=0.0)
assert deg.wpe is not None, "退化配置（use_rope=False）应有 wpe"
assert not hasattr(deg.blocks[0].attn, "q_norm"), "退化配置不应有 QK-Norm"
assert deg.blocks[0].mlp.__class__.__name__ == "GELUMLP", "退化配置的 FFN 应是 GELUMLP"
assert deg.ln_f.__class__.__name__ == "LayerNorm", "退化配置的 final norm 应是 LayerNorm"
assert deg.blocks[0].attn.n_head_kv == cfg.n_head, "use_gqa=False 时 n_head_kv 应等于 n_head"
print("[compare] 反证 C：主线默认配置行为不变（无 wpe / 有 QK-Norm / SwiGLU / RMSNorm）")

# =====================================================================
# 5) 退化配置的关键形状必须回退成 GPT-2 的取法（否则对不齐的根因在这）
# =====================================================================
assert cfg.head_dim == cfg.n_embd // cfg.n_head, "head_dim 必须回退成 n_embd // n_head"
assert cfg.intermediate_size == 4 * cfg.n_embd, "中间层必须等于 GPT-2 的 n_inner = 4·n_embd"
assert cfg.rms_eps == 1e-5, "eps 必须取 GPT-2 的 nn.LayerNorm 默认值 1e-5"
assert cfg.attention_bias is False, "bias 必须与 GPT-2 一致（False）"

# =====================================================================
# 6) 消融表的每一行都必须是「只开一个组件」
# =====================================================================
names = [r[0] for r in build_cases()]
assert names[0].startswith("全关")
for r in build_cases()[1:]:
    c = r[1]
    on = [
        c.use_gqa, c.use_rope, c.use_qk_norm,
        c.mlp_type == "swiglu", c.norm_type == "rmsnorm",
    ]
    assert sum(on) == 1, f"「{r[0]}」应当只开一个组件，实得 {on}"

print("test_compare_with_gpt2 OK  全关=0 差异 + 5 个单组件行均显著偏离 + 3 条反证")

#!/usr/bin/env bash
# =============================================================================
# init.sh — llm-learning 标准启动 + 轻量验证路径（L0 + L1 + L2 + L3 + L4）
#
# 设计原则（见 AGENTS.md §5）：
#   - 本文件只做"轻量 smoke"，不跑全量训练（全量训练 CPU 约 10 分钟，按需手动跑）
#   - torch 相关检查允许 SKIP（不阻塞），其余任何 FAIL 都返回非 0 退出码
#   - 每个检查都必须 PASS / SKIP / FAIL 三态明确，禁止"静默通过"
#
# 用法：bash init.sh          （Git Bash / Linux / macOS）
#       bash init.sh --quiet  （只输出汇总行）
#
# 解释器选择（见 feat-001）：
#   - PYTHON_BIN=... bash init.sh   显式指定解释器
#   - 否则自动挑第一个 **能 import torch** 的解释器
#   - 本机 PATH 上的 python 可能是 3.11.0a2（alpha），torch 的 DLL 在该 ABI 下
#     加载失败（WinError 127），因此不能只看 `command -v python`
# =============================================================================

set -u

QUIET=0
[ "${1:-}" = "--quiet" ] && QUIET=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR="$REPO_ROOT/gpt-learning/gpt-learning-pytorch"

PASS=0
SKIP=0
FAIL=0
FAILED_ITEMS=()

if [ "$QUIET" -eq 0 ]; then
  echo "=============================================="
  echo " llm-learning :: 轻量验证 (L0-L4)"
  echo " repo: $REPO_ROOT"
  echo "=============================================="
fi

ok()   { PASS=$((PASS+1)); [ "$QUIET" -eq 0 ] && printf "  [PASS] %s\n" "$1"; return 0; }
skip() { SKIP=$((SKIP+1)); [ "$QUIET" -eq 0 ] && printf "  [SKIP] %s — %s\n" "$1" "${2:-}"; return 0; }
fail() { FAIL=$((FAIL+1)); FAILED_ITEMS+=("$1"); [ "$QUIET" -eq 0 ] && printf "  [FAIL] %s — %s\n" "$1" "${2:-}"; return 0; }
head_() { [ "$QUIET" -eq 0 ] && printf "\n--- %s ---\n" "$1"; return 0; }

# -----------------------------------------------------------------------------
# 解释器探测：优先要一个能 import torch 的
# -----------------------------------------------------------------------------
# 理由（feat-001 的真实病因）：本机 PATH 上的 python 是 3.11.0a2（预发布 alpha），
# torch 2.1.0+cpu 的 torch_python.dll 在该 ABI 下加载失败 → OSError [WinError 127]。
# 这不是「torch 装坏了」，而是「选错了解释器」。所以探测必须以
# `import torch` 能否成功为准，而不是 `command -v python` 能否找到。
PY_FALLBACK=""
detect_python() {
  local cand
  # 1) 显式指定（最高优先级）
  if [ -n "${PYTHON_BIN:-}" ]; then
    if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
      PY_FALLBACK="$PYTHON_BIN"
      if "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
        echo "$PYTHON_BIN"; return 0
      fi
    fi
  fi
  # 2) 候选列表：第一个能 import torch 的胜出
  for cand in python python3 python3.12 python3.11 python3.10 python3.9 \
              /d/python310/python.exe; do
    command -v "$cand" >/dev/null 2>&1 || continue
    [ -z "$PY_FALLBACK" ] && PY_FALLBACK="$cand"
    if "$cand" -c "import torch" >/dev/null 2>&1; then
      echo "$cand"; return 0
    fi
  done
  # 3) 都没有能 import torch 的：退回裸 python（L1 仍可跑，L2 会 SKIP）
  echo "${PY_FALLBACK:-python}"
  return 0
}

PY="$(detect_python)"
HAS_TORCH=0
if "$PY" -c "import torch" >/dev/null 2>&1; then HAS_TORCH=1; fi

# -----------------------------------------------------------------------------
head_ "L0  环境"
# -----------------------------------------------------------------------------
if command -v "$PY" >/dev/null 2>&1; then
  PYV="$("$PY" --version 2>&1)"
  ok "python 可用 ($PYV, 解释器 = $PY)"
else
  fail "python 不可用" "PATH 中找不到 python，先装 Python 3.10+"
fi

if [ "$HAS_TORCH" -eq 1 ]; then
  ok "torch 可导入 ($("$PY" -c 'import torch; print(torch.__version__)'))"
else
  skip "torch 可导入" "候选解释器均无法 import torch，见 feature_list.json feat-001"
fi

if command -v git >/dev/null 2>&1 && git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
  ok "git 仓库就绪"
else
  fail "git 仓库不可用" "llm-learning 根目录不是 git 仓库"
fi

for f in AGENTS.md README.md harness.md feature_list.json claude-progress.md; do
  if [ -f "$REPO_ROOT/$f" ]; then
    ok "harness 文件存在: $f"
  else
    fail "harness 文件缺失: $f" "见 harness.md 第二部分初始化 SOP"
  fi
done

# -----------------------------------------------------------------------------
head_ "L1  纯 Python 层（不依赖 torch：tokenizer + 数据构造）"
# -----------------------------------------------------------------------------
if [ "$FAIL" -gt 0 ] && ! command -v "$PY" >/dev/null 2>&1; then
  skip "L1 数据层" "python 不可用，跳过"
else
  cd "$PY_DIR" || { fail "进入代码目录失败" "$PY_DIR"; }

  if [ -f data/vocab.txt ]; then
    ok "data/vocab.txt 已生成"
  else
    fail "data/vocab.txt 缺失" "跑 python data/build_vocab.py"
  fi

  "$PY" - <<'PY' >/tmp/llm_init_l1.log 2>&1
import sys
from data.tokenizer import CharTokenizer
from data.sft_data import generate_sft_dataset
from data.pretrain_data import generate_pretrain_dataset
from data.preference_data import generate_preference_dataset
from data.benchmark_data import build_benchmark_questions

tk = CharTokenizer("data/vocab.txt")
assert tk.vocab_size >= 40, f"vocab_size 过小: {tk.vocab_size}"

# 往返测试必须用词表内字符：表外字符会映射到 UNK，decode 时被跳过
sample = "".join(tk.itos[5:25])
ids = tk.encode(sample)
assert len(ids) == len(sample), "encode 长度不匹配"
assert tk.decode(ids) == sample, "decode 未还原原文"
assert tk.encode("!@#$%^&*()")[0] is not None, "表外字符应映射到 UNK 而非报错"

sft = generate_sft_dataset(n_samples=20, seed=42)
pre = generate_pretrain_dataset(n_samples=20, seed=42)
bm = build_benchmark_questions(n=5, seed=1234)
print("vocab", tk.vocab_size, "sft", len(sft), "pretrain", len(pre), "bench", len(bm))
PY
  if [ $? -eq 0 ]; then
    ok "数据层 smoke（tokenizer 编解码 + SFT/预训练/benchmark 构造）$(tail -1 /tmp/llm_init_l1.log)"
  else
    fail "数据层 smoke" "$(tail -3 /tmp/llm_init_l1.log | tr '\n' ' ')"
  fi
fi

# -----------------------------------------------------------------------------
head_ "L2  Torch 层（模型前向；torch 不可用时 SKIP 不阻塞）"
# -----------------------------------------------------------------------------
if [ "$HAS_TORCH" -ne 1 ]; then
  skip "torch 模型前向" "本机 torch 无法导入（已知：torch_python.dll 加载失败），见 feature_list.json feat-001"
else
  cd "$PY_DIR" || fail "进入代码目录失败" "$PY_DIR"

  "$PY" - <<'PY' >/tmp/llm_init_l2.log 2>&1
import torch
import config
from data.tokenizer import CharTokenizer
from model.gpt2 import GPT2

tk = CharTokenizer("data/vocab.txt")
model = GPT2(vocab_size=tk.vocab_size, n_layer=config.n_layer, n_head=config.n_head,
             n_embd=config.n_embd, n_inner=config.n_inner, block_size=config.block_size,
             dropout=0.0, bias=config.bias)

idx = torch.randint(0, tk.vocab_size, (2, 8))
out = model(idx)
logits = out[0]
assert logits.shape == (2, 8, tk.vocab_size), f"logits 形状错: {tuple(logits.shape)}"

targets = torch.randint(0, tk.vocab_size, (2, 8))
out2 = model(idx, targets=targets)
loss = out2[1]
assert loss is not None and torch.isfinite(loss), f"loss 异常: {loss}"

n_params = sum(p.numel() for p in model.parameters())
print(f"logits {tuple(logits.shape)} loss {loss.item():.4f} params {n_params}")
PY
  if [ $? -eq 0 ]; then
    ok "模型前向 smoke（logits 形状 + loss 有限 + 参数量）$(tail -1 /tmp/llm_init_l2.log)"
  else
    fail "模型前向 smoke" "$(tail -3 /tmp/llm_init_l2.log | tr '\n' ' ')"
  fi
fi

# -----------------------------------------------------------------------------
head_ "L3  仓库卫生（忽略规则是否生效）"
# -----------------------------------------------------------------------------
TRACKED_JUNK="$(git -C "$REPO_ROOT" ls-files \
  | grep -E '(^|/)(checkpoints|docs|__pycache__|\.claude|\.vscode)/|\.(pt|pth|bin)$' || true)"
if [ -z "$TRACKED_JUNK" ]; then
  ok "无产物类文件被 git 跟踪（checkpoints / docs / 缓存 / 权重）"
else
  fail "产物类文件被 git 跟踪" "$(echo "$TRACKED_JUNK" | head -3 | tr '\n' ' ')"
fi

BIG_TRACKED="$(git -C "$REPO_ROOT" ls-files -z | xargs -0 -I{} du -k "$REPO_ROOT/{}" 2>/dev/null \
  | awk '$1 > 2048 {print $2}' | head -3 || true)"
if [ -z "$BIG_TRACKED" ]; then
  ok "无 >2MB 的大文件被跟踪"
else
  fail "存在 >2MB 的大文件" "$(echo "$BIG_TRACKED" | tr '\n' ' ')"
fi

# -----------------------------------------------------------------------------
head_ "L4  Qwen3 线（目录未创建时 SKIP；随 feature 完成度逐步加重）"
# -----------------------------------------------------------------------------
QWEN_DIR="$REPO_ROOT/qwen3-learning/qwen3-pytorch"

if [ ! -d "$QWEN_DIR" ]; then
  skip "Qwen3 代码目录" "qwen3-learning/qwen3-pytorch/ 尚未创建（feature_list.json feat-011）"
else
  ok "qwen3-learning/qwen3-pytorch/ 存在"

  # --- 骨架检查（feat-011）：目录结构齐全 ---
  for f in config.py README.md model data train tests; do
    if [ -e "$QWEN_DIR/$f" ]; then
      ok "Qwen3 骨架存在: $f"
    else
      fail "Qwen3 骨架缺失: $f" "见 feature_list.json feat-011"
    fi
  done

  # --- 组件 smoke：跑 tests/run_all.py（feat-012 起逐步变重）---
  if [ -f "$QWEN_DIR/tests/run_all.py" ]; then
    if [ "$HAS_TORCH" -ne 1 ]; then
      skip "Qwen3 组件 smoke" "torch 不可用（见 feat-001）"
    else
      (cd "$QWEN_DIR" && "$PY" tests/run_all.py) >/tmp/llm_init_l4.log 2>&1
      if [ $? -eq 0 ]; then
        ok "Qwen3 组件 smoke（tests/run_all.py）$(tail -1 /tmp/llm_init_l4.log)"
      else
        fail "Qwen3 组件 smoke" "$(tail -5 /tmp/llm_init_l4.log | tr '\n' ' ')"
      fi
    fi
  fi

  # --- 端到端前向（feat-016 之后可用；vocab.json 就绪前用假词表占位）---
  if [ -f "$QWEN_DIR/model/qwen3_dense.py" ] && [ "$HAS_TORCH" -eq 1 ]; then
    (cd "$QWEN_DIR" && "$PY" - <<'PY' >/tmp/llm_init_l4_fwd.log 2>&1
import os
import torch
from config import DENSE_CONFIG
from model.qwen3_dense import Qwen3Dense

# 有真词表就用真的（feat-022 之后），没有则用假词表占位，
# 目的只是守住「import / 形状 / loss 有限」这条底线，不阻塞验证。
if os.path.exists("data/vocab.json"):
    from data.tokenizer import BPETokenizer
    tk = BPETokenizer("data/vocab.json", "data/merges.txt")
    vocab_size = tk.vocab_size
    src = "vocab.json"
else:
    vocab_size = 512
    src = "fake-vocab(512)"

model = Qwen3Dense(vocab_size=vocab_size, cfg=DENSE_CONFIG, dropout=0.0)

idx = torch.randint(0, vocab_size, (2, 8))
logits, loss, cache, extras = model(idx)
assert logits.shape == (2, 8, vocab_size), f"logits 形状错: {tuple(logits.shape)}"
assert extras == {}, f"dense 的 extras 应为空 dict，实得 {extras}"

targets = torch.randint(0, vocab_size, (2, 8))
logits, loss, _, _ = model(idx, targets=targets)
assert loss is not None and torch.isfinite(loss), f"loss 异常: {loss}"
print(f"[{src}] logits {tuple(logits.shape)} loss {loss.item():.4f} params {model.num_params():,}")
PY
    )
    if [ $? -eq 0 ]; then
      ok "Qwen3 前向 smoke（logits 形状 + loss 有限 + 参数量）$(tail -1 /tmp/llm_init_l4_fwd.log)"
    else
      fail "Qwen3 前向 smoke" "$(tail -3 /tmp/llm_init_l4_fwd.log | tr '\n' ' ')"
    fi
  else
    skip "Qwen3 前向 smoke" "model/qwen3_dense.py 尚未就绪（feat-016），或 torch 不可用"
  fi

  # --- 训练步 smoke（feat-020）：构造 dense 模型 + 走一步训练，断言 loss 有限 ---
  if [ -f "$QWEN_DIR/train/train_pretrain.py" ] && [ "$HAS_TORCH" -eq 1 ]; then
    (cd "$QWEN_DIR" && PYTHONIOENCODING=utf-8 "$PY" - <<'PY' >/tmp/llm_init_l4_train.log 2>&1
import torch
from config import DENSE_CONFIG
from train.train_pretrain import build_model, lr_at

# 走一步「真训练」：forward → loss → backward → 梯度非空 → optimizer.step
model = build_model("dense", 256, dropout=0.0)
model.train()
x = torch.randint(0, 256, (2, 8))
y = torch.randint(0, 256, (2, 8))

logits, loss, _, extras = model(x, targets=y)
assert torch.isfinite(loss), f"loss 非有限: {loss}"
assert extras == {}, f"dense 的 extras 应为空 dict，实得 {extras}"

optimizer = torch.optim.AdamW(model.parameters(), lr=DENSE_CONFIG.learning_rate)
optimizer.zero_grad(set_to_none=True)
loss.backward()
grads = [p.grad for p in model.parameters() if p.grad is not None]
assert len(grads) > 0, "没有任何参数收到梯度"
assert all(torch.isfinite(g).all() for g in grads), "梯度非有限"
optimizer.step()

# 学习率调度要能算出来（warmup 第一步 > 0，衰减结束等于 min_lr）
lr0 = lr_at(0, DENSE_CONFIG)
lr_end = lr_at(DENSE_CONFIG.lr_decay_iters, DENSE_CONFIG)
assert lr0 > 0, f"warmup 第一步 lr 应为正，实得 {lr0}"
assert abs(lr_end - DENSE_CONFIG.min_lr) < 1e-9, f"衰减结束 lr 应为 min_lr，实得 {lr_end}"
print(f"[train-step] loss {loss.item():.4f} grads {len(grads)} 个 lr0 {lr0:.2e} lr_end {lr_end:.2e}")
PY
    )
    if [ $? -eq 0 ]; then
      ok "Qwen3 训练步 smoke（forward+backward+step，loss 有限）$(tail -1 /tmp/llm_init_l4_train.log)"
    else
      fail "Qwen3 训练步 smoke" "$(tail -3 /tmp/llm_init_l4_train.log | tr '\n' ' ')"
    fi
  else
    skip "Qwen3 训练步 smoke" "train/train_pretrain.py 尚未就绪（feat-020），或 torch 不可用"
  fi

  # --- LoRA smoke（feat-023）：注入无损（B 零初始化）+ 只有 A/B 可训 ---
  if [ -f "$QWEN_DIR/model/common/lora.py" ] && [ "$HAS_TORCH" -eq 1 ]; then
    (cd "$QWEN_DIR" && PYTHONIOENCODING=utf-8 "$PY" - <<'PY' >/tmp/llm_init_l4_lora.log 2>&1
import torch
from config import DENSE_CONFIG
from model.common.lora import inject_lora, mark_only_lora_trainable
from model.qwen3_dense import Qwen3Dense

model = Qwen3Dense(vocab_size=256, cfg=DENSE_CONFIG, dropout=0.0)
model.eval()
idx = torch.randint(0, 256, (2, 8))
with torch.no_grad():
    before = model(idx)[0]

replaced = inject_lora(model, DENSE_CONFIG.lora_target_modules, r=DENSE_CONFIG.lora_r,
                       alpha=DENSE_CONFIG.lora_alpha)
with torch.no_grad():
    after = model(idx)[0]
# B 零初始化 ⇒ 注入 LoRA 后模型输出必须逐元素不变
assert torch.allclose(before, after, atol=1e-6), "B 零初始化时注入 LoRA 应无损"

trainable = mark_only_lora_trainable(model)
n_train = sum(p.numel() for p in trainable)
n_all = sum(p.numel() for p in model.parameters())
assert n_train > 0, "应有可训练的 LoRA 参数"
assert n_train < n_all * 0.1, f"可训练参数应远小于总参数：{n_train}/{n_all}"
print(f"[lora] 注入 {len(replaced)} 个模块，输出无损；可训练 {n_train:,}/{n_all:,} "
      f"({n_train/n_all:.2%})")
PY
    )
    if [ $? -eq 0 ]; then
      ok "Qwen3 LoRA smoke（注入无损 + 只训 A/B）$(tail -1 /tmp/llm_init_l4_lora.log)"
    else
      fail "Qwen3 LoRA smoke" "$(tail -3 /tmp/llm_init_l4_lora.log | tr '\n' ' ')"
    fi
  else
    skip "Qwen3 LoRA smoke" "model/common/lora.py 尚未就绪（feat-023），或 torch 不可用"
  fi

  # --- DPO smoke（feat-024）：初始 margin=0（π_θ == π_ref）+ loss=log2 + 只训 LoRA ---
  if [ -f "$QWEN_DIR/train/train_dpo.py" ] && [ "$HAS_TORCH" -eq 1 ]; then
    (cd "$QWEN_DIR" && PYTHONIOENCODING=utf-8 "$PY" - <<'PY' >/tmp/llm_init_l4_dpo.log 2>&1
import torch
from config import DENSE_CONFIG
from data.preference_data import generate_preference_dataset
from data.tokenizer import BPETokenizer
from model.common.lora import inject_lora, mark_only_lora_trainable
from model.qwen3_dense import Qwen3Dense
from train.train_dpo import dpo_loss

tk = BPETokenizer("data/vocab.json", "data/merges.txt")
pairs = generate_preference_dataset(n_samples=4, seed=DENSE_CONFIG.seed)

model = Qwen3Dense(vocab_size=tk.vocab_size, cfg=DENSE_CONFIG, dropout=0.0)
model.eval()
inject_lora(model, DENSE_CONFIG.lora_target_modules, r=DENSE_CONFIG.lora_r,
            alpha=DENSE_CONFIG.lora_alpha)
trainable = mark_only_lora_trainable(model)

loss, margin, _ = dpo_loss(model, pairs[0], tk, DENSE_CONFIG.dpo_beta, torch.device("cpu"))
# B 零初始化 ⇒ π_θ 严格等于 π_ref ⇒ margin 必为 0、loss 必为 log 2
assert abs(float(margin.item())) < 1e-4, f"初始 margin 应为 0，实得 {float(margin.item())}"
assert abs(float(loss.item()) - 0.6931472) < 1e-4, f"初始 loss 应为 log 2，实得 {float(loss.item())}"
assert torch.isfinite(loss), "loss 非有限"

n_train = sum(p.numel() for p in trainable)
n_all = sum(p.numel() for p in model.parameters())
assert n_train < n_all * 0.1, f"可训练参数应远小于总参数：{n_train}/{n_all}"
print(f"[dpo] 初始 margin={float(margin.item()):+.6f} loss={float(loss.item()):.4f}(=log2)；"
      f"可训练 {n_train:,}/{n_all:,} ({n_train/n_all:.2%})")
PY
    )
    if [ $? -eq 0 ]; then
      ok "Qwen3 DPO smoke（初始 margin=0 + loss=log2）$(tail -1 /tmp/llm_init_l4_dpo.log)"
    else
      fail "Qwen3 DPO smoke" "$(tail -3 /tmp/llm_init_l4_dpo.log | tr '\n' ' ')"
    fi
  else
    skip "Qwen3 DPO smoke" "train/train_dpo.py 尚未就绪（feat-024），或 torch 不可用"
  fi

  # --- 推理 smoke（feat-018）：有/无 KV Cache 的 greedy 输出必须逐位相同 ---
  if [ -f "$QWEN_DIR/inference.py" ] && [ "$HAS_TORCH" -eq 1 ]; then
    (cd "$QWEN_DIR" && PYTHONIOENCODING=utf-8 "$PY" - <<'PY' >/tmp/llm_init_l4_inference.log 2>&1
import torch
from config import DENSE_CONFIG
from inference import generate_naive, generate_with_cache, kv_cache_report
from model.qwen3_dense import Qwen3Dense

torch.manual_seed(DENSE_CONFIG.seed)
model = Qwen3Dense(vocab_size=64, cfg=DENSE_CONFIG, dropout=0.0)
model.eval()
prompt = torch.randint(0, 64, (1, 8), dtype=torch.long)

a = generate_naive(model, prompt, max_new_tokens=8, temperature=0.0)
b = generate_with_cache(model, prompt, max_new_tokens=8, temperature=0.0)
assert torch.equal(a, b), f"有/无 KV Cache 的 greedy 输出应逐位相同\n{a.tolist()}\n{b.tolist()}"

rep = kv_cache_report(model, prompt_len=8, max_new_tokens=56)
assert rep["ratio"] == DENSE_CONFIG.n_head_kv / DENSE_CONFIG.n_head, \
    f"GQA/MHA 占用比不对：{rep['ratio']}"
print(f"[infer] greedy 逐位一致（8 token）；KV 占用 GQA={rep['gqa_bytes']:,}B "
      f"MHA={rep['mha_bytes']:,}B（1/{DENSE_CONFIG.n_head // DENSE_CONFIG.n_head_kv}）")
PY
    )
    if [ $? -eq 0 ]; then
      ok "Qwen3 推理 smoke（有/无 KV Cache 输出一致）$(tail -1 /tmp/llm_init_l4_inference.log)"
    else
      fail "Qwen3 推理 smoke" "$(tail -3 /tmp/llm_init_l4_inference.log | tr '\n' ' ')"
    fi
  else
    skip "Qwen3 推理 smoke" "inference.py 尚未就绪（feat-018），或 torch 不可用"
  fi

  # --- 评估 smoke（feat-025）：5 段结构 + pretrain/speedup 的「GPT-2 同名列」---
  if [ -f "$QWEN_DIR/evaluate.py" ] && [ "$HAS_TORCH" -eq 1 ]; then
    (cd "$QWEN_DIR" && PYTHONIOENCODING=utf-8 "$PY" - <<'PY' >/tmp/llm_init_l4_eval.log 2>&1
import math
import os
import tempfile

import torch

import evaluate
from config import DENSE_CONFIG
from data.tokenizer import BPETokenizer
from model.qwen3_dense import Qwen3Dense

# 1) 5 段齐全
assert set(evaluate.STAGES.keys()) == {"pretrain", "moe", "lora", "dpo", "speedup"}, \
    f"stage 集合不对：{sorted(evaluate.STAGES.keys())}"

tk = BPETokenizer("data/vocab.json", "data/merges.txt")
tmpdir = tempfile.mkdtemp(prefix="qwen3_eval_")
ckpt = os.path.join(tmpdir, "fake_pretrain.pt")
torch.save({"model_state": Qwen3Dense(vocab_size=tk.vocab_size, cfg=DENSE_CONFIG,
                                      dropout=0.0).state_dict()}, ckpt)

# 2) pretrain / speedup 必须保留与 GPT-2 线逐字同名的列
GPT2_PRETRAIN = {"stage", "metric", "value", "loss", "perplexity", "n_tokens", "checkpoint"}
GPT2_SPEEDUP = {"stage", "metric", "value", "naive_avg_ms", "kv_avg_ms", "speedup",
                "naive_tokens", "kv_tokens", "n_runs", "max_new_tokens"}

orig = evaluate.PRETRAIN_DENSE
evaluate.PRETRAIN_DENSE = ckpt
try:
    r = evaluate.eval_pretrain(tk, torch.device("cpu"))
    missing = GPT2_PRETRAIN - set(r.keys())
    assert not missing, f"pretrain 缺少 GPT-2 同名列：{sorted(missing)}"
    assert abs(r["perplexity"] - math.exp(min(20.0, r["loss"]))) < 1e-9, "ppl 与 loss 不自洽"

    s = evaluate.eval_speedup(tk, torch.device("cpu"))
    missing = GPT2_SPEEDUP - set(s.keys())
    assert not missing, f"speedup 缺少 GPT-2 同名列：{sorted(missing)}"
    assert s["consistent"] is True, "有/无 KV Cache 的输出必须一致"
finally:
    evaluate.PRETRAIN_DENSE = orig

print(f"[eval] 5 段就绪；pretrain/speedup 的 GPT-2 同名列齐全；"
      f"ppl={r['perplexity']:.2f} speedup={s['speedup']:.2f}x consistent={s['consistent']}")
PY
    )
    if [ $? -eq 0 ]; then
      ok "Qwen3 评估 smoke（5 段 + GPT-2 同名列）$(tail -1 /tmp/llm_init_l4_eval.log)"
    else
      fail "Qwen3 评估 smoke" "$(tail -3 /tmp/llm_init_l4_eval.log | tr '\n' ' ')"
    fi
  else
    skip "Qwen3 评估 smoke" "evaluate.py 尚未就绪（feat-025），或 torch 不可用"
  fi

  # --- 退化对照 smoke（feat-017）：全关时 Qwen3 必须与 GPT-2 对齐 ---
  if [ -f "$QWEN_DIR/ablations/compare_with_gpt2.py" ] && [ "$HAS_TORCH" -eq 1 ]; then
    (cd "$QWEN_DIR" && PYTHONIOENCODING=utf-8 "$PY" - <<'PY' >/tmp/llm_init_l4_compare.log 2>&1
import os
import sys

sys.path.insert(0, os.getcwd())

from ablations.compare_with_gpt2 import run_all

r = run_all(threshold=1e-4, seq_len=16, batch=2, vocab_size=293)
base = r["cases"][0]
assert base["max_abs_diff"] < 1e-4, \
    f"全关（退化成 GPT-2）时 max_abs_diff 应 < 1e-4，实得 {base['max_abs_diff']:.3e}"
assert r["component_rows_passed"], "单开组件的行必须显著偏离基线，否则消融表没有说服力"
parts = ", ".join(f"{c['name'].replace('只开 ', '')}={c['max_abs_diff']:.1e}"
                  for c in r["cases"][1:])
print(f"[compare] 全关 diff={base['max_abs_diff']:.3e}（阈值 1e-4）；单开组件：{parts}")
PY
    )
    if [ $? -eq 0 ]; then
      ok "Qwen3 退化对照 smoke（全关 = GPT-2，max_abs_diff < 1e-4）$(tail -1 /tmp/llm_init_l4_compare.log)"
    else
      fail "Qwen3 退化对照 smoke" "$(tail -3 /tmp/llm_init_l4_compare.log | tr '\n' ' ')"
    fi
  else
    skip "Qwen3 退化对照 smoke" "ablations/compare_with_gpt2.py 尚未就绪（feat-017），或 torch 不可用"
  fi
fi

# -----------------------------------------------------------------------------
head_ "汇总"
# -----------------------------------------------------------------------------
echo "  PASS=$PASS  SKIP=$SKIP  FAIL=$FAIL"
if [ "$FAIL" -gt 0 ]; then
  echo ""
  echo "  失败项："
  for item in "${FAILED_ITEMS[@]}"; do echo "    - $item"; done
  echo ""
  echo "  下一步：先修 baseline，不要在此之上开新 feature（见 AGENTS.md §1）。"
  exit 1
fi

echo ""
echo "  ✅ baseline 健康。下一步："
echo "    1. 读 feature_list.json 的 phase 字段确认当前阶段"
echo "    2. 挑 priority 最小的非 passing feature，一次只做一个（WIP=1）"
echo "    3. L2/L4 被 SKIP 时先修环境（feat-001）；动训练代码再跑全量（L3）"
exit 0

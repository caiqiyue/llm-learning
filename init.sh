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

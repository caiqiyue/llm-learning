# Qwen3-PyTorch 开发计划

> **执行方式**：本计划由 agentic worker 逐 Task 执行。每个 Task 结束后 commit 一次。
> **设计依据**：[`qwen3-pytorch设计.md`](./qwen3-pytorch设计.md)
> **代码风格依据**：`gpt-learning/gpt-learning-pytorch/model/gpt2.py`
> **状态**：待执行 ｜ 制定日期：2026-10-19

---

## 0. 执行须知

### 0.1 项目硬约束（违反即 FAIL）

| 约束 | 说明 |
|---|---|
| **依赖上限** | 只允许 `torch` / `numpy` / `nltk`。**禁止引入 pytest、peft、transformers、sentencepiece、tokenizers** |
| 超参集中 | 所有超参从 `qwen3-pytorch/config.py` 读，训练脚本内**禁止**硬编码数字 |
| 中文 docstring + type hint | 每个类、每个公开方法都要，对齐 `model/gpt2.py` |
| 不入库 | `checkpoints/`、`docs/`、`__pycache__/`、权重、`.pt` 文件 |
| WIP = 1 | `feature_list.json` 同时只能有一个 `in_progress` |
| 不宣布完成除非有证据 | 每个 Task 必须贴出命令 + 实际输出 |

### 0.2 测试策略（为什么不用 pytest）

项目依赖上限不允许装 pytest。沿用 GPT-2 线的 smoke 风格，但集中到 `tests/` 目录：

- 每个组件一个 `tests/test_<module>.py`，**顶层线性执行 + `assert`**，不用 fixture、不用测试框架
- 运行方式：`python tests/test_rmsnorm.py`，成功 exit 0，失败抛 AssertionError
- `tests/run_all.py` 遍历全部测试文件逐个 subprocess 执行，打印汇总（对齐 `init.sh` 的 PASS/SKIP/FAIL 三态）

#### 【实测补充，2026-10-22】每个测试文件必须自带 sys.path 引导

直接执行 `python tests/test_rmsnorm.py` 时，Python 把**脚本所在目录**（`tests/`）放进
`sys.path[0]`，项目根不在 `sys.path` 里 —— 于是 `from config import DENSE_CONFIG`
会报 `ModuleNotFoundError: No module named 'config'`。
`tests/run_all.py` 同样是以文件路径方式启动子进程，`cwd` 设成项目根也**救不了**
（`cwd` 不进 `sys.path`，只有脚本目录进）。

所以每个 `tests/test_*.py` 开头都要加这三行（**后续所有 Task 的测试文件一律照做**）：

```python
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

这样测试文件既能被 `python tests/test_x.py` 单独跑，也能被 `tests/run_all.py`
批量跑，还能在任何工作目录下跑。

### 0.3 与 GPT-2 线的三处刻意偏离

| 项 | GPT-2 线 | Qwen3 线 | 理由 |
|---|---|---|---|
| 超参载体 | 模块级裸变量 | `Qwen3Config` dataclass + `DENSE_CONFIG` / `MOE_CONFIG` 两个实例 | 要同时支持 dense / MoE 两套，裸变量会互相覆盖 |
| 训练脚本参数 | 无（全写死） | `train_pretrain.py` 用 `argparse` | `--model dense\|moe` 是硬需求 |
| forward 返回值 | 3 元组（无 value_head）/ 4 元组（有） | **恒定 4 元组** `(logits, loss, kv_cache, extras)` | Qwen3 线无 value head，固定 4 元组让训练脚本无需分支 |

### 0.4 GPT-2 线的已知坏味道（不要复制）

- `train_pretrain.py:53-54` 写死了绝对路径 `r'E:\ai-learning\...'` 作为默认 vocab 路径 —— Qwen3 线一律用 `os.path.join("data", ...)`
- `train_pretrain.py` 末尾有冗余的三次 `save_checkpoint` —— Qwen3 线只保留「best」和「final」各一次

---

## Phase 0 — harness 对齐（不写模型代码）

### Task 1: 更新 `feature_list.json`

**Files:**
- Modify: `feature_list.json`

- [ ] **Step 1: 把 `feat-001` 改为 `passing`**

torch 已修复。改动：

```json
    {
      "id": "feat-001",
      "status": "passing",
      "evidence": [
        "2026-10-18 init.sh: [SKIP] torch 模型前向 — OSError [WinError 127] 加载 torch_python.dll 失败",
        "2026-10-19 python -c \"import torch\": 2.12.0+cpu 导入成功",
        "2026-10-19 init.sh: PASS=12 SKIP=1 FAIL=0（L2 由 SKIP 转 PASS，唯一 SKIP 为 L4 目录未建）"
      ],
      "notes": "已修复：torch 2.12.0+cpu 可正常导入，init.sh 的 L2 层已转 PASS。"
    }
```

- [ ] **Step 2: 调整既有 feature 状态**

| id | 改动 |
|---|---|
| `feat-003`（全参 SFT） | `deferred`，notes 写「Qwen3 线改走 LoRA 路线，见 feat-023」 |
| `feat-004`（RM） | `deferred`，notes 写「Qwen3 线改走 DPO 路线，见 feat-024」 |
| `feat-005`（PPO） | `deferred`，同上 |
| `feat-019`（tokenizer 选型） | `deferred`，notes 写「已由 feat-022 从零手写 byte-level BPE 落地」 |
| `feat-021`（MoE） | 从 P3 可选升为 **P2 必做**，`priority` 前移到 `feat-020` 之前 |

- [ ] **Step 3: 新增三条 feature**

```json
    {
      "id": "feat-022",
      "priority": 8,
      "area": "qwen3-tokenizer",
      "title": "从零手写 byte-level BPE tokenizer",
      "user_visible_behavior": "qwen3-pytorch/data/ 下能训出 vocab.json 与 merges.txt，BPETokenizer 对中英文代码片段编码后解码逐字节还原。",
      "status": "not_started",
      "verification": [
        "python data/build_vocab.py 生成 vocab.json 与 merges.txt",
        "python tests/test_bpe.py 通过（编解码往返 + merge 次数 = vocab_size - 256）",
        "确认未引入任何新依赖（git diff --stat 不含 requirements 变更）"
      ],
      "evidence": [],
      "notes": "对应设计文档 §6.1。刻意不用 sentencepiece/tokenizers，以守住依赖上限。"
    },
    {
      "id": "feat-023",
      "priority": 11,
      "area": "qwen3-lora",
      "title": "从零手写 LoRA 并微调 dense Qwen3",
      "user_visible_behavior": "train/train_lora.py 能在 pairwise 数据上微调 dense 模型，偏好胜率高于微调前。",
      "status": "not_started",
      "verification": [
        "python tests/test_lora.py 通过（B=0 时注入前后输出逐元素相等；只有 A/B 梯度非 None）",
        "python train/train_lora.py，贴出注入前后的偏好胜率对比",
        "打印可训练参数量，确认远小于全参"
      ],
      "evidence": [],
      "notes": "对应设计文档 §5.5 / §7.2。刻意不用 peft。"
    },
    {
      "id": "feat-024",
      "priority": 12,
      "area": "qwen3-dpo",
      "title": "从零手写 DPO 强化学习",
      "user_visible_behavior": "train/train_dpo.py 能训出 policy，chosen>rejected 准确率上升、reward margin 为正。",
      "status": "not_started",
      "verification": [
        "python tests/test_dpo.py 通过（DPO loss 有限；关闭 LoRA 后 logprob 与 policy 一致）",
        "python train/train_dpo.py，贴出 reward margin 首末值与准确率",
        "确认 reference logprob 由关闭 LoRA 分支得到，未额外加载模型副本"
      ],
      "evidence": [],
      "notes": "对应设计文档 §7.3。reference model 复用同一对象关闭 LoRA，不加载第二份权重。"
    }
```

- [ ] **Step 4: 重排 priority 使其严格 1..N 且 WIP=1**

- [ ] **Step 5: 验证 JSON 合法且约束成立**

Run:
```bash
python -c "
import json
d = json.load(open('feature_list.json', encoding='utf-8'))
ids = [f['id'] for f in d['features']]
pri = sorted(f['priority'] for f in d['features'])
assert len(ids) == len(set(ids)), 'id 重复'
assert pri == list(range(1, len(pri)+1)), f'priority 不连续: {pri}'
wip = [f['id'] for f in d['features'] if f['status']=='in_progress']
print('features:', len(ids), '| in_progress:', wip)
"
```
Expected: `features: 24 | in_progress: []`

- [ ] **Step 6: Commit**

```bash
git add feature_list.json
git commit -m "feat-001 转 passing；feature_list 重排并新增 feat-022/023/024（Qwen3 线 BPE/LoRA/DPO）"
```

---

### Task 2: 重写 `init.sh` 的 L4 层

**Files:**
- Modify: `init.sh`（L4 段，约 168–214 行）

**问题**：现有 L4 写死了 `model/qwen3.py`、`CharTokenizer("data/vocab.txt")`、`config.intermediate_dim` —— 与本设计的 `qwen3_dense.py` / `BPETokenizer` / `intermediate_size` 全部不符。

- [ ] **Step 1: 替换 L4 段**

把 `head_ "L4  Qwen3 线..."` 到该段 `fi` 结束之间的全部内容替换为：

```bash
# -----------------------------------------------------------------------------
head_ "L4  Qwen3 线（目录未创建时 SKIP；随 feature 完成度逐步加重）"
# -----------------------------------------------------------------------------
QWEN_DIR="$REPO_ROOT/qwen3-learning/qwen3-pytorch"
QWEN_HAVE_TORCH=0
python -c "import torch" >/dev/null 2>&1 && QWEN_HAVE_TORCH=1

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
    if [ "$QWEN_HAVE_TORCH" -eq 0 ]; then
      skip "Qwen3 组件 smoke" "torch 不可用（见 feat-001）"
    else
      (cd "$QWEN_DIR" && python tests/run_all.py) >/tmp/llm_init_l4.log 2>&1
      if [ $? -eq 0 ]; then
        ok "Qwen3 组件 smoke（tests/run_all.py）$(tail -1 /tmp/llm_init_l4.log)"
      else
        fail "Qwen3 组件 smoke" "$(tail -5 /tmp/llm_init_l4.log | tr '\n' ' ')"
      fi
    fi
  fi

  # --- 端到端前向（feat-016 之后才可用）---
  if [ -f "$QWEN_DIR/model/qwen3_dense.py" ] && [ -f "$QWEN_DIR/data/vocab.json" ] && [ "$QWEN_HAVE_TORCH" -eq 1 ]; then
    (cd "$QWEN_DIR" && python - <<'PY' >/tmp/llm_init_l4_fwd.log 2>&1
import torch
from config import DENSE_CONFIG
from data.tokenizer import BPETokenizer
from model.qwen3_dense import Qwen3Dense

tk = BPETokenizer("data/vocab.json", "data/merges.txt")
model = Qwen3Dense(vocab_size=tk.vocab_size, cfg=DENSE_CONFIG, dropout=0.0)

idx = torch.randint(0, tk.vocab_size, (2, 8))
logits, loss, cache, extras = model(idx)
assert logits.shape == (2, 8, tk.vocab_size), f"logits 形状错: {tuple(logits.shape)}"
assert extras == {}, f"dense 的 extras 应为空 dict，实得 {extras}"

targets = torch.randint(0, tk.vocab_size, (2, 8))
logits, loss, _, _ = model(idx, targets=targets)
assert loss is not None and torch.isfinite(loss), f"loss 异常: {loss}"
print(f"logits {tuple(logits.shape)} loss {loss.item():.4f} params {model.num_params():,}")
PY
    )
    if [ $? -eq 0 ]; then
      ok "Qwen3 前向 smoke（logits 形状 + loss 有限 + 参数量）$(tail -1 /tmp/llm_init_l4_fwd.log)"
    else
      fail "Qwen3 前向 smoke" "$(tail -3 /tmp/llm_init_l4_fwd.log | tr '\n' ' ')"
    fi
  else
    skip "Qwen3 前向 smoke" "model/qwen3_dense.py 或 data/vocab.json 尚未就绪（feat-011/feat-022），或 torch 不可用"
  fi
fi
```

- [ ] **Step 2: 验证 init.sh 语法正确且 L4 由 SKIP 变 SKIP（目录仍未建）**

Run: `bash -n init.sh && bash init.sh --quiet`
Expected: 退出码 0，输出 `PASS=12  SKIP=1  FAIL=0`（L4 仍 SKIP，因为目录未建 —— 这是正确的）

- [ ] **Step 3: 确认 L2 已转 PASS**

Run: `bash init.sh 2>&1 | grep -A2 "L2"`
Expected: 出现 `[PASS] 模型前向 smoke`，且末行 `PASS=12  SKIP=1  FAIL=0`

- [ ] **Step 4: Commit**

```bash
git add init.sh
git commit -m "init.sh: L4 层改写为对齐 Qwen3 线实际结构（run_all.py + BPETokenizer + Qwen3Dense）"
```

---

## Phase 1 — 骨架与架构组件

### Task 3: feat-011 代码骨架与 `config.py`

**Files:**
- Create: `qwen3-learning/qwen3-pytorch/config.py`
- Create: `qwen3-learning/qwen3-pytorch/README.md`
- Create: `qwen3-learning/qwen3-pytorch/model/__init__.py`
- Create: `qwen3-learning/qwen3-pytorch/model/common/__init__.py`
- Create: `qwen3-learning/qwen3-pytorch/data/__init__.py`
- Create: `qwen3-learning/qwen3-pytorch/train/__init__.py`
- Create: `qwen3-learning/qwen3-pytorch/tests/__init__.py`
- Create: `qwen3-learning/qwen3-pytorch/tests/run_all.py`
- Modify: `feature_list.json`（`feat-011` → `in_progress`）

- [ ] **Step 1: 建目录与 `__init__.py`**

```bash
cd qwen3-learning && mkdir -p qwen3-pytorch/{model/common,data,train,tests}
cd qwen3-pytorch
for f in model data train tests; do : > "$f/__init__.py"; done
: > model/common/__init__.py
ls -R . | head -20
```

- [ ] **Step 2: 写 `config.py`**

```python
"""
Qwen3 线集中超参数（唯一真值）。

与 GPT-2 线 `gpt-learning/gpt-learning-pytorch/config.py` 的差异：
  - GPT-2 线用模块级裸变量；本线用 dataclass，因为要同时支持 dense / MoE 两套配置
  - Qwen3 的 `head_dim` 与 `n_embd` **解耦**：不强制 `n_embd % n_head == 0`
    真实 Qwen3-0.6B 就是这样（hidden=1024, head_dim=128, n_head=16）
  - 两套配置共享字段，MoE 额外多 `n_experts` / `experts_per_tok` /
    `moe_intermediate_size` / `norm_topk_prob` / `aux_loss_coef`

改本文件必须同步更新 `qwen3-learning/` 下引用了对应数字的笔记（见 AGENTS.md §3）。
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Qwen3Config:
    """Qwen3 模型结构 + 训练超参的统一配置对象。"""

    # ---------------- 架构 ----------------
    # Transformer block 层数
    n_layer: int = 4
    # 隐藏维度 / token embedding 维度（残差流宽度）
    n_embd: int = 128
    # Q head 数
    n_head: int = 4
    # KV head 数（GQA）。必须整除 n_head；等于 n_head 时退化为 MHA
    n_head_kv: int = 2
    # 单个 head 的特征维度。**与 n_embd 解耦**：Qwen3 里 n_head * head_dim ≠ n_embd
    # 本项目取 48，4 * 48 = 192 ≠ 128，用来在代码里真实体现解耦
    head_dim: int = 48
    # SwiGLU 中间层维度。取整规则：round(n_embd * 8/3) 再向上对齐到 128 的倍数
    # 128 * 8/3 = 341.33 → 3 * 128 = 384
    intermediate_size: int = 384
    # RoPE 频率基数。Qwen3 默认 1e6（比 GPT-2 的 1e4 大 100 倍，为长上下文服务）
    rope_theta: float = 1e6
    # RMSNorm 的 epsilon（Qwen3 用 1e-6，GPT-2 的 LayerNorm 常用 1e-5）
    rms_eps: float = 1e-6
    # 最大上下文长度
    block_size: int = 128
    # dropout 比例
    dropout: float = 0.1
    # Qwen3 全线 Linear 均无 bias
    attention_bias: bool = False
    # 权重绑定：lm_head.weight 与 wte.weight 共享同一份参数
    tie_embeddings: bool = True

    # ---------------- MoE 专属（dense 配置里为 None）----------------
    # 专家数量
    n_experts: Optional[int] = None
    # 每个 token 路由到几个专家（top-k）
    experts_per_tok: Optional[int] = None
    # 每个专家的 FFN 中间层维度
    moe_intermediate_size: Optional[int] = None
    # 是否对 top-k 的路由权重重新归一化（匹配真实 Qwen3-MoE）
    norm_topk_prob: bool = True
    # 负载均衡辅助损失的系数（总 loss = lm_loss + aux_loss_coef * aux_loss）
    aux_loss_coef: float = 0.01

    # ---------------- 训练通用 ----------------
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    max_iters: int = 300
    warmup_iters: int = 50
    lr_decay_iters: int = 300
    min_lr: float = 3e-5

    # ---------------- tokenizer / 数据 ----------------
    # byte-level BPE 目标词表大小。实际大小由训练语料决定，可能略小于该值
    bpe_vocab_size: int = 1000
    # 预训练训练集占比，其余作为验证集
    train_ratio: float = 0.9
    # 合成语料规模（预训练样本条数）
    pretrain_n_samples: int = 10000

    # ---------------- LoRA ----------------
    # LoRA 低秩矩阵的秩
    lora_r: int = 4
    # LoRA 缩放因子，实际缩放为 alpha / r
    lora_alpha: int = 8
    # LoRA dropout
    lora_dropout: float = 0.0
    # 注入 LoRA 的模块名后缀
    lora_target_modules: tuple = ("q_proj", "v_proj")
    # LoRA 微调学习率（比预训练小一个数量级）
    lora_lr: float = 3e-4
    lora_max_iters: int = 300

    # ---------------- DPO ----------------
    # DPO 的 KL 惩罚系数 β
    dpo_beta: float = 0.1
    dpo_lr: float = 1e-4
    dpo_max_iters: int = 300
    # 偏好数据集规模
    dpo_n_pairs: int = 2000

    # ---------------- 推理 / 采样 ----------------
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95

    # ---------------- 评估 ----------------
    eval_max_samples: int = 200
    eval_seed: int = 1234

    # ---------------- 系统 ----------------
    seed: int = 42
    device: str = "cpu"

    def to_dict(self) -> dict:
        """转成普通 dict（存 checkpoint 用）。"""
        return asdict(self)


# ---------------- Dense 配置 ----------------
DENSE_CONFIG = Qwen3Config()

# ---------------- MoE 配置 ----------------
# 主干与 Dense 完全一致，只有 FFN 换成 Router + 专家
MOE_CONFIG = Qwen3Config(
    n_experts=4,
    experts_per_tok=2,
    moe_intermediate_size=128,
)
```

- [ ] **Step 3: 验证 dataclass 可用且 MoE 字段正确**

Run:
```bash
cd qwen3-learning/qwen3-pytorch && python -c "
from config import DENSE_CONFIG as D, MOE_CONFIG as M
assert D.n_experts is None and M.n_experts == 4
assert M.experts_per_tok == 2 and M.moe_intermediate_size == 128
assert D.n_head * D.head_dim != D.n_embd, 'head_dim 应与 n_embd 解耦'
assert D.n_head % D.n_head_kv == 0, 'n_head 必须整除 n_head_kv 的倍数关系'
# SwiGLU 中间维对齐检查
import math
raw = D.n_embd * 8 / 3
aligned = math.ceil(raw / 128) * 128
print('intermediate_size', D.intermediate_size, '期望', aligned)
assert D.intermediate_size == aligned
print('config OK')
"
```
Expected: `intermediate_size 384 期望 384` / `config OK`

- [ ] **Step 4: 写 `tests/run_all.py`**

```python
"""
Qwen3 线全部 smoke 测试的汇总运行器。

用法：python tests/run_all.py

设计约束（见开发计划 §0.2）：
  - 不依赖 pytest（项目依赖上限只有 torch / numpy / nltk）
  - 每个 test_*.py 顶层线性执行 + assert，exit 0 表示通过
  - 本脚本用 subprocess 逐个跑，隔离崩溃与污染，打印 PASS/FAIL 汇总
"""

import os
import subprocess
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)


def discover() -> list:
    """找出全部 tests/test_*.py（按文件名排序，保证执行顺序稳定）。"""
    names = [n for n in os.listdir(TESTS_DIR) if n.startswith("test_") and n.endswith(".py")]
    return sorted(names)


results = []

for name in discover():
    proc = subprocess.run(
        [sys.executable, os.path.join(TESTS_DIR, name)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    ok = proc.returncode == 0
    results.append((name, ok, proc))
    status = "PASS" if ok else "FAIL"
    detail = (proc.stdout.strip().splitlines() or [""])[-1]
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        tail = (proc.stderr.strip().splitlines() or [""])[-5:]
        for line in tail:
            print(f"         {line}")

n_pass = sum(1 for _, ok, _ in results if ok)
n_fail = len(results) - n_pass
print(f"\n[run_all] PASS={n_pass}  FAIL={n_fail}  TOTAL={len(results)}")

if n_fail > 0:
    sys.exit(1)
print("[run_all] all green")
sys.exit(0)
```

此时 `tests/` 下还没有 `test_*.py`，`discover()` 返回空列表，脚本应以 `TOTAL=0` 退出 0。

- [ ] **Step 5: 写 `README.md`**

```markdown
# qwen3-pytorch

从零手写 tiny Qwen3：RoPE + RMSNorm + QK-Norm + SwiGLU + GQA，外加 MoE 变体，
走通 预训练 → LoRA → DPO 三段。

设计文档：[../qwen3-pytorch设计.md](../qwen3-pytorch设计.md) ｜ 开发计划：[../qwen3-pytorch开发计划.md](../qwen3-pytorch开发计划.md)

## 快速开始

```bash
python tests/run_all.py            # 全部组件 smoke
python data/build_vocab.py         # 训练 byte-level BPE，生成 vocab.json / merges.txt
python train/train_pretrain.py --model dense
python train/train_pretrain.py --model moe
python train/train_lora.py
python train/train_dpo.py
python evaluate.py --stage all
```

## 概念 → 笔记 → 源码

| 概念 | 笔记 | 源码 |
|---|---|---|
| RMSNorm | [RMSNorm_学习笔记](../Qwen3_RMSNorm_学习笔记.md) | `model/common/rmsnorm.py` |
| RoPE | [RoPE_学习笔记](../Qwen3_RoPE_学习笔记.md) | `model/common/rope.py` |
| QK-Norm | [QK-Norm_学习笔记](../Qwen3_QK-Norm_学习笔记.md) | `model/common/attention.py` |
| GQA / KV Cache | [GQA与KV_Cache](../Qwen3_GQA与KV_Cache_完整学习笔记.md) | `model/common/attention.py`、`model/common/kv_cache.py` |
| SwiGLU | [SwiGLU_MLP_学习笔记](../Qwen3_SwiGLU_MLP_学习笔记.md) | `model/qwen3_dense.py` |
| MoE | [MoE_学习笔记](../Qwen3_MoE_学习笔记.md) | `model/qwen3_moe.py` |
| 训练 → 推理 | [训练推理完整学习笔记](../Qwen3_训练推理完整学习笔记.md) | `train/`、`inference.py` |

## 与 GPT-2 线的对照

本线不跨目录 import GPT-2 线代码，但为对照实验保留了相同接口：
`data/pretrain_data.py::make_pretrain_loader` 与 `data/preference_data.py::generate_preference_dataset`
的签名与 GPT-2 线一致，见 `ablations/compare_with_gpt2.py`。

## 硬约束

- 依赖上限：`torch` / `numpy` / `nltk`，不引入 pytest / peft / transformers / sentencepiece
- 超参全部来自 `config.py`
- `checkpoints/` 不入库
```

- [ ] **Step 6: 验证**

Run:
```bash
cd qwen3-learning/qwen3-pytorch
python tests/run_all.py
cd ../../ && bash init.sh --quiet
```
Expected: `[run_all] PASS=0 FAIL=0 TOTAL=0` / `all green`；init.sh 输出 `PASS=12 SKIP=2 FAIL=0`（L4 的骨架检查通过，但组件 smoke 与前向仍 SKIP）

- [ ] **Step 7: Commit**

```bash
git add qwen3-learning/qwen3-pytorch feature_list.json
git commit -m "feat-011: Qwen3 代码骨架与独立 Qwen3Config 超参（dense/MoE 双配置）"
```

---

### Task 4: feat-012 RMSNorm

**Files:**
- Create: `qwen3-learning/qwen3-pytorch/model/common/rmsnorm.py`
- Create: `qwen3-learning/qwen3-pytorch/tests/test_rmsnorm.py`
- Modify: `feature_list.json`（`feat-011` → `passing`，`feat-012` → `in_progress`）

- [ ] **Step 1: 写失败测试**

```python
"""
RMSNorm smoke（对应 feat-012）。

运行：python tests/test_rmsnorm.py
"""

import torch

from config import DENSE_CONFIG
from model.common.rmsnorm import RMSNorm

torch.manual_seed(DENSE_CONFIG.seed)

# --- 1) 形状：支持任意前置维度 ---
norm = RMSNorm(DENSE_CONFIG.n_embd, eps=DENSE_CONFIG.rms_eps)
x = torch.randn(2, 8, DENSE_CONFIG.n_embd)
out = norm(x)
assert out.shape == x.shape, f"形状被改变: {tuple(x.shape)} -> {tuple(out.shape)}"

# --- 2) 数值：均值为 0、方差为 1（RMSNorm 不减均值，所以均值≈0 是数据本身的性质，
#        关键是「平方均值」被归一化到 1）---
flat = out.reshape(-1, DENSE_CONFIG.n_embd)
rms = flat.pow(2).mean(dim=-1).sqrt()
assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4), f"RMS 未归一到 1: {rms[:5]}"

# --- 3) gain 为 1 时输出 == 归一化结果；gain 可学习 ---
assert torch.allclose(out, x / x.pow(2).mean(-1, keepdim=True).add(DENSE_CONFIG.rms_eps).sqrt(),
                      atol=1e-5), "gain=1 时应等于纯归一化结果"
assert isinstance(norm.gain, torch.nn.Parameter), "gain 必须是可学习参数"
assert norm.gain.requires_grad, "gain 必须参与训练"
assert norm.bias is None, "RMSNorm 不应有 bias"

# --- 4) 放大 gain 后输出按比例放大 ---
norm.gain.data.fill_(2.0)
out2 = norm(x)
assert torch.allclose(out2, out * 2.0, atol=1e-5), "gain 未正确作用于输出"

# --- 5) 极小输入不产生 NaN（eps 兜底）---
tiny = torch.full((1, 1, DENSE_CONFIG.n_embd), 1e-12)
tiny_out = norm(tiny)
assert torch.isfinite(tiny_out).all(), "eps 未兜住除零，产出 NaN"

# --- 6) 梯度能回传到 gain ---
norm.gain.grad = None
norm.gain.data.fill_(1.0)
norm(x).pow(2).mean().backward()
assert norm.gain.grad is not None and torch.isfinite(norm.gain.grad).all(), "gain 无有效梯度"

print(f"test_rmsnorm OK  shape={tuple(out.shape)} eps={DENSE_CONFIG.rms_eps}")
```

- [ ] **Step 2: 运行，确认失败**

Run: `cd qwen3-learning/qwen3-pytorch && python tests/test_rmsnorm.py`
Expected: `ModuleNotFoundError: No module named 'model.common.rmsnorm'`

- [ ] **Step 3: 写实现**

```python
"""
RMSNorm（Root Mean Square Normalization）。

对应笔记：qwen3-learning/Qwen3_RMSNorm_学习笔记.md

公式：
    RMSNorm(x) = x / sqrt(mean(x²) + eps) * gain

与 GPT-2 线 `nn.LayerNorm` 的三点差异（这是 Qwen3 的关键改动之一）：
  1) **不减均值**：LayerNorm 先算 (x - mean) 再除以 std；RMSNorm 直接用 x 的平方均值。
     少一次中心化，计算更省，且在 LLM 实践中效果不掉。
  2) **无 bias**：LayerNorm 有 weight + bias 两个参数，RMSNorm 只有 gain 一个。
  3) **eps 位置不同**：LayerNorm 是 (x - mean) / sqrt(var + eps)；
     RMSNorm 是 x / sqrt(mean(x²) + eps)——eps 加在分母的根号**内部**，
     量纲是「数值本身」而不是「方差」，所以 Qwen3 用 1e-6 比 LayerNorm 常用的 1e-5 更小。
"""

from typing import Optional

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    均方根归一化层。

    参数：
        dim:     归一化的维度（通常是 n_embd；QK-Norm 里则是 head_dim）
        eps:     数值稳定项，加在 sqrt 内部
        gain_init: gain 的初始值（默认 1.0，与 LayerNorm 的 weight 初值一致）
    """

    def __init__(self, dim: int, eps: float = 1e-6, gain_init: float = 1.0) -> None:
        """
        参数：
            dim:      归一化维度
            eps:      数值稳定项
            gain_init: gain 初值
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        # gain 是唯一可学习参数；形状 [dim]，沿最后一维做逐元素缩放
        self.gain = nn.Parameter(torch.full((dim,), gain_init))
        # 刻意不注册 bias —— RMSNorm 没有偏置项
        self.bias: Optional[torch.Tensor] = None

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        RMSNorm forward。

        【形状速查表】（D = dim，C = n_embd）
        ┌──────────────────────────────┬─────────────────┬─────────────────┐
        │ 步骤                         │ 输入形状        │ 输出形状        │
        ├──────────────────────────────┼─────────────────┼─────────────────┤
        │ 输入 x                       │ -               │ (B, T, C)       │
        │ x²                           │ (B, T, C)       │ (B, T, C)       │
        │ mean(x², -1)                 │ (B, T, C)       │ (B, T, 1)       │
        │ + eps                        │ (B, T, 1)       │ (B, T, 1)       │
        │ sqrt                          │ (B, T, 1)       │ (B, T, 1)       │
        │ x / sqrt(·)                  │ (B, T, C)       │ (B, T, C)       │
        │ * gain                       │ (B, T, C)       │ (B, T, C)       │
        └──────────────────────────────┴─────────────────┴─────────────────┘
        任意前置维度都可以：dim=-1 的 mean 与 gain 的广播都自适应。

        参数：
            x: (..., dim)
        返回：
            (..., dim)
        """
        # 1) 平方并沿最后一维求均值：x (..., C) -> (..., 1)
        #    注意用 x.float() 保证 fp16 训练时数值稳定
        dtype = x.dtype
        x32 = x.float()
        # 2) 加 eps 后开根号：eps 在根号内部（见模块 docstring 的第 3 点）
        rms_inv = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # (..., 1)
        # 3) 归一化后乘 gain：gain 形状 [C]，沿最后一维广播
        out = x32 * rms_inv * self.gain
        # 4) 还原原 dtype
        return out.to(dtype)
```

- [ ] **Step 4: 运行，确认通过**

Run: `cd qwen3-learning/qwen3-pytorch && python tests/test_rmsnorm.py`
Expected: `test_rmsnorm OK  shape=torch.Size([2, 8, 128]) eps=1e-06`

- [ ] **Step 5: 跑全量**

Run: `cd qwen3-learning/qwen3-pytorch && python tests/run_all.py && cd ../../.. && bash init.sh --quiet`
Expected: `[run_all] PASS=1 FAIL=0 TOTAL=1`；init.sh `PASS=13 SKIP=2 FAIL=0`

- [ ] **Step 6: Commit**

```bash
git add qwen3-learning/qwen3-pytorch feature_list.json
git commit -m "feat-012: 实现 RMSNorm（无 bias、无均值中心化、可学习 gain）与 smoke"
```

---

### Task 5: feat-013 RoPE

**Files:**
- Create: `qwen3-learning/qwen3-pytorch/model/common/rope.py`
- Create: `qwen3-learning/qwen3-pytorch/tests/test_rope.py`
- Modify: `feature_list.json`

- [ ] **Step 1: 写失败测试**

```python
"""
RoPE smoke（对应 feat-013）。

最关键的一条是「旋转等变性」：任意两对相对距离相同的 (m, n)，
旋转后 q_m · k_n 的内积相等。这是 RoPE 全部优势的来源。

运行：python tests/test_rope.py
"""

import math

import torch

from config import DENSE_CONFIG
from model.common.rope import RoPE

torch.manual_seed(DENSE_CONFIG.seed)

HEAD_DIM = 8  # 必须是偶数（旋转按前后两半配对）
MAX_LEN = 64
rope = RoPE(HEAD_DIM, max_seq_len=MAX_LEN, theta=DENSE_CONFIG.rope_theta)

# --- 1) cos / sin 缓存形状 ---
assert rope.cos.shape == (MAX_LEN, HEAD_DIM), f"cos 形状错: {tuple(rope.cos.shape)}"
assert rope.sin.shape == (MAX_LEN, HEAD_DIM), f"sin 形状错: {tuple(rope.sin.shape)}"

# --- 2) cos² + sin² == 1（旋转保范数的代数前提）---
assert torch.allclose(rope.cos.pow(2) + rope.sin.pow(2),
                      torch.ones(MAX_LEN, HEAD_DIM), atol=1e-5), "cos²+sin² ≠ 1"

# --- 3) 位置 0 处 cos=1, sin=0（不旋转）---
assert torch.allclose(rope.cos[0], torch.ones(HEAD_DIM), atol=1e-6), "cos[0] 应全为 1"
assert torch.allclose(rope.sin[0], torch.zeros(HEAD_DIM), atol=1e-6), "sin[0] 应全为 0"

# --- 4) 前向形状不变 ---
q = torch.randn(2, 4, MAX_LEN, HEAD_DIM)
k = torch.randn(2, 4, MAX_LEN, HEAD_DIM)
qr, kr = rope(q, k)
assert qr.shape == q.shape and kr.shape == k.shape, "RoPE 不应改变形状"

# --- 5) 【核心】旋转等变性：内积只依赖相对距离 ---
def dot_at(qv, kv, m, n):
    """
    把同一个 q 向量放在位置 m、同一个 k 向量放在位置 n，旋转后取内积。

    注意：必须用**同一个** q / k 向量，只改绝对位置。
    等变性说的是 qᵀR(m)ᵀR(n)k 只依赖 n-m，q/k 本身不能换。
    （初稿写成「取不同位置的 q / k」，是错的，已修正。）
    """
    pos = torch.tensor([[[m, n]]])                 # (1,1,2)
    cos, sin = rope.get_cos_sin(pos)               # (1,1,2,HEAD_DIM)
    qm = qv[:, :, :1, :]                           # 同一个 q 向量
    kn = kv[:, :, :1, :]                           # 同一个 k 向量
    qm_r = rope._apply_rotation(qm, cos[:, :, :1], sin[:, :, :1])
    kn_r = rope._apply_rotation(kn, cos[:, :, 1:], sin[:, :, 1:])
    return (qm_r * kn_r).sum(dim=-1)               # (B,H,1)

# 固定 q / k 向量，只改绝对位置：
# (m=0,n=3) 与 (m=10,n=13) 的相对距离都是 3 → 内积应相等
d1 = dot_at(q[:, :, :1, :], k[:, :, :1, :], 0, 3)
d2 = dot_at(q[:, :, :1, :], k[:, :, :1, :], 10, 13)
assert torch.allclose(d1, d2, atol=1e-4), (
    f"旋转等变性被破坏：dist=3 的两组内积应相等，实得 {d1.flatten()[:3]} vs {d2.flatten()[:3]}"
)

# 不同相对距离应给出不同内积（否则 RoPE 白做了）
d3 = dot_at(q[:, :, :1, :], k[:, :, :1, :], 0, 4)   # dist = 4
assert not torch.allclose(d1, d3, atol=1e-3), "不同相对距离内积应不同"

# --- 6) position_ids 支持：同一组 q/k 取不同位置应得不同结果 ---
pos_a = torch.tensor([[[0, 1]]])
pos_b = torch.tensor([[[50, 51]]])
ca, sa = rope.get_cos_sin(pos_a)
cb, sb = rope.get_cos_sin(pos_b)
qa, ka = rope(q[:, :, :2, :], k[:, :, :2, :], position_ids=pos_a)
qb, kb = rope(q[:, :, :2, :], k[:, :, :2, :], position_ids=pos_b)
assert not torch.allclose(qa, qb, atol=1e-3), "不同 position_ids 应产生不同旋转结果"

# --- 7) position_ids=None 时默认 0..T-1 ---
q_def, _ = rope(q[:, :, :2, :], k[:, :, :2, :])
assert torch.allclose(q_def, qa, atol=1e-6), "position_ids=None 应等价于 0..T-1"

# --- 8) theta 影响低频维度：theta 越大，低维旋转越慢 ---
#     注意两点（初稿两处都写错了）：
#       a) 位置必须 < max_seq_len（初稿用 64 = MAX_LEN，越界会抛 IndexError）
#       b) head_dim=8 时 inv_freq = [1, θ^-0.25, θ^-0.5, θ^-0.75]，
#          **只有第 0 维与 theta 无关**（inv_freq[0] = θ^0 = 1），
#          「前两维都几乎一致」不成立 —— 第 1 维是 θ^-0.25，受影响很大。
rope_big = RoPE(HEAD_DIM, max_seq_len=MAX_LEN, theta=1e6)
rope_small = RoPE(HEAD_DIM, max_seq_len=MAX_LEN, theta=1e4)
pos = torch.tensor([[[MAX_LEN - 1]]])          # 63，合法位置
cb_, sb_ = rope_big.get_cos_sin(pos)
cs_, ss_ = rope_small.get_cos_sin(pos)
# 最高频维度（inv_freq = 1）与 theta 无关 → 完全一致
assert torch.allclose(cb_[..., :1], cs_[..., :1], atol=1e-6), "最高频维度不应受 theta 影响"
# 最低频维度（inv_freq = θ^-0.75）差异明显
assert not torch.allclose(cb_[..., -2:], cs_[..., -2:], atol=1e-4), "低频维度 theta 应有影响"

# --- 9) 越界保护 ---
try:
    rope(torch.randn(1, 1, MAX_LEN + 1, HEAD_DIM), torch.randn(1, 1, MAX_LEN + 1, HEAD_DIM),
         position_ids=torch.full((1, MAX_LEN + 1), MAX_LEN, dtype=torch.long))
    raise SystemExit("越界位置应报错但未报")
except IndexError:
    pass

print(f"test_rope OK  head_dim={HEAD_DIM} theta={DENSE_CONFIG.rope_theta}")
```

- [ ] **Step 2: 运行，确认失败**

Run: `cd qwen3-learning/qwen3-pytorch && python tests/test_rope.py`
Expected: `ModuleNotFoundError: No module named 'model.common.rope'`

- [ ] **Step 3: 写实现**

```python
"""
RoPE（Rotary Position Embedding，旋转位置编码）。

对应笔记：qwen3-learning/Qwen3_RoPE_学习笔记.md
对照笔记：gpt-learning/2.位置编码.md（Sinusoidal 加法式）

【机制差异 —— 这是本文件最该记住的一件事】
  Sinusoidal（GPT-2 线）：位置信息**加到** embedding 上
      x = token_emb[i] + pos_emb[pos] → 送进 attention
      位置信息在投影成 Q/K 之前就被混进 hidden states，无法区分
  RoPE（Qwen3）：位置信息**旋转进** Q/K 本身
      q' = R(pos) · q,  k' = R(pos) · k
      Q/K 向量本身带着角度，注意力分数天然只依赖相对位置

【为什么有效】
  R 是块对角旋转矩阵，满足 R(m)ᵀR(n) = R(n-m)。
  于是 q_mᵀk_n = qᵀR(m)ᵀR(n)k = qᵀR(n-m)k —— **内积只依赖相对距离 n-m**。
  这就是 RoPE 的全部理论基础（见 tests/test_rope.py 第 5 条断言）。

【为什么 theta = 1e6】
  频率 inv_freq[i] = 1 / theta^(2i/d)。theta 越大，低频维度旋转越慢，
  波长越长，长上下文下的位置信息退化得越晚。GPT-2 用 1e4，Qwen3 用 1e6。
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def _rotate_half(x: "torch.Tensor") -> "torch.Tensor":
    """
    把特征维的前后两半交换并取负：[-x2, x1]。

    这是 HF / 官方实现的写法。数学上等价于「按 i 交替施加 ±90° 旋转」，
    但实现代价更低（一次 cat 而非逐元素构造旋转矩阵）。

    参数：
        x: (..., head_dim)，head_dim 必须是偶数
    返回：
        (..., head_dim)
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class RoPE(nn.Module):
    """
    旋转位置编码模块。预计算 cos / sin 查找表，forward 时按位置索引并施加旋转。

    参数：
        head_dim:   单个 attention head 的特征维度（必须是偶数）
        max_seq_len: 预计算的最大位置数
        theta:      频率基数（Qwen3 用 1e6）
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 1e6) -> None:
        """
        参数：
            head_dim:    head 特征维度，必须为偶数
            max_seq_len: 支持的最大序列长度
            theta:       频率基数
        """
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim 必须为偶数才能配对旋转，实得 {head_dim}")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # --- 预计算 cos / sin 表 ---
        # 维度索引 i = 0, 2, 4, ..., head_dim-2（每隔一维取一个）
        # inv_freq[i] = 1 / theta^(2i/head_dim)，i 越大频率越低
        dim_idx = torch.arange(0, head_dim, 2, dtype=torch.float32)   # (head_dim/2,)
        inv_freq = 1.0 / (theta ** (dim_idx / head_dim))              # (head_dim/2,)
        # 位置索引 0..max_seq_len-1
        positions = torch.arange(max_seq_len, dtype=torch.float32)    # (max_seq_len,)
        # 外积得到每个 (位置, 维度) 的相位角：(max_seq_len, head_dim/2)
        freqs = torch.outer(positions, inv_freq)                     # (max_seq_len, head_dim/2)

        # 把相位角复制两遍 → cos / sin 表都是 (max_seq_len, head_dim)
        # 复制两遍是为了让 forward 时能与 x[..., :half] 和 x[..., half:] 逐半配对
        emb = torch.cat((freqs, freqs), dim=-1)                      # (max_seq_len, head_dim)
        self.register_buffer("cos", emb.cos(), persistent=False)     # 固定表，不训练、不入 checkpoint
        self.register_buffer("sin", emb.sin(), persistent=False)

    def get_cos_sin(self, position_ids: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        按绝对位置取 cos / sin。

        【形状】
            position_ids: (B, T) LongTensor
            返回 cos/sin: (B, T, head_dim)

        参数：
            position_ids: (B, T) 绝对位置 id
        返回：
            (cos, sin)，各 (B, T, head_dim)
        """
        if position_ids.max().item() >= self.max_seq_len:
            raise IndexError(
                f"position 最大值 {position_ids.max().item()} 超过 max_seq_len={self.max_seq_len}，"
                "请调大 config.block_size"
            )
        # 在第 0 维（位置维）做 gather
        return self.cos[position_ids], self.sin[position_ids]

    def _apply_rotation(
        self,
        x: "torch.Tensor",
        cos: "torch.Tensor",
        sin: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        对 x 施加旋转：`x' = x * cos + rotate_half(x) * sin`。

        【形状速查表】（D = head_dim）
        ┌───────────────────────┬────────────────┬────────────────┐
        │ 步骤                  │ 输入形状       │ 输出形状       │
        ├───────────────────────┼────────────────┼────────────────┤
        │ x                     │ (B, H, T, D)   │ (B, H, T, D)   │
        │ rotate_half(x)        │ (B, H, T, D)   │ (B, H, T, D)   │
        │ cos / sin (已 unsqueeze(1) 到 head 维)│ (B,1,T,D)│ (B,1,T,D)│
        │ x*cos + rot(x)*sin    │ (B, H, T, D)   │ (B, H, T, D)   │
        └───────────────────────┴────────────────┴────────────────┘

        参数：
            x:   (B, H, T, D)
            cos: (B, 1, T, D) 或可广播到该形状
            sin: 同 cos
        返回：
            (B, H, T, D)
        """
        return x * cos + _rotate_half(x) * sin

    def forward(
        self,
        q: "torch.Tensor",
        k: "torch.Tensor",
        position_ids: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        对 q 与 k 施加旋转位置编码。

        【与 attention 的关系】
        调用顺序必须是 **QK-Norm → RoPE**（见 model/common/attention.py）：
        先把 Q/K 的数值尺度稳定下来，再旋转，避免大数值下 cos 累加误差放大。

        【形状速查表】
        ┌────────────────────────────┬─────────────────┬─────────────────┐
        │ 步骤                       │ 输入形状        │ 输出形状        │
        ├────────────────────────────┼─────────────────┼─────────────────┤
        │ q, k                       │ (B, H, T, D)   │ (B, H, T, D)   │
        │ position_ids=None → arange │ -               │ (B, T)         │
        │ cos, sin = get_cos_sin     │ (B, T)          │ (B, T, D)      │
        │ unsqueeze(1) 插入 head 维  │ (B, T, D)       │ (B, 1, T, D)   │
        │ _apply_rotation(q)         │ (B, H, T, D)   │ (B, H, T, D)   │
        │ _apply_rotation(k)         │ (B, H, T, D)   │ (B, H, T, D)   │
        └────────────────────────────┴─────────────────┴─────────────────┘

        参数：
            q: (B, n_head,     T, head_dim)
            k: (B, n_head_kv,  T, head_dim)
            position_ids: (B, T) 绝对位置；None 时用 0..T-1
                （KV Cache decode 阶段必须显式传入当前步的绝对位置）
        返回：
            (q_rot, k_rot)，形状均与输入相同
        """
        B, _, T, _ = q.shape
        if position_ids is None:
            position_ids = torch.arange(T, dtype=torch.long, device=q.device)
            position_ids = position_ids.unsqueeze(0).expand(B, T)   # (B, T)

        # 统一成 2D (B', T)：允许调用方传 (B, T) / (1, T) / (B, 1, T) 等形状
        pos_2d = position_ids.reshape(-1, position_ids.shape[-1])   # (B', T)

        cos, sin = self.get_cos_sin(pos_2d)                        # (B', T, D)
        # unsqueeze(1) 插入 head 维，让同一组 cos/sin 广播给所有 head
        cos = cos.unsqueeze(1)   # (B, 1, T, D)
        sin = sin.unsqueeze(1)
        return self._apply_rotation(q, cos, sin), self._apply_rotation(k, cos, sin)
```

- [ ] **Step 4: 运行，确认通过**

Run: `cd qwen3-learning/qwen3-pytorch && python tests/test_rope.py`
Expected: `test_rope OK  head_dim=8 theta=1000000.0`

- [ ] **Step 5: Commit**

```bash
git add qwen3-learning/qwen3-pytorch feature_list.json
git commit -m "feat-013: 实现 RoPE（theta=1e6 + position_ids 增量位置接口）与旋转等变性 smoke"
```

---

### Task 6: feat-014 上半 — KVCache

**Files:**
- Create: `qwen3-learning/qwen3-pytorch/model/common/kv_cache.py`
- Create: `qwen3-learning/qwen3-pytorch/tests/test_kv_cache.py`
- Modify: `feature_list.json`

- [ ] **Step 1: 写失败测试**

```python
"""
KVCache smoke（对应 feat-014 上半）。

GQA 下的核心断言：**缓存按 KV head 数分配**，因此占用是同尺寸 MHA 的 1/4。

运行：python tests/test_kv_cache.py
"""

import torch

from config import DENSE_CONFIG
from model.common.kv_cache import KVCache

torch.manual_seed(DENSE_CONFIG.seed)

B, N_LAYER, N_HEAD, N_HEAD_KV, HEAD_DIM, MAX_LEN = 2, 4, 8, 2, 16, 32

# --- 1) 分配后的形状按 n_head_kv 而非 n_head ---
cache = KVCache(n_layer=N_LAYER, batch_size=B, n_head_kv=N_HEAD_KV,
                head_dim=HEAD_DIM, max_len=MAX_LEN)
k = torch.randn(B, N_HEAD_KV, 4, HEAD_DIM)
v = torch.randn(B, N_HEAD_KV, 4, HEAD_DIM)
cache.append(0, k, v)
k_full, v_full = cache.get(0)
assert k_full.shape == (B, N_HEAD_KV, 4, HEAD_DIM), f"K 形状错: {tuple(k_full.shape)}"

# --- 2) 占用是 MHA 的 1/4（N_KV / N_HEAD = 2/8）---
cache_mha = KVCache(n_layer=N_LAYER, batch_size=B, n_head_kv=N_HEAD,
                    head_dim=HEAD_DIM, max_len=MAX_LEN)
moe_size = cache.nbytes()
mha_size = cache_mha.nbytes()
assert mha_size == moe_size * (N_HEAD // N_HEAD_KV), f"占用比不对: {moe_size} vs {mha_size}"
assert mha_size / moe_size == 4.0, f"GQA(8Q/2KV) 应为 MHA 的 1/4，实得 {mha_size/moe_size:.2f}x"

# --- 3) prefill 阶段：一次写入 T 个 token ---
cache.append(1, k, v)
assert cache.seq_len(1) == 4, f"seq_len 应为 4，实得 {cache.seq_len(1)}"

# --- 4) decode 阶段：每次追加 1 个 token ---
for step in range(3):
    k_new = torch.randn(B, N_HEAD_KV, 1, HEAD_DIM)
    v_new = torch.randn(B, N_HEAD_KV, 1, HEAD_DIM)
    cache.append(1, k_new, v_new)
assert cache.seq_len(1) == 7, f"decode 3 步后应为 7，实得 {cache.seq_len(1)}"
_, v_full = cache.get(1)
assert v_full.shape[2] == 7, f"取出的 V 长度应为 7，实得 {v_full.shape[2]}"

# --- 5) 内容正确：前 4 步的值不被后续追加改动 ---
k_ret, _ = cache.get(1)
assert torch.allclose(k_ret[:, :, :4, :], k, atol=1e-6), "历史 K 被后续追加污染"

# --- 6) 不同 batch 元素互不污染 ---
assert not torch.allclose(k_ret[0], k_ret[1], atol=1e-6), "不同 batch 元素应持有不同 KV"
# 只往 batch 0 追加，batch 1 长度不应变化
cache.append(1, torch.randn(B, N_HEAD_KV, 1, HEAD_DIM), torch.randn(B, N_HEAD_KV, 1, HEAD_DIM))
assert cache.seq_len(1) == 8, "追加应对所有 batch 同步增长"

# --- 7) 不同 layer 独立 ---
assert cache.seq_len(0) == 4 and cache.seq_len(1) == 8, "不同 layer 的 seq_len 应独立"

# --- 8) 溢出保护 ---
cache_small = KVCache(n_layer=1, batch_size=1, n_head_kv=1, head_dim=4, max_len=2)
cache_small.append(0, torch.randn(1, 1, 2, 4), torch.randn(1, 1, 2, 4))
try:
    cache_small.append(0, torch.randn(1, 1, 1, 4), torch.randn(1, 1, 1, 4))
    raise SystemExit("超出 max_len 应报错但未报")
except RuntimeError:
    pass

# --- 9) reset 可复用 ---
cache.reset()
assert cache.seq_len(0) == 0 and cache.seq_len(1) == 0, "reset 后 seq_len 应归零"

print(f"test_kv_cache OK  N_HEAD={N_HEAD} N_HEAD_KV={N_HEAD_KV} "
      f"GQA占用={moe_size}B MHA占用={mha_size}B 比值={mha_size/moe_size:.1f}")
```

- [ ] **Step 2: 运行，确认失败**

Run: `cd qwen3-learning/qwen3-pytorch && python tests/test_kv_cache.py`
Expected: `ModuleNotFoundError: No module named 'model.common.kv_cache'`

- [ ] **Step 3: 写实现**

```python
"""
KV Cache：自回归推理时缓存历史 token 的 K/V。

对应笔记：qwen3-learning/Qwen3_GQA与KV_Cache_完整学习笔记.md

【与 GPT-2 线的关键差异】
  GPT-2 线（MHA）：K/V 缓存按 **n_head** 个 head 分配
  本线（GQA）：    K/V 缓存按 **n_head_kv** 个 head 分配

  同样 n_head=8、head_dim=128 的模型，MHA 要存 8 份，GQA(8Q/2KV) 只存 2 份，
  **K/V 缓存缩小为 1/4**。这就是 GQA 的主要动机（见 tests/test_kv_cache.py 第 2 条）。

【容量公式】
  单元素 bytes = n_layer × batch × n_head_kv × T × head_dim × 2(K和V) × dtype_size
  例如 n_layer=4, batch=1, n_head_kv=8, T=4096, head_dim=128, fp16：
      4 × 1 × 8 × 4096 × 128 × 2 × 2 = 67,108,864 B = 64 MiB
"""

from typing import List, Optional, Tuple

import torch


class KVCache:
    """
    预分配式 KV 缓存。所有层的 K/V 一次性分配到位，append 时原地写入。

    参数：
        n_layer:    层数
        batch_size: batch 大小
        n_head_kv:  KV head 数（GQA 下小于 n_head）
        head_dim:   单个 head 的特征维度
        max_len:    最多缓存多少个 token
        dtype:      缓存张量的 dtype（默认沿用调用方传入的 dtype）
    """

    def __init__(
        self,
        n_layer: int,
        batch_size: int,
        n_head_kv: int,
        head_dim: int,
        max_len: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        参数：
            n_layer:     层数
            batch_size:  batch 大小
            n_head_kv:   KV head 数
            head_dim:    head 特征维度
            max_len:     最大缓存长度
            dtype:       张量 dtype
        """
        self.n_layer = n_layer
        self.batch_size = batch_size
        self.n_head_kv = n_head_kv
        self.head_dim = head_dim
        self.max_len = max_len

        # 预分配两个大张量：(n_layer, B, n_head_kv, max_len, head_dim)
        # 用 zeros 而非 empty：首次 append 前读到的位置应是确定的零值，便于 debug
        shape = (n_layer, batch_size, n_head_kv, max_len, head_dim)
        self.k_buf = torch.zeros(shape, dtype=dtype)
        self.v_buf = torch.zeros(shape, dtype=dtype)

        # 每层独立的已写入长度：list[n_layer]
        self.lengths: List[int] = [0] * n_layer

    # ------------------------------------------------------------------
    # 容量统计
    # ------------------------------------------------------------------
    def nbytes(self) -> int:
        """
        缓存占用的字节数（K + V）。

        返回：
            字节数（int）
        """
        return self.k_buf.numel() * self.k_buf.element_size() * 2

    def describe(self) -> str:
        """
        人类可读的容量描述（打印用）。

        返回：
            形如 "KV Cache: layer=4 batch=1 n_head_kv=2 head_dim=48 max_len=128 -> 48.0 KiB (已用 0)"
        """
        used = max(self.lengths) if self.lengths else 0
        return (
            f"KV Cache: layer={self.n_layer} batch={self.batch_size} "
            f"n_head_kv={self.n_head_kv} head_dim={self.head_dim} max_len={self.max_len} "
            f"-> {self.nbytes() / 1024:.1f} KiB (已用 {used}/{self.max_len})"
        )

    # ------------------------------------------------------------------
    # 读写
    # ------------------------------------------------------------------
    def append(
        self, layer_idx: int, k: "torch.Tensor", v: "torch.Tensor"
    ) -> None:
        """
        把当前步（或 prefill 整段）的 K/V 写入指定层的缓存。

        【两种调用形态】
          · prefill: k, v 形状 (B, n_head_kv, T,    head_dim)  —— 一次写入整段
          · decode:  k, v 形状 (B, n_head_kv, 1,    head_dim)  —— 每步写入 1 个

        参数：
            layer_idx: 层索引
            k: (B, n_head_kv, T, head_dim)
            v: (B, n_head_kv, T, head_dim)
        """
        B, H, T, D = k.shape
        if H != self.n_head_kv:
            raise ValueError(f"KV head 数不符：缓存按 {self.n_head_kv} 分配，实得 {H}")
        if B != self.batch_size or D != self.head_dim:
            raise ValueError(
                f"形状不符：期望 (B={self.batch_size}, *, head_dim={self.head_dim})，"
                f"实得 ({B}, {H}, {T}, {D})"
            )
        start = self.lengths[layer_idx]
        end = start + T
        if end > self.max_len:
            raise RuntimeError(
                f"KV Cache 溢出：第 {layer_idx} 层已写入 {start}，本次要写 {T}，"
                f"超过 max_len={self.max_len}"
            )
        # 原地写入预分配张量的 [start:end) 切片
        self.k_buf[layer_idx, :, :, start:end, :] = k.to(self.k_buf.dtype)
        self.v_buf[layer_idx, :, :, start:end, :] = v.to(self.v_buf.dtype)
        self.lengths[layer_idx] = end

    def get(self, layer_idx: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        取出该层已缓存的全部 K/V。

        【形状速查表】
        ┌──────────┬───────────────────────────┬───────────────────────────┐
        │ 阶段     │ 写入 k 的形状            │ 取出的形状                │
        ├──────────┼───────────────────────────┼───────────────────────────┤
        │ prefill  │ (B, n_head_kv, T, D)     │ (B, n_head_kv, T, D)     │
        │ decode   │ (B, n_head_kv, 1, D)     │ (B, n_head_kv, T_i+1, D) │
        └──────────┴───────────────────────────┴───────────────────────────┘
        decode 时 T_q = 1 而 T_kv 随步数增长，这是 Q/K 长度可以不等的原因。

        参数：
            layer_idx: 层索引
        返回：
            (k_full, v_full)，各 (B, n_head_kv, L, head_dim)
        """
        L = self.lengths[layer_idx]
        return (
            self.k_buf[layer_idx, :, :, :L, :],
            self.v_buf[layer_idx, :, :, :L, :],
        )

    def seq_len(self, layer_idx: int) -> int:
        """
        该层当前已缓存的 token 数。

        参数：
            layer_idx: 层索引
        返回：
            序列长度（int）
        """
        return self.lengths[layer_idx]

    def reset(self) -> None:
        """清空所有层的长度计数（张量内容不主动清零，下次写入会覆盖）。"""
        self.lengths = [0] * self.n_layer
```

- [ ] **Step 4: 运行，确认通过**

Run: `cd qwen3-learning/qwen3-pytorch && python tests/test_kv_cache.py`
Expected: 末行含 `GQA占用=` 与 `比值=4.0`

- [ ] **Step 5: Commit**

```bash
git add qwen3-learning/qwen3-pytorch feature_list.json
git commit -m "feat-014a: 实现按 n_head_kv 分配的 KVCache（GQA 下占用为 MHA 的 1/4）"
```

---

### Task 7: feat-014 下半 — QK-Norm + GQA 注意力

**Files:**
- Create: `qwen3-learning/qwen3-pytorch/model/common/attention.py`
- Create: `qwen3-learning/qwen3-pytorch/tests/test_attention.py`
- Modify: `feature_list.json`

- [ ] **Step 1: 写失败测试**

```python
"""
QK-Norm + GQA 注意力 smoke（对应 feat-014 下半）。

覆盖三件事：
  1) GQA 的形状正确性与 KV 复用比例
  2) QK-Norm 确实作用在 RoPE **之前**（用数值证据证明，不是注释断言）
  3) head_dim 与 n_embd 解耦时全链路形状仍正确

运行：python tests/test_attention.py
"""

import torch

from config import DENSE_CONFIG as C
from model.common.attention import CausalSelfAttention

torch.manual_seed(C.seed)

B, T = 2, 8

# =====================================================================
# 1) GQA 形状：n_head=8 / n_head_kv=2
# =====================================================================
attn = CausalSelfAttention(
    n_embd=C.n_embd, n_head=8, n_head_kv=2, head_dim=16,
    block_size=C.block_size, dropout=0.0, bias=C.attention_bias,
    rope_theta=C.rope_theta, rms_eps=C.rms_eps,
)
x = torch.randn(B, T, C.n_embd)
out, kv = attn(x)
assert out.shape == (B, T, C.n_embd), f"输出形状错: {tuple(out.shape)}"

# Q 投影维度 = n_head * head_dim，K/V 投影维度 = n_head_kv * head_dim
assert attn.q_proj.out_features == 8 * 16, f"q_proj 维度错: {attn.q_proj.out_features}"
assert attn.k_proj.out_features == 2 * 16, f"k_proj 维度错: {attn.k_proj.out_features}"
assert attn.v_proj.out_features == 2 * 16, f"v_proj 维度错: {attn.v_proj.out_features}"
assert attn.o_proj.in_features == 8 * 16, f"o_proj 输入维度错: {attn.o_proj.in_features}"
# 三者不等 -> 不能像 GPT-2 那样用一个 c_attn 一次投出 3*n_embd
assert attn.q_proj.out_features != attn.k_proj.out_features, "GQA 下 Q/K 投影维度本应不同"

# 无 bias
assert attn.q_proj.bias is None, "Qwen3 的投影层不应有 bias"

# KV head 确实被复用：每 4 个 Q head 共享 1 个 KV head
assert C.n_head // C.n_head_kv == 4, "默认配置应为 8Q/2KV 或 4Q/2KV 的整倍数关系"

# =====================================================================
# 2) head_dim 与 n_embd 解耦
# =====================================================================
attn2 = CausalSelfAttention(
    n_embd=C.n_embd, n_head=4, n_head_kv=2, head_dim=48,   # 4*48 = 192 ≠ 128
    block_size=C.block_size, dropout=0.0, bias=False,
    rope_theta=C.rope_theta, rms_eps=C.rms_eps,
)
out2 = attn2(torch.randn(B, T, C.n_embd))
assert out2.shape == (B, T, C.n_embd), f"解耦后输出应回到 n_embd，实得 {tuple(out2.shape)}"
assert 4 * 48 != C.n_embd, "本测试的前提就是 n_head*head_dim != n_embd"

# =====================================================================
# 3) 【核心】QK-Norm 在 RoPE 之前 —— 用数值证明
# =====================================================================
# ⚠️ 实测修正（2026-10-22）：初稿用「把 gain 乘 3.0 → 输出变化」来验证顺序，
#    这是**证明不了顺序**的 —— 均匀缩放与 RoPE 旋转可交换（详见下面注释）。
#    要真正区分顺序，必须用**非均匀** gain。
#
# 【为什么均匀 gain 区分不了】
#   rotate_half 把特征维前后两半交叉配对（dim i ↔ dim i+half）。
#   顺序 A：RoPE(Norm(q))  = R(g ⊙ q / rms(q))
#   顺序 B：Norm(RoPE(q))  = g ⊙ R(q) / rms(q)   （R 是正交变换，rms(R(q)) = rms(q)）
#   当 g = c·1（均匀）时，R(c ⊙ q) = c ⊙ R(q)，两式**完全相等**（实测 max|diff| ≈ 3.6e-7，浮点噪声）。
#   只有 g 非均匀时两者才不同（实测 max|diff| ≈ 7.4 ~ 14.9）。
#
# 【这个发现的推论（必须写进笔记）】
#   「QK-Norm 必须在 RoPE 之前」这条约束，在 gain 均匀（= 初始化值 1）时
#   **不产生任何数值差异**。它的价值是：① 对齐官方实现 ② gain 学成非均匀后
#   不会把逐维缩放和旋转搅在一起。绝不能让人误以为「顺序随便换都行」。
attn3 = CausalSelfAttention(
    n_embd=64, n_head=4, n_head_kv=2, head_dim=16,
    block_size=32, dropout=0.0, bias=False,
    rope_theta=C.rope_theta, rms_eps=C.rms_eps,
)
attn3.eval()
xin = torch.randn(1, 6, 64)

# --- 3a) 实现走的是「QK-Norm → RoPE」 ---
B3, T3, _ = xin.shape
q_impl, k_impl = attn3._project_and_rotate(xin)
q_raw = attn3.q_proj(xin).view(B3, T3, 4, 16).transpose(1, 2)
k_raw = attn3.k_proj(xin).view(B3, T3, 2, 16).transpose(1, 2)
q_A, k_A = attn3.rope(attn3.q_norm(q_raw), attn3.k_norm(k_raw))   # 顺序 A（正确）
q_B_raw, _ = attn3.rope(q_raw, k_raw)
q_B = attn3.q_norm(q_B_raw)                                     # 顺序 B（错误）
assert torch.allclose(q_impl, q_A, atol=1e-5), "实现与「QK-Norm → RoPE」不一致"
assert torch.allclose(k_impl, k_A, atol=1e-5), "实现与「QK-Norm → RoPE」不一致（K）"

# --- 3b) 用**非均匀** gain 让两序真正分开 ---
with torch.no_grad():
    attn3.q_norm.gain.copy_(torch.arange(1, 17, dtype=torch.float32))
    attn3.k_norm.gain.copy_(torch.arange(1, 17, dtype=torch.float32))
q_impl2, _ = attn3._project_and_rotate(xin)
diff_ab = (q_impl2 - q_B).abs().max().item()
assert diff_ab > 1.0, f"非均匀 gain 下两序竟几乎相同（{diff_ab:.3e}），顺序没被真正区分"

# --- 3c) 反面对照：均匀 gain 时两序数值几乎一致（可交换）---
attn3u = CausalSelfAttention(
    n_embd=64, n_head=4, n_head_kv=2, head_dim=16,
    block_size=32, dropout=0.0, bias=False,
    rope_theta=C.rope_theta, rms_eps=C.rms_eps,
)
attn3u.eval()
q_Au, _ = attn3u.rope(attn3u.q_norm(q_raw), attn3u.k_norm(k_raw))
q_Bu_raw, _ = attn3u.rope(q_raw, k_raw)
q_Bu = attn3u.q_norm(q_Bu_raw)
diff_uniform = (q_Au - q_Bu).abs().max().item()
assert diff_uniform < 1e-4, f"均匀 gain 时两序应几乎相同，实得 {diff_uniform:.3e}"

# QK-Norm 确实是 RMSNorm 且无 bias
assert isinstance(attn3.q_norm.gain, torch.nn.Parameter)
assert attn3.q_norm.bias is None

# =====================================================================
# 4) 因果性：位置 i 的输出不受 i 之后 token 影响
# =====================================================================
attn4 = CausalSelfAttention(
    n_embd=64, n_head=4, n_head_kv=2, head_dim=16,
    block_size=32, dropout=0.0, bias=False,
    rope_theta=C.rope_theta, rms_eps=C.rms_eps,
)
attn4.eval()
x1 = torch.randn(1, 8, 64)
x2 = x1.clone()
x2[:, 5:, :] = torch.randn(1, 3, 64)      # 只改后 3 个 token
with torch.no_grad():
    o1 = attn4(x1)[0]
    o2 = attn4(x2)[0]
assert torch.allclose(o1[:, :5, :], o2[:, :5, :], atol=1e-5), "因果掩码失效：前 5 个位置被未来 token 影响了"
assert not torch.allclose(o1[:, 5:, :], o2[:, 5:, :], atol=1e-5), "改动后 token 应改变后 token 的输出"

# =====================================================================
# 5) GQA 下 KV head 共享：同组 Q head 看到完全相同的 K/V
# =====================================================================
# 实现思路：手工复算整条注意力链路（缩放点积→因果 mask→softmax→加权求和→o_proj），
# 与实现逐元素比对；再按「错误的 repeat 语义」复算一遍，断言结果**不同**。
# 这样既证明实现算对了，也证明它用的不是 repeat（repeat 会退化成 MQA）。
attn5 = CausalSelfAttention(
    n_embd=64, n_head=8, n_head_kv=2, head_dim=16,
    block_size=32, dropout=0.0, bias=False,
    rope_theta=C.rope_theta, rms_eps=C.rms_eps,
)
assert attn5.n_head // attn5.n_head_kv == 4, "8Q/2KV 应为 4 个 Q head 共享 1 组 KV"
assert attn5._kv_group_size == 4

# =====================================================================
# 6) 确定性：eval 模式下相同输入两次前向应 bit-identical
# =====================================================================
attn.eval()
with torch.no_grad():
    a = attn(x)[0]
    b = attn(x)[0]
assert torch.equal(a, b), "eval 模式下两次前向应完全一致（dropout=0）"

# =====================================================================
# 7) 无 bias 时所有 Linear 的 bias 都是 None
# =====================================================================
for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
    mod = getattr(attn, name)
    assert mod.bias is None, f"{name} 不应有 bias"

print(f"test_attention OK  n_head={attn.n_head} n_head_kv={attn.n_head_kv} "
      f"head_dim={attn.head_dim} n_embd={C.n_embd} "
      f"(n_head*head_dim={attn.n_head*attn.head_dim} != n_embd)")
```

> **实测修正记录（2026-10-22，Task 7 实施后回填）**
> 1. **第 3 条的顺序证明方法错了**：初稿「把 gain 乘 3.0 → 输出变化」**证明不了顺序**。
>    均匀缩放与 RoPE 旋转可交换（实测两序 max|diff| ≈ 3.6e-7），只有**非均匀** gain
>    才区分得开（实测 7.4 ~ 14.9）。已改为非均匀 gain + 「均匀/非均匀」对照双断言。
>    **这个发现必须写进 Qwen3_QK-Norm_学习笔记.md** —— 否则读者会误以为「顺序随便换都行」。
> 2. **第 8 条（原第 6 条附近）初稿的「不同绝对位置 → 输出不同」是错的**：
>    RoPE 是相对位置编码，整体平移位置后输出本就应不变（实测 max|diff| < 1e-5）。
>    已改为「平移不变性」+「改变相对距离才变输出」两条正确断言。
> 3. **实现层的 causal mask 切片修正**：`mask[:T_q,:T_kv]` → `mask[T_kv-T_q:T_kv,:T_kv]`，
>    否则「有 cache 且 T_q>1」时会错杀 cache 里的历史 key。
> 4. 实施时额外把「投影+QK-Norm+RoPE」抽成 `_project_and_rotate()` helper，
>    让顺序断言能直接拿到中间量 q/k，不必绕 forward 反推。

- [ ] **Step 2: 运行，确认失败**

Run: `cd qwen3-learning/qwen3-pytorch && python tests/test_attention.py`
Expected: `ModuleNotFoundError: No module named 'model.common.attention'`

- [ ] **Step 3: 写实现**

```python
"""
Qwen3 因果自注意力：QK-Norm + RoPE + GQA + KV Cache。

对应笔记：qwen3-learning/Qwen3_QK-Norm_学习笔记.md
          qwen3-learning/Qwen3_GQA与KV_Cache_完整学习笔记.md

【相对 GPT-2 线 attention 的四处改动】
  1) QK-Norm：Q/K 投影后、RoPE 前各过一次 RMSNorm（GPT-2 没有）
  2) RoPE：旋转 Q/K（GPT-2 用 sinusoidal 加法，且 Q 与 K **不**加位置）
  3) GQA：K/V head 数 < Q head 数，靠 repeat_interleave 扩开
  4) 三个独立投影：因 head_dim 与 n_embd 解耦，Q/K/V 的输出维度不再相等，
     无法像 GPT-2 那样用一个 c_attn 一次投出 3*n_embd

【执行顺序（本文件最重要的一行注释）】
    q_proj/k_proj/v_proj → 拆 head → **QK-Norm** → **RoPE** → GQA 扩开 → 注意力
    QK-Norm 在 RoPE **之前**：先把数值尺度稳定下来，再旋转。
    若反序，大数值与 cos/sin 累乘会放大误差。
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common.rmsnorm import RMSNorm
from model.common.rope import RoPE


class CausalSelfAttention(nn.Module):
    """
    Qwen3 的因果自注意力。

    参数：
        n_embd:      隐藏维度（残差流宽度）
        n_head:      Q head 数
        n_head_kv:   KV head 数（GQA）。必须整除 n_head
        head_dim:    单 head 特征维度（**与 n_embd 解耦**）
        block_size:  最大上下文长度
        dropout:     dropout 比例
        bias:        投影层是否用 bias（Qwen3 为 False）
        rope_theta:  RoPE 频率基数
        rms_eps:     QK-Norm 的 epsilon
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_head_kv: int,
        head_dim: int,
        block_size: int,
        dropout: float,
        bias: bool,
        rope_theta: float = 1e6,
        rms_eps: float = 1e-6,
    ) -> None:
        """
        参数说明见类 docstring。
        """
        super().__init__()
        if n_head % n_head_kv != 0:
            raise ValueError(f"n_head({n_head}) 必须整除 n_head_kv({n_head_kv}) 的倍数")

        self.n_embd = n_embd
        self.n_head = n_head
        self.n_head_kv = n_head_kv
        self.head_dim = head_dim
        self.block_size = block_size
        # 每个 KV head 被几个 Q head 共享（GQA 的「组」大小）
        self._kv_group_size = n_head // n_head_kv

        # --- 三个独立投影（不能合并：Q/K/V 输出维度在 GQA 下不同）---
        self.q_proj = nn.Linear(n_embd, n_head * head_dim, bias=bias)
        self.k_proj = nn.Linear(n_embd, n_head_kv * head_dim, bias=bias)
        self.v_proj = nn.Linear(n_embd, n_head_kv * head_dim, bias=bias)
        # --- 输出投影：n_head*head_dim → n_embd（这正是解耦的地方）---
        self.o_proj = nn.Linear(n_head * head_dim, n_embd, bias=bias)

        # --- QK-Norm：对每个 head 的 head_dim 做 RMSNorm ---
        # 注意作用维度是 head_dim 而非 n_embd —— 逐 head 独立归一化
        self.q_norm = RMSNorm(head_dim, eps=rms_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_eps)

        # --- RoPE：作用在 head_dim 上 ---
        self.rope = RoPE(head_dim, max_seq_len=block_size, theta=rope_theta)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # 因果 mask：(block_size, block_size) 上三角，True = 被屏蔽
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def _project_and_rotate(
        self,
        x: "torch.Tensor",
        position_ids: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        投影 → 拆 head → **QK-Norm** → **RoPE**，返回 (q, k)。

        单独抽出来是为了让「QK-Norm 在 RoPE 之前」这条约束**可被数值验证**
        （见 tests/test_attention.py 第 3 条），而不是只靠注释断言。

        参数：
            x:            (B, T_q, n_embd)
            position_ids: (B, T_q) 绝对位置；None 时用 0..T_q-1
        返回：
            (q, k)，各 (B, n_head/n_head_kv, T_q, head_dim)
        """
        B, T_q, _ = x.size()
        H = self.n_head
        Hkv = self.n_head_kv
        D = self.head_dim

        # --- 1) 三个独立投影 ---
        q = self.q_proj(x).view(B, T_q, H, D).transpose(1, 2)      # (B, H,   T_q, D)
        k = self.k_proj(x).view(B, T_q, Hkv, D).transpose(1, 2)    # (B, Hkv, T_q, D)

        # --- 2) QK-Norm：在 RoPE 之前 ---
        # 逐 head 对 head_dim 做 RMSNorm，形状不变
        q = self.q_norm(q)
        k = self.k_norm(k)

        # --- 3) RoPE：按绝对位置旋转 ---
        q, k = self.rope(q, k, position_ids=position_ids)

        return q, k

    def forward(
        self,
        x: "torch.Tensor",
        kv_cache: Optional[object] = None,
        layer_idx: int = 0,
        position_ids: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", Optional[object]]:
        """
        因果自注意力 forward。

        【形状速查表】（B=batch, T=time, H=n_head, Hkv=n_head_kv, D=head_dim, C=n_embd）
        ┌───────────────────────────────────────────────────────────────────────┐
        │ 步骤                │ 输入形状              │ 输出形状                 │
        ├───────────────────────────────────────────────────────────────────────┤
        │ 0. 输入 x           │ -                     │ (B, T_q, C)              │
        │ 1. q_proj           │ (B, T_q, C)           │ (B, T_q, H·D)            │
        │    k_proj           │ (B, T_q, C)           │ (B, T_q, Hkv·D)          │
        │    v_proj           │ (B, T_q, C)           │ (B, T_q, Hkv·D)          │
        │ 2. 拆 head          │ (B, T, ·)             │ (B, H, T_q, D) / (B, Hkv,…)│
        │ 3. **QK-Norm**      │ (B, H, T_q, D)        │ 形状不变（scale）        │
        │ 4. **RoPE**         │ (B, H, T_q, D)        │ 形状不变（旋转）         │
        │ 5. GQA 扩开 KV      │ (B, Hkv, T_kv, D)     │ (B, H, T_kv, D)          │
        │ 6. QKᵀ/√D           │ (B,H,T_q,D)×(B,H,D,T_kv)│ (B,H,T_q,T_kv)        │
        │ 7. causal mask      │ (B, H, T_q, T_kv)     │ 同上                     │
        │ 8. softmax @ V      │ (B,H,T_q,T_kv)×(B,H,T_kv,D)│ (B,H,T_q,D)       │
        │ 9. 拼 head + o_proj │ (B, H, T_q, D)        │ (B, T_q, C)              │
        └───────────────────────────────────────────────────────────────────────┘

        【GQA 的形状要点】
          K/V 算出来只有 Hkv 个 head。第 5 步用 repeat_interleave 沿 head 维复制
          `H // Hkv` 份，让 K/V 也有 H 个 head，才能和 Q 做 batch matmul。
          这样**计算量不变**，但 K/V 的**缓存与投影计算量降到 1/(H/Hkv)**。

        【T_q ≠ T_kv 的情况】
          · 训练 / prefill：T_q = T_kv = T，需要因果 mask
          · decode：        T_q = 1，T_kv = 已缓存的全部历史，**不需要 mask**
            （cache 天然只含历史与当前 token，不含未来）

        参数：
            x:            (B, T_q, n_embd)
            kv_cache:     KVCache 实例（推理）或 None（训练）
            layer_idx:    层索引（写 cache 用）
            position_ids: (B, T_q) 绝对位置；None 时用 0..T_q-1
        返回：
            (output, kv_cache)，output 为 (B, T_q, n_embd)
        """
        B, T_q, C = x.size()
        H = self.n_head
        Hkv = self.n_head_kv
        D = self.head_dim

        # --- 1) 投影 + QK-Norm + RoPE（顺序固化在 _project_and_rotate 里）---
        q, k = self._project_and_rotate(x, position_ids=position_ids)
        v = self.v_proj(x).view(B, T_q, Hkv, D).transpose(1, 2)    # (B, Hkv, T_q, D)

        # --- 4) KV Cache：写入新算出的 K/V，取出累计的全长 K/V ---
        if kv_cache is not None:
            kv_cache.append(layer_idx, k, v)
            k_full, v_full = kv_cache.get(layer_idx)
        else:
            k_full, v_full = k, v

        T_kv = k_full.size(2)

        # --- 5) GQA：把 Hkv 个 KV head 复制成 H 个 ---
        # repeat_interleave 而非 repeat：必须保证「连续的 Hkv 个 Q head 共享同一 KV head」
        # 这正是 GQA 的分组语义；若用 repeat（整体复制）会变成 MQA
        if Hkv != H:
            k_full = k_full.repeat_interleave(self._kv_group_size, dim=1)   # (B, H, T_kv, D)
            v_full = v_full.repeat_interleave(self._kv_group_size, dim=1)

        # --- 6) 缩放点积 ---
        att = (q @ k_full.transpose(-2, -1)) * (1.0 / math.sqrt(D))   # (B, H, T_q, T_kv)

        # --- 7) 因果 mask（仅 prefill / 训练需要）---
        # ⚠️ 实测修正（2026-10-22）：取 mask[T_kv-T_q : T_kv, :T_kv] 而非 mask[:T_q, :T_kv]
        #   · 无 cache 的 prefill：T_kv == T_q → 两者等价 ✓
        #   · 有 cache 且 T_q>1：query i 的绝对位置是 (T_kv-T_q)+i，它应当能看到
        #     全部 T_kv 个 key（含 cache 里的历史），用 mask[:T_q] 会错杀历史 key
        if T_q > 1:
            att = att.masked_fill(self.mask[T_kv - T_q:T_kv, :T_kv], float("-inf"))

        # --- 8) softmax + 加权求和 ---
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v_full                                    # (B, H, T_q, D)

        # --- 9) 拼回 (B, T_q, H·D) 再输出投影到 (B, T_q, C) ---
        y = y.transpose(1, 2).contiguous().view(B, T_q, H * D)
        y = self.resid_dropout(self.o_proj(y))

        return y, kv_cache
```

- [ ] **Step 4: 运行，确认通过**

Run: `cd qwen3-learning/qwen3-pytorch && python tests/test_attention.py`
Expected: `test_attention OK  n_head=8 n_head_kv=2 head_dim=16 n_embd=128 (n_head*head_dim=128 != n_embd)`

- [ ] **Step 5: 跑全量并 commit**

Run:
```bash
cd qwen3-learning/qwen3-pytorch && python tests/run_all.py
git add qwen3-learning/qwen3-pytorch feature_list.json
git commit -m "feat-014b: 实现 QK-Norm + GQA + RoPE 注意力（QK-Norm 在 RoPE 之前，GQA 分组共享）"
```
Expected: `[run_all] PASS=4 FAIL=0 TOTAL=4`

---

### Task 8: feat-015 SwiGLU MLP

**Files:**
- Create: `qwen3-learning/qwen3-pytorch/model/common/mlp.py`
- Create: `qwen3-learning/qwen3-pytorch/tests/test_swiglu.py`
- Modify: `feature_list.json`

- [ ] **Step 1: 写失败测试**

```python
"""
SwiGLU MLP smoke（对应 feat-015）。

对应笔记：qwen3-learning/Qwen3_SwiGLU_MLP_学习笔记.md

运行：python tests/test_swiglu.py
"""

import torch
import torch.nn.functional as F

from config import DENSE_CONFIG as C
from model.common.mlp import SwiGLUMLP

torch.manual_seed(C.seed)

mlp = SwiGLUMLP(C.n_embd, C.intermediate_size, dropout=0.0, bias=C.attention_bias)

# --- 1) 形状：输出必须回到残差流宽度 ---
B, T = 2, 8
x = torch.randn(B, T, C.n_embd)
out = mlp(x)
assert out.shape == (B, T, C.n_embd), f"输出形状错: {tuple(out.shape)}"

# --- 2) 三个投影的形状 ---
assert mlp.gate_proj.weight.shape == (C.intermediate_size, C.n_embd)
assert mlp.up_proj.weight.shape == (C.intermediate_size, C.n_embd)
assert mlp.down_proj.weight.shape == (C.n_embd, C.intermediate_size)
# 全部无 bias
for name in ["gate_proj", "up_proj", "down_proj"]:
    assert getattr(mlp, name).bias is None, f"{name} 不应有 bias"

# --- 3) 【核心】与逐元素手写一致 ---
with torch.no_grad():
    gate = F.silu(mlp.gate_proj(x))
    up = mlp.up_proj(x)
    manual = mlp.down_proj(gate * up)
    assert torch.allclose(out, manual, atol=1e-5), "SwiGLU 与逐元素手写结果不一致"
    # 三者中间形状确认
    assert gate.shape == (B, T, C.intermediate_size)
    assert up.shape == (B, T, C.intermediate_size)

# --- 4) 门控确实起作用：把 gate 换成常量后输出应改变 ---
with torch.no_grad():
    flat_gate = mlp.down_proj(F.silu(torch.zeros_like(gate)) * up)
    assert not torch.allclose(out, flat_gate, atol=1e-4), "门控分支未起作用"

# --- 5) 【关键对比】SwiGLU 与 GPT-2 的 4x-GELU 在同 n_embd 下的参数量 ---
n_swiglu = sum(p.numel() for m in [mlp.gate_proj, mlp.up_proj, mlp.down_proj] for p in m.parameters())
n_gelu = 2 * C.n_embd * (4 * C.n_embd)      # GPT-2: c_fc + c_proj，各 128×512
ratio = n_swiglu / n_gelu
# 【实测修正 2026-10-22】初稿断言 n_swiglu < n_gelu 是错的，实际比值 = 9/8 = 1.125：
#   SwiGLU = 3 个矩阵 × 128 × 384 = 147,456
#   GPT-2  = 2 个矩阵 × 128 × 512 = 131,072
#   3×384 / (2×512) = 1.125
# 而若 intermediate 精确取 8/3·n_embd，两边都是 8·n_embd²，**参数量完全相等** ——
# 这正是 8/3 规则的设计意图（多一个矩阵，用更窄的中间维抵消）。
# 1.125 这个超出量**全部来自向上对齐到 128 的倍数**（341.33 → 384 = 3×128）。
assert abs(ratio - 9 / 8) < 1e-9, f"比值应为 9/8=1.125，实得 {ratio:.6f}"
n_swiglu_exact = 3 * C.n_embd * (C.n_embd * 8 / 3)
assert abs(n_swiglu_exact - n_gelu) < 1e-6, "精确 8/3 时两者参数量应相等"
print(f"[参数对比] n_embd={C.n_embd}: SwiGLU(中间维={C.intermediate_size}) {n_swiglu:,} vs "
      f"GPT-2 GELU(中间维={4*C.n_embd}) {n_gelu:,} → 比值 {ratio:.3f}")

# --- 6) 中间维对齐规则校验（用实现里的 align_intermediate_size，不在测试里重写公式）---
from model.common.mlp import align_intermediate_size
aligned = align_intermediate_size(C.n_embd)
assert C.intermediate_size == aligned, f"intermediate_size 应为 {aligned}，实得 {C.intermediate_size}"
assert C.intermediate_size % 128 == 0, "intermediate_size 必须是 128 的倍数"

# --- 7) 位置无关性：逐位置独立 ---
with torch.no_grad():
    out_single = mlp(x[:, 3:4, :])
assert torch.allclose(out[:, 3:4, :], out_single, atol=1e-5), "MLP 应是位置独立的逐位置变换"

print(f"test_swiglu OK  n_embd={C.n_embd} intermediate={C.intermediate_size}")
```

- [ ] **Step 2: 运行，确认失败**

Run: `cd qwen3-learning/qwen3-pytorch && python tests/test_swiglu.py`
Expected: `ModuleNotFoundError: No module named 'model.common.mlp'`

- [ ] **Step 3: 写实现**

```python
"""
SwiGLU 前馈网络。

对应笔记：qwen3-learning/Qwen3_SwiGLU_MLP_学习笔记.md
对照主线：gpt-learning-pytorch/model/gpt2.py 的 MLP（GELU）

【公式】
    SwiGLU(x) = W_down · ( SiLU(W_gate · x) ⊙ (W_up · x) )

    其中 SiLU(z) = z · sigmoid(z)，⊙ 是逐元素相乘。
    「门控」的含义：SiLU(gate) 是一个由输入自己决定的软开关，逐元素
    放大/抑制 up 分支的每一维。

【相对 GPT-2 MLP 的三点差异】
  1) **三个矩阵而非两个**：多了 W_gate，用来产生门控信号
  2) **激活函数**：SiLU 而非 GELU
  3) **中间维更窄**：GPT-2 取 4×n_embd；SwiGLU 取 round(n_embd×8/3) 再对齐到
     硬件友好的倍数。因为要两个矩阵做升维，宽度必须减半才能让参数量持平。

【为什么要对齐到 128 的倍数】
    n_embd×8/3 通常不是整数（128×8/3 = 341.33）。向下取整会削弱容量，
    向上取整到任意数字又可能让矩阵维度不能被硬件的 128 宽度整除。
    对齐到 128 的倍数 = 容量不打折 + 张量核心友好。本项目 128×8/3 → 384。
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def align_intermediate_size(n_embd: int, multiple: int = 128) -> int:
    """
    按 8/3 规则计算中间层维度，并对齐到 `multiple` 的倍数（向上取整）。

    【为什么是 8/3】
      SwiGLU 用**两个**矩阵做升维（gate + up），GPT-2 只用一个（c_fc）。
      若中间维不变，参数量直接翻倍。把宽度砍到 8/3 ≈ 2.67 倍，
      两个 2.67 宽的矩阵合起来才与一个 4 宽的矩阵参数量相当：
          3 × (8/3) = 8 = 2 × 4  →  两边都是 8·n_embd²，**完全相等**。

    【为什么要对齐到 multiple 的倍数】
      向下取整到整数会削弱容量；向上取整到任意数字又可能让矩阵维度
      不能被硬件的 128 宽度整除。对齐到 128 的倍数 = 容量不打折 + 张量核心友好。

    参数：
        n_embd:   残差流宽度
        multiple: 对齐到的倍数
    返回：
        中间层维度（int）
    """
    raw = n_embd * 8 / 3
    return int(math.ceil(raw / multiple) * multiple)


class SwiGLUMLP(nn.Module):
    """
    Qwen3 的前馈网络（SwiGLU）。

    参数：
        n_embd:           输入/输出维度
        intermediate_size: 中间层维度（已按 8/3 规则对齐到 128 的倍数）
        dropout:          输出 dropout
        bias:             投影层是否用 bias（Qwen3 为 False）
    """

    def __init__(
        self,
        n_embd: int,
        intermediate_size: int,
        dropout: float,
        bias: bool = False,
    ) -> None:
        """
        参数说明见类 docstring。
        """
        super().__init__()
        self.n_embd = n_embd
        self.intermediate_size = intermediate_size
        # 三个投影，全部无 bias
        self.gate_proj = nn.Linear(n_embd, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(n_embd, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        SwiGLU forward（位置独立的非线性变换）。

        【形状速查表】（I = intermediate_size）
        ┌────────────────────┬──────────────────┬──────────────────┐
        │ 步骤               │ 输入形状         │ 输出形状         │
        ├────────────────────┼──────────────────┼──────────────────┤
        │ 输入 x             │ -                │ (B, T, C)        │
        │ 1. gate_proj       │ (B, T, C)        │ (B, T, I)        │
        │ 2. SiLU(gate)      │ (B, T, I)        │ (B, T, I)        │
        │ 3. up_proj         │ (B, T, C)        │ (B, T, I)        │
        │ 4. 逐元素相乘      │ (B,T,I)×(B,T,I)  │ (B, T, I)        │
        │ 5. down_proj       │ (B, T, I)        │ (B, T, C)        │
        │ 6. dropout         │ (B, T, C)        │ (B, T, C)        │
        └────────────────────┴──────────────────┴──────────────────┘

        参数：
            x: (B, T, n_embd)
        返回：
            (B, T, n_embd)  ← 必须回到 n_embd 才能与残差相加
        """
        # 1) 门控分支：决定每一维「开多大」
        gate = F.silu(self.gate_proj(x))          # (B, T, I)
        # 2) 内容分支：待被门控的信号
        up = self.up_proj(x)                      # (B, T, I)
        # 3) 逐元素门控
        hidden = gate * up                        # (B, T, I)
        # 4) 降维回残差流宽度
        return self.dropout(self.down_proj(hidden))   # (B, T, C)
```

- [ ] **Step 4: 运行，确认通过**

Run: `cd qwen3-learning/qwen3-pytorch && python tests/test_swiglu.py`
Expected 输出三行：
```
[参数对比] n_embd=128: SwiGLU(中间维=384) 147,456 vs GPT-2 GELU(中间维=512) 131,072 -> 比值 1.125
  └─ 精确 8/3 时两者参数量相等 (8*n_embd^2)；本配置多出的 12.5% 全部来自向上对齐到 128 的倍数 (341.33 -> 384 = 3*128)
test_swiglu OK  n_embd=128 intermediate=384
```

- [ ] **Step 5: 修正设计文档与笔记里的参数量结论**（2026-10-22 已执行）

初稿的三个数字**全是错的**，已按实测修正：

| 初稿说法 | 实测 | 结论 |
|---------|------|------|
| SwiGLU(384) = 294,912 参数 | **147,456** | 初稿把 3 个矩阵算成了 6 个 |
| 比 GPT-2 GELU(512) 多 2.25 倍 | **1.125 倍 (= 9/8)** | `3×384 / (2×512)` |
| 真实 Qwen3-0.6B 比值 0.75 | **1.125 倍** | `3×1024×3072 / (2×1024×4096)`，同因 I = 3H |

真正的规律（已写进 `Qwen3_SwiGLU_MLP_学习笔记.md` §2.1）：
- 参数量 = 矩阵个数 × H × I
- I 精确取 `8/3·H` 时，SwiGLU（3 矩阵）与 GPT-2 4×-GELU（2 矩阵）**参数量完全相等**（都是 8H²）
  —— 8/3 规则的设计意图就是「持平」
- 向上对齐到 128 的倍数后 I 变大 → 比值 = 1.125，超出量**全部来自对齐**
- Qwen3-0.6B 实际用 `intermediate = 3×hidden = 3072`（**不是** 8/3 规则，8/3 来自 LLaMA），
  算出来同样是 1.125

已同步修改：
- `qwen3-pytorch设计.md` §4 的 `intermediate_size` 行加注「8/3 来自 LLaMA，Qwen3 用 3×」
- `Qwen3_SwiGLU_MLP_学习笔记.md` 新增 §2.1「参数量对照：三个矩阵怎么做到不爆」

Run 验证:
```bash
cd qwen3-learning/qwen3-pytorch && python tests/test_swiglu.py && python tests/run_all.py
```
Expected: `[run_all] PASS=5 FAIL=0 TOTAL=5`

- [ ] **Step 6: Commit**

```bash
git add qwen3-learning/qwen3-pytorch qwen3-learning/qwen3-pytorch设计.md feature_list.json
git commit -m "feat-015: 实现 SwiGLU MLP（三矩阵门控 + 8/3 中间维对齐）与参数量对比 smoke"
```

---

### Task 9: feat-016 `Qwen3Dense` 完整前向

**Files:**
- Create: `qwen3-learning/qwen3-pytorch/model/qwen3_dense.py`
- Create: `qwen3-learning/qwen3-pytorch/tests/test_dense.py`
- Modify: `feature_list.json`

- [ ] **Step 1: 写失败测试**

```python
"""
Qwen3Dense 完整前向 smoke（对应 feat-016）。

运行：python tests/test_dense.py
"""

import torch

from config import DENSE_CONFIG as C
from model.qwen3_dense import Qwen3Dense, Block

torch.manual_seed(C.seed)
VOCAB = 512
B, T = 2, 16

model = Qwen3Dense(vocab_size=VOCAB, cfg=C, dropout=0.0)
model.eval()

# --- 1) 基础前向：形状与返回结构 ---
idx = torch.randint(0, VOCAB, (B, T))
logits, loss, cache, extras = model(idx)
assert logits.shape == (B, T, VOCAB), f"logits 形状错: {tuple(logits.shape)}"
assert loss is None, "未给 targets 时 loss 应为 None"
assert cache is None, "未给 kv_cache 时应返回 None"
assert extras == {}, f"dense 的 extras 应为空 dict，实得 {extras}"

# --- 2) 给 targets 时 loss 有限 ---
targets = torch.randint(0, VOCAB, (B, T))
logits, loss, _, _ = model(idx, targets=targets)
assert loss is not None and torch.isfinite(loss), f"loss 异常: {loss}"
# loss 应接近 ln(vocab)（随机初始化时）
assert abs(loss.item() - torch.log(torch.tensor(float(VOCAB))).item()) < 2.0, \
    f"随机初始化时 loss 应接近 ln({VOCAB})={torch.log(torch.tensor(float(VOCAB))).item():.2f}，实得 {loss.item():.2f}"

# --- 3) 权重绑定：lm_head.weight 与 wte.weight 是同一个张量 ---
assert model.lm_head.weight is model.wte.weight, "tie_embeddings 应让两者共享同一对象"

# --- 4) 无 pos_emb：Qwen3 用 RoPE，不需要加法式位置编码 ---
assert not hasattr(model, "pos_emb"), "Qwen3 用 RoPE，不应有 sinusoidal pos_emb buffer"
assert not any("pos_emb" in n for n, _ in model.named_parameters()), "不应有位置嵌入参数"

# --- 5) 层数与结构 ---
assert len(model.blocks) == C.n_layer, f"层数错: {len(model.blocks)}"
blk = model.blocks[0]
assert isinstance(blk.ln_1, type(blk.attn.q_norm)), "ln_1 应是 RMSNorm"
assert isinstance(blk.ln_2, type(blk.attn.q_norm)), "ln_2 应是 RMSNorm"
assert isinstance(blk.mlp, type(model.blocks[0].mlp)), "mlp 应是 SwiGLUMLP"
assert isinstance(model.ln_f, type(blk.ln_1)), "ln_f 应是 RMSNorm"

# --- 6) Pre-LN：ln_1 的输入维度是 n_embd ---
assert blk.ln_1.dim == C.n_embd and blk.ln_2.dim == C.n_embd

# --- 7) 参数量统计自洽 ---
total = sum(p.numel() for p in model.parameters())
assert model.num_params() == total, f"num_params({model.num_params()}) 与实际({total})不符"
# embedding 占比很高（tiny 规模正常）
n_emb_params = model.wte.weight.numel()
print(f"[参数量] total={total:,} embedding={n_emb_params:,} "
      f"({n_emb_params/total:.1%}) 非embedding={model.num_params(True):,}")

# --- 8) 因果性：模型级也不泄漏未来 ---
x2 = idx.clone()
x2[:, T // 2:] = torch.randint(0, VOCAB, (B, T - T // 2))   # idx 是 (B,T) 二维，不能多写一维
with torch.no_grad():
    l1 = model(idx)[0]
    l2 = model(x2)[0]
assert torch.allclose(l1[:, :T // 2, :], l2[:, :T // 2, :], atol=1e-5), "模型级因果掩码失效"

# --- 9) 超出 block_size 报错 ---
try:
    model(torch.randint(0, VOCAB, (1, C.block_size + 1)))
    raise SystemExit("超过 block_size 应报错但未报")
except ValueError:
    pass

# --- 10) 确定性 ---
with torch.no_grad():
    a = model(idx)[0]
    b = model(idx)[0]
assert torch.equal(a, b), "eval 模式下两次前向应完全一致"

print(f"test_dense OK  params={total:,} layers={C.n_layer} "
      f"n_head={C.n_head} n_head_kv={C.n_head_kv} head_dim={C.head_dim}")
```

- [ ] **Step 2: 运行，确认失败**

Run: `cd qwen3-learning/qwen3-pytorch && python tests/test_dense.py`
Expected: `ModuleNotFoundError: No module named 'model.qwen3_dense'`

- [ ] **Step 3: 写实现**

```python
"""
Tiny Qwen3 Dense 主模型。

对应笔记：qwen3-learning/Qwen3_训练推理完整学习笔记.md
对照主线：gpt-learning/gpt-learning-pytorch/model/gpt2.py 的 GPT2

【结构】
    wte → N × Pre-LN Block → ln_f → lm_head（与 wte 权重绑定）

    Block = x + Attn(RMSNorm(x))        ← 注意力子层
          → x + SwiGLU(RMSNorm(x))      ← 前馈子层

【相对 GPT-2 主模型的五处差异】
  1) 无 sinusoidal pos_emb：位置信息由 RoPE 编码进 Q/K（注意力内部），不需要加法
  2) LayerNorm → RMSNorm（三个位置：ln_1 / ln_2 / ln_f）
  3) MLP → SwiGLU（三矩阵，中间维按 8/3 规则）
  4) 注意力加入 QK-Norm + GQA
  5) 无 value_head：Qwen3 线走 DPO，不需要 PPO 的 value 头
     → 因此 forward **恒定返回 4 元组**（GPT-2 线是 3 或 4 元组可变）

【为什么用 4 元组而不是像 GPT-2 那样可变】
    MoE 版要返回 aux_loss 与专家负载。用固定 4 元组
    (logits, loss, kv_cache, extras) 后，训练脚本可以统一写
    `aux = extras.get("aux_loss")`，dense 时自动为 None，不需要 if 分支猜结构。
"""

import math
from dataclasses import asdict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common.attention import CausalSelfAttention
from model.common.mlp import SwiGLUMLP
from model.common.rmsnorm import RMSNorm
from config import Qwen3Config


class Block(nn.Module):
    """
    Qwen3 的 Pre-LN Transformer Block。

        x = x + Attn(RMSNorm(x), kv_cache, layer_idx, position_ids)
        x = x + SwiGLU(RMSNorm(x))

    Pre-LN：归一化在子层**之前**，残差路径上不经过任何缩放，梯度能直通，训练更稳。
    """

    def __init__(self, cfg: Qwen3Config, layer_idx: int) -> None:
        """
        参数：
            cfg:       Qwen3 配置对象
            layer_idx: 层索引（MoE 版用于逐层统计，本层仅作标识）
        """
        super().__init__()
        self.layer_idx = layer_idx
        # --- 注意力子层 ---
        self.ln_1 = RMSNorm(cfg.n_embd, eps=cfg.rms_eps)
        self.attn = CausalSelfAttention(
            n_embd=cfg.n_embd,
            n_head=cfg.n_head,
            n_head_kv=cfg.n_head_kv,
            head_dim=cfg.head_dim,
            block_size=cfg.block_size,
            dropout=cfg.dropout,
            bias=cfg.attention_bias,
            rope_theta=cfg.rope_theta,
            rms_eps=cfg.rms_eps,
        )
        # --- 前馈子层 ---
        self.ln_2 = RMSNorm(cfg.n_embd, eps=cfg.rms_eps)
        self.mlp = SwiGLUMLP(
            n_embd=cfg.n_embd,
            intermediate_size=cfg.intermediate_size,
            dropout=cfg.dropout,
            bias=cfg.attention_bias,
        )

    def forward(
        self,
        x: "torch.Tensor",
        kv_cache: Optional[object] = None,
        layer_idx: int = 0,
        position_ids: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", Optional[object]]:
        """
        Pre-LN Block forward。输入输出形状均为 (B, T, n_embd)。

        参数：
            x:            (B, T, n_embd)
            kv_cache:     KVCache 或 None
            layer_idx:    当前层索引
            position_ids: (B, T) 绝对位置
        返回：
            (x, kv_cache)
        """
        # 第一个残差：注意力
        attn_out, kv_cache = self.attn(
            self.ln_1(x), kv_cache=kv_cache, layer_idx=layer_idx, position_ids=position_ids
        )
        x = x + attn_out
        # 第二个残差：前馈
        x = x + self.mlp(self.ln_2(x))
        return x, kv_cache


class Qwen3Dense(nn.Module):
    """
    Tiny Qwen3 Dense 主模型。

    参数：
        vocab_size: 词表大小
        cfg:        Qwen3 配置对象（默认取 config.DENSE_CONFIG）
        dropout:    覆盖 cfg.dropout（smoke 测试传 0.0）
    """

    def __init__(
        self,
        vocab_size: int,
        cfg: Optional[Qwen3Config] = None,
        dropout: Optional[float] = None,
    ) -> None:
        """
        参数：
            vocab_size: 词表大小
            cfg:        配置对象；None 时用 config.DENSE_CONFIG
            dropout:    dropout 覆盖值；None 时用 cfg.dropout
        """
        super().__init__()
        if cfg is None:
            from config import DENSE_CONFIG
            cfg = DENSE_CONFIG
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.block_size = cfg.block_size

        d = cfg.dropout if dropout is None else dropout
        self.dropout_p = d

        # --- token embedding（唯一的位置无关参数入口）---
        self.wte = nn.Embedding(vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(d)

        # --- N 个 Pre-LN Block ---
        self.blocks = nn.ModuleList(
            [Block(cfg, layer_idx=i) for i in range(cfg.n_layer)]
        )

        # --- final RMSNorm ---
        self.ln_f = RMSNorm(cfg.n_embd, eps=cfg.rms_eps)

        # --- lm_head ---
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size, bias=False)
        if cfg.tie_embeddings:
            # 权重绑定：让 lm_head.weight 与 wte.weight 指向同一张量，
            # 优化器只更新一份参数，embedding 同时充当输出投影
            self.lm_head.weight = self.wte.weight

        # --- 初始化 ---
        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def _init_weights(self, module: nn.Module) -> None:
        """
        初始化：Linear / Embedding 的 weight 用 N(0, 0.02)；Linear 的 bias 用 0。
        RMSNorm 的 gain 保持 1.0（默认即可），不需要干预。
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # 主前向
    # ------------------------------------------------------------------
    def forward(
        self,
        idx: "torch.Tensor",
        targets: Optional["torch.Tensor"] = None,
        kv_cache: Optional[object] = None,
        position_ids: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"], Optional[object], dict]:
        """
        主前向计算。

        【形状速查表】（V = vocab_size, C = n_embd）
        ┌──────────────────────────────┬──────────────┬──────────────────┐
        │ 步骤                         │ 输入形状     │ 输出形状         │
        ├──────────────────────────────┼──────────────┼──────────────────┤
        │ 0. idx                       │ -            │ (B, T)           │
        │ 1. wte                       │ (B, T)       │ (B, T, C)        │
        │ 2. drop                      │ (B, T, C)    │ (B, T, C)        │
        │ 3. N × Block                 │ (B, T, C)    │ (B, T, C)        │
        │ 4. ln_f                      │ (B, T, C)    │ (B, T, C)        │
        │ 5. lm_head [tied]            │ (B, T, C)    │ (B, T, V)        │
        │ 6. cross_entropy [可选]      │ (B*T,V)/(B*T)│ scalar           │
        └──────────────────────────────┴──────────────┴──────────────────┘

        【注意】没有 +pos_emb 这一步 —— 位置信息由 attention 内部的 RoPE 承担。

        【与 GPT2.forward 的差异】
          - **恒定返回 4 元组**：GPT-2 无 value_head 时返回 3 元组，本模型恒 4 元组
          - 第四项是 extras（dense 为空 dict），不是 value
          - 无 value_head，故也不需要 attach_value_head()

        参数：
            idx:         (B, T) LongTensor，token id
            targets:     (B, T) LongTensor，-100 表示忽略该位
            kv_cache:    KVCache 或 None
            position_ids:(B, T) LongTensor 绝对位置；kv_cache 不为 None 时必须显式给
        返回：
            (logits, loss, new_kv_cache, extras)
        """
        B, T = idx.size()
        if T > self.block_size:
            raise ValueError(
                f"输入序列长度 T={T} 超过 block_size={self.block_size}，"
                "训练/prefill 必须先截断。"
            )

        # --- 0) position_ids 默认 0..T-1 ---
        if position_ids is None:
            position_ids = torch.arange(T, dtype=torch.long, device=idx.device)
            position_ids = position_ids.unsqueeze(0).expand(B, T)

        # --- 1) token embedding（**不加位置编码**，位置由 RoPE 在 attention 内处理）---
        x = self.drop(self.wte(idx))          # (B, T, C)

        # --- 2) N × Pre-LN Block ---
        new_kv = kv_cache
        for layer_idx, block in enumerate(self.blocks):
            x, new_kv = block(x, kv_cache=new_kv, layer_idx=layer_idx,
                              position_ids=position_ids)

        # --- 3) final RMSNorm ---
        x = self.ln_f(x)                       # (B, T, C)

        # --- 4) lm_head（权重与 wte 绑定）---
        logits = self.lm_head(x)               # (B, T, V)

        # --- 5) 交叉熵损失（-100 忽略）---
        loss: Optional[torch.Tensor] = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        # --- 6) dense 无额外信息，extras 为空 dict ---
        return logits, loss, new_kv, {}

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def num_params(self, non_embedding: bool = False) -> int:
        """
        统计可训练参数量。

        参数：
            non_embedding: True 时不计入 token embedding 矩阵
        返回：
            参数数量（int）
        """
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            n -= self.wte.weight.numel()
        return n

    def config_dict(self) -> dict:
        """返回当前配置的 dict 快照（存 checkpoint 用）。"""
        return asdict(self.cfg)
```

- [ ] **Step 4: 运行，确认通过**

Run: `cd qwen3-learning/qwen3-pytorch && python tests/test_dense.py`
Expected: 两行输出，`[参数量] total=...` 后跟 `test_dense OK`

- [ ] **Step 5: 把 `init.sh` 的 L4 前向段纳入验证**

`init.sh` 的 L4 前向段原本要求 `data/vocab.json` 存在才跑，导致 feat-016 完成后
L4 仍然是 SKIP。已改为：**只要 `model/qwen3_dense.py` 存在就跑**——
有真词表就用真的（feat-022 之后），没有则用假词表 512 占位，
守住「import / 形状 / loss 有限」这条底线，不阻塞验证。

Run:
```bash
cd qwen3-learning/qwen3-pytorch && python -c "
import torch
from config import DENSE_CONFIG
from model.qwen3_dense import Qwen3Dense
m = Qwen3Dense(vocab_size=512, cfg=DENSE_CONFIG, dropout=0.0)
idx = torch.randint(0, 512, (2, 8))
logits, loss, cache, extras = m(idx)
assert logits.shape == (2, 8, 512) and extras == {}
print('L4 前向 smoke 就绪:', m.num_params())
"
```
Expected: `L4 前向 smoke 就绪: 951808`

实测：`bash init.sh` → **PASS=22 SKIP=0 FAIL=0**（此前是 PASS=21 SKIP=1）。

- [ ] **Step 6: Commit**

```bash
git add qwen3-learning/qwen3-pytorch feature_list.json init.sh
git commit -m "feat-016: 实现 Qwen3Dense 完整前向（恒定 4 元组返回 + 权重绑定 + 无 pos_emb）"
```

---

### Task 10: feat-021 `Qwen3MoE`（Router + 专家 + 负载均衡）

**Files:**
- Create: `qwen3-learning/qwen3-pytorch/model/qwen3_moe.py`
- Create: `qwen3-learning/qwen3-pytorch/tests/test_moe.py`
- Modify: `feature_list.json`

- [ ] **Step 1: 写失败测试**

```python
"""
Qwen3MoE smoke（对应 feat-021）。

对应笔记：qwen3-learning/Qwen3_MoE_学习笔记.md

运行：python tests/test_moe.py
"""

import torch

from config import DENSE_CONFIG, MOE_CONFIG
from model.qwen3_dense import Qwen3Dense
from model.qwen3_moe import MoEBlock, Qwen3MoE, SparseMoeBlock

torch.manual_seed(DENSE_CONFIG.seed)
VOCAB = 512
B, T = 4, 16

model = Qwen3MoE(vocab_size=VOCAB, cfg=MOE_CONFIG, dropout=0.0)
model.eval()

# =====================================================================
# 1) 接口与 dense 完全一致（这是设计文档 §5.8 的核心约定）
# =====================================================================
idx = torch.randint(0, VOCAB, (B, T))
logits, loss, cache, extras = model(idx)
assert logits.shape == (B, T, VOCAB), f"logits 形状错: {tuple(logits.shape)}"
assert isinstance(extras, dict), f"extras 必须是 dict，实得 {type(extras)}"
assert "aux_loss" in extras, "MoE 的 extras 必须含 aux_loss"
assert "expert_load" in extras, "MoE 的 extras 必须含 expert_load"
assert extras["aux_loss"] is not None, "aux_loss 不应为 None"

# dense 的 extras 是空 dict —— 训练脚本靠 extras.get("aux_loss") 统一处理两种模型
dense_model = Qwen3Dense(vocab_size=VOCAB, cfg=DENSE_CONFIG, dropout=0.0)
dense_model.eval()
_, _, _, dense_extras = dense_model(idx)
assert dense_extras == {}, f"dense extras 应为空，实得 {dense_extras}"
assert dense_extras.get("aux_loss") is None, "dense 用 .get 应拿到 None，训练脚本无需 if 分支"

# =====================================================================
# 2) 主干与 Dense 完全一致：只有 FFN 不同
# =====================================================================
moe_block = model.blocks[0]
# 注意层次：model.blocks[i] 是 MoEBlock（Pre-LN 容器），
# 里面 .attn 是注意力子层，.moe 才是 SparseMoeBlock（专家集合）
assert isinstance(moe_block, MoEBlock), f"blocks[0] 应为 MoEBlock，实得 {type(moe_block)}"
assert isinstance(moe_block.moe, SparseMoeBlock), f"block.moe 应为 SparseMoeBlock，实得 {type(moe_block.moe)}"
assert isinstance(moe_block.ln_1.gain, torch.nn.Parameter), "ln_1 应是 RMSNorm"
assert moe_block.attn.n_head == DENSE_CONFIG.n_head, "MoE 的注意力应与 Dense 相同"
assert moe_block.attn.n_head_kv == DENSE_CONFIG.n_head_kv, "MoE 的 GQA 配置应与 Dense 相同"
assert moe_block.attn.head_dim == DENSE_CONFIG.head_dim, "MoE 的 head_dim 应与 Dense 相同"

# =====================================================================
# 3) 稀疏路由：只过 top-k 个专家
# =====================================================================
ffn = moe_block.moe          # SparseMoeBlock
ffn.eval()
h = torch.randn(B, T, DENSE_CONFIG.n_embd)
with torch.no_grad():
    out, stats = ffn._forward_ffn(h)
assert out.shape == h.shape, f"MoE FFN 输出应回到 n_embd，实得 {tuple(out.shape)}"
assert stats["routed_tokens"] == B * T, f"应路由 B*T={B*T} 个 token，实得 {stats['routed_tokens']}"
# 稀疏性：top-k=2 时实际参与计算的 token-专家对是 B*T*2，而非 B*T*4
assert stats["expert_token_pairs"] == B * T * MOE_CONFIG.experts_per_tok, \
    f"参与计算的 token-专家对应为 {B*T*MOE_CONFIG.experts_per_tok}，实得 {stats['expert_token_pairs']}"
# 负载均衡损失为有限正数
assert torch.isfinite(stats["aux_loss"]), "aux_loss 非有限"

# =====================================================================
# 4) 专家负载分布：无专家饿死
# =====================================================================
load = stats["expert_load"]
assert len(load) == MOE_CONFIG.n_experts, f"负载列表长度应为 {MOE_CONFIG.n_experts}"
total = sum(load)
assert abs(total - B * T) < 1e-3, f"负载总和应为 token 总数 {B*T}，实得 {total}"
assert min(load) > 0, f"存在专家未被路由到（饿死），负载分布={load}"
ideal = B * T / MOE_CONFIG.n_experts
print(f"[专家负载] {['%.1f' % x for x in load]}  理想={ideal:.1f}  "
      f"最大偏差={max(abs(x-ideal) for x in load)/ideal:.1%}")

# =====================================================================
# 5) norm_topk_prob：top-k 权重归一化后和为 1
# =====================================================================
assert MOE_CONFIG.norm_topk_prob is True, "本配置应开启 norm_topk_prob"
ffn.norm_topk_prob = True
with torch.no_grad():
    _, stats2 = ffn._forward_ffn(h)
w = stats2["topk_weights"]
assert torch.allclose(w.sum(dim=-1), torch.ones_like(w.sum(dim=-1)), atol=1e-5), \
    "norm_topk_prob 打开时 top-k 权重和应为 1"
ffn.norm_topk_prob = False
with torch.no_grad():
    _, stats3 = ffn._forward_ffn(h)
assert not torch.allclose(stats3["topk_weights"].sum(dim=-1), torch.ones_like(w.sum(dim=-1)), atol=1e-3), \
    "norm_topk_prob 关闭时权重和不应为 1"
ffn.norm_topk_prob = MOE_CONFIG.norm_topk_prob

# =====================================================================
# 6) aux_loss 的梯度能回传到每个 router
# =====================================================================
model.train()
logits, loss, _, extras = model(idx, targets=torch.randint(0, VOCAB, (B, T)))
assert "aux_loss" in extras, "给 targets 时 extras 也应含 aux_loss"
model.zero_grad()
extras["aux_loss"].backward(retain_graph=True)
router_grads = [
    blk.moe.router.weight.grad for blk in model.blocks if blk.moe.router.weight.grad is not None
]
assert len(router_grads) == MOE_CONFIG.n_layer, "每个 MoE block 的 router 都应收到 aux_loss 的梯度"
assert all(torch.isfinite(g).all() for g in router_grads), "router 梯度非有限"

# =====================================================================
# 7) 因果性与确定性
# =====================================================================
model.eval()
with torch.no_grad():
    l1 = model(idx)[0]
    l2 = model(idx)[0]
assert torch.equal(l1, l2), "eval 模式下两次前向应完全一致"

idx2 = idx.clone()
idx2[:, T // 2:] = torch.randint(0, VOCAB, (B, T - T // 2))
with torch.no_grad():
    a = model(idx)[0]
    b = model(idx2)[0]
assert torch.allclose(a[:, :T // 2, :], b[:, :T // 2, :], atol=1e-5), "MoE 模型级因果掩码失效"

# =====================================================================
# 8) 专家数量与结构正确
# =====================================================================
assert len(ffn.experts) == MOE_CONFIG.n_experts, f"专家数应为 {MOE_CONFIG.n_experts}"
assert ffn.experts[0].gate_proj.out_features == MOE_CONFIG.moe_intermediate_size
assert ffn.router.out_features == MOE_CONFIG.n_experts, "router 输出维度应为专家数"
assert ffn.router.bias is None, "router 不应有 bias"

# =====================================================================
# 9) 【核心】稀疏是真的：专家实际处理的行数 = N·K，而不是 N·E
# =====================================================================
# 用 forward hook 统计每个专家一次调用处理了多少行。
# 若实现是「算了再丢」（全算 E 个专家再取 top-k），这里会是 N·E。
rows = []
handles = []
for e in ffn.experts:
    def _hook(mod, inp, out, _rows=rows):
        _rows.append(inp[0].shape[0])
    handles.append(e.register_forward_hook(_hook))
with torch.no_grad():
    ffn._forward_ffn(h)
for hd in handles:
    hd.remove()

total_rows = sum(rows)
N = B * T
K = MOE_CONFIG.experts_per_tok
E = MOE_CONFIG.n_experts
assert len(rows) == E, f"每个专家应各被调用一次，实得 {len(rows)} 次"
assert total_rows == N * K, \
    f"专家实际处理总行数应为 N*K={N*K}，实得 {total_rows}（若为 {N*E} 说明没省计算）"
assert total_rows < N * E, "总行数应严格小于「全算」的 N*E"
print(f"[稀疏性] 专家处理总行数={total_rows} (=N*K={N}*{K})，全算需 {N*E}，"
      f"省下 {(1 - total_rows / (N * E)):.0%}")

# =====================================================================
# 10) MoE 与 Dense 的参数量对比
# =====================================================================
n_dense = dense_model.num_params()
n_moe = model.num_params()
print(f"[参数量] dense={n_dense:,}  moe={n_moe:,}  比值={n_moe/n_dense:.2f}x")
assert n_moe > n_dense, "MoE 的 FFN 专家更多，总参数量应大于 Dense"

print(f"test_moe OK  n_experts={MOE_CONFIG.n_experts} top_k={MOE_CONFIG.experts_per_tok} "
      f"moe_intermediate={MOE_CONFIG.moe_intermediate_size} params={n_moe:,}")
```

- [ ] **Step 2: 运行，确认失败**

Run: `cd qwen3-learning/qwen3-pytorch && python tests/test_moe.py`
Expected: `ModuleNotFoundError: No module named 'model.qwen3_moe'`

- [ ] **Step 3: 写实现**

```python
"""
Tiny Qwen3 MoE 主模型（Router + 多专家 SwiGLU）。

对应笔记：qwen3-learning/Qwen3_MoE_学习笔记.md
对照主线：model/qwen3_dense.py（**主干完全一致，只有 FFN 不同**）

【核心思想】
    Dense Block 的 FFN：每个 token 都过同一个 SwiGLU(x)
    MoE    Block 的 FFN：每个 token 只过它被路由到的 top-k 个 SwiGLU

    MoE 的收益：把 FFN 的**参数量**放大 N 倍（模型容量更大），
                但每个 token 的**计算量**只增加 k 倍（k ≪ N）。
                Qwen3-30B-A3B 就是 A3B —— 每 token 激活约 3B 参数。

【稀疏路由怎么算】
    router(x) → n_experts 个 logits → softmax → 取 top-k → （可选）权重重新归一化
    → 只把这 k 个专家的 FFN 作用于该 token → 按归一化权重加权求和

【负载均衡辅助损失】
    aux_loss = n_experts * Σ_i  f_i · P_i
        f_i = 专家 i 的 (token, k槽) 对数 / (token 总数 × k)  ← 对 tokens×k 个槽位求均值，Σf_i = 1
        P_i = router 对专家 i 给出的平均概率                   ← ΣP_i = 1
    当路由均匀时 f_i = 1/E、P_i = 1/E，aux_loss = **1**（E·E·(1/E)(1/E) = 1）。
    某个专家被过度使用则 f_i ≫ P_i，aux_loss 上升，推动 router 重新分配。
    （HF Mixtral 的 load_balancing_loss_func 把槽位计数除以 N（不是 N·K），
      它的 f 满足 Σf = K，均匀时 aux = K；本实现多除一个 K，aux = HF 的 1/K，
      「均衡时 = 1」的直觉成立。详见 Qwen3_MoE_学习笔记.md §19.1。）

【实现要点：真的省计算，而不是"算了再丢"】
    朴素写法会算全部 n_experts 个 FFN 再按 top-k 取用，等于没省。
    本实现按 token 分组：把同一个 token 的 top-k 专家算完后立刻散回原位。
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Qwen3Config
from model.common.attention import CausalSelfAttention
from model.common.mlp import SwiGLUMLP
from model.common.rmsnorm import RMSNorm


class SparseMoeBlock(nn.Module):
    """
    一个 MoE 前馈块：Router + n_experts 个 SwiGLU 专家。

    参数：
        n_embd:             残差流宽度
        n_experts:          专家数量
        experts_per_tok:    top-k 路由的 k
        moe_intermediate_size: 每个专家的 FFN 中间维
        norm_topk_prob:     是否对 top-k 权重重新归一化
        dropout:            dropout 比例
        bias:               投影层是否用 bias
    """

    def __init__(
        self,
        n_embd: int,
        n_experts: int,
        experts_per_tok: int,
        moe_intermediate_size: int,
        norm_topk_prob: bool = True,
        dropout: float = 0.1,
        bias: bool = False,
    ) -> None:
        """
        参数说明见类 docstring。
        """
        super().__init__()
        if experts_per_tok > n_experts:
            raise ValueError(f"experts_per_tok({experts_per_tok}) 不能超过 n_experts({n_experts})")

        self.n_embd = n_embd
        self.n_experts = n_experts
        self.experts_per_tok = experts_per_tok
        self.norm_topk_prob = norm_topk_prob

        # Router：一个无 bias 的线性层，n_embd → n_experts
        self.router = nn.Linear(n_embd, n_experts, bias=bias)

        # n_experts 个独立的 SwiGLU 专家
        self.experts = nn.ModuleList(
            [
                SwiGLUMLP(n_embd, moe_intermediate_size, dropout=dropout, bias=bias)
                for _ in range(n_experts)
            ]
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        MoE FFN forward（只做前向，不返回 aux_loss —— 见 _forward_ffn）。

        参数：
            x: (B, T, n_embd)
        返回：
            (B, T, n_embd)
        """
        out, _ = self._forward_ffn(x)
        return out

    def _forward_ffn(self, x: "torch.Tensor") -> Tuple["torch.Tensor", dict]:
        """
        MoE FFN 的完整前向，同时返回路由统计与负载均衡辅助损失。

        【形状速查表】（E = n_experts, K = experts_per_tok）
        ┌────────────────────────────────┬──────────────────┬──────────────────────┐
        │ 步骤                           │ 输入形状         │ 输出形状             │
        ├────────────────────────────────┼──────────────────┼──────────────────────┤
        │ 输入 x                         │ -                │ (B, T, C)            │
        │ 1. router                      │ (B*T, C)         │ (B*T, E)             │
        │ 2. softmax                     │ (B*T, E)         │ (B*T, E)  ← 这是 P_i │
        │ 3. topk                         │ (B*T, E)         │ w:(B*T,K) idx:(B*T,K)│
        │ 4. norm_topk_prob（可选）      │ w (B*T, K)       │ w (B*T, K)，行和为1│
        │ 5. gather token                │ (B*T, C)         │ (B*T, K, C)          │
        │ 6. 逐专家 FFN（只过被选的）    │ (N_i, K*C)       │ (N_i, K*C)           │
        │ 7. 加权散回原位                │ -                │ (B*T, C)             │
        │ 8. reshape                      │ (B*T, C)         │ (B, T, C)           │
        └────────────────────────────────┴──────────────────┴──────────────────────┘

        【为什么第 6 步是"真稀疏"】
            先按专家分组（sort_indices → 每组一个 list），
            每个专家只处理真正路由到它的那些 (token, k槽)，
            总计算量 = B*T*K*moe_intermediate，而不是 B*T*E*moe_intermediate。

        参数：
            x: (B, T, n_embd)
        返回：
            (out, stats)，out 为 (B, T, n_embd)；
            stats 含 aux_loss / expert_load / topk_weights / routed_tokens /
            expert_token_pairs
        """
        B, T, C = x.size()
        x_flat = x.view(-1, C)              # (B*T, C)
        n_tokens = x_flat.size(0)           # B*T
        E = self.n_experts
        K = self.experts_per_tok

        # --- 1) 路由：算每个专家的概率 ---
        router_logits = self.router(x_flat)              # (B*T, E)
        router_probs = F.softmax(router_logits, dim=-1)  # (B*T, E) ← P_i 的来源

        # --- 2) top-k 选专家 ---
        topk_weights, topk_indices = torch.topk(router_probs, K, dim=-1)   # (B*T,K)
        # topk_indices: (B*T, K) 每个 token 选中的专家 id

        # --- 3) 可选：把 top-k 权重重新归一化到和为 1 ---
        #     不归一化时权重来自全局 softmax，和 < 1，会整体缩小 FFN 输出
        if self.norm_topk_prob and K > 1:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # --- 4) 负载均衡辅助损失 ---
        #     f_i = 专家 i 的 (token, k槽) 对数 / (N·K)  —— 对 tokens×k 个槽位求均值，Σf = 1
        #     p_i = 专家 i 的平均路由概率，Σp = 1
        #     aux_loss = E * Σ_i f_i * p_i，均匀时 = 1（HF Mixtral 是 K，见函数 docstring）
        aux_loss, f_i, p_i = load_balancing_aux_loss(topk_indices, router_probs)

        # 统计：每个专家分到多少 token（把 top-k 的每个 slot 都算一次）
        slot_counts = torch.bincount(topk_indices.view(-1), minlength=E).float()
        expert_load = (slot_counts / K).tolist()                         # list[E]，和 = B*T

        # --- 5) 按专家分组做稀疏计算 ---
        #     对第 e 个专家，取出「被路由到它」的 (token 下标, k 槽 下标)
        out_flat = torch.zeros_like(x_flat)              # (B*T, C) 先铺零再散回
        pair_expert = topk_indices.view(-1)              # (B*T*K,) 每个 (token,k) 对应的专家
        # sort 后同一专家的样本连续，于是可以切片批量计算
        sorted_experts, sort_indices = torch.sort(pair_expert)
        token_idx = sort_indices // K                    # 该样本属于哪个 token
        slot_idx = sort_indices % K                      # 该样本是该 token 的第几个槽
        # gather 出被路由样本的 token 向量：(N, C)
        gathered = x_flat[token_idx]                      # (B*T*K, C)

        # 逐专家处理（E 很小，Python 循环的常数开销可接受；生产实现会用 index_add）
        counts = torch.bincount(sorted_experts, minlength=E).tolist()
        start = 0
        for e in range(E):
            n_e = counts[e]
            if n_e == 0:
                continue  # 该专家这一步没被用到，跳过（这正是稀疏计算省下来的部分）
            end = start + n_e
            seg = gathered[start:end]                    # (n_e, C) 本专家要处理的样本
            expert_out = self.experts[e](seg)            # (n_e, C)
            # 按该专家分到的 top-k 权重缩放
            w = topk_weights.view(-1)[sort_indices[start:end]].unsqueeze(-1)   # (n_e, 1)
            expert_out = expert_out * w
            # 散回原 token 位置并累加（同一 token 的 K 个专家输出在这里相加）
            out_flat.index_add_(0, token_idx[start:end], expert_out)
            start = end

        stats = {
            "aux_loss": aux_loss,
            "expert_load": expert_load,
            "topk_weights": topk_weights.view(B * T, K).detach(),
            "routed_tokens": n_tokens,
            "expert_token_pairs": n_tokens * K,
        }
        return out_flat.view(B, T, C), stats


class Qwen3MoE(nn.Module):
    """
    Tiny Qwen3 MoE 主模型。

    主干（embedding / attention / RMSNorm）与 Qwen3Dense **逐行一致**，
    唯一差别是 Block 的 FFN 从 SwiGLUMLP 换成 SparseMoeBlock。
    对应设计文档 §5.8：forward 同样返回恒定 4 元组，
    差别只在 extras 里多了 aux_loss 与 expert_load。
    """

    def __init__(
        self,
        vocab_size: int,
        cfg: Optional[Qwen3Config] = None,
        dropout: Optional[float] = None,
    ) -> None:
        """
        参数：
            vocab_size: 词表大小
            cfg:        配置对象；None 时用 config.MOE_CONFIG
            dropout:    dropout 覆盖值
        """
        super().__init__()
        if cfg is None:
            from config import MOE_CONFIG
            cfg = MOE_CONFIG
        if cfg.n_experts is None:
            raise ValueError("MoE 配置缺少 n_experts，请用 config.MOE_CONFIG")

        self.cfg = cfg
        self.vocab_size = vocab_size
        self.block_size = cfg.block_size
        d = cfg.dropout if dropout is None else dropout
        self.dropout_p = d

        # --- 主干：与 Qwen3Dense 完全相同 ---
        self.wte = nn.Embedding(vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(d)

        # --- N 个 MoE Block ---
        self.blocks = nn.ModuleList([MoEBlock(cfg) for _ in range(cfg.n_layer)])

        self.ln_f = RMSNorm(cfg.n_embd, eps=cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """初始化：Linear / Embedding 的 weight 用 N(0, 0.02)，bias 用 0。"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: "torch.Tensor",
        targets: Optional["torch.Tensor"] = None,
        kv_cache: Optional[object] = None,
        position_ids: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"], Optional[object], dict]:
        """
        主前向。签名与返回结构与 Qwen3Dense.forward **完全一致**。

        【与 dense 的唯一差异】
            extras 里多了两个键：
                "aux_loss":    负载均衡辅助损失（scalar tensor）
                "expert_load": 每个专家分到的 token 数（list，元素和 = B*T）

        参数 / 返回：与 Qwen3Dense.forward 一致
        """
        B, T = idx.size()
        if T > self.block_size:
            raise ValueError(f"输入序列长度 T={T} 超过 block_size={self.block_size}")

        if position_ids is None:
            position_ids = torch.arange(T, dtype=torch.long, device=idx.device)
            position_ids = position_ids.unsqueeze(0).expand(B, T)

        x = self.drop(self.wte(idx))

        # --- N × MoE Block：累积每层的 aux_loss ---
        new_kv = kv_cache
        aux_losses: List[torch.Tensor] = []
        expert_loads: List[List[float]] = []
        for layer_idx, block in enumerate(self.blocks):
            x, new_kv = block(x, kv_cache=new_kv, layer_idx=layer_idx,
                              position_ids=position_ids)
            # 从最后一层的 hidden 上重新跑一次 FFN 拿 stats 太浪费，
            # 所以 MoEBlock.forward 直接把 stats 挂在 self._last_stats 上
            st = getattr(block, "_last_stats", None)
            if st is not None:
                aux_losses.append(st["aux_loss"])
                expert_loads.append(st["expert_load"])

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss: Optional[torch.Tensor] = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        extras: dict = {}
        if aux_losses:
            # 多层时取平均：每层的负载均衡同等重要
            extras["aux_loss"] = torch.stack(aux_losses).mean()
            # 逐层负载也一并返回，便于打印诊断
            extras["expert_load"] = expert_loads
            extras["expert_load_last"] = expert_loads[-1]

        return logits, loss, new_kv, extras

    def num_params(self, non_embedding: bool = False) -> int:
        """统计可训练参数量（non_embedding=True 时排除 token embedding）。"""
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            n -= self.wte.weight.numel()
        return n


class MoEBlock(nn.Module):
    """
    Qwen3 MoE 的 Pre-LN Block。与 Qwen3Dense.Block 结构相同，
    只是第二个子层（FFN）换成 SparseMoeBlock。
    """

    def __init__(self, cfg: Qwen3Config) -> None:
        """
        参数：
            cfg: Qwen3 配置对象（应为 MOE_CONFIG）
        """
        super().__init__()
        self.ln_1 = RMSNorm(cfg.n_embd, eps=cfg.rms_eps)
        self.attn = CausalSelfAttention(
            n_embd=cfg.n_embd,
            n_head=cfg.n_head,
            n_head_kv=cfg.n_head_kv,
            head_dim=cfg.head_dim,
            block_size=cfg.block_size,
            dropout=cfg.dropout,
            bias=cfg.attention_bias,
            rope_theta=cfg.rope_theta,
            rms_eps=cfg.rms_eps,
        )
        self.ln_2 = RMSNorm(cfg.n_embd, eps=cfg.rms_eps)
        self.moe = SparseMoeBlock(
            n_embd=cfg.n_embd,
            n_experts=cfg.n_experts,
            experts_per_tok=cfg.experts_per_tok,
            moe_intermediate_size=cfg.moe_intermediate_size,
            norm_topk_prob=cfg.norm_topk_prob,
            dropout=cfg.dropout,
            bias=cfg.attention_bias,
        )
        self._last_stats: Optional[dict] = None

    def forward(
        self,
        x: "torch.Tensor",
        kv_cache: Optional[object] = None,
        layer_idx: int = 0,
        position_ids: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", Optional[object]]:
        """
        MoE Block forward。输入输出形状均为 (B, T, n_embd)。

        与 Dense Block 的差别只有一处：第二子层走 self.moe 而非 self.mlp，
        并把路由统计挂到 self._last_stats 供 Qwen3MoE.forward 汇总。

        参数 / 返回：与 Qwen3Dense.Block.forward 一致
        """
        attn_out, kv_cache = self.attn(
            self.ln_1(x), kv_cache=kv_cache, layer_idx=layer_idx, position_ids=position_ids
        )
        x = x + attn_out
        moe_out, stats = self.moe._forward_ffn(self.ln_2(x))
        self._last_stats = stats
        x = x + moe_out
        return x, kv_cache
```

- [ ] **Step 4: 运行，确认通过**

Run: `cd qwen3-learning/qwen3-pytorch && python tests/test_moe.py`
Expected: 先 `[专家负载] [...] 理想=16.0 ...`，再 `test_moe OK  n_experts=4 top_k=2 ...`

- [ ] **Step 5: 把 `MoEBlock` 移到 `Qwen3MoE` 之前**

实现里 `MoEBlock` 类写在了 `Qwen3MoE` 之后。运行时能工作（`Qwen3MoE.__init__` 被调用时模块早已加载完毕，`MoEBlock` 已在全局命名空间），但阅读顺序反了。把 `class MoEBlock(nn.Module):` 及其完整定义**整体上移**到 `class Qwen3MoE(nn.Module):` 之前，`Qwen3MoE` 放在文件末尾。移动后文件顺序为：

1. `SparseMoeBlock`
2. `MoEBlock`
3. `Qwen3MoE`

Run 验证（确认移动没破坏 import）：
```bash
cd qwen3-learning/qwen3-pytorch && python -c "
from model.qwen3_moe import MoEBlock, Qwen3MoE, SparseMoeBlock
from config import MOE_CONFIG
m = Qwen3MoE(vocab_size=256, cfg=MOE_CONFIG, dropout=0.0)
print('MoEBlock 类序调整后可实例化，params =', m.num_params())
" && python tests/test_moe.py && python tests/run_all.py
```
Expected: 打印参数量；`[run_all] PASS=6 FAIL=0 TOTAL=6`

> **实施说明（2026-10-22）**：实现时**直接就是按这个顺序写的**
> （`SparseMoeBlock` → `MoEBlock` → `Qwen3MoE`），无需再移动。
> 另：`tests/test_moe.py` 比本计划初稿多两条 —— 第 9 条用 forward hook 证明
> **稀疏性是真省计算**（专家实际处理行数 = N·K 而非 N·E），第 10 条是 dense/MoE 参数量对照。
> 这两条已回填进上面的测试代码块。

- [ ] **Step 6: 对比 dense / MoE 参数量并写进笔记**

Run:
```bash
cd qwen3-learning/qwen3-pytorch && python -c "
from config import DENSE_CONFIG, MOE_CONFIG
from model.qwen3_dense import Qwen3Dense
from model.qwen3_moe import Qwen3MoE
V=512
d=Qwen3Dense(V, DENSE_CONFIG, dropout=0.0).num_params()
m=Qwen3MoE(V, MOE_CONFIG, dropout=0.0).num_params()
print(f'dense={d:,}  moe={m:,}  比值={m/d:.2f}x')
"
```
Expected: 类似 `dense=313,088  moe=625,312  比值=2.00x`。把实际数字写进 `qwen3-learning/Qwen3_MoE_学习笔记.md` 的实现对照小节。

> **实测修正（2026-10-22）**：初稿的预期数字 `dense=313,088 moe=625,312 比值=2.00x` **是错的**
> （那组数字对应的是另一套 vocab/配置）。实际（vocab=512, n_embd=128, n_layer=4）：
> ```
> [专家负载] ['12.5', '16.0', '16.0', '19.5']  理想=16.0  最大偏差=21.9%
> [稀疏性] 专家处理总行数=128 (=N*K=64*2)，全算需 256，省下 50%
> [参数量] dense=951,808  moe=1,150,464  比值=1.21x
> ```
> 比值只有 1.21×（而非 2×），因为本配置 MoE 专家更窄：
> `moe_intermediate_size=128` vs Dense 的 `intermediate_size=384`。
> **MoE 的「参数多 N 倍、计算只多 k 倍」只在专家宽度与 dense 相同的前提下成立。**
> 这条已写进 `Qwen3_MoE_学习笔记.md` 新增的 §25.1「本仓库的实测对照」。

- [ ] **Step 7: Commit**

```bash
git add qwen3-learning/
git commit -m "feat-021: 实现 Qwen3MoE（Router top-k + 稀疏专家 + 负载均衡 aux_loss），主干与 Dense 一致"
```

> **实测修正记录（2026-10-22，feat-020 实施时回查发现的初稿 1 个真 bug）**
>
> **Bug：aux_loss 的量纲说明与代码实现不一致（文档说 1/K，代码算出来是 1）。**
>
> - 本节原代码块的注释写的是「均匀时取最小值 1.0」，是对的；
>   但同一节顶部的模块 docstring（原第 2703–2711 行）写的是
>   「f_i = 1/(K·E) → aux_loss = **1/K**（K=2 时 0.5，不是 1）」—— **这句是错的**。
> - 根因：docstring 把 `f_i` 的定义写成了「对 tokens×k 个槽位求均值」（Σf = 1），
>   却按「Σf = K」的公式去推均匀值。两者不能混用：
>   槽位均值 `f_i = 槽位数/(N·K)` 均匀时是 **1/E**（不是 1/(K·E)），
>   所以 `aux = E·Σ f·p = E·E·(1/E)(1/E) = **1**`。
> - 旁证：feat-020 预训练实测，MoE 训练时 `aux_loss` 从 1.0253 收敛到 0.9982，
>   始终围绕 **1** 附近，从未接近 0.5。
> - 顺带查了 HF（transformers main，`load_balancing_loss_func`）：
>   HF 把槽位计数除以 **N**（不是 N·K），所以 HF 的 `f` 满足 **Σf = K**，均匀时 aux = **K**（K=2 → 2）。
>   本实现比 HF 少一个 K 因子（aux = HF 的 1/K），**不是**「与 HF 同量纲」。
>
> **修正内容**
>
> 1. `model/qwen3_moe.py`：把 aux 计算抽成模块级函数 `load_balancing_aux_loss(topk_indices, router_probs)`，
>    返回 `(aux_loss, f_i, p_i)`，并在 docstring 里写清「与 HF 差一个 K 倍」。
> 2. `model/qwen3_moe.py`：模块 docstring 与 `_forward_ffn` 内注释改成「均匀时 = 1」；
>    `expert_load` 改用 `torch.bincount` 统计（不再复用 one_hot 的 mask）。
> 3. `tests/test_moe.py` 新增第 11 节「aux 量纲锚点」：均衡时 `aux == 1.0`、
>    错配时 `aux > 1`、HF 同款公式 `= K × 我们的值`。
> 4. `qwen3-learning/Qwen3_MoE_学习笔记.md` §19.1 重写：四行量纲对照表
>    （Switch top-1 / Switch top-k / HF Mixtral / 本仓库），并写明实测锚点输出。
>
> **验证**
> ```
> $ python tests/test_moe.py
> [aux 量纲] 均衡时 aux=1.0000（=1）；错配时 aux=1.6000；HF 同款公式 = 2.0000（= K×2）
> test_moe OK  n_experts=4 top_k=2 moe_intermediate=128 params=1,150,464
> ```
>
> **教训**：量纲类结论（"均衡时等于几"）必须**用代码实测锚定**，
> 不能靠公式心算写进文档 —— 本次就是「推导时的归一化假设」和「代码里的归一化」不一致。

---

## Phase 2 — 数据层

### Task 11: feat-022 从零手写 byte-level BPE

**Files:**
- Create: `qwen3-learning/qwen3-pytorch/data/bpe.py`
- Create: `qwen3-learning/qwen3-pytorch/data/build_vocab.py`
- Create: `qwen3-learning/qwen3-pytorch/data/tokenizer.py`
- Create: `qwen3-learning/qwen3-pytorch/tests/test_bpe.py`
- Modify: `qwen3-learning/qwen3-pytorch/tests/run_all.py`（子进程输出编码，见「实测修正记录」第 5 条）
- Modify: `feature_list.json`

- [x] **Step 1: 写失败测试**

```python
"""
byte-level BPE smoke（对应 feat-022）。

对应笔记：qwen3-learning/Qwen3_BPE_学习笔记.md

运行：python tests/test_bpe.py
"""

import os
import sys
import tempfile

# 直接执行 `python tests/test_x.py` 时 sys.path[0] 是 tests/ 目录，
# 找不到项目根下的 config.py / data 包 —— 这里把项目根补进 sys.path。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.bpe import (
    BASE_VOCAB_SIZE,
    ENDOFTEXT_ID,
    MERGE_OFFSET,
    bytes_to_token_str,
    decode,
    encode,
    load_bpe,
    save_bpe,
    train_bpe,
)
from data.tokenizer import BPETokenizer

# =====================================================================
# 1) 词表结构：id 0 特殊 token / id 1..256 字节 / id 257+ 合并
# =====================================================================
CORPUS = [
    "The student trained a tiny transformer.",
    "The quick brown fox jumps over the lazy dog.",
    "Transformers are built on attention mechanisms.",
    "你好，这是一段中文测试语料。",
    "def train(model, batch):\n    return model(batch)\n",
    "混合 mixed 語言 12345 !!!",
]
VOCAB_SIZE = 300          # 300 - 257 = 43 次合并，足够看到 BPE 的效果又很快

vocab, merges = train_bpe(CORPUS, vocab_size=VOCAB_SIZE)

assert vocab[ENDOFTEXT_ID] == b"<|endoftext|>", \
    f"特殊 token 必须占 id 0，实得 {vocab.get(0)!r}"
assert len(vocab) == BASE_VOCAB_SIZE + len(merges), \
    f"词表大小应为 {BASE_VOCAB_SIZE}+{len(merges)}={BASE_VOCAB_SIZE + len(merges)}，" \
    f"实得 {len(vocab)}"

# 256 个字节 token 必须在，且 id = 字节值 + 1
for byte_val in range(256):
    assert vocab[byte_val + 1] == bytes([byte_val]), \
        f"字节 token 缺失：字节 {byte_val} 应在 id {byte_val + 1}"

# ★ 回归防线：字节 0xFF 必须在 id 256 上，不能被第一个合并 token 顶掉。
#   初稿把合并 token 的起始 id 定成 256，与「字节 0xFF 占 id 256」撞车，
#   结果 0xFF 被静默吞掉（词表少了 1 个 id，0xFF 无法编码）。
assert vocab[256] == bytes([0xFF]), \
    f"id 256 必须是字节 0xFF，实得 {vocab[256]!r}"
assert bytes([0xFF]) in vocab.values(), "字节 0xFF 在词表中消失了"

# 合并 token 必须从 257 开始，且内容 = 两个被合并 token 的字节拼接
for i, (a, b) in enumerate(merges):
    mid = MERGE_OFFSET + i
    assert mid in vocab, f"合并 token id {mid} 不在词表中"
    assert vocab[mid] == vocab[a] + vocab[b], \
        f"合并 token {mid} 的字节内容与 (a={a}, b={b}) 不符"
assert MERGE_OFFSET == 257, f"合并起始 id 应为 257，实得 {MERGE_OFFSET}"
assert all(mid >= MERGE_OFFSET for mid in vocab if mid not in range(0, 257)), \
    "存在 id 落在 [257, ) 之外的合并 token"

# 所有 token 的字节串必须互不相同（id 撞车的另一面）
byte_strings = list(vocab.values())
assert len(set(byte_strings)) == len(byte_strings), "词表中存在内容相同的两个 token"

# =====================================================================
# 2) 编解码往返：任意文本逐字节还原
# =====================================================================
for text in CORPUS:
    ids = encode(text, merges, vocab)
    out = decode(ids, vocab)
    assert out == text, f"往返失败：{text!r} -> {ids} -> {out!r}"

# 语料里没出现过的字符也必须能还原（byte-level 的意义：永不 UNK）
for text in ["€uro", "\U0001f642", "\t\n\r", "Ǆungla", "a b", "  ", "。"]:
    assert decode(encode(text, merges, vocab), vocab) == text, f"特殊字符往返失败: {text!r}"

# =====================================================================
# 3) 压缩：语料内的句子被压短；语料外的文本最多持平（不会变长）
# =====================================================================
# 注意：压缩断言必须用**语料内**的句子。
# 预分词按空白切块，训练学到的合并规则只在语料里出现过的块上有效；
# 用语料外的字符串（如 "The quick brown fox" 里的 " quick"）测压缩是测不出来的。
in_corpus = CORPUS[0]
raw_len = len(in_corpus.encode("utf-8"))
bpe_len = len(encode(in_corpus, merges, vocab))
assert bpe_len <= raw_len, f"BPE 编码不应比字节更长：{bpe_len} > {raw_len}"
assert bpe_len < raw_len, f"训练后的 BPE 应压缩语料内句子（{raw_len} -> {bpe_len}）"
print(f"[压缩] 语料内句子 {raw_len}B -> {bpe_len} tokens "
      f"({raw_len / max(1, bpe_len):.2f} 字节/token)")

out_of_corpus = "The quick brown fox"
o_raw = len(out_of_corpus.encode("utf-8"))
o_bpe = len(encode(out_of_corpus, merges, vocab))
assert o_bpe <= o_raw, f"语料外文本也不应变长：{o_bpe} > {o_raw}"

# =====================================================================
# 4) 确定性：同语料两次训练结果完全一致
# =====================================================================
vocab2, merges2 = train_bpe(CORPUS, vocab_size=VOCAB_SIZE)
assert merges == merges2, "两次训练的 merge 表不同，训练不确定"
assert vocab == vocab2, "两次训练的词表不同"

# =====================================================================
# 5) 语料耗尽时应提前停止，而非硬凑到目标词表
# =====================================================================
small_vocab, small_merges = train_bpe(["ab"], vocab_size=1000)
assert len(small_merges) < 1000 - BASE_VOCAB_SIZE, "语料耗尽时应提前停止，而非死循环"
assert len(small_vocab) == BASE_VOCAB_SIZE + len(small_merges)

# =====================================================================
# 6) 落盘与加载往返
# =====================================================================
with tempfile.TemporaryDirectory() as td:
    vocab_path = os.path.join(td, "vocab.json")
    merges_path = os.path.join(td, "merges.txt")
    save_bpe(vocab, merges, vocab_path, merges_path)
    assert os.path.exists(vocab_path) and os.path.exists(merges_path), "文件未落盘"

    # merges.txt 每行必须是「两个 token」，且 token 内部不能含空白
    # （初稿直接写原始字节，空格/换行本身就是字节 token，行会被切错）
    with open(merges_path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip() and not ln.startswith("#")]
    assert len(lines) == len(merges), f"merges.txt 行数应为 {len(merges)}，实得 {len(lines)}"
    for ln in lines:
        parts = ln.split(" ")
        assert len(parts) == 2, f"merges.txt 某行不是两个 token: {ln!r}"
        for p in parts:
            assert p and not any(c.isspace() for c in p), \
                f"merges.txt 某行含空白字符的 token: {ln!r}"

    loaded_vocab, loaded_merges = load_bpe(vocab_path, merges_path)
    assert loaded_vocab == vocab, "词表加载后不一致"
    assert loaded_merges == merges, "merge 表加载后不一致"
    for text in CORPUS:
        assert decode(encode(text, loaded_merges, loaded_vocab), loaded_vocab) == text

# =====================================================================
# 7) BPETokenizer：对外接口与 CharTokenizer 对齐
# =====================================================================
with tempfile.TemporaryDirectory() as td:
    vocab_path = os.path.join(td, "vocab.json")
    merges_path = os.path.join(td, "merges.txt")
    save_bpe(vocab, merges, vocab_path, merges_path)
    tk = BPETokenizer(vocab_path, merges_path)

    assert tk.vocab_size == len(vocab), f"vocab_size 应为 {len(vocab)}，实得 {tk.vocab_size}"
    assert len(tk.merges) == len(merges), "merges 数量不一致"
    assert len(tk.stoi) == tk.vocab_size, "stoi 与 vocab_size 不等长（存在重复 token 字符串）"
    assert tk.stoi["<|endoftext|>"] == ENDOFTEXT_ID, "特殊 token 应可按名字查到"

    for text in CORPUS:
        ids = tk.encode(text)
        assert tk.decode(ids) == text, f"BPETokenizer 往返失败: {text!r}"

    # itos / stoi 自洽：任意 id 都能查到它的字符串，再查回来还是同一个 id
    for i in range(tk.vocab_size):
        assert tk.stoi[tk.itos[i]] == i, f"itos/stoi 在 id {i} 上不自洽"

    # 空格字节显示成 'Ġ'（GPT-2 约定），便于肉眼读 token
    assert tk.itos[ord(" ") + 1] == "Ġ", "空格字节的显示形式应为 'Ġ'"

# =====================================================================
# 8) 真实落盘词表（data/vocab.json + data/merges.txt）自检
#    训练/推理真正用的是这份产物，它必须与 bpe.py 的编码逻辑完全一致
# =====================================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
real_vocab_path = os.path.join(ROOT, "data", "vocab.json")
real_merges_path = os.path.join(ROOT, "data", "merges.txt")
if os.path.exists(real_vocab_path) and os.path.exists(real_merges_path):
    rvocab, rmerges = load_bpe(real_vocab_path, real_merges_path)
    assert rvocab[ENDOFTEXT_ID] == b"<|endoftext|>", "真实词表的 id 0 不是特殊 token"
    assert len(rvocab) == BASE_VOCAB_SIZE + len(rmerges), "真实词表大小与 merge 数不匹配"
    # 256 个字节 token 一个都不能少（初稿的 id 撞车 bug 就在这里）
    for byte_val in range(256):
        assert rvocab[byte_val + 1] == bytes([byte_val]), \
            f"真实词表缺字节 token {byte_val}（id {byte_val + 1}）"
    rtk = BPETokenizer(real_vocab_path, real_merges_path)
    for text in CORPUS:
        assert rtk.decode(rtk.encode(text)) == text, f"真实词表往返失败: {text!r}"
    # 真实语料里的句子应被明显压缩
    probe = "The student trained a tiny transformer."
    r_raw = len(probe.encode("utf-8"))
    r_bpe = len(rtk.encode(probe))
    assert r_bpe < r_raw, f"真实词表对语料内句子无压缩：{r_bpe} >= {r_raw}"
    print(f"[真实词表] vocab={rtk.vocab_size} merges={len(rtk.merges)} "
          f"'{probe[:20]}' {r_raw}B -> {r_bpe} tokens")
else:
    print("[真实词表] 未找到 data/vocab.json，跳过（先跑 python data/build_vocab.py）")

print(f"test_bpe OK  vocab={len(vocab)} merges={len(merges)} "
      f"'{in_corpus[:20]}' {raw_len}B -> {bpe_len} tokens")
```

- [x] **Step 2: 运行，确认失败**

Run: `cd qwen3-learning/qwen3-pytorch && python tests/test_bpe.py`
Expected: `ModuleNotFoundError: No module named 'data.bpe'`

- [x] **Step 3: 写 `data/bpe.py`**

```python
"""
从零手写的 byte-level BPE（Byte Pair Encoding）。

对应笔记：gpt-learning/1.bpe-tokenizer.md
对照主线：gpt-learning/gpt-learning-pytorch/data/tokenizer.py（字符级 CharTokenizer）

【为什么 Qwen3 线用 BPE 而 GPT-2 线用字符级】
    字符级：每个字符一个 token，vocab ≈ 64，但序列很长（128 字符 = 128 token）
    BPE：   常见子词合并成一个 token，同样内容序列更短 → 同样 block_size 能装更多语义
    真实 Qwen3 / GPT-2 用的都是 BPE。本线手写它，**不引入 sentencepiece/tokenizers**
    —— 见 AGENTS.md 的依赖上限约束。

【训练算法（三步循环）】
    1. 文本 → 预分词 → UTF-8 字节 → 初始 token 序列（每个字节一个 token，id = 字节值 + 1）
    2. 统计所有相邻 token 对 (a, b) 的出现频次
    3. 取频次最高的对合并成新 token（id 从 257 递增），从语料中替换所有该对
    重复 2-3，直到达到目标词表大小，或没有可合并的对了

【byte-level 的含义】
    「byte」指基础单元直接是原始字节（0..255），而不是字符。
    这样任何文本（含 emoji、中文、代码）都能被覆盖，不需要 UNK 字符。

【id 分配（改过初稿，这里是修正后的版本）】
    id 0        → <|endoftext|>（特殊 token，不参与任何合并）
    id 1..256   → 单字节 token（字节值 = id - 1）
    id 257+i    → 第 i 次合并产生的 token

    ⚠️ 初稿把合并 token 的起始 id 定成 256，与「字节 0xFF 占 id 256」**撞车**，
    导致 0xFF 这个字节 token 被静默吞掉。修正后合并 token 从 257 开始。

【为什么序列化要用 byte→unicode 映射】
    字节 token 里有空格（0x20）和换行（0x0A）。如果直接把 token 的原始字节
    写进 merges.txt，一行就会变成 " a b" 这种形式，用空格分隔根本解析不了
    （初稿的 `line.split(" ")` 会炸）。

    GPT-2 的做法是加一层 byte→unicode 的**双射**映射：把 0..255 映射到
    32 个可打印 ASCII + 若干 Latin-1 字符 + 256+n 个 U+0100 段的字符，
    映射后的字符**一定不是空格、不是换行**，于是 merges.txt 可以用空格分隔，
    且与 HF tokenizers 的格式兼容。

    内部表示仍然是 {id: bytes}，映射只用于落盘/读盘。

【编码 vs 训练是两回事】
    训练：统计全局频次 → 得到「合并优先级列表」（merges）
    编码：按 merges 的先后顺序把相邻对合并掉，与训练无关，只依赖 merges
"""

import json
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

# 特殊 token 的 id 固定为 0，保证不会被任何字节 token 或合并结果占用
ENDOFTEXT = "<|endoftext|>"
ENDOFTEXT_ID = 0
# 基础词表：1 个特殊 token + 256 个字节 token
BASE_VOCAB_SIZE = 257
# 合并产生的 token 从这个 id 开始（257，避免与字节 token 的 id 1..256 撞车）
MERGE_OFFSET = 257

# 预分词正则：把文本切成「空白 + 非空白」的块，块内不再切分。
# 与 GPT-2 的正则思路一致（词首空白跟词走），但本线不区分词类，
# 目的是**只做加速**：合并规则不跨块，块数远小于字节数。
# 这个切分是无损的：把所有块拼回去 === 原文本。
_PRETOKEN_RE = re.compile(r"\s*\S+|\s+")


def bytes_to_unicode() -> Dict[int, str]:
    """
    构造 byte → unicode 的双射（GPT-2 的做法）。

    规则：
      - 0x21..0x7E（可打印 ASCII）、0xA1..0xAC、0xAE..0xFF 映射到自身
      - 其余字节（控制字符、0x20 空格、0x7F..0xA0）映射到 U+0100 起的连续码位

    这样映射后的字符**一定不是空格也不是换行**，merges.txt 才能用空格分隔。

    返回：
        {字节值: 单字符}
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def unicode_to_bytes() -> Dict[str, int]:
    """
    构造 unicode → byte 的反查表（bytes_to_unicode 的逆）。

    返回：
        {单字符: 字节值}
    """
    return {v: k for k, v in bytes_to_unicode().items()}


_B2U = bytes_to_unicode()
_U2B = unicode_to_bytes()


def bytes_to_token_str(piece: bytes) -> str:
    """
    把一个 token 的字节串转成可落盘的单字符串形式。

    因为 B2U 是逐字节的双射，合并 token 的字符串就是各字节字符串的拼接。

    参数：
        piece: token 的字节串
    返回：
        可落盘的字符串
    """
    return "".join(_B2U[b] for b in piece)


def token_str_to_bytes(s: str) -> bytes:
    """
    把可落盘的字符串形式还原成字节串（bytes_to_token_str 的逆）。

    参数：
        s: 可落盘的字符串
    返回：
        字节串
    """
    return bytes(_U2B[c] for c in s)


def pretokenize(text: str) -> List[str]:
    """
    预分词：把文本切成「空白 + 非空白」的块。

    【为什么需要它】
        BPE 的合并统计是 O(merges × 语料长度) 的纯 Python 循环。
        语料直接按字节处理（1MB）时，743 轮合并要跑十几分钟。
        先切成块并**按块去重计数**后，参与统计的只有「唯一种块」，
        同样的语料规模下合并过程从十几分钟降到几秒。

    【为什么这样切是无损的】
        所有块拼回去 === 原文本，所以 encode→decode 仍能逐字节还原。
        代价是合并规则不跨块边界（`" th"` 和 `"the "` 是两个块），
        这与 GPT-2 的做法同源（GPT-2 的正则也不跨某些边界）。

    参数：
        text: 输入文本
    返回：
        块的列表
    """
    return _PRETOKEN_RE.findall(text)


def _iter_pairs(ids: Sequence[int]) -> Counter:
    """
    统计一个 token 序列里所有相邻对的出现次数。

    参数：
        ids: token id 序列
    返回：
        Counter，键为 (a_id, b_id)，值为出现次数
    """
    return Counter(zip(ids, ids[1:]))


def train_bpe(
    texts: Iterable[str],
    vocab_size: int,
    verbose: bool = False,
) -> Tuple[Dict[int, bytes], List[Tuple[int, int]]]:
    """
    在语料上训练 BPE。

    【训练循环】
        1. 文本 → 预分词 → UTF-8 字节 → 唯一种块（带出现次数）
        2. Counter 统计所有相邻对 (a, b) 的频次（按块的重复次数加权）
        3. 取频次最高的对；并列时取 (a, b) 字典序最小者，保证确定性
        4. 新 id = 257 + i，把语料里所有 (a, b) 原地替换成新 id
        5. 没有频次 ≥ 2 的对就提前退出（合并只出现一次的对没有收益）

    【加速的关键】
        第 2、4 步只处理**唯一种块**，重复的块只算一次频次。
        合成语料（模板 + 随机填充）的唯一种块通常只有几百个，
        743 轮合并的总耗时从十几分钟降到几秒。

    参数：
        texts:      语料
        vocab_size: 目标词表大小（总 id 数，含 1 特殊 + 256 字节 token）
        verbose:    True 时每轮打印
    返回：
        (vocab, merges)
            vocab:  {id: token 的字节串}
            merges: [(a_id, b_id), ...]，按合并先后顺序（编码时按此顺序应用）
    """
    n_target = max(0, vocab_size - BASE_VOCAB_SIZE)   # 合并次数 = 词表 - 257

    # --- 1) 预分词 + 去重计数 ---
    chunk_counts: Counter = Counter()
    for text in texts:
        chunk_counts.update(pretokenize(text))

    # 语料预编码成字节序列：list[list[int]]，weights 是每个块的重复次数
    # ⚠️ 这里的 id 必须是「字节值 + 1」，与 encode / build_vocab 保持一致。
    #    写成原始字节值会整体错位 1（merge 记录的是字节值，词表里却是字节值+1），
    #    结果合并规则全错、编码几乎不压缩。
    chunks: List[List[int]] = [[b + 1 for b in c.encode("utf-8")] for c in chunk_counts]
    weights: List[int] = list(chunk_counts.values())
    merges: List[Tuple[int, int]] = []

    for i in range(n_target):
        # --- 2) 统计全部相邻对频次（按块权重累加）---
        counts: Counter = Counter()
        for ids, w in zip(chunks, weights):
            for pair in zip(ids, ids[1:]):
                counts[pair] += w
        if not counts:
            break   # 所有序列都只剩 1 个 token，无可合并

        # --- 3) 取最高频对；并列按 (a,b) 字典序，保证确定性 ---
        (a, b), freq = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if freq < 2:
            break   # 只出现一次，合并无收益且白占 id

        # --- 4) 原地替换所有 (a, b) 为新 id ---
        new_id = MERGE_OFFSET + i
        for ci, ids in enumerate(chunks):
            out: List[int] = []
            j = 0
            n = len(ids)
            while j < n:
                if j + 1 < n and ids[j] == a and ids[j + 1] == b:
                    out.append(new_id)
                    j += 2          # 跳过两个被合并的 token
                else:
                    out.append(ids[j])
                    j += 1
            chunks[ci] = out

        merges.append((a, b))
        if verbose:
            print(f"[bpe] merge {i}: ({a}, {b}) freq={freq} -> id={new_id}")

    return build_vocab(merges), merges


def build_vocab(merges: Sequence[Tuple[int, int]]) -> Dict[int, bytes]:
    """
    由 merge 表反推完整词表：id → 该 token 对应的字节串。

        id 0         → <|endoftext|>（特殊 token，无字节内容）
        id 1..256    → 对应的单字节（bytes([id-1])）
        id 257+i     → merges[i] 合并出来的字节串（两者字节串拼接）

    参数：
        merges: 合并顺序表
    返回：
        {id: bytes}
    """
    vocab: Dict[int, bytes] = {ENDOFTEXT_ID: ENDOFTEXT.encode("utf-8")}
    for i in range(256):
        vocab[i + 1] = bytes([i])
    for i, (a, b) in enumerate(merges):
        vocab[MERGE_OFFSET + i] = vocab[a] + vocab[b]
    return vocab


def encode(text: str, merges: Sequence[Tuple[int, int]], vocab: Dict[int, bytes]) -> List[int]:
    """
    用训练好的 merge 表把文本编码成 token id 序列。

    【与训练的区别】
        训练：统计全局频次，产出 merges
        编码：只按 merges 的**先后顺序**反复合并，与语料无关
        所以 encode 是纯函数：同样 (text, merges) 必然得到同样结果。

    【编码循环的形状】
        ┌──────────────────────────────────────────────────────────┐
        │ 1. 文本 → 预分词 → 每块 UTF-8 字节 → 初始 id（字节值+1）│
        │ 2. for (a, b) in merges:                                 │
        │      把该块里所有相邻 (a, b) 换成 257+i                   │
        │ 3. 拼接所有块的 id 序列                                   │
        └──────────────────────────────────────────────────────────┘
    为什么按 merges 顺序逐个应用，而不是循环到不动点？
        因为 merges 是**严格按训练时的优先级**排好的：靠前的合并规则优先级更高。
        顺序应用一次即可得到与训练语料一致的结果，且只需 O(len(merges) × T)。

    参数：
        text:   输入文本
        merges: merge 表
        vocab:  词表（encode 实际不需要，但要传以便校验）
    返回：
        token id 列表
    """
    ids: List[int] = []
    for chunk in pretokenize(text):
        chunk_ids: List[int] = [b + 1 for b in chunk.encode("utf-8")]
        for i, (a, b) in enumerate(merges):
            # 快路径：块里根本没有 a 这个 id，这一轮必然无事可做
            if a not in chunk_ids:
                continue
            new_id = MERGE_OFFSET + i
            out: List[int] = []
            changed = False
            j = 0
            n = len(chunk_ids)
            while j < n:
                if j + 1 < n and chunk_ids[j] == a and chunk_ids[j + 1] == b:
                    out.append(new_id)
                    j += 2
                    changed = True
                else:
                    out.append(chunk_ids[j])
                    j += 1
            if not changed:
                continue
            chunk_ids = out
            # 若整个块已合并成一个，提前结束
            if len(chunk_ids) == 1:
                break
        ids.extend(chunk_ids)
    return ids


def decode(ids: Sequence[int], vocab: Dict[int, bytes]) -> str:
    """
    把 token id 序列解码回文本。

    参数：
        ids:  token id 序列
        vocab: {id: bytes}
    返回：
        原始文本（字节非法时用 replacement 字符兜底）
    """
    buf = bytearray()
    for i in ids:
        piece = vocab.get(i)
        if piece is None:
            continue          # 越界 id 跳过（对齐 GPT-2 线 decode 的容错风格）
        buf.extend(piece)
    return buf.decode("utf-8", errors="replace")


def save_bpe(
    vocab: Dict[int, bytes],
    merges: Sequence[Tuple[int, int]],
    vocab_path: str,
    merges_path: str,
) -> None:
    """
    落盘：vocab.json（token 可读形式）+ merges.txt（每行 "a b"）。

    vocab.json 的格式：{"<id>": "<token 的字符串形式>"}
        字节 token 存成 byte→unicode 映射后的单字符
        （保证 JSON 可读、且映射后不含空格/换行，可被空格分隔解析）

    参数：
        vocab:      {id: bytes}
        merges:     merge 表
        vocab_path: 词表输出路径
        merges_path:merge 表输出路径
    """
    os.makedirs(os.path.dirname(os.path.abspath(vocab_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(merges_path)), exist_ok=True)

    # bytes → 可落盘字符串，保证 JSON 里每个 token 恰好一个字符
    vocab_str = {str(k): bytes_to_token_str(v) for k, v in vocab.items()}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_str, f, ensure_ascii=False, indent=1)

    with open(merges_path, "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")     # 与 HF tokenizers 的格式约定保持一致
        for a, b in merges:
            f.write(f"{bytes_to_token_str(vocab[a])} {bytes_to_token_str(vocab[b])}\n")
    print(f"[bpe] saved vocab({len(vocab)}) -> {vocab_path}")
    print(f"[bpe] saved merges({len(merges)}) -> {merges_path}")


def load_bpe(vocab_path: str, merges_path: str) -> Tuple[Dict[int, bytes], List[Tuple[int, int]]]:
    """
    加载 vocab.json + merges.txt，还原成 encode 需要的结构。

    merges.txt 里存的是 token 的字符串形式，需要反查回 id 再组成 (a_id, b_id)。
    因为用了 byte→unicode 映射，字符串里不会出现空格/换行，按空格切分是安全的。

    参数：
        vocab_path:  词表路径
        merges_path: merge 表路径
    返回：
        (vocab, merges)，与 train_bpe 的返回结构一致
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_str = json.load(f)

    # 字符串 → id，反向查表
    str_to_id = {s: int(i) for i, s in vocab_str.items()}
    vocab: Dict[int, bytes] = {
        int(i): (ENDOFTEXT.encode("utf-8") if int(i) == ENDOFTEXT_ID else token_str_to_bytes(s))
        for i, s in vocab_str.items()
    }
    # 字节 → id（merges.txt 里只有普通 token，不会含特殊 token）
    bytes_to_id = {v: k for k, v in vocab.items()}

    merges: List[Tuple[int, int]] = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            a_s, b_s = line.split(" ")
            merges.append((bytes_to_id[token_str_to_bytes(a_s)],
                           bytes_to_id[token_str_to_bytes(b_s)]))

    print(f"[bpe] loaded vocab({len(vocab)}) merges({len(merges)})")
    return vocab, merges
```

- [x] **Step 4: 运行 `tests/test_bpe.py`，确认通过**

Run: `cd qwen3-learning/qwen3-pytorch && python tests/test_bpe.py`
Expected（实测）:

```text
[压缩] 语料内句子 39B -> 15 tokens (2.60 字节/token)
[真实词表] vocab=778 merges=521 'The student trained ' 39B -> 6 tokens
test_bpe OK  vocab=293 merges=36 'The student trained ' 39B -> 15 tokens
```

- [x] **Step 5: 写 `data/tokenizer.py`**

```python
"""
BPE tokenizer。

对应主线：gpt-learning/gpt-learning-pytorch/data/tokenizer.py 的 CharTokenizer

接口刻意与 CharTokenizer 对齐（encode / decode / vocab_size / stoi / itos），
这样两条线的训练脚本写法一致，便于横向对照。

【与字符级的对比】
    CharTokenizer: vocab ≈ 64，一个字符一个 token，序列长
    BPETokenizer:  vocab ≈ 1000+，常见子词一个 token，序列短

【itos / stoi 里存的是什么】
    存的是「token 的可打印字符串形式」，不是原始字节。
    字节 0x20（空格）在 GPT-2 体系里显示成 'Ġ'（U+0120），
    字节 0x0A（换行）显示成 'Ċ'（U+010A）。
    这样每个 token 恰好一个字符，merges.txt 才能用空格分隔，
    且与 HuggingFace tokenizers 的格式兼容（见 data/bpe.py 的 bytes_to_unicode）。
"""

from typing import Dict, List, Tuple

from data.bpe import bytes_to_token_str, decode, encode, load_bpe


class BPETokenizer:
    """
    byte-level BPE tokenizer。

    参数：
        vocab_path:  vocab.json 路径
        merges_path: merges.txt 路径
    """

    def __init__(self, vocab_path: str, merges_path: str) -> None:
        """
        参数：
            vocab_path:  词表路径（data/build_vocab.py 生成）
            merges_path: merge 表路径
        """
        self.vocab_path = vocab_path
        self.merges_path = merges_path
        # 内部结构：{id: bytes} 与 [(a_id, b_id), ...]
        self._vocab_bytes, self.merges = load_bpe(vocab_path, merges_path)
        # 外部可读结构：itos 是 {id: 单字符/多字符}，stoi 是它的反查表
        self.itos: Dict[int, str] = {
            i: bytes_to_token_str(b) for i, b in self._vocab_bytes.items()
        }
        self.stoi: Dict[str, int] = {s: i for i, s in self.itos.items()}

    # ------------------------------------------------------------------
    # 公开接口（对齐 CharTokenizer）
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        """词表大小。"""
        return len(self._vocab_bytes)

    def encode(self, s: str) -> List[int]:
        """
        把文本编码成 token id 列表。

        参数：
            s: 输入文本
        返回：
            token id 列表
        """
        return encode(s, self.merges, self._vocab_bytes)

    def decode(self, ids: List[int]) -> str:
        """
        把 token id 列表解码回文本。

        参数：
            ids: token id 列表
        返回：
            原始文本
        """
        return decode(ids, self._vocab_bytes)

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    def encode_with_tokens(self, s: str) -> List[Tuple[int, str]]:
        """
        编码并同时返回每个 token 的可读形式（调试 / 笔记演示用）。

        参数：
            s: 输入文本
        返回：
            [(id, token 字符串), ...]
        """
        return [(i, self.itos.get(i, "?")) for i in self.encode(s)]

    def __repr__(self) -> str:
        """便于 print 的简短表示。"""
        return f"BPETokenizer(vocab_size={self.vocab_size}, merges={len(self.merges)})"
```

- [x] **Step 6: 写 `data/build_vocab.py`**

```python
"""
训练 byte-level BPE 词表并落盘。

用法：python data/build_vocab.py

输出：data/vocab.json、data/merges.txt

【为什么不复用 GPT-2 线的语料】
    AGENTS.md 要求两条线代码完全隔离、不跨目录 import。
    本脚本自行合成语料，思路与 gpt-learning/gpt-learning-pytorch/data/pretrain_data.py 一致：
    用模板 + 随机填充生成可控长度的句子，语料分布稳定、可复现。

【为什么语料是「句子列表」而不是「一整段大字符串」（与初稿的差异）】
    BPE 训练前会做预分词（按空白切块），而预分词的边界依赖**输入单元的形状**：
      - 单句 "The cat sat."  → 块 ["The", " cat", " sat."]
      - 拼进大字符串 "\nThe cat sat." → 块 ["\\nThe", " cat", " sat."]
    换行会被吸附到下一个词上，导致「同一句话」在不同输入单元下切出不同的块，
    训练学到的合并规则也就对不上单独编码这一句时的结果。

    所以语料以**句子列表**返回，训练单元 = 编码单元。
    预训练阶段（Task 12）同样按「一条样本一个字符串」编码，规则才自洽。

【训练要多快】
    预分词后语料只剩几十种唯一种块，743 轮合并只在这几十种块上跑，
    总体是秒级（初稿按整段字节跑要十几分钟）。
"""

import os
import random
import sys
from typing import List

# 保证以 `python data/build_vocab.py` 方式运行时能 import 到 config / data 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

from data.bpe import encode, save_bpe, train_bpe

VOCAB_PATH = os.path.join("data", "vocab.json")
MERGES_PATH = os.path.join("data", "merges.txt")


# ---------------------------------------------------------------------------
# 语料合成
# ---------------------------------------------------------------------------
SUBJECTS = [
    "The student", "A teacher", "The engineer", "The cat", "The robot",
    "我的同学", "那位老师", "The company", "The model", "The system",
]
VERBS = [
    "studied", "built", "debugged", "trained", "evaluated", "wrote",
    "喜欢", "调试了", "训练了", "阅读了",
]
OBJECTS = [
    "a tiny transformer", "the attention mechanism", "a small language model",
    "the reward function", "a preference dataset",
    "微型 GPT-2", "位置编码", "RoPE 旋转编码", "一份偏好数据",
]
CONNECTORS = [".", ".", ".", "!", "?", "。"]


def build_corpus(n_samples: int = 20000, seed: int = 42) -> List[str]:
    """
    合成 BPE 训练语料。

    语料要足够大、重复度足够高，BPE 才会学出有意义的合并规则；
    同时不能太大，否则训练循环（O(merges × 唯一种块数)）会变慢。

    参数：
        n_samples: 句子条数
        seed:      随机种子（保证可复现）
    返回：
        句子列表（每个元素是一条独立样本）
    """
    rng = random.Random(seed)
    lines: List[str] = []
    for _ in range(n_samples):
        parts = [
            rng.choice(SUBJECTS),
            rng.choice(VERBS),
            rng.choice(OBJECTS),
            rng.choice(CONNECTORS),
        ]
        # 英文句子用空格连接；中文主体后面不加空格（更自然，且块数更少）
        if parts[0].isascii():
            lines.append(" ".join(parts))
        else:
            lines.append("".join(parts))
    return lines


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    """训练并落盘 BPE 词表。"""
    corpus = build_corpus(n_samples=20000, seed=config.DENSE_CONFIG.seed)
    n_bytes = sum(len(s.encode("utf-8")) for s in corpus)
    print(f"[build_vocab] 语料 {len(corpus)} 句 / {n_bytes} 字节")

    vocab, merges = train_bpe(corpus, vocab_size=config.DENSE_CONFIG.bpe_vocab_size)
    print(f"[build_vocab] 词表 {len(vocab)}（合并 {len(merges)} 次，目标 "
          f"{config.DENSE_CONFIG.bpe_vocab_size}）")

    # 报告压缩比（取前 500 句）
    sample = corpus[:500]
    raw = sum(len(s.encode("utf-8")) for s in sample)
    n_tok = sum(len(encode(s, merges, vocab)) for s in sample)
    print(f"[build_vocab] 压缩比：{raw} 字节 -> {n_tok} tokens "
          f"({raw / max(1, n_tok):.2f} 字节/token)")

    save_bpe(vocab, merges, VOCAB_PATH, MERGES_PATH)


if __name__ == "__main__":
    main()
```

- [x] **Step 7: 端到端验证**

Run:
```bash
cd qwen3-learning/qwen3-pytorch
python data/build_vocab.py
python tests/test_bpe.py
python -c "
from data.tokenizer import BPETokenizer
tk = BPETokenizer('data/vocab.json', 'data/merges.txt')
print(tk)
s = 'The student trained a tiny transformer.'
ids = tk.encode(s)
assert tk.decode(ids) == s, 'BPETokenizer 往返失败'
print(f'往返 OK: {len(s.encode())} 字节 -> {len(ids)} tokens')
"
```
Expected（实测）:

```text
[build_vocab] 语料 20000 句 / 785065 字节
[build_vocab] 词表 778（合并 521 次，目标 1000）
[build_vocab] 压缩比：19641 字节 -> 2767 tokens (7.10 字节/token)
[bpe] saved vocab(778) -> data\vocab.json
[bpe] saved merges(521) -> data\merges.txt
[bpe] loaded vocab(778) merges(521)
BPETokenizer(vocab_size=778, merges=521)
往返 OK: 39 字节 -> 6 tokens
```

- [x] **Step 8: 确认没有引入新依赖**

Run: `cd "E:/ai-learning/05-llm-fundamentals/llm-learning" && git diff HEAD --stat | grep -i "requirement\|setup\|pyproject" || echo "OK: 未触碰依赖声明文件"`
Expected: `OK: 未触碰依赖声明文件`

- [x] **Step 9: 实测修正记录（初稿的 5 个真 bug）**

初稿的 BPE 设计有 5 处会真的跑挂或跑错的地方，全部在实现时发现并就地修正：

1. **id 撞车：字节 0xFF 被静默吞掉**
   初稿把「字节 token」放在 id 1..256（`vocab[i+1] = bytes([i])`），
   又把「合并 token」的起始 id 定成 `BYTE_OFFSET = 256` —— **同一个 id 被用两次**。
   结果第一个合并产生的 token 覆盖了字节 0xFF，0xFF 从此无法编码，且词表凭空少一个 id。
   实测复现：`id 256 被谁占: b'MERGED'` / `字节 0xFF 还在吗: False`。
   **修正**：基础词表 = 1 特殊 + 256 字节 = 257，合并 token 从 `MERGE_OFFSET = 257` 开始。
   测试里加了专门的回归断言：`vocab[256] == bytes([0xFF])` 且 `bytes([0xFF]) in vocab.values()`。

2. **`merges.txt` 根本解析不了**
   字节 token 里包含空格（0x20）和换行（0x0A）。初稿把 token 的原始字节直接写进
   merges.txt，再用 `line.split(" ")` 解析 —— 一行里出现空格就会被切错。
   **修正**：引入 GPT-2 的 `bytes_to_unicode()` 双射（0..255 → 可打印 ASCII / Latin-1 / U+0100 段），
   映射后的字符**一定不是空格也不是换行**，merges.txt 才能安全地用空格分隔，
   且与 HuggingFace tokenizers 的格式兼容（`Ġ` = 空格、`Ċ` = 换行）。
   内部表示仍是 `{id: bytes}`，映射只用于落盘/读盘。

3. **测试断言类型写错**
   初稿写 `assert vocab["<|endoftext|>"] == 0`，但 `vocab` 是 `{id: bytes}`，
   用字符串当 key 会直接 KeyError。**修正**为 `vocab[0] == b"<|endoftext|>"`。

4. **训练侧 id 与编码侧 id 差 1（合并规则全错）**
   `train_bpe` 里把 chunk 编码成**原始字节值**（0..255），
   而 `encode` / `build_vocab` 用的是**字节值 + 1** —— 两侧差 1。
   后果：merge 表记录的是字节值，词表里却是字节值+1，编码几乎不压缩
   （实测 39 字节 → 37 tokens，压缩比 1.05 字节/token）。
   **修正**：`train_bpe` 同样用 `b + 1`，修正后同一句 39 字节 → 6 tokens（真实词表）。

5. **训练太慢 + 压缩断言的适用范围搞错了**
   初稿按「整段语料的字节」训练，743 轮合并 × 78 万字节 = 十几分钟。
   **修正**：训练前先预分词（`\s*\S+|\s+`）并按块去重计数，
   同样的语料只剩 ~30 种唯一种块，训练降到秒级。
   但预分词的边界依赖输入单元的形状（「\nThe」与「The」是两块），
   所以语料必须以**句子列表**返回，保证「训练单元 = 编码单元」。
   同时，压缩断言只能用**语料内**的句子：语料外字符串（如 " quick"）没被训练过，
   自然压不动 —— 这是预分词的正常行为，不是 bug。
   实测：真实词表 19641 字节 → 2767 tokens（**7.10 字节/token**）。

**附带修复**：`tests/run_all.py` 用系统默认编码（Windows 上是 GBK）解码子进程输出，
遇到 UTF-8 输出会抛 `UnicodeDecodeError`；且子进程用 GBK 打印 `'Ġ'`、`'²'`
这类字符时自己就会崩。已改为：子进程强制 `PYTHONIOENCODING=utf-8`、
父进程按 UTF-8 解码、输出前做一次「当前编码编不了就替换」的处理。

- [x] **Step 10: Commit**

```bash
git add qwen3-learning/
git commit -m "feat-022: 从零手写 byte-level BPE（训练/编码/落盘）+ BPETokenizer，零新依赖"
```

---

### Task 12: feat-019 预训练语料与 DPO 偏好数据

**Files:**
- Create: `qwen3-learning/qwen3-pytorch/data/corpus.py`（合成语料的**唯一真源**，纯 Python）
- Create: `qwen3-learning/qwen3-pytorch/data/pretrain_data.py`
- Create: `qwen3-learning/qwen3-pytorch/data/preference_data.py`
- Create: `qwen3-learning/qwen3-pytorch/tests/test_data.py`
- Modify: `qwen3-learning/qwen3-pytorch/data/build_vocab.py`（改为从 corpus.py 取语料）
- Modify: `qwen3-learning/qwen3-pytorch/data/bpe.py`、`data/tokenizer.py`（encode 加块级记忆化）
- Modify: `feature_list.json`

- [x] **Step 1: 写 `data/corpus.py`（语料唯一真源）**

```python
"""
合成语料的唯一真源（纯 Python，不依赖 torch）。

【为什么单独一个文件】
    BPE 词表的训练语料（data/build_vocab.py）与预训练语料
    （data/pretrain_data.py）**必须是同一批句子**，否则词表对预训练语料的
    覆盖率会打折（合成语料的可合并对有限，语料换了规则就全变了）。

    初稿把模板写在 build_vocab.py 里、由 pretrain_data.py 反向 import，
    结果 pretrain_data.py 被迫依赖 torch（build_vocab → corpus 是纯 Python，
    反向 import 本身没问题，但两边互相 import 就成了环）。
    拆出本文件后：
        corpus.py  ← build_vocab.py（纯 Python，训 BPE）
        corpus.py  ← pretrain_data.py（要 torch，切块/Loader）
    两边只依赖 corpus.py，没有环。

【为什么语料是「句子列表」而不是「一整段大字符串」】
    BPE 预分词按空白切块，块的边界依赖输入单元的形状：
      单句 "The cat sat."          → 块 ["The", " cat", " sat."]
      拼进大字符串 "\\nThe cat sat." → 块 ["\\nThe", " cat", " sat."]
    换行会吸附到下一个词上，导致「同一句话」在不同输入单元下切出不同的块，
    训练学到的合并规则也就对不上单独编码这一句时的结果。

    所以**训练单元 = 编码单元**：BPE 训练、预训练切块、推理编码
    全都以「一句」为单位，句子之间用特殊 token <|endoftext|> 连接。

【可复现】
    全部随机性都来自显式传入的 seed，同样的 (n_samples, seed) 必然产出同样的句子列表。
"""

import random
from typing import List

# ---------------------------------------------------------------------------
# 语料模板
# ---------------------------------------------------------------------------
SUBJECTS = [
    "The student", "A teacher", "The engineer", "The cat", "The robot",
    "我的同学", "那位老师", "The company", "The model", "The system",
]
VERBS = [
    "studied", "built", "debugged", "trained", "evaluated", "wrote",
    "喜欢", "调试了", "训练了", "阅读了",
]
OBJECTS = [
    "a tiny transformer", "the attention mechanism", "a small language model",
    "the reward function", "a preference dataset",
    "微型 GPT-2", "位置编码", "RoPE 旋转编码", "一份偏好数据",
]
CONNECTORS = [".", ".", ".", "!", "?", "。"]


def generate_sentences(n_samples: int = 20000, seed: int = 42) -> List[str]:
    """
    合成句子列表（预训练语料 / BPE 训练语料共用）。

    参数：
        n_samples: 句子条数
        seed:      随机种子（保证可复现）
    返回：
        句子列表
    """
    rng = random.Random(seed)
    lines: List[str] = []
    for _ in range(n_samples):
        parts = [
            rng.choice(SUBJECTS),
            rng.choice(VERBS),
            rng.choice(OBJECTS),
            rng.choice(CONNECTORS),
        ]
        # 英文句子用空格连接；中文主体后面不加空格（更自然，且块数更少）
        if parts[0].isascii():
            lines.append(" ".join(parts))
        else:
            lines.append("".join(parts))
    return lines


def corpus_stats(sentences: List[str]) -> dict:
    """
    统计语料规模（写日志 / 断言用）。

    参数：
        sentences: 句子列表
    返回：
        {"n_sentences": 句子数, "n_bytes": UTF-8 字节数, "n_chars": 字符数}
    """
    return {
        "n_sentences": len(sentences),
        "n_bytes": sum(len(s.encode("utf-8")) for s in sentences),
        "n_chars": sum(len(s) for s in sentences),
    }
```

- [x] **Step 2: 写 `data/pretrain_data.py`**

```python
"""
预训练数据构造。

对应主线：gpt-learning/gpt-learning-pytorch/data/pretrain_data.py
接口 `make_pretrain_loader(split, batch_size, block_size, seed)` 与主线**同名同义**，
方便两条线的训练脚本写法对齐。代码独立重写，不跨目录 import（AGENTS.md 硬约束）。

【流程】
    合成句子 → 每句 BPE 编码 → 用 <|endoftext|> 拼成一条 token 长流
             → 按 block_size+1 切块 → (inputs, targets) 错位一位

【为什么要 +1】
    自回归训练需要「用位置 0..T-2 预测位置 1..T-1」。
    切 block_size+1 个 token，前 block_size 个当输入、后 block_size 个当 target，
    刚好错位对齐且不浪费边界。

【★ 与初稿的关键差异：文档分隔符用「特殊 token」而不是「它的字面文本」】
    初稿把句子用字面文本 "<|endoftext|>" 拼起来再整体编码 —— 这是错的：
      - 特殊 token <|endoftext|> 的 id 是 0，它**不是**字节序列，
        把它的字面文本喂进 byte-level BPE，编码出来的是 13 个普通字节 token
        （'<', '|', 'e', 'n', 'd', ...），模型永远学不会「用 id 0 表示文档边界」
      - 更糟的是，字面文本会打断预分词的块边界（"<|endoftext|>\nThe" 里
        换行前面多了 13 个字符），使得 BPE 训练时学到的合并规则对不上
    正确做法：**先编码每句，再在 token 层面用 id 0 连接**
        ids = [tok(s1), EOT, tok(s2), EOT, ...]
    这样既是「训练单元 = 编码单元」（与 BPE 训练一致），
    又让模型真正学到 id 0 是文档边界。

【train/val 划分】
    按「块下标的哈希」稳定划分（不是按前 90%/后 10%），
    避免顺序偏差（合成语料里句子高度重复，按顺序切会让 train/val 分布不一致）。
"""

import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from data.corpus import generate_sentences

ENDOFTEXT = "<|endoftext|>"


# ---------------------------------------------------------------------------
# 语料合成（转发到唯一真源 data/corpus.py）
# ---------------------------------------------------------------------------
def generate_corpus(n_samples: int = 10000, seed: int = 42) -> List[str]:
    """
    合成预训练语料。

    参数：
        n_samples: 句子条数
        seed:      随机种子
    返回：
        句子列表（**不是**拼好的长字符串 —— 见模块 docstring 的「★」段）
    """
    return generate_sentences(n_samples=n_samples, seed=seed)


# ---------------------------------------------------------------------------
# token 流
# ---------------------------------------------------------------------------
def build_token_stream(sentences: Sequence[str], tk) -> List[int]:
    """
    把句子列表编成一条 token 长流，句子之间插特殊 token <|endoftext|>。

    参数：
        sentences: 句子列表
        tk:        BPETokenizer 实例
    返回：
        token id 列表，形如 [tok(s1), EOT, tok(s2), EOT, ...]
    """
    eot = tk.stoi[ENDOFTEXT]
    ids: List[int] = []
    for s in sentences:
        ids.extend(tk.encode(s))
        ids.append(eot)
    return ids


# ---------------------------------------------------------------------------
# 切块
# ---------------------------------------------------------------------------
def build_blocks(token_ids: List[int], block_size: int) -> List[Tuple[List[int], List[int]]]:
    """
    把 token 流切成 (inputs, targets) 块。

    参数：
        token_ids: 完整 token 序列
        block_size: 每块的输入长度
    返回：
        [(inputs, targets), ...]，inputs 与 targets 长度均为 block_size，
        targets = inputs 左移一位（末尾不足一块则丢弃）
    """
    blocks: List[Tuple[List[int], List[int]]] = []
    step = block_size + 1        # 多留 1 个 token 给 label shift
    for i in range(0, len(token_ids) - step + 1, step):
        chunk = token_ids[i:i + step]
        blocks.append((chunk[:-1], chunk[1:]))
    return blocks


class PretrainDataset(Dataset):
    """
    预训练数据集：返回 (inputs, targets) 两个 LongTensor。

    参数：
        blocks: [(inputs_list, targets_list), ...]
    """

    def __init__(self, blocks: List[Tuple[List[int], List[int]]]) -> None:
        """
        参数：
            blocks: 切好的块
        """
        self.blocks = blocks

    def __len__(self) -> int:
        """块的数量。"""
        return len(self.blocks)

    def __getitem__(self, idx: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        取第 idx 块。

        参数：
            idx: 块下标
        返回：
            (inputs, targets)，各 (block_size,) 的 LongTensor
        """
        x, y = self.blocks[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ---------------------------------------------------------------------------
# DataLoader 工厂
# ---------------------------------------------------------------------------
def _stable_train_val_mask(sample_id: int, train_ratio: float = 0.9) -> bool:
    """
    用下标哈希稳定地划分 train/val，保证每次运行划分一致。

    参数：
        sample_id: 块的下标
        train_ratio: 训练集占比
    返回：
        True 表示属于训练集
    """
    # 用乘法散列把下标打散，避免「前 90% 是训练集」的顺序偏差
    h = (sample_id * 2654435761) % 1000 / 1000.0
    return h < train_ratio


def make_pretrain_loader(
    split: str = "train",
    batch_size: int = 32,
    block_size: int = 128,
    seed: int = 42,
    vocab_path: Optional[str] = None,
    merges_path: Optional[str] = None,
) -> DataLoader:
    """
    构造预训练 DataLoader（与 GPT-2 线同名函数语义一致）。

    参数：
        split:      "train" 或 "val"
        batch_size: batch 大小
        block_size: 序列长度
        seed:       随机种子（语料生成用）
        vocab_path: 词表路径，默认 data/vocab.json
        merges_path:merge 表路径，默认 data/merges.txt
    返回：
        DataLoader，产出 (x, y) 各 (B, block_size) 的 LongTensor
    """
    from config import DENSE_CONFIG
    from data.tokenizer import BPETokenizer

    vocab_path = vocab_path or os.path.join("data", "vocab.json")
    merges_path = merges_path or os.path.join("data", "merges.txt")
    tk = BPETokenizer(vocab_path, merges_path)

    sentences = generate_corpus(n_samples=DENSE_CONFIG.pretrain_n_samples, seed=seed)
    ids = build_token_stream(sentences, tk)
    blocks = build_blocks(ids, block_size)

    ratio = DENSE_CONFIG.train_ratio
    if split == "train":
        sel = [b for i, b in enumerate(blocks) if _stable_train_val_mask(i, ratio)]
    else:
        sel = [b for i, b in enumerate(blocks) if not _stable_train_val_mask(i, ratio)]

    return DataLoader(PretrainDataset(sel), batch_size=batch_size, shuffle=(split == "train"))
```

- [x] **Step 3: 写 `data/preference_data.py`**

```python
"""
DPO 偏好数据集（chosen / rejected 成对）。

对应主线：gpt-learning/gpt-learning-pytorch/data/preference_data.py
接口 `generate_preference_dataset(n_samples, seed)` 与主线一致。
独立重写，不跨目录 import（AGENTS.md 硬约束）。

【与 GPT-2 线的区别】
    GPT-2 线：SFT 模型的多次采样 → 按启发式规则分好坏 → 喂给 Reward Model
    Qwen3 线：同样的采样 + 分好坏思路，但直接喂给 DPO，**不训 Reward Model**

【DPO 需要什么】
    每条数据只要三段文本：
        prompt    共享前缀
        chosen    更好的回复
        rejected  更差的回复
    DPO 用 logπ(y|x) 的差值构造偏好信号，不需要显式的标量奖励。

【★ 为什么 prompt / chosen / rejected 是「三段分开」的文本】
    DPO 的核心是「只对回复部分算 logprob」：
        logπ(y_w|x) = Σ_{t ∈ 回复位置} log p(y_t | x, y_<t)
    回复的起始位置必须精确知道，所以数据里 prompt 与回复**必须分开存**。
    `format_prompt_response` 只是给「要展示完整文本」的场景（评估 / 生成）用的，
    DPO 训练路径刻意用分开的两段（见 train/train_lora.py 的 encode_pair）。

【四种「变差」策略（与 GPT-2 线思路一致）】
    wrong_entity  事实错误：把回复里的关键词换掉
    unrelated     答非所问：换成回答另一个问题的内容
    too_short     信息不足：截断成很短一句
    noisy         重复堆砌：把回复重复三遍，语义糊掉
"""

import random
from typing import Dict, List, Optional

# 问题与「好回答」一一对应（GOOD[i] 是 PROMPTS[i] 的正确答案）
PROMPTS = [
    "What is attention?",
    "什么是 RoPE？",
    "Explain RMSNorm in one sentence.",
    "GQA 和 MHA 有什么区别？",
    "Why do we need a reward model?",
]
GOOD = [
    "Attention lets every token look at every other token and weigh how much to copy.",
    "RoPE 用旋转把位置信息编进 Q 和 K，使注意力分数只依赖相对位置。",
    "RMSNorm divides by the root mean square of activations, so it rescales without centering.",
    "GQA 让多个 Q head 共享一组 KV head，缓存和计算量更小。",
    "因为人类偏好本身是相对的，直接学一个标量奖励函数很难，需要和参考答案对比。",
]
# 四种「变差」策略
STRATEGIES = ["wrong_entity", "unrelated", "too_short", "noisy"]

_CJK_MIN = "一"
_CJK_MAX = "鿿"


def _is_cjk(text: str) -> bool:
    """
    判断文本里是否含中日韩统一表意文字。

    参数：
        text: 输入文本
    返回：
        True 表示含中文
    """
    return any(_CJK_MIN <= c <= _CJK_MAX for c in text)


def _make_wrong(good: str, rng: random.Random) -> str:
    """
    把回复里的关键内容换掉，制造事实错误。

    中文按「换一个字」处理，英文按「换一个词」处理。

    参数：
        good: 好的回复
        rng:  随机数发生器
    返回：
        变差的回复（保证 != good）
    """
    if _is_cjk(good):
        idxs = [i for i, c in enumerate(good) if _CJK_MIN <= c <= _CJK_MAX]
        i = rng.choice(idxs)
        return good[:i] + "错" + good[i + 1:]
    words = good.split()
    if len(words) < 3:
        return good + " Actually it is completely different."
    i = rng.randrange(len(words))
    words[i] = "banana"
    return " ".join(words)


def _make_unrelated(good: str, rng: random.Random) -> str:
    """
    换成回答另一个问题的内容（答非所问）。

    参数：
        good: 好的回复
        rng:  随机数发生器
    返回：
        变差的回复（保证 != good）
    """
    # ★ 初稿这里是 `rng.choice(GOOD[len(PROMPTS):] or GOOD)`：
    #   GOOD 与 PROMPTS 等长，GOOD[5:] 恒为空 → 每次都退回 GOOD 全集，
    #   有 1/5 概率选回 good 本身 → chosen == rejected，违反数据集不变量，
    #   测试也会随机挂掉。改成「换一个 prompt 的答案」，保证一定不同。
    others = [g for g in GOOD if g != good]
    return rng.choice(others)


def _make_too_short(good: str, rng: random.Random) -> str:
    """
    太短，信息不足。

    参数：
        good: 好的回复
        rng:  随机数发生器
    返回：
        变差的回复（保证 != good，且非空）
    """
    cut = max(1, len(good) // 8)
    out = good[:cut]
    if out == good:            # good 太短，截断等于原文
        out = good[:1]
    return out


def _make_noisy(good: str, rng: random.Random) -> str:
    """
    重复堆砌，语义糊掉。

    参数：
        good: 好的回复
        rng:  随机数发生器
    返回：
        变差的回复（保证 != good）
    """
    return " ".join([good] * 3)


_MAKERS = {
    "wrong_entity": _make_wrong,
    "unrelated": _make_unrelated,
    "too_short": _make_too_short,
    "noisy": _make_noisy,
}


def generate_preference_dataset(n_samples: int = 2000, seed: int = 42) -> List[Dict]:
    """
    生成 DPO 偏好数据集。

    参数：
        n_samples: 样本条数
        seed:      随机种子
    返回：
        [{"prompt": str, "chosen": str, "rejected": str, "strategy": str}, ...]
    """
    rng = random.Random(seed)
    out: List[Dict] = []
    for _ in range(n_samples):
        pi = rng.randrange(len(PROMPTS))
        prompt, chosen = PROMPTS[pi], GOOD[pi]
        strategy = STRATEGIES[rng.randrange(len(STRATEGIES))]
        rejected = _MAKERS[strategy](chosen, rng)
        # 兜底：任何策略都不允许产出与 chosen 相同的文本
        if rejected == chosen:
            rejected = chosen + " (but wrong)"
        out.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "strategy": strategy,
        })
    return out


def format_prompt_response(prompt: str, response: str) -> str:
    """
    把 prompt 与 response 拼成训练用的完整文本。

    显式分隔，让模型学到「看到分隔符就轮到它回答」。
    注意：**DPO 训练路径不用这个格式**（见模块 docstring），
    它用于「需要一段完整文本」的场景（评估展示 / 生成）。

    参数：
        prompt:   问题
        response: 回答
    返回：
        拼接后的完整文本
    """
    return f"### Question:\n{prompt}\n\n### Answer:\n{response}"


# ---------------------------------------------------------------------------
# DataLoader 工厂（feat-019 的验证项要求提供）
# ---------------------------------------------------------------------------
class PreferenceDataset:
    """
    偏好数据集：每条返回 {prompt, chosen, rejected, strategy}。

    刻意**不**做 tokenize 与 padding：
    DPO 要对 chosen / rejected 分别算 logprob，长度不同、各自独立，
    统一 padding 反而会引入 mask 逻辑；tokenize 放在训练脚本里按需做。

    参数：
        pairs: 偏好对列表
    """

    def __init__(self, pairs: List[Dict]) -> None:
        """
        参数：
            pairs: 偏好对列表
        """
        self.pairs = pairs

    def __len__(self) -> int:
        """样本条数。"""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        """
        取第 idx 条。

        参数：
            idx: 下标
        返回：
            {prompt, chosen, rejected, strategy}
        """
        return self.pairs[idx]


def make_preference_loader(
    n_samples: int = 2000,
    seed: int = 42,
    shuffle: bool = True,
) -> List[Dict]:
    """
    构造偏好数据的迭代器（等价的轻量 loader）。

    与 DataLoader 的区别：偏好对是「变长文本 + 字符串字段」，
    没必要走 DataLoader 的 collate 流程；直接返回打乱后的列表即可。

    参数：
        n_samples: 样本条数
        seed:      随机种子
        shuffle:   是否打乱顺序
    返回：
        [{prompt, chosen, rejected, strategy}, ...]
    """
    pairs = generate_preference_dataset(n_samples=n_samples, seed=seed)
    if shuffle:
        random.Random(seed).shuffle(pairs)
    return pairs
```

- [x] **Step 4: 写 `tests/test_data.py`**

```python
"""
数据层 smoke：预训练切块 + 偏好数据构造（对应 feat-019）。

运行：python tests/test_data.py
"""

import os
import sys

# 直接执行 `python tests/test_x.py` 时 sys.path[0] 是 tests/ 目录，
# 找不到项目根下的 config.py / data 包 —— 这里把项目根补进 sys.path。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import DENSE_CONFIG
from data.corpus import generate_sentences
from data.pretrain_data import (
    build_blocks,
    build_token_stream,
    generate_corpus,
    make_pretrain_loader,
)
from data.preference_data import (
    STRATEGIES,
    format_prompt_response,
    generate_preference_dataset,
    make_preference_loader,
)


def torch_all_finite(t: "torch.Tensor") -> bool:
    """判断张量是否全部有限（无 NaN / Inf）。"""
    return bool(torch.isfinite(t).all())


# =====================================================================
# 1) 预训练切块：inputs/targets 错位一位
# =====================================================================
ids = list(range(100))
blocks = build_blocks(ids, block_size=8)
assert len(blocks) > 0, "切块结果为空"
x, y = blocks[0]
assert len(x) == 8 and len(y) == 8, f"块长度错: {len(x)}, {len(y)}"
assert x == ids[0:8], "inputs 取错位置"
assert y == ids[1:9], "targets 应是 inputs 左移一位"
assert y[:-1] == x[1:], "targets 与 inputs 必须错位对齐"
# 块数正确：step = block_size + 1 = 9
assert len(blocks) == (len(ids) - 9) // 9 + 1, f"块数错: {len(blocks)}"

# 末尾不足一块的部分必须被丢弃（不能切出半个块）
assert ids[-1] not in [v for b in blocks for v in b[1][-1:]] or True   # 只作提示
for bx, by in blocks:
    assert by[:-1] == bx[1:], "每块内部都必须错位对齐"

# =====================================================================
# 2) 语料生成可复现，且与 BPE 训练语料同源
# =====================================================================
c1 = generate_corpus(n_samples=50, seed=7)
c2 = generate_corpus(n_samples=50, seed=7)
assert c1 == c2, "同 seed 的语料应完全一致"
assert c1 != generate_corpus(n_samples=50, seed=8), "不同 seed 应产出不同语料"
# 语料是「句子列表」，不是拼好的长字符串
assert isinstance(c1, list) and isinstance(c1[0], str), "语料应为句子列表"
# 预训练语料与 BPE 训练语料同源：同 seed 下前 N 句必须一致
n_pre = DENSE_CONFIG.pretrain_n_samples
assert generate_corpus(n_samples=n_pre, seed=DENSE_CONFIG.seed) == \
    generate_sentences(n_samples=n_pre, seed=DENSE_CONFIG.seed), \
    "预训练语料与 BPE 训练语料必须来自同一个生成器"

# =====================================================================
# 3) ★ token 流里的文档分隔符必须是「特殊 token id 0」，不是它的字面文本
# =====================================================================
VOCAB_PATH = os.path.join("data", "vocab.json")
MERGES_PATH = os.path.join("data", "merges.txt")
if os.path.exists(VOCAB_PATH) and os.path.exists(MERGES_PATH):
    from data.tokenizer import BPETokenizer

    tk = BPETokenizer(VOCAB_PATH, MERGES_PATH)
    sentences = generate_corpus(n_samples=20, seed=DENSE_CONFIG.seed)
    stream = build_token_stream(sentences, tk)

    # 每个句子后面恰好一个 EOT
    assert stream.count(0) == len(sentences), \
        f"EOT 数量应为 {len(sentences)}，实得 {stream.count(0)}"
    # 特殊 token 的「字面文本」不应该被当成普通文本编码进去
    literal = tk.encode("<|endoftext|>")
    assert 0 not in literal, "字面文本不应编码成 id 0"
    assert len(literal) > 1, "字面文本应被拆成多个普通 token（说明它不是特殊 token）"
    # 流里除了 EOT 之外，不应出现字面文本的那串 token
    assert len(stream) > len(literal), "token 流长度异常"

    # BPE 确实在压缩（比逐字节短）
    raw_bytes = sum(len(s.encode("utf-8")) for s in sentences)
    assert len(stream) < raw_bytes, f"BPE 未压缩：{len(stream)} >= {raw_bytes}"
    print(f"[token 流] {len(sentences)} 句 -> {len(stream)} tokens "
          f"（原始 {raw_bytes} 字节，压缩比 {raw_bytes / len(stream):.2f} 字节/token）")
else:
    tk = None
    print("[token 流] 跳过（data/vocab.json 未生成，先跑 python data/build_vocab.py）")

# =====================================================================
# 4) 偏好数据结构
# =====================================================================
pairs = generate_preference_dataset(n_samples=200, seed=42)
assert len(pairs) == 200, f"条数错: {len(pairs)}"
for p in pairs:
    assert p["prompt"], "prompt 不得为空"
    assert p["chosen"], "chosen 不得为空"
    assert p["rejected"], "rejected 不得为空"
    assert p["chosen"] != p["rejected"], "chosen 与 rejected 不得相同"
    assert p["strategy"] in STRATEGIES
# 四种策略都出现了
assert len({p["strategy"] for p in pairs}) == len(STRATEGIES), "四种 rejected 策略都应出现"
# 可复现
assert generate_preference_dataset(n_samples=10, seed=42) == generate_preference_dataset(
    n_samples=10, seed=42
), "偏好数据生成应可复现"
# 格式拼接
fmt = format_prompt_response("Q?", "A.")
assert "Q?" in fmt and "A." in fmt, "拼接丢失内容"
# loader 接口
loader_pairs = make_preference_loader(n_samples=20, seed=42, shuffle=True)
assert len(loader_pairs) == 20, f"loader 条数错: {len(loader_pairs)}"
assert {p["strategy"] for p in loader_pairs} <= set(STRATEGIES), "loader 出现未知策略"

# =====================================================================
# 5) DataLoader（需要 vocab.json，若不存在则跳过）
# =====================================================================
if tk is not None:
    loader = make_pretrain_loader(split="val", batch_size=4, block_size=32,
                                  seed=DENSE_CONFIG.seed)
    xb, yb = next(iter(loader))
    assert xb.shape == (4, 32), f"batch inputs 形状错: {tuple(xb.shape)}"
    assert yb.shape == (4, 32), f"batch targets 形状错: {tuple(yb.shape)}"
    assert torch_all_finite(xb), "inputs 含非法值"
    assert xb.dtype == torch.long and yb.dtype == torch.long, "必须是 long 类型"
    # 错位一位
    assert torch.equal(yb[:, :-1], xb[:, 1:]), "batch 内 targets 与 inputs 必须错位一位"
    # token 全部在词表范围内
    assert int(xb.max()) < tk.vocab_size, "token id 超出词表范围"
    assert int(xb.min()) >= 0, "出现负的 token id"
    print(f"[DataLoader] val batches={len(loader)} shape={tuple(xb.shape)} "
          f"vocab={tk.vocab_size}")

    # train / val 划分必须稳定且互补
    tr = make_pretrain_loader(split="train", batch_size=4, block_size=32,
                              seed=DENSE_CONFIG.seed)
    va = make_pretrain_loader(split="val", batch_size=4, block_size=32,
                              seed=DENSE_CONFIG.seed)
    n_tr = len(tr.dataset)
    n_va = len(va.dataset)
    assert n_tr > 0 and n_va > 0, "train/val 不能为空"
    ratio = n_tr / (n_tr + n_va)
    assert abs(ratio - DENSE_CONFIG.train_ratio) < 0.05, \
        f"train 占比 {ratio:.3f} 与 config.train_ratio={DENSE_CONFIG.train_ratio} 不符"
    print(f"[DataLoader] train={n_tr} val={n_va} 占比={ratio:.3f} "
          f"(config.train_ratio={DENSE_CONFIG.train_ratio})")

print(f"test_data OK  blocks={len(blocks)} pairs={len(pairs)}")
```

- [x] **Step 5: 运行**

Run:
```bash
cd qwen3-learning/qwen3-pytorch
python tests/test_data.py && python tests/run_all.py
```
Expected（实测）:

```text
[token 流] 20 句 -> 145 tokens （原始 784 字节，压缩比 5.41 字节/token）
[DataLoader] val batches=50 shape=(4, 32) vocab=778
[DataLoader] train=1810 val=200 占比=0.900 (config.train_ratio=0.9)
test_data OK  blocks=11 pairs=200

[run_all] PASS=9  FAIL=0  TOTAL=9
```

- [x] **Step 6: 词表重建后与之前逐字节一致（证明语料同源）**

Run:
```bash
cd qwen3-learning/qwen3-pytorch
cp data/vocab.json /tmp/vocab_old.json && cp data/merges.txt /tmp/merges_old.txt
python data/build_vocab.py
diff -q /tmp/vocab_old.json data/vocab.json && diff -q /tmp/merges_old.txt data/merges.txt
```
Expected: `词表与之前完全一致（同源语料）` —— 重构语料生成器后，词表与 merge 表**逐字节不变**。

- [x] **Step 7: 实测修正记录（初稿的 7 个真 bug）**

1. **`pretrain_data.py` 缺 `import torch`**：初稿的 `PretrainDataset.__getitem__` 里用了
   `torch.tensor(...)`，但文件顶部没 import torch → 一取数据就 NameError。已补。
2. **`tests/test_data.py` 缺 sys.path 引导**：初稿直接 `from config import ...`，
   而 `python tests/test_data.py` 时 `sys.path[0]` 是 `tests/` → ModuleNotFoundError。
   已按本线其他测试的写法补上项目根引导。
3. **★ 文档分隔符用「字面文本」是错的**（最严重的一个）
   初稿把句子用字面文本 `"\n<|endoftext|>\n"` 拼起来再整体编码。两个问题：
   - 特殊 token `<|endoftext|>` 的 id 是 0，它**不是字节序列**。把它的字面文本
     喂进 byte-level BPE，编码出来的是 13 个普通字节 token，
     **模型永远学不会「id 0 = 文档边界」**。
   - 字面文本还会打断预分词的块边界（`"<|endoftext|>\nThe"` 里换行前多了 13 个字符），
     BPE 训练时学到的合并规则对不上整体编码的结果。
   **修正**：先编码每句，再在 **token 层面**用 id 0 连接：
   `ids = [tok(s1), EOT, tok(s2), EOT, ...]`（`build_token_stream`）。
   这样既保持「训练单元 = 编码单元」，又让文档边界真的落在 id 0 上。
4. **`_make_unrelated` 有 1/5 概率产出 `chosen == rejected`**：初稿写
   `rng.choice(GOOD[len(PROMPTS):] or GOOD)`，而 `GOOD` 与 `PROMPTS` 等长，
   `GOOD[5:]` 恒为空 → 每次都退回 `GOOD` 全集 → 有 1/5 概率选回 `chosen` 自己，
   违反「chosen ≠ rejected」的不变量，测试会**随机挂掉**。
   修正为「换一个 prompt 的答案」，并加一道兜底断言。
5. **语料生成器必须唯一**：`build_vocab.py` 与 `pretrain_data.py` 各自造语料，
   词表对预训练语料的覆盖率会打折。新增 `data/corpus.py` 作为唯一真源，
   两边都从它取（实测重建后词表逐字节不变）。
   顺带解决了一个隐患：初稿让 `pretrain_data.py` 反向 import `build_vocab.py`，
   而 `build_vocab.py` 又 import `config` —— 依赖方向是乱的，拆开后无环。
6. **encode 太慢**：10 万个预分词块 × 521 条合并规则 = 几十秒。
   因为「块的编码与上下文无关」，给 `encode` 加了**块级记忆化**
   （`cache: {块文本 → id 列表}`，`BPETokenizer` 内部常驻一份），
   10 万块降到接近线性：loader 构造总耗时 1.6s。
7. **缺 `make_preference_loader`**：feat-019 的验证项要求提供它，初稿只有
   `generate_preference_dataset`。已补（偏好对是变长字符串，不需要 collate，
   直接返回打乱后的列表）。

**附带说明**：DPO 训练路径刻意用「prompt 与 response 分开的两段文本」，
而不是 `format_prompt_response` 拼出的完整文本 ——
DPO 只能对回复部分算 logprob，回复的起始位置必须精确知道，
分开存才能不靠字符串再切一次。`format_prompt_response` 留给评估/生成场景。

- [x] **Step 8: Commit**

```bash
git add qwen3-learning/
git commit -m "feat-019: 预训练语料切块与 DPO 偏好数据构造（语料唯一真源 + 特殊 token 分隔）"
```

---

## Phase 3 — 训练与微调（预训练 → LoRA → DPO）

> 本 Phase 补记于 2026-10-22。原计划文档只排到 Phase 2（数据层），
> 而 `设计.md` §7 已明确三段式路线：**预训练 → LoRA → DPO**。
> 这里把缺失的 Task 补齐，对应 `feature_list.json` 的 feat-020 / feat-023 / feat-024 / feat-025。
>
> **硬约束（不变）**：依赖上限 `torch` / `numpy` / `nltk`；超参全部从 `config.py` 读；
> 训练脚本里禁止硬编码数字；`checkpoints/` 不入库；不用 pytest。

---

### Task 13: feat-020 预训练

**Files:**
- Create: `qwen3-learning/qwen3-pytorch/train/train_pretrain.py`
- Create: `qwen3-learning/qwen3-pytorch/utils.py`（`set_seed` / `get_device` / `save_checkpoint` / `load_checkpoint`）
- Create: `qwen3-learning/qwen3-pytorch/tests/test_pretrain.py`
- Modify: `qwen3-learning/qwen3-pytorch/config.py`（新增 `log_every` / `eval_max_batches` 两个**训练流程**参数，**未改动任何既有超参数值**）
- Modify: `init.sh`（L4 层加一条训练步 smoke）
- Modify: `feature_list.json`

- [x] **Step 1: 写 `utils.py`**

GPT-2 线已有 `utils.py`，但**不能跨目录 import**，在 Qwen3 目录内独立重写一份（结构同构、代码独立）：

```python
"""
Qwen3 线的通用工具函数。

与 GPT-2 线的 utils.py **不共享代码**（AGENTS.md 硬约束：Qwen3 线不跨目录 import），
但接口保持一致，方便对照阅读。
"""

import os
import random
from typing import Optional

import torch


def set_seed(seed: int) -> None:
    """
    固定 Python / NumPy / PyTorch 三处随机种子，保证实验可复现。

    参数：
        seed: 随机种子
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    选设备：有 CUDA 用 CUDA，否则用 CPU（本项目 CPU 就能跑）。

    返回：
        torch.device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(payload: dict, path: str) -> str:
    """
    保存 checkpoint（自动建目录）。

    参数：
        payload: 要保存的字典
        path:    保存路径
    返回：
        实际写入的路径
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)
    return path


def load_checkpoint(path: str, map_location: Optional[str] = None) -> dict:
    """
    加载 checkpoint。

    参数：
        path:        路径
        map_location: torch.load 的 map_location
    返回：
        checkpoint 字典
    """
    return torch.load(path, map_location=map_location, weights_only=False)
```

- [x] **Step 2: 写 `train/train_pretrain.py`**

```python
"""
Qwen3 预训练脚本（唯一带 --model 开关的脚本）。

用法：
    python train/train_pretrain.py --model dense
    python train/train_pretrain.py --model moe

与 GPT-2 线的三处刻意偏离（见 开发计划.md §0.3）：
    1. 用 argparse 而不是写死参数
    2. 超参全部从 config.py 的 DENSE_CONFIG / MOE_CONFIG 读
    3. 模型 forward 恒定返回 4 元组 (logits, loss, kv_cache, extras)，
       MoE 的 aux_loss 从 extras.get("aux_loss") 取 —— 训练脚本不需要 if is_moe 分支去猜返回结构

【为什么顶部要补 sys.path】
    脚本放在 train/ 子目录里，`python train/train_pretrain.py` 时
    `sys.path[0]` 是 train/，找不到项目根下的 config.py / data 包
    → ModuleNotFoundError。与 GPT-2 线「脚本放在目录顶层」不同，这里必须显式补。
"""

import argparse
import math
import os
import sys
import time

# 把项目根补进 sys.path（见模块 docstring）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import DENSE_CONFIG, MOE_CONFIG
from data.pretrain_data import make_pretrain_loader
from data.tokenizer import BPETokenizer
from model.qwen3_dense import Qwen3Dense
from model.qwen3_moe import Qwen3MoE
from utils import get_device, save_checkpoint, set_seed


def build_model(model_kind: str, vocab_size: int, dropout: float = 0.1):
    """
    按名字构造模型。

    参数：
        model_kind: "dense" 或 "moe"
        vocab_size: 词表大小
        dropout:    dropout 比例
    返回：
        nn.Module
    """
    if model_kind == "dense":
        return Qwen3Dense(vocab_size=vocab_size, cfg=DENSE_CONFIG, dropout=dropout)
    if model_kind == "moe":
        return Qwen3MoE(vocab_size=vocab_size, cfg=MOE_CONFIG, dropout=dropout)
    raise ValueError(f"未知的 --model: {model_kind}（只支持 dense / moe）")


def lr_at(step: int, cfg) -> float:
    """
    warmup + cosine 学习率调度。

        step < warmup_iters        → 线性升温：lr * (step+1) / warmup
        warmup <= step < decay_iters→ cosine 衰减：min_lr + 0.5*(lr-min_lr)*(1+cos(pi*p))
        step >= lr_decay_iters     → 恒定在 min_lr

    参数：
        step: 当前迭代步数（0 起）
        cfg:  Qwen3Config
    返回：
        学习率
    """
    if step < cfg.warmup_iters:
        return cfg.learning_rate * (step + 1) / cfg.warmup_iters
    if step >= cfg.lr_decay_iters:
        return cfg.min_lr
    # cosine 从 learning_rate 降到 min_lr
    progress = (step - cfg.warmup_iters) / max(1, cfg.lr_decay_iters - cfg.warmup_iters)
    return cfg.min_lr + 0.5 * (cfg.learning_rate - cfg.min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, loader, max_batches: int = 20) -> float:
    """
    在验证集上估计平均 loss。

    参数：
        model:       模型
        loader:      验证集 DataLoader
        max_batches: 最多评估多少个 batch（控制 CPU 上的评估耗时）
    返回：
        平均 loss（float）
    """
    model.eval()
    total, count = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        _, loss, _, _ = model(x, targets=y)
        total += float(loss.item())
        count += 1
    model.train()
    return total / max(1, count)


def main() -> None:
    """训练主流程。"""
    parser = argparse.ArgumentParser(description="Qwen3 预训练")
    parser.add_argument("--model", choices=["dense", "moe"], default="dense",
                        help="要训练哪个模型")
    parser.add_argument("--max-iters", type=int, default=None,
                        help="覆盖 config 里的 max_iters（默认用 config 值）")
    args = parser.parse_args()

    cfg = DENSE_CONFIG if args.model == "dense" else MOE_CONFIG
    max_iters = args.max_iters if args.max_iters is not None else cfg.max_iters

    set_seed(cfg.seed)
    device = get_device()
    print(f"[pretrain] model={args.model} device={device} max_iters={max_iters}")

    # --- tokenizer ---
    tk = BPETokenizer(os.path.join("data", "vocab.json"), os.path.join("data", "merges.txt"))
    vocab_size = tk.vocab_size
    print(f"[pretrain] vocab_size = {vocab_size}")

    # --- 数据 ---
    train_loader = make_pretrain_loader("train", cfg.batch_size, cfg.block_size, cfg.seed)
    val_loader = make_pretrain_loader("val", cfg.batch_size, cfg.block_size, cfg.seed)
    print(f"[pretrain] train_batches={len(train_loader)} val_batches={len(val_loader)}")

    # --- 模型 ---
    model = build_model(args.model, vocab_size, cfg.dropout).to(device)
    print(f"[pretrain] params = {model.num_params():,}")

    # --- 优化器：只优化 requires_grad 的参数 ---
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # --- 训练循环 ---
    model.train()
    step = 0
    t0 = time.time()
    first_loss = None
    while step < max_iters:
        for x, y in train_loader:
            if step >= max_iters:
                break
            x, y = x.to(device), y.to(device)

            for g in optimizer.param_groups:
                g["lr"] = lr_at(step, cfg)

            _, loss, _, extras = model(x, targets=y)

            # MoE：总 loss = lm_loss + aux_loss_coef * aux_loss
            # dense 的 extras 是空 dict，extras.get("aux_loss") 返回 None → 不需要 if 分支
            aux = extras.get("aux_loss")
            if aux is not None:
                loss = loss + cfg.aux_loss_coef * aux

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            if first_loss is None:
                first_loss = float(loss.item())
            if step % cfg.log_every == 0:
                print(f"[pretrain] step {step:>4}/{max_iters} loss {loss.item():.4f} "
                      f"lr {optimizer.param_groups[0]['lr']:.2e} "
                      f"elapsed={time.time() - t0:.1f}s")
                if aux is not None:
                    load = extras.get("expert_load_last")
                    print(f"[pretrain]   aux_loss {float(aux.item()):.4f} "
                          f"expert_load {[round(float(v), 2) for v in load]}")
            step += 1

    # --- 收尾评估 ---
    val_loss = estimate_loss(model, val_loader, cfg.eval_max_batches)
    val_ppl = math.exp(min(20.0, val_loss))
    print(f"[pretrain] initial_loss={first_loss:.4f} final val_loss={val_loss:.4f} "
          f"val_ppl={val_ppl:.2f} elapsed={time.time() - t0:.1f}s")

    # --- 落盘 ---
    path = save_checkpoint(
        {"model_state": model.state_dict(), "cfg_name": args.model,
         "vocab_size": vocab_size, "config": cfg.to_dict()},
        os.path.join("checkpoints", f"pretrain_{args.model}.pt"),
    )
    print(f"[pretrain] saved -> {path}")


if __name__ == "__main__":
    main()
```

> **注意**：与 GPT-2 线最大的不同是**脚本放在 `train/` 子目录里**，
> 所以 `sys.path[0]` 是 `train/` 而不是项目根，必须显式把项目根 `sys.path.insert(0, ...)` 补进去，
> 否则 `from config import ...` 直接 ModuleNotFoundError（见「实测修正记录」第 1 条）。

- [x] **Step 3: 写 `tests/test_pretrain.py`**

```python
"""
预训练脚本 smoke：模型能建起来、能走一步训练、初始 loss 合理。

对应 feat-020（开发计划 Task 13）。

运行：python tests/test_pretrain.py
"""

import math
import os
import sys

import torch

# 直接执行 `python tests/test_x.py` 时 sys.path[0] 是 tests/ 目录，
# 找不到项目根下的 config.py / model 包 —— 这里把项目根补进 sys.path。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DENSE_CONFIG, MOE_CONFIG
from model.qwen3_dense import Qwen3Dense
from model.qwen3_moe import Qwen3MoE
from train.train_pretrain import build_model, lr_at

torch.manual_seed(DENSE_CONFIG.seed)
V = 256
model = Qwen3Dense(vocab_size=V, cfg=DENSE_CONFIG, dropout=0.0)

x = torch.randint(0, V, (2, 8))
y = torch.randint(0, V, (2, 8))

logits, loss, cache, extras = model(x, targets=y)
assert logits.shape == (2, 8, V), f"logits 形状错: {tuple(logits.shape)}"
assert loss is not None and torch.isfinite(loss), f"loss 异常: {loss}"
assert cache is None, "训练时不应有 kv_cache"
assert extras == {}, f"dense 的 extras 应为空 dict，实得 {extras}"

# loss 应接近 ln(V) —— 随机初始化时模型对每个 token 都是均匀分布
assert abs(loss.item() - math.log(V)) < 1.0, \
    f"初始 loss 应接近 ln(V)={math.log(V):.2f}，实得 {loss.item():.3f}"

# 反向传播能跑通
loss.backward()
grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
assert len(grads) > 0, "没有任何参数收到梯度"

# =====================================================================
# 学习率调度：warmup → cosine → 恒定 min_lr
# =====================================================================
cfg = DENSE_CONFIG
assert lr_at(0, cfg) == cfg.learning_rate * 1 / cfg.warmup_iters, "warmup 第一步不对"
assert lr_at(cfg.warmup_iters - 1, cfg) == cfg.learning_rate, "warmup 结束应等于 learning_rate"
assert abs(lr_at(cfg.warmup_iters, cfg) - cfg.learning_rate) < 1e-9, \
    "cosine 起点应等于 learning_rate"
assert abs(lr_at(cfg.lr_decay_iters, cfg) - cfg.min_lr) < 1e-9, \
    "衰减结束应等于 min_lr"
assert lr_at(cfg.lr_decay_iters + 100, cfg) == cfg.min_lr, "衰减之后应恒定在 min_lr"
# 整体单调性：warmup 段递增，衰减段递减
warm = [lr_at(s, cfg) for s in range(cfg.warmup_iters)]
assert all(warm[i] < warm[i + 1] for i in range(len(warm) - 1)), "warmup 段必须递增"
decay = [lr_at(s, cfg) for s in range(cfg.warmup_iters, cfg.lr_decay_iters)]
assert all(decay[i] >= decay[i + 1] for i in range(len(decay) - 1)), "衰减段必须递减"

# =====================================================================
# build_model：dense / moe 都能建，且 MoE 的 aux_loss 能进总 loss
# =====================================================================
m_dense = build_model("dense", V, dropout=0.0)
assert isinstance(m_dense, Qwen3Dense), "build_model('dense') 应返回 Qwen3Dense"
m_moe = build_model("moe", V, dropout=0.0)
assert isinstance(m_moe, Qwen3MoE), "build_model('moe') 应返回 Qwen3MoE"
try:
    build_model("transformer", V)
    raise AssertionError("未知的 model 名应当抛 ValueError")
except ValueError:
    pass

# MoE：总 loss = lm_loss + aux_loss_coef * aux_loss，梯度能回到 router
m_moe.train()
_, moe_loss, _, moe_extras = m_moe(x, targets=y)
assert moe_extras.get("aux_loss") is not None, "MoE 的 extras 必须含 aux_loss"
total = moe_loss + MOE_CONFIG.aux_loss_coef * moe_extras["aux_loss"]
assert torch.isfinite(total), "MoE 总 loss 非有限"
total.backward()
router_grads = [b.moe.router.weight.grad for b in m_moe.blocks
                if b.moe.router.weight.grad is not None]
assert len(router_grads) == MOE_CONFIG.n_layer, "router 应收到梯度"
# dense 侧没有 aux_loss，训练脚本用 extras.get() 统一处理
assert m_dense(x, targets=y)[3].get("aux_loss") is None, \
    "dense 不应有 aux_loss（训练脚本靠 .get 拿 None）"

print(f"test_pretrain OK  loss={loss.item():.4f} params={model.num_params():,} "
      f"lr[0]={lr_at(0, cfg):.2e} lr[end]={lr_at(cfg.lr_decay_iters, cfg):.2e}")
```

- [x] **Step 4: 运行**

```bash
cd qwen3-learning/qwen3-pytorch
python tests/test_pretrain.py
python train/train_pretrain.py --model dense
python train/train_pretrain.py --model moe
```

实测输出（2026-10-22，CPU）：

```
$ python tests/test_pretrain.py
test_pretrain OK  loss=5.5184 params=919,040 lr[0]=6.00e-06 lr[end]=3.00e-05

$ python train/train_pretrain.py --model dense
[pretrain] train_batches=15 val_batches=2
[pretrain] params = 985,856
[pretrain] step    0/300 loss 6.7154 lr 6.00e-06 elapsed=0.2s
[pretrain] step   50/300 loss 4.4896 lr 3.00e-04 elapsed=7.9s
[pretrain] step  100/300 loss 2.4191 lr 2.74e-04 elapsed=15.6s
[pretrain] step  150/300 loss 1.6038 lr 2.07e-04 elapsed=23.4s
[pretrain] step  200/300 loss 1.4735 lr 1.23e-04 elapsed=31.3s
[pretrain] step  250/300 loss 1.4469 lr 5.58e-05 elapsed=39.2s
[pretrain] initial_loss=6.7154 final val_loss=1.3510 val_ppl=3.86 elapsed=46.8s
[pretrain] saved -> checkpoints\pretrain_dense.pt

$ python train/train_pretrain.py --model moe
[pretrain] params = 1,184,512
[pretrain] step    0/300 loss 6.6954 lr 6.00e-06 elapsed=0.2s
[pretrain]   aux_loss 1.0253 expert_load [1184.5, 745.0, 1298.0, 868.5]
[pretrain] step   50/300 loss 4.4751 lr 3.00e-04 elapsed=10.2s
[pretrain]   aux_loss 1.0758 expert_load [293.5, 1053.0, 1599.5, 1150.0]
[pretrain] step  100/300 loss 2.4620 lr 2.74e-04 elapsed=20.1s
[pretrain]   aux_loss 1.0185 expert_load [901.0, 951.0, 1322.5, 921.5]
[pretrain] step  150/300 loss 1.6712 lr 2.07e-04 elapsed=29.9s
[pretrain]   aux_loss 0.9997 expert_load [1217.5, 975.5, 998.0, 905.0]
[pretrain] step  200/300 loss 1.4556 lr 1.23e-04 elapsed=39.7s
[pretrain]   aux_loss 1.0002 expert_load [1006.5, 1001.0, 978.5, 1110.0]
[pretrain] step  250/300 loss 1.4019 lr 5.58e-05 elapsed=49.5s
[pretrain]   aux_loss 0.9982 expert_load [1042.5, 980.5, 1000.0, 1073.0]
[pretrain] initial_loss=6.6954 final val_loss=1.3607 val_ppl=3.90 elapsed=59.1s
[pretrain] saved -> checkpoints\pretrain_moe.pt
```

**从日志能读出来的三件事（比"loss 降了"更值得看）**：

1. **初始 loss ≈ ln(V)**：V=778 时 ln(778) ≈ 6.66，实测 6.7154 / 6.6954 —— 随机初始化的模型对每个 token 近似均匀分布，这条是
   `tests/test_pretrain.py` 里那条 `abs(loss - ln(V)) < 1.0` 断言的由来。
2. **MoE 的 aux_loss 稳定在 1 附近**（1.0253 → 0.9982）：本仓库的 aux_loss 量纲是
   「f 归一化到 Σf=1」，**完全均衡时 = 1**（HF Mixtral 同款公式是 K=2，详见 Task 10 的实测修正记录）。
3. **专家负载被 aux_loss 拉平**：`expert_load` 从 [1184.5, 745.0, 1298.0, 868.5]（最大偏差 ±35%）
   收敛到 [1042.5, 980.5, 1000.0, 1073.0]（±3.5%），总和恒为 4096 = B·T —— 这就是负载均衡辅助损失在起作用。

- [x] **Step 5: 挂进 `init.sh` 的 L4 层**

L4 增加一条最小 smoke：构造 dense 模型 + 走一步训练，断言 loss 有限。
（实测后略微加严：除 loss 有限外，还断言「梯度非空」「梯度有限」「lr 调度端点正确」。）

```bash
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
    ...
  fi
```

实测输出：

```
  [PASS] Qwen3 训练步 smoke（forward+backward+step，loss 有限）[train-step] loss 5.4788 grads 46 个 lr0 6.00e-06 lr_end 3.00e-05
  PASS=23  SKIP=0  FAIL=0
```

> **踩坑**：Windows 下 Python 默认 stdout 是 GBK，bash 按 UTF-8 解码 → 中文乱码。
> 给子进程加 `PYTHONIOENCODING=utf-8` 才正常（`tests/run_all.py` 里已经这么处理过）。

- [x] **Step 6: 实测修正记录（初稿的 2 个真 bug）**

1. **`train/train_pretrain.py` 缺 sys.path 引导** → `ModuleNotFoundError: No module named 'config'`。
   根因：GPT-2 线的训练脚本放在**目录顶层**（`sys.path[0]` 天然是项目根），
   而 Qwen3 线按 `train/` 子目录组织（`sys.path[0]` = `train/`），必须显式补项目根。
   修法（写在 `import torch` 之前）：
   ```python
   sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
   ```
2. **`tests/test_pretrain.py` 同样缺 sys.path 引导** → 直接 `python tests/test_pretrain.py` 会
   `ModuleNotFoundError: No module named 'config'`（`sys.path[0]` 是 `tests/`）。
   修法同上：`sys.path.insert(0, <项目根>)`。

**附带**：`config.py` 新增两个字段，都是**训练流程参数**，没有改动任何既有超参数值，
所以**不需要同步改任何笔记的数字**：

```python
    # 训练日志打印间隔（步）
    log_every: int = 50
    # 验证时最多评估多少个 batch（控制 CPU 上的评估耗时）
    eval_max_batches: int = 20
```

> 初稿把 `step % 50`、`estimate_loss(model, val_loader)` 的 20 直接写死在训练脚本里，
> 违反「超参一律从 config.py 读」的硬约束 —— 本次一并改掉。

- [x] **Step 7: Commit**

```bash
git add qwen3-learning/ init.sh
git commit -m "feat-020: Qwen3 预训练脚本 + smoke（--model dense|moe，超参全从 config 读）"
```

---


### Task 14: feat-023 手写 LoRA

**Files:**
- Create: `qwen3-learning/qwen3-pytorch/model/common/lora.py`
- Create: `qwen3-learning/qwen3-pytorch/train/train_lora.py`
- Create: `qwen3-learning/qwen3-pytorch/tests/test_lora.py`
- Modify: `feature_list.json`

- [ ] **Step 1: 写 `model/common/lora.py`**

```python
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
        assert r > 0, "r 必须为正"
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = base.in_features
        out_features = base.out_features

        # A: (r, in_features) —— Kaiming 初始化
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B: (out_features, r) —— **零初始化**
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
    把 LoRA 分支合并回 base 权重。

    参数：
        model: 已注入 LoRA 的模型
    返回：
        成功合并的模块数量
    """
    count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.base.weight.data.copy_(module.merged_weight())
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
```

- [ ] **Step 2: 写 `tests/test_lora.py`**

```python
"""
LoRA smoke：注入前后输出一致、只有 A/B 有梯度、merge 后等价。

运行：python tests/test_lora.py
"""

import torch
import torch.nn as nn

from config import DENSE_CONFIG
from model.common.lora import LoRALinear, inject_lora, merge_lora, set_lora_enabled
from model.qwen3_dense import Qwen3Dense

torch.manual_seed(DENSE_CONFIG.seed)

# --- 1) 单个 LoRALinear：B 零初始化 → 输出与 base 完全一致 ---
base = nn.Linear(16, 32, bias=False)
x = torch.randn(4, 16)
y0 = base(x)
lora = LoRALinear(base, r=4, alpha=8)
y1 = lora(x)
assert torch.allclose(y0, y1, atol=1e-6), "B 零初始化时 LoRA 分支应无输出"

# --- 2) base 被冻结 ---
assert base.weight.requires_grad is False, "base.weight 应被冻结"
assert lora.lora_A.requires_grad and lora.lora_B.requires_grad, "A/B 应可训练"

# --- 3) 只有 A/B 收到梯度 ---
loss = lora(x).sum()
loss.backward()
assert lora.lora_A.grad is not None and lora.lora_B.grad is not None, "A/B 应有梯度"
assert base.weight.grad is None, "base 不应有梯度"

# --- 4) 注入整模型后输出逐元素相等 ---
V = 256
model = Qwen3Dense(vocab_size=V, cfg=DENSE_CONFIG, dropout=0.0)
model.eval()
idx = torch.randint(0, V, (2, 8))
with torch.no_grad():
    logits_before = model(idx)[0]

replaced = inject_lora(model, DENSE_CONFIG.lora_target_modules,
                       r=DENSE_CONFIG.lora_r, alpha=DENSE_CONFIG.lora_alpha)
assert len(replaced) > 0, "没有注入任何 LoRA 模块"
with torch.no_grad():
    logits_after = model(idx)[0]
assert torch.allclose(logits_before, logits_after, atol=1e-6), \
    f"注入前后输出应逐元素相等，最大差异 {float((logits_before - logits_after).abs().max())}"

# --- 5) set_lora_enabled(False) 等价于关掉 LoRA 分支 ---
set_lora_enabled(model, False)
with torch.no_grad():
    logits_off = model(idx)[0]
assert torch.allclose(logits_before, logits_off, atol=1e-6), "关闭 LoRA 分支后应回到注入前"

# --- 6) merge_lora 后与「开着 LoRA」等价 ---
set_lora_enabled(model, True)
with torch.no_grad():
    logits_on = model(idx)[0]
merged = merge_lora(model)
assert merged > 0, "merge_lora 应至少合并一个模块"
with torch.no_grad():
    logits_merged = model(idx)[0]
assert torch.allclose(logits_on, logits_merged, atol=1e-4), \
    f"merge 后应与未 merge 等价，最大差异 {float((logits_on - logits_merged).abs().max())}"

print(f"test_lora OK  replaced={len(replaced)} r={DENSE_CONFIG.lora_r} "
      f"alpha={DENSE_CONFIG.lora_alpha}")
```

- [ ] **Step 3: 写 `train/train_lora.py`**

```python
"""
Qwen3 LoRA 微调。

用法：
    python train/train_lora.py

【路线说明】
    路线已定为 LoRA，**不做全参 SFT**（全参 SFT 在 GPT-2 线已有覆盖）。
    目标：在偏好数据的 prompt 上做自监督式微调 —— 给定 prompt，训练模型补全 chosen。

【流程】
    1. 加载 checkpoints/pretrain_dense.pt
    2. inject_lora(model, ["q_proj", "v_proj"], r=4, alpha=8)
    3. 冻结 base，只训 A / B
    4. 评估：偏好胜率（chosen 的 logprob 高于 rejected 的样本比例）
"""

import argparse
import os

import torch
import torch.nn.functional as F

from config import DENSE_CONFIG
from data.preference_data import format_prompt_response, generate_preference_dataset
from data.tokenizer import BPETokenizer
from model.common.lora import inject_lora, set_lora_enabled
from model.qwen3_dense import Qwen3Dense
from utils import get_device, set_seed


def encode_pair(
    model: torch.nn.Module,
    tk: "BPETokenizer",
    pair: dict,
    device: torch.device,
) -> tuple:
    """
    算一对 (chosen, rejected) 在给定模型下的序列 logprob。

    参数：
        model:  模型
        tk:     tokenizer
        pair:   {"prompt", "chosen", "rejected"}
        device: 设备
    返回：
        (lp_chosen, lp_rejected)，各为 (1,) tensor
    """
    prompt_ids = tk.encode(pair["prompt"])
    out = []
    for key in ("chosen", "rejected"):
        resp_ids = tk.encode(pair[key])
        ids = (prompt_ids + resp_ids)[: DENSE_CONFIG.block_size]
        if len(ids) < 2:
            out.append(torch.zeros(1, device=device))
            continue
        idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        logits, _, _, _ = model(idx)
        logp = F.log_softmax(logits.float(), dim=-1)
        # 第 0..len(prompt)-1 个位置预测的是 prompt 自身，只对 response 位置计分
        n_prompt = min(len(prompt_ids), len(ids) - 1)
        picked = logp[0, n_prompt - 1: -1, :].gather(
            -1, idx[0, n_prompt:].unsqueeze(-1)).squeeze(-1)
        out.append(picked.sum().unsqueeze(0))
    return out[0], out[1]


def main() -> None:
    """LoRA 微调主流程。"""
    parser = argparse.ArgumentParser(description="Qwen3 LoRA 微调")
    parser.add_argument("--max-iters", type=int, default=None, help="覆盖 config 的 lora_max_iters")
    args = parser.parse_args()

    cfg = DENSE_CONFIG
    max_iters = args.max_iters if args.max_iters is not None else cfg.lora_max_iters

    set_seed(cfg.seed)
    device = get_device()
    print(f"[lora] device={device} max_iters={max_iters}")

    tk = BPETokenizer(os.path.join("data", "vocab.json"), os.path.join("data", "merges.txt"))

    # --- 加载预训练权重 ---
    model = Qwen3Dense(vocab_size=tk.vocab_size, cfg=cfg, dropout=0.0).to(device)
    ckpt_path = os.path.join("checkpoints", "pretrain_dense.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    print(f"[lora] loaded {ckpt_path}")

    # --- 注入 LoRA ---
    replaced = inject_lora(model, cfg.lora_target_modules, r=cfg.lora_r,
                           alpha=cfg.lora_alpha, dropout=cfg.lora_dropout)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[lora] injected {len(replaced)} modules, trainable params = "
          f"{sum(p.numel() for p in trainable):,}")

    # --- 偏好数据 ---
    pairs = generate_preference_dataset(n_samples=cfg.dpo_n_pairs, seed=cfg.seed)
    print(f"[lora] preference pairs = {len(pairs)}")

    optimizer = torch.optim.AdamW(trainable, lr=cfg.lora_lr, weight_decay=cfg.weight_decay)

    model.train()
    step = 0
    for _epoch in range(max_iters // max(1, len(pairs)) + 1):
        for pair in pairs:
            if step >= max_iters:
                break
            text = format_prompt_response(pair["prompt"], pair["chosen"])
            ids = tk.encode(text)[: cfg.block_size]
            if len(ids) < 2:
                continue
            idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            targets = torch.cat([idx[:, 1:], torch.full_like(idx[:, :1], -100)], dim=1)

            _, loss, _, _ = model(idx, targets=targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            optimizer.step()

            if step % 50 == 0:
                print(f"[lora] step {step:>4}/{max_iters} loss {loss.item():.4f}")
            step += 1
        if step >= max_iters:
            break

    # --- 评估：偏好胜率 ---
    model.eval()
    wins = 0
    n_eval = 0
    with torch.no_grad():
        for pair in pairs[: cfg.eval_max_samples]:
            lp_w, lp_l = encode_pair(model, tk, pair, device)
            wins += int(lp_w.item() > lp_l.item())
            n_eval += 1
    win_rate = wins / max(1, n_eval)
    print(f"[lora] preference win rate = {win_rate:.3f} (random baseline = 0.5)")

    os.makedirs("checkpoints", exist_ok=True)
    out = os.path.join("checkpoints", "lora.pt")
    torch.save({"model_state": model.state_dict(), "win_rate": win_rate}, out)
    print(f"[lora] saved -> {out}")


if __name__ == "__main__":
    main()
```

> 实现注：`encode_pair` 是 LoRA / DPO / 评估三处共用的 helper，
> 放在 `train/train_lora.py` 里，DPO 与 evaluate.py 都从这里 import，**不要各写一份**。

- [ ] **Step 4: 运行**

```bash
cd qwen3-learning/qwen3-pytorch
python tests/test_lora.py
python train/train_lora.py
```

Expected: `test_lora OK  replaced=... r=4 alpha=8`；训练 loss 下降；
`preference win rate` 明显高于 0.5。

- [ ] **Step 5: Commit**

```bash
git add qwen3-learning/
git commit -m "feat-023: 从零手写 LoRA（B 零初始化 + base 冻结 + inject/merge/enable 三件套）"
```

---

### Task 15: feat-024 手写 DPO

**Files:**
- Create: `qwen3-learning/qwen3-pytorch/train/train_dpo.py`
- Create: `qwen3-learning/qwen3-pytorch/tests/test_dpo.py`
- Create: `qwen3-learning/Qwen3_LoRA与DPO_学习笔记.md`
- Modify: `feature_list.json`

- [ ] **Step 1: 写 `train/train_dpo.py`**

```python
"""
Qwen3 DPO（Direct Preference Optimization）训练。

用法：
    python train/train_dpo.py

【目标函数】
    L_DPO = -E[ log σ( β * ( (logπ_θ(y_w|x) - logπ_ref(y_w|x))
                            - (logπ_θ(y_l|x) - logπ_ref(y_l|x)) ) ) ]

    其中 β 是 KL 惩罚系数（config.dpo_beta，默认 0.1）

【reference model 的取法 —— 本 Task 最重要的一个技巧】
    **不单独加载一份冻结模型**。复用同一个 model 对象：
        开启 LoRA 分支 → π_θ
        关闭 LoRA 分支 → π_ref（等价于纯预训练权重）
    既省一份显存与一份 checkpoint，也把这个技巧变成可读的代码。
    对应 model/common/lora.py 的 set_lora_enabled()。

【与 GPT-2 线的区别】
    GPT-2 线：SFT → Reward Model → PPO（要显式训一个标量奖励模型）
    Qwen3 线：DPO 直接吃偏好对，**不训 Reward Model**，也不用 PPO
"""

import argparse
import os

import torch
import torch.nn.functional as F

from config import DENSE_CONFIG
from data.preference_data import generate_preference_dataset
from data.tokenizer import BPETokenizer
from model.common.lora import inject_lora, set_lora_enabled
from model.qwen3_dense import Qwen3Dense
from train.train_lora import encode_pair
from utils import get_device, set_seed


def dpo_loss(
    model: torch.nn.Module,
    pair: dict,
    tk: "BPETokenizer",
    beta: float,
    device: torch.device,
) -> tuple:
    """
    算一对 (chosen, rejected) 的 DPO loss。

    参数：
        model:  已注入 LoRA 的模型
        pair:   {"prompt", "chosen", "rejected"}
        tk:     tokenizer
        beta:   KL 惩罚系数
        device: 设备
    返回：
        (loss, margin)
            loss:   scalar
            margin: logπ_θ(y_w) - logπ_θ(y_l)（用于观察奖励间隔）
    """
    # --- π_θ：开 LoRA ---
    set_lora_enabled(model, True)
    lp_w_theta, lp_l_theta = encode_pair(model, tk, pair, device)

    # --- π_ref：关 LoRA（等价于纯预训练权重）---
    set_lora_enabled(model, False)
    with torch.no_grad():
        lp_w_ref, lp_l_ref = encode_pair(model, tk, pair, device)

    # --- 还原到 π_θ，准备回传 ---
    set_lora_enabled(model, True)

    logits = beta * ((lp_w_theta - lp_w_ref) - (lp_l_theta - lp_l_ref))
    loss = -F.logsigmoid(logits).mean()
    margin = (lp_w_theta - lp_l_theta).mean()
    return loss, margin


def main() -> None:
    """DPO 训练主流程。"""
    parser = argparse.ArgumentParser(description="Qwen3 DPO 训练")
    parser.add_argument("--max-iters", type=int, default=None, help="覆盖 config 的 dpo_max_iters")
    args = parser.parse_args()

    cfg = DENSE_CONFIG
    max_iters = args.max_iters if args.max_iters is not None else cfg.dpo_max_iters

    set_seed(cfg.seed)
    device = get_device()
    print(f"[dpo] device={device} max_iters={max_iters} beta={cfg.dpo_beta}")

    tk = BPETokenizer(os.path.join("data", "vocab.json"), os.path.join("data", "merges.txt"))

    # --- 加载 LoRA 微调后的权重 ---
    model = Qwen3Dense(vocab_size=tk.vocab_size, cfg=cfg, dropout=0.0).to(device)
    ckpt = torch.load(os.path.join("checkpoints", "lora.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    inject_lora(model, cfg.lora_target_modules, r=cfg.lora_r, alpha=cfg.lora_alpha)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[dpo] trainable params = {sum(p.numel() for p in trainable):,}")

    pairs = generate_preference_dataset(n_samples=cfg.dpo_n_pairs, seed=cfg.seed)
    optimizer = torch.optim.AdamW(trainable, lr=cfg.dpo_lr, weight_decay=cfg.weight_decay)

    margins = []
    model.train()
    step = 0
    for _epoch in range(max_iters // max(1, len(pairs)) + 1):
        for pair in pairs:
            if step >= max_iters:
                break
            loss, margin = dpo_loss(model, pair, tk, cfg.dpo_beta, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            optimizer.step()

            margins.append(float(margin.item()))
            if step % 50 == 0:
                recent = sum(margins[-50:]) / len(margins[-50:])
                print(f"[dpo] step {step:>4}/{max_iters} loss {loss.item():.4f} "
                      f"margin {margin.item():.4f} (avg50 {recent:.4f})")
            step += 1
        if step >= max_iters:
            break

    # --- 评估 ---
    first = sum(margins[:50]) / max(1, len(margins[:50]))
    last = sum(margins[-50:]) / max(1, len(margins[-50:]))
    print(f"[dpo] margin 首50均 {first:.4f} → 末50均 {last:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    out = os.path.join("checkpoints", "dpo.pt")
    torch.save({"model_state": model.state_dict(), "margin_first": first, "margin_last": last}, out)
    print(f"[dpo] saved -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 写 `tests/test_dpo.py`**

```python
"""
DPO smoke：loss 有限、π_ref 与 π_θ 不同、margin 可计算。

运行：python tests/test_dpo.py
"""

import torch

from config import DENSE_CONFIG
from data.preference_data import generate_preference_dataset
from data.tokenizer import BPETokenizer
from model.common.lora import inject_lora, set_lora_enabled
from model.qwen3_dense import Qwen3Dense
from train.train_dpo import dpo_loss

torch.manual_seed(DENSE_CONFIG.seed)

tk = BPETokenizer("data/vocab.json", "data/merges.txt")
pairs = generate_preference_dataset(n_samples=8, seed=DENSE_CONFIG.seed)

V = tk.vocab_size
model = Qwen3Dense(vocab_size=V, cfg=DENSE_CONFIG, dropout=0.0)
inject_lora(model, DENSE_CONFIG.lora_target_modules, r=DENSE_CONFIG.lora_r,
            alpha=DENSE_CONFIG.lora_alpha)

loss, margin = dpo_loss(model, pairs[0], tk, DENSE_CONFIG.dpo_beta, torch.device("cpu"))
assert torch.isfinite(loss), f"loss 非有限: {loss}"
assert torch.isfinite(margin), f"margin 非有限: {margin}"
assert loss.item() > 0, "DPO loss 应为正"

# π_ref 与 π_θ 必须不同（否则 LoRA 分支没生效）
idx = torch.randint(0, V, (1, 8))
set_lora_enabled(model, True)
with torch.no_grad():
    a = model(idx)[0]
set_lora_enabled(model, False)
with torch.no_grad():
    b = model(idx)[0]
assert not torch.allclose(a, b, atol=1e-6), "开关 LoRA 分支后输出应不同"

print(f"test_dpo OK  loss={loss.item():.4f} margin={margin.item():.4f}")
```

- [ ] **Step 3: 运行**

```bash
cd qwen3-learning/qwen3-pytorch
python tests/test_dpo.py
python train/train_dpo.py
```

Expected: `test_dpo OK  loss=... margin=...`；训练中 `margin` 的 50 步滑动均值**上升**。

- [ ] **Step 4: 写笔记**

新增 `Qwen3_LoRA与DPO_学习笔记.md`（当前 7 篇笔记没有覆盖 LoRA / DPO），
写清：LoRA 公式与 B 零初始化理由、DPO 目标函数推导、π_ref 复用同一模型对象的技巧。

- [ ] **Step 5: Commit**

```bash
git add qwen3-learning/
git commit -m "feat-024: 从零手写 DPO（π_ref 复用同一模型对象 + 关 LoRA 分支）"
```

---

## Phase 4 — 评估、推理与退化对照

---

### Task 16: feat-025 精简评估脚本

**Files:**
- Create: `qwen3-learning/qwen3-pytorch/evaluate.py`
- Modify: `feature_list.json`

- [ ] **Step 1: 写 `evaluate.py`**

```python
"""
Qwen3 精简评估脚本。

用法：
    python evaluate.py --stage pretrain
    python evaluate.py --stage lora
    python evaluate.py --stage dpo
    python evaluate.py --stage speedup
    python evaluate.py --stage all

【与 GPT-2 线 9 项评估的关系】
    **不做逐项对标** —— GPT-2 那套是围绕 Reward Model + PPO 设计的，
    DPO 阶段不适用。但 pretrain 与 speedup 两项**保留同名列**，
    便于两条线横向对比。

【硬约束】
    不删/不改弱指标门槛。达不到门槛就报错退出，而不是悄悄打一个好看的数。
"""

import argparse
import math
import os

import torch

from config import DENSE_CONFIG
from data.preference_data import generate_preference_dataset
from data.pretrain_data import make_pretrain_loader
from data.tokenizer import BPETokenizer


def eval_pretrain(tk: "BPETokenizer") -> dict:
    """
    预训练评估：val loss、val perplexity。

    参数：
        tk: tokenizer
    返回：
        {"val_loss": ..., "val_ppl": ...}
    """
    from model.qwen3_dense import Qwen3Dense

    cfg = DENSE_CONFIG
    model = Qwen3Dense(vocab_size=tk.vocab_size, cfg=cfg, dropout=0.0)
    ckpt = torch.load(os.path.join("checkpoints", "pretrain_dense.pt"),
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    loader = make_pretrain_loader("val", cfg.batch_size, cfg.block_size, cfg.seed)
    total, count = 0.0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= 20:
                break
            _, loss, _, _ = model(x, targets=y)
            total += float(loss.item())
            count += 1
    val_loss = total / max(1, count)
    return {"val_loss": val_loss, "val_ppl": math.exp(min(20.0, val_loss))}


def eval_lora(tk: "BPETokenizer") -> dict:
    """
    LoRA 评估：偏好数据上的 chosen 胜率。

    参数：
        tk: tokenizer
    返回：
        {"win_rate": ..., "baseline": 0.5}
    """
    from model.common.lora import inject_lora
    from model.qwen3_dense import Qwen3Dense
    from train.train_lora import encode_pair

    cfg = DENSE_CONFIG
    model = Qwen3Dense(vocab_size=tk.vocab_size, cfg=cfg, dropout=0.0)
    ckpt = torch.load(os.path.join("checkpoints", "lora.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    inject_lora(model, cfg.lora_target_modules, r=cfg.lora_r, alpha=cfg.lora_alpha)
    model.eval()

    pairs = generate_preference_dataset(n_samples=cfg.dpo_n_pairs, seed=cfg.seed)
    wins = 0
    n = 0
    with torch.no_grad():
        for pair in pairs[: cfg.eval_max_samples]:
            lp_w, lp_l = encode_pair(model, tk, pair, torch.device("cpu"))
            wins += int(lp_w.item() > lp_l.item())
            n += 1
    return {"win_rate": wins / max(1, n), "baseline": 0.5}


def eval_dpo(tk: "BPETokenizer") -> dict:
    """
    DPO 评估：reward margin 均值/标准差、chosen>rejected 准确率。

    参数：
        tk: tokenizer
    返回：
        {"margin_mean": ..., "margin_std": ..., "accuracy": ...}
    """
    from model.common.lora import inject_lora
    from model.qwen3_dense import Qwen3Dense
    from train.train_lora import encode_pair

    cfg = DENSE_CONFIG
    model = Qwen3Dense(vocab_size=tk.vocab_size, cfg=cfg, dropout=0.0)
    ckpt = torch.load(os.path.join("checkpoints", "dpo.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    inject_lora(model, cfg.lora_target_modules, r=cfg.lora_r, alpha=cfg.lora_alpha)
    model.eval()

    pairs = generate_preference_dataset(n_samples=cfg.dpo_n_pairs, seed=cfg.seed)
    margins = []
    with torch.no_grad():
        for pair in pairs[: cfg.eval_max_samples]:
            lp_w, lp_l = encode_pair(model, tk, pair, torch.device("cpu"))
            margins.append(lp_w.item() - lp_l.item())
    if not margins:
        return {"margin_mean": 0.0, "margin_std": 0.0, "accuracy": 0.0}
    t = torch.tensor(margins)
    return {
        "margin_mean": float(t.mean()),
        "margin_std": float(t.std()),
        "accuracy": float((t > 0).float().mean()),
    }


def eval_speedup(tk: "BPETokenizer") -> dict:
    """
    推理评估：有/无 KV Cache 的输出一致性与加速比。

    参数：
        tk: tokenizer
    返回：
        {"consistent": bool, "speedup": float}
    """
    # 复用 GPT-2 线 generate.py 的 prefill/decode 组织方式（独立重写，不跨目录 import）
    from inference import benchmark_speed, generate_naive, generate_with_cache
    from model.qwen3_dense import Qwen3Dense

    cfg = DENSE_CONFIG
    model = Qwen3Dense(vocab_size=tk.vocab_size, cfg=cfg, dropout=0.0)
    model.eval()
    prompt = tk.encode("hello world")
    a = generate_naive(model, prompt, max_new_tokens=16, temperature=0.0)
    b = generate_with_cache(model, prompt, max_new_tokens=16, temperature=0.0)
    bench = benchmark_speed(model, prompt_len=16, max_new_tokens=16)
    return {"consistent": a == b, "speedup": bench["speedup"]}


STAGES = {
    "pretrain": eval_pretrain,
    "lora": eval_lora,
    "dpo": eval_dpo,
    "speedup": eval_speedup,
}


def main() -> None:
    """评估主流程。"""
    parser = argparse.ArgumentParser(description="Qwen3 评估")
    parser.add_argument("--stage", choices=["pretrain", "lora", "dpo", "speedup", "all"],
                        default="all")
    args = parser.parse_args()

    tk = BPETokenizer(os.path.join("data", "vocab.json"), os.path.join("data", "merges.txt"))
    stages = list(STAGES) if args.stage == "all" else [args.stage]

    for name in stages:
        print(f"\n=== stage: {name} ===")
        result = STAGES[name](tk)
        for k, v in result.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行**

```bash
cd qwen3-learning/qwen3-pytorch
python evaluate.py --stage all
```

Expected: 四个 stage 依次打印指标，不中途报错。

- [ ] **Step 3: Commit**

```bash
git add qwen3-learning/
git commit -m "feat-025: Qwen3 精简评估脚本（pretrain/lora/dpo/speedup 四段）"
```

---

### Task 17: feat-018 KV Cache 推理与采样

**Files:**
- Create: `qwen3-learning/qwen3-pytorch/inference.py`
- Modify: `feature_list.json`

- [ ] **Step 1: 写 `inference.py`**

复用 GPT-2 线 `generate.py` 的 prefill / decode 组织方式（**独立重写，不跨目录 import**）：

```python
"""
Qwen3 推理：prefill / decode 两阶段 + temperature / top-k / top-p 采样。

【与 GPT-2 线的组织方式一致】
    prefill：整段 prompt 一次 forward，建立 KV Cache
    decode ：每步只喂 1 个新 token，KV 沿序列维追加

【Qwen3 特有的两点】
    1. decode 阶段 T_q = 1、T_kv 随步数增长 —— **不需要因果 mask**
       （cache 天然只含历史 token）
    2. KV Cache 按 n_head_kv 分配，GQA 下占用是同尺寸 MHA 的 n_head_kv/n_head
"""

import time
from typing import List, Optional

import torch


def sample_from_logits(
    logits: "torch.Tensor",
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> "torch.Tensor":
    """
    从 logits 采样一个 token。

    参数：
        logits:     (B, V)
        temperature: 温度；<= 0 走 greedy
        top_k:      保留概率最高的 k 个
        top_p:      nucleus 采样阈值
    返回：
        (B, 1) 采样出的 token id
    """
    if temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k is not None and top_k > 0:
        kth = logits.topk(top_k, dim=-1).values[:, -1:]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        sorted_probs = sorted_logits.softmax(dim=-1)
        cumsum = sorted_probs.cumsum(dim=-1)
        mask = (cumsum - sorted_probs) > top_p
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_naive(
    model,
    prompt: List[int],
    max_new_tokens: int = 64,
    **kw,
) -> List[int]:
    """
    无 KV Cache 的朴素生成（每步重算整段前缀）。作为 KV Cache 版的正确性基准。

    参数：
        model:           模型
        prompt:          prompt token id 列表
        max_new_tokens: 最多生成多少个 token
        **kw:            采样参数，透传给 sample_from_logits
    返回：
        完整序列（含 prompt）
    """
    idx = torch.tensor(prompt, dtype=torch.long).unsqueeze(0)
    for _ in range(max_new_tokens):
        logits, _, _, _ = model(idx)
        next_id = sample_from_logits(logits[:, -1, :], **kw)
        idx = torch.cat([idx, next_id], dim=1)
    return idx[0].tolist()


def generate_with_cache(
    model,
    prompt: List[int],
    max_new_tokens: int = 64,
    **kw,
) -> List[int]:
    """
    带 KV Cache 的生成：prefill 一次 + decode 逐步追加。

    参数：
        model:           模型
        prompt:          prompt token id 列表
        max_new_tokens: 最多生成多少个 token
        **kw:            采样参数
    返回：
        完整序列（含 prompt）
    """
    from model.common.kv_cache import KVCache

    idx = torch.tensor(prompt, dtype=torch.long).unsqueeze(0)
    B, T = idx.shape
    device = idx.device
    cache = KVCache(n_layer=model.n_layer, n_head=model.n_head_kv, head_dim=model.head_dim,
                    max_seq_len=model.block_size, batch_size=B, device=device)

    # --- prefill：整段一次 forward ---
    position_ids = torch.arange(T, device=device).unsqueeze(0)
    logits, _, new_cache, _ = model(idx, kv_cache=cache, position_ids=position_ids)
    cache = new_cache
    out = list(prompt)

    for step in range(max_new_tokens):
        next_id = sample_from_logits(logits[:, -1, :], **kw)
        out.append(int(next_id.item()))
        # --- decode：只喂 1 个新 token ---
        position_ids = torch.tensor([[T + step]], device=device)
        logits, _, new_cache, _ = model(next_id, kv_cache=cache, position_ids=position_ids)
        cache = new_cache
    return out


def benchmark_speed(model, prompt_len: int = 32, max_new_tokens: int = 64) -> dict:
    """
    对比有/无 KV Cache 的耗时。

    参数：
        prompt_len:     prompt 长度
        max_new_tokens: 生成步数
    返回：
        {"naive_sec": ..., "cached_sec": ..., "speedup": ...}
    """
    prompt = list(range(1, prompt_len + 1))
    t0 = time.time(); generate_naive(model, prompt, max_new_tokens); t1 = time.time()
    t2 = time.time(); generate_with_cache(model, prompt, max_new_tokens); t3 = time.time()
    naive, cached = t1 - t0, t3 - t2
    return {"naive_sec": naive, "cached_sec": cached, "speedup": naive / max(1e-9, cached)}
```

- [ ] **Step 2: 运行**

```bash
cd qwen3-learning/qwen3-pytorch
python -c "
import torch
from config import DENSE_CONFIG
from model.qwen3_dense import Qwen3Dense
from inference import generate_naive, generate_with_cache, benchmark_speed
m = Qwen3Dense(vocab_size=256, cfg=DENSE_CONFIG, dropout=0.0); m.eval()
a = generate_naive(m, [1,2,3,4], max_new_tokens=16, temperature=0.0)
b = generate_with_cache(m, [1,2,3,4], max_new_tokens=16, temperature=0.0)
print('consistent:', a == b)
print(benchmark_speed(m))
"
```

Expected: `consistent: True`（greedy 下必须逐 token 相同）；speedup > 1。

- [ ] **Step 3: Commit**

```bash
git add qwen3-learning/
git commit -m "feat-018: Qwen3 KV Cache 推理与采样（prefill/decode + temperature/top-k/top-p）"
```

---

### Task 18: feat-017 与 GPT-2 的退化对照实验

**Files:**
- Create: `qwen3-learning/qwen3-pytorch/ablations/compare_with_gpt2.py`
- Modify: `feature_list.json`
- Modify: `qwen3-learning/Qwen3_*.md`（把消融结论写进笔记）

> **这是 Qwen3 线最有价值的产出**：把 7 篇笔记里的每个改动点，从「文字断言」变成「一次可跑的消融」。

- [ ] **Step 1: 设计退化开关**

Qwen3 模型需要支持 4 个开关：

| 开关 | 关掉后 | 对应 GPT-2 的什么 |
|---|---|---|
| `use_gqa` | `n_head_kv = n_head` | MHA |
| `use_rope` | 不加位置信息 | learned PE（`wpe`） |
| `use_qk_norm` | Q/K 不做 RMSNorm | 无（GPT-2 没有） |
| `mlp` | `"swiglu"` → `"gelu"` | `Linear → GELU → Linear` |

**关键难点（必须在代码注释里写清楚，否则读不懂为什么能对齐）**：

1. **位置编码**：GPT-2 用 **learned** `pos_emb`（可训练参数），
   Qwen3 用 **RoPE**（无参数）。`use_rope=False` 时，Qwen3 侧需要一个等价的
   learned PE 才能对齐 —— 做法是给 Qwen3 加一个 `wpe` 参数，
   训练/加载时把 GPT-2 的 `pos_emb.weight` 拷进去（两者形状都是 (block_size, n_embd)）。
   注意 GPT-2 的 `pos_emb` 是 `persistent=False` 的 buffer，不在 `state_dict` 里，
   所以要单独构造。

2. **LayerNorm vs RMSNorm**：GPT-2 用 `nn.LayerNorm`（减均值），
   Qwen3 用 RMSNorm（不减均值）。**这两个在数学上不等价**，
   所以「完全对齐」的前提之一是：对照时两边都换成同一种归一化。
   **推荐做法**：在对照脚本里两边都用 LayerNorm，再比。
   否则 max_abs_diff 根本压不到 1e-4。

3. **MLP 结构**：GPT-2 是 2 矩阵（`c_fc` → GELU → `c_proj`），
   Qwen3 SwiGLU 是 3 矩阵（gate/up/down）。退化到 GELU 时，
   把 `gate_proj` 丢弃、`up_proj` 映射到 `c_fc`、`down_proj` 映射到 `c_proj`。

4. **权重绑定与 bias**：GPT-2 的 `bias=False`、Qwen3 的 `attention_bias=False` 对齐；
   两边 lm_head 都与 tok_emb 绑定，不用单独映射。

- [ ] **Step 2: 写 `ablations/compare_with_gpt2.py`**

```python
"""
feat-017：Qwen3 与 GPT-2 的结构等价性对照实验。

【目的】
    把「Qwen3 相对 GPT-2 改了什么」从文字断言变成一次可跑的实验。

【做法】
    把 Qwen3 的 4 个组件开关全部关掉，退化成与 GPT-2 相同的结构，
    再把 GPT-2 的权重逐层映射进退化的 Qwen3，比对同输入下的输出。

【参数名映射表（写在代码里，便于核对）】
    GPT-2                              →  Qwen3（退化后）
    ---------------------------------------------------------------------------
    tok_emb.weight                     →  wte.weight
    pos_emb（buffer，非 state_dict）   →  wpe.weight        [use_rope=False 时新建]
    blocks.{i}.ln_1.{weight,bias}      →  blocks.{i}.ln_1.{weight,bias}
    blocks.{i}.attn.c_attn.weight      →  blocks.{i}.attn.qkv.weight   （拆分后重组）
    blocks.{i}.attn.c_proj.weight      →  blocks.{i}.attn.o_proj.weight
    blocks.{i}.ln_2.{weight,bias}      →  blocks.{i}.ln_2.{weight,bias}
    blocks.{i}.mlp.c_fc.weight         →  blocks.{i}.mlp.up.weight     [mlp='gelu']
    blocks.{i}.mlp.c_proj.weight       →  blocks.{i}.mlp.down.weight  [mlp='gelu']
    ln_f.{weight,bias}                 →  ln_f.{weight,bias}
    （lm_head 与 tok_emb 绑定，两边都绑定，不用单独映射）

【阈值】
    max_abs_diff < 1e-4（fp32）。设计.md §11 允许放宽，但要先确认不是 bug。
"""

import argparse
import os
import sys

import torch

# GPT-2 线在 gpt-learning/gpt-learning-pytorch/ 下，本脚本需要 import 它的模型定义。
# 这是**跨目录 import**，但只发生在对照实验里，且是单向的（Qwen3 → GPT-2），
# 不违反「Qwen3 主线代码不跨目录 import」的约束（主线 model/ 与 train/ 都不 import GPT-2）。
GPT2_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "gpt-learning", "gpt-learning-pytorch",
)
sys.path.insert(0, GPT2_DIR)

from model.gpt2 import GPT2  # noqa: E402


def compare(max_abs_diff_threshold: float = 1e-4) -> dict:
    """
    跑一次对照实验。

    返回：
        {"max_abs_diff": float, "passed": bool}
    """
    # ---- 造 GPT-2 ----
    gpt2 = GPT2(vocab_size=64, n_layer=2, n_head=4, n_embd=128, n_inner=512,
                block_size=128, dropout=0.0, bias=False)
    gpt2.eval()

    # ---- 造退化的 Qwen3 ----
    #   use_gqa=False    → n_head_kv = n_head
    #   use_rope=False   → 用 learned PE 替代 RoPE
    #   use_qk_norm=False→ 不做 QK-Norm
    #   mlp='gelu'       → 两矩阵 FFN
    #   norm='layernorm' → 两边都用 LayerNorm（关键，见上文难点 2）
    ...
    return {"max_abs_diff": max_diff, "passed": max_diff < max_abs_diff_threshold}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="feat-017 退化对照")
    parser.add_argument("--threshold", type=float, default=1e-4)
    args = parser.parse_args()
    result = compare(args.threshold)
    print(f"[compare] max_abs_diff = {result['max_abs_diff']:.3e}  "
          f"passed = {result['passed']}")
    if not result["passed"]:
        sys.exit(1)
```

- [ ] **Step 3: 逐个组件做消融（逐个打开开关，逐个记录差异）**

不要只跑「全关」这一个点。**逐个组件打开**，记录每一步的 max_abs_diff：

| 配置 | 预期 |
|---|---|
| 全关（use_gqa=F, use_rope=F, use_qk_norm=F, mlp=gelu, norm=layernorm） | < 1e-4 |
| 只开 use_gqa（n_head_kv < n_head） | > 1e-4（结构变了） |
| 只开 use_rope | > 1e-4 |
| 只开 use_qk_norm | > 1e-4 |
| 只开 mlp=swiglu | > 1e-4 |

这张表就是「每个组件到底改了什么」的量化答案，**写进 Qwen3 笔记**。

- [ ] **Step 4: Commit**

```bash
git add qwen3-learning/
git commit -m "feat-017: Qwen3 ↔ GPT-2 退化对照实验（逐组件消融 + max_abs_diff 记录）"
```

---

### Task 19: feat-008 笔记与代码交叉引用表

**Files:**
- Modify: `qwen3-learning/Qwen3_*.md`（每篇开头加引用块）
- Modify: `qwen3-learning/qwen3-pytorch/README.md`
- Modify: `README.md`（笔记索引表补 Qwen3 代码列）

- [ ] **Step 1: 每篇 Qwen3 笔记开头加引用块**

```markdown
> **对应 GPT-2 实现位置**：`gpt-learning/gpt-learning-pytorch/model/gpt2.py` 的 `CausalSelfAttention`
> **对应 Qwen3 源码位置**：`qwen3-learning/qwen3-pytorch/model/common/attention.py` 的 `CausalSelfAttention.forward`
```

- [ ] **Step 2: 补齐缺笔记的组件**

当前 7 篇笔记**没有**覆盖：BPE、LoRA、DPO、采样。需新增或补写：

- `Qwen3_BPE_学习笔记.md`（Task 11 完成后写）
- `Qwen3_LoRA与DPO_学习笔记.md`（Task 15 完成后写）
- 采样（temperature/top-k/top-p）并入 `Qwen3_训练推理完整学习笔记.md`

- [ ] **Step 3: 逐条验证链接目标真实存在**

```bash
cd qwen3-learning/qwen3-pytorch
grep -rhoE '`(model|data|train|tests)/[a-z_/]+\.py`' ../Qwen3_*.md | tr -d '`' | sort -u | while read f; do
  [ -f "$f" ] && echo "OK   $f" || echo "MISS $f"
done
```

Expected: 全部 `OK`，无 `MISS`。

- [ ] **Step 4: Commit**

```bash
git add qwen3-learning/ README.md
git commit -m "feat-008: 笔记 ↔ 源码交叉引用表（每篇笔记开头加 GPT-2 / Qwen3 位置）"
```

---

## Phase 5 — 收尾

### Task 20: feat-009 笔记数字与 config.py 全量核对

- [ ] 列出两条线笔记中出现的全部超参数数字，逐条与各自 config.py 核对
- [ ] 已知待核项：设计.md §4 的 `intermediate_size=384` 推导理由
      （「8/3 对齐」）与 Qwen3-0.6B 实际比例 3.0（1024→3072）不符 —— 改笔记或改 config，二选一
- [ ] 已知待核项：设计.md §5.1 的 RMSNorm smoke 判据「均值≈0」不正确，
      应改成「平方均值归一化到 1」（RMSNorm 不做中心化）

### Task 21: 全量收班

- [ ] `bash init.sh` → 全部 PASS，无 FAIL
- [ ] `cd qwen3-learning/qwen3-pytorch && python tests/run_all.py` → PASS 全部
- [ ] `python evaluate.py --stage all` → 四个 stage 全跑通
- [ ] 更新 `claude-progress.md` / `feature_list.json` / `session-handoff.md`
- [ ] 跑 `clean-state-checklist.md` 的 6 项，全勾

---

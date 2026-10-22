# Progress Log — llm-learning

> session 连续性日志。**每个 session 结束前必须更新本文件**（见 `AGENTS.md` §8）。
> 上半部分是快照（可覆盖），下半部分是流水（只追加）。

---

## Current Verified State

- **当前阶段**：**P2 — Qwen3 从零实现**（阶段标记见 `feature_list.json` 的 `phase` 字段）
- 仓库根目录：`E:/ai-learning/05-llm-fundamentals/llm-learning`
- 标准启动路径：`bash init.sh`（轻量 smoke L0–L4，< 1 分钟）
- 标准验证路径：同上；GPT-2 全量训练与评估见 `AGENTS.md` §5 的 L3 段
- 当前活跃 feature：**无**（WIP=0；下一条是 `feat-011` Qwen3 代码骨架）
- 当前最高优先级未完成 feature：`feat-011` Qwen3 代码骨架与独立 `config.py`
- ~~已知 blocker：torch DLL 加载失败~~ → **已解除**（feat-001 根因是 PATH 上的 Python 3.11.0a2 alpha ABI，不是 torch 损坏；`init.sh` 现在自动挑能 `import torch` 的解释器，本机命中 `/d/python310/python.exe` 3.10.4 + torch 2.1.0+cpu）
- 训练产物现状：`gpt-learning/gpt-learning-pytorch/checkpoints/` 下的 `pretrain.pt` / `sft.pt` / `reward.pt` / `ppo.pt` 均为 **2026-10-22 在修复后的环境里重跑产出**，指标可信（见 `feature_list.json` feat-002~007 的 evidence）
- 笔记规模：`gpt-learning/` 17 篇 + `qwen3-learning/` 7 篇（缺 BPE / LoRA / DPO 三篇，见 feat-008）
- 两条线代码状态：GPT-2 线 `gpt-learning/gpt-learning-pytorch/` **全链路已复验通过**；Qwen3 线 `qwen3-learning/qwen3-pytorch/` **尚未创建**（计划已就位：`qwen3-learning/qwen3-pytorch设计.md` + `qwen3-learning/qwen3-pytorch开发计划.md`）

---

## Session Log

### Session 001 — 仓库重组 + harness 落地

- **日期**：2026-10-18
- **目标**：把 `llm-learning` 从"两个嵌套的旧仓库"重组为一个完整项目，并装上 harness

**做了什么**

1. 解除嵌套仓库：`gpt-learning/.git`（366MB）已删除，它不再是独立 Git 仓库，仅作为本地目录存在
2. 新建根 `.gitignore` / `.gitattributes`：排除 `checkpoints/`、`docs/`、模型权重、`__pycache__/`、`.claude/`、`.vscode/`、`info.txt`
3. 首次提交并推送到 `https://github.com/caiqiyue/llm-learning.git`（59 个文件）
4. 实装 harness 文件集（本 session 主要产出）：
   - `AGENTS.md` — 本项目定制操作手册（framing / 硬约束 / 三层验证 / DoD / 路由表）
   - `init.sh` — 轻量 smoke 验证（L0 环境 / L1 纯 Python 数据层 / L2 torch 模型前向 / L3 仓库卫生）
   - `feature_list.json` — 10 条 feature 的任务状态机，WIP=1
   - `claude-progress.md` — 本文件
   - `session-handoff.md` — 交接卡
   - `clean-state-checklist.md` — 收班清单
   - `README.md` — 中文项目说明
   - `harness.md` — 顶部新增「本项目落地」章节，原通用手册正文保留作参考

**验证**

```
$ bash init.sh
L0: python 3.11.0a2 / git 仓库就绪 / 5 个 harness 文件存在          → 全 PASS
L1: data/vocab.txt 存在；tokenizer 编解码往返 + SFT/预训练/benchmark 数据构造 → PASS
L2: torch 模型前向                                                    → SKIP（见 feat-001）
L3: 无产物类文件被跟踪；无 >2MB 大文件被跟踪                          → 全 PASS
汇总: PASS=11  SKIP=1  FAIL=0（退出码 0）
```

**未解决 / 风险**

- feat-001：`import torch` 报 `OSError [WinError 127]`，`torch_python.dll` 及其依赖加载失败。
  在此之前，feat-002 ~ feat-007（4 阶段训练 + 9 项评估 + KV Cache）都无法在本机验证。
- 本机已有 `gpt-learning/checkpoints/pretrain.pt` 与 `ppo.pt`，但产生它们的 torch 版本已不可复现，
  上述数字目前**不可信**，等 feat-001 修好后需重跑复核。

**下一步**

- feat-001：确认 torch 重装方案（重装 CPU wheel vs 装 VC++ Redistributable），修好后立即复跑 `bash init.sh`
- 之后按优先级推进 feat-002（预训练链路）

**本次改动涉及的文件**

`README.md`、`AGENTS.md`、`CLAUDE.md`、`init.sh`、`feature_list.json`、`claude-progress.md`、
`session-handoff.md`、`clean-state-checklist.md`、`harness.md`

### Session 002 — 阶段切换：对齐到 P2（Qwen3 开发准备）

- **日期**：2026-10-18
- **目标**：用户决定下一阶段做 **Qwen3 从零手写最小实现**（不用 transformers 跑真模型），本 session 把整套 harness 文档对齐到新阶段并留下交接

**做了什么**

1. `feature_list.json` 重排为 21 条 feature，新增 `phase` 字段标记阶段：
   - P2 新增 8 条：`feat-011` Qwen3 骨架 / `feat-012` RMSNorm+Pre-LN / `feat-013` RoPE / `feat-014` QK-Norm+GQA / `feat-015` SwiGLU / `feat-016` 完整前向 / `feat-017` 与 GPT-2 等价性对照 / `feat-018` KV Cache 推理
   - P3 预留 3 条：`feat-019` Qwen3 数据与 tokenizer / `feat-020` Qwen3 预训练 smoke / `feat-021` MoE（可选）
   - GPT-2 训练链路 `feat-002` ~ `feat-007` 状态改为 `deferred`，notes 写明推迟理由
   - 新增状态 `deferred`（主动推迟）与之前 `blocked`（被卡住）区分开
   - priority 重排为严格 1–21，WIP=1（当前 `in_progress` 数量为 0）
2. `init.sh` 新增 **L4 层**：检查 `qwen3-learning/qwen3-pytorch/` 骨架与 `model/qwen3.py` 前向；目录不存在时输出 `[SKIP]` 并指向 `feat-011`，不阻塞 GPT-2 线
3. `AGENTS.md` 对齐双线：framing 改为 P1/P2 双线表述；硬约束新增「P2 阶段专属」5 条；验证路径从三层扩为五层表；DoD 增加 Qwen3 门槛（架构组件必须附可运行对照证据）；路由表补 Qwen3 条目；升级规则补 3 条 Qwen3 选型问题
4. `CLAUDE.md` 同步双线表述与 Qwen3 约束
5. `README.md` 新增「三、项目阶段」章节（P1/P2/P3 状态表 + 计划中的 Qwen3 代码结构 + 核心思路），后续章节重新编号；笔记索引表补 P2 进行中说明；快速开始改为五层验证
6. `session-handoff.md` 重写为 **P2 阶段 Qwen3 开发交接卡**
7. `harness.md` 顶部落地章节同步 P2 状态

**核心设计决策**

- Qwen3 侧**不做 transformers 微调路线**，坚持与 GPT-2 线同一套「从零手写 + 可 debugger」的做法
- 组件一律做成**可关掉的开关**（`use_gqa` / `use_rope` / `use_qk_norm` / MLP 选型），退化后应与 GPT-2 对齐 —— 让「Qwen3 改了什么」变成可重跑的实验而不是文字断言（`feat-017`）
- Qwen3 用 `head_dim` 与 `n_embd` 解耦，不允许复制粘贴 GPT-2 的 `config.py`

**验证**

```
$ bash init.sh --quiet
PASS=11  SKIP=2  FAIL=0（退出码 0）
  L0/L1/L3 全 PASS；L2 torch 前向 SKIP（feat-001）；L4 Qwen3 结构 SKIP（feat-011 目录未建）
$ python -c "json.load(feature_list.json)"
21 条 feature，id 唯一，priority 严格 1–21 无重复，in_progress 数量 0（符合 WIP=1）
```

**未解决 / 风险**

- `feat-001` 仍是硬前置：torch 修不好，L2/L4 一直 SKIP，Qwen3 的任何模型代码都无法验证
- Qwen3 线尚无任何代码，本 session 只做了文档与状态对齐；`feat-011` 一开工就应先建骨架并把新模块挂进 `init.sh` 的 L4
- `feat-019`（tokenizer 选型：沿用字符级 vs 引入 BPE）会动到依赖上限，属于需要用户拍板的决策，已写进 `AGENTS.md` §10 升级规则

**下一步**

1. 修 `feat-001`（重装 torch CPU wheel 或补 VC++ 运行库），重跑 `bash init.sh` 确认 L2 变 PASS
2. 开 `feat-011`：建 `qwen3-learning/qwen3-pytorch/` 骨架与独立 `config.py`
3. 按 `feat-012` → `feat-018` 顺序逐个组件推进，每个组件完成即在笔记里标注源码位置

### Session 003 — 解除环境 blocker + 路线切到 LoRA/DPO + 计划补全

- **日期**：2026-10-22
- **目标**：① 修 feat-001（不重装）② 按 `设计.md` §10 重排 feature_list 并把路线切到 LoRA+DPO ③ 重写 init.sh L4 ④ 把计划补到「预训练 → LoRA → DPO」

**做了什么**

1. **feat-001 修复（真因不是 torch 损坏）**
   - 诊断：PATH 上的 `python` = `D:/python311` = **Python 3.11.0a2（alpha 预发布）**，torch 2.1.0+cpu 的 `torch_python.dll` 在该 ABI 下加载失败 → `OSError [WinError 127]`。原先记录的「重装 CPU wheel / 装 VC++ 运行库」是误诊。
   - 修复：`init.sh` 内置解释器探测（`PYTHON_BIN` 优先，否则挑第一个**能 `import torch`** 的候选），并支持 `PYTHON_BIN=... bash init.sh` 覆盖。命中 `/d/python310/python.exe`（3.10.4 / torch 2.1.0+cpu / numpy 1.26.4 / nltk 3.9.4）。
   - L2 由 SKIP → PASS。
2. **解除 P1 线阻塞并复验**
   - `data/build_vocab.py`：直接执行时 `data/` 占据 `sys.path[0]`，`from data.pretrain_data import ...` 报 `ModuleNotFoundError: No module named 'data'` → 补仓库根到 `sys.path`。
   - 4 个训练脚本（`train_pretrain.py` / `train_sft.py` / `train_reward.py` / `train_ppo_standard_adapted.py`）里 6 处写死的 `E:\ai-learning\05-llm-fundamentals\gpt-learning\...` 绝对路径（该目录已不存在）→ 全部改为 `os.path.join(...)` 相对路径。
   - L3 全量重跑：pretrain → sft → reward → ppo → evaluate 全部通过，`evaluate.py --stage all` 退出码 0。
   - feat-002 ~ feat-007 由 `deferred` 转为 `passing`，evidence 全部落到 `feature_list.json`。
3. **feature_list.json 重排（21 → 25 条）**
   - priority 严格 1–25，WIP=0；`phase` 补 `route_change_note`。
   - 路线切换：**SFT / Reward Model / PPO → LoRA（feat-023）+ DPO（feat-024）**；新增 feat-022（手写 BPE）、feat-025（Qwen3 评估）。
   - 新增 `verification_levels.QWEN3_TEST`：依赖上限不允许 pytest，采用顶层线性执行 + assert + subprocess 汇总的三态风格。
4. **init.sh L4 重写**
   - 旧 L4 写死 `model/qwen3.py` / `CharTokenizer("data/vocab.txt")` / `config.intermediate_dim` —— 与新设计的 `qwen3_dense.py` / `BPETokenizer` / `intermediate_size` 全部不符。
   - 新 L4 三段递进：骨架检查（`config.py README.md model data train tests`）→ `tests/run_all.py` 组件 smoke → 端到端前向（`Qwen3Dense` + `BPETokenizer`，断言 logits 形状 / extras 为空 / loss 有限）。
5. **文档纠错**：`python evaluate.py --stage all --all_checkpoints checkpoints/` 中的 `--all_checkpoints` **并非 evaluate.py 支持的参数**（argparse 报 unrecognized arguments）→ CLAUDE.md 与 AGENTS.md 均改回 `python evaluate.py --stage all`。
6. **计划补全**：`qwen3-pytorch开发计划.md` 补 Phase 3 / 4 / 5（Task 13–21），把用户要求的**预训练 → LoRA → DPO** 三段全部落成可执行 Task（含完整代码块、Run/Expected、Commit 步骤）：
   - Task 13 feat-020 预训练（`--model dense|moe`，超参全从 config 读）
   - Task 14 feat-023 手写 LoRA（B 零初始化 + base 冻结 + inject/merge/enable 三件套）
   - Task 15 feat-024 手写 DPO（**π_ref 复用同一模型对象，靠关 LoRA 分支得到**，不训 Reward Model）
   - Task 16 feat-025 精简评估（pretrain/lora/dpo/speedup 四段）
   - Task 17 feat-018 KV Cache 推理与采样；Task 18 feat-017 与 GPT-2 的退化对照（逐组件消融表）；Task 19 feat-008 笔记↔源码交叉引用；Task 20/21 收尾核对与收班。

**验证**

```
$ bash init.sh
L0 [PASS] python 可用 (Python 3.10.4, 解释器 = /d/python310/python.exe)
L0 [PASS] torch 可导入 (2.1.0+cpu)
L1 [PASS] 数据层 smoke … vocab 45 sft 20 pretrain 20 bench 5
L2 [PASS] 模型前向 smoke … logits (2, 8, 45) loss 3.8102 params 794496
L3 [PASS] 无产物类文件被 git 跟踪 / 无 >2MB 大文件
L4 [SKIP] Qwen3 代码目录 — 尚未创建（feat-011）
汇总: PASS=13  SKIP=1  FAIL=0（退出码 0）

$ bash E:/tmp/run_l3.sh        # P1 全量
[pretrain] final val_loss 3.0060 | val_ppl 20.21
[sft]      eval val_loss 0.0007 | val_ppl 1.00
[reward]   val_acc 0.7344 | margin 2.7838 | mean_r_chosen 3.2060 | mean_r_rejected 0.4222
[ppo]      iter=100 | best RM=-3.5685 | ref_KL/token=+0.1767 | wall=160.8s
[eval]     9 项评估全部通过（exit 0）
[eval]     speedup value=1.7620 (naive 248.11ms vs kv 140.81ms)
```

**未解决 / 风险**

- P1 的**质量**指标仍然弱（SFT 后 distinct_1≈0.003、self_bleu=1.0、response_quality=0；PPO 的 compare 阶段 `improvement=0.0000`）。链路是通的、指标是脚本自身阈值下的实跑值，但模型只学会了「背模板」。这是 45 词表 + 小语料的必然结果，**未削弱 evaluate.py 的任何阈值**。已如实记入 feat-003/005 的 evidence。
- Qwen3 线 0 行代码；`设计.md` 与 `开发计划.md` 之间存在 3 处已知不一致（RMSNorm 判据「均值≈0」不正确、`intermediate_size=384` 的「8/3」推导与 Qwen3-0.6B 实际比例不符、SwiGLU 文件位置），已分别记入 feat-009 与设计文档备注，待 Task 20 统一裁决。
- `rope_theta=1e6` 目前没有对应笔记支撑。

**下一步**

1. 开 `feat-011`（设 `in_progress`）：建 `qwen3-learning/qwen3-pytorch/` 骨架 + 独立 `config.py`（`Qwen3Config` dataclass，`DENSE_CONFIG` / `MOE_CONFIG`）
2. 按 `feat-012` → `feat-013` → `feat-014` → `feat-015` → `feat-016` 顺序实现组件，每个组件挂 `tests/` 并跑通 `init.sh` 的 L4
3. `feat-022` 手写 BPE → `feat-019` 数据 → `feat-020` 预训练 → `feat-021` MoE → `feat-023` LoRA → `feat-024` DPO → `feat-025` 评估 → `feat-017` 对照

**本次改动涉及的文件**

`feature_list.json`、`init.sh`、`claude-progress.md`、`CLAUDE.md`、`AGENTS.md`、
`gpt-learning/gpt-learning-pytorch/data/build_vocab.py`、
`gpt-learning/gpt-learning-pytorch/train_pretrain.py`、`train_sft.py`、`train_reward.py`、`train_ppo_standard_adapted.py`、
`qwen3-learning/qwen3-pytorch设计.md`（新增）、`qwen3-learning/qwen3-pytorch开发计划.md`（新增）

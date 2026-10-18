# AGENTS.md — llm-learning Harness 操作手册

> 本文件是 **给 AI agent 读的操作手册**，README.md 是给人读的项目说明，两者不要混写。
> 通用方法论见 [`harness.md`](harness.md)（5 子系统 / 任务拆解 / 反模式全集）。

---

## 0. 项目是什么（framing，承重段）

**llm-learning 是一个 LLM 学习仓库**：主线是「从零手写 tiny GPT-2 → 预训练 → SFT → Reward Model → PPO 全流程 + 9 项评估」，副线是「Qwen3 现代架构笔记」。它同时是**笔记库**和**可运行代码库**，两者必须对得上。

**Agent 在本项目应优先优化什么（按优先级）：**

1. MUST — **笔记与代码一致**。笔记里出现的公式、超参、代码片段必须能在 `gpt-learning-pytorch/` 里找到对应出处；不一致就是错误。
2. MUST — **可复现**。任何人 clone 下来按 README 跑，CPU 上 10 分钟内能跑完 4 阶段训练 + 评估。
3. SHOULD — **可读性优先于性能**。这是教学代码，宁可慢 10 倍，也要让每一行都能 debugger 跟进去。
4. SHOULD — 注释与 docstring 用中文，风格对齐 `gpt-learning-pytorch/model/gpt2.py`。
5. MUST NOT — 引入重量级新依赖。当前依赖上限：`torch` / `numpy` / `nltk`，加依赖前先在 README 里说明理由。

---

## 1. Startup Workflow（每个 session 开头必跑 6 步）

```bash
pwd                                    # 1. 确认在 llm-learning 根目录
cat claude-progress.md                # 2. 读当前已验证状态
cat feature_list.json                 # 3. 读任务状态机
git log --oneline -5                   # 4. 看最近 5 次 commit
bash init.sh                          # 5. 跑轻量验证，确认 baseline 健康
git status --short                    # 6. 工作区是否干净
```

**任一步失败 → 不开新 feature，先修 baseline**，并在 `feature_list.json` 里新增一行记录这个故障（status: `blocked`）。

---

## 2. Working Rules（干活时的 6 条铁律）

| # | 规则 | 违反后果 |
|---|------|----------|
| 1 | **WIP = 1** — 一次只推进一个 feature（`in_progress` 只能有一行） | 多任务半成品 |
| 2 | **不宣布完成除非有可运行证据** — 贴出实际命令输出 | 假 passing |
| 3 | **不靠自评验收** — 同一份代码自己打分一定偏松，必须有可重跑的日志 | Generation-Evaluation Asymmetry |
| 4 | **不删/不改弱测试与评估阈值让结果好看** | 评估腐烂 |
| 5 | **不碰 active feature 之外的文件**（顺手重构是最常见的 scope creep） | 难以 review |
| 6 | **用仓库文件当唯一记录**，不依赖聊天记录 | 跨 session 失忆 |

**实时监控 4 个信号：**

- 同一 verify 连续失败 ≥ 3 次 → 转 `blocked` + 写 `notes`
- 当前 feature 卡住 ≥ 20 分钟无进展 → 转 `blocked` + 写 `notes`
- 本 session 已开第 2 个 feature → 立刻关掉它
- `init.sh` 出现 FAIL → 先修环境，不要在此之上做任何实现

---

## 3. Hard Constraints（本项目专属，不可违反）

```
MUST NOT  把 checkpoints/、docs/、.claude/、.vscode/、*.pt、*.pth 提交进 git（.gitignore 已覆盖，提交前用 git status 核对）
MUST NOT  在训练脚本里硬编码超参 —— 一律从 config.py 读
MUST NOT  删除或降低 evaluate.py 的评估阈值/指标来让结果"更好看"
MUST NOT  引入 torch/numpy/nltk 之外的重量级依赖（lightning / transformers 等）而不先在 README 说明
MUST      Python 代码保留中文 docstring + type hint，风格对齐 model/gpt2.py
MUST      新增学习笔记遵循 gpt-learning/ 下的编号命名（N.主题.md），Qwen3 笔记遵循 Qwen3_主题_学习笔记.md
MUST      改动 config.py 中任何超参后，说明它影响哪几章笔记
MUST      commit 前跑过 bash init.sh，且结果贴进 commit message 或 evidence
MUST NOT  引入 TensorFlow 作为主线（TF 子项目已从仓库移除，只保留 PyTorch）
MUST NOT  把学习者的个人进度/临时 TODO 混进对外笔记文件
```

---

## 4. Required Artifacts（这些文件是 system of record）

| 文件 | 作用 | 何时更新 |
|------|------|----------|
| `feature_list.json` | 任务状态机，唯一事实来源 | 开 feature、转状态、填 evidence |
| `claude-progress.md` | session 日记 + 当前已验证状态 | 每个 session 结束 |
| `session-handoff.md` | 跨 session 交接卡 | context 用满 / 长 session 结束 |
| `init.sh` | 标准启动 + 轻量验证路径 | 验证路径变化时 |
| `evidence/` | 验证输出存档（可 git tracked，也可 gitignore） | 每次验证通过后 |

---

## 5. 本项目的 Verification 路径（三层）

**不跑全量训练**做 baseline 验证 —— 轻量 smoke 就够；全量训练是**按需**跑的。

```bash
# L0+L1+L2：轻量 smoke（< 1 分钟，每个 session 必跑）
bash init.sh

# L3：全量训练与评估（CPU 约 10 分钟，只在动到训练/评估代码时跑）
cd gpt-learning/gpt-learning-pytorch
python data/build_vocab.py
python train_pretrain.py && python train_sft.py && python train_reward.py && python train_ppo.py
python evaluate.py --stage all --all_checkpoints checkpoints/
```

**注意**：L2（torch 相关）在本机可能因 `torch_python.dll` 加载失败而 **SKIP** 而非 FAIL —— 这是已知的本机环境问题（见 `feature_list.json` feat-001），不阻塞纯 Python 数据层的工作。

---

## 6. Definition of Done（本项目的 4 条 AND）

一个 feature 只有**同时满足**下面 4 条才允许标 `passing`：

```
[ ] 目标行为已实现（代码改动 或 笔记改动 二选一，明确是哪个）
[ ] 对应层级的验证实际跑过（代码改动 → init.sh + 受影响的训练/评估脚本；笔记改动 → 抽样核对 ≥3 处与代码一致）
[ ] evidence 非空：命令 + 实际输出摘要已写进 feature_list.json
[ ] 从干净环境出发不需要任何手工修补（下一个 session 能直接继续）
```

**笔记类 feature 的额外门槛**：笔记中的数字/超参必须与 `config.py` 逐一核对过；引用的代码行号或函数名必须真实存在。

---

## 7. Routing Map（信息去哪找）

| 想了解 | 去读 |
|--------|------|
| harness 通用方法论、5 子系统、反模式全集 | [`harness.md`](harness.md) |
| 代码该按什么顺序读、读到哪了 | [`gpt-learning/代码阅读顺序与进度.md`](gpt-learning/代码阅读顺序与进度.md) |
| 超参含义与取值依据 | [`gpt-learning/GPT2训练数学公式与超参详解.md`](gpt-learning/GPT2训练数学公式与超参详解.md) |
| 训练链路怎么跑、评估有哪些指标 | [`gpt-learning/gpt-learning-pytorch/README.md`](gpt-learning/gpt-learning-pytorch/README.md) |
| 当前在做什么、下一步做什么 | [`feature_list.json`](feature_list.json) + [`claude-progress.md`](claude-progress.md) |

---

## 8. End of Session（5 步，按顺序）

```
1. 更新 claude-progress.md：写清本 session 验证了什么、结果是什么
2. 更新 feature_list.json：翻状态 + 填 evidence + 记录 blocked/notes
3. 记录未解决风险到 session-handoff.md
4. commit（message 必须引用 feature ID，格式见 harness.md §5.3）
5. 跑 clean-state-checklist.md 的 6 项，全勾才收班
```

---

## 9. Context Pressure Protocol

| 阈值 | 动作 |
|------|------|
| **50%** | 每做一个决定就顺手写进 `claude-progress.md`，不要攒到最后 |
| **70%** | 停止开新 feature；当前 feature 能收尾就收尾，否则转 `blocked`；写完整 `session-handoff.md` |
| **85%** | 立即执行交班 5 步，提交干净状态，下个 session 从 progress + handoff + git log 重建 |
| **95%** | 只写一段话 handoff，不验证不修复，立刻 commit 退出 |

---

## 10. Escalation（什么时候该问用户）

- **架构决策**（换模型规模、换训练框架、引入新依赖）→ 问用户
- **笔记与代码冲突**（代码变了但不知笔记该怎么改）→ 问用户，不要自行取舍
- **超参调整**（lr / 层数 / block_size 改动）→ 先在 README 说明影响，问用户确认
- **重复验证失败** → 转 `blocked`，写清失败现象，在 progress 里问
- **范围不明确** → 回到 active feature 的 `user_visible_behavior` 逐字读

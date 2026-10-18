# Progress Log — llm-learning

> session 连续性日志。**每个 session 结束前必须更新本文件**（见 `AGENTS.md` §8）。
> 上半部分是快照（可覆盖），下半部分是流水（只追加）。

---

## Current Verified State

- 仓库根目录：`E:/ai-learning/05-llm-fundamentals/llm-learning`
- 标准启动路径：`bash init.sh`（轻量 smoke，< 1 分钟）
- 标准验证路径：同上；全量训练与评估见 `AGENTS.md` §5 的 L3 段
- 当前活跃 feature：**无**（feat-001 被环境问题卡住，等待决定如何修 torch）
- 当前最高优先级未完成 feature：`feat-001` 让本机 torch 可正常导入
- 已知 blocker：torch DLL 加载失败（`WinError 127`），导致 `init.sh` 的 L2 层 SKIP，全量训练/评估暂不可跑
- 训练产物现状：`gpt-learning/checkpoints/` 下有 `pretrain.pt`、`ppo.pt`（旧版本，本机当前 torch 不可用，无法复验）
- 笔记规模：`gpt-learning/` 17 篇笔记 + `qwen3-learning/` 7 篇笔记

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

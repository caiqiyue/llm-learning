# CLAUDE.md — llm-learning

> 本文件是 `AGENTS.md` 的 Claude Code 入口版，**规则本体在 [`AGENTS.md`](AGENTS.md)**。
> 两者冲突时以 `AGENTS.md` 为准；本文件只放 Claude Code 特有的启动约定。

## 开工前必做（Claude Code 会话开始时执行）

```bash
pwd && cat claude-progress.md && cat feature_list.json && git log --oneline -5
bash init.sh          # 退出码非 0 时不要开新 feature，先修 baseline
```

## 这个仓库是什么

LLM 学习仓库：主线「从零手写 tiny GPT-2 → 预训练 → SFT → Reward Model → PPO 全流程 + 9 项评估」，副线「Qwen3 现代架构笔记」。**笔记和代码必须对得上**，这是第一优先级。

## 本项目专属硬约束（摘要，全文见 AGENTS.md §3）

- 不提交 `checkpoints/`、`docs/`、权重、缓存、IDE 配置
- 超参一律从 `config.py` 读，训练脚本里不硬编码
- 不删/不改弱 `evaluate.py` 的指标和阈值
- 依赖上限 `torch` / `numpy` / `nltk`，加新依赖先问
- Python 代码保留中文 docstring + type hint，对齐 `model/gpt2.py`
- 改 `config.py` 要说明影响哪几章笔记
- commit 前跑过 `bash init.sh`

## 工作节奏

- **WIP = 1**：`feature_list.json` 里同时只能有一个 `in_progress`
- **不宣布完成除非有可运行证据**：把命令和实际输出贴出来
- 改训练/评估代码 → 跑 L3 全量（CPU 约 10 分钟）；只改笔记 → 跑 `init.sh` + 抽样核对 ≥3 处
- 收班 5 步：更新 progress → 更新 feature_list → 记 handoff → commit → 跑 `clean-state-checklist.md`

## 已知环境问题

本机 `import torch` 报 `WinError 127`（`torch_python.dll` 加载失败），`init.sh` 的 L2 层会 **SKIP**。
这不是新引入的故障，见 `feature_list.json` 的 `feat-001`。纯 Python 数据层（tokenizer / 数据构造）不受影响。

## 常用命令

| 用途 | 命令 |
|------|------|
| 轻量验证 | `bash init.sh` |
| 代码目录 | `cd gpt-learning/gpt-learning-pytorch` |
| 重建词表 | `python data/build_vocab.py` |
| 全量训练 | `python train_pretrain.py && python train_sft.py && python train_reward.py && python train_ppo.py` |
| 全量评估 | `python evaluate.py --stage all --all_checkpoints checkpoints/` |

## 延伸阅读

- `AGENTS.md` — 操作手册（硬约束、验证路径、DoD、路由表）
- `harness.md` — harness 通用方法论手册（5 子系统 / 任务拆解 / 反模式全集）
- `README.md` — 项目总览与笔记索引
- `gpt-learning/代码阅读顺序与进度.md` — 代码该怎么读、读到哪了

# Session Handoff — llm-learning

> 跨 session 交接卡。**context 用满（≥70%）或长 session 结束前必须填写**（见 `AGENTS.md` §9）。
> 规则：用指针代替贴全文；不带密钥；一次 session 只关心当前任务。

---

## Verified Now（当前哪些是已验证的）

- **能用**：`bash init.sh` 的 L0 / L1 / L3 三层全绿（Python 3.11 可用、纯 Python 数据层可用、仓库忽略规则生效）
- **不能用**：torch 相关的一切 —— `import torch` 报 `WinError 127`（feat-001）
- **已跑过的验证命令**：`bash init.sh`（2026-10-18，PASS=8 / SKIP=1 / FAIL=0）

## Changed This Session（本次 session 改了什么）

- 仓库重组：删除嵌套的 `gpt-learning/.git`，整个 `llm-learning` 成为一个仓库
- 新增根 `.gitignore` / `.gitattributes`，排除 `checkpoints/`、`docs/`、权重、缓存、IDE 配置
- 首次提交并推送到 `https://github.com/caiqiyue/llm-learning.git`
- 落地 harness 文件集：`AGENTS.md` / `init.sh` / `feature_list.json` / `claude-progress.md` / `session-handoff.md` / `clean-state-checklist.md` / `README.md`，并在 `harness.md` 顶部加了本项目落地章节

## Broken Or Unverified（坏的 / 没验证的）

- **已知缺陷**：`import torch` 失败 → `OSError [WinError 127] Error loading torch_python.dll`。torch 安装不完整或缺 VC++ 运行库。
- **未验证路径**：`train_pretrain.py` / `train_sft.py` / `train_reward.py` / `train_ppo.py` / `evaluate.py` / `inference.py` 全部**没有在当前环境跑过**
- **不可信数据**：`gpt-learning/checkpoints/pretrain.pt`、`ppo.pt` 由旧 torch 产出，当前环境无法复验其指标，笔记里引用的相关数字在复跑前不要当作事实
- **风险**：feat-002 ~ feat-007 全部依赖 feat-001；feat-001 不解决，训练类 feature 无法推进

## Next Best Step（下一步做什么）

- **最高优先级未完成 feature**：`feat-001` 让本机 torch 可正常导入
- **为什么是它**：它是所有训练/评估类 feature 的前置条件，且不依赖任何其他 feature
- **什么算通过**：`python -c "import torch"` 有版本输出；`bash init.sh` 的 L2 层从 SKIP 变 PASS
- **这一步不许动的**：不改任何训练代码的超参或逻辑；不改笔记中的结论性数字

## Commands（常用命令）

- 启动 + 轻量验证：`bash init.sh`
- 安静模式（只看汇总）：`bash init.sh --quiet`
- 进入代码目录：`cd gpt-learning/gpt-learning-pytorch`
- 重建词表：`python data/build_vocab.py`
- 全量训练（L3，CPU 约 10 分钟）：
  `python train_pretrain.py && python train_sft.py && python train_reward.py && python train_ppo.py`
- 全量评估：`python evaluate.py --stage all --all_checkpoints checkpoints/`
- 看当前状态：`cat feature_list.json` + `cat claude-progress.md`

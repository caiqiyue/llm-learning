# Progress Log — llm-learning

> session 连续性日志。**每个 session 结束前必须更新本文件**（见 `AGENTS.md` §8）。
> 上半部分是快照（可覆盖），下半部分是流水（只追加）。

---

## Current Verified State

- **当前阶段**：**P2 — Qwen3 从零实现**（阶段标记见 `feature_list.json` 的 `phase` 字段）
- 仓库根目录：`E:/ai-learning/05-llm-fundamentals/llm-learning`
- 标准启动路径：`bash init.sh`（轻量 smoke L0–L4，< 1 分钟）
- 标准验证路径：同上；GPT-2 全量训练与评估见 `AGENTS.md` §5 的 L3 段
- 当前活跃 feature：**无**（feat-001 被环境问题卡住；P2 尚未开工）
- 当前最高优先级未完成 feature：`feat-001` 让本机 torch 可正常导入 → 之后 `feat-011` Qwen3 代码骨架
- 已知 blocker：torch DLL 加载失败（`WinError 127`），导致 `init.sh` 的 L2/L4 层 SKIP，**任何模型代码都无法验证**
- 训练产物现状：`gpt-learning/checkpoints/` 下有 `pretrain.pt`、`ppo.pt`（旧 torch 版本产出，当前环境不可复验，指标暂不可信）
- 笔记规模：`gpt-learning/` 17 篇 + `qwen3-learning/` 7 篇
- 两条线代码状态：GPT-2 线 `gpt-learning/gpt-learning-pytorch/` 代码齐全待复验；Qwen3 线 `qwen3-learning/qwen3-pytorch/` **尚未创建**

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

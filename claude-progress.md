# Progress Log — llm-learning

> session 连续性日志。**每个 session 结束前必须更新本文件**（见 `AGENTS.md` §8）。
> 上半部分是快照（可覆盖），下半部分是流水（只追加）。

---

## Current Verified State

- **当前阶段**：**P2 — Qwen3 从零实现**（阶段标记见 `feature_list.json` 的 `phase` 字段）
- 仓库根目录：`E:/ai-learning/05-llm-fundamentals/llm-learning`
- 标准启动路径：`bash init.sh`（轻量 smoke L0–L4，< 1 分钟）
- 标准验证路径：同上；GPT-2 全量训练与评估见 `AGENTS.md` §5 的 L3 段
- 当前活跃 feature：**无**（WIP=0；下一条是 `feat-023` 手写 LoRA）
- 当前最高优先级未完成 feature：`feat-023` Qwen3 手写 LoRA（→ feat-024 手写 DPO → feat-025 评估）
- ~~已知 blocker：torch DLL 加载失败~~ → **已解除**（feat-001 根因是 PATH 上的 Python 3.11.0a2 alpha ABI，不是 torch 损坏；`init.sh` 现在自动挑能 `import torch` 的解释器，本机命中 `/d/python310/python.exe` 3.10.4 + torch 2.1.0+cpu）
- 训练产物现状：`gpt-learning/gpt-learning-pytorch/checkpoints/` 下的 `pretrain.pt` / `sft.pt` / `reward.pt` / `ppo.pt` 均为 **2026-10-22 在修复后的环境里重跑产出**，指标可信（见 `feature_list.json` feat-002~007 的 evidence）
- 笔记规模：`gpt-learning/` 17 篇 + `qwen3-learning/` 7 篇（缺 BPE / LoRA / DPO 三篇，见 feat-008）
- 两条线代码状态：GPT-2 线 `gpt-learning/gpt-learning-pytorch/` **全链路已复验通过**；Qwen3 线 `qwen3-learning/qwen3-pytorch/` **组件层 + 数据层 + 预训练已就位**（RMSNorm / RoPE / KVCache / QK-Norm+GQA 注意力 / SwiGLU / Qwen3Dense / Qwen3MoE / byte-level BPE / 语料与偏好数据 / `--model dense|moe` 预训练脚本），`bash init.sh` L4 已有「组件 smoke + 前向 smoke + 训练步 smoke」三条；**LoRA / DPO / 评估侧尚未开工**

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

---

### Session 004 — Qwen3 线组件层从零实现（feat-011 → feat-022）

- **日期**：2026-10-22
- **目标**：按 `qwen3-pytorch开发计划.md` 把 Qwen3 线的**组件层**全部手写出来，每个组件都要有可运行的对照/消融证据（仓库硬约束：不能只靠文字断言）

**做了什么**（每条一个独立 commit，WIP=1，测试先失败后通过）

1. **feat-011 骨架**：`qwen3-learning/qwen3-pytorch/` 落地，`config.py` 用 `Qwen3Config` dataclass 统一 dense / MoE 两套超参（不复制 GPT-2 线的 config）。
2. **feat-012 RMSNorm**：无 bias、无均值中心化、可学习 gain；smoke 断言「平方均值归一化到 1」而不是「均值≈0」。
3. **feat-013 RoPE**：`theta=1e6` + `position_ids` 增量位置接口 + 旋转等变性 smoke。
4. **feat-014a KVCache**：按 `n_head_kv` 分配，GQA 下占用 = MHA 的 1/4。
5. **feat-014b 注意力**：QK-Norm + GQA + RoPE 三合一，**QK-Norm 在 RoPE 之前**。
6. **feat-015 SwiGLU**：三矩阵门控（无 bias），`8/3` 中间维对齐规则 + 参数量对照 smoke。
7. **feat-016 Qwen3Dense**：完整前向，**恒定 4 元组返回** `(logits, loss, kv_cache, extras)` + 权重绑定 + 无 pos_emb。
8. **feat-021 Qwen3MoE**：Router top-k + 真稀疏专家（forward hook 证明只算 N·K 行）+ 负载均衡 aux_loss。
9. **feat-022 byte-level BPE**：`data/bpe.py` + `data/tokenizer.py` + `data/build_vocab.py`，零新依赖。

**四条可迁移的实测结论**（已写进笔记，不是拍脑袋）

- **QK-Norm 在 RoPE 前 vs 后，用均匀 gain 测不出来**：两者数值几乎相同（max|diff| ≈ 3.6e-07），因为 `rotate_half` 把第 i 维和第 i+half 维配对，逐维 gain 与旋转不对易。**只有非均匀 gain 才暴露差异（7.38 ~ 14.9）**。⇒ 任何用 `gain.mul_(3.0)` 这类均匀增益的测试都**证明不了顺序**。
- **SwiGLU 与 GPT-2 4×-GELU 的参数量差 = `矩阵数 × H × I`**。I 严格取 8/3·H 时两者**完全相等**（都是 8H²，这正是 8/3 规则的目的）；本仓库把 341.33 向上取整到 384（=3×128），比值 = 9/8 = **1.125×**，超出部分全部来自取整。另注：Qwen3-0.6B 实际用的是 **3×hidden**，8/3 是 LLaMA 的规则。
- **aux_loss 的量纲有 Switch 版和 Mixtral 版两套**：Switch 论文版 `L=N·Σf_i p_i` 均匀时 = 1；本仓库实现对 tokens×k 槽位求均值（Σf=1）、系数是专家数 E，均匀时同样 = **1**；HF Mixtral 版把槽位计数除以 N（Σf=K），均匀时 = **K**（本仓库 K=2 → 2）。三者只差归一化，**与 HF 的数值不能直接比大小（要乘 K）**。已写进 `Qwen3_MoE_学习笔记.md` §19.1。
  > ⚠️ 这条结论在 Session 004 写成「均匀时 = 1/K（0.5）」，**是错的**，已在 Session 005 用预训练日志（aux_loss 稳定在 1.0 附近）实测纠正。
- **RoPE 是相对位置编码**：把所有绝对位置整体平移，注意力输出**不变**（max|diff| < 1e-5）；只有改变**相对距离**才会改变输出。初稿测试「不同绝对位置应产生不同输出」本身是错的，已改成「整体平移不变 + 插入 token 改变相对距离则输出改变」两条。

**feat-022 修正的 5 个真 bug**（开发计划 Task 11 的初稿设计里全都有）

1. **id 撞车** —— 字节 token 占 id 1..256，合并 token 却从 256 开始 → **字节 0xFF 被静默吞掉**。改为 `BASE_VOCAB_SIZE=257 / MERGE_OFFSET=257`，并加回归断言。
2. **`merges.txt` 解析不了** —— 空格/换行本身就是字节 token，`line.split(" ")` 必错。改用 GPT-2 的 byte→unicode 双射（`Ġ`=空格、`Ċ`=换行），格式与 HF tokenizers 兼容。
3. **测试断言类型写错** —— `vocab["<|endoftext|>"] == 0`（vocab 是 `{id: bytes}`）→ 改为 `vocab[0] == b"<|endoftext|>"`。
4. **训练侧与编码侧 id 差 1** —— `train_bpe` 用原始字节值、`encode`/词表用字节值+1 → 合并规则全错（39B → 37 tokens）。修正后同句 → 6 tokens。
5. **训练太慢 + 压缩断言的适用范围** —— 改为预分词后按块去重计数（训练单元 = 编码单元，语料以句子列表返回），十几分钟 → 秒级；压缩断言只能用**语料内**句子。

**验证**

```
$ python data/build_vocab.py
[build_vocab] 语料 20000 句 / 785065 字节
[build_vocab] 词表 778（合并 521 次，目标 1000）
[build_vocab] 压缩比：19641 字节 -> 2767 tokens (7.10 字节/token)

$ python tests/test_bpe.py
[压缩] 语料内句子 39B -> 15 tokens (2.60 字节/token)
[真实词表] vocab=778 merges=521 'The student trained ' 39B -> 6 tokens
test_bpe OK  vocab=293 merges=36 'The student trained ' 39B -> 15 tokens

$ python tests/run_all.py
  [PASS] test_attention.py … [PASS] test_bpe.py … [PASS] test_dense.py … [PASS] test_kv_cache.py
  [PASS] test_moe.py … [PASS] test_rmsnorm.py … [PASS] test_rope.py … [PASS] test_swiglu.py
[run_all] PASS=8  FAIL=0  TOTAL=8

$ bash init.sh
L4 [PASS] Qwen3 前向 smoke（logits 形状 + loss 有限 + 参数量）[vocab.json] logits (2, 8, 778) loss 6.6282 params 985,856
汇总: PASS=22  SKIP=0  FAIL=0
```

**未解决 / 风险**

- **BPE 词表只有 778 < 目标 1000**：合成语料的可合并对用尽（521 次合并后没有频次 ≥ 2 的对），属于语料多样性问题，不是实现 bug（`config.py` 已注明「实际大小由训练语料决定」）。
- **BPE 语料与预训练语料是两个生成器**：`build_vocab.py` 与（待写的）`data/pretrain_data.py` 各自合成语料，词表对预训练语料的覆盖率会打折。Task 12 需把语料生成器收敛成**唯一真值**（建议放到 `data/pretrain_data.py`，`build_vocab.py` 改为 import 它）并重新生成词表。
- **MoE 的「参数多 N 倍、计算只多 k 倍」在本仓库不成立**：专家更窄（128 vs 384），全模型只多 1.21×。已在笔记 §25.1 写清前提条件。
- 笔记仍有缺口：`Qwen3_BPE_学习笔记.md` / `Qwen3_LoRA与DPO_学习笔记.md` 未写（按计划在 feat-008 / Task 19 补）。

**下一步**

1. 开 `feat-019`：写 `data/pretrain_data.py` + `data/preference_data.py`（语料生成器唯一真源，DPO 偏好对至少 3 种「变差」策略）
2. `feat-020` 预训练 smoke（`--model dense|moe`，超参全从 config 读）
3. `feat-023` 手写 LoRA → `feat-024` 手写 DPO（π_ref 靠关 LoRA 分支得到）→ `feat-025` 评估
4. `feat-018` KV Cache 推理与采样 → `feat-017` 与 GPT-2 的退化对照 → `feat-008` 笔记↔源码交叉引用
5. `feat-009` 笔记数字全量核对 → 收班

**本次改动涉及的文件**

`qwen3-learning/qwen3-pytorch/`：`config.py`、`model/common/{rmsnorm,rope,kv_cache,attention,mlp}.py`、
`model/{qwen3_dense,qwen3_moe}.py`、`data/{bpe,tokenizer,build_vocab}.py`、
`tests/{test_rmsnorm,test_rope,test_kv_cache,test_attention,test_swiglu,test_dense,test_moe,test_bpe}.py`、
`tests/run_all.py`、`data/{vocab.json,merges.txt}`（生成产物）、
`qwen3-learning/qwen3-pytorch开发计划.md`（Task 7–11 就地纠错）、`qwen3-learning/qwen3-pytorch设计.md`（§4 中间维说明）、
`qwen3-learning/Qwen3_{SwiGLU_MLP,MoE}_学习笔记.md`（新增 §2.1 / §19.1 / §25.1）、
`feature_list.json`、`claude-progress.md`

### Session 005 — Qwen3 数据层 + 预训练（feat-019 → feat-020）+ aux_loss 量纲纠错

- **日期**：2026-10-22
- **目标**：把 Qwen3 线从「组件层」推到「能真训练」：补齐语料/偏好数据层（feat-019），
  再写预训练脚本（feat-020），最后按三段式路线进入 LoRA / DPO。

**做了什么**

1. **feat-019 数据层**（commit `b60006c`）
   - `data/corpus.py` —— 合成语料的**唯一真源**（纯 Python，不依赖 torch）。
     BPE 训练语料与预训练语料从此同源，重建词表后 `vocab.json` / `merges.txt` **逐字节一致**。
   - `data/pretrain_data.py` —— `build_token_stream` / `build_blocks` / `PretrainDataset` / `make_pretrain_loader`；
     **文档分隔符用特殊 token id 0**，不用它的字面文本。
   - `data/preference_data.py` —— 5 组 prompt × 5 组 chosen × 4 种「变差」策略（wrong_entity / unrelated / too_short / noisy）。
   - `tests/test_data.py` —— 5 组断言（切块错位、语料可复现、分隔符、偏好数据不变量、DataLoader 形状与 train/val 划分）。
2. **feat-020 预训练**（本次）
   - `utils.py`（set_seed / get_device / save_checkpoint / load_checkpoint，独立重写不跨目录 import）
   - `train/train_pretrain.py` —— 唯一带 `--model dense|moe` 的脚本，**超参全部来自 config.py**，
     模型恒定返回 4 元组，MoE 的 aux_loss 从 `extras.get("aux_loss")` 取（训练脚本无 dense/MoE 分支）。
   - `tests/test_pretrain.py` —— 7 组断言（4 元组契约 / 初始 loss ≈ ln(V) / 梯度 / lr 调度端点与单调性 /
     build_model 报错 / MoE 总 loss 与 router 梯度 / dense 无 aux_loss）。
   - `init.sh` L4 新增第三条 smoke：「Qwen3 训练步 smoke」（forward → backward → step，断言 loss 有限 + 梯度非空 + lr 端点正确）。
   - `config.py` 只**新增**两个训练流程字段 `log_every` / `eval_max_batches`，**未改动任何既有超参数值** ⇒ 不需要同步改笔记数字。
3. **aux_loss 量纲纠错**（feat-021 的回查修正）
   - 原文档说「均匀时 aux_loss = 1/K（K=2 → 0.5）」，**与代码行为不符**：f_i 定义为「对 tokens×k 槽位求均值」时，
     均匀值是 1/E，aux = E·E·(1/E)(1/E) = **1**。
   - 查 transformers main 的 HF `load_balancing_loss_func`：HF 把槽位计数除以 N（Σf=K），均匀时 = **K**（=2），
     所以**本实现 = HF 的 1/K**，并非同量纲。
   - 修法：抽出模块级 `load_balancing_aux_loss(topk_indices, router_probs)`；代码 docstring / 注释 / 笔记 §19.1 / 开发计划 Task 10 全部改对；
     `tests/test_moe.py` 新增第 11 节「aux 量纲锚点」把数值钉死。

**三条可迁移的实测结论**

- **量纲类结论必须用代码实测锚定**：本次 aux_loss 的教训 —— 推导时的归一化假设（Σf=K）和代码里的归一化（Σf=1）不一致，
  公式心算出来的「0.5」和代码实际输出的「1.0」差一倍。写进文档前必须有一条能红的断言兜底。
- **Windows 下子进程输出编码**：`bash` 按 UTF-8 解码，Windows Python 默认 stdout 是 GBK → 中文乱码。
  证据命令一律加 `PYTHONIOENCODING=utf-8`（`tests/run_all.py` 里对子进程也这么做了）。
- **脚本放在子目录 = 必须补 sys.path**：GPT-2 线的脚本在目录顶层（`sys.path[0]` 天然是项目根），
  Qwen3 线按 `train/`、`tests/` 子目录组织，不显式补项目根就是 ModuleNotFoundError。

**验证**

```
$ python tests/test_pretrain.py
test_pretrain OK  loss=5.5184 params=919,040 lr[0]=6.00e-06 lr[end]=3.00e-05

$ python train/train_pretrain.py --model dense
[pretrain] params = 985,856
[pretrain] step    0/300 loss 6.7154 ... step 250/300 loss 1.4469 lr 5.58e-05
[pretrain] initial_loss=6.7154 final val_loss=1.3510 val_ppl=3.86 elapsed=46.8s
[pretrain] saved -> checkpoints\pretrain_dense.pt

$ python train/train_pretrain.py --model moe
[pretrain] params = 1,184,512
[pretrain]   aux_loss 1.0253 expert_load [1184.5, 745.0, 1298.0, 868.5]
[pretrain]   aux_loss 0.9982 expert_load [1042.5, 980.5, 1000.0, 1073.0]
[pretrain] initial_loss=6.6954 final val_loss=1.3607 val_ppl=3.90 elapsed=59.1s

$ python tests/test_moe.py
[aux 量纲] 均衡时 aux=1.0000（=1）；错配时 aux=1.6000；HF 同款公式 = 2.0000（= K×2）

$ python tests/run_all.py
[run_all] PASS=10  FAIL=0  TOTAL=10

$ bash init.sh
  [PASS] Qwen3 训练步 smoke（forward+backward+step，loss 有限）[train-step] loss 5.4788 grads 46 个 lr0 6.00e-06 lr_end 3.00e-05
  PASS=23  SKIP=0  FAIL=0
```

**未解决 / 风险**

- **MoE 在本仓库体现不出容量优势**：val_loss 1.3607 vs dense 1.3510，MoE 反而略差 ——
  语料太简单（合成句式）+ 专家更窄（128 vs 384，参数量只多 1.21×）+ 只训 300 步。
  结论要写进笔记，别让读者误以为「MoE 一定更好」。
- **专家负载是被 aux_loss 拉平的，但只在末段可见**：初期最大偏差 ±35%，250 步后 ±3.5%（总和恒为 B·T）。
- 笔记缺口仍在：`Qwen3_BPE_学习笔记.md` / `Qwen3_LoRA与DPO_学习笔记.md`（feat-008 / Task 19 补）。

**下一步**

1. `feat-023` 手写 LoRA（`model/common/lora.py` + `train/train_lora.py` + `tests/test_lora.py`）
2. `feat-024` 手写 DPO（π_θ = LoRA 开着 / π_ref = LoRA 关着，无 Reward Model 无 PPO）
3. `feat-025` 精简评估脚本 → `feat-018` KV Cache 推理与采样 → `feat-017` 与 GPT-2 的退化对照
4. `feat-008` 笔记↔源码交叉引用（含补 BPE / LoRA-DPO 两篇笔记）→ `feat-009` 数字全量核对 → 收班

**本次改动涉及的文件**

`qwen3-learning/qwen3-pytorch/`：`utils.py`、`train/train_pretrain.py`、`tests/test_pretrain.py`、
`config.py`（+log_every/eval_max_batches）、`model/qwen3_moe.py`（aux 抽函数 + docstring 纠错）、
`tests/test_moe.py`（+第 11 节量纲锚点）、`tests/run_all.py`（UTF-8 子进程解码）、
`qwen3-learning/qwen3-pytorch开发计划.md`（Task 10 纠错 + Task 13 就地换成真代码）、
`qwen3-learning/Qwen3_MoE_学习笔记.md`（§19.1 重写）、
仓库根 `init.sh`（L4 +训练步 smoke）、`feature_list.json`、`claude-progress.md`、`session-handoff.md`

### Session 006 — Qwen3 LoRA 微调（feat-023）

**本次做完的事**

1. 手写 LoRA（`model/common/lora.py`）：`LoRALinear` + `inject_lora` / `merge_lora` /
   `set_lora_enabled` / `lora_parameters` / `mark_only_lora_trainable`，**不引入 peft**。
2. 手写 LoRA 训练脚本（`train/train_lora.py`）：加载预训练权重 → 注入 → 冻结 base 只训 A/B →
   给定 prompt 补全 chosen → 评估偏好胜率 → 落盘 `checkpoints/lora.pt`。
3. 写 `tests/test_lora.py`（9 组断言），挂进 `init.sh` 的 L4 层。
4. 跑消融，据此把 `config.py` 的 `lora_lr` 3e-4 → 1e-3、`lora_max_iters` 300 → 1500。
5. 写 `qwen3-learning/Qwen3_LoRA与DPO_学习笔记.md` 的 LoRA 半部（§1–§9）。

**★ 三个结构性坑（都不报错，只静默给错结论）**

- **`merge_lora` 合并后分支还开着** → ΔW 被算两遍。合并的语义是「把旁路收编进主干」，
  所以合并完必须把 `scaling` 置 0。**初稿自带的第 6 条测试（merge 前后等价）本来就红。**
- **注入 LoRA 只冻结了被包装的那一个 Linear** → embedding / RMSNorm / MLP / lm_head 仍可训，
  「LoRA 微调」悄悄变成「大部分参数都在训」。补 `mark_only_lora_trainable`。
- **loss 把 prompt 也算进去了** → 与「给定 prompt 补全 chosen」的目标矛盾。
  改成按回复起始位置掩码（prompt 段与末位 -100）。

**★ 最重要的一条：偏好指标的长度偏置**

总 logprob 胜率系统性偏袒**短回复**（`too_short` 策略的 rejected 只有 6.3 token，
该策略胜率**恒为 0.000**），把整体胜率天花板压到 ~0.75；长度归一口径又反过来偏袒
「可预测」的文本。**只看整体胜率会误判「模型没进步」。**
这正是 DPO 减去参考模型 π_ref 的原因 —— π_ref 把「长度」「可预测性」这类静态偏置同时减掉。

**消融表**

| 配方 | loss | 总 logprob 胜率 | 长度归一胜率 | 可训练参数 |
|---|---|---|---|---|
| r=4 attn lr=0.001 steps=300 | 8.463 → 6.121 | 0.515 → 0.570 | 0.400 → <_io.TextIOWrapper name='C:/Users/ADMINI~1/AppData/Local/Temp/claude/E--ai-learning-05-llm-fundamentals-llm-learning/4091d65f-ec7e-4a9a-9393-f9eb3fa38550/tasks/bwfl72rnk.output' mode='r' encoding='utf-8'> | 8,704 |
| r=4 attn lr=0.0003 steps=1500 | 8.463 → 5.926 | 0.515 → 0.590 | 0.400 → <_io.TextIOWrapper name='C:/Users/ADMINI~1/AppData/Local/Temp/claude/E--ai-learning-05-llm-fundamentals-llm-learning/4091d65f-ec7e-4a9a-9393-f9eb3fa38550/tasks/bwfl72rnk.output' mode='r' encoding='utf-8'> | 8,704 |
| r=4 attn lr=0.001 steps=1500 | 8.463 → 5.701 | 0.515 → 0.595 | 0.400 → <_io.TextIOWrapper name='C:/Users/ADMINI~1/AppData/Local/Temp/claude/E--ai-learning-05-llm-fundamentals-llm-learning/4091d65f-ec7e-4a9a-9393-f9eb3fa38550/tasks/bwfl72rnk.output' mode='r' encoding='utf-8'> | 8,704 |
| r=16 attn+mlp lr=0.001 steps=300 | 8.415 → 5.319 | 0.515 → 0.605 | 0.400 → <_io.TextIOWrapper name='C:/Users/ADMINI~1/AppData/Local/Temp/claude/E--ai-learning-05-llm-fundamentals-llm-learning/4091d65f-ec7e-4a9a-9393-f9eb3fa38550/tasks/bwfl72rnk.output' mode='r' encoding='utf-8'> | 133,120 |
| r=16 attn+mlp lr=0.001 steps=1500 | 8.415 → 4.153 | 0.515 → 0.640 | 0.400 → <_io.TextIOWrapper name='C:/Users/ADMINI~1/AppData/Local/Temp/claude/E--ai-learning-05-llm-fundamentals-llm-learning/4091d65f-ec7e-4a9a-9393-f9eb3fa38550/tasks/bwfl72rnk.output' mode='r' encoding='utf-8'> | 133,120 |
| r=64 attn+mlp lr=0.001 steps=300 | 8.348 → 4.460 | 0.515 → 0.575 | 0.400 → <_io.TextIOWrapper name='C:/Users/ADMINI~1/AppData/Local/Temp/claude/E--ai-learning-05-llm-fundamentals-llm-learning/4091d65f-ec7e-4a9a-9393-f9eb3fa38550/tasks/bwfl72rnk.output' mode='r' encoding='utf-8'> | 532,480 |
| r=64 attn+mlp lr=0.003 steps=300 | 8.348 → 6.258 | 0.515 → 0.520 | 0.400 → <_io.TextIOWrapper name='C:/Users/ADMINI~1/AppData/Local/Temp/claude/E--ai-learning-05-llm-fundamentals-llm-learning/4091d65f-ec7e-4a9a-9393-f9eb3fa38550/tasks/bwfl72rnk.output' mode='r' encoding='utf-8'> | 532,480 |
| 全参 lr=0.001 steps=300 | 8.450 → 1.217 | 0.515 → 0.715 | 0.400 → <_io.TextIOWrapper name='C:/Users/ADMINI~1/AppData/Local/Temp/claude/E--ai-learning-05-llm-fundamentals-llm-learning/4091d65f-ec7e-4a9a-9393-f9eb3fa38550/tasks/bwfl72rnk.output' mode='r' encoding='utf-8'> | 985,856 |

**验证**

```
$ python tests/test_lora.py
[可训练参数] 8,704 / 927,744 (0.94%)
test_lora OK  replaced=8 r=4 alpha=8 scaling=2.0

$ python train/train_lora.py
[lora] injected 8 modules: ['blocks.0.attn.q_proj', ..., 'blocks.3.attn.v_proj']
[lora] trainable params = 8,704 / 994,560 (0.88%)
[lora] BEFORE  win_rate=0.515  win_rate_per_token=0.400  mean_logp(chosen)=-603.26
[lora]   too_short      n=47   总logprob胜率=0.000  长度归一胜率=0.404  len(chosen)=72.0 len(rejected)=6.3
[lora] step  850/1500 loss 5.7365 lr 1.00e-03
[lora] AFTER   win_rate=0.595  win_rate_per_token=0.745  mean_logp(chosen)=-389.48
[lora] 胜率 0.515 -> 0.595 (随机基线 0.5)，长度归一口径 0.400 -> 0.745
[lora] saved -> checkpoints\lora.pt

$ python tests/run_all.py
[run_all] PASS=11  FAIL=0  TOTAL=11

$ bash init.sh
  [PASS] Qwen3 LoRA smoke（注入无损 + 只训 A/B）[lora] 注入 8 个模块，输出无损；可训练 8,704/927,744 (0.94%)
  PASS=24  SKIP=0  FAIL=0
```

**未解决 / 风险**

- **LoRA 在本任务上打不过全参微调**：这是内容型任务（要模型说出特定文本），
  全参能压到 loss ~1.06，r=4 的 LoRA 只能到 ~5.7。**结论要写进笔记**：
  LoRA 是风格/格式型任务的主场，内容型任务上明显吃力。
- **胜率是训练内指标**：合成偏好数据只有 5 个 prompt × 5 个 chosen，训练集与评估集同分布，
  证明的是「LoRA 确实让模型更偏向 chosen」，**不能当泛化能力**。
- 笔记缺口：`Qwen3_LoRA与DPO_学习笔记.md` 的 **DPO 半部（§8）** 待 feat-024 补；
  `Qwen3_BPE_学习笔记.md` 待 feat-008 补。

**下一步**

1. `feat-024` 手写 DPO（π_θ = LoRA 开着 / π_ref = LoRA 关着，无 Reward Model 无 PPO）
2. `feat-025` 精简评估脚本 → `feat-018` KV Cache 推理与采样 → `feat-017` 与 GPT-2 的退化对照
3. `feat-008` 笔记↔源码交叉引用（含补 BPE 笔记 + LoRA-DPO 笔记的 DPO 半部）→ `feat-009` 数字全量核对 → 收班

**本次改动涉及的文件**

`qwen3-learning/qwen3-pytorch/`：`model/common/lora.py`、`train/train_lora.py`、`tests/test_lora.py`、
`config.py`（lora_lr / lora_max_iters 两个值）、
`qwen3-learning/Qwen3_LoRA与DPO_学习笔记.md`（新建，LoRA 半部）、
`qwen3-learning/qwen3-pytorch开发计划.md`（Task 14 就地换成真代码 + 实测修正记录）、
仓库根 `init.sh`（L4 +LoRA smoke）、`feature_list.json`、`claude-progress.md`、`session-handoff.md`

### Session 007 — Qwen3 DPO（feat-024）+ 一个模型层的静默 bug

**本次做完的事**

1. 手写 DPO（`train/train_dpo.py`）：目标函数、隐式奖励、reward margin、
   `evaluate_dpo`（margin 均值/标准差 + dpo_acc + 绝对胜率 + 按策略分组）。
2. 给 `model/common/lora.py` 加 **`strip_lora`**（把 LoRALinear 换回普通 nn.Linear）。
3. 写 `tests/test_dpo.py`（9 组断言），挂进 `init.sh` 的 L4 层。
4. 补 `Qwen3_LoRA与DPO_学习笔记.md` 的 §8（DPO 全节）+ §6.4 + 把 §9 教训扩到 7 条。
5. **★ 顺带修掉一个模型层的静默 bug：`dropout` 覆盖没传到 Block**（见下）。

**★ 最隐蔽的一个 bug：π_ref 锚错了**

标准 DPO 的 KL 锚点是 **SFT 模型**。初稿加载完 `lora.pt` 后**没有 merge**，
直接 inject 了一个新 LoRA —— 于是「关掉 LoRA 分支」得到的是**预训练模型**，
不是 SFT 模型。后果：DPO 的 KL 项会跟刚做完的 LoRA SFT 互相拉扯。

**它不会让代码崩、不会让 loss 变 NaN、指标还会照涨 —— 只是整个实验锚在错的基线上。**

修法是三步：

```text
inject → load lora.pt  →  merge_lora（ΔW 折进 base ⇒ 模型 == SFT 模型）
                        →  strip_lora（换回普通 Linear）
                        →  inject_lora（新的，B 零初始化）
```

之后「关掉分支」精确等于 SFT 模型，**同一个模型对象**搞定两套策略，不加载第二份权重。

**★ 由此得到一条可运行的不变量**（被守了三次：训练脚本 assert / 单测第 3 组 / init.sh smoke）：

> **初始 margin 恒为 0、loss 恒为 log 2 = 0.6931**（因为 B 零初始化 ⇒ π_θ ≡ π_ref）。

这是「π_ref 锚对了没有」的唯一可运行证据。

**★ 第二个静默 bug：`dropout` 覆盖没传到 Block（模型层）**

做消融时发现「同一份 DPO 配方跑两遍，margin 差 0.028（+0.1859 vs +0.1582）」，
而 `train/train_dpo.py` 自己重跑两遍完全一致 ⇒ 差异来自消融脚本「训练前多做了一次 win_flags」。

追下去发现根因在模型层：

- `Qwen3Dense.__init__` / `Qwen3MoE.__init__` 算了覆盖值 `d`，
  却只把它给了 `self.drop`（embedding 的 Dropout），
  **Block 里写的是 `dropout=cfg.dropout`** ⇒ `dropout=0.0` 只关掉了一半；
- 注意力/MLP 的 Dropout 一直是 `p=0.1`，而 **p>0 的 Dropout 在 train 模式下会推进全局 torch RNG**
  （实测：p=0.0 不推进，p=0.1 推进）⇒ 「训练前多做 200 次前向」就会改变训练轨迹。

修法：`Block` / `MoEBlock` 接受并真正传递 `dropout` 覆盖值。
回归断言写进 `tests/test_dense.py` 第 13 条与 `tests/test_moe.py` 末尾
（**结构断言 + 行为断言各一条**；已实测退回 bug 版本时断言立刻报错）。

**影响面**：SFT/DPO 阶段真的没有 dropout ⇒ **LoRA 与 DPO 的全部数字都变了**
⇒ 重跑了 LoRA 主训练 + LoRA 8 行消融 + DPO 主训练 + DPO 8 行消融，并更新了笔记。
**预训练不受影响**（显式传 `cfg.dropout=0.1`），重跑核对
`initial_loss=6.7154 → val_loss=1.3510 val_ppl=3.86`，与修复前**逐位一致**。

**★ 最重要的实测发现 ——「margin 和绝对胜率是两回事」**

| 配方 | reward margin | dpo_acc | 绝对 win_rate | 长度归一 | 翻转数 |
|---|---|---|---|---|---|
| 300 步 / lr 1e-4 | +0.2831 | **0.945** | 0.605 → 0.605 | 0.745 → 0.780 | **0**/200 |
| 1500 步 / lr 1e-3 | **+4.6512** | 0.780 | 0.605 → **0.665** | 0.745 → **0.900** | 12/200 |

**弱配方下 dpo_acc 冲到 0.945（看着很美），绝对胜率却 0/200 翻转 —— 一个样本都没动。**
**DPO 优化的是「相对 π_ref 的间隔」，不是「绝对排序」。**

⇒ **报告 DPO 结果必须同时给 reward margin（目标口径）与绝对 win_rate（业务口径）**，
只给前者是自我欺骗，只给后者会误判「没训好」。

**★ 顺带实证了 LoRA 笔记 §5.4 的论断，还多了一层**

`too_short` 策略（rejected 只有 6.3 token）的**绝对 win_rate 恒为 0.000**。
弱配方下它的 dpo_acc 是 **1.000**（DPO 判得很对，静态长度偏置在相减时被减掉了）；
强配方下掉到 **0.277**、margin 变负 ——
**DPO 减掉 π_ref 的偏置之后，会按新目标重新造一个偏置。**

**★ π_ref 取法的结论**

标准做法（锚 SFT 模型）与消融（锚预训练模型）**都能把业务指标推上去，终点几乎一样**
（0.665 vs 0.650）。但两行的 margin 参照系不同、起点基线不同，不可直接比大小；
**默认仍取 `--ref-mode sft`**。

**消融表**

| π_ref | 步数 | lr | β | reward margin | dpo_acc | 绝对 win_rate | 翻转数 |
|---|---|---|---|---|---|---|---|
| sft | 300 | 0.0001 | 0.1 | +0.0000 → +0.2831 | 0.000 → 0.945 | 0.605 → 0.605 | 0 |
| sft | 1500 | 0.0001 | 0.1 | +0.0000 → +1.6338 | 0.000 → 0.710 | 0.605 → 0.630 | 5 |
| sft | 1500 | 0.001 | 0.1 | +0.0000 → +4.6512 | 0.000 → 0.780 | 0.605 → 0.665 | 12 |
| sft | 1500 | 0.001 | 1.0 | +0.0000 → +8.4384 | 0.000 → 0.950 | 0.605 → 0.625 | 4 |
| sft | 1500 | 0.001 | 0.01 | +0.0000 → +1.0109 | 0.000 → 0.710 | 0.605 → 0.715 | 22 |
| pretrained | 1500 | 0.001 | 0.1 | +0.0000 → +8.4640 | 0.000 → 0.930 | 0.515 → 0.650 | 27 |
| sft | 1500 | 0.0003 | 0.1 | +0.0000 → +4.1526 | 0.000 → 0.715 | 0.605 → 0.665 | 12 |
| sft | 3000 | 0.001 | 0.1 | +0.0000 → +5.0101 | 0.000 → 0.920 | 0.605 → 0.665 | 12 |

**验证**

```
$ python tests/test_dpo.py
[strip] inject → strip 往返无损
[strip] merge → strip 保住 SFT 的 ΔW
[dpo] 初始 margin=+0.000000 loss=0.6931（= log 2）
[dpo] 闭式校验通过：margin=-0.0124 loss=0.6994
[dpo] 梯度只到 16 个 LoRA 张量（16 个有梯度）
[dpo] 一步优化 margin +0.0158 → +0.2230
[dpo] evaluate_dpo 自洽：dpo_acc=0.750 margin=+0.0798
[dpo] β=0 边界通过
test_dpo OK  beta=0.1 loss=0.6994 margin=-0.0124 dpo_acc=0.750

$ python train/train_dpo.py
[dpo] π_ref = SFT 模型：merge 8 个模块 / strip 8 个模块，base 已含 SFT 的 ΔW
[dpo] trainable params = 8,704 / 994,560 (0.88%)
[dpo] BEFORE  margin=+0.0000 ± 0.0000  dpo_acc=0.000  win_rate=0.605  win_rate_per_token=0.745
[dpo] step    0/1500 loss 0.6931 margin +0.0000 (avg50 +0.0000)
[dpo] AFTER   margin=+4.6512 ± 4.3952  dpo_acc=0.780  win_rate=0.665  win_rate_per_token=0.900
[dpo] margin 首50均 +0.1230 → 末50均 +3.2850
[dpo] saved -> checkpoints\dpo.pt

$ python tests/run_all.py
[run_all] PASS=12  FAIL=0  TOTAL=12

$ bash init.sh
  [PASS] Qwen3 DPO smoke（初始 margin=0 + loss=log2）[dpo] 初始 margin=+0.000000 loss=0.6931(=log2)；可训练 8,704/994,560 (0.88%)
  PASS=25  SKIP=0  FAIL=0
```

**未解决 / 风险**

- **dpo_acc 与绝对胜率会背离**：强配方 dpo_acc 只有 0.780（弱配方 0.945），
  但业务指标涨得更多。**dpo_acc 是训练内指标**，评估集 = 训练集的前 200 对。
- **单对 margin 极端化**：`margin = +4.6512 ± 4.3952`（标准差 > 均值），
  lr=1e-3 下 DPO 在逐对硬拟合。
- **笔记缺口**：`Qwen3_BPE_学习笔记.md`（feat-008 补）。

**下一步**

1. `feat-025` 精简评估脚本 → `feat-018` KV Cache 推理与采样 → `feat-017` 与 GPT-2 的退化对照
2. `feat-008` 笔记↔源码交叉引用（补 BPE 笔记）→ `feat-009` 数字全量核对 → 收班

**本次改动涉及的文件**

`qwen3-learning/qwen3-pytorch/`：`train/train_dpo.py`、`tests/test_dpo.py`、
`model/common/lora.py`（+strip_lora）、`model/qwen3_dense.py`（**dropout 覆盖修复**）、
`model/qwen3_moe.py`（**dropout 覆盖修复**）、`tests/test_dense.py`（+回归断言）、
`tests/test_moe.py`（+回归断言）、`config.py`（dpo_lr / dpo_max_iters / 注释）、
`qwen3-learning/Qwen3_LoRA与DPO_学习笔记.md`（§6.2 / §6.4 / §7 / §8 / §9）、
`qwen3-learning/qwen3-pytorch开发计划.md`（Task 15 就地换成真代码 + 实测修正记录）、
仓库根 `init.sh`（L4 +DPO smoke）、`feature_list.json`、`claude-progress.md`、`session-handoff.md`

### Session 008 — Qwen3 KV Cache 推理与采样（feat-018）

**本次做完的事**

1. 新增 `qwen3-learning/qwen3-pytorch/inference.py`：
   `sample_from_logits` / `make_kv_cache` / `generate_naive` / `generate_with_cache` /
   `benchmark_speed` / `kv_cache_bytes` / `kv_cache_report`。
2. 新增 `tests/test_inference.py`（12 组断言），自动挂进 `tests/run_all.py`。
3. `init.sh` L4 加一条推理 smoke（greedy 逐位一致 + GQA/MHA 占用比）。
4. 补 `Qwen3_训练推理完整学习笔记.md` §10：实现与采样差异、实测、自测题。
5. **修正计划里的一个顺序 bug**：feat-025 的 `speedup` 段依赖 `inference.py`，
   而原计划把 feat-025 排在 feat-018 前面 ⇒ 实际执行顺序改为 feat-018 → feat-025，
   开发计划 Task 16 已就地标注。

**★ 最硬的一条结论：有/无 KV Cache 的 greedy 输出逐位相同**

不只是「差不多」—— 两条断言一起守：

- token 序列：`torch.equal(seq_naive, seq_kv) == True`（greedy，12 步）
- 逐步 logits：12 步的 `max|diff| = 5.96e-07`

> logits 不是逐位相同，差在 6e-7：**naive 每次把整条序列（T_q = P+s）一起前向，
> cache 路径每步只前向 1 个 token（T_q = 1），matmul 的规约顺序不同 ⇒ fp32 末位差异。**
> 差在 1e-7 量级，argmax 完全不受影响。

**★ 「decode 不施加因果 mask」的反向实验**

做法：prefill 之后把 cache 里**除最后一个之外**的历史 K/V 全部清零，再走一步 decode。

- 若 decode 真读了历史 → 输出**必须改变**
- 若 decode 错误地把历史屏蔽了 → 输出**不会变**

实测输出**变了** ⇒ decode 确实读了历史。
（结构断言也守着：prefill 后每层缓存长度 = prompt 长度，decode 一步后 +1。）

**★ 与 GPT-2 线的 speedup 横向对比**（同 prompt 长度 64、同生成 64、同一台 CPU、greedy、3 次取平均）：

| | naive_avg_ms | kv_avg_ms | speedup |
|---|---|---|---|
| **GPT-2**（MHA） | 207.92 | 118.94 | **1.748** |
| **Qwen3**（GQA） | 327.31 | 163.74 | **1.999** |

- **Qwen3 的绝对速度更慢**（naive 慢 ~57%）：三个独立投影 + QK-Norm 两趟 RMSNorm +
  RoPE 旋转 + SwiGLU 三矩阵，单层做的事比 GPT-2 多。
- **但加速比反而更高**（2.00x vs 1.75x）：GQA 让 cache 的**读取量减半**，cache 路径省得更多。

> **「模型更复杂」与「推理更快」是两回事。**
> Qwen3 用结构复杂度换来了 GQA 的内存/带宽优势。

**KV Cache 占用**（T = 128，fp32）：GQA **393,216 B** vs 同尺寸 MHA **786,432 B** = **1/2**。

**★ 逐条堵上 GPT-2 笔记（`gpt-learning/12.采样方式.md` §15）指出的 4 个坑**

| # | GPT-2 线的问题 | Qwen3 线的处理 |
|---|---|---|
| ① | 左 padding 没有 attention mask | 本线只做 batch=1 单条生成，不引入 padding |
| ② | finished 样本没冻结输出 | eos 直接 break，**并补上 prefill 后第一个 token 的 eos 检查**（GPT-2 线循环从第 2 个 token 才开始查，第一个 token 若是 eos 不会被拦下） |
| ③ | KVCache 的 device/dtype 可能不匹配 | 显式传模型参数的 dtype |
| ④ | `cache.get()` 每次 clone 整段缓存 | 直接返回切片 view |

采样链的执行顺序与 top-p 的 mask 写法与 GPT-2 线**逐字一致**
（`mask = (cumsum - sorted_probs) > top_p`，保证第一次越过 top_p 的 token 仍被保留）。

**★ 修正计划初稿的 4 个真 bug**

① **top-p 的 scatter 写错了** —— 初稿 `logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)`
把 `sorted_logits` 散射回**自己**，得到的不是「按原位置还原」；
正确写法是散射进一个空张量 `torch.empty_like(logits).scatter(-1, sorted_idx, sorted_logits)`。
② **KVCache 构造参数名对不上**（初稿 `n_head=`/`max_seq_len=`/`device=`，
实际签名是 `n_layer/batch_size/n_head_kv/head_dim/max_len/dtype`）。
③ **模型属性取不到**（初稿 `model.n_layer`，实际在 `model.cfg` 上）。
④ **★ benchmark_speed 的列名不满足 feat-025 的验证项** —— 初稿返回
`naive_sec/cached_sec/speedup`，而要求是「与 GPT-2 线同名的列」
⇒ 实现改成 `naive_avg_ms / kv_avg_ms / speedup / naive_tokens / kv_tokens`。

**验证**

```
$ python tests/test_inference.py
test_inference OK  greedy一致=逐位相同 logits_max|diff|=5.96e-07 speedup=1.33x KV占用(GQA/MHA)=196608/393216B 比值=1/2

$ python tests/run_all.py
[run_all] PASS=13  FAIL=0  TOTAL=13

$ bash init.sh
  [PASS] Qwen3 推理 smoke（有/无 KV Cache 输出一致）[infer] greedy 逐位一致（8 token）；KV 占用 GQA=196,608B MHA=393,216B（1/2）
  PASS=26  SKIP=0  FAIL=0
```

**未解决 / 风险**

- **speedup 只有 ~2x**：CPU 上模型太小，prefill 的固定开销占比高。
  真实部署里长 prompt + 长生成才会看到 KV Cache 的收益（本仓库 block_size 只有 128）。
- **只做了 batch=1**：GPT-2 线笔记指出的「左 padding 无 attention mask」问题在本线**被绕开**
  而不是被解决。要做批量推理必须先补 attention mask。

**下一步**

1. `feat-025` 精简评估脚本（pretrain / lora / dpo / speedup 四段 + MoE 段）
2. `feat-017` 与 GPT-2 的退化对照实验 → `feat-008` 笔记↔源码交叉引用 → `feat-009` 数字全量核对 → 收班

**本次改动涉及的文件**

`qwen3-learning/qwen3-pytorch/`：`inference.py`（新）、`tests/test_inference.py`（新）、
`qwen3-learning/Qwen3_训练推理完整学习笔记.md`（+§10）、
`qwen3-learning/qwen3-pytorch开发计划.md`（Task 17 就地换成真代码 + Task 16 标注顺序纠正）、
仓库根 `init.sh`（L4 +推理 smoke）、`feature_list.json`

### Session 009 — Qwen3 精简评估脚本（feat-025）

**本次做完的事**

1. 新增 `qwen3-learning/qwen3-pytorch/evaluate.py`（5 段：pretrain / moe / lora / dpo / speedup + `--stage all`）。
2. 新增 `tests/test_evaluate.py`（7 组断言），自动挂进 `tests/run_all.py`。
3. `init.sh` L4 加一条评估 smoke（5 段结构 + GPT-2 同名列 + 一致性）。
4. `qwen3-pytorch/README.md` 新增「与 GPT-2 线 9 项评估的差异」一节（feat-025 的验收项）。
5. `Qwen3_训练推理完整学习笔记.md` 补 §11（评估）。
6. **修正计划初稿的 9 个真 bug**（开发计划 Task 16 已就地换成真代码）。

**★ 评估脚本的数字与训练日志完全对得上**

| 指标 | 训练日志 | 评估脚本 |
|---|---|---|
| 预训练 val_loss / val_ppl | 1.3510 / 3.86 | 1.3510（batch 均值口径）/ 3.861 |
| LoRA 胜率（前 → 后） | 0.515 → 0.605 | 0.5150 → 0.6050 |
| DPO margin / dpo_acc / 绝对胜率 | +4.6512 / 0.780 / 0.665 | +4.6512 / 0.7800 / 0.6650 |

> **做法**：各段的评估函数**直接复用训练脚本里的实现**
> （`train.train_pretrain.estimate_loss` / `train.train_lora.evaluate_pairs` / `train.train_dpo.evaluate_dpo`），
> **不重写**。重写就会出现「两边口径不同、数字对不上」的老问题。

**★ pretrain 的两种 loss 口径（都打印，是为了能跟训练日志对上）**

- `loss` / `perplexity`：**token 加权**口径 —— 与 GPT-2 线 `eval_perplexity` 同口径（可直接横向比）
- `loss_batch_mean`：**batch 均值**口径 —— 与 `train/train_pretrain.py` 打印的 `val_loss` 同口径

实测 1.3485（token 加权）vs 1.3510（batch 均值），差 0.19%（差异只来自最后一个不满的 batch）。

> **「同名」还不够，「同口径」才够**：同一个叫 `perplexity` 的字段，
> 一个 token 加权、一个 batch 均值，数字放在一起比就是在比两件不同的事。

**★ MoE 段的两个新指标**

- **`aux_loss = 0.9978`** —— 本实现除以 `N·K`，**完全均衡时 = 1**（HF Mixtral 是 K，Switch top-1 是 1）
  ⇒ 0.9978 说明 router 没塌。
- **专家负载 = [0.2439, 0.2395, 0.2437, 0.2729]，max/min = 1.140** ——
  判读负载分布要看 **max/min 比值**而不是单个值：单个专家占 27% 看着不多，
  但若另一个占 0% 则 max/min = ∞，那才是塌缩。

**★ 与 GPT-2 线 9 项评估的差异（已写进 README）**

| GPT-2 线 | Qwen3 线 | 说明 |
|---|---|---|
| `pretrain` | ✅ `pretrain` | 同名列 + 同口径 |
| `speedup` | ✅ `speedup` | 同名列 |
| `sft` | 🔁 `lora` | 对应物：LoRA 前后胜率对比 |
| `reward`（RM pairwise acc） | 🔁 `dpo` | 本线没有 RM；对应物是 margin 判对率 |
| `ppo`（reward / KL） | 🔁 `dpo` | 本线没有 PPO；对应物是隐式奖励 margin |
| `compare`（SFT vs PPO） | 🔁 `lora` | 本线没有 PPO；对应物是 LoRA 前→后对比 |
| `benchmark` | ❌ 无 | 语料是合成的，没有 QA benchmark |
| `diversity` / `response_quality` | ⏳ 未实现 | 生成质量类，feat-025 范围外 |
| — | ➕ `moe` | **Qwen3 独有**：专家负载分布 + aux_loss |

**★ 「不删 / 不改弱门槛」必须有可运行证据**

门槛分两层：

| 层 | 写在哪儿 | 特点 |
|---|---|---|
| ① **自洽性门槛** | 各 `eval_*()` 里 | 跟模型好坏无关，随机初始化模型也该过 ⇒ 测试与 smoke 靠它 |
| ② **方向性门槛** | `check_directional()`，由 `main()` 施加 | 只有训好的 checkpoint 才该过 ⇒ 达不到就报错退出 |

**两层都实测过会触发**：
第 5 组把 `inference.generate_naive` 换成返回错误结果 ⇒ `eval_speedup` 必须抛 AssertionError；
第 5b 组分别喂「胜率 0.4 / margin −0.1 / speedup 0.9」⇒ `check_directional` 必须抛 AssertionError。

checkpoint 缺失时**跳过该段并打印原因**（那是「还没跑到那一步」，不是「指标不合格」）。

**★ 修正计划初稿的 9 个真 bug**（其中 3 个会「静默给出错误数字」，最危险）

① **`strict=False` 会静默加载不上权重** —— `lora.pt`/`dpo.pt` 的 state_dict 是**注入 LoRA 之后**的键名
（`q_proj.base.weight` / `.lora_A` / `.lora_B`），与未注入的模型对不上，
`strict=False` 只警告不报错 ⇒ **权重全部没加载，评估跑的是随机模型**。
改成：**先 `inject_lora` 再 `load_state_dict(strict=True)`**。

② **DPO 的 margin 算错了** —— 初稿 `margins.append(lp_w.item() - lp_l.item())` 是**绝对 logprob 差**，
既没减 π_ref 也没乘 β ⇒ **看着像 margin，其实完全不是一回事**。
DPO 的 margin 必须是 `β·[(Δw_θ − Δw_ref) − (Δl_θ − Δl_ref)]`。

③ **`encode_pair` 的返回值被解包错了** —— 实际返回 `((lp_w, n_w), (lp_l, n_l))`，
初稿按两个值解包 ⇒ `lp_w` 变成 tuple，后面 `.item()` 直接 AttributeError。

④⑤ **`benchmark_speed` / `generate_naive` 的参数名与签名不符**，且 prompt 必须是 `(B,P)` tensor 不是 list。

⑥ **`a == b` 返回的是 tensor 不是 bool**。

⑦ **缺 MoE 段** —— 设计文档 §8 要求 MoE 阶段额外输出专家负载分布与 aux_loss。

⑧ **返回列名不满足验收项** —— 初稿 pretrain 返回 `{val_loss, val_ppl}`、speedup 返回 `{consistent, speedup}`，
而验收项要求「与 GPT-2 线同名列」。

⑨ **方向性门槛缺失 + baseline 是写死的 0.5** —— 那个 0.5 不是**测出来的**基线。
改成：关掉 LoRA 分支（模型 == 预训练模型）**实测**基线（0.515），并新增 `check_directional()`。

**验证**

```
$ python evaluate.py --stage all
[eval] stage=pretrain ppl=3.852  loss=1.3485  n_tokens=6400
[eval]   （batch 均值口径 loss=1.3510 ppl=3.861 —— 与训练日志的 val_loss 同口径）
[eval] stage=moe     ppl=3.889  loss=1.3582  aux_loss=0.9978（coef=0.01）
[eval]   专家负载（4 个）= [0.2439, 0.2395, 0.2437, 0.2729]
[eval]   负载 max/min = 1.140
[eval] stage=lora    win_rate=0.6050  （基线 0.5150，Δ=+0.0900）
[eval] stage=dpo     margin=+4.6512 ± 4.3952  accuracy=0.7800  win_rate=0.6650  (beta=0.1, ref=sft)
[eval] stage=speedup speedup=1.770x  naive=284.57ms  kv=160.82ms  consistent=True
[eval]   KV Cache 占用 GQA=304,128B  MHA=608,256B  比值=0.500
[eval] 跑了 5/5 段：['pretrain', 'moe', 'lora', 'dpo', 'speedup']

$ python tests/test_evaluate.py
test_evaluate OK  5 段结构 + GPT-2 同名列 + 跳过路径 + 一致性门槛

$ python tests/run_all.py
[run_all] PASS=14  FAIL=0  TOTAL=14

$ bash init.sh
  [PASS] Qwen3 评估 smoke（5 段 + GPT-2 同名列）[eval] 5 段就绪；pretrain/speedup 的 GPT-2 同名列齐全；ppl=765.62 speedup=1.73x consistent=True
  PASS=27  SKIP=0  FAIL=0
```

**未解决 / 风险**

- **diversity / response_quality / benchmark 三项没做**：属「生成质量」类，
  本线语料是合成的，做出来的数字意义有限。已在 README 明确标注「⏳ 未实现 / ❌ 无」。
- **`too_short` 那一组的绝对胜率恒为 0**：不是 bug，是「总 logprob 偏袒长回复」的必然结果；
  长度归一口径（0.9000）才是能用的那个。
- **speedup 只有 ~1.7x**：CPU + block_size=128 + 极小模型，prefill 固定开销占比高。

**下一步**

1. `feat-017` 与 GPT-2 的**退化对照实验**（本线最有价值的产出）
2. `feat-008` 笔记↔源码交叉引用（补 BPE 笔记）→ `feat-009` 数字全量核对 → 收班

**本次改动涉及的文件**

`qwen3-learning/qwen3-pytorch/`：`evaluate.py`（新）、`tests/test_evaluate.py`（新）、`README.md`（+评估差异一节）、
`qwen3-learning/Qwen3_训练推理完整学习笔记.md`（+§11）、
`qwen3-learning/qwen3-pytorch开发计划.md`（Task 16 就地换成真代码 + 9 个 bug 记录）、
仓库根 `init.sh`（L4 +评估 smoke）、`feature_list.json`

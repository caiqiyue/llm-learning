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

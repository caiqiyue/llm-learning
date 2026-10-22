# llm-learning

> **从零手写 tiny GPT-2，跑通「预训练 → SFT → Reward Model → PPO」全流程**；副线**从零手写 tiny Qwen3**
> （RoPE / RMSNorm / QK-Norm / SwiGLU / GQA / MoE + byte-level BPE，走「预训练 → LoRA → DPO」）。
> 配套 26 篇中文学习笔记（GPT-2 17 篇 + Qwen3 9 篇）。
> 笔记和代码在同一仓库，**每个结论都能落到具体代码行上**。

模型只有约 0.8M 参数，CPU 上 10 分钟能跑完整个 RLHF 链路 —— 规模小到可以逐行 debugger，规模完整到能看清每个环节在干什么。

---

## 一、这个仓库解决什么问题

读 LLM 论文和看教程，最容易卡在两件事上：

1. **只讲原理，看不到代码** —— 知道了 PPO 的公式，但不知道 `rollout` 到底怎么组织、KL 项加在哪一行
2. **只贴代码，不讲为什么** —— 一份 200 行的 `train_ppo.py`，没人告诉你哪几行是关键

这个仓库的做法是：**每个知识点一篇笔记，笔记对应一段可运行的代码**。GPT-2 主线用最朴素的方式写（不追求性能，只追求能看懂），Qwen3 笔记再讲现代架构怎么在同样的位置做改进。

---

## 二、目录结构

```
llm-learning/
├── README.md                    ← 你在这里
├── AGENTS.md                    ← AI agent 操作手册（硬约束 / 验证路径 / DoD）
├── CLAUDE.md                    ← Claude Code 的入口版规则
├── harness.md                   ← harness 通用方法论手册
├── init.sh                      ← 一键轻量验证（< 1 分钟）
├── feature_list.json            ← 任务状态机：当前阶段 + 现在该做什么
├── claude-progress.md           ← session 进度日志
├── session-handoff.md           ← 跨 session 交接卡
├── clean-state-checklist.md     ← 收班清单
│
├── gpt-learning/                        ← 【P1 线】GPT-2 + RLHF 全流程
│   ├── 0~14 章系列笔记 + 数学公式详解 + 代码阅读顺序
│   ├── requirements.txt
│   ├── checkpoints/                     ← 训练产物（不入库）
│   ├── docs/                            ← 本地资料（不入库）
│   └── gpt-learning-pytorch/            ← 实现（PyTorch）
│       ├── config.py                    ← 所有超参集中在这里
│       ├── data/                        ← 合成数据构造（纯 Python，不依赖 torch）
│       ├── model/gpt2.py                ← GPT-2 架构 + KV Cache
│       ├── train_pretrain.py            ← ① 预训练
│       ├── train_sft.py                 ← ② 监督微调
│       ├── train_reward.py              ← ③ 奖励模型
│       ├── train_ppo.py                 ← ④ PPO 强化学习
│       ├── evaluate.py                  ← 9 项评估
│       ├── generate.py / inference.py   ← 采样 / KV Cache 推理
│       └── utils.py
│
└── qwen3-learning/                      ← 【P2 线】Qwen3 现代架构
    ├── Qwen3_*.md × 9                   ← 笔记（每篇开头带 GPT-2 / Qwen3 源码交叉引用）
    └── qwen3-pytorch/                   ← 实现（从零手写，预训练 → LoRA → DPO 全链路）
```

> `checkpoints/` 和 `docs/` 被 `.gitignore` 排除：权重文件和本地草稿留在本机，不进仓库。

---

## 三、项目阶段

| 阶段 | 内容 | 状态 |
|------|------|------|
| **P1** | GPT-2 + RLHF 全流程：17 篇笔记 + 可运行代码 | **全链路已复验通过**（torch 环境已修好，训练 + 9 项评估全绿） |
| **P2（当前）** | Qwen3 从零实现：RoPE / RMSNorm / QK-Norm / SwiGLU / GQA / MoE + 预训练 → LoRA → DPO | **9 篇笔记 + 全部实现就位**，收尾中（交叉引用 + 数字全量核对） |
| **P3** | Qwen3 更大规模、更多任务 | 规划中 |

当前阶段以 [`feature_list.json`](feature_list.json) 的 `phase` 字段为准。

### P2 的 Qwen3 代码结构（已建成）

```
qwen3-learning/qwen3-pytorch/
├── config.py          # 独立超参：n_head_kv / head_dim / rope_theta / rms_eps …
├── model/
│   ├── common/        # rmsnorm / rope / attention(QK-Norm+GQA) / mlp(SwiGLU) / kv_cache / lora / norm / pos_emb
│   ├── qwen3_dense.py # 完整前向（接口对齐 GPT2.forward）
│   └── qwen3_moe.py   # Router + 专家 + 负载均衡辅助损失
├── data/              # byte-level BPE tokenizer + 语料 + 预训练/偏好数据
├── train/             # train_pretrain.py → train_lora.py → train_dpo.py
├── inference.py       # KV Cache prefill/decode + temperature/top-k/top-p 采样
├── evaluate.py        # 精简 5 段评估（pretrain / moe / lora / dpo / speedup）
├── ablations/         # 与 GPT-2 的退化对照（feat-017，唯一跨目录引用 GPT-2 线的地方）
└── tests/             # 16 个 test_*.py + run_all.py 汇总
```

**核心思路**：不靠"读懂论文"，而是**把 Qwen3 的每个组件做成一个可关掉的开关**（`use_gqa` / `use_rope` / `use_qk_norm` / `mlp_type` / `norm_type`），
关掉之后退化出的模型与 GPT-2 **逐位相同**（实测 `max_abs_diff = 0.000e+00`）——
这样"Qwen3 到底改了什么、为什么改"就变成一次可重跑的实验，而不是一段文字（`feat-017` 已完成）。

---

## 四、笔记索引

### 主线：`gpt-learning/` —— 跟着代码走

| 笔记 | 内容 |
|------|------|
| [0. GPT 全流程详解](gpt-learning/0.GPT全流程详解.md) | 总览：从架构到 RLHF 推理的一页式地图 |
| [1. BPE Tokenizer](gpt-learning/1.bpe-tokenizer.md) | 文本怎么变成 token id；GPT-2 的 Byte-level BPE |
| [2. 位置编码](gpt-learning/2.位置编码.md) | 模型怎么知道 token 的顺序；Sinusoidal vs RoPE |
| [3. Prefill](gpt-learning/3.prefill.md) | 生成之前模型做了什么；KV Cache 怎么来的 |
| [4. Weight Tying](gpt-learning/4.weight-tying.md) | 输入 embedding 和输出分类器共用一张表 |
| [5. 损失函数](gpt-learning/5.损失函数.md) | 错位一位预测下一个 token；`ignore_index` 怎么 mask |
| [6. 训练流程](gpt-learning/6.训练流程.md) | 完整训练循环、warmup、cosine decay、梯度裁剪 |
| [7. SFT 训练](gpt-learning/7.sft-train.md) | 监督微调：prompt 屏蔽、只对回答算 loss |
| [8. 训练奖励模型](gpt-learning/8.训练奖励模型.md) | pairwise 排序损失怎么设计 |
| [9. 强化学习](gpt-learning/9.强化学习.md) | RL 基础到 PPO 的推导脉络 |
| [10. GPT 全流程训练 + 代码解析](gpt-learning/10.gpt全流程训练+代码解析.md) | 逐行读训练脚本（问答式拆解） |
| [11. model.train()](gpt-learning/11.model-train.md) | 训练模式 / 评估模式到底切换了什么 |
| [12. 采样方式](gpt-learning/12.采样方式.md) | prefill / decode、temperature、top-k、top-p |
| [13. GPT-2 评估](gpt-learning/13.gpt2评估.md) | 9 项指标分别看什么、往哪个方向看 |
| [14. GPT + PPO](gpt-learning/14.gpt-ppo.md) | 4 个模型的角色、GAE、双 clip、KL 惩罚 |
| [GPT-2 训练数学公式与超参详解](gpt-learning/GPT2训练数学公式与超参详解.md) | 每个超参的公式依据和取值范围 |
| [代码阅读顺序与进度](gpt-learning/代码阅读顺序与进度.md) | 按什么顺序读代码、目前进度 |

### 副线：`qwen3-learning/` —— 现代架构在哪些位置做了改进

| 笔记 | 对照主线 | Qwen3 源码 |
|------|----------|-----------|
| [Qwen3 RoPE](qwen3-learning/Qwen3_RoPE_学习笔记.md) | 对比笔记 2 的 Sinusoidal 位置编码 | [`model/common/rope.py`](qwen3-learning/qwen3-pytorch/model/common/rope.py) |
| [Qwen3 RMSNorm](qwen3-learning/Qwen3_RMSNorm_学习笔记.md) | 对比 LayerNorm 在 Pre-LN 结构里的作用 | [`model/common/rmsnorm.py`](qwen3-learning/qwen3-pytorch/model/common/rmsnorm.py) |
| [Qwen3 QK-Norm](qwen3-learning/Qwen3_QK-Norm_学习笔记.md) | 注意力数值稳定性 | [`model/common/attention.py`](qwen3-learning/qwen3-pytorch/model/common/attention.py) |
| [Qwen3 SwiGLU MLP](qwen3-learning/Qwen3_SwiGLU_MLP_学习笔记.md) | 对比主线 `model/gpt2.py` 的 GELU MLP | [`model/common/mlp.py`](qwen3-learning/qwen3-pytorch/model/common/mlp.py) |
| [Qwen3 MoE](qwen3-learning/Qwen3_MoE_学习笔记.md) | 从稠密 FFN 到稀疏专家 | [`model/qwen3_moe.py`](qwen3-learning/qwen3-pytorch/model/qwen3_moe.py) |
| [Qwen3 GQA 与 KV Cache](qwen3-learning/Qwen3_GQA与KV_Cache_完整学习笔记.md) | 对比主线笔记 3、12 的 MHA + KV Cache | [`model/common/attention.py`](qwen3-learning/qwen3-pytorch/model/common/attention.py)、[`model/common/kv_cache.py`](qwen3-learning/qwen3-pytorch/model/common/kv_cache.py) |
| [Qwen3 BPE Tokenizer](qwen3-learning/Qwen3_BPE_学习笔记.md) | 对比主线字符级 `CharTokenizer`（主线**没有** BPE 代码） | [`data/bpe.py`](qwen3-learning/qwen3-pytorch/data/bpe.py)、[`data/tokenizer.py`](qwen3-learning/qwen3-pytorch/data/tokenizer.py) |
| [Qwen3 LoRA 与 DPO](qwen3-learning/Qwen3_LoRA与DPO_学习笔记.md) | 对比主线全参 SFT + RM + PPO（主线**没有** LoRA/DPO） | [`model/common/lora.py`](qwen3-learning/qwen3-pytorch/model/common/lora.py)、[`train/train_lora.py`](qwen3-learning/qwen3-pytorch/train/train_lora.py)、[`train/train_dpo.py`](qwen3-learning/qwen3-pytorch/train/train_dpo.py) |
| [Qwen3 训练推理完整学习笔记](qwen3-learning/Qwen3_训练推理完整学习笔记.md) | 端到端训练与推理流程 | [`train/train_pretrain.py`](qwen3-learning/qwen3-pytorch/train/train_pretrain.py)、[`inference.py`](qwen3-learning/qwen3-pytorch/inference.py)、[`evaluate.py`](qwen3-learning/qwen3-pytorch/evaluate.py) |

> **每篇笔记开头都有「对应 GPT-2 实现位置 / 对应 Qwen3 源码位置 / 对应可跑测试」三行**，
> 点开就能跳到具体文件；这些链接由 `qwen3-learning/qwen3-pytorch/tests/test_note_refs.py` **自动校验**
> —— 链接指向的文件不存在时，测试直接红。

> **笔记里的超参数字 ↔ `config.py` 也自动校验**：
> `qwen3-learning/qwen3-pytorch/tests/test_config_sync.py` 把两条线 27 篇笔记里
> 形如 `` `超参名 = 值` `` 的写法抽出来，跟各自的 `config.py` 逐条对齐 ——
> **改完 `config.py` 忘了改笔记，测试直接红**。
> 不属于本仓库配置的取值（教学举例、官方模型参数、算法定义表……）必须**登记理由**。

---

## 五、快速开始

### 环境准备

```bash
# 依赖只有三个
pip install -r gpt-learning/gpt-learning-pytorch/requirements.txt   # torch / numpy / nltk
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

> ✅ **torch 环境已就绪**：本机命中 `Python 3.10.4 + torch 2.1.0+cpu`。
> `init.sh` 会自动挑一个**能 `import torch` 的解释器**（PATH 上的 Python 3.11.0a2 alpha ABI 不兼容 torch）。
> 仍报 `WinError 127` 时先跑一遍 `bash init.sh`，它会明确告诉你哪一层可用。

### 一键轻量验证

```bash
bash init.sh
```

约 1 分钟，分五层检查：L0 环境 → L1 纯 Python 数据层 → L2 GPT-2 模型前向 → L3 仓库卫生 → L4 Qwen3 全链路
（组件 smoke / 前向 / 训练步 / LoRA / DPO / 推理 / 评估 / 退化对照，共 8 条）。

### 完整训练流程（CPU 约 10 分钟，P1 线）

```bash
cd gpt-learning/gpt-learning-pytorch

python data/build_vocab.py        # 扫描合成数据生成词表

python train_pretrain.py          # ① 预训练 → checkpoints/pretrain.pt
python train_sft.py               # ② SFT     → checkpoints/sft.pt
python train_reward.py            # ③ 奖励模型 → checkpoints/reward.pt
python train_ppo.py               # ④ PPO     → checkpoints/ppo.pt
```

### 评估

```bash
python evaluate.py --stage all --all_checkpoints checkpoints/
```

一次输出 9 项指标：困惑度、pairwise accuracy、Distinct-N、Self-BLEU、token entropy、
response quality、SFT vs PPO、toy benchmark、KV speedup。单项评估见
[`gpt-learning-pytorch/README.md`](gpt-learning/gpt-learning-pytorch/README.md)。

### Qwen3 线完整流程（CPU 约 3 分钟，P2 线）

```bash
cd qwen3-learning/qwen3-pytorch

python data/build_vocab.py               # 训练并落盘 byte-level BPE 词表（vocab 778）

python train/train_pretrain.py --model dense   # ① 预训练 → checkpoints/pretrain_dense.pt
python train/train_pretrain.py --model moe     # ①' 预训练（MoE 变体）
python train/train_lora.py               # ② LoRA 微调（r=4，注入 q_proj / v_proj）
python train/train_dpo.py                # ③ DPO（π_ref = LoRA 后的 SFT 模型）

python evaluate.py --stage all            # 精简 5 段评估
python ablations/compare_with_gpt2.py    # 与 GPT-2 的退化对照（6 行消融表）
python tests/run_all.py                  # 17 个 smoke 汇总
```

单项说明见 [`qwen3-pytorch/README.md`](qwen3-learning/qwen3-pytorch/README.md)。

### 推荐学习顺序

1. 读笔记 **0** 拿到全局地图
2. 按 [代码阅读顺序与进度](gpt-learning/代码阅读顺序与进度.md) 读 `model/gpt2.py` → `data/` → `train_pretrain.py`
3. 每读完一个模块，回头读对应章节笔记（2 / 4 / 5）
4. 跑通预训练后读 6、7，再依次跑 SFT → RM → PPO，每跑通一步读一篇对应笔记
5. 最后读 Qwen3 副线（9 篇），对照着看现代架构改了什么；
   每篇笔记开头都有「对应 GPT-2 实现位置 / 对应 Qwen3 源码位置 / 对应可跑测试」三行，可直接跳过去读

---

## 六、代码风格约定

| 约定 | 说明 |
|------|------|
| 超参集中 | 所有超参在 `config.py`，训练脚本不硬编码 |
| 中文注释 | 每个类、每个方法都有中文 docstring，风格对齐 `model/gpt2.py` |
| 延迟 import | 模块顶层延迟 import torch，避免 torch 故障炸掉整个数据层 |
| 形状注释 | 关键函数 docstring 里附「输入输出形状速查表」 |
| 可复现 | 随机种子显式写死，`eval` 模式同输入应 bit-identical |

---

## 七、Harness 机制（给 AI agent 的工作装备）

本仓库对 AI 协作做了完整约束，规则都写在文件里而不是靠聊天记录：

| 文件 | 作用 |
|------|------|
| [`AGENTS.md`](AGENTS.md) | 操作手册：项目定位与阶段、硬约束、五层验证路径、Definition of Done |
| [`init.sh`](init.sh) | 标准启动 + 轻量验证，一个命令知道环境是否健康（L0–L4） |
| [`feature_list.json`](feature_list.json) | 任务状态机：含 `phase` 阶段标记，WIP=1，每条 feature 带可执行验证步骤 |
| [`claude-progress.md`](claude-progress.md) | session 日记 + 当前已验证状态 |
| [`session-handoff.md`](session-handoff.md) | 跨 session 交接卡（当前记录：P2 阶段 Qwen3 开发接手须知） |
| [`clean-state-checklist.md`](clean-state-checklist.md) | 收班清单（6 项通用 + 3 项本项目特有） |
| [`harness.md`](harness.md) | 通用方法论手册：5 子系统、任务拆解、反模式全集 |

给 agent 的最短指令：

```bash
cat AGENTS.md && cat feature_list.json && bash init.sh
```

---

## 八、本地不提交的内容

`.gitignore` 已排除以下内容，它们保留在本机磁盘上：

- `checkpoints/`、`*.pt` / `*.pth` / `*.bin` 等模型权重
- `docs/` 本地资料（图、规格草稿）
- `__pycache__/`、`.claude/`、`.vscode/`
- 日志、缓存、虚拟环境

查看当前被忽略的文件：`git status --ignored`

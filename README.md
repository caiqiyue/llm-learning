# llm-learning

> **从零手写 tiny GPT-2，跑通「预训练 → SFT → Reward Model → PPO」全流程**，配套 24 篇中文学习笔记。
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
├── feature_list.json            ← 任务状态机：现在该做什么
├── claude-progress.md           ← session 进度日志
├── session-handoff.md           ← 跨 session 交接卡
├── clean-state-checklist.md     ← 收班清单
│
├── gpt-learning/                        ← 【主线】GPT-2 + RLHF 全流程
│   ├── 0~14 章系列笔记 + 数学公式详解 + 代码阅读顺序
│   ├── requirements.txt
│   ├── checkpoints/                     ← 训练产物（不入库）
│   ├── docs/                            ← 本地资料（不入库）
│   └── gpt-learning-pytorch/            ← 唯一主线实现（PyTorch）
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
└── qwen3-learning/                      ← 【副线】Qwen3 现代架构笔记（7 篇）
```

> `checkpoints/` 和 `docs/` 被 `.gitignore` 排除：权重文件和本地草稿留在本机，不进仓库。

---

## 三、笔记索引

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

| 笔记 | 对照主线 |
|------|----------|
| [Qwen3 RoPE](qwen3-learning/Qwen3_RoPE_学习笔记.md) | 对比笔记 2 的 Sinusoidal 位置编码 |
| [Qwen3 RMSNorm](qwen3-learning/Qwen3_RMSNorm_学习笔记.md) | 对比 LayerNorm 在 Pre-LN 结构里的作用 |
| [Qwen3 QK-Norm](qwen3-learning/Qwen3_QK-Norm_学习笔记.md) | 注意力数值稳定性 |
| [Qwen3 SwiGLU MLP](qwen3-learning/Qwen3_SwiGLU_MLP_学习笔记.md) | 对比主线 `model/gpt2.py` 的 GELU MLP |
| [Qwen3 MoE](qwen3-learning/Qwen3_MoE_学习笔记.md) | 从稠密 FFN 到稀疏专家 |
| [Qwen3 GQA 与 KV Cache](qwen3-learning/Qwen3_GQA与KV_Cache_完整学习笔记.md) | 对比主线笔记 3、12 的 MHA + KV Cache |
| [Qwen3 训练推理完整学习笔记](qwen3-learning/Qwen3_训练推理完整学习笔记.md) | 端到端训练与推理流程 |

---

## 四、快速开始

### 环境准备

```bash
# 依赖只有三个
pip install -r gpt-learning/gpt-learning-pytorch/requirements.txt   # torch / numpy / nltk
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

> ⚠️ **已知问题**：部分 Windows 环境下 `import torch` 会报 `WinError 127`（`torch_python.dll` 加载失败）。
> 先跑一遍 `bash init.sh` 确认环境，它会明确告诉你哪一层可用。

### 一键轻量验证

```bash
bash init.sh
```

约 1 分钟，分四层检查：环境 → 纯 Python 数据层 → torch 模型前向 → 仓库卫生。

### 完整训练流程（CPU 约 10 分钟）

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

### 推荐学习顺序

1. 读笔记 **0** 拿到全局地图
2. 按 [代码阅读顺序与进度](gpt-learning/代码阅读顺序与进度.md) 读 `model/gpt2.py` → `data/` → `train_pretrain.py`
3. 每读完一个模块，回头读对应章节笔记（2 / 4 / 5）
4. 跑通预训练后读 6、7，再依次跑 SFT → RM → PPO，每跑通一步读一篇对应笔记
5. 最后读 Qwen3 副线，对照着看现代架构改了什么

---

## 五、代码风格约定

| 约定 | 说明 |
|------|------|
| 超参集中 | 所有超参在 `config.py`，训练脚本不硬编码 |
| 中文注释 | 每个类、每个方法都有中文 docstring，风格对齐 `model/gpt2.py` |
| 延迟 import | 模块顶层延迟 import torch，避免 torch 故障炸掉整个数据层 |
| 形状注释 | 关键函数 docstring 里附「输入输出形状速查表」 |
| 可复现 | 随机种子显式写死，`eval` 模式同输入应 bit-identical |

---

## 六、Harness 机制（给 AI agent 的工作装备）

本仓库对 AI 协作做了完整约束，规则都写在文件里而不是靠聊天记录：

| 文件 | 作用 |
|------|------|
| [`AGENTS.md`](AGENTS.md) | 操作手册：项目定位、硬约束、三层验证路径、Definition of Done |
| [`init.sh`](init.sh) | 标准启动 + 轻量验证，一个命令知道环境是否健康 |
| [`feature_list.json`](feature_list.json) | 任务状态机，WIP=1，每条 feature 都带可执行验证步骤 |
| [`claude-progress.md`](claude-progress.md) | session 日记 + 当前已验证状态 |
| [`session-handoff.md`](session-handoff.md) | context 用满时的交接卡 |
| [`clean-state-checklist.md`](clean-state-checklist.md) | 收班 6 项自查 |
| [`harness.md`](harness.md) | 通用方法论手册：5 子系统、任务拆解、反模式全集 |

给 agent 的最短指令：

```bash
cat AGENTS.md && cat feature_list.json && bash init.sh
```

---

## 七、本地不提交的内容

`.gitignore` 已排除以下内容，它们保留在本机磁盘上：

- `checkpoints/`、`*.pt` / `*.pth` / `*.bin` 等模型权重
- `docs/` 本地资料（图、规格草稿）
- `__pycache__/`、`.claude/`、`.vscode/`
- 日志、缓存、虚拟环境

查看当前被忽略的文件：`git status --ignored`

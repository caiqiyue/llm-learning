# Tiny GPT-2 + RLHF (PyTorch)

> ~0.8M 参数 GPT-2 跑通 预训练 → SFT → Reward Model → PPO 全流程；含 KV Cache 推理 + 9 项评估。

## 简介

本目录是项目 **PyTorch 实现**（主学）。模型与 TF 版**结构镜像**但**不共享代码**，全部从零手写并带详细中文注释，方便逐行 debugger。

## 安装

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

## 数据准备

```bash
python data/build_vocab.py
```

会扫描所有合成数据 → 写 `data/vocab.txt`（45 个字符，含 5 个 special token + 40 个字母数字符号）。

## 训练流程（4 步，按顺序）

```bash
python train_pretrain.py     # ① 预训练，输出 checkpoints/pretrain.pt  (lr=3e-4, 10 epoch)
python train_sft.py          # ② SFT，    输出 checkpoints/sft.pt       (lr=1e-5,  5 epoch)
python train_reward.py       # ③ RM，     输出 checkpoints/reward.pt    (lr=1e-5,  5 epoch)
python train_ppo.py          # ④ PPO，    输出 checkpoints/ppo.pt       (lr=1e-5,  100 rollout)
```

每步预计耗时 30s-2min（CPU），端到端总时间 < 10 分钟。

## 评估

```bash
python evaluate.py --stage pretrain  --checkpoint checkpoints/pretrain.pt
python evaluate.py --stage sft       --checkpoint checkpoints/sft.pt
python evaluate.py --stage reward    --checkpoint checkpoints/reward.pt
python evaluate.py --stage ppo       --checkpoint checkpoints/ppo.pt
python evaluate.py --stage compare   --sft sft.pt --ppo ppo.pt --rm reward.pt
python evaluate.py --stage diversity --checkpoint checkpoints/ppo.pt
python evaluate.py --stage benchmark --checkpoint checkpoints/pretrain.pt
python evaluate.py --stage speedup   --checkpoint checkpoints/ppo.pt
python evaluate.py --stage all       --all_checkpoints checkpoints/
```

9 个评估函数：PPL、pairwise accuracy、Distinct-N、Self-BLEU、token entropy、response quality、SFT vs PPO、toy benchmark、KV speedup。

## 训练口径（三条线必须一致，feat-027）

| 约定 | 在哪 | 为什么 |
|------|------|--------|
| `labels` / `target_ids` 与 `input_ids` **同下标** | `data/sft_data.py` / `data/pretrain_data.py` | 错位（"logits[t] 预测 token t+1"）**只在模型里做一次**（`model/gpt2.py`：`CE(logits[:, :-1], targets[:, 1:])`）⇒ 两条数据线口径必然一致，不会一边左移一边不左移 |
| 特殊 token（`<bos>` `<sep>` `<eos>`）**整体一个 id** | `data/tokenizer.py` 的 `encode`（最长匹配优先） | 否则 `BOS/EOS/SEP` 三个 id 永远不会被产生 ⇒ 模型学不会自己停，生成永远跑满 `max_new_tokens` |
| 回答按 **token id 个数**切，不按字符串长度切 | `evaluate._generate_responses` / `generate.generate_response` | `UNK` 解码回来是 5 字符的 `"<unk>"`，按字符数切会错位，每条回答都被塞上 `"? <unk>sep<unk> "` 前缀 |

> 这三条都有**机器守卫**：`bash init.sh` 的 L1（数据线同下标 + 特殊 token 单 id）
> 与 L2（模型 loss 确实错位）会直接把改坏的代码判红。

## 推理示例

```python
from inference import KVCache, GPT2WithKVCache
from data.tokenizer import CharTokenizer
from model.gpt2 import GPT2
import torch

tok = CharTokenizer(vocab_path="data/vocab.txt")
model = GPT2(vocab_size=tok.vocab_size, n_layer=4, n_head=4, n_embd=128,
             n_inner=512, block_size=128, dropout=0.0, bias=False)
sd = torch.load("checkpoints/ppo.pt", map_location="cpu")
model.load_state_dict(sd["model"])
model.eval()

wrapped = GPT2WithKVCache(model)
cache = KVCache(n_layer=4, n_head=4, head_dim=32, max_seq_len=128)

prompt = torch.tensor([[2] + tok.encode("Once upon a time, the ") + [1]])  # <bos> ... <sep>
pos = torch.arange(prompt.size(1)).unsqueeze(0)
logits, cache = wrapped(prompt, cache, pos)
# 再 decode 一个 token: position = prompt_len
...
```

## 文件清单

| 文件 | 用途 |
|---|---|
| `model/gpt2.py` | GPT-2 主模型（4 Block、weight tying、KV cache 接口） |
| `model/value_head.py` | PPO value head |
| `data/*.py` | 4 个数据集生成器 + tokenizer + benchmark |
| `train_pretrain.py` | 阶段 ① 预训练 |
| `train_sft.py` | 阶段 ② SFT（prompt-masked loss） |
| `train_reward.py` | 阶段 ③ Reward Model（pairwise ranking） |
| `train_ppo.py` | 阶段 ④ PPO（GAE + clipped objective） |
| `inference.py` | KVCache + 增量 forward + batched_generate |
| `generate.py` | generate_naive / generate_kv / benchmark_speed |
| `evaluate.py` | 9 个评估函数 + argparse 入口 |
| `utils.py` / `config.py` | 工具与超参 |

## 跨框架对比

详见 [`../docs/cross-framework-comparison.md`](../docs/cross-framework-comparison.md)。

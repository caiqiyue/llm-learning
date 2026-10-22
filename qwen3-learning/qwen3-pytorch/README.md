# qwen3-pytorch

从零手写 tiny Qwen3：RoPE + RMSNorm + QK-Norm + SwiGLU + GQA，外加 MoE 变体，
走通 预训练 → LoRA → DPO 三段。

设计文档：[../qwen3-pytorch设计.md](../qwen3-pytorch设计.md) ｜ 开发计划：[../qwen3-pytorch开发计划.md](../qwen3-pytorch开发计划.md)

## 快速开始

```bash
python tests/run_all.py            # 全部组件 smoke
python data/build_vocab.py         # 训练 byte-level BPE，生成 vocab.json / merges.txt
python train/train_pretrain.py --model dense
python train/train_pretrain.py --model moe
python train/train_lora.py
python train/train_dpo.py
python evaluate.py --stage all
```

## 概念 → 笔记 → 源码

| 概念 | 笔记 | 源码 |
|---|---|---|
| RMSNorm | [RMSNorm_学习笔记](../Qwen3_RMSNorm_学习笔记.md) | `model/common/rmsnorm.py` |
| RoPE | [RoPE_学习笔记](../Qwen3_RoPE_学习笔记.md) | `model/common/rope.py` |
| QK-Norm | [QK-Norm_学习笔记](../Qwen3_QK-Norm_学习笔记.md) | `model/common/attention.py` |
| GQA / KV Cache | [GQA与KV_Cache](../Qwen3_GQA与KV_Cache_完整学习笔记.md) | `model/common/attention.py`、`model/common/kv_cache.py` |
| SwiGLU | [SwiGLU_MLP_学习笔记](../Qwen3_SwiGLU_MLP_学习笔记.md) | `model/qwen3_dense.py` |
| MoE | [MoE_学习笔记](../Qwen3_MoE_学习笔记.md) | `model/qwen3_moe.py` |
| 训练 → 推理 | [训练推理完整学习笔记](../Qwen3_训练推理完整学习笔记.md) | `train/`、`inference.py` |

## 与 GPT-2 线的对照

本线不跨目录 import GPT-2 线代码，但为对照实验保留了相同接口：
`data/pretrain_data.py::make_pretrain_loader` 与 `data/preference_data.py::generate_preference_dataset`
的签名与 GPT-2 线一致，见 `ablations/compare_with_gpt2.py`。

## 硬约束

- 依赖上限：`torch` / `numpy` / `nltk`，不引入 pytest / peft / transformers / sentencepiece
- 超参全部来自 `config.py`
- `checkpoints/` 不入库

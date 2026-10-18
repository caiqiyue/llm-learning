"""
debug_sft_data.py — 单步调试 SFT 数据（含 prompt-masked loss）

跑法（在 gpt-learning-pytorch/ 目录下）：
    python data/debug_sft_data.py

目的：
  - 看 prompt 模板 → response 模板 的映射逻辑（pick_response_templates）
  - 看占位符填充结果
  - ★ 看 format_sft_sample 的完整拼接格式 "<bos> prompt <sep> response <eos>"
  - ★ 看 SFTDataset 的关键：prompt_len + labels = -100 mask
  - ★ 看 sft_collate_fn 的 padding：batch 内不同长度对齐 + 对 padding 也设 -100

★ 这是 SFT 训练的关键 debug 脚本！（prompt 部分的 label 是否被正确 mask？）
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PJT = os.path.dirname(_HERE)
sys.path.insert(0, _PJT)

from data.sft_data import (  # noqa: E402
    SFT_PROMPT_TEMPLATES,
    FACTUAL_TEMPLATES,
    ACTION_TEMPLATES,
    DESCRIPTIVE_TEMPLATES,
    pick_response_templates,
    generate_sft_dataset,
    format_sft_sample,
)
from data.tokenizer import CharTokenizer, BOS_ID, EOS_ID, SEP_ID, PAD_ID  # noqa: E402

# torch 环境检测
try:
    import torch
    from data.sft_data import SFTDataset, sft_collate_fn, make_sft_loader
    TORCH_OK = True
    TORCH_ERR = None
except Exception as e:
    TORCH_OK = False
    TORCH_ERR = repr(e)
    SFTDataset = sft_collate_fn = make_sft_loader = None


def hr(title: str):
    print(f"\n{'=' * 64}\n {title}\n{'=' * 64}")


def visualize(tk, ids, sep_token=None):
    """把 token id 列表可视化为 [id]char 形式。"""
    parts = []
    for tid in ids[:60]:
        c = tk.itos[tid] if 0 <= tid < tk.vocab_size else "?"
        c = c if c != "\n" else "\\n"
        # 高亮关键 token
        if sep_token is not None and tid == sep_token:
            c = f"«{c}»"
        parts.append(f"[{tid}]{c!r}")
    print("    " + " ".join(parts))


def main():
    vocab_path = os.path.join(_HERE, "vocab.txt")
    if not os.path.exists(vocab_path):
        print("[!] vocab.txt 不存在，请先跑: python data/build_vocab.py")
        return

    tk = CharTokenizer(vocab_path)

    hr("Step 1: 模板池概览")
    print(f"prompt 模板数: {len(SFT_PROMPT_TEMPLATES)}")
    print(f"factual response 模板数: {len(FACTUAL_TEMPLATES)}")
    print(f"action response 模板数: {len(ACTION_TEMPLATES)}")
    print(f"descriptive response 模板数: {len(DESCRIPTIVE_TEMPLATES)}")

    hr("Step 2: prompt 模板 → response 模板的映射（关键！）")
    print("下面演示每个 prompt 模板对应哪些 response 模板：\n")
    for i, p in enumerate(SFT_PROMPT_TEMPLATES):
        resps = pick_response_templates(p)
        category = (
            "ACTION"      if resps is ACTION_TEMPLATES else
            "DESCRIPTIVE" if resps is DESCRIPTIVE_TEMPLATES else
            "FACTUAL"
        )
        print(f"[{i:2}] ({category})")
        print(f"     prompt: {p!r}")
        for r in resps:
            print(f"     → resp: {r!r}")
        print()

    hr("Step 3: 单样本生成 + 占位符填充")
    samples = generate_sft_dataset(n_samples=5, seed=42)
    print(f"取前 5 条样本:\n")
    for s in samples:
        print(f"id={s['id']}")
        print(f"  prompt:   {s['prompt']}")
        print(f"  response: {s['response']}")
        print()

    hr("Step 4: format_sft_sample 的拼接格式 (核心★)")
    for s in samples[:2]:
        full = format_sft_sample(s["prompt"], s["response"])#'<bos> Tell me about a cat. <sep> The rabbit is quiet and lives in the mountain. <eos>'
        print(f"\nprompt:   {s['prompt']!r}")
        print(f"response: {s['response']!r}")
        print(f"\n→ format_sft_sample 后的完整字符串:")
        print(f"  {full!r}")
        print(f"\n→ tokenize:")
        ids = tk.encode(full)
        print(f"  ids = {ids}")
        visualize(tk, ids)
        # 检查关键 token id
        print(f"\n特殊 token id 位置:")
        for name, tid in [("BOS_ID", BOS_ID), ("SEP_ID", SEP_ID), ("EOS_ID", EOS_ID)]:
            positions = [i for i, x in enumerate(ids) if x == tid]
            print(f"  {name}={tid}  出现在位置: {positions}")

    hr("Step 5: SFTDataset.__init__ —— prompt_len + labels 的关键实现 (核心★)")
    print("★ prompt_len = tokenize('<bos>{prompt}<sep>') 的长度")
    print("★ labels = [-100]*prompt_len + input_ids[prompt_len:]")
    print("  → response + <eos> 部分才是 loss 计算区域\n")

    if not TORCH_OK:
        print(f"[跳过] torch 不可用，无法演示 SFTDataset")
        print(f"       原因: {TORCH_ERR}")
    else:
        ds = SFTDataset(samples)#看懂这一个即可
        print(f"Dataset 长度: {len(ds)}")
        print(f"\n3 条样本的 prompt_len 统计:")
        for i in range(3):
            item = ds[i]
            x = item["input_ids"].tolist()
            y = item["labels"].tolist()
            prompt_len = item["prompt_len"].item()
            n_masked = sum(1 for v in y if v == -100)
            n_real = sum(1 for v in y if v != -100)
            print(f"\nsample[{i}]: total_len={len(x)}  prompt_len={prompt_len}  masked={n_masked}  real_loss={n_real}")
            print(f"  input_ids  : {x}")
            print(f"  labels     : {y}")
            print(f"  mask 区段  : [{' '.join(str(v) for v in y[:prompt_len])}]  ← -100 (=masked)")
            print(f"  算 loss 区段: [{' '.join(str(v) for v in y[prompt_len:])}]  ← 真实 token id")

            # ★ 把 token id 翻译回字符，验证
            prompt_chars = "".join(tk.itos[t] if t in tk.itos else "?" for t in x[:prompt_len])
            resp_chars = "".join(tk.itos[t] if t in tk.itos else "?" for t in x[prompt_len:])
            print(f"  prompt 部分字符串: {prompt_chars!r}")
            print(f"  resp   部分字符串: {resp_chars!r}")

    hr("Step 6: sft_collate_fn 的 padding (B, L_max)")
    if not TORCH_OK:
        print(f"[跳过] torch 不可用，无法演示 sft_collate_fn")
        print(f"       预期行为：批量 pad 到 L_max, padding 位置 labels=-100")
    else:
        print("★ 不同样本长度不一致，collate 时 pad 到 batch 内最大长度")
        print("★ padding 位置也设为 -100（在 labels 上），不让 padding 算 loss\n")

        test_dataset = SFTDataset(samples[:5])
        batch = sft_collate_fn([test_dataset[0], test_dataset[1], test_dataset[2]])
        print(f"Collate 3 条样本后:")
        print(f"  input_ids:  shape={tuple(batch['input_ids'].shape)}  dtype={batch['input_ids'].dtype}")
        print(f"  labels:     shape={tuple(batch['labels'].shape)}    dtype={batch['labels'].dtype}")
        print(f"  prompt_len: shape={tuple(batch['prompt_len'].shape)} dtype={batch['prompt_len'].dtype}")
        print(f"\n详细形状: 输入 {batch['input_ids'].shape[0]} 条样本 -> B={batch['input_ids'].shape[0]}, L={batch['input_ids'].shape[1]}")
        print(f"  prompt_len: {batch['prompt_len'].tolist()}")

        print(f"\n每行展开:")
        print(f"{'#':<3}{'input_ids':<40}{'labels':<40}{'masked?':<10}")
        x = batch["input_ids"]
        y = batch["labels"]
        for i in range(x.size(0)):
            x_str = " ".join(str(t.item()) for t in x[i])
            y_str = " ".join(f"{t.item():>3}" for t in y[i])
            masked_pos = sum(1 for v in y[i].tolist() if v == -100)
            total = y[i].size(0)
            print(f"{i:<3}{x_str:<40}{y_str:<40}-100数:{masked_pos}/{total}")

    hr("Step 7: make_sft_loader")
    if not TORCH_OK:
        print(f"[跳过] torch 不可用，无法演示 make_sft_loader")
        print(f"       预期输出: dict{{input_ids:(B, L), labels:(B, L), prompt_len:(B,)}}")
    else:
        loader = make_sft_loader(split="train", batch_size=4, seed=42, n_samples=50)
        print(f"loader 类型: {type(loader).__name__}")
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= 2:
                break
            print(f"\nbatch[{batch_idx}]:")
            print(f"  input_ids:  shape={tuple(batch['input_ids'].shape)}")
            print(f"  labels:     shape={tuple(batch['labels'].shape)}")
            print(f"  prompt_len: shape={tuple(batch['prompt_len'].shape)}")
            labels = batch["labels"]
            masked_per_sample = (labels == -100).sum(dim=1).tolist()
            print(f"  每条样本 -100 数 (masked): {masked_per_sample}")
            prompt_len = batch["prompt_len"].tolist()
            match = all(m >= p for m, p in zip(masked_per_sample, prompt_len))
            print(f"  masked ≥ prompt_len? {match}  (应该全部 True)")

    print("\n" + "=" * 64)
    print(" [OK] sft_data debug 完成")
    print("=" * 64)


if __name__ == "__main__":
    main()

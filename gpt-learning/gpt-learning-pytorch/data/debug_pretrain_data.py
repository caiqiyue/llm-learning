"""
debug_pretrain_data.py — 单步调试 Pretrain 数据生成 + 切块 + DataLoader

跑法（在 gpt-learning-pytorch/ 目录下）：
    python data/debug_pretrain_data.py

目的：
  - 看模板填充流程
  - 看 90/10 train/val 切分（按 id 稳定哈希）
  - 看 format_pretrain_for_lm 的拼接 + 切块（输入 target 错位 1 位）
  - 看 DataLoader batch shape  (B, T) -> (B, T) <- target

★ 这一份 debug 涵盖了**整条 pretrain 数据管道**，是最重要的 debug 脚本之一。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PJT = os.path.dirname(_HERE)
sys.path.insert(0, _PJT)

from data.pretrain_data import (  # noqa: E402
    TEMPLATES,
    ENTITIES,
    generate_pretrain_sample,
    generate_pretrain_dataset,
    format_pretrain_for_lm,
    _stable_train_val_mask,
)

# torch 环境检测（很多步骤需要它）
try:
    import torch
    from data.pretrain_data import make_pretrain_loader, PretrainDataset
    TORCH_OK = True
    TORCH_ERR = None
except Exception as e:
    TORCH_OK = False
    TORCH_ERR = repr(e)

from data.tokenizer import UNK_ID, PAD_ID  # noqa: E402


def hr(title: str):
    print(f"\n{'=' * 64}\n {title}\n{'=' * 64}")


def visualize_token_sequence(tk, ids, max_show=40):
    """把 token id 列表可视化为 [id]char。"""
    out = []
    for i, tid in enumerate(ids[:max_show]):
        c = tk.itos[tid] if 0 <= tid < tk.vocab_size else "?"
        c = c if c != "\n" else "\\n"
        out.append(f"[{tid}]{c!r}")
    if len(ids) > max_show:
        out.append(f"... (+{len(ids)-max_show} more)")
    print("    " + " ".join(out))


def main():
    vocab_path = os.path.join(_HERE, "vocab.txt")
    if not os.path.exists(vocab_path):
        print("[!] vocab.txt 不存在，请先跑: python data/build_vocab.py")
        return

    from data.tokenizer import CharTokenizer
    tk = CharTokenizer(vocab_path)

    hr("Step 1: 模板与实体池概览")
    print(f"模板数量: {len(TEMPLATES)}")
    for i, t in enumerate(TEMPLATES):
        print(f"  [{i}] {t}")

    print(f"\n实体池（key → 候选数）:")
    for k, v in ENTITIES.items():
        print(f"  {k:12} = {len(v)} 个  (前 5: {v[:5]})")

    hr("Step 2: 单样本生成（3 条不同模板）")
    import random
    rng = random.Random(0)
    for _ in range(3):
        sample = generate_pretrain_sample(rng)
        print(f"  → {sample}")

    hr("Step 3: 数据集生成（n=20，可视化前 5 条）")
    n = 20
    ds = generate_pretrain_dataset(n_samples=n, seed=42)
    print(f"数据集大小: {len(ds)}")
    print(f"\n字段: {{'text': str, 'id': int}}")
    print(f"前 5 条:")
    for s in ds[:5]:
        print(f"  [{s['id']:2}] {s['text'][:80]}{'...' if len(s['text'])>80 else ''}")

    hr("Step 4: 90/10 切分（看稳定哈希效果）")
    train_ids = [s["id"] for s in ds if _stable_train_val_mask(s["id"])]
    val_ids = [s["id"] for s in ds if not _stable_train_val_mask(s["id"])]
    print(f"train 数: {len(train_ids)}  ({len(train_ids)/n*100:.0f}%)")
    print(f"val   数: {len(val_ids)}    ({len(val_ids)/n*100:.0f}%)")
    print(f"⚠️  n=20 样本太少，比例可能偏离 90/10；n=10000 时比例稳定在 90/10")
    print(f"\n★ 同一个 id 在不同次运行切分结果完全一致（SHA-256 稳定哈希，无全局 RNG 依赖）")

    hr("Step 5: format_pretrain_for_lm 关键步骤（核心！）")
    print("输入: 1000 条文本  →  大字符串  →  tokenize  →  切块")
    print("接下来以一个 toy 数据集演示完整流程")
    toy_samples = generate_pretrain_dataset(n_samples=20, seed=42)
    print(f"\n玩具数据集: 20 条样本")

    # 5.1 拼接
    big_text = "\n".join(s["text"] for s in toy_samples)
    print(f"\n【5.1 拼接】big_text = '\\n'.join(texts)")
    print(f"  big_text 前 100 字符: {big_text[:100]!r}")
    print(f"  big_text 长度: {len(big_text)}")

    # 5.2 tokenize
    ids = tk.encode(big_text)
    print(f"\n【5.2 tokenize】 ids = tk.encode(big_text)")
    print(f"  ids 长度: {len(ids)}  (注意：每个英文单词的每个字母单独成一个 token)")
    print(f"  字符数/token 数 = {len(big_text)/len(ids):.2f}")
    visualize_token_sequence(tk, ids, max_show=30)

    # 5.3 切块
    block_size = 64  # 临时调小一点，看清结构
    n_blocks = len(ids) // block_size
    print(f"\n【5.3 切块】 block_size={block_size}, n_blocks={n_blocks}")

    chunks = format_pretrain_for_lm(toy_samples, block_size=block_size)
    print(f"\n★ format_pretrain_for_lm 返回 List[dict]，每个 dict 含:")
    print(f"   input_ids  (长度 {len(chunks[0]['input_ids'])})")
    print(f"   target_ids (长度 {len(chunks[0]['target_ids'])})")
    print(f"\n【第一块可视化】")
    chunk = chunks[0]
    print(f"input_ids : {chunk['input_ids']}")
    print(f"target_ids: {chunk['target_ids']}")

    # 关键点：target = input 错位 1 位
    print(f"\n★ 关键观察：target_ids[i] 应该是 input_ids[i+1]:")
    all_match = all(
        chunk["target_ids"][i] == chunk["input_ids"][i + 1]
        for i in range(block_size - 1)
    )
    # 最后一格 target_ids[-1] = input_ids[-1] (自监督，不是错位)
    last_ok = chunk["target_ids"][-1] == chunk["input_ids"][-1]
    print(f"  前 {block_size-1} 格 target=input[1:]? {'[OK]' if all_match else '[!]'}")
    print(f"  最后一格 target=input[-1] (兜底)? {'[OK]' if last_ok else '[!]'}")

    print(f"\n【并排打印前 20 格】")
    print(f"{'位置':<6}{'input':<10}{'target':<10}{'input_char':<14}{'target_char':<14}{'差 1 位?'}")
    for i in range(20):
        if i >= block_size:
            break
        ic = tk.itos[chunk["input_ids"][i]]
        tc = tk.itos[chunk["target_ids"][i]]
        ic_disp = ic.replace("\n", "\\n")
        tc_disp = tc.replace("\n", "\\n")
        # 错位检查
        if i < block_size - 1:
            check = chunk["target_ids"][i] == chunk["input_ids"][i + 1]
        else:
            check = chunk["target_ids"][i] == chunk["input_ids"][i]
        mark = "[OK]" if check else "[!]"
        print(f"{i:<6}{chunk['input_ids'][i]:<10}{chunk['target_ids'][i]:<10}{ic_disp!s:<14}{tc_disp!s:<14}{mark}")

    hr("Step 6: 用真实 block_size=128 走一遍完整管道")
    real_samples = generate_pretrain_dataset(n_samples=100, seed=42)#生成了100个样本
    chunks = format_pretrain_for_lm(real_samples, block_size=128)#100个样本，拼接成一个大段文字，然后是每个字符不是单词，是每个字符，a,b,c,d..... 直接转成id，然后就是 按照 block-size 切分成一个一个 训练样本，所以 也就说可能 每个 样本都不一定是语义完整的句子，但是没关系，只要预测下一个词就行了
    print(f"100 条样本 → {len(chunks)} 个块 (block_size=128)")
    print(f"每块 input/target 都是 128 长")

    hr("Step 7: PretrainDataset.__getitem__")
    if not TORCH_OK:
        print(f"[跳过] torch 不可用，无法演示 Dataset.__getitem__")
        print(f"       原因: {TORCH_ERR}")
    else:
        samples = generate_pretrain_dataset(n_samples=10, seed=42)
        train_samples = [s for s in samples if _stable_train_val_mask(s["id"])]
        dataset = PretrainDataset(train_samples, block_size=64)
        print(f"Dataset 长度: {len(dataset)}")
        print(f"\n__getitem__(0) 返回一个 batch sample（单条）:")
        item = dataset[0]
        print(f"  type: {type(item)}  (tuple of 2 tensors)")
        print(f"  x: shape={item[0].shape} dtype={item[0].dtype}  前 10: {item[0][:10].tolist()}")
        print(f"  y: shape={item[1].shape} dtype={item[1].dtype}  前 10: {item[1][:10].tolist()}")

    hr("Step 8: make_pretrain_loader (DataLoader)")
    if not TORCH_OK:
        print(f"[跳过] torch 不可用，无法演示 DataLoader")
        print(f"       DataLoader 输出预期形状: x=(B, T_block), y=(B, T_block)，y[:, :-1]=x[:, 1:]")
    else:
        loader = make_pretrain_loader(split="train", batch_size=8, block_size=64, seed=42, n_samples=100)
        print(f"loader 类型: {type(loader).__name__}")
        print(f"\n★ 每个 batch 是 (x, y) tuple")
        print(f"★ x 和 y 都是 (B, T) shape  ←  B=batch_size=8, T=block_size=64")

        for batch_idx, (x, y) in enumerate(loader):
            if batch_idx >= 2:
                break
            print(f"\nbatch[{batch_idx}]:")
            print(f"  x: shape={tuple(x.shape)}  dtype={x.dtype}  device={x.device}")
            print(f"  y: shape={tuple(y.shape)}  dtype={y.dtype}  device={y.device}")
            print(f"  x[0, :15]: {x[0, :15].tolist()}")
            print(f"  y[0, :15]: {y[0, :15].tolist()}")
            xb, yb = x, y
            all_ok = (yb[:, :-1] == xb[:, 1:]).all().item()
            print(f"  y[:, :-1] == x[:, 1:] ? {all_ok}  ('true' = 错位正确)")

    print("\n" + "=" * 64)
    print(" [OK] pretrain_data debug 完成")
    print("=" * 64)


if __name__ == "__main__":
    main()

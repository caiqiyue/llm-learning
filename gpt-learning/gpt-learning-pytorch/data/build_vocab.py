"""
扫描所有合成数据集 → 写 vocab.txt。

字符表结构（与 dataset_construction.md §6 / tokenizer.py 锁死的 id 一致）：
  - 前 5 行是 special tokens，顺序固定：
        <unk>=0, <pad>=1, <bos>=2, <eos>=3, <sep>=4
  - 从第 6 行起是按 Unicode 排序的普通字符
  - 总 vocab_size = 5 + len(chars)，预期约 64

用法：
    # 在 gpt-learning-pytorch/ 目录下：
    python data/build_vocab.py        # 直接执行（无 if __name__ 包装）
    python -m data.build_vocab        # 同上（module 形式）

约束（用户明确要求）：
  - 顶部不写 if __name__ == "__main__"
  - 文件最后直接调用 build_vocab()，构成线性"导入即执行"脚本
  - 与 tokenizer.py 共用 SPECIAL_TOKENS 常量，保证 id 顺序锁死
"""

import os

# 5 个特殊 token；顺序决定 id，必须与 tokenizer.py 的 UNK_ID=0, PAD_ID=1, BOS_ID=2, EOS_ID=3, SEP_ID=4 一致
SPECIAL_TOKENS = ["<unk>", "<pad>", "<bos>", "<eos>", "<sep>"]


def build_vocab(save_path: str = None) -> int:
    """
    收集所有出现过的字符，写入 vocab.txt。

    数据来源（必须 P1 实现完成才可工作）：
      - pretrain_data: 2000 条文本
      - sft_data:      1000 条 (prompt + response)
      - preference_data: 1000 条 (prompt + chosen + rejected)

    参数：
        save_path: 输出文件路径；默认 data/vocab.txt
    返回：
        写入的 token 总数（含 5 个 special tokens）
    """
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), "vocab.txt")

    # 延迟 import：避免 P0 stub 阶段 import 错误
    from data.pretrain_data import generate_pretrain_dataset
    from data.sft_data import generate_sft_dataset
    from data.preference_data import generate_preference_dataset

    # 收集所有文本
    texts = []
    for s in generate_pretrain_dataset(n_samples=2000, seed=42):
        texts.append(s["text"])
    for s in generate_sft_dataset(n_samples=1000, seed=42):
        texts.append(s["prompt"])
        texts.append(s["response"])
    for s in generate_preference_dataset(n_samples=1000, seed=42):
        texts.append(s["prompt"])
        texts.append(s["chosen"])
        texts.append(s["rejected"])

    # 收集所有出现过的字符，按 Unicode 排序
    chars = sorted(set("".join(texts)))

    # 写文件：先 special tokens，再普通字符
    with open(save_path, "w", encoding="utf-8") as f:
        for tok in SPECIAL_TOKENS:
            f.write(tok + "\n")
        for c in chars:
            f.write(c + "\n")

    total = len(SPECIAL_TOKENS) + len(chars)
    print(f"[build_vocab] wrote {total} tokens to {save_path}")
    print(f"[build_vocab]   - {len(SPECIAL_TOKENS)} special tokens: {SPECIAL_TOKENS}")
    print(f"[build_vocab]   - {len(chars)} chars: {''.join(chars)}")
    return total


# ---------------------------------------------------------------------------
# 线性脚本：导入即执行（**无** if __name__ 包装）
# 用户跑 `python data/build_vocab.py` 或 `python -m data.build_vocab` 都会执行到这里
# ---------------------------------------------------------------------------
build_vocab()

"""
预训练数据：合成"童话"语料。
模板 + 实体池 → 大量短文本，让模型学到字符级语言结构。

设计目标（详见 dataset_construction.md §2）：
  - 用 5 个固定模板 + 8 个实体池随机填充
  - 共生成 10,000 条短文本
  - 训练集 90% / 验证集 10%，按 id 的稳定哈希切分（保证可复现）
  - 切分成 block_size=128 的固定长块做 Causal LM 训练

提供：
  - TEMPLATES: 5 个模板字符串（verbatim 来自 §2.2）
  - ENTITIES: 8 个实体池（verbatim 来自 §2.3）
  - generate_pretrain_sample(rng): 解析占位符并随机替换
  - generate_pretrain_dataset(n_samples, seed=42) -> List[dict]，每条 {'text': str, 'id': int}
  - format_pretrain_for_lm(samples, block_size=128): 拼接 + tokenize + 切分固定长块
  - PretrainDataset: torch.utils.data.Dataset
  - make_pretrain_loader(split, batch_size, block_size, seed=42) -> DataLoader
"""

import hashlib
import random
from typing import List, Dict

# 延迟导入 torch：仅在构造 DataLoader / Dataset 时需要，
# 纯数据生成函数（generate_pretrain_dataset / format_pretrain_for_lm）不依赖 torch。

# 复用同包的 tokenizer（避免跨文件硬编码特殊 token id）
from data.tokenizer import CharTokenizer, BOS_ID, EOS_ID


# ---------------------------------------------------------------------------
# 模板与实体池（与 dataset_construction.md §2.2 / §2.3 完全一致，**禁止修改**）
# ---------------------------------------------------------------------------
TEMPLATES = [
    "Once upon a time, the {animal} went to the {place}. The {animal} saw a {object}. The {animal} said: \"{phrase}\".",
    "In the {place}, there lived a {adjective} {animal}. The {animal} liked to {action} every day.",
    "A {animal} and a {animal2} met in the {place}. They became friends and {action} together.",
    "The {adjective} {animal} found a {object} near the {place}. It was a {adjective2} day.",
    "One day, the {animal} decided to {action} at the {place}. What a {adjective} adventure!",
]

ENTITIES = {
    "animal":    ["cat", "dog", "fox", "bear", "rabbit", "mouse", "wolf", "deer", "owl", "frog"],
    "place":     ["forest", "castle", "river", "mountain", "cave", "garden", "village", "bridge", "lake", "hill"],
    "object":    ["tree", "rock", "flower", "mushroom", "leaf", "stone", "berry", "feather", "shell", "stick"],
    "action":    ["run", "jump", "sleep", "sing", "dance", "play", "explore", "rest", "watch", "listen"],
    "adjective": ["small", "big", "happy", "quiet", "brave", "clever", "gentle", "curious", "shy", "loud"],
    "adjective2":["sunny", "rainy", "cloudy", "windy", "foggy", "warm", "cool", "bright", "still", "peaceful"],
    "phrase":    ["hello", "goodbye", "thank you", "let's play", "be careful", "follow me", "stay close", "I'm tired"],
    # animal2 与 animal 同分布，方便"两个动物相遇"类模板
    "animal2":   ["cat", "dog", "fox", "bear", "rabbit", "mouse", "wolf", "deer", "owl", "frog"],
}


# ---------------------------------------------------------------------------
# 单样本生成
# ---------------------------------------------------------------------------
def generate_pretrain_sample(rng: random.Random) -> str:
    """
    随机选一个模板，并把里面所有 {key} 占位符按 ENTITIES[key] 随机替换。
    每次替换一个占位符后从 start+1 继续找，避免位置错乱。

    参数：
        rng: 显式传入的随机源（保证可复现，不污染全局 random）
    返回：
        一条合成童话文本
    """
    template = rng.choice(TEMPLATES) # 'Once upon a time, the {animal} went to the {place}. The {animal} saw a {object}. The {animal} said: "{phrase}".', 'In the {place}, there lived a {adjective} {animal}. The {animal} liked to {action} every day.', 'A {animal} and a {animal2} met in the {place}. They became friends and {action} together.', 'The {adjective} {animal} found a {object} near the {place}. It was a {adjective2} day.', 'One day, the {animal} decided to {action} at the {place}. What a {adjective} adventure!'
    sample = template
    # 循环替换：每轮处理最左边的一对 { }。
    # 用 index 查找 + 切片拼接，简单清晰；不引入正则表达式。
    while "{" in sample:
        start = sample.index("{")
        end = sample.index("}", start)
        key = sample[start + 1:end]
        # 防御：模板里出现未在 ENTITIES 里的 key 时，原样保留
        if key in ENTITIES:
            value = rng.choice(ENTITIES[key])
        else:
            value = "{" + key + "}"
        sample = sample[:start] + value + sample[end + 1:]
    return sample #'Once upon a time, the cat went to the cave. The bear saw a mushroom. The fox said: "goodbye".'


def generate_pretrain_dataset(n_samples: int = 10000, seed: int = 42) -> List[Dict]:
    """
    批量生成 n_samples 条 pretrain 文本。

    返回：List[dict]，每个 dict 含
        - text: str, 完整文本
        - id:   int, 样本编号（0..n_samples-1）
    """
    rng = random.Random(seed)
    dataset = []
    for i in range(n_samples):
        text = generate_pretrain_sample(rng) #'Once upon a time, the cat went to the cave. The bear saw a mushroom. The fox said: "goodbye".'
        dataset.append({"text": text, "id": i})
    return dataset


# ---------------------------------------------------------------------------
# 切分为 block_size 固定长块（Causal LM 训练样本）
# ---------------------------------------------------------------------------
def format_pretrain_for_lm(samples: List[Dict], block_size: int = 128) -> List[Dict]:
    """
    把一摞文本拼接成一个长序列，再用 block_size 切分为固定长块。
    每块的 input_ids 和 target_ids 都是 block_size 长，target = input 整体左移一位
    （即 target[t] = input[t+1]，最后一个位置的 target 设为 0，作为填充无影响）。

    注意：拼接前在每个文本之间插入一个换行作为边界，避免两段末尾字符意外相接。
    tokenizer 的 vocab 一定包含 '\\n'（来自 char 集）。

    参数：
        samples: 来自 generate_pretrain_dataset 的样本列表
        block_size: 每个训练样本的长度
    返回：
        List[dict]，每个 dict 含
            - input_ids:  List[int] 长度 = block_size
            - target_ids: List[int] 长度 = block_size
    """
    tokenizer = CharTokenizer(vocab_path=_vocab_path())

    # 拼接所有文本（每个之间加 '\\n' 隔开）
    big_text = "\n".join(s["text"] for s in samples) #拼接了8993条样本
    # 整段 tokenize
    ids = tokenizer.encode(big_text) #696476 整个大文本全部 tokenizer

    # 切成 block_size 整数倍，丢弃最后不足一块的尾巴
    n_blocks = len(ids) // block_size # 5441 128个token 作为一个 sample
    chunks = []
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size
        chunk = ids[start:end]
        # Causal LM 标准目标：target = input 向左错一位
        # target[t] = input[t+1]，最后一格用 input[-1] 填充（用 0 即可，因为 loss 会自然 mask）
        # 但更常见的做法是 target = input[:-1] 然后补一个 padding；为了简单 + 与 char-level 自洽，
        # 我们把 target[t] 设为 chunk[t+1]（如果存在），不存在则用 chunk[t]（最后一个 token 自监督）
        target = chunk[1:] + [chunk[-1]] # 答案就是 prompt 往后移一位，比如说 prompt是 1-100 那么答案就是2-101
        chunks.append({
            "input_ids": chunk,
            "target_ids": target,
        })
    return chunks


# ---------------------------------------------------------------------------
# 训练/验证划分（按 id 的稳定哈希）
# ---------------------------------------------------------------------------
def _stable_train_val_mask(sample_id: int, train_ratio: float = 0.9) -> bool:
    """
    用 id 的 SHA-256 哈希取前 4 字节当 uint32，再除以 2^32 得到 [0, 1) 的小数。
    < train_ratio 则归入训练集，否则验证集。
    保证：同一 id 在任何时候切分结果一致（不依赖 Python 字典顺序或全局 RNG）。
    """
    h = hashlib.sha256(str(sample_id).encode("utf-8")).digest()
    n = int.from_bytes(h[:4], "big")
    frac = n / 2 ** 32
    return frac < train_ratio


def _vocab_path():
    """内部辅助：定位 vocab.txt（与本文件同目录）。"""
    import os
    return os.path.join(os.path.dirname(__file__), "vocab.txt")


# ---------------------------------------------------------------------------
# PyTorch Dataset + DataLoader
# ---------------------------------------------------------------------------
class PretrainDataset:
    """预训练固定长块数据集。

    在 __init__ 时一次性把整份样本切成固定长块，缓存到内存。
    __getitem__ 直接返回 tensor 化的 (input_ids, target_ids)。

    注：实现为鸭子类型（不显式继承 torch Dataset），但与 torch DataLoader 协议完全兼容。
    延迟导入 torch，仅在 __getitem__ 实际被调用时才依赖它。
    """

    def __init__(self, samples: List[Dict], block_size: int = 128):
        # samples 是 generate_pretrain_dataset 的输出
        self.chunks = format_pretrain_for_lm(samples, block_size=block_size)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        import torch  # 延迟导入
        chunk = self.chunks[idx]
        x = torch.tensor(chunk["input_ids"], dtype=torch.long)
        y = torch.tensor(chunk["target_ids"], dtype=torch.long)
        return x, y


def _make_split(samples: List[Dict], split: str, train_ratio: float = 0.9) -> List[Dict]:
    """按 split 过滤样本：'train' 取前 train_ratio，'val' 取剩余。"""
    train_samples = [s for s in samples if _stable_train_val_mask(s["id"], train_ratio)]
    val_samples = [s for s in samples if not _stable_train_val_mask(s["id"], train_ratio)]
    if split == "train":
        return train_samples
    elif split in ("val", "valid", "validation"):
        return val_samples
    else:
        raise ValueError(f"unknown split: {split} (use 'train' or 'val')")


def make_pretrain_loader(
    split: str = "train",
    batch_size: int = 32,
    block_size: int = 128,
    seed: int = 42,
    n_samples: int = 10000,
):
    """
    一站式构造 DataLoader：
      1. generate_pretrain_dataset 生成 n_samples 条原始文本
      2. 按 id 稳定哈希 90/10 切分
      3. format_pretrain_for_lm 切成 block_size 固定长块
      4. 包成 DataLoader

    返回的 batch 是 (x, y) tensor 对，y = x 整体左移一位。
    """
    import torch  # 延迟导入
    from torch.utils.data import DataLoader
    raw = generate_pretrain_dataset(n_samples=n_samples, seed=seed)
    sub = _make_split(raw, split=split) #8993条数据
    dataset = PretrainDataset(sub, block_size=block_size)
    shuffle = (split == "train")
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        generator=g,
    )
    return loader

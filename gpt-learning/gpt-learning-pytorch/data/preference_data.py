"""
偏好数据：合成 (prompt, chosen, rejected) 三元组。
让 RM（奖励模型）学到"哪个回答好、哪个不好"的判别能力。

设计目标（详见 dataset_construction.md §4）：
  - chosen  = 复用 SFT 的 response
  - rejected = 5 种策略之一（带概率分布）：
      * wrong_entity   30%  换掉 response 里的主体动物名
      * unrelated      30%  随机抽一个**其他** SFT 的 response
      * noisy_keywords 20%  保留关键词但打乱顺序
      * too_short      10%  只保留 1-2 个词
      * too_long       10%  重复 5 次首句
  - 拒绝**简单扰动**（如只换 1-2 个词），用**结构性差异**让 RM 学更深

提供：
  - build_rejected(prompt, chosen_response, rng, other_responses) -> str
  - generate_preference_dataset(n_samples, seed=42) -> List[dict]
  - PreferenceDataset + make_preference_loader

约束：
  - 复用 SFT 的 prompt 字段（保证 SFT 和 Preference 同一 id 的 prompt 完全一致）
  - 数据加载返回 (chosen_input_ids, rejected_input_ids)，**不**做 padding；
    后续 train_reward.py 用自定义 collate 处理
"""

import hashlib
import random
from typing import List, Dict

# 延迟导入 torch：仅在 DataLoader / Dataset 实际构造时使用
from data.tokenizer import CharTokenizer
from data.sft_data import generate_sft_dataset
from data.pretrain_data import ENTITIES


# ---------------------------------------------------------------------------
# 5 种 rejected 构造策略 + 概率分布
# ---------------------------------------------------------------------------
# (strategy_name, weight)
REJECTED_STRATEGIES = [
    ("wrong_entity",   0.30),
    ("unrelated",      0.30),
    ("noisy_keywords", 0.20),
    ("too_short",      0.10),
    ("too_long",       0.10),
]
_STRATEGY_NAMES = [s for s, _ in REJECTED_STRATEGIES]
_STRATEGY_WEIGHTS = [w for _, w in REJECTED_STRATEGIES]


def _pick_animal_from_prompt(prompt: str) -> str:
    """
    从 prompt 里找动物名（与 SFT 模板一致，动物是 prompt 文本中某 ENTITIES['animal'] 的元素）。
    找不到时返回 ''，调用方会 fallback。
    """
    for animal in ENTITIES["animal"]:
        # 用单词边界匹配，避免 'cat' 命中 'scatter' 之类
        if f" {animal}" in f" {prompt}" or f" {animal}." in prompt or f" {animal}?" in prompt:
            return animal
    return ""


def _pick_strategy(rng: random.Random) -> str:
    """按权重随机抽一种策略。"""
    return rng.choices(_STRATEGY_NAMES, weights=_STRATEGY_WEIGHTS, k=1)[0]


def _build_wrong_entity(chosen_response: str, prompt: str, rng: random.Random) -> str:
    """
    策略 1（30%）：把 chosen_response 里的动物名换成另一个动物。
    如果 chosen_response 里没出现任何动物名，则 fallback：随机加一个动物前缀。
    """
    # 先尝试找 chosen_response 里出现的动物（用空格包夹的方式）
    animals_in_response = [a for a in ENTITIES["animal"] if f" {a}" in f" {chosen_response}" or f" {a}." in chosen_response]
    prompt_animal = _pick_animal_from_prompt(prompt)
    if animals_in_response:
        original = rng.choice(animals_in_response)
        # 换掉
        others = [a for a in ENTITIES["animal"] if a != original]
        replacement = rng.choice(others)
        # 替换时小心大小写
        new = chosen_response.replace(original, replacement)
        if new != chosen_response:
            return new
    # fallback：若 response 没有可替换的动物，用 prompt 的动物名换掉
    if prompt_animal:
        others = [a for a in ENTITIES["animal"] if a != prompt_animal]
        replacement = rng.choice(others)
        return f"A {replacement} is a small creature that likes to play."
    # 终极 fallback
    return "A dog is a small creature that likes to play."


def _build_unrelated(chosen_response: str, rng: random.Random, other_responses: List[str]) -> str:
    """
    策略 2（30%）：从所有**其他** SFT response 里随机抽一条当 rejected。
    若 other_responses 为空（边界情况），用 chosen_response 自身打乱来 fallback。
    """
    if other_responses:
        return rng.choice(other_responses)
    # fallback：打乱自身
    words = chosen_response.split()
    rng.shuffle(words)
    return " ".join(words)


def _build_noisy_keywords(chosen_response: str, rng: random.Random) -> str:
    """
    策略 3（20%）：保留关键词但打乱词序。
    """
    words = chosen_response.split()
    if len(words) <= 1:
        return chosen_response
    rng.shuffle(words)
    return " ".join(words)


def _build_too_short(chosen_response: str) -> str:
    """
    策略 4（10%）：只保留第 1 个词。
    极端短答：只一个词。
    """
    words = chosen_response.split()
    if words:
        return words[0]
    return "something"


def _build_too_long(chosen_response: str) -> str:
    """
    策略 5（10%）：重复 chosen 5 次组成超长无意义文本。
    """
    return " ".join([chosen_response] * 5)


# ---------------------------------------------------------------------------
# 对外暴露：构造单条 rejected
# ---------------------------------------------------------------------------
def build_rejected(
    prompt: str,
    chosen_response: str,
    rng: random.Random,
    other_responses: List[str] = None,
) -> str:
    """
    按权重抽一种策略，构造 rejected 文本（兼容旧接口：只返回文本）。

    参数：
        prompt:           SFT prompt（用于 wrong_entity 策略中查找动物名）
        chosen_response:  对应的 chosen response
        rng:              显式传入的随机源
        other_responses:  "unrelated" 策略候选池（其他 SFT response）
    返回：
        rejected 文本
    """
    text, _ = _build_rejected_with_strategy(prompt, chosen_response, rng, other_responses)
    return text


def _build_rejected_with_strategy(
    prompt: str,
    chosen_response: str,
    rng: random.Random,
    other_responses: List[str] = None,
):
    """
    与 build_rejected 一样，但同时返回 (rejected_text, strategy_name)。
    内部使用，generate_preference_dataset 借此记录 rejected_strategy 字段。
    """
    if other_responses is None:
        other_responses = []
    strategy = _pick_strategy(rng)
    if strategy == "wrong_entity":
        text = _build_wrong_entity(chosen_response, prompt, rng)
    elif strategy == "unrelated":
        text = _build_unrelated(chosen_response, rng, other_responses)
    elif strategy == "noisy_keywords":
        text = _build_noisy_keywords(chosen_response, rng)
    elif strategy == "too_short":
        text = _build_too_short(chosen_response)
    elif strategy == "too_long":
        text = _build_too_long(chosen_response)
    else:
        # 不可达
        text = chosen_response
        strategy = "unknown"
    return text, strategy


# ---------------------------------------------------------------------------
# 批量生成
# ---------------------------------------------------------------------------
def generate_preference_dataset(n_samples: int = 5000, seed: int = 42) -> List[Dict]:
    """
    批量生成 n_samples 条 preference 三元组。

    关键：直接复用 SFT 数据集，保证同 id 的 prompt 与 SFT 完全一致。
    """
    rng = random.Random(seed)
    sft = generate_sft_dataset(n_samples=n_samples, seed=seed)#5000 个 样本

    # 准备 unrelated 策略用的候选池：所有 SFT response（除了当前这条）
    all_responses = [s["response"] for s in sft] # 5000个 回复

    dataset = []
    for i, s in enumerate(sft):
        prompt = s["prompt"]
        chosen = s["response"] #正例
        other_responses = [r for j, r in enumerate(all_responses) if j != i] #准备好负例
        # 复用同一个 rng 以保证整体可复现
        rejected, strategy = _build_rejected_with_strategy( #使用不同的策略去生成负例
            prompt, chosen, rng, other_responses=other_responses
        )
        dataset.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "rejected_strategy": strategy,
            "id": s["id"],
        })
    return dataset


# ---------------------------------------------------------------------------
# PyTorch Dataset + DataLoader
# ---------------------------------------------------------------------------
def _vocab_path() -> str:
    import os
    return os.path.join(os.path.dirname(__file__), "vocab.txt")


def _stable_train_val_mask(sample_id: int, train_ratio: float = 0.9) -> bool:
    h = hashlib.sha256(str(sample_id).encode("utf-8")).digest()
    n = int.from_bytes(h[:4], "big")
    return (n / 2 ** 32) < train_ratio
"""

代码	一句话
_stable_train_val_mask	"用 hash 把每个 id 永久 决定该进哪一桶"
make_preference_loader	"把生成、切分、tokenize、打乱、batch 打包成一个函数"

"""


def _make_full_text(prompt: str, response: str) -> str:
    """Reward Model 输入格式：与 SFT 训练一致即可（<bos> prompt <sep> response <eos>）。"""
    return f"<bos> {prompt} <sep> {response} <eos>"


class PreferenceDataset:
    """
    偏好数据集：返回 {chosen_input_ids, rejected_input_ids}（均不 padding）。
    后续 train_reward.py 用自定义 collate 处理变长序列。

    鸭子类型实现：__getitem__ 才延迟导入 torch。
    """

    def __init__(self, samples: List[Dict]):
        self.tokenizer = CharTokenizer(vocab_path=_vocab_path())
        self._cached: List[Dict] = []
        for s in samples:
            chosen_text = _make_full_text(s["prompt"], s["chosen"])
            rejected_text = _make_full_text(s["prompt"], s["rejected"])
            self._cached.append({
                "chosen_input_ids":   self.tokenizer.encode(chosen_text),
                "rejected_input_ids": self.tokenizer.encode(rejected_text),
            })

    def __len__(self):
        return len(self._cached)

    def __getitem__(self, idx):
        import torch  # 延迟导入
        item = self._cached[idx]
        return {
            "chosen_input_ids":   torch.tensor(item["chosen_input_ids"],   dtype=torch.long),
            "rejected_input_ids": torch.tensor(item["rejected_input_ids"], dtype=torch.long),
        }


def _make_split(samples: List[Dict], split: str) -> List[Dict]:
    train_samples = [s for s in samples if _stable_train_val_mask(s["id"])]
    val_samples = [s for s in samples if not _stable_train_val_mask(s["id"])]
    if split == "train":
        return train_samples
    if split in ("val", "valid", "validation"):
        return val_samples
    raise ValueError(f"unknown split: {split}")


def preference_collate_fn(batch: List[Dict]):
    """
    把不等长序列 pad 到本 batch 最大长度，padding id = 0（<unk>）。
    - chosen_input_ids / rejected_input_ids 都 pad
    - 暂不返回 mask；train_reward.py 可以用 (input_ids != 0) 自行构造 mask
    """
    import torch  # 延迟导入
    max_len = max(
        max(len(b["chosen_input_ids"]), len(b["rejected_input_ids"]))
        for b in batch
    )
    chosen_list = []
    rejected_list = []
    for b in batch:
        n_c = len(b["chosen_input_ids"])
        n_r = len(b["rejected_input_ids"])
        chosen_list.append(
            torch.cat([b["chosen_input_ids"], torch.zeros(max_len - n_c, dtype=torch.long)])
        )
        rejected_list.append(
            torch.cat([b["rejected_input_ids"], torch.zeros(max_len - n_r, dtype=torch.long)])
        )
    return {
        "chosen_input_ids":   torch.stack(chosen_list,   dim=0),
        "rejected_input_ids": torch.stack(rejected_list, dim=0),
    }


def make_preference_loader(
    split: str = "train",
    batch_size: int = 32,
    seed: int = 42,
    n_samples: int = 5000,
):
    """
    一站式构造 Preference DataLoader。
    返回 batch 形如 {chosen_input_ids: (B, L), rejected_input_ids: (B, L)}。
    """
    import torch  # 延迟导入
    from torch.utils.data import DataLoader
    raw = generate_preference_dataset(n_samples=n_samples, seed=seed) # 5000个 用于训练奖励模型的数据集
    sub = _make_split(raw, split=split)#要么训练集，要么测试集
    dataset = PreferenceDataset(sub)
    shuffle = (split == "train")
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=preference_collate_fn,
        generator=g,
    )

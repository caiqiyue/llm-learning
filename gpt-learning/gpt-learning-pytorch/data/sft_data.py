"""
SFT 数据：合成 (prompt, response) 对。
让模型学会"以 prompt 为条件，生成 response"。

设计目标（详见 dataset_construction.md §3）：
  - 10 个 prompt 模板（覆盖"What is / Where does / Describe / How does ..."等指令风格）
  - 3 类 response 模板（事实 / 动作 / 描述），按 prompt 类型映射
  - 拼接格式：\"<bos> {prompt} <sep> {response} <eos>\"
  - 训练时 prompt 部分 label 设为 -100（prompt-masked loss）
  - prompt_len 字段 = tokenize(\"<bos>{prompt}<sep>\") 的长度
  - 同时为 PPO 阶段提供 get_ppo_prompts（不需要 response）

提供：
  - SFT_PROMPT_TEMPLATES / FACTUAL_TEMPLATES / ACTION_TEMPLATES / DESCRIPTIVE_TEMPLATES
  - pick_response_templates(prompt_template): 把 prompt 模板映射到 response 模板列表
  - generate_sft_dataset(n_samples, seed=42) -> List[dict]
  - format_sft_sample(prompt, response): 拼成完整训练字符串
  - SFTDataset + make_sft_loader
  - get_ppo_prompts(n=1000, seed=42) -> List[dict]，每条 {'prompt', 'id'}
"""

import hashlib
import random
import re
from typing import List, Dict

# 延迟导入 torch：仅在 DataLoader / Dataset 实际构造时使用
from data.tokenizer import CharTokenizer


# ---------------------------------------------------------------------------
# 模板（与 dataset_construction.md §3.2 / §3.3 完全一致，**禁止修改**）
# ---------------------------------------------------------------------------
SFT_PROMPT_TEMPLATES = [
    "What is a {animal}?",
    "Tell me about a {animal}.",
    "Where does a {animal} live?",
    "What does a {animal} like to do?",
    "Describe a {adjective} {animal}.",
    "What is in the {place}?",
    "Tell me a story about a {animal}.",
    "How does a {animal} {action}?",
    "Why is a {animal} {adjective}?",
    "What is the {object} used for?",
]

# 事实型：对应 "What is a {animal}?" / "Tell me about a {animal}."
FACTUAL_TEMPLATES = [
    "A {animal} is a {adjective} creature that lives in the {place}.",
    "A {animal} is a small animal that likes to {action}.",
    "The {animal} is {adjective} and lives in the {place}.",
]

# 动作型：对应 "What does a {animal} like to do?" / "How does a {animal} {action}?"
ACTION_TEMPLATES = [
    "A {animal} likes to {action} in the {place}.",
    "The {animal} often {action}s when it is {adjective}.",
    "{animal}s enjoy {action}ing in the {place}.",
]

# 描述型：对应 "Describe a {adjective} {animal}." / "Tell me a story about a {animal}."
DESCRIPTIVE_TEMPLATES = [
    "The {adjective} {animal} is a {adjective2} creature that likes to {action}.",
    "It is {adjective} and {adjective2}, often seen in the {place}.",
]

# 复用 pretrain_data 的实体池，保持数据分布一致
from data.pretrain_data import ENTITIES


# ---------------------------------------------------------------------------
# prompt 模板 → response 模板的映射
# ---------------------------------------------------------------------------
def pick_response_templates(prompt_template: str) -> List[str]:
    """
    根据 prompt 模板的关键词，返回匹配的 response 模板列表。

    匹配规则（按优先级）：
      - 包含 "like to do" 或 "How does"  → ACTION
      - 包含 "Describe" 或 "story"        → DESCRIPTIVE
      - 其他                              → FACTUAL
    """
    if "like to do" in prompt_template or "How does" in prompt_template:
        return ACTION_TEMPLATES
    if "Describe" in prompt_template or "story" in prompt_template:
        return DESCRIPTIVE_TEMPLATES
    return FACTUAL_TEMPLATES


# ---------------------------------------------------------------------------
# 占位符替换（与 pretrain_data 同源，但 SFT 的 response 模板可能含 "adjective2" 等）
# ---------------------------------------------------------------------------
def _fill_placeholders(template: str, rng: random.Random) -> str:
    """
    解析 {key} 占位符并替换。复用与 pretrain_data 相同的逻辑。
    未在 ENTITIES 里的 key 原样保留（防御性）。
    """
    sample = template
    while "{" in sample:
        start = sample.index("{")
        end = sample.index("}", start)
        key = sample[start + 1:end]
        if key in ENTITIES:
            value = rng.choice(ENTITIES[key])
        else:
            value = "{" + key + "}"
        sample = sample[:start] + value + sample[end + 1:]
    return sample


# ---------------------------------------------------------------------------
# 单条 SFT 样本生成
# ---------------------------------------------------------------------------
def _generate_sft_sample(rng: random.Random, idx: int) -> Dict:
    """
    生成一条 SFT 样本：随机挑 prompt 模板 → 填充 → 挑匹配的 response 模板 → 填充。
    返回 dict 含 {prompt, response, prompt_template, response_template, id}。
    """
    prompt_template = rng.choice(SFT_PROMPT_TEMPLATES) #'Tell me about a {animal}.'
    prompt = _fill_placeholders(prompt_template, rng) #'Tell me about a cat.'

    response_templates = pick_response_templates(prompt_template)
    response_template = rng.choice(response_templates) #'The {animal} is {adjective} and lives in the {place}.'
    response = _fill_placeholders(response_template, rng) # 'The rabbit is quiet and lives in the mountain.'

    return {
        "prompt": prompt,
        "response": response,
        "prompt_template": prompt_template,
        "response_template": response_template,
        "id": idx,
    }


def generate_sft_dataset(n_samples: int = 5000, seed: int = 42) -> List[Dict]:
    """批量生成 n_samples 条 SFT 样本。"""
    rng = random.Random(seed)
    return [_generate_sft_sample(rng, i) for i in range(n_samples)]


# ---------------------------------------------------------------------------
# 拼接格式
# ---------------------------------------------------------------------------
def format_sft_sample(prompt: str, response: str) -> str:
    """
    拼成训练用完整序列：\"<bos> {prompt} <sep> {response} <eos>\"。
    注意特殊 token 用字面量写入，tokenize 时由 tokenizer 映射到对应 id。
    """
    return f"<bos> {prompt} <sep> {response} <eos>"


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------
def _vocab_path() -> str:
    import os
    return os.path.join(os.path.dirname(__file__), "vocab.txt")


def _stable_train_val_mask(sample_id: int, train_ratio: float = 0.9) -> bool:
    """同 pretrain_data：用 id 的稳定哈希做 90/10 切分。"""
    h = hashlib.sha256(str(sample_id).encode("utf-8")).digest()
    n = int.from_bytes(h[:4], "big")
    return (n / 2 ** 32) < train_ratio


# ---------------------------------------------------------------------------
# PyTorch Dataset + DataLoader
# ---------------------------------------------------------------------------
class SFTDataset:
    """
    SFT 数据集：返回 {input_ids, labels, prompt_len}。
    - input_ids: tokenize(完整序列)
    - labels: 与 input_ids 同长；prompt 部分（含 <sep>）设为 -100，只在 response + <eos> 上算 loss
    - prompt_len: tokenize("<bos>{prompt}<sep>") 的长度，便于 train_sft.py 直接定位 mask 起点

    实现为鸭子类型（与 torch DataLoader 协议兼容），__getitem__ 才延迟 import torch。
    """

    def __init__(self, samples: List[Dict]):
        self.tokenizer = CharTokenizer(vocab_path=_vocab_path())
        # 预计算并缓存：每条样本的 (input_ids, labels, prompt_len)，这一部分长度没有统一
        self._cached: List[Dict] = []
        for s in samples:
            full_text = format_sft_sample(s["prompt"], s["response"]) #'<bos> Tell me about a cat. <sep> The rabbit is quiet and lives in the mountain. <eos>'
            prompt_part = f"<bos> {s['prompt']} <sep>"  #'<bos> Tell me about a cat. <sep>'

            input_ids = self.tokenizer.encode(full_text) # len=85
            prompt_len = len(self.tokenizer.encode(prompt_part)) # prompt 32
            # 【feat-027】labels 与 input_ids **同下标**：labels[i] = input_ids[i]，
            #   只是 prompt 部分（含 <sep>）置 -100。
            #   错位（"logits[t] 预测 token t+1"）由**模型内部**完成：
            #   model/gpt2.py 里是 F.cross_entropy(logits[:, :-1], targets[:, 1:])。
            #   ★ 之前模型的 loss **不左移**，而这里 labels 又与 input 同下标
            #     ⇒ SFT 教模型"输出**当前** token"（copy）
            #     ⇒ 推理时 prompt 以空格结尾，模型把最后一个 token 复制出来
            #     ⇒ 无限吐空格（实测 48 个 token 全是空格 id=5）。
            #   现在错位只在模型里做一次 ⇒ 预训练线与 SFT 线口径必然一致。
            labels = [-100] * prompt_len + input_ids[prompt_len:] # len=85，前 32 个位置是 -100（代表 prompt）
            self._cached.append({
                "input_ids": input_ids,
                "labels": labels,
                "prompt_len": prompt_len,
            })

    def __len__(self):
        return len(self._cached)

    def __getitem__(self, idx):
        import torch  # 延迟导入
        item = self._cached[idx]
        return {
            "input_ids":  torch.tensor(item["input_ids"],   dtype=torch.long),
            "labels":     torch.tensor(item["labels"],      dtype=torch.long),
            "prompt_len": torch.tensor(item["prompt_len"],  dtype=torch.long),
        }


def _make_split(samples: List[Dict], split: str) -> List[Dict]:
    train_samples = [s for s in samples if _stable_train_val_mask(s["id"])]
    val_samples = [s for s in samples if not _stable_train_val_mask(s["id"])]
    if split == "train":
        return train_samples
    if split in ("val", "valid", "validation"):
        return val_samples
    raise ValueError(f"unknown split: {split}")


def sft_collate_fn(batch: List[Dict]):
    """
    自定义 collate：把不等长序列 pad 到本 batch 的最大长度。
    padding token = 0（<unk>，与 tokenizer.PAD_ID=1 不同；这里用 0 是为了在 char-level 下不引入特殊语义）。
    labels 中 padding 位置设为 -100，避免对 padding 算 loss。
    """
    import torch  # 延迟导入
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids_list = []
    labels_list = []
    prompt_len_list = []
    for b in batch:
        n = len(b["input_ids"])
        pad = max_len - n
        # 用 0 作为 padding id（vocab 中 0 = <unk>，不会与正常 token 混淆）
        x = torch.cat([b["input_ids"], torch.zeros(pad, dtype=torch.long)])
        y = torch.cat([b["labels"],    torch.full((pad,), -100, dtype=torch.long)])
        input_ids_list.append(x)
        labels_list.append(y)
        prompt_len_list.append(b["prompt_len"])
    return {
        "input_ids":  torch.stack(input_ids_list,  dim=0),
        "labels":     torch.stack(labels_list,     dim=0),
        "prompt_len": torch.stack(prompt_len_list, dim=0),
    }


def make_sft_loader(
    split: str = "train",
    batch_size: int = 32,
    seed: int = 42,
    n_samples: int = 5000,
):
    """
    一站式构造 SFT DataLoader。
    返回 batch 形如 {input_ids: (B, L), labels: (B, L), prompt_len: (B,)}。
    """
    import torch  # 延迟导入
    from torch.utils.data import DataLoader
    raw = generate_sft_dataset(n_samples=n_samples, seed=seed)
    sub = _make_split(raw, split=split) #4518条
    dataset = SFTDataset(sub)
    shuffle = (split == "train")
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=sft_collate_fn, #把不等长的  sample 多扔少补 
        generator=g,
    )


# ---------------------------------------------------------------------------
# PPO prompt 池（dataset_construction.md §5）
# ---------------------------------------------------------------------------
def get_ppo_prompts(n: int = 1000, seed: int = 42) -> List[Dict]:
    """
    从 SFT 数据集前 n 条抽 prompt 作为 PPO 训练用的 prompt 池。
    返回 [{'prompt': str, 'id': int}, ...]。

    关键：PPO 阶段 response 由 policy 实时生成，所以这里**只**要 prompt。
    """
    sft = generate_sft_dataset(n_samples=n, seed=seed)
    return [{"prompt": s["prompt"], "id": s["id"]} for s in sft]

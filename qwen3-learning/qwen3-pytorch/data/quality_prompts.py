"""
生成质量的**探针集**：两条线并排评估时用的**同一批问题**。

【为什么需要它】
    GPT-2 线的 `evaluate.py` 用 `get_ppo_prompts`（SFT 模板 + 实体池）当生成质量
    指标的输入；Qwen3 线要实现同名同定义的指标（distinct-n / self-BLEU /
    token entropy / response quality），就也需要**同一批问题** ——
    否则两个数字量的不是一件事，没法并排。

    探针集必须满足两条：
      1. **与领域无关**：问的是「猫/森林/石头」这类中性问题，
         不是「注意力机制是什么」这类只对某一条线在域内的问题；
      2. **可复现**：同样的 (n_samples, seed) 必然给出同一批问题。

【★ 与 GPT-2 线的关系：对齐物 + 自动守卫】
    实体池与 GPT-2 线 `data/pretrain_data.py` 的 `ENTITIES` **逐字相同**。
    原因：response quality 的规则里有一条「回答里出现了某个动物名」，
    动物表不一样 ⇒ 两个 `response_quality` 根本不是同一个指标。

    ★ 本文件**不 import** GPT-2 线（AGENTS.md 硬约束：Qwen3 线不跨目录 import）。
      「两边还对不对得上」由 `ablations/compare_quality.py` 里的断言自动检查：
      哪天有人改了 GPT-2 的实体池而忘了改这里，对照脚本会**直接报错退出**。
      （这比「复制一份就没人管了」可靠 —— 复制品会腐烂，断言不会。）

【这个探针集在两条线里的位置】
    GPT-2 线：动物问题 = 它的 SFT 训练分布 ⇒ **域内**
    Qwen3 线：动物问题 = 它没见过的分布 ⇒ **域外**
    ⇒ 并排表里必须标出这一点，否则会被误读成「架构谁更好」。
    （Qwen3 的域内探针见 `get_corpus_prompts`，只用于 distinct / self-BLEU 这类
      **与领域无关**的指标 —— 动物规则对它不适用。）
"""

import random
from typing import Dict, List

# ---------------------------------------------------------------------------
# 实体池：与 GPT-2 线 data/pretrain_data.py 的 ENTITIES 逐字相同
# （由 ablations/compare_quality.py 的断言保证不会漂移）
# ---------------------------------------------------------------------------
QUALITY_ENTITIES: Dict[str, List[str]] = {
    "animal": ["cat", "dog", "fox", "bear", "rabbit", "mouse", "wolf", "deer", "owl", "frog"],
    "place": ["forest", "castle", "river", "mountain", "cave", "garden", "village", "bridge", "lake", "hill"],
    "object": ["tree", "rock", "flower", "mushroom", "leaf", "stone", "berry", "feather", "shell", "stick"],
    "action": ["run", "jump", "sleep", "sing", "dance", "play", "explore", "rest", "watch", "listen"],
    "adjective": ["small", "big", "happy", "quiet", "brave", "clever", "gentle", "curious", "shy", "loud"],
    "adjective2": ["sunny", "rainy", "cloudy", "windy", "foggy", "warm", "cool", "bright", "still", "peaceful"],
    "phrase": ["hello", "goodbye", "thank you", "let's play", "be careful", "follow me", "stay close", "I'm tired"],
    # animal2 与 animal 同分布，方便「两个动物相遇」类模板
    "animal2": ["cat", "dog", "fox", "bear", "rabbit", "mouse", "wolf", "deer", "owl", "frog"],
}

# ---------------------------------------------------------------------------
# 探针模板：与 GPT-2 线 data/sft_data.py 的 SFT_PROMPT_TEMPLATES 同构
# （同一批问题类型 ⇒ 两条线面对的是「难度相当」的输入）
# ---------------------------------------------------------------------------
QUALITY_PROMPT_TEMPLATES = [
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


def _fill_placeholders(template: str, rng: random.Random) -> str:
    """
    把模板里最左边的一对 `{key}` 替换掉，循环到没有占位符为止。

    参数：
        template: 模板字符串
        rng:      随机源（显式传入 ⇒ 可复现，不污染全局 random）
    返回：
        填好占位符的字符串
    """
    sample = template
    while "{" in sample:
        start = sample.index("{")
        end = sample.index("}", start)
        key = sample[start + 1:end]
        # 防御：模板里出现未登记的 key 时，原样保留
        value = rng.choice(QUALITY_ENTITIES[key]) if key in QUALITY_ENTITIES else "{" + key + "}"
        sample = sample[:start] + value + sample[end + 1:]
    return sample


def get_quality_prompts(n_samples: int = 30, seed: int = 1234) -> List[str]:
    """
    取 n_samples 条**动物探针**（两条线共用的那批问题）。

    参数：
        n_samples: 要多少条
        seed:      随机种子（可复现）
    返回：
        问题字符串列表
    """
    rng = random.Random(seed)
    prompts: List[str] = []
    for _ in range(n_samples):
        template = rng.choice(QUALITY_PROMPT_TEMPLATES)
        prompts.append(_fill_placeholders(template, rng))
    return prompts


def get_corpus_prompts(n_samples: int = 30, seed: int = 1234) -> List[str]:
    """
    取 n_samples 条**Qwen3 语料域内**的探针（陈述句，不是问题）。

    【用途与限制】
        用来回答「Qwen3 在自己的分布上到底行不行」——
        动物探针是 Qwen3 的**域外**输入，在它上面的分数低是**预期的**，
        不能当成「Qwen3 生成能力差」的证据。

        ⚠️ 这里**只适用与领域无关的指标**（distinct-n / self-BLEU / token entropy）。
           `response_quality` 的「含动物名」规则对语料域内的回答不适用
           （Qwen3 的回答里本来就该讲 transformer，不该讲猫），
           所以域内探针**不出** response_quality —— 不硬凑一个没意义的数字。

    参数：
        n_samples: 要多少条
        seed:      随机种子（可复现）
    返回：
        陈述句列表
    """
    from data.corpus import OBJECTS, SUBJECTS, VERBS

    rng = random.Random(seed)
    prompts: List[str] = []
    for _ in range(n_samples):
        subject = rng.choice(SUBJECTS)
        verb = rng.choice(VERBS)
        obj = rng.choice(OBJECTS)
        # 英文主体用空格连接，中文主体直接拼接（与 corpus.generate_sentences 一致）
        if subject.isascii():
            prompts.append(f"{subject} {verb} {obj}.")
        else:
            prompts.append(f"{subject}{verb}{obj}。")
    return prompts

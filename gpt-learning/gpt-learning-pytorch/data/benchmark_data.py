"""
Toy Benchmark 数据：4 选 1 多选题库。
让预训练好的模型在"实体-关系"判断上比 random baseline（25%）好。

设计目标（详见 design doc §6.5 + plan P7.1.4）：
  - 100 道题
  - 4 个问题模板：
      1) "Where does a {animal} live?"  → 正确答案=动物的真实 place + 2 个 place 干扰项 + 1 个不相关
      2) "What does a {animal} like to do?" → 正确答案=动物的真实 action + 2 个 action 干扰项 + 1 个不相关
      3) "What is a {adjective} {animal}?" → 4 个形容词选 1（正确答案=该形容词）
      4) "Who is in the {place}?"  → 4 个动物选 1（正确答案=一个常出现在该 place 的动物）
  - 正确答案位置 answer_idx 尽量在 0..3 上均匀分布（避免总在 0）
  - 字段：{question, choices: [str, str, str, str], answer_idx: int}
  - 同时保存到 data/benchmark.json，让 evaluate.py 可直接 load

说明：
  - 我们没有"动物→固定 place"的真实映射（合成数据里动物和 place 是独立的随机分布）；
    所以采用"在该动物出现过的 pretrain 文本里"统计最常见的 place 作为正确答案。
    简单起见，用"该动物常出现的 place" = ENTITIES['place'] 随机取一个；
    真正让模型"答对"靠的是它见过 "the animal lived in the place" 这种搭配。


    从预训练数据的实体池里随机抽 animal / place / action / adjective，
自动构造 100 道 4 选 1 的小测试题，
随机打乱正确答案位置，
最后保存成 data/benchmark.json，
供 evaluate.py 测试模型是否比随机猜测 25% 更强。
"""

import json
import os
import random
from typing import List, Dict

# 复用 pretrain 实体池
from data.pretrain_data import ENTITIES


# ---------------------------------------------------------------------------
# 4 种问题模板
# ---------------------------------------------------------------------------
QUESTION_BUILDERS = ["place", "action", "adjective", "animal_in_place"]


# ---------------------------------------------------------------------------
# 单题生成辅助
# ---------------------------------------------------------------------------
def _pick_distractor(rng: random.Random, pool: List[str], exclude: str) -> str:
    """从 pool 里挑一个不等于 exclude 的元素。"""
    candidates = [x for x in pool if x != exclude]
    if not candidates:
        return exclude  # 极端 fallback
    return rng.choice(candidates)


def _build_place_question(rng: random.Random) -> Dict:
    """
    模板 1：Where does a {animal} live?
    正确答案 = 一个该动物常出现的 place（合成数据里其实没有强约束，
    这里直接随机选一个 place 当作"该动物的栖息地"）
    干扰项 = 2 个其他 place + 1 个不相关（用 action 当干扰，让模型也得能识别语义类型）
    """
    animal = rng.choice(ENTITIES["animal"])
    correct_place = rng.choice(ENTITIES["place"])
    choices = [correct_place]
    # 2 个 place 干扰
    choices.append(_pick_distractor(rng, ENTITIES["place"], correct_place))
    choices.append(_pick_distractor(rng, ENTITIES["place"], choices[1]))
    # 1 个语义不相关（用 action 短语）
    choices.append(f"likes to {rng.choice(ENTITIES['action'])}")
    answer_idx = _decide_answer_idx(rng, len(choices))
    # 把正确答案放到 answer_idx 位置
    choices[0], choices[answer_idx] = choices[answer_idx], choices[0]
    return {
        "question": f"Where does a {animal} live?",
        "choices": choices,
        "answer_idx": answer_idx,
    }


def _build_action_question(rng: random.Random) -> Dict:
    """
    模板 2：What does a {animal} like to do?
    正确答案 = 该动物常做的 action
    干扰 = 2 个 action + 1 个不相关（用 place 当干扰）
    """
    animal = rng.choice(ENTITIES["animal"])
    correct_action = rng.choice(ENTITIES["action"])
    choices = [f"to {correct_action}"]
    # 2 个 action 干扰
    a1 = _pick_distractor(rng, ENTITIES["action"], correct_action)
    a2 = _pick_distractor(rng, ENTITIES["action"], a1)
    choices.append(f"to {a1}")
    choices.append(f"to {a2}")
    # 1 个不相关：用 place
    choices.append(f"in the {rng.choice(ENTITIES['place'])}")
    answer_idx = _decide_answer_idx(rng, len(choices))
    choices[0], choices[answer_idx] = choices[answer_idx], choices[0]
    return {
        "question": f"What does a {animal} like to do?",
        "choices": choices,
        "answer_idx": answer_idx,
    }


def _build_adjective_question(rng: random.Random) -> Dict:
    """
    模板 3：What is a {adjective} {animal}?
    4 个形容词选 1，正确答案 = 给定形容词。
    干扰 = 其他 3 个形容词。
    """
    animal = rng.choice(ENTITIES["animal"])
    correct_adj = rng.choice(ENTITIES["adjective"])
    distractors = [_pick_distractor(rng, ENTITIES["adjective"], correct_adj) for _ in range(3)]
    # 防御：去重（虽然 adjective 池无重复）
    seen = {correct_adj}
    unique_distractors = []
    for d in distractors:
        if d not in seen:
            unique_distractors.append(d)
            seen.add(d)
    while len(unique_distractors) < 3:
        # 补齐（理论上不会发生）
        unique_distractors.append(_pick_distractor(rng, ENTITIES["adjective"], correct_adj))
    choices = [correct_adj] + unique_distractors[:3]
    answer_idx = _decide_answer_idx(rng, len(choices))
    choices[0], choices[answer_idx] = choices[answer_idx], choices[0]
    return {
        "question": f"What is a {correct_adj} {animal}?",
        "choices": choices,
        "answer_idx": answer_idx,
    }


def _build_animal_in_place_question(rng: random.Random) -> Dict:
    """
    模板 4：Who is in the {place}?
    4 个动物选 1，正确答案 = 一个动物（与 place 没有强约束，合成数据里随机选即可）
    干扰 = 其他 3 个动物。
    """
    place = rng.choice(ENTITIES["place"])
    correct_animal = rng.choice(ENTITIES["animal"])
    distractors = []
    seen = {correct_animal}
    while len(distractors) < 3:
        d = _pick_distractor(rng, ENTITIES["animal"], correct_animal)
        if d not in seen:
            distractors.append(d)
            seen.add(d)
    choices = [correct_animal] + distractors
    answer_idx = _decide_answer_idx(rng, len(choices))
    choices[0], choices[answer_idx] = choices[answer_idx], choices[0]
    return {
        "question": f"Who is in the {place}?",
        "choices": choices,
        "answer_idx": answer_idx,
    }


def _decide_answer_idx(rng: random.Random, n_choices: int) -> int:
    """
    在 [0, n_choices) 内随机选一个位置作为正确答案位置。
    不再偏向 0；均匀分布。
    """
    return rng.randrange(0, n_choices)


# ---------------------------------------------------------------------------
# 批量生成
# ---------------------------------------------------------------------------
def build_benchmark_questions(n: int = 100, seed: int = 1234) -> List[Dict]:
    """
    生成 n 道题。每种模板各占 1/4。
    返回：List[dict]，每个 dict 含
        - question:   str
        - choices:    List[str] 长度 4
        - answer_idx: int ∈ [0, 3]
    """
    rng = random.Random(seed)
    builders = [
        _build_place_question,
        _build_action_question,
        _build_adjective_question,
        _build_animal_in_place_question,
    ]
    items = []
    for i in range(n):
        builder = builders[i % len(builders)]
        items.append(builder(rng))
    return items


# ---------------------------------------------------------------------------
# 持久化
# ---------------------------------------------------------------------------
def save_benchmark(items: List[Dict], save_path: str) -> None:
    """把题库写到 JSON 文件。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def load_benchmark(load_path: str) -> List[Dict]:
    """从 JSON 文件读题库。"""
    with open(load_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 线性脚本：直接执行（**无** if __name__ 包装）
# ---------------------------------------------------------------------------
_ITEMS = build_benchmark_questions(n=100, seed=1234)
_SAVE_PATH = os.path.join(os.path.dirname(__file__), "benchmark.json")
save_benchmark(_ITEMS, _SAVE_PATH)
print(f"[benchmark_data] wrote {len(_ITEMS)} questions to {_SAVE_PATH}")

"""
DPO 偏好数据集（chosen / rejected 成对）。

对应主线：gpt-learning/gpt-learning-pytorch/data/preference_data.py
接口 `generate_preference_dataset(n_samples, seed)` 与主线一致。
独立重写，不跨目录 import（AGENTS.md 硬约束）。

【与 GPT-2 线的区别】
    GPT-2 线：SFT 模型的多次采样 → 按启发式规则分好坏 → 喂给 Reward Model
    Qwen3 线：同样的采样 + 分好坏思路，但直接喂给 DPO，**不训 Reward Model**

【DPO 需要什么】
    每条数据只要三段文本：
        prompt    共享前缀
        chosen    更好的回复
        rejected  更差的回复
    DPO 用 logπ(y|x) 的差值构造偏好信号，不需要显式的标量奖励。

【★ 为什么 prompt / chosen / rejected 是「三段分开」的文本】
    DPO 的核心是「只对回复部分算 logprob」：
        logπ(y_w|x) = Σ_{t ∈ 回复位置} log p(y_t | x, y_<t)
    回复的起始位置必须精确知道，所以数据里 prompt 与回复**必须分开存**。
    `format_prompt_response` 只是给「要展示完整文本」的场景（评估 / 生成）用的，
    DPO 训练路径刻意用分开的两段（见 train/train_lora.py 的 encode_pair）。

【四种「变差」策略（与 GPT-2 线思路一致）】
    wrong_entity  事实错误：把回复里的关键词换掉
    unrelated     答非所问：换成回答另一个问题的内容
    too_short     信息不足：截断成很短一句
    noisy         重复堆砌：把回复重复三遍，语义糊掉
"""

import random
from typing import Dict, List, Optional

# 问题与「好回答」一一对应（GOOD[i] 是 PROMPTS[i] 的正确答案）
PROMPTS = [
    "What is attention?",
    "什么是 RoPE？",
    "Explain RMSNorm in one sentence.",
    "GQA 和 MHA 有什么区别？",
    "Why do we need a reward model?",
]
GOOD = [
    "Attention lets every token look at every other token and weigh how much to copy.",
    "RoPE 用旋转把位置信息编进 Q 和 K，使注意力分数只依赖相对位置。",
    "RMSNorm divides by the root mean square of activations, so it rescales without centering.",
    "GQA 让多个 Q head 共享一组 KV head，缓存和计算量更小。",
    "因为人类偏好本身是相对的，直接学一个标量奖励函数很难，需要和参考答案对比。",
]
# 四种「变差」策略
STRATEGIES = ["wrong_entity", "unrelated", "too_short", "noisy"]

_CJK_MIN = "一"
_CJK_MAX = "鿿"


def _is_cjk(text: str) -> bool:
    """
    判断文本里是否含中日韩统一表意文字。

    参数：
        text: 输入文本
    返回：
        True 表示含中文
    """
    return any(_CJK_MIN <= c <= _CJK_MAX for c in text)


def _make_wrong(good: str, rng: random.Random) -> str:
    """
    把回复里的关键内容换掉，制造事实错误。

    中文按「换一个字」处理，英文按「换一个词」处理。

    参数：
        good: 好的回复
        rng:  随机数发生器
    返回：
        变差的回复（保证 != good）
    """
    if _is_cjk(good):
        idxs = [i for i, c in enumerate(good) if _CJK_MIN <= c <= _CJK_MAX]
        i = rng.choice(idxs)
        return good[:i] + "错" + good[i + 1:]
    words = good.split()
    if len(words) < 3:
        return good + " Actually it is completely different."
    i = rng.randrange(len(words))
    words[i] = "banana"
    return " ".join(words)


def _make_unrelated(good: str, rng: random.Random) -> str:
    """
    换成回答另一个问题的内容（答非所问）。

    参数：
        good: 好的回复
        rng:  随机数发生器
    返回：
        变差的回复（保证 != good）
    """
    # ★ 初稿这里是 `rng.choice(GOOD[len(PROMPTS):] or GOOD)`：
    #   GOOD 与 PROMPTS 等长，GOOD[5:] 恒为空 → 每次都退回 GOOD 全集，
    #   有 1/5 概率选回 good 本身 → chosen == rejected，违反数据集不变量，
    #   测试也会随机挂掉。改成「换一个 prompt 的答案」，保证一定不同。
    others = [g for g in GOOD if g != good]
    return rng.choice(others)


def _make_too_short(good: str, rng: random.Random) -> str:
    """
    太短，信息不足。

    参数：
        good: 好的回复
        rng:  随机数发生器
    返回：
        变差的回复（保证 != good，且非空）
    """
    cut = max(1, len(good) // 8)
    out = good[:cut]
    if out == good:            # good 太短，截断等于原文
        out = good[:1]
    return out


def _make_noisy(good: str, rng: random.Random) -> str:
    """
    重复堆砌，语义糊掉。

    参数：
        good: 好的回复
        rng:  随机数发生器
    返回：
        变差的回复（保证 != good）
    """
    return " ".join([good] * 3)


_MAKERS = {
    "wrong_entity": _make_wrong,
    "unrelated": _make_unrelated,
    "too_short": _make_too_short,
    "noisy": _make_noisy,
}


def generate_preference_dataset(n_samples: int = 2000, seed: int = 42) -> List[Dict]:
    """
    生成 DPO 偏好数据集。

    参数：
        n_samples: 样本条数
        seed:      随机种子
    返回：
        [{"prompt": str, "chosen": str, "rejected": str, "strategy": str}, ...]
    """
    rng = random.Random(seed)
    out: List[Dict] = []
    for _ in range(n_samples):
        pi = rng.randrange(len(PROMPTS))
        prompt, chosen = PROMPTS[pi], GOOD[pi]
        strategy = STRATEGIES[rng.randrange(len(STRATEGIES))]
        rejected = _MAKERS[strategy](chosen, rng)
        # 兜底：任何策略都不允许产出与 chosen 相同的文本
        if rejected == chosen:
            rejected = chosen + " (but wrong)"
        out.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "strategy": strategy,
        })
    return out


def format_prompt_response(prompt: str, response: str) -> str:
    """
    把 prompt 与 response 拼成训练用的完整文本。

    显式分隔，让模型学到「看到分隔符就轮到它回答」。
    注意：**DPO 训练路径不用这个格式**（见模块 docstring），
    它用于「需要一段完整文本」的场景（评估展示 / 生成）。

    参数：
        prompt:   问题
        response: 回答
    返回：
        拼接后的完整文本
    """
    return f"### Question:\n{prompt}\n\n### Answer:\n{response}"


# ---------------------------------------------------------------------------
# DataLoader 工厂（feat-019 的验证项要求提供）
# ---------------------------------------------------------------------------
class PreferenceDataset:
    """
    偏好数据集：每条返回 {prompt, chosen, rejected, strategy}。

    刻意**不**做 tokenize 与 padding：
    DPO 要对 chosen / rejected 分别算 logprob，长度不同、各自独立，
    统一 padding 反而会引入 mask 逻辑；tokenize 放在训练脚本里按需做。

    参数：
        pairs: 偏好对列表
    """

    def __init__(self, pairs: List[Dict]) -> None:
        """
        参数：
            pairs: 偏好对列表
        """
        self.pairs = pairs

    def __len__(self) -> int:
        """样本条数。"""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        """
        取第 idx 条。

        参数：
            idx: 下标
        返回：
            {prompt, chosen, rejected, strategy}
        """
        return self.pairs[idx]


def make_preference_loader(
    n_samples: int = 2000,
    seed: int = 42,
    shuffle: bool = True,
) -> List[Dict]:
    """
    构造偏好数据的迭代器（等价的轻量 loader）。

    与 DataLoader 的区别：偏好对是「变长文本 + 字符串字段」，
    没必要走 DataLoader 的 collate 流程；直接返回打乱后的列表即可。

    参数：
        n_samples: 样本条数
        seed:      随机种子
        shuffle:   是否打乱顺序
    返回：
        [{prompt, chosen, rejected, strategy}, ...]
    """
    pairs = generate_preference_dataset(n_samples=n_samples, seed=seed)
    if shuffle:
        random.Random(seed).shuffle(pairs)
    return pairs

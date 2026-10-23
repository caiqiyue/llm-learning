"""
生成质量的 4 个指标：distinct-n / self-BLEU / token entropy / response quality。

【为什么单独一个文件】
    这 4 个指标是**纯函数**（吃「一批回答」或「一批 logits」），
    与「用哪个模型生成的」无关。放在同一个文件里，两条线才能
    **用同一份代码**算同一个指标 —— 口径一致要靠「同一份实现」，
    不能靠「两边各写一份、写得一模一样」（那只是「看起来一样」）。

【★ 与 GPT-2 线同名同定义】
    4 个指标的定义逐条对齐 GPT-2 线 `evaluate.py` 的同名函数：

    | 指标            | GPT-2 线的函数           | 本文件的函数                |
    |-----------------|-------------------------|----------------------------|
    | distinct_n      | `eval_distinct_n`       | `distinct_n`               |
    | self_bleu       | `eval_self_bleu`        | `self_bleu`                |
    | token_entropy   | `eval_token_entropy`    | `token_entropy_from_logits`|
    | response_quality| `eval_response_quality` | `response_quality`          |

    「同名同口径」不是一句承诺，而是**一条断言**：
    `ablations/compare_quality.py` 会把**同一批固定回答**分别喂给
    「GPT-2 线自己的 `evaluate.py`」和「本文件」，
    两边算出来的值必须**完全相等**，否则直接报错退出。

【★ 本文件可以被 GPT-2 线的进程 import】
    为了做上面那条交叉验证，GPT-2 线那边要能调用**本文件的实现**
    （否则就是「两边各算各的」，谈不上口径一致）。
    所以本文件**只依赖 torch / nltk / 标准库**，不 import 任何项目内模块：
      · 顶层不 import `inference` / `model` / `data`
      · 需要 Qwen3 模型的那两个函数在**函数体内**才 import（惰性）
    ⇒ GPT-2 线的进程按文件路径加载本文件即可，不会碰到包名冲突。

【跨线不可比的量：token entropy】
    熵的绝对值依赖词表大小（均匀分布时 H = log V），
    GPT-2 线的词表 ~64 字符、Qwen3 线 ~778 子词 ⇒ **绝对值不能直接比**。
    本文件额外给 `token_entropy_normalized` = H / log(V)，
    那个才是跨线可比的量。
"""

import math
from typing import Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# 0) 探针集实体表（按**文件路径**加载，避开 `data` 包名冲突）
# ---------------------------------------------------------------------------
#   为什么不用 `from data.quality_prompts import ...`：
#   GPT-2 线的进程里 `data` 这个包名指的是 **GPT-2 的 data/**，
#   那里没有 quality_prompts.py ⇒ 直接 import 会 ImportError。
#   按绝对路径加载就没有这个问题（也不污染 sys.modules 里的 `data`）。
_PROBE_TABLES: Optional[tuple] = None


def load_probe_tables() -> tuple:
    """
    取探针集自带的动物表与动词表。

    返回：
        (animals, verbs) 两个 list
    """
    global _PROBE_TABLES
    if _PROBE_TABLES is None:
        import importlib.util
        import os

        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "data", "quality_prompts.py")
        spec = importlib.util.spec_from_file_location("qwen3_quality_prompts", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法加载探针集：{path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _PROBE_TABLES = (list(mod.QUALITY_ENTITIES["animal"]),
                         list(mod.QUALITY_ENTITIES["action"]))
    return _PROBE_TABLES

# ---------------------------------------------------------------------------
# 1) distinct-n：字符级 n-gram 多样性
# ---------------------------------------------------------------------------
def distinct_n(responses: Sequence[str], n_gram: int = 1) -> float:
    """
    字符级 n-gram 多样性 = unique n-gram 数 / n-gram 总数。

    越接近 1 越多样（每个 n-gram 都不重样），越接近 0 越单调（复读）。

    【口径与 GPT-2 线的 `eval_distinct_n` 逐字一致】
      · 在**回答文本**上算，不含 prompt
      · 长度 < n_gram 的回答整条跳过（不贡献 n-gram）
      · 分母为 0 时返回 0.0

    参数：
        responses: 一批回答（字符串）
        n_gram:    n-gram 的阶数
    返回：
        多样性 ∈ [0, 1]
    """
    if n_gram < 1:
        raise ValueError(f"n_gram 必须 >= 1，实得 {n_gram}")
    all_ngrams: List[str] = []
    for r in responses:
        if len(r) < n_gram:
            continue
        all_ngrams.extend(r[i:i + n_gram] for i in range(len(r) - n_gram + 1))
    return len(set(all_ngrams)) / max(len(all_ngrams), 1)


# ---------------------------------------------------------------------------
# 2) self-BLEU：批内互相重复程度
# ---------------------------------------------------------------------------
def self_bleu(responses: Sequence[str]) -> float:
    """
    Self-BLEU：每条回答对其它 N-1 条的 BLEU 平均。**越低越多样**。

    【口径与 GPT-2 线的 `eval_self_bleu` 逐字一致】
      · 回答按**字符**切成 list（不是按词）
      · 参考句 = 批内其它非空回答
      · nltk 的 `sentence_bleu` + `SmoothingFunction().method1`
      · nltk 不可用时返回 0.0（与 GPT-2 线同样的降级）

    参数：
        responses: 一批回答
    返回：
        self-BLEU ∈ [0, 1]（nltk 不可用时为 0.0）
    """
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    except Exception:
        return 0.0

    tokenized = [list(r) for r in responses]
    smoothing = SmoothingFunction().method1
    bleu_scores: List[float] = []
    for i, hyp in enumerate(tokenized):
        if not hyp:
            continue
        refs = [tokenized[j] for j in range(len(tokenized)) if j != i and tokenized[j]]
        if not refs:
            continue
        try:
            bleu_scores.append(sentence_bleu(refs, hyp, smoothing_function=smoothing))
        except Exception:
            bleu_scores.append(0.0)
    return sum(bleu_scores) / max(len(bleu_scores), 1)


# ---------------------------------------------------------------------------
# 3) token entropy：生成每一步的 next-token 分布熵
# ---------------------------------------------------------------------------
def entropy_from_probs(probs: "torch.Tensor") -> float:
    """
    一个离散分布的熵 H(p) = -Σ p_i log p_i（自然对数，单位 nat）。

    参数：
        probs: (B, V) 的概率分布
    返回：
        熵（float）
    """
    import torch

    # p=0 的位置 log(0) = -inf ⇒ 用 (p + 1e-10) 兜住，再乘 0 消掉
    return float(-(probs * (probs + 1e-10).log()).sum(dim=-1).mean().item())


def token_entropy_from_logits(
    logits_groups: Sequence,
    temperature: float = 1.0,
) -> float:
    """
    生成过程中每一步 next-token 分布熵的**平均值**。

    【口径与 GPT-2 线的 `eval_token_entropy` 逐字一致】
      · logits 先除以 temperature，再 softmax
      · 每步算一次 H
      · **先在样本内对步数取平均，再对样本取平均**
        （不是把所有步摊平一起平均 —— 提前 EOS 的短样本权重会更小，
          两种算法在样本长度不同时会给出不同的数）

    参数：
        logits_groups: 每个样本的逐步 logits。
                       元素是「样本」的列表，样本内每个元素是
                       (B, V) 或 (B, L, V) 的 logits（B 一般是 1）；
                       也接受「扁平的逐步 logits 列表」当只有 1 个样本
        temperature: 采样温度（与生成时用的温度必须一致）
    返回：
        平均熵（float）
    """
    import torch

    if not logits_groups:
        return 0.0

    # 兼容「扁平列表」：元素是 Tensor 而不是 list ⇒ 当成 1 个样本
    groups = []
    for item in logits_groups:
        if isinstance(item, (list, tuple)):
            groups.append(list(item))
        else:
            groups.append([item])

    per_sample: List[float] = []
    for steps in groups:
        if not steps:
            continue
        entropies: List[float] = []
        for logits in steps:
            # 两种形状都支持：
            #   (B, V)    —— Qwen3 的 generate_naive 返回的「最后一个位置」的 logits
            #   (B, L, V) —— 整条序列的 logits（取最后一个位置）
            last = logits[:, -1, :] if logits.dim() == 3 else logits
            scaled = last / max(temperature, 1e-6)
            probs = torch.softmax(scaled, dim=-1)
            entropies.append(entropy_from_probs(probs))
        per_sample.append(sum(entropies) / len(entropies))
    if not per_sample:
        return 0.0
    return sum(per_sample) / len(per_sample)


# ---------------------------------------------------------------------------
# 4) response quality：rule-based 评分（满分 1.0）
# ---------------------------------------------------------------------------
def response_quality_detail(
    prompts: Sequence[str],
    responses: Sequence[str],
    animals: Optional[Sequence[str]] = None,
    verbs: Optional[Sequence[str]] = None,
) -> dict:
    """
    逐条打分的 rule-based 质量评分，并给出**每条规则的命中率**。

    【4 条规则（与 GPT-2 线的 `eval_response_quality` 逐字一致，每条 0.25 分）】
      1) 长度在 10~50 字符
      2) 回答里出现了动物名（取自动物表）
      3) 回答里出现了常见动词（取自动词表）
      4) 回答以 `.` / `!` / `?` 之一结尾（先 rstrip 再判断）

    参数：
        prompts:   对应的 prompt（当前实现不需要，但保留以便将来做
                   「回答是否回应了问题」这类规则 —— 口径与 GPT-2 线保持一致）
        responses: 一批回答
        animals:   动物表；None 时用 `data.quality_prompts.QUALITY_ENTITIES["animal"]`
        verbs:     动词表；None 时用 `data.quality_prompts.QUALITY_ENTITIES["action"]`
    返回：
        {
            "value":         均分（含 4 条规则）,
            "value_ex_length": **扣掉长度规则**后的均分（其余 3 条的命中率）,
            "by_rule":       {规则名: 命中率},
            "mean_chars":    回答的平均字符数,
            "n":             条数,
        }
    """
    if animals is None or verbs is None:
        default_animals, default_verbs = load_probe_tables()
        animals = animals if animals is not None else default_animals
        verbs = verbs if verbs is not None else default_verbs

    animals = list(animals)
    verbs = list(verbs)
    n = len(responses)
    if n == 0:
        return {"value": 0.0, "by_rule": {}, "n": 0}

    hits = {"length": 0, "entity": 0, "verb": 0, "ending": 0}
    total = 0.0
    for resp in responses:
        score = 0.0
        if 10 <= len(resp) <= 50:
            score += 0.25
            hits["length"] += 1
        if any(a in resp for a in animals):
            score += 0.25
            hits["entity"] += 1
        if any(v in resp for v in verbs):
            score += 0.25
            hits["verb"] += 1
        if resp.rstrip().endswith((".", "!", "?")):
            score += 0.25
            hits["ending"] += 1
        total += score

    return {
        "value": total / n,
        # 扣掉「长度」规则后的均分：长度规则是**按字符级 tokenizer 标定**的
        # （10~50 字符），Qwen3 线的 BPE 粒度不同 ⇒ 回答普遍更长，
        #  这一条规则对 Qwen3 系统性不利。扣掉它之后的分数才是
        #  「不依赖 tokenizer 粒度」的可比量。
        "value_ex_length": (hits["entity"] + hits["verb"] + hits["ending"]) / (3 * n),
        "by_rule": {k: v / n for k, v in hits.items()},
        "mean_chars": sum(len(r) for r in responses) / n,
        "n": n,
    }


def response_quality(
    prompts: Sequence[str],
    responses: Sequence[str],
    animals: Optional[Sequence[str]] = None,
    verbs: Optional[Sequence[str]] = None,
) -> float:
    """
    response quality 的总分（rule-based，满分 1.0）。

    参数：
        prompts:   对应的 prompt
        responses: 一批回答
        animals:   动物表（None ⇒ 用探针集自带的那份）
        verbs:     动词表（None ⇒ 用探针集自带的那份）
    返回：
        质量分 ∈ [0, 1]
    """
    return response_quality_detail(prompts, responses, animals, verbs)["value"]


# ---------------------------------------------------------------------------
# 5) Qwen3 侧的生成（GPT-2 线不用这一段 —— 它有自己的 generate）
# ---------------------------------------------------------------------------
def generate_responses(
    model: "torch.nn.Module",
    tk,
    prompts: Sequence[str],
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.95,
    eos_id: Optional[int] = None,
    prompt_template: str = "### Question:\n{prompt}\n\n### Answer:\n",
) -> List[str]:
    """
    用 Qwen3 模型把一批 prompt 生成成回答（**不含 prompt 本身**）。

    【prompt 模板】
        用 `### Question:\\n{prompt}\\n\\n### Answer:\\n` —— 与
        `train/train_lora.py` 的 `PROMPT_TEMPLATE` 完全一致，
        这样探针输入对 LoRA 模型是**在域内**的格式。

    参数：
        model:           Qwen3 模型
        tk:              tokenizer
        prompts:         一批 prompt
        max_new_tokens:  最多生成多少 token
        temperature:     采样温度
        top_k:           top-k
        top_p:           top-p
        eos_id:          早停 token
        prompt_template: prompt 的格式化模板
    返回：
        回答字符串列表
    """
    import torch

    from inference import generate_naive

    responses: List[str] = []
    for prompt in prompts:
        ids = tk.encode(prompt_template.format(prompt=prompt))
        if not ids:
            responses.append("")
            continue
        idx = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
        out = generate_naive(
            model, idx,
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, eos_id=eos_id,
        )
        full = tk.decode(out[0].tolist())
        # 去掉 prompt 那一段，只留模型生成的
        prefix = tk.decode(ids)
        resp = full[len(prefix):] if full.startswith(prefix) else full
        responses.append(resp)
    return responses


def generate_responses_with_logits(
    model: "torch.nn.Module",
    tk,
    prompts: Sequence[str],
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.95,
    eos_id: Optional[int] = None,
    prompt_template: str = "### Question:\n{prompt}\n\n### Answer:\n",
):
    """
    同 `generate_responses`，但**同时返回每一步的 logits**（算 token entropy 用）。

    返回：
        (responses, logits_per_sample)
        logits_per_sample[i] 是第 i 条样本的逐步 logits（每个是 (B, V)）
    """
    import torch

    from inference import generate_naive

    responses: List[str] = []
    logits_per_sample: List[List["torch.Tensor"]] = []
    for prompt in prompts:
        ids = tk.encode(prompt_template.format(prompt=prompt))
        if not ids:
            responses.append("")
            logits_per_sample.append([])
            continue
        idx = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
        out, logits_list = generate_naive(
            model, idx,
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, eos_id=eos_id,
            return_logits=True,
        )
        full = tk.decode(out[0].tolist())
        prefix = tk.decode(ids)
        resp = full[len(prefix):] if full.startswith(prefix) else full
        responses.append(resp)
        logits_per_sample.append(list(logits_list))
    return responses, logits_per_sample

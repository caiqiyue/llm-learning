"""
GPT-2 线的生成质量跑分 —— **在 GPT-2 自己的进程里**执行。

【为什么必须开子进程】
    Qwen3 线与 GPT-2 线**都叫 `data` 包、也都叫 `model` 包**。
    同一个 Python 进程里，`import data.tokenizer` 只能解析到其中一个 ——
    所以本文件**不**在 Qwen3 的进程里跑，而是被
    `ablations/compare_quality.py` 用 `PYTHONPATH=<GPT-2 目录>` 拉起来：
        · `data` / `model` / `evaluate` 全部解析到 **GPT-2 线** ✅
        · 共享的 `quality.py` 用**文件路径**加载（不依赖包名）✅

【这个进程干三件事】
    1. 用 **GPT-2 自己的** `_generate_responses` 生成回答；
    2. 用**共享的** `quality.py` 算 4 个指标（与 Qwen3 线同一份代码）；
    3. **交叉验证**：把回答固定住，再跑 **GPT-2 自己的** `eval_distinct_n` /
       `eval_self_bleu` / `eval_response_quality` / `eval_token_entropy`，
       它们的值必须与第 2 步的值**完全相等** ——
       这就是「同名同口径」的机器证明（不相等 ⇒ 父进程报错退出）。

【用法（由 compare_quality.py 调用，不要手工跑）】
    python _gpt2_quality_runner.py --payload <in.json> --out <out.json>
"""

import argparse
import importlib.util
import io
import json
import math
import os
import sys
from typing import Dict, List

import torch


def _load_shared_quality():
    """
    按**文件路径**加载 Qwen3 线的 `quality.py`（共享指标实现）。

    为什么不用 `import quality`：本进程的 `PYTHONPATH` 指向 GPT-2 线，
    而且 Qwen3 的项目根一旦进了 `PYTHONPATH`，`data` / `model` 就会
    被 Qwen3 的抢走（两个包同名）⇒ 只能按绝对路径加载。

    返回：
        quality 模块
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(here), "quality.py")
    spec = importlib.util.spec_from_file_location("qwen3_shared_quality", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载共享指标实现：{path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qwen3_shared_quality"] = mod
    spec.loader.exec_module(mod)
    return mod


def _import_gpt2_evaluate():
    """
    import GPT-2 线的 `evaluate.py`，**但不执行它的命令行入口**。

    【为什么要绕这一下】
        GPT-2 线的 `evaluate.py` 在**模块层**就做了两件事：
            parser = argparse.ArgumentParser(...)
            args = parser.parse_args()
        然后是一串 `if args.stage == ...:` 的分派。
        直接 `import evaluate` ⇒ ① argparse 会因为没有 `--stage` 直接报错退出；
        ② 就算给了 `--stage`，也会**顺手把那一整个 stage 跑一遍**。

        这里把 `argparse.ArgumentParser.parse_args` 临时换成一个返回
        「stage = __none__」的假命名空间 ⇒ 模块层的分派**一个分支都不会命中**，
        但所有 `eval_*` 函数都已经定义好了，可以随便调用。
        import 完立刻把 `parse_args` 换回去。

    返回：
        GPT-2 线的 evaluate 模块
    """
    import argparse

    class _DummyArgs:
        """假的命令行参数（stage 不匹配任何分支 ⇒ 什么都不跑）。"""
        stage = "__none__"
        checkpoint = ""
        sft = ""
        ppo = ""
        rm = ""
        device = "cpu"

    original = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = lambda self, *a, **k: _DummyArgs()
    try:
        import evaluate as gpt2_eval
    finally:
        argparse.ArgumentParser.parse_args = original
    return gpt2_eval


def _greedy_entropy_groups(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    eos_id: int,
) -> List[List["torch.Tensor"]]:
    """
    沿**贪心**路径走一遍 GPT-2 的模型，收集每一步的 logits。

    【为什么要「贪心」】
        GPT-2 自己的 `eval_token_entropy` 是**采样**的（`torch.multinomial`），
        两次跑出来的路径不同 ⇒ 熵也不同，没法逐值比对。
        把采样换成 argmax 之后，路径是确定的，
        「GPT-2 自己的实现」与「共享实现」就能算出**同一个数**。

    ★ 这一段的算法与 GPT-2 线 `eval_token_entropy` 的循环**逐行对应**
      （含「超过 block_size 就截断到末尾」这一步）。

    参数：
        model:           GPT-2 模型
        tokenizer:       GPT-2 的 CharTokenizer
        prompts:         一批 prompt
        max_new_tokens:  最多走多少步
        temperature:     温度（与 GPT-2 线 config.temperature 一致）
        eos_id:          早停 token
    返回：
        每个样本的逐步 logits 列表
    """
    groups: List[List["torch.Tensor"]] = []
    for prompt_str in prompts:
        full_prompt = f"<bos> {prompt_str} <sep> "
        prompt_ids = tokenizer.encode(full_prompt)
        if not prompt_ids:
            continue
        x = torch.tensor([prompt_ids], dtype=torch.long)
        steps: List["torch.Tensor"] = []
        for _ in range(max_new_tokens):
            if x.size(1) > model.block_size:
                x = x[:, -model.block_size:]
            out = model(x)
            logits = out[0]
            steps.append(logits[:, -1, :])
            probs = torch.softmax(logits[:, -1, :] / max(temperature, 1e-6), dim=-1)
            next_id = probs.argmax(dim=-1, keepdim=True)
            if int(next_id.item()) == eos_id:
                break
            x = torch.cat([x, next_id], dim=1)
        groups.append(steps)
    return groups


def main() -> int:
    """跑 GPT-2 侧的生成质量并把结果写成 JSON。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", required=True, help="输入 JSON（探针 / 参数 / checkpoint）")
    ap.add_argument("--out", required=True, help="输出 JSON")
    args = ap.parse_args()

    with io.open(args.payload, encoding="utf-8") as f:
        payload = json.load(f)
    prompts: List[str] = payload["prompts"]
    ckpt: str = payload["checkpoint"]
    max_new_tokens: int = payload["max_new_tokens"]
    n_entropy: int = payload["n_entropy_samples"]
    temperature: float = payload["temperature"]
    seed: int = payload.get("seed", 42)
    device = "cpu"

    # ★ 固定随机种子 ⇒ 采样出来的路径可复现，两次跑的数字必须一模一样。
    #   （生成是**采样**（temperature 0.8），不固定种子的话每次跑数字都在动，
    #    那样「并排对照」就没有意义了。）
    torch.manual_seed(seed)

    quality = _load_shared_quality()

    # --- GPT-2 线的模块（在本进程里，`data` / `model` / `evaluate` 都是 GPT-2 的）---
    gpt2_eval = _import_gpt2_evaluate()
    from data.pretrain_data import ENTITIES

    model, tokenizer, vocab_size = gpt2_eval._load_model(
        ckpt, with_value_head=False, device=device)

    # ★ 让 GPT-2 自己的 eval_* 直接复用**已经加载好的**模型。
    #   不然每调一个 eval_* 就重新 load 一次 checkpoint（5 次），
    #   两对模型跑下来要好几分钟；模型是同一个、状态不变 ⇒ 复用完全等价。
    gpt2_eval._load_model = lambda *a, **k: (model, tokenizer, vocab_size)

    # --- 1) GPT-2 自己的生成 ---
    responses = gpt2_eval._generate_responses(
        model, tokenizer, prompts, max_new_tokens=max_new_tokens, device=device
    )

    # --- 2) 共享实现的 4 个指标 ---
    rq = quality.response_quality_detail(prompts, responses)
    shared: Dict[str, object] = {
        "distinct_1": quality.distinct_n(responses, 1),
        "distinct_2": quality.distinct_n(responses, 2),
        "self_bleu": quality.self_bleu(responses),
        "response_quality": rq["value"],
        "response_quality_ex_length": rq["value_ex_length"],
        "response_quality_by_rule": rq["by_rule"],
        "mean_response_chars": rq["mean_chars"],
    }

    # --- 3) 交叉验证：固定回答 + 固定 prompt，跑 GPT-2 自己的 evaluate 函数 ---
    fixed = list(responses)
    gpt2_eval._generate_responses = lambda *a, **k: list(fixed)
    # ★ 一定要**按 n 截断**：GPT-2 的 `eval_token_entropy(n_samples=20)` 会取
    #   「前 20 条 prompt」，如果不截断，它会遍历全部 30 条 ⇒
    #   两边算的样本集不一样 ⇒ 熵对不上（差了 2.8%）。
    gpt2_eval.get_ppo_prompts = lambda n=50, seed=42: [
        {"prompt": p, "id": i} for i, p in enumerate(prompts[:n])
    ]

    own: Dict[str, object] = {}
    own["distinct_1"] = gpt2_eval.eval_distinct_n(
        ckpt, n_gram=1, n_samples=len(prompts), device=device)["value"]
    own["distinct_2"] = gpt2_eval.eval_distinct_n(
        ckpt, n_gram=2, n_samples=len(prompts), device=device)["value"]
    own["self_bleu"] = gpt2_eval.eval_self_bleu(
        ckpt, n_samples=len(prompts), device=device)["value"]
    own["response_quality"] = gpt2_eval.eval_response_quality(
        ckpt, n_samples=len(prompts), device=device)["value"]

    # token entropy：把采样换成贪心，让路径确定 ⇒ 两边能算出同一个数
    orig_multinomial = torch.multinomial
    torch.multinomial = lambda p, num_samples=1, **kw: p.argmax(dim=-1, keepdim=True)
    try:
        own["token_entropy"] = gpt2_eval.eval_token_entropy(
            ckpt, n_samples=n_entropy, max_new_tokens=max_new_tokens, device=device)["value"]
    finally:
        torch.multinomial = orig_multinomial

    # 共享实现算的熵（同一批 prompt、同一段贪心路径）
    groups = _greedy_entropy_groups(
        model, tokenizer, prompts[:n_entropy], max_new_tokens, temperature, gpt2_eval.EOS_ID
    )
    shared["token_entropy"] = quality.token_entropy_from_logits(groups, temperature=temperature)
    # 归一化熵：H / log(V) —— 熵的**绝对值**依赖词表大小（均匀分布时 H = log V），
    # GPT-2 线词表 ~64 字符、Qwen3 线 ~778 子词 ⇒ 绝对值跨线不可比，归一化后才可比。
    shared["token_entropy_normalized"] = shared["token_entropy"] / math.log(tokenizer.vocab_size)

    # --- 4) 实体表对齐检查（防止探针集与 GPT-2 的 ENTITIES 漂移）---
    out = {
        "responses": responses,
        "shared": shared,
        "own": own,
        "gpt2_entities": {k: list(v) for k, v in ENTITIES.items()},
        "n_samples": len(prompts),
    }
    with io.open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

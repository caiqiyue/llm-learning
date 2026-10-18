"""
debug_preference_data.py — 单步调试 Preference 数据 + 5 种 rejected 策略

跑法（在 gpt-learning-pytorch/ 目录下）：
    python data/debug_preference_data.py

目的：
  - 看 generate_preference_dataset 的 prompt/chosen/rejected 三元组
  - ★ 看 5 种 rejected 策略的样例（wrong_entity / unrelated / noisy_keywords / too_short / too_long）
  - 看每种策略实际出现频率（5000 条样本的分布）
  - 看 Dataset 和 collate_fn


后面 train_reward.py 会做这件事：


r_chosen  = reward_model(input_ids=chosen_batch)     # (B,)
r_rejected = reward_model(input_ids=rejected_batch)  # (B,)
loss = -log sigmoid(r_chosen - r_rejected).mean()   # ★ 一对 scalar
⚠️ 两个 input_ids 必须形状完全一致（同一个 (B, L)），reward model 才能用 PyTorch 矩阵算。
"""

import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_PJT = os.path.dirname(_HERE)
sys.path.insert(0, _PJT)

from data.preference_data import (  # noqa: E402
    REJECTED_STRATEGIES,
    build_rejected,
    generate_preference_dataset,
)
from data.sft_data import generate_sft_dataset, format_sft_sample  # noqa: E402

# torch 环境检测
try:
    import torch
    from data.preference_data import PreferenceDataset, preference_collate_fn, make_preference_loader
    TORCH_OK = True
    TORCH_ERR = None
except Exception as e:
    TORCH_OK = False
    TORCH_ERR = repr(e)
    PreferenceDataset = preference_collate_fn = make_preference_loader = None


def hr(title: str):
    print(f"\n{'=' * 64}\n {title}\n{'=' * 64}")


def main():
    vocab_path = os.path.join(_HERE, "vocab.txt")
    if not os.path.exists(vocab_path):
        print("[!] vocab.txt 不存在，请先跑: python data/build_vocab.py")
        return

    hr("Step 1: 5 种 rejected 策略定义")
    print(f"{'策略名':<18}{'概率':<8}{'描述'}")
    print("-" * 64)
    strategy_desc = {
        "wrong_entity":   "把 chosen 里的动物名换成另一个动物",
        "unrelated":      "从其他 SFT response 里抽一条当 rejected",
        "noisy_keywords": "保留关键词但打乱词序",
        "too_short":      "只保留第 1 个词",
        "too_long":       "重复 5 次变成超长",
    }
    for name, weight in REJECTED_STRATEGIES:
        print(f"{name:<18}{weight:<8.0%} {strategy_desc[name]}")

    hr("Step 2: 单条样本示例 —— 5 种策略各自展示一条")
    print("\n下面强制构造 5 条样本，每条覆盖一种 rejected 策略：\n")
    # 用同一对 prompt / chosen 跑 5 次，每次指定不同策略
    sft_samples = generate_sft_dataset(n_samples=200, seed=42)
    base = sft_samples[0]

    # 强制覆盖策略：直接调用 _build_* 函数
    import random
    from data import preference_data as pd_mod
    rng = random.Random(0)

    # builder 签名不同, 按 name 分别调用
    example_prompts = ["wrong_entity", "unrelated", "noisy_keywords", "too_short", "too_long"]

    chosen_resp = base["response"]
    prompt = base["prompt"]
    print(f"基础: prompt={prompt!r}")
    print(f"      chosen={chosen_resp!r}\n")

    for name in example_prompts:
        # 不同 builder 的签名不一样；这里分别调用
        if name == "wrong_entity":
            new_text = pd_mod._build_wrong_entity(chosen_resp, prompt, rng)
        elif name == "unrelated":
            new_text = pd_mod._build_unrelated(chosen_resp, rng, [s["response"] for s in sft_samples[1:]])
        elif name == "noisy_keywords":
            new_text = pd_mod._build_noisy_keywords(chosen_resp, rng)
        elif name == "too_short":
            new_text = pd_mod._build_too_short(chosen_resp)
        elif name == "too_long":
            new_text = pd_mod._build_too_long(chosen_resp)
        print(f"━━━ {name} ━━━")
        print(f"  chosen  : {chosen_resp}")
        print(f"  rejected: {new_text}")
        print()

    hr("Step 3: 真实数据集 —— 5000 条样本的策略分布")
    n = 5000
    samples = generate_preference_dataset(n_samples=n, seed=42)
    counter = Counter(s["rejected_strategy"] for s in samples)
    print(f"样本数: {n}\n")
    print(f"{'策略':<18}{'实际数':<10}{'实际比例':<12}{'期望比例':<12}{'偏差'}")
    print("-" * 64)
    for name, weight in REJECTED_STRATEGIES:
        actual = counter.get(name, 0)
        actual_pct = actual / n
        expected_pct = weight
        diff = actual_pct - expected_pct
        print(f"{name:<18}{actual:<10}{actual_pct:<12.2%}{expected_pct:<12.0%}{diff:+.2%}")

    hr("Step 4: 5 条样本的 chosen vs rejected 并排对比")
    for s in samples[:5]:
        print(f"\n━━━━━━━━━━━━━━━━ id={s['id']} ━━━━━━━━━━━━━━━━")
        print(f"  prompt:    {s['prompt']}")
        print(f"  chosen:    {s['chosen']}")
        print(f"  rejected:  {s['rejected'][:120]}{'...' if len(s['rejected'])>120 else ''}")
        print(f"  strategy:  {s['rejected_strategy']}")

    hr("Step 5: ★ 关键代码解读 —— 复用 SFT 数据集保证 id 一致")
    print("generate_preference_dataset 内部直接调用 generate_sft_dataset，")
    print("保证同 id 的 prompt 与 SFT 数据集完全一致：\n")

    sft = generate_sft_dataset(n_samples=5, seed=42)
    pref = generate_preference_dataset(n_samples=5, seed=42)
    for sf, pr in zip(sft, pref):
        match = "[OK]" if sf["prompt"] == pr["prompt"] else "[!]"
        print(f"  id={sf['id']} prompt sft={sf['prompt']!r}")
        print(f"            pref={pr['prompt']!r}  (一致？{match})")

    hr("Step 6: PreferenceDataset + collate_fn")
    if not TORCH_OK:
        print(f"[跳过] torch 不可用，无法演示 PreferenceDataset / collate")
        print(f"       原因: {TORCH_ERR}")
    else:
        samples_small = generate_preference_dataset(n_samples=10, seed=42)
        ds = PreferenceDataset(samples_small)#看懂上面和这两行即可
        print(f"Dataset 长度: {len(ds)}")

        # 取前 3 条看单样本
        for i in range(3):
            item = ds[i]
            c = item["chosen_input_ids"].tolist()
            r = item["rejected_input_ids"].tolist()
            print(f"\nsample[{i}]:")
            print(f"  chosen_input_ids:   len={len(c):3}  {c[:30]}{'...' if len(c)>30 else ''}")
            print(f"  rejected_input_ids: len={len(r):3}  {r[:30]}{'...' if len(r)>30 else ''}")

        # 看 collate
        print(f"\n★ 关键: chosen 和 rejected 长度都 pad 到 batch 的 max(len(chosen), len(rejected))")
        batch = preference_collate_fn([ds[0], ds[1], ds[2]])#这个也需要看懂
        print(f"  chosen_input_ids:   shape={tuple(batch['chosen_input_ids'].shape)}")
        print(f"  rejected_input_ids: shape={tuple(batch['rejected_input_ids'].shape)}")

        # 看 padding 数量
        ch = batch["chosen_input_ids"]
        rj = batch["rejected_input_ids"]
        for i in range(ch.size(0)):
            n_pad_c = (ch[i] == 0).sum().item()
            n_pad_r = (rj[i] == 0).sum().item()
            total_c = ch[i].size(0)
            total_r = rj[i].size(0)
            print(f"\n  sample[{i}]: chosen_len={total_c} pad={n_pad_c}  ({n_pad_c}/{total_c} = {n_pad_c/total_c*100:.0f}%)")
            print(f"             rejected_len={total_r} pad={n_pad_r}  ({n_pad_r}/{total_r} = {n_pad_r/total_r*100:.0f}%)")

    hr("Step 7: make_preference_loader")
    if not TORCH_OK:
        print(f"[跳过] torch 不可用，无法演示 make_preference_loader")
        print(f"       预期输出: chosen_input_ids:(B, L), rejected_input_ids:(B, L) — 形状一致")
    else:
        loader = make_preference_loader(split="train", batch_size=8, seed=42, n_samples=100)
        print(f"loader 类型: {type(loader).__name__}")
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= 1:
                break
            print(f"\nbatch[{batch_idx}]:")
            print(f"  chosen_input_ids:   {tuple(batch['chosen_input_ids'].shape)}")
            print(f"  rejected_input_ids: {tuple(batch['rejected_input_ids'].shape)}")
            # ★ 验证关键：chosen 和 rejected 长度一致
            assert batch["chosen_input_ids"].shape == batch["rejected_input_ids"].shape, "shape mismatch!"
            print(f"  OK chosen 和 rejected 形状一致 (都是 {batch['chosen_input_ids'].shape})")

    print("\n" + "=" * 64)
    print(" [OK] preference_data debug 完成")
    print("=" * 64)


if __name__ == "__main__":
    main()

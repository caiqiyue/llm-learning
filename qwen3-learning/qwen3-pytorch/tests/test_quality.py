"""
生成质量指标的 smoke（对应 feat-026）。

测的是 `quality.py` 里的**纯函数**：
  1. distinct_n     —— 手算能对上的 n-gram 多样性
  2. self_bleu      —— 全同 ⇒ 1.0；全异 ⇒ 低
  3. entropy        —— 均匀分布的熵 = log V；确定性分布 = 0
  4. response_quality —— 4 条规则逐条可验
  5. 探针集         —— 可复现、长度对得上、实体表与 GPT-2 线一致
  6. 口径           —— 共享实现与 GPT-2 线自己的 evaluate.py 逐值相等

对应笔记：qwen3-learning/Qwen3_训练推理完整学习笔记.md §生成质量评估

运行：python tests/test_quality.py
"""

import math
import os
import sys

# 直接执行 `python tests/test_quality.py` 时 sys.path[0] 是 tests/ 目录，
# 找不到项目根下的 config.py / quality.py —— 这里把项目根补进 sys.path。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import quality
from data.quality_prompts import (
    QUALITY_ENTITIES,
    get_corpus_prompts,
    get_quality_prompts,
)

# =====================================================================
# 1) distinct_n：手算能对上
# =====================================================================
# "abab" 的 1-gram = {a, b} 共 2 个，位置 4 个 ⇒ 2/4 = 0.5
assert quality.distinct_n(["abab"], 1) == 0.5, quality.distinct_n(["abab"], 1)
# 2-gram："abab" → {"ab", "ba", "ab"} 共 2 个不同，3 个位置 ⇒ 2/3
assert abs(quality.distinct_n(["abab"], 2) - 2 / 3) < 1e-12
# 全部相同 ⇒ 1 个 n-gram / 6 个位置（两条 "xxx" 各 3 个位置）
assert quality.distinct_n(["xxx", "xxx"], 1) == 1 / 6
# 空串不贡献 n-gram（长度 < n_gram 整条跳过），"ab" 的 1-gram 全不重样 ⇒ 1.0
assert quality.distinct_n(["", "ab"], 1) == 1.0
# 长度 < n_gram 的整条被跳过 ⇒ 分母为 0 ⇒ 返回 0.0
assert quality.distinct_n(["a"], 2) == 0.0
# n_gram 必须 >= 1
try:
    quality.distinct_n(["abc"], 0)
    raise AssertionError("n_gram=0 应该报错")
except ValueError:
    pass
print("[quality] distinct_n 手算一致")

# =====================================================================
# 2) self_bleu：全同 ⇒ 1.0（完全复读）；完全互异 ⇒ 明显更低
# =====================================================================
same = ["hello world"] * 4
sb_same = quality.self_bleu(same)
assert abs(sb_same - 1.0) < 1e-9, f"全相同的回答 self_bleu 应为 1.0，实得 {sb_same}"

varied = ["abcdefgh", "ijklmnop", "qrstuvwx", "yz012345"]
sb_var = quality.self_bleu(varied)
assert 0.0 <= sb_var <= 1.0, f"self_bleu 越界：{sb_var}"
assert sb_var < sb_same, f"互异的回答 self_bleu 应低于全同：{sb_var} vs {sb_same}"
# 空回答不参与打分（GPT-2 线的同一口径）
assert 0.0 <= quality.self_bleu(["", ""]) <= 1.0
print(f"[quality] self_bleu：全同={sb_same:.4f} 全异={sb_var:.4f}")

# =====================================================================
# 3) 熵：均匀分布 = log V；确定性分布 = 0
# =====================================================================
V = 8
uniform = torch.full((1, V), 1.0 / V)
ent_uniform = quality.entropy_from_probs(uniform)
assert abs(ent_uniform - math.log(V)) < 1e-6, \
    f"均匀分布的熵应等于 log(V)={math.log(V):.6f}，实得 {ent_uniform}"

onehot = torch.zeros((1, V))
onehot[0, 3] = 1.0
ent_onehot = quality.entropy_from_probs(onehot)
assert ent_onehot < 1e-6, f"确定性分布的熵应为 0，实得 {ent_onehot}"
print(f"[quality] 熵：均匀={ent_uniform:.6f}（= log {V} = {math.log(V):.6f}） "
      f"确定性={ent_onehot:.2e}")

# 3b) token_entropy_from_logits：按样本分组，先样本内平均再样本间平均
#     用「k 个等概率取值」的分布造熵：H = log k（取值精确、可手算）
#     样本 A：2 步，H = log 2 与 log 8 ⇒ 样本内均值 (log2+log8)/2 = 1.3863
#     样本 B：1 步，H = log 64 = 4.1589 ⇒ 样本内均值 4.1589
#     样本间均值 = (1.3863 + 4.1589) / 2 = 2.7726
#     ⚠️ 如果错误地「把所有步摊平一起平均」会得到 (log2+log8+log64)/3 = 2.3105
def _uniform_logits(k: int, v: int = 64) -> "torch.Tensor":
    """造一个「在 k 个 token 上等概率」的 logits（其熵恰为 log k）。"""
    logits = torch.full((1, v), -1e9)
    logits[0, :k] = 0.0
    return logits


H2, H8, H64 = math.log(2), math.log(8), math.log(64)
expect_grouped = ((H2 + H8) / 2 + H64) / 2
expect_pooled = (H2 + H8 + H64) / 3
groups = [
    [_uniform_logits(2), _uniform_logits(8)],
    [_uniform_logits(64)],
]
ent = quality.token_entropy_from_logits(groups, temperature=1.0)
assert abs(ent - expect_grouped) < 1e-4, \
    f"按样本分组平均应得 {expect_grouped:.4f}，实得 {ent}"
assert abs(ent - expect_pooled) > 1e-3, \
    "分组平均与摊平平均不该相等（否则这个断言测不出分组逻辑）"
# 扁平列表（只有 1 个样本）也要能用
ent_flat = quality.token_entropy_from_logits(
    [_uniform_logits(2), _uniform_logits(8)], temperature=1.0)
assert abs(ent_flat - (H2 + H8) / 2) < 1e-4, f"单样本应得 {(H2 + H8) / 2:.4f}，实得 {ent_flat}"
# 空输入 ⇒ 0.0
assert quality.token_entropy_from_logits([]) == 0.0
print(f"[quality] token entropy 按样本分组平均：{ent:.4f}"
      f"（摊平平均会是 {expect_pooled:.4f}）")

# =====================================================================
# 4) response_quality：4 条规则逐条可验
# =====================================================================
animals = ["dog", "cat"]
verbs = ["runs", "jumps"]
prompts = ["What is a dog?"] * 4
# 回答 1：长度 19（10~50 ✓）、含动物 ✓、含动词 ✓、以 . 结尾 ✓ ⇒ 1.0
# 回答 2：长度 2 ✗、不含动物 ✗、不含动词 ✗、不以标点结尾 ✗ ⇒ 0.0
# 回答 3：长度 62 ✗、含动物 ✓、含动词 ✓、不以标点结尾 ✗ ⇒ 0.5
# 回答 4：长度 12 ✓、不含动物 ✗、含动词 ✓、不以标点结尾 ✗ ⇒ 0.5
responses = [
    "The dog runs fast.",                       # 19 字符
    "no",                                       # 2 字符
    "cat jumps" + "x" * 53,                     # 62 字符，不以标点结尾
    "it runs here",                             # 12 字符
]
d = quality.response_quality_detail(prompts, responses, animals=animals, verbs=verbs)
assert abs(d["value"] - (1.0 + 0.0 + 0.5 + 0.5) / 4) < 1e-9, \
    f"总分算错：{d['value']}"
assert abs(d["by_rule"]["length"] - 2 / 4) < 1e-9, d["by_rule"]
assert abs(d["by_rule"]["entity"] - 2 / 4) < 1e-9, d["by_rule"]
assert abs(d["by_rule"]["verb"] - 3 / 4) < 1e-9, d["by_rule"]
assert abs(d["by_rule"]["ending"] - 1 / 4) < 1e-9, d["by_rule"]
# 扣掉长度规则之后的分数（跨 tokenizer 粒度可比的那一个）
assert abs(d["value_ex_length"] - (2 + 3 + 1) / (3 * 4)) < 1e-9, d["value_ex_length"]
assert d["n"] == 4
assert abs(d["mean_chars"] - sum(len(r) for r in responses) / 4) < 1e-9
# 空输入不炸
empty = quality.response_quality_detail([], [])
assert empty["value"] == 0.0 and empty["n"] == 0
print(f"[quality] response_quality 规则命中：{d['by_rule']} 总分={d['value']:.4f}")

# =====================================================================
# 5) 探针集：可复现、条数对得上、实体表与 GPT-2 线一致
# =====================================================================
p1 = get_quality_prompts(10, 1234)
p2 = get_quality_prompts(10, 1234)
assert p1 == p2, "同一 seed 必须给出同一批探针（否则对照不可复现）"
assert len(p1) == 10, len(p1)
assert len(get_quality_prompts(30, 1234)) == 30
# 不同 seed ⇒ 不同的探针（否则 seed 白设了）
assert get_quality_prompts(10, 999) != p1
# 域内探针也要能生成，且条数一致
c1 = get_corpus_prompts(10, 1234)
assert len(c1) == 10 and c1 == get_corpus_prompts(10, 1234)
# 探针必须都是非空字符串
for tpl in p1:
    assert isinstance(tpl, str) and tpl.strip(), f"探针不合法：{tpl!r}"
print(f"[quality] 探针集可复现：n=10 seed=1234 例：{p1[0]!r}")

# --- 实体表必须与 GPT-2 线的 ENTITIES 一致（否则 response quality 不是同一件事）---
# ⚠️ 必须开**子进程**查：GPT-2 线的 `data/pretrain_data.py` 内部 `from data.tokenizer import ...`，
#    而本进程里 `data` 包已经指向 **Qwen3 的** data（同名包，两条线各有一份）
#    ⇒ 在本进程里 import 会拿到 Qwen3 的 tokenizer ⇒ ImportError。
#    子进程里把 PYTHONPATH / cwd 指向 GPT-2 线，`data` 就只可能是 GPT-2 的。
GPT2_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))),
    "gpt-learning", "gpt-learning-pytorch")
gpt2_data = os.path.join(GPT2_DIR, "data", "pretrain_data.py")
if os.path.exists(gpt2_data):
    import json
    import subprocess
    env = dict(os.environ)
    env["PYTHONPATH"] = GPT2_DIR + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, "-c",
         "import json\n"
         "from data.pretrain_data import ENTITIES\n"
         "print(json.dumps({k: list(v) for k, v in ENTITIES.items()}))\n"],
        cwd=GPT2_DIR, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace")
    assert proc.returncode == 0, f"取 GPT-2 实体表失败：{proc.stderr[-800:]}"
    gpt2_entities = json.loads(proc.stdout.strip().splitlines()[-1])
    for key in ("animal", "action"):
        assert list(gpt2_entities[key]) == list(QUALITY_ENTITIES[key]), \
            f"实体表 `{key}` 与 GPT-2 线不一致 ⇒ response quality 不是同一个指标"
    print("[quality] 探针实体表与 GPT-2 线 ENTITIES 一致")
else:
    print(f"[quality] 跳过实体表核对（找不到 {gpt2_data}）")

# =====================================================================
# 6) 口径：共享实现 vs GPT-2 线自己的 evaluate.py —— 必须逐值相等
#    这一条是「同名同口径」的机器证明。
# =====================================================================
if os.path.exists(os.path.join(GPT2_DIR, "evaluate.py")):
    import importlib.util
    import json
    import subprocess
    import tempfile

    # 用 runner 子进程跑一次真正的交叉验证（跟 ablations/compare_quality.py 同一条路径）
    runner = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "ablations", "_gpt2_quality_runner.py")
    tmpdir = tempfile.mkdtemp(prefix="quality_cross_")
    payload = os.path.join(tmpdir, "payload.json")
    out = os.path.join(tmpdir, "out.json")
    probes = get_quality_prompts(6, 1234)
    with open(payload, "w", encoding="utf-8") as f:
        json.dump({
            "prompts": probes,
            "checkpoint": os.path.join(GPT2_DIR, "checkpoints", "sft.pt"),
            "max_new_tokens": 8,
            "n_entropy_samples": 4,
            "temperature": 0.8,
            "seed": 1234,
        }, f, ensure_ascii=False)
    env = dict(os.environ)
    env["PYTHONPATH"] = GPT2_DIR + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, runner, "--payload", payload, "--out", out],
        cwd=GPT2_DIR, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise AssertionError(
            f"runner 失败（退出码 {proc.returncode}）：\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
    with open(out, encoding="utf-8") as f:
        res = json.load(f)
    for key in ("distinct_1", "distinct_2", "self_bleu",
                "response_quality", "token_entropy"):
        a = float(res["own"][key])
        b = float(res["shared"][key])
        assert abs(a - b) <= 1e-9, \
            f"{key} 口径不一致：GPT-2 自己算 {a}，共享实现算 {b}"
    print("[quality] 口径交叉验证：5 个指标与 GPT-2 线逐值相等")
else:
    print("[quality] 跳过口径交叉验证（找不到 GPT-2 线的 evaluate.py）")

print("test_quality OK  4 个指标手算一致 + 探针可复现 + 口径与 GPT-2 线相等")

# qwen3-pytorch

从零手写 tiny Qwen3：RoPE + RMSNorm + QK-Norm + SwiGLU + GQA，外加 MoE 变体，
走通 预训练 → LoRA → DPO 三段。

设计文档：[../qwen3-pytorch设计.md](../qwen3-pytorch设计.md) ｜ 开发计划：[../qwen3-pytorch开发计划.md](../qwen3-pytorch开发计划.md)

## 快速开始

```bash
python tests/run_all.py            # 全部组件 smoke
python data/build_vocab.py         # 训练 byte-level BPE，生成 vocab.json / merges.txt
python train/train_pretrain.py --model dense
python train/train_pretrain.py --model moe
python train/train_lora.py
python train/train_dpo.py
python evaluate.py --stage all
python ablations/compare_with_gpt2.py   # 与 GPT-2 的退化对照（feat-017）
python ablations/compare_quality.py     # 与 GPT-2 的生成质量并排对照（feat-026）
```

## 概念 → 笔记 → 源码

| 概念 | 笔记 | 源码 |
|---|---|---|
| RMSNorm | [RMSNorm_学习笔记](../Qwen3_RMSNorm_学习笔记.md) | `model/common/rmsnorm.py` |
| RoPE | [RoPE_学习笔记](../Qwen3_RoPE_学习笔记.md) | `model/common/rope.py` |
| QK-Norm | [QK-Norm_学习笔记](../Qwen3_QK-Norm_学习笔记.md) | `model/common/attention.py` |
| GQA / KV Cache | [GQA与KV_Cache](../Qwen3_GQA与KV_Cache_完整学习笔记.md) | `model/common/attention.py`、`model/common/kv_cache.py` |
| SwiGLU | [SwiGLU_MLP_学习笔记](../Qwen3_SwiGLU_MLP_学习笔记.md) | `model/common/mlp.py` |
| MoE | [MoE_学习笔记](../Qwen3_MoE_学习笔记.md) | `model/qwen3_moe.py` |
| byte-level BPE | [BPE_学习笔记](../Qwen3_BPE_学习笔记.md) | `data/bpe.py`、`data/tokenizer.py` |
| LoRA / DPO | [LoRA与DPO_学习笔记](../Qwen3_LoRA与DPO_学习笔记.md) | `model/common/lora.py`、`train/train_lora.py`、`train/train_dpo.py` |
| 训练 → 推理 | [训练推理完整学习笔记](../Qwen3_训练推理完整学习笔记.md) | `train/`、`inference.py` |
| 与 GPT-2 的退化对照 | [训练推理完整学习笔记 §12](../Qwen3_训练推理完整学习笔记.md) | `ablations/compare_with_gpt2.py` |
| 生成质量评估 | [训练推理完整学习笔记 §13](../Qwen3_训练推理完整学习笔记.md) | `quality.py`、`data/quality_prompts.py`、`ablations/compare_quality.py` |

> **每篇笔记开头都有「对应 GPT-2 实现位置 / 对应 Qwen3 源码位置 / 对应可跑测试」三行**，
> 链接的可点开性由 `python tests/test_note_refs.py` 自动校验（链接指向的文件不存在 ⇒ 测试直接红）。

### 笔记里的数字 ↔ config.py：自动守卫（feat-009）

```bash
python tests/test_config_sync.py
```

把两条线的**笔记**（17 篇 GPT-2 + 9 篇 Qwen3 + `qwen3-pytorch设计.md` = 27 篇）
里形如 `` `超参名 = 值` `` 的写法抽出来，
跟**各自**的 `config.py`（直接 import，不是正则猜）逐条对齐。
对不上 ⇒ 测试直接红，并给出「笔记写的 / config 实际值」。

**不是本仓库配置**的取值（教学举例、官方模型参数、算法定义表、退化开关值……）
必须在本文件 `EXCEPTIONS` 登记簿里**写明理由分类**，否则也算红。
登记簿里「已不再命中」的条目会被自动统计出来，提醒删掉。

> `../qwen3-pytorch开发计划.md` **不进自动守卫** —— 它是分层工程文档
> （含初稿旧数字、各 Task 的 Expected 输出、对历史错误的复盘引用），
> 改为**每次重写该 Task 时人工核对**。

实测：`命中超参断言 88 处 / 例外登记 25 条（全部命中，过时 0）/ 跨线或推导量跳过 99 处`。

### 笔记里的**派生量** ↔ 代码实测：自动守卫（feat-028）

```bash
python tests/test_derived_sync.py
```

`test_config_sync.py` 守的是「写成 `键 = 值` 的**超参主张**」，
但笔记里还有一类**派生量**：它们**不是 config 里的值，而是算出来的** ——
参数量（794,496 / 985,856 / 8,704 / 1,150,464 …）、词表大小（45 / 778）、
单层 FFN 参数量（SwiGLU 147,456 vs GPT-2 GELU 131,072）及其比值（1.125 / 1.21x）。
config 守卫**看不见它们**，而它们恰恰是**最容易漂移**的：
改任何一个 `n_embd` / `n_layer` / `vocab_size`，几十个参数量数字会**同时**变成错的。

★ 这个守卫**不手写期望值，而是现场把模型搭出来数参数**：
GPT-2 按 `config.py` 造模型数 `parameters()`；Qwen3 造 dense / MoE / 注入 LoRA 后数；
词表大小直接走两条线**自己的 tokenizer**（不数行 —— 词表里有「空格」这种行，`strip()` 会数掉）。

| 规则 | 查什么 |
|---|---|
| R1 | 算式自洽：「45 × 128 = 5,760」必须算得对（含链式等号） |
| R2a | 紧邻式参数量：`参数量 = 794,496` / `11520 个参数` |
| R2b | **行级**：这一行只要出现参数关键词，行上**每个**参数化数字都要对得上（表格里的数字也跑不掉） |
| R3 | 词表大小 = 实测词表大小 |
| R4 | `8,704 / 994,560 (0.88%)` 三元组：分子分母是实测值，百分比真的算得出来 |
| R5 | `比值 = 1.21x` 真的等于该行两个数字之比 |
| R6 | `可训练总计: A + B + C = N` 里的 N = 模型实测总参数量 |

**归属判定**：每个数字先按它前面最近的关键词判（`GPT-2` → GPT-2 侧，
`Qwen3 / SwiGLU / dense / moe / LoRA` → Qwen3 侧），判不出来按**文件所在目录**判。
⇒ 「GPT-2 的 794,496」和「Qwen3 的 794,496」会被分别对待。

> **为什么不做「散文里的『128 维』『4 层』」那种守卫**：量过了，不可行 ——
> 「层」在本仓语义冲突（`init.sh` 的 **L4 层** 18 处）、「维」的例子大多是教学玩具值或官方模型值。
> 详见守卫文件头部的「为什么不用正则抓散文」一节。

实测：`核对笔记 27 篇 / 派生量主张 41 处 / 例外登记 8 条（全部命中，过时 0）`。

## 与 GPT-2 线的对照

**主线代码（`model/` 与 `train/`）不 import GPT-2 线的任何代码** ——
两条线各自独立，避免「抄一份 GPT-2 的 config 再改」。

唯一的例外是 **`ablations/` 目录**，它是**单向**引用 GPT-2 线的
（Qwen3 → GPT-2），且只为了做对照实验：

| 脚本 | 引用方式 | 为什么这么写 |
|---|---|---|
| `ablations/compare_with_gpt2.py` | `importlib.util.spec_from_file_location` 按**文件路径**加载 GPT-2 的 `model/gpt2.py`，模块名取 `gpt2_ref` | 本仓的 `model` 包名**已被 Qwen3 线占用**（`qwen3-pytorch/model/`）。若走 `sys.path.insert` + `from model.gpt2 import GPT2`，`import model.gpt2` 会解析到 Qwen3 的 `model/` 里 ⇒ `ImportError` |

另外，为对照实验保留了相同接口：
`data/pretrain_data.py::make_pretrain_loader` 与 `data/preference_data.py::generate_preference_dataset`
的签名与 GPT-2 线一致。

### 退化对照：把「Qwen3 改了什么」变成一个数字（feat-017）

给 Qwen3 加 5 个**退化开关**（默认值 = 主线行为 ⇒ 不改变任何既有数字），
把它们全关掉就退化成 GPT-2 的结构；再把 GPT-2 的权重按显式映射表搬进去，比同一批输入的 logits。

```bash
python ablations/compare_with_gpt2.py
```

| 配置 | max_abs_diff | 相对差 | 结论 |
|---|---|---|---|
| 全关（退化成 GPT-2） | **0.000e+00** | 0 | **逐位相同** —— 退化是真的退化成 GPT-2，映射表没漏项 |
| 只开 `use_gqa` | 7.514e-02 | 1.01e-01 | K/V head 数减半 ⇒ 注意力看到的 K/V 变了 |
| 只开 `use_rope` | 1.541e+00 | 2.08e+00 | **行为改变最大** —— 位置信息的表达方式换掉了 |
| 只开 `use_qk_norm` | 9.334e-03 | 1.26e-02 | **最小** —— 只缩放 Q/K 的尺度，不改信息量 |
| 只开 `mlp_type=swiglu` | 7.187e-02 | 9.68e-02 | 多一个门控分支（GPT-2 无对应物） |
| 只开 `norm_type=rmsnorm` | 5.449e-01 | 7.34e-01 | 减不减均值、有没有 bias —— 对照实验必须两边同步切 |

三个**前提**（少一个就永远对不齐，详见笔记 §12.2）：
① `head_dim` 回退成 `n_embd // n_head`；② 中间层回退成 GPT-2 的 `n_inner = 4·n_embd`；
③ 归一化两边同步切到 LayerNorm 且 eps 一样（`1e-5`）。

结论与解读见 [`Qwen3_训练推理完整学习笔记.md` §12](../Qwen3_训练推理完整学习笔记.md)。

### 评估指标：与 GPT-2 线 9 项评估的差异（feat-025 的验收项 + feat-026 补齐生成质量）

GPT-2 线的 `evaluate.py` 有 9 个评估函数（8 个 `--stage` + `diversity` 里打包的 4 个），
整套是围绕 **Reward Model + PPO** 设计的。
Qwen3 线走的是 **LoRA + DPO**：没有显式奖励模型、没有 PPO、没有采样。
⇒ **两条线的评估不做逐项对标**，但 **pretrain 与 speedup 两项保留与 GPT-2 线逐字同名的列**。

| GPT-2 线 | GPT-2 的评估函数 | Qwen3 线 | 说明 |
|---|---|---|---|
| `pretrain` | `eval_perplexity` | ✅ `pretrain` | **同名列**：`stage / metric / value / loss / perplexity / n_tokens / checkpoint`；口径也相同（val set 的 token 加权平均 cross_entropy） |
| `speedup` | `eval_speedup` | ✅ `speedup` | **同名列**：`stage / metric / value / naive_avg_ms / kv_avg_ms / speedup / naive_tokens / kv_tokens / n_runs / max_new_tokens` |
| `sft` | `eval_sft` | 🔁 `lora` | 对应物：LoRA 前后偏好胜率对比（`baseline → win_rate`）。列名不通用（SFT 与 LoRA 是不同的训练方式） |
| `reward` | `eval_pairwise_accuracy`（RM） | 🔁 `dpo` | **本线没有 RM**。对应物是 DPO 的 **margin 判对率** `accuracy`（`margin > 0` 的比例） |
| `ppo` | `eval_ppo`（reward / KL） | 🔁 `dpo` | **本线没有 PPO**。对应物是 DPO 的**隐式奖励 margin**（`r_θ(y) = β(logπ_θ − logπ_ref)`） |
| `compare` | `eval_sft_vs_ppo` | 🔁 `lora` | **本线没有 PPO**。对应物是 LoRA 段里的「注入前（= 预训练模型）→ 训练后」胜率对比 |
| `benchmark` | `eval_benchmark`（QA 准确率） | ❌ 无 | 本线语料是**合成语料**，没有 QA benchmark 可做 |
| `diversity` | `eval_distinct_n` / `eval_self_bleu` / `eval_token_entropy` | ✅ `quality` | **同名同口径**（feat-026）：实现放在 `quality.py`，两条线**共用同一份代码**；`ablations/compare_quality.py` 用 GPT-2 线自己的 `evaluate.py` 在同一批回答上重算，**逐值相等**（实测差 0.000e+00） |
| `response_quality` | `eval_response_quality` | ✅ `quality` | **同名同口径**（feat-026）：4 条规则各 0.25（长度 10~50 / 含动物名 / 含动词 / 以 `.!?` 结尾） |
| — | — | ➕ `moe` | **Qwen3 独有**：专家负载分布 + `aux_loss`（GPT-2 线没有 MoE） |

**结论一句话**：GPT-2 线的评估是「**生成质量 + RLHF 三件套**」；
Qwen3 线的评估是「**训练质量（loss/ppl）+ 偏好对齐（margin/胜率）+ 推理效率（speedup）+ 生成质量**」。
两条线能在 **pretrain**、**speedup**、**生成质量** 三处并排比数字 ——
但生成质量那一处**带着下面三条不可比性**，别把它当「谁更好」的结论。

### 生成质量（`quality` 段）的三条不可比性（feat-026）

并排数字本身是可信的（口径已被交叉验证），但**结论要打折**：

| 不可比性 | 具体表现 | 怎么读 |
|---|---|---|
| **探针的域内外** | 动物探针对 GPT-2 是**域内**（它的预训练语料就是动物童话），对 Qwen3 是**域外**（Qwen3 训在 transformer 主题语料上） | 表 1 只说明「同一把尺子下的绝对水平」，**不是**「架构谁更好」 |
| **tokenizer 粒度** | `response_quality` 的长度规则（10~50 **字符**）按字符级 tokenizer 标定；Qwen3 用 BPE，回答普遍更长 ⇒ 这一条规则对 Qwen3 系统性不利 | 同时看 `response_quality_ex_length`（扣掉长度规则）与每条规则的命中率 |
| **词表大小** | `token_entropy` 的绝对值依赖词表大小（均匀分布时 H = log V；GPT-2 ~64 字符 vs Qwen3 778 子词） | 跨线只看 `token_entropy_normalized` = H / log(V) |

跑一遍对照：`python ablations/compare_quality.py`（会同时打**两对**模型：微调对 + 底模对，
并打印口径交叉验证表 —— 不相等直接报错退出）。

### 评估脚本的「不删/不改弱门槛」

`evaluate.py` 只设**可自证的门槛**，达不到就**报错退出**而不是打一个好看的数：

| 段 | 门槛 |
|---|---|
| `pretrain` / `moe` | `perplexity == exp(loss)` 自洽；`n_tokens > 0`；`aux_loss` 有限；专家负载之和 == `B*T` |
| `lora` | 胜率 ∈ [0, 1] |
| `dpo` | checkpoint 的 LoRA `B` 必须非零（B 全零 ⇒ DPO 没训过）；`margin_std >= 0`；准确率 ∈ [0, 1] |
| `speedup` | **有/无 KV Cache 的 greedy 输出必须逐位相同**，不一致直接报错退出 |
| `quality` | 4 个指标全部 ∈ [0, 1]；`token_entropy` 有限且 > 0；探针数 > 0（都是**自洽性**门槛，与模型好坏无关） |

checkpoint 缺失时**跳过该段并打印原因**（那是「还没跑到那一步」，不是「指标不合格」）。

## 硬约束

- 依赖上限：`torch` / `numpy` / `nltk`，不引入 pytest / peft / transformers / sentencepiece
- 超参全部来自 `config.py`
- `checkpoints/` 不入库

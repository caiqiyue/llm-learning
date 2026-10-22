# Session Handoff — llm-learning

> **本卡用途**：把项目交给「下一阶段（Session 006+）」的接手者，主题是 **P2 — Qwen3 从零实现**。
> **context 用满（≥70%）或长 session 结束前必须更新本卡**（见 `AGENTS.md` §9）。
> 规则：用指针代替贴全文；不带密钥；一次 session 只关心当前任务。

---

## 一句话现状

GPT-2 线**全链路已复验通过**（torch 环境已修好，训练 + 9 项评估全绿）；
Qwen3 线的**组件层 + 数据层 + 预训练 + LoRA + DPO + KV Cache 推理已从零手写完并真跑通**
（RMSNorm / RoPE / KVCache / QK-Norm+GQA 注意力 / SwiGLU / Qwen3Dense / Qwen3MoE / byte-level BPE /
语料与偏好数据 / `--model dense|moe` 预训练 / 手写 LoRA 微调 / 手写 DPO / prefill-decode 推理与采样），
**精简评估脚本已跑通**（pretrain / moe / lora / dpo / speedup 五段），
**与 GPT-2 的退化对照实验已跑通**（全关时 `max_abs_diff = 0` 逐位相同，5 个组件各自的影响量已量化），
**9 篇笔记的源码交叉引用已补齐**（BPE 笔记新建 + 每篇开头三行引用块 + 一个可运行的链接校验测试），
**笔记数字 ↔ `config.py` 的全量核对也做完了**（真错 2 处已改笔记 + 一个可运行的数字一致性守卫）。
**25 个 feature 全部 passing**，只剩收班。

---

## 当前阶段（P2：Qwen3 从零实现）

**范围**：从零手写 tiny Qwen3 —— RoPE + RMSNorm + QK-Norm + SwiGLU + GQA（后期 MoE）+ byte-level BPE，
然后走 **预训练 → LoRA 微调 → DPO**（路线已从 SFT/Reward Model/PPO 切到 LoRA+DPO，见 `feature_list.json` 的 `phase`）。
**不做什么**：不引入 `transformers` / `peft` / `accelerate` / `sentencepiece` / `tokenizers`，不跑真 Qwen3-0.6B。

**核心方法论 —— 组件全部做成可关掉的开关**（feat-017 的退化对照实验，**已跑通**）：

```
use_gqa      = False   → 退化为 MHA（n_head_kv = n_head）
use_rope     = False   → 退化为加法式 sinusoidal wpe（不是「无位置编码」！）
use_qk_norm  = False   → 退化为裸注意力（Q/K 不做 RMSNorm）
mlp_type     = "gelu"  → 退化为 GPT-2 的两矩阵 FFN（Linear → GELU → Linear）
norm_type    = "layernorm" → 退化为 nn.LayerNorm（两边必须同步切，eps 也要一样）
```

**开关的默认值 = 主线行为 ⇒ 加开关不改变任何既有训练 / 评估数字**（已由
`tests/test_compare_with_gpt2.py` 第 4 组断言守住）。

退化后的 Qwen3 与 GPT-2 在同输入下对齐（`max_abs_diff < 1e-4`，fp32）——
**实测 `max_abs_diff = 0.000e+00`（逐位相同）**。
这把「Qwen3 到底改了什么、为什么改」从**文字断言**变成**可重跑的实验**（`feat-017`）。

**★ 但对齐要满足三个「前提」**（少一个就永远对不齐，详见笔记 §12.2）：
① `head_dim` 回退成 `n_embd // n_head`（Qwen3 主线故意解耦，`4×48 ≠ 128` ⇒ **形状都映射不了**）；
② 中间层回退成 GPT-2 的 `n_inner = 4·n_embd`（512，不是 8/3 规则的 384）；
③ 归一化两边同步切到 LayerNorm，且 **eps 一样**（1e-5，不是 Qwen3 的 `rms_eps=1e-6`）。

---

## Verified Now（哪些是已验证的）

- `bash init.sh` → **PASS=28 SKIP=0 FAIL=0**（2026-10-23，L4 已有八条：组件 smoke / 前向 smoke / **训练步 smoke** / **LoRA smoke** / **DPO smoke** / **推理 smoke** / **评估 smoke** / **退化对照 smoke**）
- `cd qwen3-learning/qwen3-pytorch && python tests/run_all.py` → **PASS=16 FAIL=0 TOTAL=16**
- Qwen3 前向 smoke 已用**真实 BPE 词表**（vocab 778）跑通：`logits (2, 8, 778) loss 6.6897 params 985,856`
- Qwen3 训练步 smoke：`[train-step] loss 5.4788 grads 46 个 lr0 6.00e-06 lr_end 3.00e-05`
- **预训练真跑通**（CPU，脚本在 `train/train_pretrain.py`，超参全从 `config.py` 读）：
  - `--model dense`：`params = 985,856`，`initial_loss=6.7154 → val_loss=1.3510 val_ppl=3.86`，46.8s
  - `--model moe`：`params = 1,184,512`，`initial_loss=6.6954 → val_loss=1.3607 val_ppl=3.90`，59.1s，
    `aux_loss 1.0253 → 0.9982`，`expert_load` 从 ±35% 偏差被拉平到 ±3.5%
- BPE 端到端：语料 20000 句 / 785065 字节 → 词表 778（521 次合并）→ **压缩比 7.10 字节/token**
- 语料同源已证明：`data/corpus.py` 是唯一真源，重建词表后 `vocab.json` / `merges.txt` **逐字节一致**
- **LoRA 真跑通**（CPU，`train/train_lora.py`）：
  - `trainable params = 8,704 / 994,560 (0.88%)`（基座 985,856 全冻结，只训 A/B）
  - `BEFORE win_rate=0.515 win_rate_per_token=0.400 mean_logp(chosen)=-603.26`
    → `AFTER win_rate=0.605 win_rate_per_token=0.745 mean_logp(chosen)=-385.99`
  - ⚠️ 这组数字是**修完 dropout 覆盖 bug 之后**重测的（修复前是 0.595 / 0.745 / -389.48），见下面的「静默 bug」条目
  - 「注入即无损」实测成立：注入前后 logits `torch.allclose(atol=1e-6)`
  - `tests/test_lora.py` 9 组断言全过 → `test_lora OK  replaced=8 r=4 alpha=8 scaling=2.0`
- **DPO 真跑通**（CPU，`train/train_dpo.py`，不训 RM、不做 PPO、不采样）：
  - `π_ref = SFT 模型`：靠 **merge → strip → 重新 inject** 三步在同一模型对象上变出来
  - `trainable params = 8,704 / 994,560 (0.88%)`（只训 A/B）
  - `BEFORE margin=+0.0000 ± 0.0000 dpo_acc=0.000 win_rate=0.605`
    → `AFTER margin=+4.6512 ± 4.3952 dpo_acc=0.780 win_rate=0.665 win_rate_per_token=0.900`
  - **不变量**：B 零初始化 ⇒ `π_θ ≡ π_ref` ⇒ 初始 margin 恒为 0、loss 恒为 `log 2 = 0.6931`
    （被守了三次：训练脚本 assert / `tests/test_dpo.py` 第 3 组 / `init.sh` 的 DPO smoke）
  - `tests/test_dpo.py` 9 组断言全过 → `test_dpo OK  beta=0.1 loss=0.6994 margin=-0.0124 dpo_acc=0.750`
- **★ DPO 消融（步数 / lr / β / π_ref 取法）**：

| π_ref | 步数 | lr | β | reward margin | dpo_acc | 绝对 win_rate | 翻转数 |
|---|---|---|---|---|---|---|---|
| sft | 300 | 0.0001 | 0.1 | +0.0000 → +0.2831 | 0.000 → 0.945 | 0.605 → 0.605 | 0 |
| sft | 1500 | 0.0001 | 0.1 | +0.0000 → +1.6338 | 0.000 → 0.710 | 0.605 → 0.630 | 5 |
| sft | 1500 | 0.001 | 0.1 | +0.0000 → +4.6512 | 0.000 → 0.780 | 0.605 → 0.665 | 12 |
| sft | 1500 | 0.001 | 1.0 | +0.0000 → +8.4384 | 0.000 → 0.950 | 0.605 → 0.625 | 4 |
| sft | 1500 | 0.001 | 0.01 | +0.0000 → +1.0109 | 0.000 → 0.710 | 0.605 → 0.715 | 22 |
| pretrained | 1500 | 0.001 | 0.1 | +0.0000 → +8.4640 | 0.000 → 0.930 | 0.515 → 0.650 | 27 |
| sft | 1500 | 0.0003 | 0.1 | +0.0000 → +4.1526 | 0.000 → 0.715 | 0.605 → 0.665 | 12 |
| sft | 3000 | 0.001 | 0.1 | +0.0000 → +5.0101 | 0.000 → 0.920 | 0.605 → 0.665 | 12 |

- **KV Cache 推理 + 采样真跑通**（`qwen3-learning/qwen3-pytorch/inference.py`）：
  - `sample_from_logits` / `generate_naive` / `generate_with_cache` / `benchmark_speed` /
    `kv_cache_bytes` / `kv_cache_report`；`benchmark_speed` 的**列名与 GPT-2 线逐字相同**
  - **有/无 KV Cache 的 greedy 输出逐位相同**（token 序列 `torch.equal` == True；
    逐步 logits 12 步 `max|diff| = 5.96e-07`）
  - **decode 不施加因果 mask 的反向实验**：prefill 后清零「除最后一个之外」的历史 K/V，
    decode 输出**必须改变**（实测变了）
  - **与 GPT-2 线的 speedup 横向对比**（同 prompt 长度 64、同生成 64、同一台 CPU）：
    GPT-2 **1.748x** vs Qwen3 **1.999x** —— Qwen3 绝对更慢（327 vs 208 ms）但加速比更高
    （GQA 让 cache 读取量减半）
  - **KV Cache 占用**（T=128，fp32）：GQA **393,216 B** vs 同尺寸 MHA **786,432 B** = **1/2**
  - `tests/test_inference.py` 12 组断言全过
- **精简评估脚本真跑通**（`qwen3-learning/qwen3-pytorch/evaluate.py`，5 段）：
  - `pretrain` / `moe` / `lora` / `dpo` / `speedup`，`--stage all` 一次跑完
  - **评估数字与训练日志完全一致**：val_loss 1.3510、LoRA 胜率 0.515→0.605、
    DPO margin +4.6512 / dpo_acc 0.780 / 绝对胜率 0.665
    （**做法：评估函数直接复用训练脚本的实现**，不重写）
  - **pretrain 同时给两种 loss 口径**：token 加权（与 GPT-2 线同口径，可横向比）
    与 batch 均值（与训练日志的 val_loss 同口径）—— 1.3485 vs 1.3510
  - **MoE 段**：aux_loss 0.9978（均衡值 = 1 ⇒ router 没塌）、专家负载
    [0.2439, 0.2395, 0.2437, 0.2729]，**max/min = 1.140**
  - **speedup 段**：1.770x（naive 284.57ms / kv 160.82ms），有/无 KV Cache 的输出**逐位相同**
  - **门槛分两层**（自洽性写在 `eval_*()`、方向性写在 `check_directional()`），
    **两层都实测过会触发**（第 5 组 / 第 5b 组）
  - `tests/test_evaluate.py` 7 组断言全过
- **消融结论**：低秩约束 + 注入点决定「能学到什么」——
  全参微调 loss ~1.06 vs r=4 LoRA loss ~5.7（**内容型任务上 LoRA 明显吃力**，
  LoRA 的主场是风格/格式型任务）
- **偏好指标的长度偏置已量化**：`too_short` 策略（rejected 6.3 token vs chosen 72 token）
  的总 logprob 胜率**恒为 0.000**，把整体胜率天花板压到 ~0.75
  ⇒ **只看整体胜率会误判「模型没进步」**；这也正是 DPO 减 π_ref 的原因
- **★ 与 GPT-2 的退化对照真跑通**（`qwen3-learning/qwen3-pytorch/ablations/compare_with_gpt2.py`）：
  - 做法是**反过来做** —— 给 Qwen3 加 5 个退化开关（默认值 = 主线行为），
    全关掉就退化成 GPT-2 的结构，再把 GPT-2 的权重按显式映射表搬进去，比 logits
  - **全关那一行 = `max_abs_diff = 0.000e+00`（逐位相同，比 1e-4 阈值还严）**
    ⇒ 退化是真的退化成 GPT-2、映射表没漏项也没多项
  - 逐组件消融（每行只开一个）：`use_rope` **1.541e+00**（相对差 2.08，**最大**）＞
    `norm=rmsnorm` 5.449e-01（0.73）＞ `use_gqa` 7.514e-02（1.01e-1）＞
    `mlp=swiglu` 7.187e-02（9.68e-2）＞ `use_qk_norm` 9.334e-03（**1.26e-2，最小** ——
    它只缩放 Q/K 的尺度、不改信息量）
  - **三条反证**（防「碰巧通过」）：A 不加载权重 ⇒ `max_abs_diff = 1.013`（差得远）；
    B 映射表逐条 `torch.equal`；C 主线默认配置不变（无 `wpe` / 有 `q_norm` / SwiGLU / RMSNorm）
  - `ablations/` 是本仓**唯一**跨目录引用 GPT-2 线的地方（单向 Qwen3 → GPT-2）；
    主线 `model/` 与 `train/` 依然不 import GPT-2 线任何代码
  - 结论写进 `Qwen3_训练推理完整学习笔记.md` §12 + 5 篇组件笔记各一行「量化实证」指针
- **★ 笔记 ↔ 源码交叉引用真跑通**（feat-008）：
  - **9 篇 Qwen3 笔记**开头各加三行引用块（放在 H1 之后，恰好三行）：
    「对应 GPT-2 实现位置」/「对应 Qwen3 源码位置」/「对应可跑测试」——
    **链接文字是仓库根相对路径，链接目标是相对当前文件的路径**（两者都要，缺一不可）
  - **诚实标注「主线没有对应实现」**：GPT-2 线的代码里**没有 BPE**（字符级 `CharTokenizer`，
    vocab 45，中文全 UNK）、**没有 MoE**、**没有 LoRA / DPO**
  - **补齐唯一缺过的笔记 `Qwen3_BPE_学习笔记.md`**（BPE 代码 feat-022 落地，笔记一直欠着）
  - 实测：词表 **778** / 合并 **521** / 压缩比 **7.10 字节/token**；
    单句 `"The student trained a tiny transformer."` 39 B → **6 tokens**
    （GPT-2 线 `CharTokenizer` 同一句 **39 token**）
  - **可运行的交叉引用校验** `tests/test_note_refs.py`：
    `[refs] 笔记 9 篇 / 交叉引用链接 78 条全部可点开 / 正文路径 107 处全部存在`
    ⇒ 链接指向的文件不存在、某篇笔记漏了引用块、根 README 漏了某篇笔记，**测试直接红**
  - 根 `README.md` 的 Qwen3 索引表补「Qwen3 源码」列并覆盖全部 9 篇；
    同时把根 README 里几处**过期描述**改成现状（目录树 / 阶段表 / 代码结构 / torch 环境 / 核心思路）
- GPT-2 线：`python evaluate.py --stage all` 退出码 0，9 项评估全过
- 各 feature 的详细证据在 `feature_list.json` 的 `evidence` 数组里（feat-011 ~ feat-025 都有）

### feat-009 — 笔记数字 ↔ `config.py` 全量核对（Session 012）

- **真错只有 2 处，都改了笔记（config 才是真值）**：
  `gpt-learning/0.GPT全流程详解.md` §2 写 `warmup_iters = 200` / `lr_decay_iters = 2000`，
  GPT-2 的 `config.py` 实际是 **`50` / `200`**。顺带修掉了同一段**抄错的 `get_lr` 源码**
  （笔记写 `learning_rate * it / warmup_iters`，仓库实际是
  `learning_rate * (it + 1) / max(1, config.warmup_iters)`）。
- **待核项 ① 设计.md §4 `intermediate_size=384` → 不是问题，不改**：
  原文已写明「`128 × 8/3 = 341.3` → 对齐到 128 的倍数 ⇒ 384 = 3×n_embd」+
  「8/3 规则来自 LLaMA，Qwen3 实际用 3×」。官方 Qwen3-0.6B 是 1024 → 3072（3×），**一致**。
- **★ 待核项 ② 设计.md §5.1 RMSNorm smoke 判据「均值≈0、方差≈1」→ 是真错，已改笔记**：
  RMSNorm **不减均值**，「均值≈0」只是随机输入恰好对称时的巧合。
  改成「**平方均值归一化到 1**（RMS ≈ 1）」并指向 `tests/test_rmsnorm.py` 的可跑断言。
  （`开发计划.md` 同一处**早已是正确写法**，只有 `设计.md` 漏改 —— 「同一事实写在两处就会漂移」的活例子。）
- **★ 新增守卫 `tests/test_config_sync.py`**（`run_all.py` 自动发现 ⇒ `init.sh` L4 自动覆盖）：
  扫两条线 **27 篇**笔记（17 GPT-2 + 9 Qwen3 + `设计.md`），跳过围栏代码块，
  抽 `` `超参名 = 值` `` 跟**各自**的 `config.py`（直接 import，不靠正则猜）逐条比对。
  实测：`命中超参断言 88 处 / 例外登记 25 条（命中 25，过时 0）/ 跳过 99 处`。
  **不属于本仓库配置的取值必须登记理由**（7 类：ALGO / SWITCH / TEACHING / UPSTREAM / SMOKE / PHASE / SIGNATURE）。
- **★ 守卫上线当天就抓到了我自己**：新写的计划章节引用了「GPT-2 的旧值 `warmup_iters = 200`」，
  守卫按「目录 → config」规则拿 Qwen3 的 config 去比 ⇒ 误报。
  **这暴露了一个真盲点：跨线比较的文档不适用「目录决定 config」的规则**
  ⇒ `开发计划.md` 改为**人工核对**，盲点写进守卫注释（不是「把报错关掉」，是「换一种对的核对方式」）。
- **顺手复核的非 config 类数字全部一致**：GPT-2 可训练参数 **794,496**；
  MoE **951,808 / 1,150,464 / 1.21x**；LoRA **8,704 / 994,560 = 0.88%**；
  设计.md §12 的 6 行 `max_abs_diff`（复跑**逐位一致**）。
- **⚠️ 唯一例外：实测值（speedup）不能当超参审计** —— 笔记写 1.748x / 1.999x，
  同机复跑是 **1.6992x / 1.841x**。**结论不变，但小数点不可当常数** ⇒ 已给该表加免责说明。

## Broken Or Unverified（坏的 / 没验证的）

- **Qwen3 的评估侧只做了「精简 5 段」**：GPT-2 线那 9 项里的
  `diversity`（distinct-n / self-BLEU / token entropy）与 `response_quality` **本线未实现**，
  `benchmark` **本线没有**（合成语料没有 QA 题库）—— 差异已在
  `qwen3-learning/qwen3-pytorch/README.md` 逐项列出。
- **LoRA / DPO 的胜率都是「训练内指标」**：合成偏好数据只有 5 个 prompt × 5 个 chosen，训练集与评估集同分布 ——
  它证明的是「LoRA 确实让模型更偏向 chosen」，**不能当泛化能力**
- **BPE 词表只有 778 < 目标 1000**：合成语料的可合并对用尽（521 次后没有频次 ≥ 2 的对），属语料多样性问题，非实现 bug
- **MoE 在本仓库体现不出容量优势**：val_loss 1.3607 vs dense 1.3510（MoE 反而略差）——
  语料太简单 + 专家更窄（参数量只多 1.21×）+ 只训 300 步。**写笔记时别暗示「MoE 一定更好」**（前提条件见笔记 §25.1）
- **笔记缺口**：**已清零** —— `Qwen3_BPE_学习笔记.md` 已补齐（feat-008），
  `Qwen3_LoRA与DPO_学习笔记.md` 早前已补齐 LoRA 半部（§1–§7）与 DPO 半部（§8），§9 教训 7 条。
  `qwen3-learning/` 现有 **9 篇**笔记，**每篇开头都有三行源码交叉引用**（由 `tests/test_note_refs.py` 守住）
- **交叉引用只到「文件 + 类/函数」级，不到行号级** —— 行号会随代码改动漂移，写进笔记反而变成错误信息。
  另外**裸写的不带目录的文件名**（`config.py` / `evaluate.py`）**不在自动校验范围内**
  （机器判不准它指主线还是本线），这部分靠人工
- **feat-017 的结论只在 fp32 / 小尺寸下成立**（vocab=293、n_layer=4、n_embd=128、seq_len=16）：
  「全关逐位相同」**不能外推**到别的 dtype / 别的形状。
  且单开组件时新引入的参数（SwiGLU 的 gate）是**随机初值** ⇒ 那几行的 diff 适合比**量级**，
  不等于「这个组件贡献了多少准确率」
- **本仓库的 aux_loss 与 HF 数值不能直接比大小**：本实现 = HF Mixtral 同款公式的 1/K（见下面的陷阱表）
- **feat-009 守卫的三个已知盲区**（都写进了 `tests/test_config_sync.py` 的注释）：
  ① **跨线比较的文档没有自动守卫** —— `开发计划.md` 引用 GPT-2 的值时会误报，已移出自动范围、改为人工核对；
  ② **裸文件名不在校验范围** —— 笔记里不带目录的 `config.py` / `evaluate.py` 机器判不准指哪条线；
  ③ **只覆盖「写成 `键 = 值` 的主张」** —— 写成「128 维」「4 层」的散文数字仍靠人读
  （本次人工扫了关键派生量：参数量 / 消融表，但**没有自动化**）。

## Next Best Step（按顺序，不要跳）

| 顺序 | feature | 做什么 | 为什么是它 |
|------|---------|--------|--------|
| — | **无** | — | **25 个 feature 全部 passing**，P2 阶段（Qwen3 从零实现）已收班 |

**下一个 session 建议做的第一件事**（不在 `feature_list.json` 里，因为还没立项）：
读本卡的「Broken Or Unverified」段落，挑一条已知缺口**立项**成新的 feature。
目前最值得做的三条：
① **评估侧补齐生成质量指标**（`diversity` / `response_quality`）—— 唯一让两条线能真正并排的缺口；
② **feat-009 守卫的第 3 个盲点自动化**（散落在散文里的「128 维」「4 层」目前靠人读）；
③ **把 GPT-2 线的 `get_lr` 等实现细节也做成「笔记抄源码必须逐字一致」的守卫**
   （本次手抄的 `get_lr` 就抄错了，是靠人眼发现的）。

**这一步不许动的**：不改 GPT-2 线的任何训练代码或超参；不在 `qwen3-learning/qwen3-pytorch/` 之外 import；不削弱 `evaluate.py` 的任何指标与阈值。

---

## 已知的「文字断言 vs 可跑实验」陷阱（别再踩）

| 结论 | 关键点 |
|------|--------|
| QK-Norm 在 RoPE 前 vs 后 | **用均匀 gain 测不出差异**（max\|diff\| ≈ 3.6e-07）。`rotate_half` 把第 i 维与 i+half 维配对，逐维 gain 与旋转不对易。必须用**非均匀 gain** 才暴露（7.38 ~ 14.9） |
| RoPE 是相对位置编码 | 绝对位置**整体平移**不改变输出（max\|diff\| < 1e-5）；只有改变**相对距离**才改变输出 |
| SwiGLU vs GPT-2 4×-GELU 参数量 | 差 = `矩阵数 × H × I`。I 严格取 8/3·H 时**完全相等**；本仓库取整到 384 → 比值 9/8 = 1.125×。Qwen3-0.6B 实际用 3×hidden，8/3 是 LLaMA 的规则 |
| MoE aux_loss | Switch 论文版均匀时 = 1；**本仓库版均匀时也 = 1**（f 对 tokens×k 槽位求均值、Σf=1）；HF Mixtral 版把槽位计数除以 N（Σf=K），均匀时 = **K**（K=2 → 2）。**本实现 = HF 的 1/K**。⚠️ 曾把这条写成「1/K = 0.5」，已被预训练日志（aux≈1.0）与 `tests/test_moe.py` 第 11 节实测纠正 |
| BPE 的 id 分配 | 合并 token 从 **257** 开始（不是 256 —— 256 是字节 0xFF 的位置，撞车会把它静默吞掉） |
| 量纲类结论 | **必须用代码实测锚定**，不能公式心算后直接写进文档（aux_loss 就是这么错了一倍） |
| Windows 子进程输出编码 | bash 按 UTF-8 解码、Windows Python 默认 stdout 是 GBK → 中文乱码。证据命令加 `PYTHONIOENCODING=utf-8` |
| 脚本放在子目录 | Qwen3 线脚本在 `train/` / `tests/` 里，`sys.path[0]` 不是项目根 → **必须显式 `sys.path.insert(0, <项目根>)`** |
| **偏好胜率的长度偏置** | 总 logprob 口径偏袒**短回复**（`too_short` 策略胜率**恒为 0.000**，因为 rejected 只有 6.3 token），把整体胜率天花板压到 ~0.75；长度归一口径反过来偏袒「可预测」的文本。**必须按策略分组看，否则误判「没进步」** |
| **旁路注入的「结构性 bug」** | `merge_lora` 合并后分支还开着 → ΔW 算两遍；`inject_lora` 只冻被包装的那一个 Linear → embedding/Norm/MLP/lm_head 仍可训；loss 掩码把 prompt 也算进去。**三条都不报错、只静默给错结论 ⇒ 必须用「等价性断言」守，不能靠看 loss 曲线** |
| **DPO 的 π_ref 是「哪一个」模型** | 标准做法锚 **SFT 模型**。本仓库「关掉 LoRA 分支」得到的是**预训练模型** —— 锚错**不报错、loss 也降、指标照涨**，只是整个实验锚在错的基线上。解法：`merge_lora` → `strip_lora`（新增）→ 重新 `inject_lora`。**必须用「初始 margin 恒为 0 / loss 恒为 log 2」这条可运行断言来钉** |
| **DPO 目标指标 ≠ 业务指标** | 强配方 margin +4.65 / dpo_acc 0.78 / 绝对胜率 0.605→0.665；**弱配方 dpo_acc 冲到 0.945，绝对胜率却是 0/200 翻转**。**报告时必须同时给 margin（目标口径）与 win_rate（业务口径）**，只给前者是自我欺骗 |
| **覆盖参数没传到子模块 = 静默的半配置** | `Qwen3Dense(dropout=0.0)` 只关掉了 embedding 的 dropout，注意力/MLP 一直是 `cfg.dropout=0.1`；而 **p>0 的 Dropout 在 train 模式下会推进全局 torch RNG**⇒ 「训练前多做几次评估」就改变训练轨迹（同一配方 margin 差 0.028）。**每个覆盖开关都要配一条等价性断言（结构 + 行为各一条）** |
| **KV Cache 与 naive 的 logits 不会逐位相同，但 token 会** | 两条路径的 matmul 规约顺序不同（naive 一次前向 T_q = P+s，cache 每步 T_q = 1）⇒ fp32 末位差异 ~1e-7。**验证「KV Cache 改没改结果」要断言 token 序列，而不是断言 logits 逐位相等**；想更强就断言逐步 logits 的 `max|diff| < 1e-4` |
| **「decode 不 mask」要用反向实验证** | 正向断言（decode 输出对）证明不了「没被 mask」。正确做法：**污染历史 K/V ⇒ 输出必须改变** —— 若历史被屏蔽了，污染不会影响输出 |
| **`strict=False` 会让你「评估了一个随机模型」而不报错** | `lora.pt` / `dpo.pt` 的 state_dict 是**注入 LoRA 之后**的键名（`q_proj.base.weight` / `.lora_A` / `.lora_B`），与未注入的模型对不上。`load_state_dict(strict=False)` **只警告不抛异常** ⇒ 权重全部没加载。正确做法：**先 `inject_lora` 再 `load_state_dict(strict=True)`**，键名对不上就当场红 |
| **「同名」不等于「同口径」** | 同一个叫 `perplexity` 的字段，GPT-2 线是 **token 加权**、本线训练日志是 **batch 均值**（1.3485 vs 1.3510）。直接并排比就是在比两件不同的事。**两个口径都打印**才能既横向比又纵向对 |
| **LoRA ≠ 弱化版全参** | 低秩约束 + 注入点决定「能学到什么」：内容型任务上全参 loss ~1.06 vs r=4 LoRA ~5.7。**LoRA 是风格/格式型任务的主场** |
| **「全关 < 阈值」不等于实验成功** | 对照实验的基线（两边都退化成同一个东西）如果**两边根本没跑**，也会「通过」。**必须有反证**：不加载权重时必须差得很远（实测 1.013 vs 阈值 1e-4），且映射表要逐条 `torch.equal` 核对 |
| **退化开关缺一个「前提」就永远对不齐** | 对齐 GPT-2 需要三个前提同时满足：① `head_dim` 回退成 `n_embd // n_head`（Qwen3 主线故意解耦 ⇒ **形状都映射不了**，不是精度问题）；② 中间层回退成 `4·n_embd`；③ 归一化两边同步切且 **eps 一样**（1e-5 vs 1e-6 差 ~1%，会和「减不减均值」混在一起） |
| **`model` 包名会被占用 ⇒ 跨目录 import 必须按文件路径加载** | 本仓 `model` 已被 Qwen3 线占用，`sys.path.insert` + `from model.gpt2 import GPT2` 会解析到 Qwen3 的 `model/` ⇒ `ImportError`（加 path 也救不了，`model` 已进 `sys.modules`）。正确做法：`importlib.util.spec_from_file_location(path)`，模块名另取（如 `gpt2_ref`） |
| **「GPT-2 用 learned PE」是事实错误** | 本仓 GPT-2 的 `pos_emb` 是 `register_buffer(..., persistent=False)` 的**固定 sinusoidal 表** —— 不训练、**且不在 `state_dict` 里**。写对照实验时「关掉 RoPE = 不加位置信息」也是错的：必须补上**加法式** sinusoidal PE，否则两边连「有没有位置信息」都不同，根本不是同一个模型 |
| **对照实验的 diff ≠ 组件的「贡献」** | 单开某个组件时，新引入的参数（如 SwiGLU 的 gate）是**随机初值** ⇒ diff 是「结构差异 + 随机新参数」的合成效果。**适合比量级，不等于「这个组件贡献了多少准确率」**；且结论只在测过的 dtype / 尺寸下成立，不要外推 |
| **「链接可点开」要能自动发现它坏了** | 文档里的链接过期是**静默腐烂**：没人点就没人发现。正确做法是**一个可跑的测试**（`tests/test_note_refs.py`）：链接目标不存在 / 漏了引用块 / 索引表漏了某篇 ⇒ 测试直接红 |
| **查路径时必须先排除三类假阳性** | ① 含空格（`python tests/x.py`、`n_embd // n_head` 是命令/自然语言）；② 带盘符（`E:/tmp/x.py` 在仓库外）；③ 产物目录（`checkpoints/*.pt` 不入库，**「本地存在」≠「链接有效」**）。另外 `config.json` 指的是 **HuggingFace 模型的配置文件**（外部文件），不是仓库文件 |
| **「笔记里的数字」要自动守，不能靠人读** | 人工承诺守不住 —— `config.py` 改一个数字，几十篇笔记不会自动跟着变。正确做法是**一个可跑的守卫**（`tests/test_config_sync.py`）：把笔记里 `` `键 = 值` `` 抽出来跟 config 逐条比对，**改 config 忘了改笔记 ⇒ 测试直接红** |
| **审计数字先解决「定位」，别先核对** | v1 采集器（键名 + 附近有数字）捞出 **919** 处候选，人工根本看不动；收紧到「`` `键 = 值` `` + **跳过围栏代码块**」之后只剩 **110** 处才可审。**跳过围栏代码块是关键** —— 否则会把文档里贴的源码当成笔记主张 |
| **守卫必须允许「显式例外」，且例外要带原因** | 没有例外机制的守卫会逼着人把「教学举例」「官方模型参数」「算法定义表」也改成 config 值 —— 那是**把笔记改坏**。例外要**分类**（ALGO/TEACHING/UPSTREAM/SMOKE/SWITCH/PHASE/SIGNATURE），并能被自动统计「哪些已过时」 |
| **「目录 → config」的规则对跨线文档不成立** | `qwen3-learning/` 里的文档引用了 **GPT-2** 的值时，守卫会拿 Qwen3 的 config 去比 ⇒ 误报。**这不是 bug，是守卫的适用范围问题** —— 跨线比较的文档（如 `开发计划.md`）应**移出自动守卫、改为人工核对**，而不是把报错关掉 |
| **实测值不是超参** | 耗时 / 加速比随机器负载浮动（笔记写 1.748x / 1.999x，复跑 1.6992x / 1.841x）。**只能写「同机复跑」并注明波动范围，不能锁死小数点**；`evaluate.py` 对 speedup 的门槛也应该是**方向性**的（> 1）而不是固定阈值 |
| **「同一个事实写在两处就会漂移」** | 设计.md §5.1 的 RMSNorm 判据写「均值≈0」（错），而 `开发计划.md` 同一处**早已是正确写法**（「平方均值归一化到 1」）—— 两处只改一处，就出现矛盾。**判据类文字必须能指向一个可跑断言** |

---

## 需要用户拍板的决策（别自行决定）

| 决策 | 出现在 | 说明 |
|------|--------|------|
| ~~LoRA 的注入点与超参~~ | `feat-023` | **已定**：`lora_r=4 / lora_alpha=8 / lora_target_modules=("q_proj","v_proj")`；`lora_lr` 3e-4→**1e-3**、`lora_max_iters` 300→**1500**（消融实测依据） |
| ~~DPO 的 β 与训练步数~~ | `feat-024` | **已定**：`dpo_beta=0.1`、`dpo_lr` 1e-4 → **1e-3**、`dpo_max_iters` 300 → **1500**（消融实测依据，见上面的表） |
| DPO 偏好数据要不要引入真实数据集 | `feat-024` | 目前是合成数据，引入真实数据会动依赖上限（**未做**） |
| 评估要不要和 GPT-2 线的 9 项对齐 | `feat-025`（已决） | **不对齐**：只保留 pretrain / speedup 逐字同名列，其余 7 项要么有对应物（lora/dpo）要么本线没有（benchmark）要么未实现（diversity / response_quality），差异写在 README |

---

## Commands（常用命令）

```bash
# 启动 + 轻量验证（L0-L4）
bash init.sh
bash init.sh --quiet            # 只看汇总行

# GPT-2 线（代码目录已存在，全链路已复验）
cd gpt-learning/gpt-learning-pytorch
python data/build_vocab.py
python train_pretrain.py && python train_sft.py && python train_reward.py && python train_ppo.py
python evaluate.py --stage all

# Qwen3 线（组件 + 数据 + 预训练已就位）
cd qwen3-learning/qwen3-pytorch
python data/build_vocab.py       # 重建 BPE 词表（vocab.json / merges.txt）
python tests/run_all.py          # 17 个 smoke 汇总
python tests/test_pretrain.py    # 单个 smoke
python tests/test_lora.py        # LoRA 的 9 组断言
python train/train_pretrain.py --model dense   # 预训练（dense，~47s CPU）
python train/train_pretrain.py --model moe     # 预训练（MoE，~59s CPU）
python train/train_lora.py       # LoRA 微调（r=4 q/v, lr=1e-3, 1500 步）
python train/train_dpo.py        # DPO（π_ref = SFT 模型，merge → strip → 重新 inject）
python tests/test_dpo.py         # DPO 的 9 组断言
python tests/test_inference.py   # KV Cache 推理与采样（12 组断言）
python tests/test_evaluate.py    # 精简评估（7 组断言）
python ablations/compare_with_gpt2.py   # 与 GPT-2 的退化对照（feat-017，6 行消融表）
python tests/test_compare_with_gpt2.py # 退化对照的 6 组断言（含 3 条反证）
python tests/test_note_refs.py    # 笔记 ↔ 源码交叉引用校验（feat-008，链接全可点开）
python tests/test_config_sync.py  # 笔记数字 ↔ config.py 一致性校验（feat-009，88 处断言）

# 状态
cat feature_list.json            # 含 phase 阶段标记 + 每条的 evidence
cat claude-progress.md           # 上次做到哪
git log --oneline -5
```

## 相关文件速查

| 想做什么 | 看哪个 |
|----------|--------|
| 开工前必读 | `AGENTS.md` §0 framing / §3 硬约束 / §5 验证路径 |
| 当前阶段与任务顺序 | `feature_list.json` 的 `phase` + priority 最小的非 passing 项 |
| Qwen3 设计 / 实施计划 | `qwen3-learning/qwen3-pytorch设计.md`、`qwen3-learning/qwen3-pytorch开发计划.md`（Task 7–20 已就地纠错，含「实测修正记录」与「9 个真 bug」清单；Task 20 = feat-009 数字核对，含例外分类表与 4 条经验） |
| Qwen3 笔记原文 | `qwen3-learning/Qwen3_*.md`（**9 篇**，BPE 已补齐；每篇开头有 GPT-2 / Qwen3 源码交叉引用；训练推理篇含 §10 推理 + §11 评估 + **§12 退化对照**） |
| 笔记数字一致性守卫 | `qwen3-learning/qwen3-pytorch/tests/test_config_sync.py`（feat-009；27 篇笔记 / 88 处断言 / 25 条例外登记；改 `config.py` 忘了改笔记 ⇒ 红灯） |
| GPT-2 对照实现 | `gpt-learning/gpt-learning-pytorch/model/gpt2.py`（**只能按文件路径 import**，见陷阱表） |
| 对照实验脚本 | `qwen3-learning/qwen3-pytorch/ablations/compare_with_gpt2.py`（本仓唯一跨目录引用 GPT-2 线的地方，单向） |
| 收班自查 | `clean-state-checklist.md` |

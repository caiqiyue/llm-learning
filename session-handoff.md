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
**精简评估脚本已跑通**（pretrain / moe / lora / dpo / speedup / **quality** 六段），
**与 GPT-2 的退化对照实验已跑通**（全关时 `max_abs_diff = 0` 逐位相同，5 个组件各自的影响量已量化），
**生成质量已与 GPT-2 线并排**（4 个指标**共用同一份实现** `quality.py`，
并用 GPT-2 线**自己的** `evaluate.py` 做**口径交叉验证** ⇒ 5 个指标**逐值相等**，差 0.000e+00），
**9 篇笔记的源码交叉引用已补齐**（BPE 笔记新建 + 每篇开头三行引用块 + 一个可运行的链接校验测试），
**笔记数字 ↔ `config.py` 的全量核对也做完了**（真错 2 处已改笔记 + 一个可运行的数字一致性守卫），
**P1 线的 SFT 塌陷也修好了**（feat-027：三个互相咬合的缺陷 —— 训练目标与推理目标不一致 /
特殊 token 不可达 / 评估切片错位；修完 PPO 第一次真的在优化奖励），
**笔记里的「派生量」也纳入机器守卫了**（feat-028：参数量 / 词表大小 / 算式 / 比值 ↔ **代码实测**，
期望值不手写而是**现场把模型搭出来数参数**），
**笔记里手抄的代码也纳入机器守卫了**（feat-029：把笔记里那段代码**真的跑起来**，
和源码里那段**比输出** —— 不要求逐字一致，只要求**行为一致**），
**笔记里的 `#L` 行号锚点也纳入机器守卫了**（feat-030：行号**越界**要红、
单行锚点**指错行**也要红 —— 「链接点得开」≠「链接指得准」）。
**30 个 feature 全部 passing**（本卡「Next Best Step」给出下一批候选）。

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

- `bash init.sh` → **PASS=30 SKIP=0 FAIL=0**（2026-10-23，L4 已有九条：组件 smoke / 前向 smoke / **训练步 smoke** / **LoRA smoke** / **DPO smoke** / **推理 smoke** / **评估 smoke** / **生成质量 smoke** / **退化对照 smoke**；L1 另加「**P1 训练口径守卫（feat-027）**」、L2 另加「模型 loss 确实错位」断言）
- `cd qwen3-learning/qwen3-pytorch && python tests/run_all.py` → **PASS=20 FAIL=0 TOTAL=20**（feat-028 新增第 19 个 `test_derived_sync.py`；feat-029 新增第 20 个 `test_note_code_sync.py`；feat-030 **没有新增文件**——是给 `test_note_refs.py` 加了第 6 节锚点校验）
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
  （**feat-027 修完之后**：pretrain ppl 20.24 / sft distinct_1 0.0119 self_bleu 0.8579
  response_quality 0.3600 / ppo distinct_1 0.0159 response_quality 0.5150 /
  compare `sft_rm_score −2.7321` `ppo_rm_score +0.9188` `improvement +3.6509`）
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

### feat-027 — P1 线 SFT 塌陷的根因与修法（Session 014）

- **症状**：`sft.pt` / `ppo.pt` 对**任何** prompt 都输出同一串常量
  （公共后缀 47 字符 / 平均 48.0 字符 = **98%**）；`pretrain.pt` 正常（0%）。
  PPO 相对 SFT 零提升（`sft_rm_score = ppo_rm_score = 0.0696`，`improvement = 0.0000`）。
- **根因**：**训练目标 ≠ 推理目标**。`data/sft_data.py` 的 labels 与 input_ids **同下标**，
  而 `model/gpt2.py` 的 loss **不做错位** ⇒ SFT 教模型「输出**当前** token」（copy）
  ⇒ 推理时 prompt 以空格结尾 ⇒ 模型把最后一个 token 复制出来 ⇒ **无限吐空格**
  （实测 48 个 token 全是空格 id=5，第一步 top-1 logit 14.949 vs 第二名 'A' 3.237）。
  预训练线当时在**数据侧**左移、SFT 线没左移 ⇒ **两条数据线口径不一致**
  （比任何一条单独写错更危险：不报错，只是让模型悄悄学成另一个任务）。
- **另外两个互相咬合的缺陷**：
  ② `encode` 逐字符查表 ⇒ `"<sep>"` 变成 5 个字符 ⇒ **BOS/EOS/SEP 的 id 永远不可达**
  ⇒ 模型从没学过吐 EOS ⇒ 生成永远跑满 `max_new_tokens`；
  ③ 评估用**字符串长度**切解码后的文本，而 `UNK` 解码回来是 5 字符的 `"<unk>"`
  ⇒ 每条回答都被塞上 `"? <unk>sep<unk> "` 前缀 ⇒ 污染 `distinct_n` / `self_bleu` / `response_quality`。
- **修法（定下来的唯一口径）**：**错位只在模型里做一次**
  —— 两条数据线都保持「labels 与 input 同下标」，模型内部
  `F.cross_entropy(logits[:, :-1, :], targets[:, 1:])`；
  特殊 token 整体一个 id（`encode` 改「最长匹配优先」）；回答按 **token id 个数**切。
  **为什么不用「数据侧预左移」**：笔记 5/7 原本写的就是这一套（HF 标准形式），
  改代码迁就另一种口径 = 让笔记变成错的；而且错位只在一处，第三条数据线不容易跑偏。
- **修完的效果**（L3 全链重跑 + `evaluate.py --stage all`，退出码 0）：

| 指标 | 修之前 | 修之后 |
|---|---|---|
| 退化判据（公共后缀占比） | 47 / 48.0 = **98%** | 0 / 32.0 = **0%** |
| SFT `self_bleu` | 1.0000（恒定输出） | 0.8579 |
| SFT `token_entropy` | 0.0000 | 1.3094 |
| SFT `response_quality` | 0.2500 | 0.3600 |
| `sft_rm_score` | 0.0696 | **−2.7321** |
| `ppo_rm_score` | 0.0696 | **+0.9188** |
| PPO `improvement` | **0.0000** | **+3.6509** |
| PPO 训练中的 RM 走势 | 无（常量） | 前 5 次 −1.4…−3.7 → 后 5 次 **+1.3…+2.6** |

  ⇒ **塌陷不是「小模型学不会」，是训练目标写错了**；修完之后 PPO 第一次真的在优化奖励。
- **机器守卫**：`init.sh` L1「P1 训练口径守卫（feat-027）」+
  L2「模型 loss 必须等于手工算的 `CE(logits[:,:-1], targets[:,1:])`」
  ⇒ 改坏口径直接判红，不再靠人读代码发现。
- **受影响的笔记**：`gpt-learning/5.损失函数.md` §14、`gpt-learning/7.sft-train.md` §16、
  `gpt-learning/13.gpt2评估.md` §十八、`gpt-learning/gpt-learning-pytorch/README.md`
  「训练口径」小节、`qwen3-learning/Qwen3_训练推理完整学习笔记.md` §13.7。

### feat-028 — 笔记里的「派生量」也纳入机器守卫（Session 015）

- **由来**：feat-009 收班时写下的「守卫的第三个盲点 —— 散文里的『128 维』『4 层』」，
  在本卡的「Next Best Step ②」里被留成待办。**这次先把两种字面读法都实测了一遍。**
- **★ ②③ 的字面读法都不可行（量过了，别再试）**：
  ① **散文数字扫描**（正则抓「128 维」「4 层」）会变成**误报机器**：
    「层」在本仓**语义冲突** —— `init.sh` 的 **L4 层**（18 处）、「4 层 Transformer」、
    SwiGLU 的 3 个矩阵，全是「层」；「维」的命中多是**教学玩具值**（`n_embd = 4`）
    或**官方模型值**（768 维）。真正带 key 名的散文主张只有 **12 处**，其中一半还是跨线引用。
  ② **逐字一致的源码守卫**（交接文档原「Next Best Step ③」）：实测 **164/164** 个笔记代码块
    都是**故意简化**的摘录，**没有一个**与源文件逐字相同 ⇒ 要做这个守卫，
    得先把 164 处笔记代码块改成逐字一致 —— 那是**改内容**，不是**加守卫**，
    而且守卫上线当天就是 164 条红灯。**⇒ 这条不要再做了**（除非先改笔记）。
  ⇒ 能自动化的交集只剩「**能算出真值**」的那一类：**参数量 / 词表大小 / 算式 / 比值**。
- **新守卫**：`qwen3-learning/qwen3-pytorch/tests/test_derived_sync.py`
  （`tests/run_all.py` **自动发现** ⇒ `init.sh` L4 一起覆盖，**不用改 init.sh**）。
  **立身之本：不手写期望值，而是现场把模型搭出来数参数** ——
  GPT-2 按 `config.py` 造模型数 `parameters()` 得 794,496（词表 45，走 `CharTokenizer.vocab_size`，
  **不数行**：词表里有「空格」这种行，`strip()` 会数成 44）；
  Qwen3 造 dense 985,856 / 注入 LoRA 后 994,560 / LoRA 可训练 8,704 / MoE 1,184,512 /
  SwiGLU 单层 FFN 147,456。改配置 ⇒ 期望值跟着变 ⇒ 笔记不改就红。
- **7 条规则**：R1 算式自洽（含链式等号）· R2a 紧邻式参数量 · **R2b 行级** ·
  R3 词表大小 · R4 `N / M (P%)` 三元组 · R5 比值 · R6「可训练总计」= 模型实测总参数。
- **★ R2b 为什么必须有**：关键词紧邻的正则（R2a）**抓不到表格里的数字** ——
  `| 总可训练参数 | 见 §1.9 | **794,496** |` 里关键词和数字被 `|` 隔开，
  那些参数量**一直没人核对过**。补上行级规则后覆盖从 **21 处 → 41 处**，
  新增的 20 处里有 **6 处是表格 / 散文里从来没被核对过的**。
- **归属判定**（分层）：数字**前面最近的关键词**（`GPT-2`/`GELU` → GPT-2 侧；
  `Qwen3`/`SwiGLU`/`dense`/`moe`/`LoRA` → Qwen3 侧）→ 判不出来就按**文件所在目录**判
  ⇒ 「GPT-2 的 794,496」和「Qwen3 的 794,496」分别对待。
  归属启发式判错、但数字本身是实测值时只报**提示**不拦 CI。
- **★ 反证 4 连（改完全部复原）**：
  ① `config.py` 的 `n_layer` 4→5 → **15 处**不一致；
  ② 笔记 794,496→794,497 → **1 处**（正是 R2a 抓不到、被 R2b 抓到的那一行）；
  ③ 百分比 0.88%→0.99% → **2 处**；④ 算式 131,072→131,073 → **1 处**。
- **跑通**：`bash init.sh` → **PASS=30 SKIP=0 FAIL=0**；
  `tests/run_all.py` → **PASS=19 FAIL=0 TOTAL=19**；
  单独跑 → `核对笔记 27 篇；派生量主张 41 处；例外登记 8 条（命中 8，过时 0）`。
- **顺手改了 2 处笔记**（把主张变成**可核对**的写法）：
  `gpt-learning/0.GPT全流程详解.md` 的「TF 端（含 embedding transpose 差异）: ~ 800,256」
  → 加上「参数量」三个字（才进 R2b 范围，由 UPSTREAM 例外放行）；
  守卫原来遇「字节」就**跳过整行** → 改成**只涂掉那个数字**
  （BPE 笔记「语料 20000 句 / 785065 字节 → 目标词表 1000」现在能核对「目标词表 1000」）。
- **已知边界**：`qwen3-pytorch开发计划.md` 不进自动守卫（**人工核对**）·
  测量值类数字（耗时 / loss / 采样类）不在范围内 · 不带参数上下文的分数不进守卫。

### feat-029 — 笔记里「手抄的代码」也纳入机器守卫（Session 016）

- **由来**：feat-028 收班时把交接文档「Next Best Step ③（逐字一致守卫）」判为**不可行**。
  feat-029 **不去硬做 ③**，而是**换一个可落地的问法**。
- **★ 换问法：不要求「长得一样」，要求「行为一样」** ——
  教学笔记**必须**简化（少 docstring、改写变量名、只留主干），
  所以「逐字一致」从设计上就不该是目标；真正的目标是**行为一致**
  （学习者照着笔记敲出来的代码，必须和仓库里跑的是同一件事）。
  实测：笔记版 `get_lr` 把 `config.xxx` 改写成局部变量、还多了一行 clamp ⇒ 输出**逐个相同** ⇒ 放行。
- **新守卫**：`qwen3-learning/qwen3-pytorch/tests/test_note_code_sync.py`
  （`tests/run_all.py` **自动发现** ⇒ `init.sh` L4 一起覆盖，**不用改 init.sh**）。
  **配对 8 条 / 覆盖 4 篇笔记**：
  `get_lr`（2 篇 × 11 个 iter）· `_build_sinusoidal_pos_emb`（3 篇 × 3 组尺寸）·
  `KVCache.append`（比**调用后**的 cache 状态）· `mark_only_lora_trainable`
  （比**可训练参数个数 + 每个参数的 requires_grad 分布**）· `set_lora_enabled`（比 **scaling 值**）。
- **★ 绝不 import 被测的源码模块** —— 源码函数用 `ast.get_source_segment` **摘出来**再 `exec`。
  原因**实测踩过**：`train_pretrain.py` 是**顶层线性脚本**（没有 `__main__` 保护），
  import 它会**把整个预训练重跑一遍**（打印 150 轮迭代）并**覆盖 `checkpoints/pretrain.pt`**。
- **★ 有状态的对象：输入要一样、对象要各造一份** —— `append` 会写 cache、`set_lora_enabled` 会改 scaling。
  第一版只做了一半（两边各造 cache，但**各自 random 出 K/V**）⇒ 比的是两份随机数，
  测出 `max|diff| = 2.502e+00` 的假红。修法：随机数在工厂**外面**生成一次、两边共用。
- **★ 阈值用「绝对 + 相对」双阈值**（`atol=1e-9, rtol=1e-6`）——
  学习率量级只有 `1e-4`，纯绝对阈值 `1e-6` 会让「少一次 +1」这种真错误只剩 `6.000e-06` 的差，卡在阈值边上很脆。
- **★ 反证 3 连（改完全部复原）**：
  ① `get_lr` 的 `(it + 1)` 抄成 `it` → **红**（`|diff| = 6.000e-06`）——
  这正是本仓**历史上真实发生过**的错；
  ② `mark_only_lora_trainable` 的 `id(p) in lora_ids` 抄成 `True` → **红**（`62 vs 16` 个可训练参数）；
  ③ `math.log(10000.0)` 抄成 `math.log(100000.0)` → **红**（`max|diff| = 1.999e+00`）。
  三次都**只红被扰动的那一条** ⇒ 定位精确，没有误伤。
- **跑通**：`bash init.sh` → **PASS=30 SKIP=0 FAIL=0**；
  `tests/run_all.py` → **PASS=20 FAIL=0 TOTAL=20**；
  单独跑 → `核对 8 条「笔记手抄代码 ↔ 源码」配对，全部行为一致（笔记 4 篇）`。
- **顺手给 4 篇笔记加了「本段代码由机器守卫核对」的醒目标记** ——
  `0.GPT全流程详解.md` / `GPT2训练数学公式与超参详解.md` / `2.位置编码.md` / `Qwen3_LoRA与DPO_学习笔记.md`。
- **已知边界**：① 只覆盖**函数级**配对，且是**手工登记**的 8 条
  （不是「自动发现所有能跑的笔记代码块」—— 大部分笔记代码块是片段，缺上下文跑不起来）；
  ② 配对表里的输入点写死，覆盖的是**该函数的行为契约**，不是全输入空间；
  ③ 本 feature 只加守卫、**没动任何训练/评估代码** ⇒ 不需要跑 L3 全量训练。

### feat-030 — 笔记里的 `#L` 行号锚点纳入机器守卫（Session 017）

- **由来**：feat-029 收班时做**全仓锚点普查**，发现全仓 **16** 个 `#L` 锚点里 **4 个失效**，
  而且失效的不是「行号偏了」而是**整段章节在描述一个已经不存在的实现**。
- **★ 根因两层**：① `train_ppo.py` 已被改造成 **14 行的 thin wrapper**
  （只 `from train_ppo_standard_adapted import *`），真实实现在 `train_ppo_standard_adapted.py`（1337 行）
  ⇒ 锚点行号 262 / 271 / 303 / 392 / 397 **全部越界**；
  ② 笔记引用的 `compute_kl` / reward shaping / reward normalization **在重构后的代码里已经不存在**
  （现在对应 `score_reward_model` / `whiten_masked` / `compute_gae`）。
- **★ 核心教训：「链接点得开」≠「链接指得准」** —— 原来的守卫只查「目标文件存不存在」，
  于是这类腐烂**安安静静躺了很久**（**没人点就没人发现**）。行号锚点是一种**会随代码改动自动腐烂**的引用。
- **守卫**：加在 `qwen3-learning/qwen3-pytorch/tests/test_note_refs.py` 的第 6 节 `_check_anchors()`，
  扫**全仓** markdown（跳过 `.git` / `checkpoints` / `docs` / `__pycache__` / `.claude` / `.vscode`），两条规则：
  - **R1（范围）**：目标文件存在，且 `1 <= start <= end <= 文件行数`；
  - **R2（单行锚点必须精确）**：`file.py#L123` 的语义是「就是这一行」
    ⇒ 紧跟其后的 ```python 代码块**第一条语句**必须**就是**源码第 123 行（去注释、去空白后逐字相等）。
    —— **在范围内也可能是指错行**，本仓实测 **3 处**。
- **★ 测过但明确不做**的第三条规则（R3：「代码块要逐行落在锚点区间内」）：
  实测 **18** 个「锚点 + 代码块」配对里 **8 个对不上**，逐条看全是**风格差异**而不是错误
  （笔记省 docstring、把多行签名写成一行、去掉 type hint）⇒ 做成守卫就是一台误报机器。
  故**多行锚点只做 R1**。这与 feat-029 的「逐字一致不可行」同源：**教学笔记必须简化，守卫只能判「真错」**。
- **修的内容**（不是只改行号）：**重写** `gpt-learning/GPT2训练数学公式与超参详解.md` 的 PPO 章节
  （§3.4.3 / §4.1 / §4.2 / §4.3 / §4.4 / §4.5），锚点改指 `train_ppo_standard_adapted.py`：
  policy loss `L1012-L1048`、value loss `L1077-L1093`、token 级 KL `L842-L847`、
  reward shaping `L839-L847`、RM 奖励注入 `L862-L867`、`whiten_masked` `L661-L673`、
  reward whitening 开关 `L178-L179`、`compute_gae` `L676-L728`、advantage whitening `L928-L929`。
  锚点总数 **16 → 22**（删 4 个坏的，加 10 个新的）。
- **★ 顺带修的 3 个「在范围内但指错行」的单行锚点**：
  `model/gpt2.py#L222`（实际是 `att = self.attn_dropout(att)`，笔记引的是 GELU）→ `#L293`；
  `model/gpt2.py#L262`（实际是 `super().__init__()`，笔记引的是 `self.ln_1 = nn.LayerNorm(n_embd)`）→ `#L336`；
  `train_reward.py#L203`（实际是**注释行**，笔记引的是 `-F.logsigmoid(r_chosen - r_rejected).mean()`）→ `#L218`。
- **★ 顺带修的 4 处**内容**事实错误**（不只锚点错，正文本身也错）：
  ① advantage whitening 原文写「未启用」，实际**始终启用**；
  ② reward whitening 原文当成默认开启，实际**默认关闭**
  （`cfg("ppo_reward_whiten", False)`，且 `config.py` 里**根本没有这个键** ⇒ 静默走代码里的 fallback 0.05
  —— 这是个**静默默认值陷阱**）；
  ③ 原文写「代码里没有 `compute_kl()` 这个函数」，现在补上**真实的 token 级 KL**；
  ④ reward shaping / reward normalization 的公式与代码对不上，改成真实口径，并讲清
  **RM 的奖励只加在最后一个 action 上**。
- **★ 反证 2 连（改完全部复原）**：① 单行锚点 `#L293` → `#L294` → **红**
  （`单行锚点指错行 → …/gpt2.py#L294 / 笔记里引用的：x = F.gelu(x) / 源码第 294 行：（空行 / 越界）`）；
  ② 多行锚点的 end 改成越界值 → **红**（`锚点行号越界`）。两次都**只红被扰动的那一条**。
- **★ 守卫当场抓到我自己的错**（不是脚本）：写 `qwen3-pytorch/README.md` 的 feat-030 段落时，
  我举了个锚点例子，把路径写成了 `…/model/gpt2.py#L293`（`…` 是**省略号不是路径**）
  ⇒ 守卫立刻红（`锚点目标文件不存在`），`init.sh` 从 PASS=30 掉到 **FAIL=1**；
  改成普通行内代码 `model/gpt2.py#L293` 后恢复绿 ⇒ 这条守卫**不是摆设**，连「自己写的文档」也照抓。
- **跑通**：`python tests/test_note_refs.py` →
  `源码行号锚点 22 处全部在范围内` / `test_note_refs OK  笔记 9 篇 / 链接 82 条 / 路径 121 处 / 锚点 22 处`；
  `tests/run_all.py` → **PASS=20 FAIL=0 TOTAL=20**；`bash init.sh` → **PASS=30 SKIP=0 FAIL=0**。
  本 feature 只改守卫与笔记、**没动任何训练/评估代码** ⇒ 不需要跑 L3 全量训练。
- **已知边界**：① R2 只比「代码块**第一条语句**」与「源码那一行」，**块里其余行不比对**
  （R3 实测会误报）；② R2 只在锚点后 3 个非空行内出现 ```python 围栏时生效，没围栏就只做 R1；
  ③ `0.GPT全流程详解.md` 的**多行锚点**（`inference.py#L280-L287` 等）目前只保证「范围对」，
  区间**并没有逐字包含**笔记引用的那几行（**故意不查**，见上面的 R3 实测）。

## Broken Or Unverified（坏的 / 没验证的）

- ~~**★ P1 线：SFT / PPO checkpoint 退化成常量输出**~~ **✅ 已修（feat-027，Session 014）**
  症状：`sft.pt` / `ppo.pt` 对任何 prompt 都输出同一串常量（公共后缀 47/48 = 98%），
  PPO 相对 SFT 零提升（`rm_score = 0.0696` / `improvement = 0.0000`）。
  **根因**：训练目标 ≠ 推理目标（SFT 的 labels 同下标 + 模型 loss 不左移 ⇒ 学成「复制当前 token」），
  外加「特殊 token 不可达」与「评估切片错位」两个缺陷。
  修完：公共后缀 0%（0/32.0）、`ppo_rm_score` **+0.9188**、`improvement` **+3.6509**。
  细节与新数字见上面「feat-027」小节。
- **Qwen3 的评估侧是「精简 6 段」**（feat-025 六段 + feat-026 补齐生成质量）：
  GPT-2 线那 9 项里的 `benchmark` **本线没有**（合成语料没有 QA 题库）—— 差异已在
  `qwen3-learning/qwen3-pytorch/README.md` 逐项列出。
- **★ 生成质量虽然已经能并排，但并排数字带着三条不可比性**（feat-026）：
  ① **探针的域内外** —— 动物探针对 GPT-2 是**域内**（它的预训练语料就是动物童话）、
  对 Qwen3 是**域外**（Qwen3 训在 transformer 主题语料上）⇒ **不能**读成「架构谁更好」；
  ② **tokenizer 粒度** —— `response_quality` 的长度规则（10~50 **字符**）按字符级 tokenizer 标定，
  Qwen3 用 BPE、回答普遍更长 ⇒ 这一条规则对 Qwen3 **系统性不利**（跨线只看 `response_quality_ex_length`）；
  ③ **词表大小** —— 熵的绝对值依赖 V（均匀分布时 H = log V；GPT-2 ~64 字符 vs Qwen3 778 子词）
  ⇒ 跨线只看 `token_entropy_normalized` = H / log(V)。
  详见 `qwen3-learning/Qwen3_训练推理完整学习笔记.md` §13.5。
- **LoRA / DPO 的胜率都是「训练内指标」**：合成偏好数据只有 5 个 prompt × 5 个 chosen，训练集与评估集同分布 ——
  它证明的是「LoRA 确实让模型更偏向 chosen」，**不能当泛化能力**
- **BPE 词表只有 778 < 目标 1000**：合成语料的可合并对用尽（521 次后没有频次 ≥ 2 的对），属语料多样性问题，非实现 bug
- **MoE 在本仓库体现不出容量优势**：val_loss 1.3607 vs dense 1.3510（MoE 反而略差）——
  语料太简单 + 专家更窄（参数量只多 1.21×）+ 只训 300 步。**写笔记时别暗示「MoE 一定更好」**（前提条件见笔记 §25.1）
- **笔记缺口**：**已清零** —— `Qwen3_BPE_学习笔记.md` 已补齐（feat-008），
  `Qwen3_LoRA与DPO_学习笔记.md` 早前已补齐 LoRA 半部（§1–§7）与 DPO 半部（§8），§9 教训 7 条。
  `qwen3-learning/` 现有 **9 篇**笔记，**每篇开头都有三行源码交叉引用**（由 `tests/test_note_refs.py` 守住）
- ~~**交叉引用只到「文件 + 类/函数」级，不到行号级**~~ **✅ 部分已修（feat-030，Session 017）** ——
  当时的顾虑是「行号会随代码改动漂移，写进笔记反而变成错误信息」。
  现在的做法是**两全**：行号锚点**可以写**，但**必须由机器盯着**
  （R1 范围 + R2 单行锚点必须精确 ⇒ 漂移了测试就红）。
  ⚠️ 仍然**裸写的不带目录的文件名**（`config.py` / `evaluate.py`）**不在自动校验范围内**
  （机器判不准它指主线还是本线），这部分靠人工
- ~~**★ P1 线 PPO 章节与重构后的代码脱节**~~ **✅ 已修（feat-030，Session 017）** ——
  `gpt-learning/GPT2训练数学公式与超参详解.md` 的 4 个 `#L` 锚点
  （`train_ppo.py#L262-L268` / `#L271-L303` / `#L392-L393` / `#L397-L399`）指向
  `gpt-learning-pytorch/train_ppo.py` —— 而那个文件现在只有 **14 行**（thin wrapper，
  真实实现在 `train_ppo_standard_adapted.py`，1337 行）⇒ **4 个锚点全部越界**。
  更严重：那几段被引用的代码（`compute_kl`、reward shaping、reward normalization）
  在**重构后已经不存在**了 —— 现在对应的是 `score_reward_model` / `whiten_masked` / `compute_gae`
  ⇒ 当时**整段 PPO 章节在描述一个已经不存在的实现**。
  现在**整章重写**（§3.4.3 / §4.1 / §4.2 / §4.3 / §4.4 / §4.5），锚点改指真实实现，
  并顺带修掉 4 处**内容事实错误**（详见上面 feat-030 小节）。
- **feat-017 的结论只在 fp32 / 小尺寸下成立**（vocab=293、n_layer=4、n_embd=128、seq_len=16）：
  「全关逐位相同」**不能外推**到别的 dtype / 别的形状。
  且单开组件时新引入的参数（SwiGLU 的 gate）是**随机初值** ⇒ 那几行的 diff 适合比**量级**，
  不等于「这个组件贡献了多少准确率」
- **本仓库的 aux_loss 与 HF 数值不能直接比大小**：本实现 = HF Mixtral 同款公式的 1/K（见下面的陷阱表）
- **feat-009 守卫的三个已知盲区**（都写进了 `tests/test_config_sync.py` 的注释）：
  ① **跨线比较的文档没有自动守卫** —— `开发计划.md` 引用 GPT-2 的值时会误报，已移出自动范围、改为人工核对；
  ② **裸文件名不在校验范围** —— 笔记里不带目录的 `config.py` / `evaluate.py` 机器判不准指哪条线；
  ③ ~~**只覆盖「写成 `键 = 值` 的主张」**~~ **✅ 已修（feat-028，Session 015）** ——
     散文里「128 维」「4 层」这种**字面读法实测不可行**（见下面的 feat-028 段），
     改成守卫**能算出真值**的那一类：**参数量 / 词表大小 / 算式 / 比值**
     （`tests/test_derived_sync.py`，覆盖 21 → 41 处主张）。
     ⚠️ 剩下的**真·散文数字**（无 key、无法算真值）**仍然只能靠人读** —— 这是设计取舍，不是遗漏。

## Next Best Step（按顺序，不要跳）

> **`feature_list.json` 的 30 条 feature 全部 passing** ⇒ 下面三条是**建议新立项**的候选，
> 都需要**先写进 `feature_list.json`（priority 31/32/33）再动手**（AGENTS.md 的 WIP=1 规矩）。

| 顺序 | 建议 feature | 做什么 | 为什么是它 |
|------|---------|--------|--------|
| 1 | **feat-031** | 给**多行锚点**加**标识符级**校验（feat-030 的 R3 软化版）：不比逐字，只比「笔记代码块里出现的每个 `标识符(` 调用 / 赋值目标，是否在锚点区间内出现过」 | **feat-030 留下的唯一已知盲区**：R3 实测 18 个配对里 8 个对不上，但**全是风格差异** ⇒ 逐字不行；但「标识符在不在区间里」既不误报又能抓真错 —— 实测 `0.GPT全流程详解.md` 的 `inference.py#L280-L287` 区间里**根本没有**笔记引用的 `position_ids = ...` 那一行 |
| 2 | **feat-032** | 让 **MoE 在本仓语料上体现出容量优势**（换/扩合成语料 + 加训练步数 + 固定 seed 重测） | 现在 MoE val_loss 1.3607 反而**略差**于 dense 1.3510 ⇒ 结论只能是「本仓语料太简单 + 专家更窄（参数量只多 1.21×）+ 只训 300 步」。**「MoE 一定更好」现在是常识不是实验** —— 把它变成实验是 P3 的核心内容 |
| 3 | **feat-033** | 给笔记里**裸写的文件名**（`config.py` / `evaluate.py`）定一个**消歧约定**并纳入自动校验 | feat-009 守卫的已知盲区 ②：**机器判不准它指主线还是本线**。约定比猜测便宜（例如笔记里一律写成 `qwen3-config.py` / `gpt2-config.py`） |

**下一个 session 建议做的第一件事**：
把上表的 **feat-031 立项**（写进 `feature_list.json`，priority 31，status `not_started`），
再动手实现 —— 它承接刚做完的 feat-030，是同一条「笔记 ↔ 源码锚点」链上的最后一个已知缺口。

**已完成的这条链（留个记录，别重复劳动）**：
① ~~**修 P1 线的 SFT 塌陷**~~ **✅ 已做（feat-027，Session 014）** ——
   `sft.pt` / `ppo.pt` 不再吐常量（公共后缀 0%），`ppo_rm_score` 0.0696 → +0.9188，
   `improvement` 0.0000 → +3.6509；口径写进了 `init.sh` 的 L1/L2 守卫；
② ~~**feat-009 守卫的第 3 个盲点自动化**~~ **✅ 已做（feat-028，Session 015）** —
   散文数字的字面读法**实测不可行**（「层」语义冲突 18 处、「维」多为教学/官方值）；
   改成守卫**能算出真值的派生量**（参数量 / 词表 / 算式 / 比值），
   期望值**现场把模型搭出来数**（`tests/test_derived_sync.py`，41 处主张）；
③ ~~**把 GPT-2 线的 `get_lr` 等实现细节也做成「笔记抄源码必须逐字一致」的守卫」**~~
   **❌ 已实测不可行（feat-028，Session 015）** ——
   **164/164** 个笔记代码块都是**故意简化**的摘录，没有一个与源文件逐字相同；
   要做这个守卫得先改 164 处笔记（那是改内容，不是加守卫），上线当天就是 164 条红灯。
   ⇒ **换一条路：改成「行为一致」守卫** ——
   **✅ 已做（feat-029，Session 016）**：把笔记里那段代码和源码里那段**都跑起来**，
   在同样输入上**比输出**（`tests/test_note_code_sync.py`，8 条配对 / 4 篇笔记）。
   ★ 关键：源码函数用 `ast` **摘出来**再 `exec`，**绝不 import**
   （`train_pretrain.py` 是顶层线性脚本，import 会重跑预训练并覆盖 checkpoint）；
④ ~~**笔记里的 `#L` 行号锚点机器校验 + 修 P1 线 PPO 章节**~~ **✅ 已做（feat-030，Session 017）** ——
   守卫：`tests/test_note_refs.py` 第 6 节（R1 范围 + R2 单行锚点必须精确）；
   内容：PPO 章节**整章重写**，锚点改指 `train_ppo_standard_adapted.py`，顺带修 4 处事实错误。

**这一步不许动的**：不在 `qwen3-learning/qwen3-pytorch/` 之外 import；不削弱 `evaluate.py` 的任何指标与阈值；
GPT-2 线的训练代码只允许在**已立项且已确认**的 feature 里改（feat-027 已改完，
它改的只有**口径**与**切片**，没有动 `config.py` 的任何超参，也没有动 `evaluate.py` 的指标与阈值）。

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
| 采样类指标的复现 | 生成是**采样**（temperature 0.8）⇒ 不固定种子，每次跑数字都在动。**生成质量类指标必须先 `torch.manual_seed`** 再算，否则笔记里记的数字明天就复现不出来 |
| 对照实验的**结论文字** | 也是**实验输出**，必须**由数据算出**（脚本里比大小再打印）。写死「域内比域外高」，而实测恰好相反 ⇒ 结论与数字互相打脸 |
| 判「模型是否退化成常量输出」 | **不能看「不同回答的数量」** —— GPT-2 的 `sft.pt` 会把 prompt 的最后一个字符（`?`/`.`）一起吐出来，30 条不同问题得到 30 条「看起来不同」的回答。用**公共后缀长度**才露馅（47/48 = 98%） |
| 跨线复用同名包 | 两条线都有 `data` 包、都有 `model` 包 ⇒ 同进程里只能解析到一个。跨线复用要么**按绝对路径加载文件**，要么开**子进程**并把 `PYTHONPATH` 指向要用的那一条线 |
| import 别人的 `evaluate.py` | GPT-2 线的 `evaluate.py` 在**模块层**就 `argparse.parse_args()` 并按 `args.stage` 分派 ⇒ 直接 import 会顺手把一整个 stage 跑一遍。用 `argparse.ArgumentParser.parse_args` 打补丁的办法把它变成「只定义函数、不执行」 |
| 「同名」≠「同口径」 | 抄两份指标实现只是**看起来**一样，迟早分叉。口径一致要靠**同一份实现** + 一条**机器证明**（同一批输入两边算出的值必须逐值相等，不相等就报错退出） |
| Windows 子进程输出编码 | bash 按 UTF-8 解码、Windows Python 默认 stdout 是 GBK → 中文乱码。证据命令加 `PYTHONIOENCODING=utf-8` |
| 脚本放在子目录 | Qwen3 线脚本在 `train/` / `tests/` 里，`sys.path[0]` 不是项目根 → **必须显式 `sys.path.insert(0, <项目根>)`** |
| **偏好胜率的长度偏置** | 总 logprob 口径偏袒**短回复**（`too_short` 策略胜率**恒为 0.000**，因为 rejected 只有 6.3 token），把整体胜率天花板压到 ~0.75；长度归一口径反过来偏袒「可预测」的文本。**必须按策略分组看，否则误判「没进步」** |
| **旁路注入的「结构性 bug」** | `merge_lora` 合并后分支还开着 → ΔW 算两遍；`inject_lora` 只冻被包装的那一个 Linear → embedding/Norm/MLP/lm_head 仍可训；loss 掩码把 prompt 也算进去。**三条都不报错、只静默给错结论 ⇒ 必须用「等价性断言」守，不能靠看 loss 曲线** |
| **DPO 的 π_ref 是「哪一个」模型** | 标准做法锚 **SFT 模型**。本仓库「关掉 LoRA 分支」得到的是**预训练模型** —— 锚错**不报错、loss 也降、指标照涨**，只是整个实验锚在错的基线上。解法：`merge_lora` → `strip_lora`（新增）→ 重新 `inject_lora`。**必须用「初始 margin 恒为 0 / loss 恒为 log 2」这条可运行断言来钉** |
| **DPO 目标指标 ≠ 业务指标** | 强配方 margin +4.65 / dpo_acc 0.78 / 绝对胜率 0.605→0.665；**弱配方 dpo_acc 冲到 0.945，绝对胜率却是 0/200 翻转**。**报告时必须同时给 margin（目标口径）与 win_rate（业务口径）**，只给前者是自我欺骗 |
| **覆盖参数没传到子模块 = 静默的半配置** | `Qwen3Dense(dropout=0.0)` 只关掉了 embedding 的 dropout，注意力/MLP 一直是 `cfg.dropout=0.1`；而 **p>0 的 Dropout 在 train 模式下会推进全局 torch RNG**⇒ 「训练前多做几次评估」就改变训练轨迹（同一配方 margin 差 0.028）。**每个覆盖开关都要配一条等价性断言（结构 + 行为各一条）** |
| **KV Cache 与 naive 的 logits 不会逐位相同，但 token 会** | 两条路径的 matmul 规约顺序不同（naive 一次前向 T_q = P+s，cache 每步 T_q = 1）⇒ fp32 末位差异 ~1e-7。**验证「KV Cache 改没改结果」要断言 token 序列，而不是断言 logits 逐位相等**；想更强就断言逐步 logits 的 `max|diff| < 1e-4` |
| **「decode 不 mask」要用反向实验证** | 正向断言（decode 输出对）证明不了「没被 mask」。正确做法：**污染历史 K/V ⇒ 输出必须改变** —— 若历史被屏蔽了，污染不会影响输出 |
| **守卫的期望值不能手写** | 手写期望值 = 守卫变成「第二份要同步的文档」，改配置时它自己就过期了。**现场把模型搭出来数参数**（`tests/test_derived_sync.py`）⇒ 期望值永远跟着代码走 |
| **「关键词紧邻」的正则抓不到表格里的数字** | `\| 总可训练参数 \| 见 §1.9 \| **794,496** \|` 里关键词和数字被 `\|` 隔开 ⇒ 紧邻式正则**一条都抓不到**（feat-028 实测：补「行级」规则后覆盖 21 → 41 处，新增的 20 处里有 6 处是**从没被核对过**的参数量） |
| **守卫要能自己作废** | 例外登记簿里「已不再命中」的条目要**自动报出来要求删除**，否则例外越积越多，最后变成「什么都豁免」 |
| **别用行级跳过** | 遇「785065 字节」就跳过**整行** ⇒ 连带把同一行真正该核对的「目标词表 1000」一起放过。**只涂掉那个数字**，其余照常核对 |
| **「逐字一致」不是好目标，换成「行为一致」** | 教学笔记**必须**简化（少 docstring、改写变量名、只留主干）⇒ 逐字守卫上线当天就是 164 条红灯（feat-028 实测 164/164 块都是简化摘录）。**把两边都跑起来比输出**才是可落地的判据：简化过的 `get_lr` 输出**逐个相同** ⇒ 放行；抄错 ⇒ 红灯 |
| **绝不 import 被测的源码模块** | 本仓的脚本大多是**顶层线性脚本**（没有 `__main__` 保护）⇒ import 就有副作用：`train_pretrain.py` import 一次会把**整个预训练重跑一遍**并**覆盖 `checkpoints/pretrain.pt`**（实测踩过）。正确做法：`ast.get_source_segment` **摘函数**再 `exec` ⇒ 零副作用 |
| **有状态的东西：输入要一样、对象要各造一份** | `KVCache.append` 会写 cache、`set_lora_enabled` 会改 scaling。**对象共用** ⇒ 第二次调用接着第一次写（假红）；**对象各造但输入各随机** ⇒ 比的是两份随机数（实测 `max|diff| = 2.502e+00` 假红）。正确做法：随机数在工厂**外面**生成一次、两边共用；只有状态对象各造各的 |
| **阈值要跟着量级走** | 学习率量级只有 `1e-4` ⇒ 纯绝对阈值 `1e-6` 会让「少一次 +1」这种真错误只剩 `6.000e-06` 的差，**刚好卡在阈值边上**。用**绝对 + 相对双阈值**（`atol=1e-9, rtol=1e-6`） |
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
| **「错位做几次」决定模型学到什么** | SFT 的 labels 与 input **同下标** + 模型 loss **不左移** ⇒ 模型学的是「输出**当前** token」（复制）⇒ 推理时无限吐空格。**预训练对、SFT 错**，因为两条数据线的错位次数不一致。**错位只允许做一次**（放在模型里），两条数据线都用同一口径 |
| **特殊 token 是多字符词表条目 ⇒ 逐字符 encode 永远够不到它** | `"<sep>"` 会被拆成 5 个字符（`'<'`→UNK）⇒ `BOS/EOS/SEP` 的 id 永不出现 ⇒ **模型学不会自己停**，生成永远跑满 `max_new_tokens`。修法：`encode` 改成「最长匹配优先」，特殊 token 整体一个 id |
| **回答不能按「字符串长度」切，要按「token id 个数」切** | `UNK` 在词表里是**整条** `"<unk>"`（5 个字符）⇒ 解码后的 prompt 前缀 47 字符 vs 原始 31 字符 ⇒ 每条回答被塞上 `"? <unk>sep<unk> "` 前缀 ⇒ 污染 `distinct_n` / `self_bleu` / `response_quality` |
| **指标异常时先怀疑度量，再怀疑模型 —— 但别把两者混成一个** | feat-026 里 GPT-2 侧 `distinct_1` 低得离谱，一半是切片 bug 造成的；可**切片不是塌陷的原因**（修完切片，公共后缀仍是 47/48 = 98%）。**两个独立根因要分别验证、分别修** |
| **「训练口径」必须写进机器守卫** | 「换个数据线就换个学到的任务」这件事不报错、只看指标才发现 ⇒ 断言要直接钉在数据与 loss 上：`labels[j]==x[j]`、末位监督目标 == EOS、以及**手工重算 `CE(logits[:,:-1], targets[:,1:])` 与模型 loss 比对** |
| **行号锚点会静默腐烂** | 「链接点得开」≠「链接指得准」。行号会随代码改动漂移，**没人点就没人发现**。本仓真实踩过：4 个锚点指向一个**只剩 14 行**的 wrapper（行号 262/271/303/392/397 **全部越界**），而它们引用的代码**在重构后已经不存在** ⇒ 整段 PPO 章节在描述一个**不存在的实现**。**行号锚点必须机器守** |
| **「在范围内」≠「指对了行」** | 范围校验（R1）只能保证行号不越界；本仓还有 **3 处**在范围内却**指着旁边代码**的锚点（`gpt2.py#L222` 实际是 dropout、`#L262` 实际是 `super().__init__()`、`train_reward.py#L203` 实际是**注释行**）⇒ 单行锚点还得再判「那一行**就是**笔记引用的那句」 |
| **别用「逐字比对」守多行锚点** | 实测 18 个「锚点 + 代码块」配对里 **8 个对不上**，逐条看全是**风格差异**（笔记省 docstring、拆开签名、去 type hint）⇒ 做成守卫就是一台误报机器。**教学笔记必须简化，守卫只能判「真错」**（与 feat-029 的「逐字一致不可行」同源） |
| **静默默认值陷阱：`config.py` 里没这个键** | `cfg("ppo_reward_whiten", False)` —— `config.py` 里**根本没有** `ppo_reward_whiten` 这个键 ⇒ 静默走代码里的 fallback，**不报错**。若按「config 里有 ⇒ 生效」去想，就会把**默认关闭**的东西写成默认开启（feat-030 实测就错在这：reward whitening 默认关闭，而 **advantage whitening 反而始终开启**） |

---

## 需要用户拍板的决策（别自行决定）

| 决策 | 出现在 | 说明 |
|------|--------|------|
| ~~LoRA 的注入点与超参~~ | `feat-023` | **已定**：`lora_r=4 / lora_alpha=8 / lora_target_modules=("q_proj","v_proj")`；`lora_lr` 3e-4→**1e-3**、`lora_max_iters` 300→**1500**（消融实测依据） |
| ~~DPO 的 β 与训练步数~~ | `feat-024` | **已定**：`dpo_beta=0.1`、`dpo_lr` 1e-4 → **1e-3**、`dpo_max_iters` 300 → **1500**（消融实测依据，见上面的表） |
| DPO 偏好数据要不要引入真实数据集 | `feat-024` | 目前是合成数据，引入真实数据会动依赖上限（**未做**） |
| 评估要不要和 GPT-2 线的 9 项对齐 | `feat-025`（已决）+ `feat-026`（已决） | **不对齐 RLHF 三件套**（sft / reward / ppo / compare）：本线走 LoRA+DPO，有对应物（lora / dpo），列名不通用；**pretrain 与 speedup 保留逐字同名列**；`benchmark` 本线没有（合成语料没有 QA 题库）；**`diversity` / `response_quality` 已于 feat-026 补齐并做成「同名同口径」**（共用 `quality.py` + 交叉验证断言）。差异写在 README |

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
python tests/run_all.py          # 20 个 smoke 汇总
python tests/test_pretrain.py    # 单个 smoke
python tests/test_lora.py        # LoRA 的 9 组断言
python train/train_pretrain.py --model dense   # 预训练（dense，~47s CPU）
python train/train_pretrain.py --model moe     # 预训练（MoE，~59s CPU）
python train/train_lora.py       # LoRA 微调（r=4 q/v, lr=1e-3, 1500 步）
python train/train_dpo.py        # DPO（π_ref = SFT 模型，merge → strip → 重新 inject）
python tests/test_dpo.py         # DPO 的 9 组断言
python tests/test_inference.py   # KV Cache 推理与采样（12 组断言）
python tests/test_evaluate.py    # 精简评估（6 段）
python evaluate.py --stage quality   # 生成质量段（feat-026）
python ablations/compare_quality.py  # 与 GPT-2 的生成质量并排 + 口径交叉验证（feat-026）
python tests/test_quality.py     # 生成质量的 8 组断言
python ablations/compare_with_gpt2.py   # 与 GPT-2 的退化对照（feat-017，6 行消融表）
python tests/test_compare_with_gpt2.py # 退化对照的 6 组断言（含 3 条反证）
python tests/test_note_refs.py    # 笔记 ↔ 源码交叉引用校验（feat-008 链接可点开 + feat-030 行号锚点不越界/不指错行）
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
| Qwen3 笔记原文 | `qwen3-learning/Qwen3_*.md`（**9 篇**，BPE 已补齐；每篇开头有 GPT-2 / Qwen3 源码交叉引用；训练推理篇含 §10 推理 + §11 评估 + **§12 退化对照** + **§13 生成质量**） |
| 笔记数字一致性守卫 | `qwen3-learning/qwen3-pytorch/tests/test_config_sync.py`（feat-009；27 篇笔记 / 88 处断言 / 25 条例外登记；改 `config.py` 忘了改笔记 ⇒ 红灯） |
| 笔记**派生量**一致性守卫 | `qwen3-learning/qwen3-pytorch/tests/test_derived_sync.py`（feat-028；27 篇笔记 / **41 处**主张 / 8 条例外登记；参数量 / 词表 / 算式 / 比值 ↔ **代码实测**；期望值现场数参数，不手写） |
| 笔记**手抄代码**一致性守卫 | `qwen3-learning/qwen3-pytorch/tests/test_note_code_sync.py`（feat-029；**8 条配对 / 4 篇笔记**；不要求逐字一致，只要求**行为一致** —— 两边都跑起来比输出；源码用 `ast` 摘出来 `exec`，**绝不 import**） |
| 笔记**行号锚点**一致性守卫 | `qwen3-learning/qwen3-pytorch/tests/test_note_refs.py` **第 6 节**（feat-030；全仓 **22 处** `#L` 锚点；**R1** 范围 + **R2** 单行锚点必须精确；多行锚点只做 R1 —— 逐字比对实测会误报） |
| GPT-2 对照实现 | `gpt-learning/gpt-learning-pytorch/model/gpt2.py`（**只能按文件路径 import**，见陷阱表） |
| 对照实验脚本 | `qwen3-learning/qwen3-pytorch/ablations/compare_with_gpt2.py`（本仓唯一跨目录引用 GPT-2 线的地方，单向） |
| 生成质量（两条线共用） | `qwen3-learning/qwen3-pytorch/quality.py`（**唯一一份指标实现**，两条线共用）+ `data/quality_prompts.py`（共享探针）+ `ablations/compare_quality.py` + `ablations/_gpt2_quality_runner.py`（GPT-2 侧子进程） |
| 收班自查 | `clean-state-checklist.md` |

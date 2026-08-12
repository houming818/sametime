# C10：从预训练到任务坍缩的 TreeHeap Pipeline

日期：2026-08-11

Claim：`S3-TREEHEAP-PRETRAIN-POSTERIOR-C10`

状态：pilot 已完成；匹配任务迁移得到支持，posterior 优势为弱支持，本 checkpoint 的多分辨率 READ 被否定。

## 1. 为什么要重新建立流水线

C09 测量的是“冻结的 `H_state` 是否容易被新 Decoder 读取”。它不是贝叶斯
坍缩实验。单参考答案的平均 NLL 只能说明模型给真实 token 分了多少概率，
不能告诉我们概率桶里还有哪些合理候选，也不能告诉我们最后生成了什么。

C10 固定一条完整航线：

```text
构造数据
-> 自然语料 Pretrain
-> 匹配的 Task Train
-> 冻结 checkpoint
-> 概率桶与实际坍缩 Proof
-> TreeHeap 因果干预
-> 自动 Report
```

Pretrain 负责形成广泛的语言先验；Task Train 负责学习如何把先验用于 WMT；
Proof 阶段不再更新参数，只观察概率如何随证据变化，以及最终坍缩成什么。

## 2. 主张

在完全相同的 Butterfly/FOLD/READ 架构中，自然文本的 next-span 预训练应当
形成随上下文变化的候选概率桶。与语料 unigram 先验相比，它应当更接近独立
语料中的经验候选分布。随后，在相同 WMT 训练预算下，预训练初始化应当优于
同一随机起点直接训练 WMT。两种收益都必须依赖 source 和 TreeHeap 的通信、
配对结构。

这是一条联合 Claim。仅仅看到更低的 Pretrain NLL，不算通过。

## 3. 固定架构

Pretrain 和 Task Train 使用同一份模型参数结构：

```text
token WRITE
-> 动态宽度 XOR Butterfly
-> learned lifting FOLD
-> root 与带地址 details
-> recursive READ
-> shared Decoder
-> token 概率桶
```

第一轮固定：

| 项目 | 数值 |
|---|---:|
| 词表 | 现有 32K 双语 SentencePiece，加 PAD/方向槽位 |
| state dim | 256 |
| hidden dim | 256 |
| 最大 leaf | 256 |
| Butterfly scale | 0.25 |
| 数值模式 | FP32 |
| 主机 | `io`，维持现有 270W 限制 |

Pretrain 不把具有词向量的 `BLANK` 写进 Butterfly。source 是一段已经完整
观察到的自然前缀，target 是紧随其后的自然文本。

## 4. Stage A：自然语料 Pretrain

只读取训练分区：

- news2016zh、wiki_zh、webtext2019zh；
- WMT Massive 的中英文单语侧，但不提供中英配对关系。

source 长度从 `4/8/16/32/64/128` 中选择，target 为后续 `32` pieces。
加入短 source，是为了让 posterior proof 使用的证据也属于模型训练分布。优化时只用：

```text
L_pre = -sum_t log P(x_t | H_context, x_<t)
```

交叉熵只负责产生梯度。它不是“推理成立”的证据。

模型更新前必须保存随机初始 state。Pretrain 最佳 checkpoint 由独立 continuation
验证集选择。

## 5. Stage B：匹配的 WMT Task Train

建立两组：

| 组 | 初始化 |
|---|---|
| `PT` | Stage A 最佳 Pretrain checkpoint |
| `SC` | Stage A 开始前保存的同一随机 state |

两组使用完全相同的 WMT 行、翻译方向、batch 顺序、optimizer updates、参数量
和验证测试集合。两组都允许训练全部参数。这样比较的是预训练初始化，不是更多
任务数据或更大模型。

## 6. Stage C：概率桶与坍缩 Proof

从独立 held-out 自然语料自动统计长度为 `4/8/16` 的重复精确上下文，以及这些上下文
之后实际出现的 token 分布：

```text
Q(token | evidence)
```

冻结模型后，对每个上下文输出：

- Top-20 token 与概率；
- 语料候选集合获得的总概率；
- 模型分布与语料经验分布的 JS divergence，并设置 OTHER 桶；
- unigram 与经验分布的 JS，作为朴素先验基线；
- greedy top-1 到底坍缩成什么，它是否属于经验候选集合；
- top-p 样本、短 continuation、entropy 和不同坍缩结果比例。

人工编写的例句只用于阅读，不参与门槛。

## 7. TreeHeap 因果干预

冻结 Pretrain checkpoint 后重复 Proof：

1. 把 source 跨样本错配；
2. 运行时把 Butterfly 切换成 `identity`；
3. 在一个 FOLD 深度把 right child 跨样本重新配对。

不再使用整树 mirror，因为同步翻转 node 与 mask 是当前置换等变 READ 的
自同构，没有辨识力。

## 8. 预注册门槛

### P0：Pipeline 有效

模型参数结构一致；`PT` 和 `SC` 的父 checkpoint hash 正确；WMT 数据流和计分
token 完全一致；梯度有限；checkpoint 可原样加载；不存在 target 直通泄漏。

### P1：形成条件先验

Pretrain 后必须同时满足：

- held-out NLL 相对初始化改善至少 `0.20`；
- 模型到经验候选分布的平均 JS，比 unigram 基线至少低 `0.02`；
- 经验候选集合的模型概率质量，比 unigram 至少高 `0.02`；
- greedy 落入经验候选集合的比例，比 unigram top-1 至少高 20%；
- 至少 25% 的上下文得到不同 greedy 结果，不能坍缩成同一个 token。

### P2：Pretrain 迁移到任务

相同 WMT 预算下，`PT` 必须比 `SC` 至少低 `0.02` validation NLL，且 chrF 和
token-BLEU4 不能下降。两组实际翻译都必须保存。

### P3：概率桶依赖 TreeHeap

wrong-source、runtime-identity、pre-FOLD pair-break 三种干预，每一种都必须让
经验 JS 恶化至少 `0.01`，或者让候选集合概率下降至少 `0.01`。wrong-source
还必须让至少 25% 的 greedy 坍缩发生改变。

### P4：实际坍缩可用

至少 95% 的上下文生成非空结果；严重相邻重复率低于 25%；任何单一 greedy
token 都不能占全部上下文的 50% 以上。

## 9. 执行顺序

```text
smoke：20 个 Pretrain updates
       PT/SC 各 20 个 Task updates
       16 个概率上下文
       只验证代码，不支持 Claim

pilot：100M Pretrain target tokens
       匹配的有限 WMT budget
       至少 256 个概率上下文
       单 seed 机制筛选

formal：只有 pilot 的 P0-P4 全部通过，才能另行预注册
```

## 10. Evidence

```text
ara/s3-generation/evidence/s3_pretrain_task_posterior_pipeline/
  config.json
  environment.json
  data_manifest.json
  initial_state.json
  pretrain/
  task/PT/
  task/SC/
  proof/
  report.json
  README.md
```

每个阶段必须记录父 checkpoint SHA、tokenizer/data hash、样本与 token 数量、
optimizer 预算、GPU 时间和完整命令。

## 11. 边界

正结果最多支持：这个有限 Pipeline 学到了随上下文变化的概率协议，并能在
匹配任务中迁移。它不能证明意识、语义理解、人类级贝叶斯最优、世界模型完整、
架构优越或产品可用。

## 12. Smoke 结果

2026-08-11，taskd 任务 `156` 在 `io` 完成了代码有效性 smoke：20 次
Pretrain 更新，PT/SC 各 20 次匹配的 Task Train 更新，16 个 posterior
上下文。九项 P0 完整性检查全部通过，包括父 checkpoint 状态 hash、两组
共用的任务数据流 hash、有限梯度和 checkpoint 严格重载。

| 观察项 | Smoke 数值 | 含义 |
|---|---:|---|
| Pretrain 初始 -> 最佳 valid NLL | `10.8661 -> 9.1168` | 优化链路可以工作 |
| PT task valid NLL | `9.0647` | 更新太少，不能评价迁移 |
| SC task valid NLL | `9.0430` | 本次 smoke 中 PT 反而差 `0.0217` |
| Native 到经验分布的 JS | `0.6750` | 差于 unigram smoke 基线 |
| Unigram 到经验分布的 JS | `0.6136` | 20-step 模型尚未胜出 |
| Native 经验候选集合概率质量 | `0.00857` | 低于 unigram 的 `0.06615` |
| 非空生成率 | `1.0` | 生成代码可以运行 |
| 严重相邻重复率 | `1.0` | 所有生成都发生重复坍缩 |
| 单一 greedy token 最大占比 | `1.0` | 所有上下文坍缩成同一 token |

这些是 20 次更新后应当如实保留的负面质量观察。它们既不能驳回 C10，
也不能支持 C10；它们证明后续 pilot 可以看到真实的条件坍缩失败，而不是
只输出一个平均 NLL。

taskd 任务 `157` 另做了 batch-32 吞吐测试：200 次 Pretrain 更新耗时
`45.13` 秒，约为每秒 `4.5K` 个 target pieces。由此估算，100M target
pieces 的 Pretrain 约需 `6.1` GPU 小时；加上匹配的 PT/SC Task Train、
posterior proof 和运行余量，第一轮完整 pilot 预计需要 `9-12` GPU 小时。

吞吐测试同时发现两项必须在 pilot 前修复的问题：

1. 960 条数据循环训练在约第 100 步后开始过拟合。pilot 必须持续读取新的
   语料窗口，不能反复循环小型内存表。
2. smoke 的 posterior 上下文集中于少量中文主题；经验桶只按末尾 2/4/8
   pieces 聚合，而模型实际看到了 128-piece 前缀。因此在任何 pilot 结果出现
   之前修订合同：Pretrain 加入 4/8/16-piece 输入；proof 按来源文档分层抽样；
   模型输入与经验分布都严格使用同一段 4/8/16-piece 证据。

Evidence：`evidence/s3_pretrain_task_posterior_pipeline/smoke/`。

修订后，taskd `158` 验证了输入与经验分布严格对齐的 4/8/16-piece posterior
路径；taskd `159` 验证了仅供 pilot 使用的持续新窗口流式读取分支。两项任务
都正常退出并通过 P0。20-step 模型质量依然不属于 Claim 证据。元数据保存在
`smoke_v2/` 和 `fresh_stream_smoke/`，可重载的大型 checkpoint 保留在 `io`。

## 13. Pilot 结果与后续结构审计

正式 pilot 已在 `io` 完成。Pretrain 共处理 100,000,768 个 target pieces；
PT 和 SC 两条 WMT 任务臂各训练 25,000 step，并严格使用同一个包含
20,198,612 pieces 的任务数据流。

| 指标 | PT | SC | PT - SC |
|---|---:|---:|---:|
| WMT test NLL | `5.403696` | `6.291975` | `-0.888279` |
| token BLEU4 | `5.065443` | `1.171101` | `+3.894342` |

Posterior 审计中，Native 到经验分布的 JS 为 `0.609404`，unigram 基线为
`0.641004`；Native 分配给经验候选集合的质量为 `0.062018`，unigram 为
`0.033277`。换成错误 source 后，`78.64%` 的 greedy token 发生变化。这支持
“已经出现 source-conditioned signal”，但 posterior 的绝对质量仍然较弱。

Pilot 同时暴露了 parent Claim 没有预料到的结构失败：Native READ 几乎把
全部质量送到了 leaf。在 256 条 held-out 样本上，Native NLL
`5.299829634` 与强制 leaf 的 `5.299829704` 相同。运行时拿掉 Butterfly 和
破坏 depth-0 配对，NLL 分别增加 `0.360457` 与 `0.391156`。因此
Butterfly/FOLD 仍有因果作用，但 learned STOP 已坍缩到 leaf 分辨率。

两个后续诊断进一步缩小了解释空间：

1. `path_shape_audit.json` 否定了单一链表路径。Top-1 leaf 质量只有
   `0.210366`，只保留 Top-1 会让 NLL 增加 `4.004781`。但将所有有效 leaf
   改成均匀池化只增加 `0.027111` NLL，因此强语义树索引也没有得到支持。
2. `observer_resolution_stop_smoke.json` 证明有限观察阈值可以剪除数值尾巴。
   `epsilon=0.001` 少访问 `8.24%` 节点，NLL 增加 `0.001791`；
   `epsilon=0.003` 少访问 `10.11%` 节点，NLL 增加 `0.010178`。但仍有超过
   `99.93%` 的质量到达 leaf，所以有限分辨率不能修复主流量坍缩。

最终拆分状态：

```text
Pretrain 到 WMT 的匹配迁移：本 pilot 支持
随上下文变化的 posterior signal：弱支持
Butterfly/FOLD 的因果参与：支持
learned 多分辨率 STOP/READ：本 checkpoint 否定
产品可用性或语义推理：不支持
```

Evidence：`evidence/s3_pretrain_task_posterior_pipeline/pilot_seed10101/`。

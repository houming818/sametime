# STONE-1 Butterfly V1 产品训练合同

日期：2026-08-05

状态：已注册并获准执行，等待 selector smoke

里程碑：`STONE-1-BUTTERFLY-V1`

Claim 系列：`S3-TREEHEAP-BUTTERFLY-PRODUCT-C07`

## 1. 这次到底要决定什么

本文不假设 TreeHeap 已经成为最终架构。它要回答的是：在双向翻译产品中，
纯 Native Butterfly 与追加 Identity replay，哪一种是更好的有限平台训练方案。

实验分两段：

1. 用同一个冻结 checkpoint 做三 seed、三实验臂的协议选择；
2. 只有获胜方案才能获得一次完整语料产品续训和制品审计。

C06 不会被悄悄改写成成功。它预注册的 Recovery 只有 `0.3909`，没有通过
`0.50` 门槛。但 C06 找到了一个很强的 Pareto 候选：和等算力追加 Butterfly
相比，追加 Identity 让跨视角 JS 改善 `0.1375`，Native NLL 只付出
`+0.00281`。C07 要验证这个交换是否稳定，并且是否能转化为产品价值。

## 2. Claim 边界

本实验能够得到的最强结论是：

> 在冻结的 V1 平台上，选定的 TreeHeap 续训方案能够产生一个可重载的中英
> 双向翻译 checkpoint；输出依赖 source、重复率受控，并保留注册的 TreeHeap
> 因果作用和验证集质量。

它不声称达到行业最佳翻译，不声称具有对话、世界知识或通用智能，也不声称
计算量优于其他架构，更不声称 20% Identity 是普遍最优比例。

## 3. 冻结的起点

| 制品 | 路径 | SHA-256 / 数值 |
|---|---|---|
| 起始 checkpoint | `ara/s3-generation/evidence/s3_treeheap_butterfly_bilingual_full/checkpoint_best.pt` | `821ce8123d78817b37ff8f0a68458fd59427a7af555f93c7c87c297f28861c1d` |
| WMT 语料 | `/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv` | `3f4a5189a6b2f06a8a928165a69e119d6e0afe71ffece2bbe7c049ecef7a44df` |
| SentencePiece | `/home/nio/datasets/wmt_massive/sp_bpe_massive.model` | `9956eff597852f8c684c4ad23243d15889da6a9b138f8fd025570147324cc731` |
| 语料规模 |  | `2,520,995,022` bytes，`14,170,275` 个原始句对 |
| checkpoint 进度 |  | epoch `3`，next line `6,400,000`，`48,811,685` 样本，`1,032,196,489` target token |
| 最佳 valid mean NLL |  | `3.4223013826` |

如果 hash、文件大小、模型键、词表大小或数据切分 hash 有任意差异，程序必须在
初始化 CUDA 之前退出。

## 4. 平台参数

| 项目 | 冻结值 |
|---|---|
| 主机 | `io` |
| GPU | NVIDIA GeForce RTX 3090，24,576 MiB |
| GPU UUID | `GPU-4c73dc30-0b6f-86fb-f1ae-24f316d8b54c` |
| 功率限制 | 270 W，由 `nvidia-power-limit.service` 强制；严禁提高 |
| 驱动 | 580.173.02 |
| PyTorch / CUDA | 2.12.0+cu130 / CUDA 13.0 |
| cuDNN | 92000 |
| SentencePiece | 0.2.1 |
| CPU | AMD Ryzen 7 5700X，8 核 16 线程 |
| 内存 | 31 GiB，无 swap |
| 注册时本地剩余磁盘 | 637 GiB |
| 队列 | `nio-taskd`，本机 19100 端口，GPU 任务严格串行 |

训练只读取 `/home/nio/datasets`，不直接读取 NAS。checkpoint 先原子写入本地，
只有完成的 best/final 制品才复制到 NAS。GPU watchdog 和功率限制属于基础设施，
不是可调实验变量。

## 5. 模型参数

| 参数 | 数值 |
|---|---:|
| 总可训练参数 | 34,448,396 |
| FP32 模型状态 | 131.41 MiB |
| 含 AdamW 状态的 checkpoint | 394.27 MiB |
| 最大 TreeHeap leaf | 256 |
| source/target 最大正文 | 253 pieces |
| state / hidden 维度 | 256 / 256 |
| Butterfly coupling scale | 0.25 |
| 动态 leaf 宽度 | 32 / 64 / 128 / 256 |
| 各宽度 batch | 64 / 32 / 16 / 8 |
| 数值模式 | FP32；C07 不引入 AMP |
| 梯度裁剪 | 1.0 |

固定数据流：

```text
token WRITE
-> 动态宽度 XOR Butterfly 通信
-> 自适应递归 FOLD
-> H_state
-> 递归 READ
-> 共享双向 Decoder
-> token 概率桶
```

不加入 Transformer、flat attention 或外部 teacher。

## 6. 优化器完整性

续训读取 checkpoint 中的 AdamW moments，然后必须显式把每个参数组的学习率
改为 `2e-4`。启动时把真实生效的学习率写入 `environment.json` 并断言。

一个 parent batch 必须只产生一次 `optimizer.step()`。Replay 臂先按照非 PAD
target token 数组合 base loss 与 replay loss，再做一次反向传播：

$$
L=\frac{n_B L_B+n_R L_R}{n_B+n_R}
$$

这样小 replay 子集不会获得第二次满强度更新。更新次数、Native token、replay
token 和各视角 token 必须分别计数。

C07 暂不加入显式 JS loss。它先验证已经观察到的 Identity replay 机制；加入
`lambda * JS` 属于另一个新 Claim。

## 7. 阶段 A：三 seed 方案选择

### 7.1 独立语料区间

选择实验使用原始行 `[6,700,000, 7,700,000)`，与 C06 区间分离。稳定 hash
切分仍会排除 valid/test 行。

seed 为 `9201/9202/9203`。同一个 seed 内，所有实验臂必须共享句子顺序、batch
顺序和确定性的 replay 样本集合。

### 7.2 三个实验臂

| 实验臂 | Native Butterfly 剂量 | 追加剂量 | 作用 |
|---|---:|---:|---|
| `N_native` | 100% | 无 | 正式任务质量基线 |
| `BB_replay` | 100% | 20% Butterfly | 等算力对照 |
| `BI_replay` | 100% | 20% Identity | 跨视角协议候选 |

BB 与 BI replay 完全相同的行和 target 文本，唯一差别是 source 坐标模式。

### 7.3 选择阶段预算

每个实验臂使用 100 万原始行。按 C06 吞吐估计，每个 seed 约需 9.3 GPU 小时：
Native 约 1.9 小时，两个 replay 臂各约 3.7 小时。三个 seed 串行预算约
28--32 GPU 小时。

墙钟 timeout 只防止故障：Native 四小时，replay 六小时。科学停止条件是注册
的数据行和 token 剂量，不是时间到了就截取一个最好结果。

### 7.4 BI 获胜门槛

只有全部通过，BI 才进入产品续训：

```text
mean(JS_BB - JS_BI)                         >= 0.10
每个 seed 的 JS_BB - JS_BI                  >= 0.05
mean(NLL_BI - NLL_BB)                       <= 0.01
每个 seed 的 NLL_BI - NLL_BB                <= 0.02
mean chrF(BI - BB)                          >= -0.30
mean token BLEU-4(BI - BB)                  >= -0.20
严重重复率增量                              <= 0.02
每个 seed 的 source-shuffle damage          >= 1.50
每个 seed 的 adjacent 结构替换损伤           > 0
Butterfly communication delta 与梯度         > 0
```

没有隐藏的加权总分。BI 要么同时满足全部约束，要么不获选。若 BI 失败，产品方案
回到 Native。BB 的主要作用是控制额外算力，除非它独立改善任务质量，否则不会
自动成为产品方案。

## 8. 阶段 B：全语料产品续训

阶段 A 签字后，获胜方案从冻结的原始 checkpoint 重新开始，而不是从“看起来
最好”的 selector seed 继续。这样不会把选择噪声带进 release checkpoint。

产品训练确定性遍历全部 `14,170,275` 原始行一次。中英两个方向继续由稳定
行 hash 规则保持平衡。科学完成条件是一整个 pass，不是墙钟时间。

| 获胜方案 | 有效剂量 | 预计时间 | 故障 timeout |
|---|---:|---:|---:|
| Native | 一次完整 Native pass | 27--32 h | 48 h |
| BI | Native 全量 + 20% Identity replay | 48--55 h | 72 h |

release 训练 seed 为 `9301`。阶段 A 提供三 seed 机制稳定性；昂贵的阶段 B 只
训练一个产品制品，不能被重新描述成三 seed 产品复现。

## 9. 中间观察窗口

| 窗口 | 已暴露原始行 | 目的 | 保存内容 |
|---|---:|---|---|
| W0 | 0 | 精确起始基线 | 指标、core/full Dreams、重载 hash |
| W1 | 100,000 | 发现 LR 错误或即时坍缩 | 轻量指标、core Dreams |
| W2 | 300,000 | 对齐 C06 尺度 | 完整指标、full Dreams、checkpoint |
| W3 | 1,000,000 | selector 终点/产品早期趋势 | 完整指标、full Dreams、checkpoint |
| W4+ | 每 2,000,000 | 产品生长曲线 | 完整指标、full Dreams、Pareto checkpoint |
| Final | 14,170,275 | 锁定产品结论 | test、CLI、best/final checkpoint |

每 25 万行额外保存原子恢复点，但只永久保留 W0/W2/W3/W4/Final。分别维护
best Native NLL、best chrF、质量约束下 best JS 和 final，不用一个隐藏总分
覆盖多目标权衡。test 只允许 Final 运行，所有 wake 决策只看 valid。

每个完整 wake 记录：

```text
按方向和长度桶统计 Native/Identity NLL
按方向和长度桶统计跨视角 JS
token BLEU-4、chrF
source shuffle、Identity、adjacent override 损伤
communication delta RMS 与 gradient norm
非空率、重复率、不同输出数、输出长度比
样本数、target/replay token、optimizer step、GPU 小时
显存峰值、温度/功率采样、checkpoint hash
```

## 10. Dreams 语料准备

Dreams 是不可变观察探针，永远不是训练 target，也不能用于选择 batch。C07
使用两个文件：

```text
dreams_product_v1_core.tsv   32 条，每个轻量 wake 都生成
dreams_product_v1_full.tsv   96 条，只在 W0/W2/W3/W4/Final 生成
```

full 集中英方向各 48 条，按下面分层：

| 类别 | 数量 | 观察目标 |
|---|---:|---|
| 简单词汇与组合 | 8 | 基础 source 依赖 |
| 主客体与主动/被动 | 16 | 有序角色和手性 |
| 否定、情态与量词辖域 | 16 | 少量 token 引发的大语义变化 |
| 时间与因果方向 | 12 | before/after、原因/结果 |
| 附着与嵌套从句 | 12 | 长程结构绑定 |
| 实体、数字、单位、日期 | 12 | 事实保留 |
| 33--64、65--128 pieces | 12 | 中长 TreeHeap 宽度 |
| 129--253 pieces | 4 | 最大递归深度附近 |
| 重复与通用回答陷阱 | 4 | 条件坍缩 |

TSV 格式：

```text
id<TAB>direction<TAB>category<TAB>source<TAB>reference<TAB>required_facts
```

准备规则：

1. 训练前写好最小对，包括主客互换、肯定/否定、before/after 和数字变化；
2. 预编码 source，清单中保存 piece 数和 width bucket；
3. 对 WMT 扫描规范化后的 source/reference；精确命中必须替换或标记污染；
4. 冻结两个 TSV 和 manifest 的 SHA-256；
5. 训练中不得改文件；临时想法写入 `dreams_exploratory.tsv`，不能影响 C07；
6. 每个 wake 的坏输出也必须保留。

自动评分包括 chrF、数字/实体保持、极性和时间关键词、重复率、长度比以及最小
对分离度。人工评价另存，不能覆盖数值结果。

## 11. 硬停止与科学降级

立即停止：

```text
loss 或梯度非有限
checkpoint 无法精确重载
制品/数据/tokenizer hash 不一致
真实生效学习率不是 2e-4
一个 parent batch 出现多次 optimizer.step
GPU 掉卡或 watchdog 报警
本地剩余磁盘低于 100 GiB
```

连续两个完整 wake 出现以下情况时暂停审核：

```text
Native valid NLL 比 W0 恶化超过 0.20
source-shuffle damage 低于 0.50
超过一半 core Dreams 产生同一个规范化输出
严重重复率超过 0.30
所有非 root 结构损伤降为零
一个翻译方向改善、另一个持续恶化
```

暂停后不能现场调参继续。先保存 evidence，再注册修复方案。

## 12. 产品 Gate

只有全部满足，才能签署产品制品：

```text
mean test NLL                              <= 3.45
valid chrF 相对 W0                         提升 >= 0.50
valid token BLEU-4 相对 W0                 下降不超过 0.20
非空生成率                                 = 1.00
严重重复率                                 <= 0.10
source-shuffle damage                      >= 1.50
Native Butterfly runtime override damage  >= 1.00
四个长度区域都保持 source-dependent
重载后固定 greedy token IDs                完全一致
CLI 声明任务                               与双向翻译 objective 一致
```

若 BI 获选，Final 还必须保持选择阶段的 JS 门。任务质量过关、结构门失败，只能叫
翻译 demo；结构门过关、质量失败，只能叫机制 checkpoint，不能发布成产品版。

## 13. 任务队列与可靠性

```text
preflight hash + Dreams 污染扫描
-> 三臂各 3 万行 smoke
-> seed 9201: N -> BB -> BI
-> seed 9202: N -> BB -> BI
-> seed 9203: N -> BB -> BI
-> selector 汇总并签署方案
-> 获胜方案全语料产品续训
-> Final 锁定评估与 reload 审计
-> NAS 归档、model card、CLI 包
-> sendme 邮件通知
```

每个任务启动五分钟后检查日志、进程、GPU 显存和功率。GPU 任务严格串行。
失败实验保留原始 evidence；记录原因后，只能从最后一个原子 checkpoint 恢复。

Evidence：

```text
ara/s3-generation/evidence/s3_stone1_butterfly_product_v1/
```

大制品：

```text
/mnt/nas/ara/s3_stone1_butterfly_product_v1/
```

必须保存 `contract.json`、`environment.json`、hash、`command.sh`、stdout/stderr、
`trace.jsonl`、wake 指标、不可变 Dreams、各臂 summary、selected/final checkpoint
和 NAS manifest。

## 14. 启动审核门

正式 selector 实验臂启动前，smoke Reviewer 必须确认：

1. 三臂代码和 one-step optimizer 不变量；
2. 数据/checkpoint/tokenizer 的精确 hash；
3. 提取后的 valid/test manifest；
4. 冻结的 Dreams 与污染扫描报告；
5. 磁盘预算和 taskd 命令；
6. C06 仍保持 `not supported as registered`，没有被产品计划改写历史。

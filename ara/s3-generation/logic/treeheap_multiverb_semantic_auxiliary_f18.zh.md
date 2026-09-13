# TreeHeap 多动词语义辅助训练

日期：2026-09-13
Claim：`S3-TREEHEAP-MULTIVERB-SEMANTIC-AUXILIARY-F18`
状态：第一次正式提交因数据容量门失败；已完成 token 化容量审计，准备在修订合同下重跑。

## 1. 起点

F15 证明 source token 数量可到达 root；F16/F17 则发现，target-side `push*` 的跨中文表达族方向从
leaf 起就不稳定，而且 root 没有相对 leaf 出现显著坍塌。因此当前证据不支持直接替换 FOLD 退火算子。

F18 测试更早的一阶问题：给 TreeHeap 一个明确但有限的跨语言目标存在性压力，能否让 leaf 与 root
共同形成可泛化的目标词方向，同时不显著破坏原翻译能力。

## 2. 数据与隔离

- 起点固定为 E01 TreeHeap-106M pass2 checkpoint；
- 语料固定为 `NioClean-ZHEN-S098-7M-v2/pairs.tsv`；
- 概念固定为 `push / like / want / eat / carry`；
- 初版计划使用 23 个中文表达族。token 化容量审计显示 `push:force` 的正例只有训练 `6/32`、评估 `3/8`，
  `eat:swallow` 的正例只有训练 `26/32`、评估 `4/8`，均无法满足配对合同；修订版剔除这两个族，使用 21 个表达族；
- 每个“概念 x 表达族”抽取 40 个 target 含目标词形的正例和 40 个负例，其中每类 32 个训练、
  8 个辅助验证；
- split 由原始 corpus 行号的稳定哈希决定，同一原始行即使命中多个概念也不得跨 split；
- F16 的 480 个原始行全部从 F18 训练与辅助验证中排除；
- source 正文限制 2--30 pieces，加入方向 token 与 EOS 后不超过 leaf 宽度 32；target 最多 31 pieces
  加 EOS。

标签是英文 target 中固定词形是否出现，不把未出现目标词的翻译称为错误。

## 3. 训练结构

在 FOLD tree 上对 base 与 extra 通道分别 masked mean 后拼接。一个共享线性 head 同时读取 leaf 与 root，
输出五个 target-presence logits：

```text
L_aux = 0.5 * [BCE(head(H_leaf), y) + BCE(head(H_root), y)]
L = L_translation + 0.20 * L_aux
```

共享 head 是尺度一致性压力：同一目标标签必须能从 leaf 与 root 的同一坐标系读出。BCE 的 `pos_weight`
仅由训练集标签频率计算。主翻译 loss 保留，防止实验退化为只会做五个二分类任务。

固定训练日程：

1. 冻结 TreeHeap，仅预热共享 head 50 steps，学习率 `2e-3`；
2. 解冻 E01 原本可训练参数，联合训练 120 steps；TreeHeap 学习率 `2e-5`，head 学习率 `5e-4`；
3. batch size 16，depth 5/6/7 确定性轮换，梯度范数裁剪到 1.0；
4. 不因 NLL、AUROC 或生成质量非单调而提前停止，只在 OOM、CUDA/Xid、NaN/Inf 或证据损坏时停。

## 4. 观察

- head 预热后与联合训练后的五概念 leaf/root 辅助验证 AUROC；
- F16 原 480 行、原 leave-one-family-out mean-difference 探针在训练前后的 FOLD 与 READ-facing
  leaf/root AUROC；
- 固定 WMT valid 子集的训练前后 NLL；
- source 冻结、模型更新、有限梯度、checkpoint 哈希与 reload；
- 保存 manifest、trace、summary 和远端 checkpoint；大 checkpoint 不拉回 Git。

## 5. 预注册门槛

- O0：21 个族均满足 32/8 的正负 train/eval 配额，F16 行号零重叠，split 无泄漏；
- O1：全部 loss、梯度、特征和指标有限；
- O2：frozen source 哈希不变，TreeHeap 可训练状态确实更新；
- O3：保存 checkpoint 可重载且状态哈希一致；
- S0：训练后 WMT mean NLL 相对训练前恶化不超过 0.20；
- P1：训练后辅助验证的五概念 root 宏平均 AUROC 不低于 0.65；
- P2：训练后辅助验证 root 与 leaf 宏平均 AUROC 差不超过 0.05；
- P3：F16 原留出集的 FOLD root AUROC 相对训练前至少提高 0.03；
- P4：F16 原留出集的 READ-facing root AUROC 相对训练前至少提高 0.03。

P1/P2 只验证辅助任务自身能否建立共享尺度方向；P3/P4 才检测它是否迁移到完全排除的 `push` 留出行。
若辅助验证成功但 F16 不提高，说明只记住了训练表达族；若 leaf 提高而 root 明显落后，才重新支持 FOLD
压缩问题；若两者共同提高且 WMT 稳定，再扩展训练规模。

## 6. 边界

这是任务训练 smoke，不是产品模型。自动词形标签只提供有限监督；成功不能等同于理解，失败也不能证明
语义无法形成。它的用途是决定下一阶应扩大语义任务，还是返回修改递归退火。

## 7. 第一次正式提交的失败记录

任务 422 在训练开始前终止：`push:force:train` 只有 `6/32` 个可长度匹配的正例。
后续任务 423 对排除 F16 行、稳定 split 和 SentencePiece 长度约束后的候选做审计，确认失败来自数据容量合同，
不是 loss、梯度或模型运行时故障。该失败证据保留，不把这些样本称为“错误数据”。修订版只收缩实验族集合，
并将候选 reservoir 上限从 256 提高到 1024，以减少边界族的抽样波动。

# TreeHeap 语义探针容量复核

日期：2026-09-13
Claim：`S3-TREEHEAP-SEMANTIC-PROBE-CAPACITY-F17`
状态：预注册，尚未观察结果。

## 1. 动机

F16 在六个中文表达族的 leave-one-family-out 测试中得到约 0.60 的 leaf AUROC、0.60 的 FOLD root
AUROC 和 0.58 的 READ-facing root AUROC。其 mean-difference 探针只使用标准化后两类均值之差，不能
校正特征协方差；若目标方向被旋转或分散在相关坐标中，该探针可能低估可恢复信息。

F17 使用 F16 已冻结的 480 行 manifest、同一 checkpoint 和同一特征位置，只把读出器升级为固定正则的
ridge 线性探针。它不修改 TreeHeap，不重新抽样，也不改变 target-side `push*` 标签。

## 2. 固定合同

- 输入 manifest SHA-256 必须为 `940c7d11f201c7df45c9ac37f0482d567134d34090f5ced4a51cdf0f5eff61ed`；
- 六族各 40 正例、40 负例，完整留出一族测试；
- 比较 `input_bow`、FOLD 与 READ-facing 的 root/middle/leaf，depth 5/6/7；
- 训练特征逐坐标标准化，常量坐标移除；
- ridge 系数固定为 `1e-3 * active_feature_count`；
- 每个外层折同时求解真实标签和五个族内乱序标签，防止计算顺序造成差异；
- 主指标仍为测试 AUROC，另报告 AP、balanced accuracy 与 score gap；
- checkpoint 前后逐 tensor 必须完全一致。

## 3. 预注册判读

- O0：manifest、行数、族平衡和 token 长度合同完全匹配 F16；
- O1：全部解、预测与指标有限；
- O2：冻结 checkpoint 未改变；
- C0：READ-facing root 乱序对照宏平均 AUROC 小于 0.60；
- P1：FOLD root 宏平均 AUROC 不低于 0.65；
- P2：READ-facing root 宏平均 AUROC 不低于 0.65；
- P3：leaf 不低于 0.70 且同阶段 root 至少低 0.10，才支持压缩损失；
- P4：若 ridge root 比 F16 同位置 mean-difference 至少提高 0.05，则 F16 的弱结果部分来自探针欠拟合。

若 ridge 仍表现为 leaf/root 同时弱，下一步先建立多动词 target-presence 辅助训练阶梯；若 ridge 使 leaf
明显变强而 root 不变，才把 FOLD 修订提升为直接候选；若 root 也变强但 Decoder 仍不能生成，则问题属于
当前 READ/Decoder 的目标坐标使用，而非信息不存在。

## 4. 边界

ridge 是更强的线性读出器，不是语义本身。高维探针可能利用分布线索，因此必须以未见表达族和乱序标签为
约束。F17 只复核 F16 的可恢复性结论，不据此宣称完整翻译能力。

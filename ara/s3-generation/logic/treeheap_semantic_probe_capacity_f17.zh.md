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

## 5. Smoke 结果

io taskd 419 完成固定 manifest 的全部六折。代码、逻辑、F16 cases 与 F16 summary 的运行哈希均与
提交 `145808c` 对应文件一致；480 行 manifest 哈希保持不变。checkpoint 前后逐 tensor 相同，270 W
限制保持，最高温度 62 摄氏度、最高显存 990 MiB，未见 CUDA、Xid 或非有限值。

宏平均 AUROC：

| 特征 | F16 mean-difference | F17 ridge |
|---|---:|---:|
| input BOW | 0.5327 | 0.5267 |
| FOLD leaf | 0.6003 | 0.5241 |
| FOLD root | 0.5974 | 0.5340 |
| READ-facing leaf | 0.6003 | 0.5241 |
| READ-facing root | 0.5842 | 0.5631 |

ridge 没有达到 0.65 的 root 可恢复门，也没有使 root 相对 F16 提高 0.05。相反，协方差校正更容易
拟合训练表达族内的方向，在完整留出新表达族时泛化下降。READ-facing root 虽高于其 leaf `0.0390`，
但两者都低于语义成立门槛，不能解释为高层形成了可靠语义。

分族 READ-facing root 约为：physical 0.57、technical 0.41、abstract 0.59、press 0.59、force 0.53、
other_tui 0.69。它复现了 F16 的异质性：`other_tui` 有局部方向，`technical` 无法跨族迁移。

O0--O2 与 C0 通过；P1、P2、两项 P3 和两项 P4 均未通过。

## 6. 当前结论

F17 没有发现被 F16 简单探针漏掉的统一线性目标方向。结合 F15，当前状态可以更精确地表述为：

1. source token 的存在与数量确实到达 root；
2. 但 target-side `push*` 的跨表达族预测方向从 leaf 起就不稳定；
3. root 没有相对 leaf 出现足够大的额外退化，因此目前不能把主因定位为递归退火压缩；
4. 当前模型更像形成了若干表达族局部协议，而非统一的跨语言动词语义协议。

下一步应先做小规模、多动词、多义项的 target-presence 辅助训练，使“同一目标词由不同中文表达触发、
相同中文线索在不应触发时受抑制”成为明确梯度目标。该训练必须同时监督 leaf 与 root，并继续保留原翻译
loss；随后用 F16/F17 原样复测。只有出现 leaf 达标、root 显著落后，才动 FOLD 算子。

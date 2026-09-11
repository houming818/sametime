# F13：TreeHeap 可学习多面方向路由

日期：2026-09-11  
Claim：`S3-TREEHEAP-LEARNED-POLYTOPE-ROUTER-F13`  
状态：smoke 已完成；工程合同、方向分化和生成健康通过，NLL 优势未通过。

## 1. 已知起点

F12 在冻结的 TreeHeap-106M checkpoint 上发现，同一递归位置的 target-conditioned 局部梯度
具有多个有效方向。8 组方向相对同步单轴方向能够利用更多梯度能量，并在 oracle 干预中降低
更多 NLL。但 F12 的每个方向都由正确 target 反传得到，不能在推理时直接使用。

F13 测试下一阶问题：只观察当前 `(left, right, merge depth)` 的小参数路由头，能否学会预测
这些分组方向，并在 held-out 数据上优于同参数量的单轴方向头。

## 2. 严格配对

base 256 维和 extra 384 维各自分成 8 个通道组。两个实验臂拥有相同形状的 projection、
逐 merge/group head 和 bias，参数总量完全相同，且逐 tensor 同初始化：

- `scalarized`：先预测 8 个 raw 值，再取均值并广播，强制所有组同步旋转；
- `grouped`：保留 8 个 raw 值，各组独立执行 F10 的有界正交旋转。

两臂都只递归并暴露 parent；不改 READ、不暴露 detail、不训练 106M checkpoint。全部 head/bias
从零开始，因此 step 0 必须和 native FOLD 精确一致。

## 3. 训练与观察

- 数据：固定 WMT 双向训练流，排除本次 valid/test pair；
- 两臂使用同一批次顺序、direction、protocol depth、seed 和优化参数；
- 每臂 300 steps，batch 8，rank 16，AdamW，学习率 `2e-3`；
- 目标：teacher-forced token NLL；
- 每 50 steps 记录 loss、梯度范数、正交能量误差与方向分化；
- 训练结束后记录逐 depth valid/test NLL、BLEU、非空率、重复率和输出样例；
- 指标非单调不提前停止。

## 4. 判据

- O0：参数量与初值严格相同，两个臂 step-zero 相对 native 的 NLL 差不超过 `1e-7`；
- O1：两臂均有有限非零梯度，正交能量误差不超过 `1e-5`；
- O2：冻结 checkpoint 前后完全一致，两个 theta checkpoint 可重载复现；
- P1：grouped 的实际 `u` 在组间发生分化，平均组内标准差大于 `1e-4`；
- P2：grouped 的 held-out test mean NLL 比 scalarized 至少低 `0.005`；
- P3：grouped 的 BLEU4 median 不低于 scalarized，非空率不低于 scalarized `0.02` 以上，
  重复率不高于 scalarized `0.02` 以上。

P1 只说明路由头使用了新增自由度。P2/P3 才分别表示概率读出和自由生成出现可迁移收益。它们
都是 smoke 证据，不授权替换默认 FOLD，也不授权自动扩大模型或延长训练。

## 5. 否证解释

若 P1 失败，说明当前训练目标或参数头无法识别 F12 的 oracle 多方向。若 P1 通过而 P2/P3
失败，说明自由度被训练使用但没有泛化，后续应研究组定义、共享 projection 的干扰或正则化，
不能仅靠追加 steps。若 grouped 与 scalarized 都改善，但差异很小，则 F12 的局部几何优势未
转化为当前尺度的学习优势。

## 6. Smoke 结果

io taskd 405 完成两个配对臂各 300 steps，用时 442.93 秒。两个臂均为 32,080 个参数，
逐 tensor 同初始化，step-zero 相对 native 的 NLL 差为 0。两臂梯度有限非零，最大能量误差
低于 `3.2e-7`，冻结的 106M checkpoint 未改变，theta 保存重载误差为 0。O0--O2 通过。

grouped 的实际方向发生了清楚分化：base/extra 的组内 `u` 标准差分别为 0.06494 和
0.00747，平均为 0.03620；P1 通过。base 比 extra 更主动地使用多方向自由度。

| 状态 | valid mean NLL | test mean NLL | BLEU4 median | nonempty min | repetition max |
|---|---:|---:|---:|---:|---:|
| native | 4.22856 | 4.20940 | 8.03773 | 1.00 | 0.00693 |
| scalarized | 4.20069 | 4.19129 | 7.74384 | 1.00 | 0.00243 |
| grouped | 4.20815 | 4.19894 | 11.60700 | 1.00 | 0.00491 |

两个可学习方向头都比 native 降低了 held-out NLL，但 scalarized 比 grouped 更低。grouped
相对 scalarized 的 valid/test NLL 分别高 0.00745/0.00765，因此 P2 未通过。这个结果不能用
“grouped 没学会”解释，因为它确实分化且仍比 native test NLL 低 0.01046。

生成侧出现相反排序。grouped 的 BLEU4 median 比 scalarized 高 3.86316，非空率相同，重复率
只高 0.00248，P3 通过。逐层 BLEU 表明收益主要来自 depth 6：

| depth | native | scalarized | grouped |
|---:|---:|---:|---:|
| 5 | 4.63107 | 4.74749 | 4.68578 |
| 6 | 8.03773 | 7.74384 | 12.27451 |
| 7 | 12.10288 | 11.29756 | 11.60700 |

样例中，grouped 在 depth 6 恢复出一条更接近参考的法律条款结构，但其他长句仍有明显幻译。
因此当前生成收益是值得追踪的信号，不是产品质量结论。

## 7. 当前结论

F13 支持“F12 观察到的分组方向可以由只读 `(left, right, depth)` 的小参数头学出”，并观察到
**NLL 与自由生成发生帕累托分叉**：单轴头拥有更好的平均 token 概率，多面头在当前小样本上
越过了更多离散生成边界。它不支持“多面头全面优于单轴头”，也不足以替换默认 FOLD。

由于 BLEU 只使用 16 条生成样本，且增益集中在 depth 6，下一步先扩大冻结 checkpoint 与
theta 的只读评估，按多个固定数据块检查 BLEU 胜率；不立即增加训练步数，不把一次 median
跳升写成稳定收益。

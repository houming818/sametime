# F11：TreeHeap 方向退火头配对训练

日期：2026-09-10  
Claim：`S3-TREEHEAP-DIRECTIONAL-THETA-TRAINING-F11`  
状态：smoke 已完成；工程合同通过，方向参数可训练，但 held-out soft/hard 增量均未支持。

## 1. 起点

F10 已确认有界正交退火满足零点精确复现、能量闭合和有限非零梯度，并在三个非饱和输入上
观察到一致的左右符号敏感性。但所有完整句仍未生成 hard `push`。F11 不把 F10 的最优固定
角度写入模型，而是让一个小型内容条件化参数头从行为损失中自行学习方向。

## 2. 配对臂

两臂冻结同一个 E01 TreeHeap-106M checkpoint，共享完全相同的随机投影初值、训练样本、深度
序列、优化器、学习率和步数：

| 臂 | parent 计算 | 目的 |
|---|---|---|
| radial | `((l+r)/sqrt(2)) * exp(log(1.5)*tanh(raw))` | F04 的等参数径向对照 |
| directional | `(c+u*d)/sqrt(1+u^2)`，`u=0.5*tanh(raw)` | F10 的方向候选 |

其中 `c=(l+r)/sqrt(2)`，`d=(l-r)/sqrt(2)`。两个参数头都是 shared projection 加逐深度
head/bias，参数名、形状和总量必须一致。所有 head/bias 从零开始，因此两臂 step 0 都严格等于
原始 FOLD。F11 只向上递归和暴露 parent，不在本次配对中改变 READ 接口或加入 detail 通道。

## 3. 数据与训练

- 训练：F04 的八条原子词与重复短组合；
- 测试：未在训练中出现的西西弗完整句、小明喜欢苹果、小明想要苹果；
- depths：`5,6,7` 循环；
- 两臂各 `90` outer steps，AdamW，学习率 `2e-3`；
- 行为目标：正概念覆盖、负概念抑制、EOS 覆盖和重复惩罚；
- 固定 WMT 小样本 NLL 只作为健康观察，不单独决定方向头成功或失败；
- 保存每个深度的自由生成、hard 概念命中、soft coverage、rank、重复和参数范围。

## 4. 判定

- O0：两臂 step-zero 文本逐 token 相同，WMT NLL 相对 native 差不超过 `1e-8`；
- O1：两臂参数量、逐 tensor 初值完全一致；
- O2：两臂均有非零有限梯度，checkpoint 冻结，保存重载评价差不超过 `1e-7`；
- P1：directional 训练后至少一个有效节点的 `|u| > 1e-4`；
- P2：directional held-out positive coverage 高于起点，且高于 radial 至少 `0.002`；
- P3：directional 的 held-out 全正概念 hard-hit 次数高于 radial；
- P4：directional 的负概念激活与期望重复均不比起点恶化超过 `0.05`。

P2/P3 是不同层级证据。只通过 P2 表示 soft 路由可训练，不表示生成能力已经形成；P3 才表示
至少越过了离散读出边界。Smoke 不授权替换默认 FOLD，也不因某次 NLL 非单调而提前中止。

## 5. 否证与后续边界

若 O0--O2 失败，本次实验无效并修复工程合同。若方向头只改善训练集、held-out 不优于等参数
径向头，当前行为目标不足以学习可迁移方向。若 soft coverage 改善但 hard generation 不变，
保留方向参数化为候选，但下一步应改进 READ/detail 接口或训练目标，不直接延长同一 smoke。

## 6. Smoke 结果

io taskd 403 完成两臂各 90 steps，用时 168.2 秒。输入哈希复核通过；GPU 保持 270 W 上限，
最高温度 65 C、最高显存 1118 MiB，未见 OOM、CUDA、NaN/Inf 或重载故障。

两臂参数量均为 30,890，逐 tensor 同初始化，step-zero 文本与 native 完全一致，WMT NLL 差
均为 0。两臂都有非零有限梯度，冻结的 106M checkpoint 未改变，保存重载一致。因此 O0--O2
全部通过，实验合同有效。

方向参数确实离开零点。base 的 u 范围为 [-0.28844, 0.27684]，extra 为
[-0.26233, 0.17530]，能量闭合误差低于 2.5e-7，P1 通过。这只证明公式可训练。

| 状态 | train positive | test positive | hard all-positive / 9 |
|---|---:|---:|---:|
| step 0 | 0.631262 | 0.271733 | 5 |
| radial | 0.630872 | 0.242628 | 4 |
| directional | 0.603751 | 0.237401 | 3 |

directional 的 held-out positive 比 radial 低 0.005227，hard all-positive 也少 1 次，所以
P2/P3 均未通过；P4 行为健康门通过。固定 WMT 小样本中 directional NLL 比 radial 高
0.017339，只作为一致的负面观察，不单独裁决实验。

分深度观察并非一律失败：directional 在 depth 6 和 7 的 soft coverage 高于 radial，但在
depth 5 明显较差，汇总后仍然落后。这提示方向自由度存在，却没有被当前共享投影和行为目标
组织成跨深度一致的协议。

## 7. 结论

F11 支持“有界正交方向头可以稳定训练”，不支持“当前词汇行为目标能够把方向头训练成
可迁移的组合协议”。不能直接延长同一训练，也不能替换默认 FOLD。下一步应先解释深度间
差异，分别审计逐深度方向分布与共享 projection 的梯度干扰，再决定修改训练目标还是 READ
接口。

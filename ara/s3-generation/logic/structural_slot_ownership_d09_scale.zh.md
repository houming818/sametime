# D09：结构槽位协议的第一档规模训练

日期：2026-08-29

Claim：`S3-STRUCTURAL-SLOT-OWNERSHIP-D09-SCALE`

状态：正式单 seed 支持（`scale_rung_supported`）；尚未达到产品可用。

## 1. 依据

D08R1 在三个 seed、三个递归深度上正式支持：相邻 subheap ownership 相对自由读取和
同容量稳定随机分组均降低 Test NLL，并通过 shuffle/zero 输入因果门。但中型训练的
BLEU4 中位数只有 `2.79--3.66`，因此只证明结构机制，没有证明生成产品。

D09 不增加新算子，不同时更换语料。它只扩大已支持 arm 的数据暴露和训练步数，回答：
结构协议继续学习后，能否把机制收益转成可测生成收益。

## 2. 固定模型与初始化

```text
source checkpoint: C12 formal seed10101 READ
protocol warm start: D08R1 formal seed10811 subheap final
warm-start rule: 使用第一个预注册 seed，不按测试结果挑选
ownership: adjacent recursive subheap
max slots: 32
depth schedule: balanced 5 / 6 / 7
source TreeHeap: frozen
language backbone: frozen
trainable: compressor READ/slot、reconstructor READ/K_up、bounded gain
loss: 完整目标 token cross entropy
```

保持语言骨架冻结是为了让本阶只检验结构协议能否继续生长。若本阶 BLEU 不提高，不能用
“全模型还没解冻”事后解释成通过；应先记录失败，再为联合训练另开 Claim。

## 3. 规模与资源

```text
seed: 10901
WMT train / valid / test: 200000 / 1000 / 1000
batch: 16
maximum steps: 25000
wake interval: 2500 steps
learning rate: 0.0005
single RTX 3090
power limit: <= 270.5 W
```

训练行仍来自 D08 使用的原始 WMT massive 固定切分，并限制 source 长度不超过 34 pieces。
本阶不声称覆盖长句，也不把高置信清洗语料混入，避免同时改变两个变量。

## 4. Wake、恢复与停止

每个 wake 记录：

- 三个深度的 valid NLL/PPL、slot 方差、槽间方差和 route 指标；
- 固定样例的生成、BLEU4、非空率与相邻重复率；
- trainable checkpoint/optimizer、step、数据哈希和 checkpoint SHA-256；
- wall time 与 GPU 状态。

保存 `checkpoint_latest.pt` 和 valid mean NLL 最低的 `checkpoint_best.pt`。中断后必须从
latest 的 optimizer 和 step 继续。step >= 10000 后，连续三个 wake 没有相对历史最佳改善
至少 `0.005` NLL，则提前停止，不把平台时间消耗在平台期。

OOM、CUDA 错误、NaN/Inf、冻结哈希改变、数据哈希改变或 checkpoint 重载失败立即停止。

## 5. 预注册门

### S0：合同

- source、语言骨架训练前后哈希不变；
- warm-start trainable 哈希与 D08R1 seed10811 checkpoint 一致；
- 训练/验证/测试行数与 SHA-256 完整；
- 无 Transformer、自注意力、flat `L x L` 路由和 target 输入 compressor。

### S1：规模学习

best 三深度 mean valid NLL 相对 step 0 改善至少 `0.10`。

### S2：输入因果

best checkpoint 至少两个深度同时满足：

```text
NLL(shuffle) - NLL(native) >= 0.10
NLL(zero)    - NLL(native) >= 0.10
```

### S3：结构与数值

- owner coverage 与 argmax coverage 均为 `1.0`；
- route overlap 小于 `0.05`；
- slot variance 大于 `1e-4`，between-slot variance 大于 `1e-8`；
- 全程梯度和 loss 有限。

### S4：生成趋势

best checkpoint 三深度 BLEU4 中位数同时满足：

```text
BLEU4 >= 5.0
BLEU4 - initial BLEU4 >= 0.5
```

且非空率 `1.0`、严重相邻重复率不高于 `0.10`。

### S5：重载

重新构造模型并加载 best trainable state 后，固定 valid 子集的 NLL 差小于 `1e-9`，state
SHA-256 完全一致。

## 6. 决策

- S0/S2/S3/S5 任一失败：结构或证据失效，停止；
- S1 失败：当前规模训练没有继续学习，停止扩容；
- S1 通过但 S4 失败：机制继续优化但没有转成生成收益，不进入更大规模；
- S0--S5 全通过：允许设计下一档长度/语料扩展，不自动启动全量训练。

Evidence：

```text
ara/s3-generation/evidence/s3_structural_slot_ownership_d09_scale/
```

## 7. 正式结果

任务 `335` 在 io 的 RTX 3090 上完成全部 `25000` steps，耗时 `6430.48` 秒
（约 `1.79` 小时），没有提前停止。最佳 checkpoint 出现在最后一步，而不是中间 wake。

| 指标 | step 0 | best step 25000 | 变化 |
|---|---:|---:|---:|
| 三深度 mean valid NLL | 5.6862 | 5.2633 | -0.4229 |
| BLEU4 中位数 | 3.7088 | 11.3552 | +7.6464 |
| 最大相邻重复率 | - | 0.0411 | 低于 0.10 门槛 |
| 非空率 | 1.0 | 1.0 | 保持 |

各深度结果：

| depth | 初始 NLL | best NLL | 初始 BLEU4 | best BLEU4 |
|---:|---:|---:|---:|---:|
| 5 | 5.8398 | 5.3400 | 3.9679 | 11.9152 |
| 6 | 5.5965 | 5.2167 | 3.7088 | 11.3552 |
| 7 | 5.6224 | 5.2333 | 3.6300 | 9.8092 |

输入因果审计也通过。对 best checkpoint 打乱 TreeHeap 状态后，三个深度的 NLL 分别增加
`1.9958 / 2.1097 / 1.9680`；把状态清零后分别增加
`1.2068 / 1.2882 / 1.2787`。因此生成改善不能解释成冻结语言骨架忽略输入后独立输出。

结构指标中，三个深度的 owner coverage 和 argmax coverage 均为 `1.0`，route overlap 为
`0.0`，槽内与槽间方差均非零。重新加载 best checkpoint 后 state SHA-256 一致，mean NLL
差为 `0.0`。S0--S5 六个预注册门全部通过。

## 8. 结论边界

D09 支持一个明确但有限的结论：D08R1 找到的相邻递归 subheap ownership 协议，在扩大到
20 万训练样本和 2.5 万步后，仍能继续降低 NLL，并把结构机制收益转成明显的 BLEU4
收益。它没有在本档规模下退化成只会通过冻结语言骨架输出的旁路。

但生成样例仍存在错译、未知字符、语言方向混乱和局部重复。BLEU4 `11.36` 是规模学习
成功的证据，不是商业翻译质量证明。由于最佳点仍在最后一步，下一步可以预注册一次相邻
训练档，区分“同一数据继续训练”与“增加高质量数据/长度覆盖”的边际收益；不得直接把本次
单 seed 结果外推为全量训练必然成功。

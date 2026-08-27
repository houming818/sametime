# D05：递归深度的自由生成长度响应

日期：2026-08-27

Claim：`S3-RECURSIVE-DEPTH-FREE-LENGTH-D05`

状态：`preregistered`

## 1. 修正 D04 的问题

D04 用 `block_size=1/2/4` 人工减少 Decoder 的执行步数，因此短输出是实验脚本制造的，
不是模型根据 TreeHeap 分辨率自主产生的。D05 删除目标分组、目标 embedding 均值和
teacher forcing，只观察冻结 checkpoint 的自由生成行为。

## 2. 实验问题

对相同输入、相同参数和相同生成预算，仅改变 Decoder 可读取的最深 TreeHeap 层：

```text
depth 5：较粗状态
depth 6：中间状态
depth 7：完整状态
```

所有深度统一最多生成 256 个 token。Decoder 从 BOS 开始逐 token 生成，只有生成 EOS
才自然停止。实验不能把 depth 5 或 depth 6 的 `max_output` 预先缩短。

若 TreeHeap 已经形成“深度对应分辨率”的私有协议，则冻结模型应当表现出稳定的长度
响应。以 depth 7 为一倍尺度，强预测为：

```text
median(L6 / L7) ~= 1/2
median(L5 / L7) ~= 1/4
```

## 3. 多次生成

每条 WMT test 输入执行：

- greedy 生成 1 次，用于确定性复查；
- temperature `0.8`、top-p `0.9` 随机生成 8 次，用于估计长度分布；
- 三个深度使用相同随机种子序列，减少抽样噪声；
- 三个独立训练 seed `10101/10102/10103` 复现。

正式实验每个 checkpoint 使用相同的 128 条 test rows。smoke 先使用 seed `10101`、
16 条 test rows、每条 3 次采样。

## 4. 主要指标

本实验不使用 NLL 判定 Claim。主要记录：

1. 各深度自由生成长度的均值、中位数和四分位数；
2. 同一句、同一采样编号下的配对比值 `L5/L7` 与 `L6/L7`；
3. 满足 `L5 <= L6 <= L7` 的配对比例；
4. EOS 命中率与 256-token 触顶率；
5. greedy 与随机采样是否给出相同方向；
6. 重复率、非空率、BLEU 和可读样例只作为退化诊断，不作为长度 Claim 的主判据。

## 5. Predict

### P0：实现与证据合同

- 三个深度使用同一输入、同一 checkpoint、同一 `max_output=256`；
- 生成过程不读取 target，不使用 teacher forcing，不按目标长度分组；
- checkpoint 哈希与 test-row 哈希完整，所有长度统计有限；
- depth `5/6/7` 均是该 TreeHeap 的合法 READ 深度。

### P1：自然单调长度响应

每个 seed 的随机采样中位数满足：

```text
median(L5) < median(L6) < median(L7)
```

且配对样本中至少 `60%` 满足 `L5 <= L6 <= L7`。至少两个 seed 通过，才支持自然
单调响应。

### P2：近似二倍尺度

至少两个 seed 同时满足：

```text
0.35 <= median(L6/L7) <= 0.65
0.125 <= median(L5/L7) <= 0.375
```

这是强预测。P1 通过而 P2 失败，只说明深度影响自然长度，不能声称每层精确对应二倍
分辨率。

### P3：EOS 可观测性

三个深度的随机采样触顶率均不超过 `10%`。若任一深度大量触顶，则该深度的真实长度
被右删失，长度比例判为 `inconclusive`，不能把 256 当作真实长度。

第一次 smoke 使用共同上限 64，观察到 depth 5/6/7 触顶率分别为
`75%/25%/2.08%`。这说明 64 不能测量浅层自然长度。该结果只用于测量窗口预检，
在三 seed 正式实验前将所有深度的共同上限统一扩大到 256；Predict 和配对判据不变。

### P4：跨 seed 复现

至少两个 seed 通过 P1；其余 seed 不得呈现稳定反向次序。否则不升级 Claim。

## 6. 结论边界

- P1/P2 通过：支持当前冻结协议中存在深度到生成长度的自然映射。
- 仅 P1 通过：支持单调长度响应，不支持精确 `1/4, 1/2, 1` 尺度。
- P1 失败：当前 checkpoint 未显示自然长度分辨率协议。
- P3 失败：生成预算不足或 EOS 机制失效，结果只算删失观察。

即使 P1/P2 全部通过，也不等于较浅输出具有正确摘要语义；语义保留需要下一项独立
Claim 验证。

## 7. Evidence 位置

```text
ara/s3-generation/evidence/s3_recursive_depth_free_generation_length_d05/
```

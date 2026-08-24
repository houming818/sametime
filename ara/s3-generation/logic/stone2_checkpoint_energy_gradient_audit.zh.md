# STONE-2 checkpoint 能量与梯度旁路审计

状态：预注册后执行

## 目的

`stone2_recursive_energy_carrier_smoke` 证明了一个确定性数值事实：当前逐层局部
归一化在高能量反向抵消树上会产生额外的递归梯度放大，而携带绝对能量的节点
状态可以消除该特例。但人工 alternating tree 不能证明自然语言 checkpoint 中
真的存在这一病灶。

本审计冻结 `S3-STONE2-INTEGRATED-C03` 的 pretrain checkpoint，读取其训练后
token embedding 与 Native XOR Butterfly 输出，在真实自然文本验证窗口上旁路
运行两种 FOLD。它不训练 decoder，也不把未适配 decoder 的 NLL 当作公式优劣。

## 固定输入

- checkpoint：C03 smoke seed `16101` 的 pretrain `checkpoint_best.pt`；
- tokenizer、语料路径和 split：完全读取 checkpoint config；
- 宽度：`4/8/16/32/64/128/256`；
- 每个宽度最多 32 个真实验证窗口；
- FOLD 数值审计转换为 float64；
- 固定 root probe，不搜索对候选有利的方向。

## 两条旁路

1. `local-current`：当前每层从左右方向状态重新计算 scale；
2. `energy-carrier`：节点携带 `(direction, absolute_energy)`，同层能量用平方和，
   沿 root-to-leaf 地址用无量纲比例连乘。

两条旁路都必须保存 detail/scale 并完成数值闭包。正式 checkpoint 参数保持冻结。

## 观测量

每一层、每个有效 sibling pair 计算共同方向比例：

```text
q = ||left + right|| / (||left|| + ||right|| + tiny)
```

`q` 越接近零，表示两个非零子树越接近反向抵消。另记录：

- 当前局部 scale 与 `1/scale` 代理；
- parent norm；
- `q < 0.01/0.05/0.10` 的比例；
- 每个样本 root probe 对全部 leaf 的梯度 norm；
- current/carrier 梯度比的 p50/p90/p99/max；
- 样本最小 `q` 与梯度比的 log-Pearson 相关；
- 两条旁路的 FOLD/UNFOLD 闭包误差与有限性；
- carrier 的路径比例乘法能量重建误差。

## 预注册判断

- A0：任一非有限值、闭包误差 `>=1e-6` 或路径能量误差 `>=1e-8`，停止；
- A1：全部 pair 中 `q<0.05` 比例至少 `0.1%`，才认为真实数据存在可测抵消尾部；
- A2：current root-gradient p99 至少为 carrier 的 2 倍，才认为候选改善尾部；
- A3：`-log10(min q)` 与 `log10(current/carrier gradient)` 的相关至少 `0.2`，
  才把梯度差异归因于抵消，而不是普通尺度差异。

只有 A0 通过，且 A1 与 A2/A3 至少形成一条一致证据链，才允许注册短训练
ablation。若 A1 不通过，toy 病灶在当前自然数据上缺少存在性；若 A1 通过但
A2/A3 均不通过，能量载体没有改善实际 Jacobian，停止修改 FOLD。

## 边界

该审计只回答当前 checkpoint 的数值条件性，不回答语言质量、语义、S7 多深度
贡献或产品效果。它不能用来宣称新 FOLD 更优，也不允许在审计后直接启动长训练。

## 执行结果

最终任务：io taskd `301`。共审计 224 个真实验证窗口，每个
`4/8/16/32/64/128/256` 宽度各 32 个。

### 抵消存在性

全部节点没有一个进入 `q<0.10`；全局最小 `q=0.58384`，样本最小 `q` 的中位数
为 `0.65610`。A1 失败。这说明人工 alternating tree 中的近反向抵消病灶没有在
当前 C03 训练后自然文本状态里出现。

### 梯度随宽度变化

| width | current 梯度 p50 | carrier 梯度 p50 | current/carrier p50 |
|---:|---:|---:|---:|
| 4 | 0.04459 | 0.03131 | 1.419 |
| 8 | 0.04404 | 0.02183 | 2.023 |
| 16 | 0.04111 | 0.01524 | 2.690 |
| 32 | 0.03622 | 0.01076 | 3.408 |
| 64 | 0.03035 | 0.00757 | 4.005 |
| 128 | 0.02153 | 0.00528 | 4.068 |
| 256 | 0.01483 | 0.00370 | 3.994 |

候选的较小梯度不是已证实的稳定收益。它随宽度从 4 到 256 下降约 8.5 倍，接近
叶数平方根增长对应的衰减；当前公式只下降约 3 倍。也就是说，当前逐层重新
归一化虽然在人工抵消点可能病态，却在这批真实状态中保留了更多长深度梯度。
在没有真实抵消事件时，以 carrier 替换它可能先削弱长程学习。

A3 的 log 相关为 `0.7283`，但它发生在 `q=0.58..0.72` 的非抵消窄区间，并与
句长/深度共同变化；由于 A1 失败，不能把该相关解释为抵消因果。A2 也未通过：
全样本 current/carrier 各自 p99 的比例不足预注册的 2 倍。

两臂闭包最大误差约 `4.0e-15`，路径乘法能量误差约 `1.07e-14`，A0 通过。
最终判定：`do_not_train_energy_carrier_yet`。

## 航线决定

不启动 energy-carrier 训练 ablation，也不修改正式 FOLD。toy 作为边界反例保留，
但当前 S7 失败不能归因于自然数据中的反向抵消。下一项数学审计应回到真实 task
loss：分离各深度对共享 READ、depth embedding、branch 和 recurrent decoder 的
参数梯度 Gram，定位负交互究竟发生在哪个参数组。

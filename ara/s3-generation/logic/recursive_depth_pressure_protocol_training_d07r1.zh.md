# D07R1：冻结语言骨架后的压力协议训练

状态：预注册，等待 proof。

Claim：`S3-RECURSIVE-DEPTH-PRESSURE-PROTOCOL-D07R1`

## 修正依据

D07 的 zero 协议比 native 协议低 `1.51-1.90` NLL。原因不是梯度不存在，而是继承的
`embedding/GRU/output` 与协议模块同时训练后，decoder 可以独立学习目标边际分布。

D07R1 保持数据、容量公式、协议槽位、递归 FOLD、三深度共享参数和全部 Gate 不变，
只增加以下隔离：

- 冻结 source TreeHeap；
- 冻结重建 decoder 继承的 token embedding、GRU、output、query、branch 与 depth
  embedding；
- 只训练压缩 READ、槽位合成、协议 FOLD 上方的 `K_up`、重建 READ 及其 gain。

完整目标交叉熵仍是唯一主损失。目标不进入压缩器，目标长度不进入容量公式。

## Predict

- R1-P0：两个冻结骨架训练前后哈希不变，梯度仅存在于协议参数。
- R1-P1：至少两个深度的 valid NLL 比初始化下降 `>=0.10`。
- R1-P2：至少两个深度中，shuffle 与 zero 都使 test NLL 增加 `>=0.10`。
- R1-P3：`NLL7 <= NLL6 + 0.05` 且 `NLL6 <= NLL5 + 0.05`。
- R1-P4：压缩 READ 与重建 READ 均有非零有限梯度，native 槽位方差大于 `1e-4`。

如果 R1-P2 仍失败，说明当前从 source TreeHeap 到有限协议槽位的映射不能形成可用的
输入条件，不允许用更多 steps 或更多语料掩盖该失败。

Evidence：

```text
ara/s3-generation/evidence/s3_recursive_depth_pressure_protocol_d07r1/
```

## Smoke 结果

taskd `329` 完成 300 steps。冻结合同成立，三个深度 valid NLL 降到
`8.049/7.998/7.997`，P1、P3、P4 通过。shuffle 相对 native 的 NLL 损失增大到
`+0.587/+0.618/+0.617`，证明协议已经包含样本相关信号。

但是 zero 相对 native 仍然更好 `1.440/1.360/1.358` NLL，R1-P2 失败。当前结果应解释
为：协议含有输入信息，但满幅协议 state 与冻结语言 decoder 的参考坐标不匹配，整体
偏置大于样本信息的收益。正式训练继续被阻止。

# TreeHeap 重复信号相干累积审计

日期：2026-09-11  
Claim：`S3-TREEHEAP-COHERENT-ACCUMULATION-F14`  
状态：只读 smoke 已完成；完整 READ 重复响应通过，root 相干累积解释未通过。

## 1. 问题

当前原生 FOLD 对一对有效子节点使用：

\[
p=(l+r)/\sqrt{2}
\]

因此，一个只出现于单个 leaf 的分量会随递归深度衰减；在多个 leaf 中方向一致的分量却可能
按 \(\sqrt{k}\) 相干累积。此前 `push` 探针在自由生成中长期不出现，重复输入又可能使它在
更高分辨率节点上越过 Decoder 的离散边界。本实验区分三个机制：

1. FOLD 的 normalized sum 使重复方向在 coarse 节点累积；
2. READ 的逐层查询或分支路由产生非线性跳变；
3. 变化只来自输入长度、tokenization 或自由生成历史。

## 2. 固定输入合同

- 使用冻结的 E01 TreeHeap-106M checkpoint；
- 方向固定为中译英，目标 token 固定为 SentencePiece 的 `push`；
- source 固定为 8 个单 token 位置，取值为 `推` 或中性控制 `看`；
- 重复计数为 `0, 1, 2, 4, 8`；对 1/2/4 使用四个确定性位置排列；
- 所有 source 都含相同方向 token、8 个内容 token 和 EOS，并统一 pad 到模型 leaf width 16；
- protocol depth 分别为 5、6、7。

固定 token 长度消除字符串长度和动态 batch width 的影响。这是合成机制探针，不代表自然 WMT
质量。后续若机制成立，再增加真实语料中的 `push`、`like`、`want` 配对样本。

## 3. 观察量

对每个样本记录目标 token 的 logit、概率、词表 rank、竞争 margin 与分布熵：

\[
M_{\mathrm{push}}=z_{\mathrm{push}}-
\log\sum_{j\ne\mathrm{push}}\exp z_j
\]

观察分为四组：

1. `cumulative READ`：从 root 开始，分别在每个 tree level 截止，记录累计 READ 的输出；
2. `depth ablation`：保持完整 root-to-leaf 路由，只去掉某一层的 READ kernel 更新；
3. `FOLD scale`：比较原生 `(l+r)/sqrt(2)` 与 `(l+r)/2`，后者只作因果干预；
4. `group gradient`：在零点计算 focus loss 对 base/extra 每个 merge/group 的梯度，一次反传
   同时获得全部局部方向。

自由生成只作输出观察，不参与门槛。模型状态在实验前后必须逐 tensor 完全一致。

## 4. 预注册判据

- O0：`推`、`看`、`push` 均为单 SentencePiece token，所有 source 真长度和 pad width 完全相同；
- O1：零参数 grouped FOLD 与 native 的最大 logits 误差不超过 `1e-7`，所有值有限；
- O2：冻结 checkpoint 在实验前后逐 tensor 完全一致；
- P1：native root 截止状态下，count 8 的平均 target margin 高于 count 1；
- P2：native 完整 READ 下，count 8 的平均 target margin 高于 count 1；
- P3：root 截止状态下，native 从 count 1 到 count 8 的 margin 增量高于 mean-FOLD；
- P4：去掉 root READ update 对 count 8 的损害大于对 count 1 的损害。

P1/P3 支持 coarse normalized-sum 累积，P2 表示该累积能传到最终 Decoder，P4 表示 root READ
对重复响应具有因果贡献。任一失败都保留结果，不以 NLL 或生成质量提前停止。

## 5. 结论边界

即使 P1--P4 全部通过，也只说明当前 checkpoint 在受控重复 token 上存在该机制，不证明自然
句子中的 `push` 缺失完全由 FOLD 引起。若 root 累积存在但完整 READ 不响应，应优先审计 READ；
若 native 与 mean-FOLD 无差异，应检查 compressor、UP convolution 或 Decoder；若各层都没有
目标方向，则问题更可能位于语料、词表或语言底座。

## 6. Smoke 结果

首次 io taskd 407 在输入合同阶段失败：预注册实现沿用了 width 32，但该 checkpoint 的
Butterfly leaf 上限为 16。任务未进入模型计算，也未改动参数。修正为 width 16 后，taskd 408
在冻结的 TreeHeap-106M checkpoint 上完成 14 个固定长度样本、三个 protocol depth 的审计，
运行 3.98 秒。`推`、`看`、`push` 都是单 token，source 真长度统一为 10，零参数 grouped
FOLD 与 native logits 误差为 0，checkpoint 逐 tensor 未改变；O0--O2 通过。

三个 depth 平均后的结果为：

| `推` 计数 | 0 | 1 | 2 | 4 | 8 |
|---:|---:|---:|---:|---:|---:|
| root margin | -15.6896 | -14.6296 | -13.7116 | -13.5912 | -14.8898 |
| full READ margin | -14.1778 | -12.9832 | -12.4918 | -11.1506 | -9.6214 |

完整 READ 的 margin 随计数单调改善，count 8 比 count 1 高 3.3618，P2 通过；但 root 在
count 8 比 count 1 低 0.2602，P1 未通过。分 depth 看，root 的 count 8 减 count 1 分别为
`+1.78, +1.62, -4.19`，失败主要来自 protocol depth 7。

mean-FOLD 的 root margin 从 count 1 的 -12.4208 改善到 count 8 的 -11.0233，增量 1.3975；
native normalized-sum 的对应增量为 -0.2602。因此 `(l+r)/sqrt(2)` 没有表现出预期的 root
优势，P3 未通过。去掉 root READ update 的平均损害在 count 1 为 0.3251，在 count 8 反而
只有 0.1571，P4 也未通过。

逐层累计结果进一步定位了变化。depth 7 中，count 8 相对 count 1 的 margin 差在 cutoff 3
仍为 -0.93，到 cutoff 4 变为 +4.74，最终 full READ 为 +3.61。levels 按 root-to-leaf
排列，cutoff 4 是倒数第二层。因此重复信号的有效读出发生在近 leaf 的累计 READ 与路由组合，
不是 root 单节点激活。所有自由生成仍未出现 `push`，说明 margin 尚未跨过 argmax 生成边界。

## 7. 当前结论与下一步

本实验否定了“重复 `推` 主要经 normalized-sum 在 root 相干放大，再激活 `push`”这一简单
机制。重复确实提高最终 target margin，但证据指向 near-leaf 累计 READ 与跨层路由。下一步
应针对 cutoff 3 到 cutoff 4 保存 frontier、branch score、READ kernel update、base/extra
logit 分量及每个 parent 的干预响应；在定位前不修改 FOLD。

# TreeHeap 重复信号相干累积审计

日期：2026-09-11  
Claim：`S3-TREEHEAP-COHERENT-ACCUMULATION-F14`  
状态：预注册，待只读 smoke。

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

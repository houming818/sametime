# F10：TreeHeap 有界正交退火显微镜

日期：2026-09-10  
Claim：`S3-TREEHEAP-ORTHOGONAL-ANNEALING-MICROSCOPE-F10`  
状态：预注册，等待 smoke。

## 1. 问题

F04--F09 只沿现有隐态方向做标量放大或缩小。结果证明径向滤镜能够改变 Decode，也发现少数
祖先链具有局部敏感性，但完整西西弗句仍无法生成 hard `push`。因此下一阶梯不再扩大标量
增益范围，而是检验左右子状态之间的方向自由度。

本实验不直接替换正式 TreeHeap，也不训练完整模型。它先回答三个较小的问题：

1. 一个零点严格等于当前 FOLD 的正交参数化能否稳定运行；
2. 左右方向变化能否产生径向增益无法表达的 Decode 变化；
3. 当前冻结 Decoder 是否能够读取正交分解产生的 detail。

## 2. 预注册公式

对有效左右子状态 `l`、`r`，先构造共有分量和差异分量：

\[
c=\frac{l+r}{\sqrt 2},\qquad d=\frac{l-r}{\sqrt 2}.
\]

用有界参数控制方向：

\[
u=0.5\tanh(a),\qquad z=\frac{1}{\sqrt{1+u^2}}.
\]

正交输出为：

\[
p=z(c+ud),\qquad q=z(-uc+d).
\]

`p` 是继续向 root 递归的 parent，`q` 是停留在当前尺度的 detail。它满足：

\[
\lVert p\rVert_2^2+\lVert q\rVert_2^2
=\lVert l\rVert_2^2+\lVert r\rVert_2^2.
\]

当 `a=0` 时，`u=0`，所以 `p=(l+r)/sqrt(2)`，严格回到当前 FOLD。`|u|<0.5` 保证第一阶梯
不会把任一子树权重压到零，也不引入角度周期。

## 3. READ 观察方式

当前 Decoder 每个深度只接收一个同维状态，不能无损地同时接收 `p` 和 `q`。因此固定同一个
递归过程，只改变暴露给 READ 的内部层：

| 模式 | 递归状态 | READ 看到的内部层 | 含义 |
|---|---|---|---|
| native | 当前平均 FOLD | parent | 原始基线 |
| parent | `p` | `p` | 只改变递归方向 |
| detail | `p` | `q` | 诊断冻结 Decoder 能否直接读差异分量 |
| joint | `p` | `(p+0.5q)/sqrt(1.25)` | 受控联合 READ，不宣称无损 |

所有模式的 leaf 层保持原样。`detail` 和 `joint` 只是接口显微镜；它们不等价于最终双通道
架构，也不能以 step-zero 不一致作为失败理由。

## 4. Smoke 合同

- 冻结 E01 第二遍语料的 TreeHeap-106M checkpoint；
- 使用 F08 的单次 `推`、八次重复 `推`、完整西西弗句和四次重复短句；
- 扫描 `a in {-4,-2,-1,-0.5,0,0.5,1,2,4}`；
- base/extra 使用同一方向参数，但分别验证能量闭合；
- 保存固定原生历史 logits、自由生成、`push` coverage/rank/hard hit；
- 只允许硬件、非有限值、冻结参数改变或证据损坏使 smoke 无效。

## 5. 判定

- O0：`parent, a=0` 与 native logits 的最大差不超过 `1e-7`；
- O1：有效节点最大相对能量闭合误差不超过 `1e-5`；
- O2：冻结模型逐 tensor 不变，全部观测有限；
- O3：零点方向参数得到非零有限行为梯度；
- P1：正负方向在至少一个非重复完整句上产生不同 coverage 或 logits；
- P2：若完整句出现 hard `push`，只记录为冻结邻域可达性，不提升为学会翻译；
- P3：若 detail/joint 改善 soft coverage 但没有 hard 文本，只说明 Decoder 对差异分量敏感。

无论 smoke 结果如何，本实验都不授权替换默认 FOLD。只有 O0--O3 通过且方向干预出现可重复的
增量，才设计下一步内容条件化参数训练。

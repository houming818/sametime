# F06：TreeHeap 解析滤镜干涉显微镜

日期：2026-09-09
Claim：`S3-TREEHEAP-ANALYTIC-FILTER-INTERFERENCE-F06`
状态：初始 smoke 在滤镜合同门失败；r1 与 r2 已完成。解析滤镜能够区分结构响应，小信号
干涉主要符合平滑非线性缩放，同时观察到接近边界的离散 Decode 换相。

## 1. 问题

F03 证明逐节点幅度干预可以改变 Decode；F04 尝试让行为 loss 寻找临时滤镜并指导可训练
退火参数；F05 用单点脉冲确认固定宽度下的 FOLD 路径局部性。训练得到的滤镜会依赖目标词和
loss，不能单独揭示 TreeHeap 对已知结构的响应规律。

F06 因此冻结模型与 F04 theta，不优化滤镜。实验使用预先定义、可复现且等 RMS 的解析滤镜组，
观察分布式结构经过递归 FOLD、卷积、READ 和 logits 后的响应及两滤镜非线性干涉。

## 2. 固定合同

- checkpoint：E01 TreeHeap-106M `checkpoint_latest.pt`；
- theta：F04 r1 `filter-guided-theta.theta.pt`，仅作为现有运行态的一部分，完全冻结；
- 标本：`西西弗每天都推着石头上山顶。`；
- depth：`7`，Decoder 观察长度：`24`；
- source heap width：固定 `32`，排除 F05 已确认的 batch 最大长度依赖；
- 分布比较使用基线贪心 token 作为固定 Decoder 历史；自由生成只作为可读输出；
- base 256d 与 extra 384d 通道分别干预，不把两通道响应混为一个结论。

## 3. 解析滤镜组

对有效 internal coordinate，记其 root-to-node 路径长度为 `d`，向右次数为 `r`。预注册六个
滤镜：

1. `uniform-gain`：所有有效坐标相同，作为纯能量缩放对照；
2. `pascal-middle`：原始值为 `C(d,r)`，强调杨辉三角中间路径类；
3. `binomial-left-p025`：原始值为 `0.25^r 0.75^(d-r)`；
4. `binomial-right-p075`：原始值为 `0.75^r 0.25^(d-r)`；
5. `haar-local-sibling`：同一 merge 内相邻 parent coordinate 按偶数 `+1`、奇数 `-1`
   交替，作为局部高通；
6. `depth-fine-minus-coarse`：以 `d` 构造从 coarse 到 fine 的深度斜坡。

除 `uniform-gain` 外，所有滤镜只在有效坐标上做零均值、单位 RMS 归一化；uniform 直接单位
RMS。实际递归干预为：

\[
u_i=\epsilon F_i,\qquad
g_i=\exp\!\left(\log(1.5)\tanh u_i\right).
\]

扫描 `epsilon = +/-0.01, +/-0.1`。归一化用于防止把“滤镜结构差异”误读成“总能量差异”。

## 4. 干涉定义

对预注册滤镜对分别计算基线、A、B、A+B 四次前向。任一观察量 `Y` 的干涉残差定义为：

\[
I_Y(A,B)=Y_{A+B}-Y_A-Y_B+Y_0.
\]

滤镜对为：

- `binomial-left-p025 + binomial-right-p075`；
- `pascal-middle + haar-local-sibling`；
- `pascal-middle + depth-fine-minus-coarse`；
- `uniform-gain + pascal-middle`。

分别在递归 FOLD 树、卷积后树和固定历史 logits 上保存 `L2` 干涉量及相对比例。FOLD 已出现
干涉表示退火递归非线性；FOLD 近线性而卷积后出现表示卷积引入耦合；前两者近线性而 READ/logits
明显非线性，表示干涉主要出现在读出与离散决策。

## 5. 预注册门

- `A0 parity/frozen`：零滤镜复制 READ 与原生路径 logits 最大差不超过 `1e-6`、token 相同，
  模型与 theta 前后逐 tensor 不变；
- `A1 analytic contract`：所有滤镜有限、有效坐标 RMS 为 `1 +/- 1e-6`，非 uniform 滤镜有效
  坐标均值绝对值不超过 `1e-6`；
- `A2 evidence complete`：每个通道、滤镜、epsilon 都产生 FOLD、卷积、路由、logits、自由生成
  和 token 光谱记录；
- `A3 interference complete`：两个通道的四组滤镜对均保存四点干涉结果；
- `A4 no target optimization`：源码不创建 optimizer、不反向传播，模型和 theta 的
  `requires_grad` 均关闭。

A0--A4 是仪器与证据门，不预设哪个滤镜必须改善 `push`，也不以一次自由生成换词宣称语义能力。

## 6. 结果边界

本 smoke 只能说明固定 checkpoint、固定句子邻域中的响应。它不选择默认滤镜，不训练 theta，
不证明杨辉结构是语言规律，也不把强干涉自动解释为好或坏。若观察到稳定差异，下一步才扩展
标本和 epsilon，并用同一解析滤镜检查跨句、跨 depth 的可重复性。

## 7. 初始 smoke 合同失败与 r1 修正

taskd `394` 在产生任何模型响应数据前失败。原因是初始预注册的 `haar-first-branch` 按
root-to-node 的第一步左右分组；固定 32 宽度下，这条 14-piece 左对齐标本的全部有效 internal
coordinate 都位于根的左半树，滤镜在有效域上退化成常数，去均值后 RMS 为 0。该失败说明
第一分支 Haar 不适用于当前稀疏有效域，不能用其输出得出模型结论。

r1 保留其他合同、滤镜、epsilon 和干涉对，只把退化滤镜改为预注册第 3 节所述的
`haar-local-sibling`。它按每层局部 parent 的奇偶位置交替，能够在左对齐子树内部形成可定义的
高通方向。r1 使用新 evidence 目录，不覆盖任务 `394`。

## 8. r1 smoke 结果

taskd `395` 在 RTX 3090 上完成，耗时 24.1 秒，A0--A4 全部通过。模型与 theta 前后逐 tensor
不变，零滤镜 READ 与原生路径一致。两个通道、六种滤镜、四档正负 epsilon 共 48 个干预臂，
以及两个通道各四组滤镜对的干涉证据均完整且有限。

等 RMS 不产生等响应。`epsilon=+0.1` 时，base 通道的 uniform、Pascal、左右二项、局部 Haar、
深度斜坡对应 logits L2 分别约为 `59.96, 33.12, 34.50, 36.07, 35.28, 32.73`。extra 通道
对应值只有 `7.37, 4.74, 5.31, 3.46, 1.77, 4.97`。因此工具已经能区分滤镜结构与通道，
并显示当前 checkpoint 的 base 读出对同幅解析干预明显更敏感；这不等于 base 更有语义。

在 `epsilon=0.1` 的 base 通道，四组干涉的相对残差为：FOLD `2.78%--5.11%`，卷积后
`3.95%--6.83%`，logits `6.35%--9.26%`。最大的 logits 相对干涉来自
`pascal-middle + haar-local-sibling`。干涉在 FOLD 已非零，经过卷积和 READ/logits 后通常扩大，
但这里只测了较大幅值，尚不能区分结构阈值与 `exp(tanh())` 的平滑曲率。

没有解析滤镜恢复完整的目标翻译或生成 `push`。滤镜改变的是附近候选、句尾和重复形态，不能
把换词本身解释为能力改善。

## 9. r2 预注册补充

r2 不修改滤镜、checkpoint、标本、模型、theta、单滤镜 epsilon 或评价方式，只增加两项证据：

1. 把零滤镜 `push/stone` token 光谱写入同一份 summary，避免跨 evidence 手工比较；
2. 对每组四点干涉同时测 `epsilon=0.01` 与 `0.1`。若相对干涉随 epsilon 约十倍下降，优先解释
   为局部平滑非线性；若在 `0.01` 仍保持同量级或出现离散 branch/token 换相，才记录小信号下的
   强结构干涉。r2 使用新 evidence 目录，不覆盖 r1。

## 10. r2 结果

taskd `396` 在 RTX 3090 上完成，耗时 26.3 秒，A0--A4 全部通过。r2 复现 r1 的全部单滤镜
结果，并完整保存 16 组通道、滤镜对、epsilon 干涉记录。模型和 theta 保持逐 tensor 不变。

### 10.1 干涉随幅度近似平滑缩放

base 通道四组滤镜对的相对干涉范围为：

| epsilon | FOLD | 卷积后 | logits |
|---:|---:|---:|---:|
| `0.01` | `0.36%--0.44%` | `0.46%--0.71%` | `0.62%--0.95%` |
| `0.1` | `2.78%--5.11%` | `3.95%--6.83%` | `6.35%--9.26%` |

extra 通道也表现出相同尺度规律：`epsilon=0.01` 时 logits 相对干涉为
`0.17%--0.37%`，到 `0.1` 时为 `2.03%--7.26%`。相对干涉随 epsilon 约一个数量级增长，
优先支持它来自 `exp(tanh())`、递归组合、卷积和 READ 的平滑非线性累积，而不是一个在任意微小
扰动下都保持固定强度的隐藏跳跃。

不过，base 的 `pascal-middle + depth-fine-minus-coarse` 与
`uniform-gain + pascal-middle` 在 `epsilon=0.01` 时都已各改变一个固定历史 argmax。连续值干涉
仍小，但部分 Decoder token 已非常靠近决策边界，因此小的连续变化可以产生离散换相。

### 10.2 工具能区分滤镜形状，而非只看总能量

所有滤镜在有效坐标上具有相同 RMS。`epsilon=0.1` 时，base 的 logits L2 响应从
`depth-fine-minus-coarse` 的 `32.73` 到 `uniform-gain` 的 `59.96`；局部 Haar 为 `35.28`，
左右二项分别为 `34.50` 与 `36.07`。因此滤镜位置、符号和路径分布会改变响应，不能用一个
“总能量”标量替代。

同一组滤镜在 extra 通道的 logits L2 只有 `1.77--7.37`，而 base 为 `32.73--59.96`。
这证明仪器可以观察到当前 checkpoint 中两条协议通道的灵敏度差异；它不证明 extra 无用，也不
排除通道输出增益造成的尺度差异。

### 10.3 分层轨迹

leaf 层没有被直接修改，FOLD 响应从第一层 parent 出现并递归传向 root。以 base 的
`pascal-middle, epsilon=0.1` 为例，root-to-leaf 各有效层的 FOLD L2 约为
`1.98, 1.98, 2.02, 2.06, 2.03, 0`；卷积后变为
`2.09, 2.15, 2.40, 2.22, 2.02, 0`。卷积没有抹掉滤镜签名，而是在中间分辨率层重新分配并
放大部分响应。

`pascal-middle + haar-local-sibling, epsilon=0.1` 的相对干涉从 FOLD 的 `4.14%`，增加到
卷积后的 `6.83%` 和 logits 的 `9.26%`。因此干涉不是只在最终 softmax 才突然出现；它从有界
递归增益的曲率开始，经过卷积和 READ 继续积累。

### 10.4 token 光谱

零滤镜下 `push` phrase coverage 为 `0.0013051`，最佳完整 surface 是 `pushed`，rank `438`；
`stone` coverage 为 `0.123102`，完整 token rank `1`。

单滤镜中，base 的 `pascal-middle, epsilon=0.1` 把 `push` coverage 提高到 `0.0013784`
（绝对增加 `0.0000733`，相对约 `5.62%`），完整 surface rank 从 `438` 移到 `415`；base 的
`haar-local-sibling, epsilon=-0.1` 提高到 `0.0013708`、rank `423`。这些变化说明解析滤镜能够
推动目标词光谱，但幅度远不足以越过生成边界，所有输出仍未出现 `push`。

## 11. 当前结论与边界

F06 支持以下存在性结论：

1. 冻结解析滤镜可以作为 TreeHeap 的结构显微镜，分辨路径形状、深度分布和通道差异；
2. 当前滤镜干涉在小信号区主要表现为平滑非线性，并沿 FOLD、卷积、READ/logits 逐级积累；
3. 连续干涉很小时仍可能触发个别离散 token 换相，说明 Decoder 决策边界是必须单独观察的层；
4. 杨辉滤镜在这个标本上把 `push` 向输出边界推动了一小段，但没有形成正确组合生成。

F06 不支持“杨辉滤镜就是理想语言规律”，也不支持将它写入默认架构。下一步若继续，应保持
解析滤镜与 checkpoint 冻结，把同一滤镜组扩展到多个动词、句型和 depth，检验响应指纹是否
跨标本重复；只有重复后，才讨论由哪一种可训练参数去逼近该规律。

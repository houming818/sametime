# F05：TreeHeap 观察仪器箱

日期：2026-09-08
Claim：`S3-TREEHEAP-OBSERVATION-INSTRUMENTS-F05`
状态：初始 smoke 已完成；FOLD 路径局部性得到支持，同时发现 token 子片段误读风险与动态宽度批依赖；r1 待验证。

## 1. 动机

F03 的节点滤镜证明临时缩放能够改变 Decode；F04 又尝试把滤镜当作老师训练退火参数。但这两步
还没有直接展示参数、隐态和 Decoder 路由如何在 TreeHeap 中运动。尤其不能把“代码使用递归”
直接写成“修正沿分辨率路径传播”。

F05 把滤镜恢复为观察仪器。模型参数 `theta` 全部冻结，仪器参数 `phi` 只在一次前向中存在，
不写回 checkpoint。实验先测出现有协议的传递规律，再决定未来是否以及怎样加入路径约束。

## 2. 固定标本

- checkpoint：E01 的 TreeHeap-106M `checkpoint_latest.pt`；
- 句子：`西西弗每天都推着石头上山顶。`；
- 主评价 depth：`7`；
- FOLD 脉冲：单个 internal node 的 `u_i` 取 `+/-0.001`、`+/-0.01`、`+/-0.1`；
- 其他滤镜坐标保持 0；
- 主通道 256d 与额外通道 384d 分开注入；
- 模型权重、输入、Decoder 历史和随机种子保持不变。

脉冲通过现有 FOLD 方程传播：

\[
H_{k+1,i}=\frac{H_{k,2i}+H_{k,2i+1}}{\sqrt{2}}
\exp\!\left(\log(1.5)\tanh u_{k,i}\right).
\]

实验不为祖先节点另设修正。祖先怎样变化完全由现有递归决定。

## 3. 仪器一：参数地图

把完整模型 `state_dict` 与可选 F04 theta 转换为可读表格，至少记录名称、模块、shape、dtype、
参数量、mean、std、min、max 和 L2 norm。它只能展示数值结构和训练漂移，不能把单个维度直接
命名为某个语义。

## 4. 仪器二：路径脉冲显微镜

对每个有效 internal node 单独注入脉冲。对于被注入节点 `i` 和任意节点 `j`，测量：

\[
R_{j\leftarrow i}(\epsilon)=
\frac{\lVert H_j(u_i=\epsilon)-H_j(0)\rVert_2}{|\epsilon|}.
\]

依据固定二叉拓扑把 `j` 分成自身、祖先、兄弟/非祖先。记录：

- 注入节点响应；
- 各级祖先响应及相对传递率；
- root 响应；
- off-path energy ratio；
- 正负脉冲的中心差分与非线性偏差。

这里的路径只指 **FOLD 构树阶段、卷积之前**。理论实现应使非祖先响应为 0；若不为 0，说明
实现、坐标映射或共享状态存在串扰。

## 5. 仪器三：READ 路由示波器

复制但不修改当前 READ 计算，同时记录每个 Decoder step、每个分辨率层的 frontier 概率、
路由熵和 argmax 节点。扰动臂使用基线生成 token 作为固定 Decoder 历史，避免第一个输出 token
变化污染后续比较。

对基线与脉冲臂记录：

- 每层 frontier 的 Jensen-Shannon divergence；
- argmax branch flip 数；
- base/extra 两通道分别的路由变化；
- 最终 logits 与原生 READ 的最大差异合同。

FOLD 路径局部而 READ 路由跳转，表示问题位于读出决策；FOLD 在卷积前已经出现 off-path
响应，才表示构树或滤镜坐标有问题。

## 6. 仪器四：token 光谱

不只保存贪心文本。对 `push/pushes/pushed/pushing` 与 `stone/stones/rock/rocks/boulder/boulders`
的 SentencePiece token，在每个 Decoder step 记录概率和词表排名，并记录整个轨迹中的最佳排名、
最大概率与 phrase coverage。它用于区分“词完全不在候选空间”和“词存在但没有越过 argmax”。

## 7. 预注册检查

- `O0 reader parity`：示波器复制的 READ 与原 READ context/logits 最大绝对差不超过 `1e-6`；
- `O1 topology locality`：卷积前所有有效单点脉冲的 off-path energy ratio 不超过 `1e-12`；
- `O2 ancestor transport`：每个有效脉冲自身响应大于 0，并至少传到一个更高祖先；
- `O3 route observed`：成功保存逐 step、逐 depth frontier；这是证据完整性门，不预设必须换路；
- `O4 spectrum observed`：成功保存目标 token 概率与排名；不预设 `push` 必须上升；
- `O5 frozen exact`：实验前后完整模型 state_dict 逐 tensor 相等。

O0--O5 是仪器可信度门，不是模型质量门。无论路由是否跳转，都保存结果。

## 8. 边界

F05 不训练退火 theta，不寻找最优滤镜，不宣称找到语言语义坐标，也不以一次脉冲决定正式架构。
它回答的是：当前 TreeHeap 在一个固定标本上，局部变化经过 FOLD、卷积、READ 和 logits 时实际
留下了什么轨迹。

## 9. 初始 smoke 结果

taskd `391` 在 RTX 3090 上完成，耗时 7.3 秒。参数地图展开了完整模型的 `112` 个 tensor、
`105,965,594` 个参数，以及 F04 guided theta 的 `6` 个 tensor、`30,890` 个参数。模型和
theta 在观察前后逐 tensor 不变，复制 READ 与原 READ 的 logits/token 一致，O0/O5 通过。

depth 7 的协议预算为 12，base/extra 各有 11 个有效 internal node。两个通道共执行 132 次
单点正负脉冲；所有卷积前响应的 off-path energy ratio 都为 `0`，每个非 root 有效脉冲均改变
自身并传到更高祖先，因此 O1/O2 通过。当前 FOLD 构树没有观察到跨兄弟子树跳跃。

进入 READ 后出现了不同现象。base 的 leaf-near `coord 0, epsilon=0.1` 在 Decoder step 22、
tree depth 5 发生一次 frontier argmax 翻转，但未改变局部输出 token；base 的高层 `coord 28`
在 step 18、depth 3 翻转一次，并改变一个局部 argmax token。两种 extra 脉冲均无分支翻转。
这说明可见的离散跳转发生在 READ 路由，不发生在 FOLD 的卷积前构树阶段；它目前只是
固定标本上的存在性观察，不代表翻转必然有害。

## 10. smoke 后发现的仪器问题与 r1

初始 token 光谱把任意子片段的最佳 rank 汇总为概念最佳值。`pushes` 被 SentencePiece 切成
`[▁p, us, hes]`，其中 `▁p` 曾达到 rank 2，但完整单 token `▁push` 的最好 rank 只有 165。
因此“push 已接近输出”是错误读法。r1 将分别报告完整单 token surface、任意 piece 与 phrase
coverage，禁止用公共子片段替代整词。

此外，固定句单独编码时 source width 为 14、动态 heap width 为 16；与另外两条测试句组成
batch 时 source width 为 17、动态 heap width 为 32。同一句的 step-0 logits 最大差已经明显
非零，贪心文本也不同。源码显示动态宽度改变会增加一个 pass-through root 层，使后续
compressor 看到的 root-to-leaf 层数与 depth embedding 对齐发生变化。这是初始 smoke 之外发现
的异常，尚需正式隔离。

r1 保留初始 evidence，新增 batch/width 检测器：比较动态宽度下 single/batch，以及固定 32
宽度下 single/batch；所有比较固定第一条句子的 Decoder 历史。若动态宽度不一致而固定宽度恢复
到 `1e-6` 以内，才把异常定位为 max-length/heap-width 依赖。r1 的路径、READ 和 token 观察统一
使用固定 32 宽度，保证固定标本不因同 batch 的其他句子改变观察坐标系。

## 11. r1 smoke 结果与等 batch 对照

taskd `392` 完成 r1，耗时 8.3 秒。O0--O5 再次通过：固定 32 宽度下 132 个卷积前
脉冲仍全部严格保持 off-path energy ratio `0`，base leaf-near 脉冲在 READ 中观察到一次
branch flip，extra 未翻转，模型和 theta 均未改变。

修正后的 token 光谱显示：`push` 概念的 phrase coverage 为 `0.001305`；完整单 token
surface 中最好的是 `pushed`，最佳 rank 仅 `438`、概率 `0.000266`。此前 rank 2/3 来自
`pushes` 的公共子片段 `▁p`，不能作为 push 接近成形的证据。`stone` 则达到完整 token rank 1，
phrase coverage 为 `0.123102`。因此当前标本是“stone 已进入强候选，push 仍远离输出边界”。

动态宽度 single/batch 的 step-0 logits 最大差为 `7.97856`，固定 Decoder 历史下全轨迹最大差
为 `8.04538`，并改变 12 个局部 argmax。固定 32 后两值降至 `5.84e-6` 和 `6.20e-6`，文本
完全相同且 argmax 变化为 0。由于预注册门是 `1e-6`，O6 仍严格记为失败，不能在看到结果后
放宽。剩余量级可能来自 batch=1 与 batch=3 使用不同浮点矩阵内核。

为分离该数值因素，r2 新增同 batch size 对照：窄组由目标句复制成三条，宽组保持三条测试句，
两组第一条输入及 Decoder 历史相同，batch size 都为 3；动态模式只让 heap width 从 16 变成
32，固定模式都使用 32。r2 预注册 `O7`：动态模式差异大于 `1e-6`，固定模式差异不超过
`1e-6`。O7 只定位动态宽度依赖，不覆盖 O6 记录。

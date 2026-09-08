# F05：TreeHeap 观察仪器箱

日期：2026-09-08
Claim：`S3-TREEHEAP-OBSERVATION-INSTRUMENTS-F05`
状态：预注册，等待 smoke。

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

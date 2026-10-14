# S3 多分辨率语言课程 A01：深度对齐训练

状态：A01 smoke 完成；原始正式任务与 R1 失败；R2 已预注册，等待 smoke。

日期：2026-10-13

## 问题

T0-R3 已经产生 488 条形式上满足
`core < short < reduced < full` 的教师合成候选，但尚未证明 TreeHeap 能把
这些目标学习为可控的 READ 分辨率。

本实验回答一个窄问题：在相同初始化、样本、四次尺度更新和优化步数下，把
不同信息量的目标对齐到不同 READ 深度，是否比“四个深度都学习完整句”更容易
形成随 READ 深度增长的生成长度和信息量。

## 数据与语义审计

- 候选集：T0-R3 `accepted_curriculum.jsonl`，固定 488 条；
- seed：`20261013`；
- 输入、候选集、SentencePiece、初始化 checkpoint 均记录 SHA-256；
- split 由 `seed + source_line` 的稳定哈希决定，训练、验证、测试互斥；
- 候选仍标记为 `teacher_synthetic`，不称为人工真值。

训练前执行两类独立审计：

1. 使用冻结的本地 `bge-reranker-v2-m3` 比较 `full↔同条尺度` 与
   `full↔打乱尺度` 的分数；
2. 记录阿拉伯数字和否定标记从 full 到各尺度的保留率。

reranker 只提供异源诊断，不是真值。除非出现机械错误、非有限值，或三个压缩
尺度的 aligned 分数全部不优于 shuffled，否则审计结果不作为任意质量停机门。

## 原始匹配双臂

初始化为冻结身份的 C03 全量双语 checkpoint。模型结构和参数数量完全相同，
两臂都允许全部参数继续学习。

对每个动态 TreeHeap，令 UNFOLD 后层数为 `L`，四个观察深度为：

\[
d=(0,\operatorname{round}((L-1)/3),
       \operatorname{round}(2(L-1)/3),L-1).
\]

在指定深度停止 READ：先由已有 branch 概率沿树传播，到该深度后将尚存概率
质量加权求和为 decoder context。没有增加新的尺度标签参数或 `lambda`。

### Arm A：full-control（已被失败证据否决为正式对照）

四个深度都以 `full` 为目标：

\[
\mathcal L_A={1\over4}\sum_{i=0}^{3}
  \operatorname{CE}(D(h_{d_i}),y_{full}).
\]

### Arm B：depth-curriculum

四个深度分别学习 `core/short/reduced/full`：

\[
\mathcal L_B={1\over4}\left[
  \operatorname{CE}(D(h_{d_0}),y_{core})+
  \operatorname{CE}(D(h_{d_1}),y_{short})+
  \operatorname{CE}(D(h_{d_2}),y_{reduced})+
  \operatorname{CE}(D(h_{d_3}),y_{full})
\right].
\]

每层 CE 先按本层非 PAD token 求平均，再对四层等权平均。因此本实验匹配更新
次数和尺度权重，但不声称匹配两臂实际处理的 target token 数；该差异必须报告。

## 测量

在 epoch `0/1/2/4/8/16/32`（smoke 只到 2）保存：

- 每层对其对应 `core/short/reduced/full` 目标的 held-out token NLL；
- 每层自由生成的平均长度、非空率、重复率和唯一输出比例；
- 单样本是否满足 `len(core)<len(short)<len(reduced)<len(full)`；
- 两臂训练更新数、目标 token 暴露量、有限梯度和峰值显存；
- 固定测试样例的四层文本。

不同目标长度之间的原始 NLL 不直接比较优劣。主要比较同一尺度、同一测试集上
Arm B 相对 Arm A 的差值，以及 Arm B 的生成长度是否随深度形成稳定顺序。

## 原始 Claim 与边界

`S3-MULTIRES-CURRICULUM-A01-C01`：深度对齐目标能够在固定小样本合同内，形成
比 full-control 更强的 READ 深度条件生成差异，同时保持 full 深度可训练。

smoke 只验证：数据隔离、同初始化、四深度前向、有限梯度、CUDA、JSON 和
确定性重载路径。质量非单调不得中止。

即使正式结果支持 C01，也只说明该 checkpoint 在该候选集上获得了深度条件行为，
不证明人工摘要质量、一般语言压缩、低功耗优势或优于 Transformer。

## 停止条件

只在以下机械异常停止：OOM、CUDA/NVIDIA Xid、NaN/Inf、输入或 checkpoint
哈希变化、split 泄漏、两臂初始参数不一致、证据 JSON 损坏、270W 功率限制解除。
不得因为中间 NLL、长度顺序或生成质量暂时恶化而提前停止。

## Smoke 结果

任务 `659` 使用 96 条固定候选运行 2 epoch，两臂各完成 24 次更新。机械合同
全部成立：同初始化、同参数量、有限梯度、互斥 split、CUDA 与证据哈希均正常。

独立 reranker 在 core/short/reduced 三个尺度上均以 100% 样本胜率区分同条候选
和打乱候选，平均 aligned-minus-shuffled logit 分别为
`11.6820/14.2890/17.1480`。但规则审计发现数字保留率只有
`0.0851/0.2340/0.5106`，否定保留率只有 `0.1613/0.2258/0.4839`。
因此候选保持整体主题相关性，但关键事实保留存在明确风险。

2 epoch 后，课程臂相对对照臂的同尺度测试 NLL 差为
`-0.3725/-0.1691/-0.0245/-0.0080`。该方向值得继续观测，但两臂的生成长度
都未形成递增阶梯；core 和 short 均跑满 128 token，粗层重复严重。Smoke
因此只释放完整训练曲线，不支持 C01 功能结论。

首次正式任务 `660` 在 full-control 的 coarse READ 上报告非有限聚合 NLL，按
合同人工终止。诊断确认 encoder levels 与 logits 均有限，但部分逐 token CE
达到约 `6.36e37`；FP32 对一批 token 求和时溢出。R1 只把 token loss 的求和与
均值改为 FP64，逐 token CE、模型前向和梯度目标不变，不做裁剪或质量补签。
同时把真正的非有限 token loss 或梯度改为即时异常。任务 `660` 保留为失败记录。

R1 任务 `661` 使用 FP64 loss 聚合后，排除了单纯的求和溢出，但
`full-control` 在第一个 epoch 仍产生非有限梯度并按合同失败。这说明“在 root
和 coarse READ 上强制预测完整句”对当前 C03 decoder 不是数值上可用的对照，
不能继续用增加训练时间来掩盖。任务 `661` 保留为第二条失败记录。

## R2 后继实验：等长度前缀对照

R2 不改模型、READ、优化器、数据、seed 或四个观察深度，只替换已经被否决的
对照目标。对尺度 (s\in\{core,short,reduced,full\})，令压缩目标长度为
(n_s)，R2 对照目标为完整句的等长度前缀并补 EOS：

\[
y^{prefix}_s=\operatorname{EOS}\left(y^{full}_{1:n_s-1}\right).
\]

两臂因此在每个深度具有相同的目标长度、PAD 形状、更新次数和 target token
暴露量：

\[
\mathcal L_{prefix}={1\over4}\sum_s
  \operatorname{CE}(D(h_{d_s}),y^{prefix}_s),
\qquad
\mathcal L_{curriculum}={1\over4}\sum_s
  \operatorname{CE}(D(h_{d_s}),y^{semantic}_s).
\]

唯一有意变化是：对照臂使用机械截断，课程臂使用教师生成的语义压缩。这个
设计测试的是“语义压缩目标是否优于同长度截断”，不再测试已被证明数值病态的
“粗层复述完整句”。

`S3-MULTIRES-CURRICULUM-A01-R2-C01`：在相同初始化、结构、深度、目标长度、
token 暴露量和更新预算下，语义压缩课程相对等长度前缀对照形成更强的深度条件
行为，同时 full 深度保持可训练。

R2 smoke 必须重新验证同初始化、等长度合同、有限 token loss、有限梯度、
CUDA、JSON 和哈希。只有机械合同通过才进入正式运行；正式质量指标仍只观察，
不得因非单调而提前停止。

## R2/R3 执行记录与 checkpoint 审计

任务 `662` 的 R2 smoke 通过同初始化、等参数、等 target token 暴露和有限梯度
合同。2 epoch 后，课程臂相对等长度前缀对照在 core/short/reduced 测试 NLL 上
分别低 `0.0247/0.0331/0.0744`，full 高 `0.0042`；两臂均未形成长度阶梯。

任务 `663`、`664`、`666` 和 `667` 在完整 488 条数据的第 10 个更新附近复现
非有限梯度。定位结果表明，`checkpoint_latest.pt` 的
`encoder.embedding.weight` 存在少数有限但接近 FP32 上限的行：全表最大绝对值
为 `3.20e38`，课程样本 `source_line=5359127` 命中 token `▁Gent`，该行最大值为
`1.50e38`。这一行经 FOLD/UNFOLD 传播后使所有深度 RMS 达到约 `10^36`，并首先
令 `decoder.branch.weight` 梯度溢出。将学习率从 `1e-4` 降到 `1e-5` 不改变该
故障，因此不能把它归因于课程目标或用梯度裁剪掩盖。

同一 C03 航线由原训练过程保存的 `checkpoint_best.pt` 通过任务 `669` 审计：
encoder embedding 最大绝对值为 `1.905`，无非有限参数。后继实验不删除样本、
不修补权重，改用这个已有的干净 best checkpoint，并继续保持两臂同初始化。

任务 `670` 在完整数据上运行 2 epoch，两臂各 `102` 次更新、各暴露 `48,550`
个 target token，数值合同全部通过。课程臂相对对照的测试 NLL 差为
`-0.303/-0.225/-0.266/+0.011`。

任务 `671` 进一步跑满 32 epoch。课程臂相对对照的最终测试 NLL 差为
`-1.366/-0.890/-0.717/-0.085`，但两臂仍未形成按深度递增的生成长度。课程臂
验证 NLL 在 16 epoch 为 `11.76/7.28/5.51/4.71`，到 32 epoch 变为
`11.34/7.81/5.81/4.98`：core 继续改善，而 short/reduced/full 回升；同期训练
NLL 仍持续下降。这是多尺度目标进入不同饱和/过拟合阶段的证据，不是单调质量门。

R4 从相同 best checkpoint、相同 seed、数据、优化器和学习率重新运行到 64
epoch，并观察 32/48/64。只有当 48 或 64 出现新的 held-out 最优点，才继续增加
到 128；否则把 16--64 的区间解释为本合同下的有效训练平台。该规则只决定是否
继续消耗预算，不改写任务 `671` 的完整结果，也不以中途质量波动终止 R4。

## R4 饱和结论

任务 `672` 跑满 64 epoch，两臂各完成 `3,264` 次更新并暴露 `1,553,600`
个 target token；同初始化、同参数量、有限梯度、270W 功率限制和全部 SHA-256
合同通过。课程臂四尺度平均验证 NLL 在 32/48/64 epoch 分别为
`7.49/7.87/8.19`，对照臂为 `8.29/8.73/9.05`。两臂都没有在 48 或 64
出现第二次下降，因此按预注册规则不继续到 128。

课程臂逐尺度最优点为：core `11.34@32`、short `7.28@16`、reduced
`5.51@16`、full `4.71@16`。64 epoch 时训练 NLL 已降到 `0.3709`，而验证
NLL 全部高于各自最优点，说明额外训练主要增加小样本记忆，没有改善 held-out
能力。

最终 64 epoch 测试上，课程臂相对等长度前缀对照的 NLL 差为
`-1.643/-1.268/-0.917/-0.205`。这支持“语义压缩目标比机械截断更适合作为
尺度监督”的诊断性结论；但两臂都未形成 `core < short < reduced < full` 的
生成长度顺序，粗尺度仍存在重复与不稳定 EOS，主观文本也未达到可用缩句质量。
因此 `S3-MULTIRES-CURRICULUM-A01-R2-C01` 只获得 NLL 层面的部分支持，功能门
不通过，不得宣称 TreeHeap 已学会多分辨率语言生成。

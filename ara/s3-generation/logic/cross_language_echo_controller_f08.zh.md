# S3-CROSS-LANGUAGE-ECHO-CONTROLLER-F08：共享滤镜控制器

状态：r1/r2 已完成；测量合同通过，soft 结构响应存在，跨语言 hard Echo 未成立。

## 1. 问题

F07 表明冻结 TreeHeap 对解析滤镜方向具有不同响应，但没有证明这些方向能够形成跨样本规律。
F08 检验：能否仅在少量单次中文词上拟合一组共享滤镜系数，并将它迁移到未参与拟合的中文词，
使英文目标词更容易被 Decoder 读出。

## 2. 数据隔离

共享系数只使用 fit 集：`喜欢 -> like`、`想要 -> want`、`吃 -> eat`、`苹果 -> apple`。

test 集不参与共享系数梯度：`推 -> push`、`搬 -> carry`、`石头 -> stone`，以及“小明吃苹果”
和“西西弗推石头”组合句。每行另有重复输入版本，只作为无滤镜参照。

## 3. 控制器

使用 F07 的 11 个解析滤镜作为基底。对第 j 个输入构造基底 `B_j`，所有输入共享同一系数向量
`a`：

\[
u_j=0.2\tanh(aB_j).
\]

因此每个递归坐标的干预严格位于 `[-0.2,0.2]`。模型和 F04 theta 完全冻结，只优化 11 个
共享系数。优化时固定无滤镜 greedy 历史，减少离散 argmax 跳变；正式评价同时保存固定历史概率
和自由生成。

## 4. 对照

1. `single-native`：单次输入，无滤镜；
2. `repeat-native`：重复输入，无滤镜；
3. `single-shared`：单次输入，使用仅由 fit 集得到的共享控制器；
4. `single-test-oracle`：每个 test 行独立拟合系数，只作为可达性上界，不属于泛化证据。

## 5. 评价与判定

- soft：正概念 phrase coverage、负概念激活、EOS、重复度和行为损失；
- hard：自由生成是否包含完整目标 token 序列；
- 保存每行文本、目标命中、滤镜 RMS/max、共享系数和优化轨迹。

完整性门：模型/theta 冻结、所有值有限、fit/test 隔离、证据完整。

结果解释：

- shared 在 test soft 与 hard 指标均改善：支持共享跨语言读出方向；
- oracle 改善而 shared 不改善：基底可达，但规律未迁移；
- oracle 也不改善：当前基底或干预位置不足；
- repeat 改善而滤镜不改善：重复语料提供了当前控制器未表达的信息或能量。

任一单词的偶然改善都不能证明通用跨语言 Echo 已完成。

## 6. R1 后的匹配能量控制修订

R1 执行后发现，单词标本只有 1 至 2 个有效 internal parent 坐标，多个结构滤镜因此退化或彼此
重合；共享控制器也接近 `+0.2` 上限。虽然 test 平均 soft coverage 从 `0.15778` 增加到
`0.18194`，hard hit rate 没有变化，且提升可能来自统一增益而非结构规律。

因此在解释 r1 前增加 r2 对照 `test-single-uniform-matched`：对每个 test 行，在其全部有效坐标
施加相同正值，该值严格匹配 shared filter 在该行有效坐标上的 RMS。r2 不改变数据、seed、优化
步数、共享系数训练或其他对照。只有 shared 优于 matched uniform，才能把增量归因于滤镜形状；
否则应解释为当前 internal FOLD 增益的 DC/能量效应。

## 7. R2 结果

运行：`io` taskd `399`，seed `11901`，耗时 `67.78 s`。四个完整性门全部通过，模型和
theta 冻结，所有干预均不超过 `0.2`。

test 集结果：

| 条件 | positive coverage | hard token hit rate |
|---|---:|---:|
| single native | 0.15778 | 0.2857 |
| matched uniform | 0.16119 | 0.2857 |
| shared filters | **0.18194** | 0.2857 |
| repeat native | **0.31140** | **0.4286** |
| test oracle | 0.17720 | 0.2857 |

shared 高于 matched uniform，说明全部 soft 增量不能由总能量解释；但 hard hit 没有增加，且
shared 远低于重复输入。因此“存在可区分的结构响应”得到支持，“共享滤镜已经完成跨语言 Echo”
未得到支持。

## 8. 逐样本解释

`推。` 只有一个有效 internal parent 坐标，结构滤镜在此退化为单一增益。native、matched
uniform、shared 的 push coverage 分别为 `0.0004577、0.0004775、0.0004775`，自由生成完全
相同，均为 `The following is the example.`。重复八次后 coverage 接近 `1.0`，并生成
`pushing push push ...`。这证明当前 checkpoint 中存在 push 的词汇通路，但单坐标 FOLD
增益不足以把它读出；重复输入虽然实现 lexical echo，同时产生严重重复，并非正常翻译。

`搬。` 同样没有 hard echo：native/shared/repeat coverage 为 `0.00237/0.00239/0.01865`。
因此重复也不是对所有词都足够。

shared 相对 uniform 的主要差异来自长句“西西弗……”：stone coverage 从 native 的
`0.07537`、uniform 的 `0.07634` 增至 `0.22189`，而 push 反而从 `0.001149` 降至
`0.000508`。控制器找到的是长句内偏向 stone/mountain 模式的结构方向，不是通用的动词直译
方向。测试 oracle 也没有增加 hard hit，说明仅靠当前 11 个 internal-parent 滤镜基底与
`0.2` 幅度，尚不能到达 isolated push/carry 的正确生成区域。

## 9. 下一边界

短输入的 internal TreeHeap 只有 1 至 2 个可干预 parent，无法承载丰富的拓扑滤镜。下一步若
继续 Echo，应将显微镜扩展到 leaf、embedding 或 READ 层，或者使用不改变目标词但提供多个
结构坐标的固定 carrier frame。否则继续优化同一 parent-gain 基底，只会重复测量单一能量旋钮。

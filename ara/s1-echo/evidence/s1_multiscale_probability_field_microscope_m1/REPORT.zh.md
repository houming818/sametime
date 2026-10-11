# M1 多尺度概率场显微镜报告

Owner: Nio / Trinity

Date: 2026-10-11

Tasks: io `652`（实现失败记录）、`653`（smoke）、`654`（formal）

## 问题

M1 同时观察两个对象：

1. T0 静态 token-type leaf 概率场是否存在可测的质量、熵、loss 和细节系数结构；
2. T1 动态 masked-token 后验能否从粗尺度后验递归加入局部分叉证据，得到细尺度后验。

它不训练新 FOLD/READ，也不把 parent 命名为语言概念。

## 完整性

正式实验使用 200,000 行 WMT 英文侧、8,192 个目标 token、4,096 个 context token、
50,000 个配对 masked-token 事件、三棵随机树及三次 learned-leaf permutation。

以下门全部通过：

- 各尺度概率质量守恒；
- leaf 熵等于各深度条件熵贡献之和；
- held-out leaf cross-entropy 等于各深度局部 cross-entropy 之和；
- 保存全部 15 个分叉参数可数值重建 16-leaf 场；
- T1 细后验等于粗后验加局部分叉对数证据后重新归一化；
- 最大后验递推闭合误差为 `1.144409e-05`；
- formal evidence 的全部 SHA-256 校验通过。

## T1 的概率金字塔

四上下文下，learned topology 每次提高分辨率带来的平均正确词对数概率增益为：

| 尺度 | mean gold log-probability gain | 正增益事件比例 |
|---|---:|---:|
| 1 -> 2 | 约 0.20 | 约 0.73 |
| 2 -> 4 | 约 0.11 | 约 0.67 |
| 4 -> 8 | 约 0.11 | 约 0.63 |
| 8 -> 16 | 约 0.10 | 约 0.62 |

总增益为 `0.522812`。其中到 `K=4` 已获得：

```math
F_4=0.607451.
```

匹配控制为：

| topology | 到 K=4 的增益占比 |
|---|---:|
| learned tree | **0.607451** |
| random tree mean | 0.263590 |
| learned-leaf permutation mean | 0.378829 |

因此，smoke 后预注册的 formal C03 通过。learned topology 不只是最终 16-leaf 场有效，
它还把较多有效判别信息安排到了较粗的前两次分裂中。

## 不是简单的系数衰减

learned tree 的 prior-weighted Haar 细节能量为：

```text
depth 0: 0.321586
depth 1: 0.158577
depth 2: 0.312618
depth 3: 0.299835
```

held-out cross-entropy 的逐层贡献为：

```text
0.518268 / 0.608847 / 0.522474 / 0.531875
```

两者都没有随深度单调下降。因此，现有证据不支持“TreeHeap 概率场等于一个按深度自然衰减的
简单幂级数”。更准确的结构是：粗层优先获得一部分有判别力的信息，但细层仍保存大量不可忽略
的修正。

## 两个失败门

### C01：少量细节系数重建

保留四个最大细节系数时：

| topology | prior-weighted JS reconstruction error |
|---|---:|
| learned | 0.028009 |
| random mean | **0.008186** |
| permutation mean | 0.040368 |

C01 失败。随机混合把概率场变得更平滑，所以更容易用少量系数近似；这不代表随机树保存了更多
预测信息。结论是：**易压缩不能单独作为结构质量指标。**

### C02：低阶 parent-to-child 多项式

二次回归在留出 token 上的 R-squared：

| topology | R-squared |
|---|---:|
| learned | 0.124078 |
| random mean | 0.038021 |
| permutation mean | **0.216238** |

C02 失败。learned tree 的局部递归性高于随机树，但 leaf permutation 更容易被这个低阶回归预测。
因此不能把一个共享的线性或二次 parent-to-child 方程宣布为 F。

## 当前结论

M1 支持：

- T0 概率单纯形可以精确改写为树上的局部条件坐标；
- T1 后验可以精确写成粗后验加逐尺度局部对数证据；
- learned topology 将显著更多的有效正确词增益前置到了 K<=4；
- 这个前置现象在新 formal 数据预算上优于随机树和 learned-leaf permutation。

M1 不支持：

- 细节系数随深度单调衰减；
- 少量最大系数重建误差可以单独评价 topology；
- 已经找到统一的低阶递归 F；
- parent 已经形成可命名句义；
- T1 已经等价于生成 Decoder。

更合适的下一问是：能否把 T1 的逐尺度局部对数证据作为 READ 的显式输入，让模型学习“何时停在
粗尺度，何时继续读取细节”，而不是继续寻找一个对所有节点通用的简单多项式。

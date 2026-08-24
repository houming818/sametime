# STONE-2 C05：连续深度调制 READ smoke

日期：2026-08-24

Claim：`S3-STONE2-CONTINUOUS-DEPTH-READ-C05`

状态：预注册，等待 smoke。

## 1. 为什么做

C04 说明解除 READ kernel 的完全共享可能有短程收益，但 coarse/middle/fine
三段式分组没有击败同参数量的交错分组。因此，不能继续枚举离散分组，也不能
直接启动长训练。

C05 检验一个更窄的结构假设：相邻 TreeHeap 深度之间应当连续变化，而不是突然
切换到另一套完整 kernel。该实验与 `IDEA-001` 隔离，不使用分层训练目标，也不
修改 loss。

## 2. 实验臂

三个实验臂从同一个 C03 PT checkpoint、同一批次流和同一初始化开始：

1. `shared`：原始共享 READ kernel；
2. `smooth_depth`：共享 kernel 加一个低秩残差，残差由归一化深度
   `z = 2d/(D-1)-1` 连续调制；
3. `shuffled_depth_control`：参数量和调用次数相同，但把 `z` 按固定交错顺序分配
   给深度，破坏相邻深度的连续关系。

低秩残差的输出层初始化为零，所以 step 0 三个臂必须严格等函数。Encoder、FOLD、
Butterfly、branch、depth embedding、recurrent cell、output 和 gain 全部冻结；
只训练共享 READ kernel，以及后两臂的低秩调制器。

## 3. 训练与评价

- seed：`16501`；
- 训练：120 updates，batch 16；
- 数据：checkpoint 记录的 WMT train/valid/test；
- 每 40 steps 记录 train/valid NLL；
- 完成后记录 test NLL、PPL、token BLEU-4、生成样例、梯度有限性与调制器范数；
- 只做 smoke，不保存产品 checkpoint。

## 4. 预注册门

- G0：step 0 logits 最大差 `<1e-6`，valid/test NLL 差 `<1e-7`；
- G1：所有梯度有限，两个低秩基都收到非零梯度；
- G2：`smooth_depth` final valid NLL 至少比 `shared` 低 `0.005`；
- G3：`smooth_depth` final valid NLL 至少比 `shuffled_depth_control` 低 `0.005`；
- G4：`smooth_depth` final test NLL 不得比 `shared` 高超过 `0.01`；
- G5：两个低秩输出矩阵的合计范数 `>1e-5`。

只有 G0..G5 全部通过，才允许多 seed 复测。G2 通过但 G3 失败，只能再次支持
“增加 READ 自由度”，不能支持“连续深度坐标”。G2 失败则停止该航线。

## 5. 执行结果

- 实现 smoke：io taskd `310`；
- 正式 smoke：io taskd `311`；
- 正式运行时长：`335.8s`；
- checkpoint SHA-256：
  `24d2b03c7a5f7441e3169ed40741544f62b1ce7db3e37ade7021558c226cb202`；
- stream SHA-256：
  `80f15f281a8f08795bd5e1f451bd75a714c78263b6fa7bd46f331b09979a16b8`。

step 0 三个臂严格等函数，训练中没有 NaN、Inf 或 OOM。120 updates 后：

| arm | trainable params | valid NLL | test NLL | token BLEU-4 |
|---|---:|---:|---:|---:|
| shared | 525,056 | 8.04866 | 8.09110 | 1.07279 |
| smooth_depth | 590,592 | 8.07938 | 8.09964 | 0.69441 |
| shuffled_depth_control | 590,592 | 8.07832 | 8.09404 | 0.71372 |

G0、G1、G4、G5 通过；G2、G3 失败。连续深度臂没有击败 shared，并且也没有
击败打乱深度坐标的同参数量对照。它在 step 40 一度达到 valid NLL `8.03718`，
随后在 step 80 反弹到 `8.16752`，step 120 回落到 `8.07938`。预注册要求使用
最终点，不能事后选择 step 40。

结论为 `continuous_depth_read_not_supported`。当前低秩线性深度调制不能作为
STONE-2 默认 READ，也不授权多 seed 或长训练。这个负结果只否定当前参数化和
训练条件，不否定深度连续性在其他受约束算子中可能存在。

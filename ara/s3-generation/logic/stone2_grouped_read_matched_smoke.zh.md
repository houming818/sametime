# STONE-2 C04：分组 READ kernel 匹配 smoke

日期：2026-08-24

Claim：`S3-STONE2-GROUPED-READ-C04`

状态：smoke 完成；解除完全共享为候选，但按分辨率分组未获支持。

## 1. 依据

C03-D02 在冻结 PT checkpoint 上用等函数解绑定证明：共享 READ kernel 的
coarse/middle 任务梯度存在冲突，而 branch 的组级梯度没有通过冲突门。这个结果
只授权修改 READ 的参数共享边界，不授权修改 FOLD、Butterfly、branch、Decoder、
数据或 loss。

## 2. 三个实验臂

三个臂都从同一个 C03 PT `checkpoint_best.pt` 开始，冻结 Encoder、FOLD、
Butterfly、branch、depth embedding、recurrent cell、output 和 gain，只训练 READ
kernel。

1. `shared`：沿用一个 READ kernel，所有深度共享；
2. `resolution_grouped`：复制为三个同初始化 kernel，depth 0..2 使用 coarse，
   3..5 使用 middle，6..末层使用 fine；
3. `interleaved_control`：同样三个 kernel、同参数量、同每深度一次调用，但按
   `depth mod 3` 分配。它打散分辨率连续性，用于区分“参数变多”与“按分辨率
   解绑定”。

所有副本从原 READ kernel 精确复制，因此 step 0 的三个臂必须具有完全相同的
logits、valid NLL 和 test NLL。

## 3. 数据与训练

- WMT train/valid/test 及 tokenizer 完全读取 checkpoint config；
- 固定 seed `16401`；
- 120 updates，batch 16；
- 三个臂使用同一份预生成 schedule 与 stream SHA-256；
- AdamW、学习率沿用 checkpoint config；
- 每 40 steps 记录 train/valid NLL；
- 不保存产品 checkpoint，不进入长训练。

## 4. 预注册门

- G0：step 0 三臂 logits 最大差 `<1e-6`，初始 valid/test NLL 差 `<1e-7`；
- G1：全部梯度有限；grouped 与 interleaved 的三个 bank 都至少一次收到非零梯度；
- G2：`resolution_grouped` final valid NLL 至少比 `shared` 低 `0.01`；
- G3：`resolution_grouped` final valid NLL 至少比 `interleaved_control` 低 `0.01`；
- G4：`resolution_grouped` final test NLL 不得比 `shared` 高超过 `0.01`；
- G5：grouped 三个 bank 的最大两两参数距离 `>1e-5`，证明发生了实际分化。

只有 G0..G5 全部通过，才把“按分辨率解绑定 READ”列为 C04 候选并允许多 seed
复测。若 G2 通过但 G3 失败，只能说明解除共享可能有益，不能说明 coarse/middle/
fine 分组正确。若 G2 失败，则 D02 的梯度冲突没有转化为短程任务收益，停止扩容。

## 5. 边界

该 smoke 只检验冻结架构上的 READ-only 适配。即使通过，也不能补签 C03 S7，
不能证明端到端长训练一定受益，不能直接启动 100M-piece Pretrain。

## 6. 执行结果

- 实现 smoke：io taskd `308`；
- 正式 smoke：io taskd `309`；
- checkpoint SHA-256：
  `24d2b03c7a5f7441e3169ed40741544f62b1ce7db3e37ade7021558c226cb202`；
- schedule SHA-256：
  `f64de353272c55d4ddfc0876c79cf749bd354fcb06f9c3e4a925999e623bdab3`。

step 0 三臂 logits、valid NLL 与 test NLL 完全相同。120 updates 后：

| arm | trainable params | valid NLL | test NLL | token BLEU-4 |
|---|---:|---:|---:|---:|
| shared | 525,056 | 8.04208 | 8.08483 | 1.08536 |
| resolution_grouped | 1,575,168 | 8.03026 | 8.07101 | 1.08534 |
| interleaved_control | 1,575,168 | 8.02922 | 8.07022 | 0.96937 |

G0、G1、G2、G4、G5 通过；G3 失败。按分辨率分组比 shared 的 valid/test NLL
分别改善 `0.01182/0.01382`，但 interleaved control 又比按分辨率分组低
`0.00104/0.00079`。因此不能把收益归因于 coarse/middle/fine 分组；更窄的结论
是，当前短程 READ-only 适配中，解除一个 kernel 的完全共享可能有益。

interleaved 的 NLL 略优但 token BLEU-4 更低，进一步说明不能只按 NLL 选择分组。
预注册决策为 `untying_candidate_but_resolution_grouping_not_supported`。本航线停在
这里，不进行分组枚举，不启动多 seed 或长训练。下一步需要先提出不依赖验证集
搜索的参数共享原理，例如连续 depth-conditioned adapter 或受约束的低秩调制。


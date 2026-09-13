# TreeHeap 高层信号可恢复性审计

日期：2026-09-13
Claim：`S3-TREEHEAP-ROOT-SIGNAL-RECOVERABILITY-F15`
状态：预注册，尚未观察结果。

## 1. 问题

F14 发现，重复 `推` 能单调改善完整 READ 的 `push` margin，却不能在 root 截止状态下稳定改善。
这至少有两种互斥解释：

1. **表示丢失**：递归 FOLD 没有把局部 `推` 信号传到 root；
2. **读出失败**：信号仍在 root 隐态中，但当前 READ/Decoder 的坐标、尺度或非线性无法访问它。

F14 只观察当前 Decoder 的目标 margin，不能区分“信息不存在”和“信息存在但读不出”。F15 冻结
同一 checkpoint，用独立线性探针测量每层隐态对 `推` 数量的可恢复性，不更新 TreeHeap 参数。

## 2. 固定合同

- checkpoint 固定为 E01 TreeHeap-106M pass2；
- 输入方向固定为中译英；`推`、中性控制 `看` 和目标 `push` 都必须是单 token；
- 八个内容位置枚举全部 `2^8 = 256` 个 `推/看` 组合，source 真长度统一为 10，pad width 为 16；
- 对 count 1--7，每个 count 内按组合序号确定性留出四分之一位置排列作为测试集；count 0 和 8 只进入训练边界；
- protocol depth 固定为 5、6、7；
- 比较原生 `(l+r)/sqrt(2)` 与只读反事实 `(l+r)/2`；
- checkpoint 前后逐 tensor 必须完全一致。

测试集的位置组合从未参与探针拟合，因此探针必须学习跨位置可复用的计数方向。另用确定性乱序训练标签
建立负对照，排除高维小样本下的偶然拟合。

## 3. 观察位置

每种退火模式和 protocol depth 都保存两套等维特征：

1. `fold`：`model.protocol` 直接产生的各层 TreeHeap 状态；
2. `read_facing`：经过 Decoder `convolve` 后，真正交给 READ 的各层状态。

同层有效节点做 masked mean；base 与 extra 通道拼接。对每一层分别用 ridge 线性探针预测归一化
计数 `y=count/8`，报告测试集 `R2`、MAE、相关系数和舍入后的精确计数率。

该探针回答“高层是否仍含有可线性恢复的 source 计数信号”，不把它偷换成跨语言语义已经形成。
`push` 的当前跨语言可读性仍由冻结 Decoder margin 单独记录。

## 4. 预注册门槛

- O0：256 个固定宽度样本完整，count 1--7 均同时存在训练与测试位置排列；
- O1：全部特征、预测和 Decoder margin 有限；
- O2：冻结 checkpoint 在实验前后逐 tensor 完全一致；
- C0：native `read_facing` root 的乱序标签对照 `R2 < 0.20`；
- P1：三个 depth 的 native `fold` root 测试 `R2` 最小值不低于 0.80；
- P2：三个 depth 的 native `read_facing` root 测试 `R2` 最小值不低于 0.80。

判读规则：

- P1 失败：支持 FOLD 高层表示丢失，优先修退火算子；
- P1 通过、P2 失败：FOLD 保留了信号，但 Decoder convolution 破坏了可恢复性；
- P1/P2 都通过，而 F14 root margin 仍失败：不能再称为“高层没有收到信号”，问题转向 READ/Decoder
  的目标方向、尺度或非线性；
- mean-FOLD 明显提高 root 可恢复性，只能作为退火尺度候选证据，不能直接替换训练中的原生算子。

## 5. 结论边界

这是合成、冻结、单 source token 的机制审计。即使 root 计数完全可恢复，也不证明 root 已形成中文 `推`
到英文 `push` 的语义协议；即使不可恢复，也只定位当前 checkpoint 和当前 operator。后续若需要验证语义，必须
在真实平行语料上用未见上下文训练 target-presence probe，再进行成对重训。

## 6. Smoke 结果

io taskd 409 首次启动时在模型加载前失败。原因是系统更新后，正在运行的 NVIDIA kernel module 为
`580.173.02`，用户态 NVML 为 `580.178`，CUDA 返回 804；失败现场保存在
`smoke_seed12601_failed_cuda804_task409`。重启 io 后，kernel module、NVML 与驱动统一为
`580.178.04`，270 W 功率限制保持不变。taskd 414 随后用同一预注册代码完成实验。
运行输入中的 logic SHA-256 已对预注册提交 `4e0c17b` 复算并完全匹配；当前文件追加结果后哈希变化，
不回写或伪造运行时的 `inputs.sha256`。

实验枚举 256 个位置组合，其中训练探针 192 个、测试 64 个。三个 protocol depth 的 root 结果为：

| 状态 | depth 5 R2 | depth 6 R2 | depth 7 R2 | 测试精确计数率 |
|---|---:|---:|---:|---:|
| native FOLD root | 0.999027 | 0.999821 | 0.999937 | 100% |
| native READ-facing root | 0.999159 | 0.999791 | 0.999841 | 100% |
| mean FOLD root | 0.999027 | 0.999821 | 0.999937 | 100% |
| mean READ-facing root | 0.999064 | 0.999790 | 0.999844 | 100% |

native READ-facing root 的乱序标签对照 R2 为 `-0.3412, -0.0593, -0.2418`，没有出现高维探针
偶然泛化。全部 FOLD 与 READ-facing 层在未见位置组合上也都达到近似 1.0 的 R2。O0--O2、C0、
P1、P2 全部通过，checkpoint 前后逐 tensor 未改变。

同一批样本经过当前 Decoder 后，count 8 相对 count 1 的 root target margin 在 depth 5、6 分别
改善 `+1.6105`、`+1.5055`，在 depth 7 却恶化 `-4.2142`。因此，depth 7 的失败不能解释为
source 计数信号没有到达 root。

## 7. 当前结论

F15 否定“递归 FOLD 已经把 `推` 的 source 计数从高层隐态中完全擦除”这一严格命题。当前
checkpoint 的 root 不仅含有该信号，而且一个位置外推的线性读出可以近乎精确地恢复它；Decoder
convolution 也没有破坏这种线性可恢复性。

这不否定退火算法存在问题。`(l+r)/sqrt(2)` 与 `(l+r)/2` 对线性探针几乎等价，却在 F14 的
非线性 Decoder margin 上表现不同，更符合“信息仍在，但能量尺度、目标方向或 READ 非线性进入了
错误工作区间”，而不是不可逆擦除。F14 所见的 near-leaf 恢复，也应改写为当前读出路径在细尺度上
更容易访问目标方向，而不是细层重新创造了 source 信息。

下一步需要 F16 真实语义可恢复性审计：从平行语料构造 source 含 `推` 且 target 含/不含 `push`
的真实上下文，按上下文模板隔离训练和测试，在每层预测 target-side `push` 是否成立。只有当 near-leaf
探针可泛化而 root 探针失败时，才支持“退火压缩丢失跨语言语义”；若 root 探针成功而原 Decoder
失败，应优先修 READ/Decoder 的坐标与尺度。

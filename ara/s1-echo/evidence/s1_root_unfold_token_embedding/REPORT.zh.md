# Root-Unfold Token Embedding 实验报告

## 问题

此前实验直接优化 `token -> leaf` 分配矩阵，并不符合目标架构。本实验改为：

```text
root = [1]
  -> [2 values]
  -> [4 values]
  -> [8/16 values]
  -> 选择一个 leaf
```

TreeHeap 内部节点共享参数并逐层拆分单位质量。最终 leaf 概率向量是
unfold 计算出来的 embedding，而不是一张可训练 token lookup 表。

## 输入边界

所有 token 的 root 质量都是 `1`。为了让确定性系统区分不同 token，
节点门同时读取固定的真实语料观测：

```text
sqrt(P_train(context | token))
```

这个观测不是可训练 embedding。实验中 `trainable_token_parameter_count=0`。
若连这一条件也移除，那么相同输入经过相同确定性函数必然得到相同输出。

## 结果

| 配置 | 容量选择初始/最终 NLL | 全局/打乱输入 NLL | LCP/随机 | 纯 argmax 占用 | 守恒误差 |
|---|---:|---:|---:|---:|---:|
| 50K 行，128 token，8 leaf | `4.38605 / 4.36042` | `4.46868 / 4.59844` | `1.70833 / 0.84115` | 全部，`10..24` | `1.19e-7` |
| 100K 行，256 token，16 leaf | `4.89755 / 4.87234` | `5.02004 / 5.29377` | `2.06120 / 0.94010` | 全部，`12..21` | `1.79e-7` |

两档等容量选择均为严格 `16..16`。纯局部 argmax 没有使用容量投影，
仍覆盖全部 leaf，因此分叉来自 root-unfold 参数，而不是容量算子凭空制造。

打乱 token 的上下文观测后，NLL 分别恶化 `0.23802` 和 `0.42143`，
硬落点一致率只有 `0.11719` 和 `0.04297`。这证明 leaf 向量依赖 token
的语料条件，而不是所有 token 共用一个固定分布。

## 可观察样例

128-token Smoke 中，`push` 从 root 的单位质量展开为：

```text
[0.0842546, 0.4977756, 0.0577371, 0.2044440,
 0.0326375, 0.0740195, 0.0434620, 0.0056696]
```

选择 leaf 1。256-token 规模中，`push` 的 16 维最大值位于 leaf 5：

```text
[0.0101005, 0.0026503, 0.1700694, 0.0422990,
 0.2294857, 0.2936626, 0.0326314, 0.0527469,
 0.0758334, 0.0310074, 0.0134267, 0.0294938,
 0.0019139, 0.0104558, 0.0034395, 0.0007838]
```

`want` 在同一档形成更尖锐的 leaf 15 概率 `0.8058221`。

## 结论边界

`S1-ROOT-UNFOLD-EMBED-C01` 在 128 和 256 token 上得到支持：共享
TreeHeap 节点能够把单位 root 质量递归展开成 token-dependent embedding，
梯度能够调整 unfold 参数，且不需要直接 token-leaf 参数。

尚未证明：

- 裸 token ID 或完全相同的无条件标量能自行分离；
- 同一 token 在不同句子中的多义动态路由；
- decoder 能利用该 embedding 完成句子生成或翻译；
- 更大词表和其他语料仍保持相同规律。

## Evidence

- implementation contract: task 496，非结论证据；
- registered Smoke: task 497；
- registered scale successor: task 498；
- host: `io.grepcode.cn`；
- GPU: RTX 3090，270 W 功率限制；
- source: `ara/s1-echo/src/s1_root_unfold_token_embedding.py`。

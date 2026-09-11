# F13-R1：多面方向生成收益益扩样复核

日期：2026-09-11
Claim：S3-TREEHEAP-LEARNED-POLYTOPE-ROUTER-F13-R1
状态：只读扩样已完成；工程合同与全量 BLEU 微增通过，分块多数和生成健康未通过。

F13 在 16 条生成样本上观察到 grouped 相对 scalarized 的 BLEU4 median 增加 3.863，但收益
主要位于 protocol depth 6。R1 固定 task 405 产生的两个 theta，不训练、不选 checkpoint、
不调整参数，只扩大测试集以判断这个生成信号是否稳定。

- 使用 64 个 held-out WMT pair，组成 128 条中英双向样本；
- 每个 theta 对每条样本、每个 protocol depth 只生成一次；
- 将 128 条按固定顺序切成 8 个互不重叠的 16 条数据块；
- 同时记录全量和逐块的 BLEU4、非空率、重复率，以及全量 valid/test NLL；
- 复核模型与 theta 的 SHA-256，评估前后不得改变。

支持稳健生成收益需同时满足：

- O0：模型与两个 theta 可重载，输出和指标全部有限；
- O1：冻结模型状态与三个输入文件的 SHA-256 前后不变；
- P1：grouped 的全量 BLEU4 median 高于 scalarized；
- P2：8 个固定块中，grouped BLEU4 median 高于 scalarized 的块数至少为 5；
- P3：grouped 非空率不比 scalarized 低 0.02 以上，重复率不比 scalarized 高 0.02 以上。

R1 不重新定义 F13 已经失败的 NLL 优势判据。即使通过，也只授权进入多 seed 配对训练，不
直接替换默认 FOLD。

## 扩样结果

io taskd 406 对 64 个 held-out pair、128 条双向样本完成只读评估，用时 198.73 秒。模型和
两个 theta 均成功重载，输入 SHA-256 与冻结模型状态前后不变，O0/O1 通过。

| 状态 | valid NLL | test NLL | BLEU4 median | nonempty min | repetition max |
|---|---:|---:|---:|---:|---:|
| native | 4.15374 | 3.96519 | 9.62601 | 0.96875 | 0.03856 |
| scalarized | 4.13233 | 3.95128 | 10.33731 | 0.96094 | 0.00494 |
| grouped | 4.13747 | 3.95229 | 10.83573 | 0.96875 | 0.03451 |

grouped 的全量 BLEU 中位数仍高于 scalarized 0.49842，P1 通过，但收益远小于原 16 条样本
的 3.86316。grouped 的 test NLL 只比 scalarized 高 0.00101，两个方向头都仍优于 native。

8 个固定块的 grouped 减 scalarized BLEU 中位数依次约为：

`+3.86, -0.43, -0.43, +0.00, -2.45, -3.15, -0.95, -0.43`

只有首块明确获益，另一个块仅有浮点量级的微弱正差，因此 grouped 只计为 2/8 胜，P2 未通过。
原 F13 的 16 条样本正好就是首块，确认当时的大幅增益是局部样本效应，不是普遍生成提升。

P3 也未通过。非空率没有恶化，但 grouped 的最坏重复率为 0.03451，scalarized 仅 0.00494，
差 0.02957，超过预注册上限；重复主要集中在 protocol depth 7。

## 当前结论

F12 的多方向局部几何仍成立，F13 也证明小头能够学习组间差异；但当前固定 8 等分通道的
grouped router 没有形成稳健的通用生成协议。它帮助了某一类样本，却在多数数据块和 depth 7
上带来干扰。下一步不应直接扩大训练，而应先对首块获益样本与后续受损样本做路由差异审计，
判断问题来自固定通道分组、depth 共享，还是缺少抑制重复的方向约束。

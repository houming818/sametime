# F13-R1：多面方向生成收益益扩样复核

日期：2026-09-11
Claim：S3-TREEHEAP-LEARNED-POLYTOPE-ROUTER-F13-R1
状态：预注册，待只读评估。

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

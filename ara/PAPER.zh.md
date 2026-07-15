# ARA 论文式总览：Semantic Prefix Routing 与 TreeHeap

状态：持续更新的研究工件  
创建时间：2026-06-22  
负责人：nio / Houming818  
Review Engineer：Codex  
仓库：SameTime  
ARA 参考：https://arxiv.org/abs/2604.24658v3

这个文件是 SameTime 公开研究记录的根 ARA manifest。

它不是传统论文。它是一份紧凑的研究状态图，目的是让人类或 AI reviewer 能够重建下面这条链：

```text
claim -> predict -> experiment -> evidence -> decision -> next step
主张   -> 预测    -> 实验       -> 证据     -> 判断     -> 下一步
```

面向人的叙事版本发布在 SPR blog 中。这个文件是给 reviewer 和后续 AI agent 使用的地图。

最新的细节以各层 `logic/claims.md`、`logic/experiments.md` 和 `evidence/` 为准；本文负责给出全局结构。

## ARA 目录结构

SameTime 使用四层 ARA 结构：

| 层 | 路径 | 用途 |
|---|---|---|
| Logic | `ara/*/logic/` | claims、predicts、problem、experiment design |
| Source | `ara/*/src/` | 可复现实验脚本和环境说明 |
| Trace | `ara/*/trace/` | 路线转向、被拒绝路径、决策 DAG |
| Evidence | `ara/*/evidence/` | summary、metrics、raw log 指针 |

主要课题：

| Topic | Path | Role |
|---|---|---|
| S1 Echo | `ara/s1-echo/` | 容量、路径 hash、上下文路由 |
| S2 Translation | `ara/s2-translation/` | Fold stack、graph builder、probability container |
| M0 TreeHeap Math | `ara/m0-treeheap-math/` | 代数、plus、kernel search、可训练性 |
| S3 Generation | `ara/s3-generation/` | 后续 generation / decoder work 的占位 |

## 状态词汇

| Status | 含义 |
|---|---|
| `verified` | 已有证据，并包含 baseline 或 falsification check |
| `supported` | 有正向 pilot evidence，但 baseline 或 scale 不完整 |
| `open` | 主张合理，但尚未测试 |
| `design` | 架构或数学设计，还没有证据支撑 |
| `rejected` | 已测试并失败 |
| `downgraded` | 早期较强主张被后续证据收窄 |

## 全局研究主张

当前全局 claim 必须刻意收窄：

```text
SPR / TreeHeap 不能靠断言替代 Transformer。
它是一族 addressable、path-aware、tree-structured operators。
这些算子可能拥有 MLP / CNN / Transformer 类似的机器学习能力，
同时在 address、substructure migration、prefix reuse、
probability container、delayed collapse 等问题上提供更强归纳偏置。
```

在提出 WMT 规模主张之前，研究必须先通过小任务证明：

```text
M0 math toolbox
-> S1 capacity and context routing
-> TreeHeap trainability
-> structural existence proofs
-> S2 translation / fold-stack tasks
```

## Claim Tree

### C0：ARA 流程主张

| ID | Source | Claim | Status | Evidence / Pointer | Falsification |
|---|---|---|---|---|---|
| C0-001 | SPR-001, SPR-008 | 每个强 claim 都必须有 evidence 和 falsification criteria。 | verified | `ara/README.md`, `ara/*/logic/claims.md`, `ara/*/evidence/README.md` | 一个 claim 在没有 evidence pointer 或 failure condition 时被升级。 |
| C0-002 | SPR-008 | 正确工作循环是 `predict -> claim -> experiment -> evidence -> trace`。 | supported | Blog SPR-008 和当前 ARA layout | 研究决策继续只依赖聊天记忆。 |
| C0-003 | SPR-019 | Blog 叙事不够；claims 必须保存在 registry artifacts 中。 | open | `PAPER.md` / `PAPER.zh.md` 是第一层 root manifest | 后续 SPR claims 与 registry 分离且不 reconciliation。 |

### C1：S1 容量与顺序主张

| ID | Source | Claim | Status | Evidence / Pointer | Falsification |
|---|---|---|---|---|---|
| C1-001 | SPR-001, SPR-002 | S1 token path hash 对 WMT14 word echo 有足够容量。 | verified | `ara/s1-echo/logic/claims.md` S1-C01, solo rate 99.7% | 同 slice / seed 下 solo rate 低于 95%。 |
| C1-002 | SPR-002 | 纯 cyclic shift 会发生 order collision；sign alternation 能打破这个对称。 | verified | `ara/s1-echo/logic/claims.md` S1-C02 | 非退化 A,B 与 B,A 在 sign alternation 后仍表示相同。 |
| C1-003 | SPR-002 | Echo reconstruction 可以在没有 learned Transformer attention 的情况下达到近似完美 BLEU。 | supported | `ara/s1-echo/logic/claims.md` S1-C03, BLEU-4 99.99 | 打乱 leaf labels 后 BLEU 仍保持，说明只证明 lookup。 |
| C1-004 | SPR-003 | Token-only routing 不编码上下文语义。 | rejected old claim / verified rejection | `ara/s1-echo/logic/claims.md` S1-C11 | 只有当 token-only route 在 polysemy 上超过 context 和 random baselines 才能重开。 |
| C1-005 | SPR-007 | Context-conditioned routing 能在 controlled proof 中编码语义区别。 | supported | `ara/s1-echo/logic/claims.md` S1-C10, S1-C13 | 真实 corpus 或 BoW/random baselines 追平或超过 context route。 |
| C1-006 | SPR-006, SPR-007 | S1b 的最小接口是 `route(token, context)`，不是 `route(token)`。 | supported | `ara/s1-echo/logic/claims.md` S1-C13 | 在 controlled 和 real tests 中 context 没有提升。 |
| C1-007 | SPR-004 | S1 output 可以作为 downstream S2 的输入契约。 | open | `ara/s1-echo/logic/claims.md` S1-C21 | 用 matched random vectors 替换 S1 vectors 后 S2 不变。 |

### C2：S2 Fold Stack 与 Graph Builder 主张

| ID | Source | Claim | Status | Evidence / Pointer | Falsification |
|---|---|---|---|---|---|
| C2-001 | SPR-005 | Semantic vectors 包含 fold-action 信息。 | verified | `ara/s2-translation/logic/claims.md` C-001 | clean split 下 fold action prediction 降到 chance。 |
| C2-002 | SPR-005 | 32D semantic space 足够完成当前 fold-action prediction task。 | verified | `ara/s2-translation/logic/claims.md` C-002 | 重新计算时 32D 明显弱于 128D。 |
| C2-003 | SPR-005 | EN->ZH 和 ZH->EN 两个方向都能做 cross-lingual fold structure prediction。 | verified | `ara/s2-translation/logic/claims.md` C-003, C-004 | cross-lingual AUC 对 frequency / BoW baselines 降到 chance。 |
| C2-004 | SPR-005 | Fold action types 可以被 small MLPs 预测。 | supported | `ara/s2-translation/logic/claims.md` C-005 | leakage control 后 baselines 追平。 |
| C2-005 | SPR-005, SPR-008 | PP/VP/NP pattern 显示不同 collapse regimes。 | verified | `ara/s2-translation/logic/claims.md` C-006..C-010 | recomputed grammar atlas 消除这些 collapse pattern。 |
| C2-006 | SPR-005 | Head、span、action detection 能用 supervised probes 完成。 | verified | `ara/s2-translation/logic/claims.md` C-011..C-013 | clean split 下 probe 失败。 |
| C2-007 | SPR-005 | Fold representation 可以用 oracle 或 template edges 重建 sentence。 | verified / supported | `ara/s2-translation/logic/claims.md` C-014, C-015 | oracle-free setting 下 reconstruction collapse。 |
| C2-008 | SPR-008 | 当前 graph assembly 瓶颈是 child/parent allocation，不是 fold action representation。 | verified | `ara/s2-translation/logic/claims.md` C-016..C-020 | oracle child/parent ablations 无法缩小 UAS gap。 |
| C2-009 | SPR-008 | Parent candidates 应作为 probability containers 保留，而不是 early argmax。 | verified for graph-builder stage | `ara/s2-translation/logic/claims.md` C-028 | top-k parent coverage 低，或后续模块无法使用 distributions。 |
| C2-010 | SPR-008 | 当前 3-epoch TreeHeap vectors 过度坍缩，不能支撑 syntax-energy claim。 | verified | `ara/s2-translation/logic/claims.md` C-027, strategy audit | 重新训练后 vectors 在 baselines 下显示可分离 syntax energy。 |
| C2-011 | SPR-008, SPR-009, SPR-010 | 历史 checkpoints 不能承载更强的 “TreeHeap syntax energy solved” claim。 | downgraded | `ara/s2-translation/evidence/strategy_audit/`, `tmerge_diagnostic/` | 新独立训练 checkpoint 通过 energy 和 syntax baselines。 |
| C2-012 | SPR-010 | 当前 world-model / background-field training evidence 只是诊断，不是正向 translation claim。 | downgraded | `ara/s2-translation/evidence/world_model_long_20260617_180554/` | 新 loss 在 controls 下显示下游 translation 或 structure gains。 |

### C3：M0 TreeHeap 代数主张

| ID | Source | Claim | Status | Evidence / Pointer | Falsification |
|---|---|---|---|---|---|
| C3-001 | SPR-011 | TreeHeap 必须先成为数学工具箱，再提出语言 claim。 | design | `ara/m0-treeheap-math/logic/problem.md`, `logic/solution/algebra.md` | 在没有 closure / inverse / projection tests 的情况下推进语言层工作。 |
| C3-002 | SPR-011, SPR-013 | Minimal TreeHeap algebra 应支持 closure、non-commutativity、inverse-like operations、projection、subheap matching、probability normalization。 | supported pilot | `ara/m0-treeheap-math/logic/predicts.md` P-MATH01; `evidence/treeheap_math_probe/` | synthetic exact mode 下 closure 或 order distinction 失败。 |
| C3-003 | SPR-012 | Subheap kernel search 是 TreeHeap 对 local structure 做 convolution 的类比。 | design / partly supported by toy | `ara/m0-treeheap-math/logic/experiments.md` existence suite B | kernel relocation 无法迁移到 seen positions 之外，或退化成 root matching。 |
| C3-004 | SPR-013 | Pure-math probe 可以建立非语言数学 pilot。 | supported | `evidence/treeheap_math_probe/README.md` | probe 无法复现 closure / inverse / subheap matching。 |
| C3-005 | SPR-014 | TreeHeap order 应由 primitive + plus 生成，而不是只由外部 index 指定。 | design | `ara/m0-treeheap-math/logic/predicts.md` P-MATH02 | 没有 plus candidate 能产生 successor、cycle 或 order margin。 |
| C3-006 | SPR-015 | Addressable TreeHeap 可以在 toy 中把 plus 用作 successor、information gain、mod-base fold、cyclic-kernel addressing。 | supported toy | `evidence/primitive_plus_probe/` | toy trace 无法复现 address successor 或 mod fold。 |
| C3-007 | SPR-016 | 在 TreeHeap encoder/decoder 之前，本地 learning stack 必须通过 linear、nonlinear、modular toy tasks。 | verified for local stack | `evidence/trainability_quiz/` | linear regression、XOR 或 modular addition deterministic rerun 失败。 |
| C3-008 | SPR-016 | 通过 trainability quiz 只能证明本地 ML stack capacity，不证明 TreeHeap language intelligence。 | verified boundary | `ara/m0-treeheap-math/logic/experiments.md` trainability interpretation | blog 或 registry 把 toy tasks 升级为 language evidence。 |

### C4：TreeHeap 存在性偏置主张

| ID | Source | Claim | Status | Evidence / Pointer | Falsification |
|---|---|---|---|---|---|
| C4-001 | SPR-017 | TreeHeap 的存在性主张不是 “它也能学”，而是 structural inductive-bias claim。 | design | Blog SPR-017 | TreeHeap 只匹配 generic MLP/Transformer，没有结构优势。 |
| C4-002 | SPR-017 | Addressable closure / mod-fold tasks 应测试 TreeHeap 是否在 length extrapolation 下保留 address 和 overwrite semantics。 | open | `ara/m0-treeheap-math/logic/predicts.md` P-EXIST01 | flatten baselines 在 extrapolated address tasks 上追平或超过 TreeHeap。 |
| C4-003 | SPR-017 | Subheap kernel relocation 应显示 local pattern 跨 unseen tree addresses 迁移。 | open | `ara/m0-treeheap-math/logic/predicts.md` P-EXIST02 | kernel 只在训练位置工作，或 false positive 很高。 |
| C4-004 | SPR-017 | Prefix compression 和 delayed collapse 应显示 shared prefixes reuse 与 calibrated candidates。 | open | `ara/m0-treeheap-math/logic/predicts.md` P-EXIST03 | prefix tree 没有 compression gain 或 probability container calibration。 |
| C4-005 | SPR-018 | 单纯 pattern matching 是错误的 B 实验；真正 TreeHeap proof 需要 learned encoder 加 conjugate query/decoder kernel。 | design | Blog SPR-018 | hand-designed pattern matching 仍是唯一可工作机制。 |
| C4-006 | SPR-018 | Learned ordered TreeHeap search 应把 key/value data 编码进 searchable tree，并用 query kernel 解码。 | open | Blog SPR-018 planned experiment | learned encoder 无法形成超过 trivial memorization 的 searchable structure。 |
| C4-007 | SPR-018 | Learned weighted prefix TreeHeap 应类似 Huffman-like compression，适合 skewed symbol distributions。 | open | Blog SPR-018 planned experiment | expected path length 无法击败 fixed-length baseline。 |

### C5：Soft TreeHeap / Gradient 主张

| ID | Source | Claim | Status | Evidence / Pointer | Falsification |
|---|---|---|---|---|---|
| C5-001 | SPR-019 | TreeHeap 需要 differentiable operator lifting，才能像 MLP / Transformer 一样训练。 | design | Blog SPR-019 | 只用 hard non-differentiable operators 就成功训练 useful TreeHeap。 |
| C5-002 | SPR-019 | Soft TreeHeap 应是 Hard TreeHeap operators 的 probability lifting：`SoftO(H)=sum_a p(a) O_a(H)`。 | mathematical design | Blog SPR-019 | one-hot soft operation 无法恢复 hard operation。 |
| C5-003 | SPR-019 | Naive soft memory write 不足以作为 TreeHeap claim，因为它更新 array slots，而不是 plus algebra。 | design / critique | Blog SPR-019 | slot interpolation 被形式化证明等价于 TreeHeap plus。 |
| C5-004 | SPR-019 | Soft Plus 应 lift TreeHeap plus：`H_next=sum_a p(a|H,x) (H plus_a x)`。 | supported pilot | `ara/m0-treeheap-math/evidence/soft_plus_probe/` | gradient 无法通过 soft plus，或 collapse 得到 illegal TreeHeap。 |
| C5-005 | SPR-019, SPR-020 | Kernel-guided Soft Plus 应使用 TreeHeap convolution kernel 决定 write/merge route。 | open / scoped pilot evidence | `ara/m0-treeheap-math/evidence/soft_plus_probe/`；GLM audit 显示当前 collapse 依赖 engineered alignment features | 在 clean-feature ablation 中 kernel-guided plus 弱于 naive memory write 或 encoder soft plus。 |
| C5-006 | SPR-019 | Multi-kernel / staged training 优于一个 “big pot” loss。 | open | Blog SPR-019 Experiment 3 loss ablation | matched budget 下 single total loss 更稳定且泛化更好。 |
| C5-007 | SPR-019 | Soft collapse 应恢复 legal Hard TreeHeap structure。 | supported pilot | `ara/m0-treeheap-math/evidence/soft_plus_probe/`: `collapse_accuracy_tau_0.05=1.0` | 更大或 noisy tests 中 collapse legality、route interpretability、hard-soft gap 失败。 |
| C5-008 | SPR-021 | C05 必须暴露 TreeHeap structure：path、subheap、recursive route/plus；否则退化成 flat soft memory。 | supported pilot | `ara/m0-treeheap-math/evidence/structural_c05_probe/`: flat/path-only test acc 0.0; subheap/path+subheap test acc 1.0 | unseen-depth relocation 中 flat address 或 path-only baselines 追平 subheap/path-subheap kernels。 |

### C7：S3 真实文本生成主张

| ID | 来源 | 主张 | 状态 | Evidence / Pointer | 证伪条件 |
|---|---|---|---|---|---|
| C7-001 | SPR-052 | 在相同的 `K=4` decoder memory 带宽下，learned TreeHeap frontier 应在 WMT 上优于 fixed/random tree frontier 和 flat 四向量压缩；替换 learned route 应导致翻译指标下降。 | 主主张未支持 / 保留弱 route 信号 | `ara/s3-generation/evidence/s3_wmt_frontier_smoke/`；learned/fixed/random/flat NLL 为 `6.5988/6.6012/6.7020/6.5080`；同 checkpoint 替换 fixed/random route 的 NLL 增量为 `+0.0097/+0.0285`，低于预注册 `0.05` 门槛 | 只有 value-preserving compose 重设计在预注册多 seed 下同时击败 flat `K=4`、fixed 和 random controls，才重新开放。 |
| C7-002 | SPR-054 | 冻结的 TreeHeap root compressor 加上每个内部地址一个有界 detail code，可以形成渐进恢复的多分辨率 codec，同时不改变原 root predictor。 | 机制支持 / 三 seed、每 seed 100万 block | `ara/s3-generation/evidence/s3_multiresolution_treeheap_pyramid/main_v2/`；MSE `1.9334 -> 0.5526`，k=64 token top-1 `0.9964`，整段 exact `0.8268`，地址错位 MSE `3.3147`，root NLL 增量 `0`，flat/Haar k=64 MSE `1.6042/1.2881` | 在 random-pair 训练、逐层消融、量化和更强同码率 baseline 完成前，结论必须保持有界。 |
| C7-003 | SPR-055 待写 | 只用 raw-token echo loss，可以联合训练共享递归的 WRITE/FOLD/DETAIL/UNFOLD/READ kernel，形成地址敏感的连续编码协议；但 echo 本身不会强迫模型形成有用的全局 root，也不能证明有限比特压缩。 | 部分支持 / 单 seed 机制结果 | `ara/s3-generation/evidence/s3_treeheap_native_codec/`；连续 v1/v2 top-1 `0.9955/0.9915`，detail 地址错位后 `0.0024/0.0014`，root 清零后仍为 `0.9956/0.9889`；632-bit v3 top-1 `0.2073` | root 消融无影响时必须拒绝 root-plus 解释。只有显式量化 rate-distortion 成立才能重新开放有限编码主张；root 的价值应改用缺失信息预测测试。 |

## SPR Blog Source Map

| SPR | File | Main role in ARA |
|---|---|---|
| 001 | `blogs/.../spr/001-problem.md` | 问题定义与 ARA 规则：每个 claim 需要 evidence / falsification |
| 002 | `blogs/.../spr/002-s1-evidence.md` | S1 echo capacity 与 order evidence |
| 003 | `blogs/.../spr/003-s1-falsification.md` | token-only semantic route 被拒绝 |
| 004 | `blogs/.../spr/004-architecture-decision.md` | SPR 层级拆分与 S1/S2 架构 framing |
| 005 | `blogs/.../spr/005-s2-fold-stack.md` | Fold stack 与 S2 方向 |
| 006 | `blogs/.../spr/006-next-experiments.md` | Baseline battle 与 claim decision workflow |
| 007 | `blogs/.../spr/007-context-proof.md` | controlled context routing proof |
| 008 | `blogs/.../spr/008-s2-strategy-audit.md` | strategy audit、probability container、历史 syntax claims 降级 |
| 009 | `blogs/.../spr/009-world-model-frames.md` | 术语：world model / background field |
| 010 | `blogs/.../spr/010-world-model-night-run.md` | night-run diagnostic；不是 positive WMT claim |
| 011 | `blogs/.../spr/011-treeheap-algebra.md` | 语言 claim 前的 TreeHeap algebra design |
| 012 | `blogs/.../spr/012-subheap-kernel-search.md` | subheap kernel / convolution-like reasoning proposal |
| 013 | `blogs/.../spr/013-treeheap-math-probe.md` | M0 pure math probe |
| 014 | `blogs/.../spr/014-primitive-plus-order.md` | primitive + plus as order source |
| 015 | `blogs/.../spr/015-primitive-plus-probe.md` | addressable plus / mod fold toy proof |
| 016 | `blogs/.../spr/016-trainability-quiz.md` | ML entrance quiz：linear、XOR、modular addition |
| 017 | `blogs/.../spr/017-treeheap-existence-proofs.md` | existence proof claims A/B/C |
| 018 | `blogs/.../spr/018-treeheap-structural-inductive-bias.md` | learned encoding 与 conjugate kernel refinement |
| 019 | `blogs/.../spr/019-soft-treeheap-gradient.md` | soft algebra、kernel-guided soft plus、multi-kernel training |
| 020 | `blogs/.../spr/020-soft-treeheap-audit.md` | GLM audit、ARA scope repair、clean-kernel next proof |
| 021 | `blogs/.../spr/021-c05-structural-proof.md` | C05 structural proof：path、subheap、recursive plus |
| 022-033 | `blogs/.../spr/022-*.md` 到 `033-*.md` | 后续数学底座、kernel、S1、decoder 进展；以各层 registry 为准 |
| 052 | `blogs/.../spr/052-treeheap-frontier-bottleneck.md` | 固定带宽 WMT frontier 结果；主要优势 Claim 未支持 |
| 053 | `blogs/.../spr/053-treeheap-algebraic-operator-codec.md` | 学习短算子程序并交给固定 TreeHeap executor；未见地址阳性、未见深度失败 |
| 054 | `blogs/.../spr/054-treeheap-multiresolution-pyramid.md` | 冻结 root、带地址 detail codec 与三 seed 率失真 Evidence |

Canonical local blog source：

```text
../../blogs/www.grepcode.cn/src/spr/
../../blogs/www.lostmap.cn/src/spr/
```

Public blog URLs：

```text
https://www.grepcode.cn/spr/
https://www.lostmap.cn/spr/
```

## Evidence Registry Pointers

| Evidence | Supports |
|---|---|
| `ara/s1-echo/evidence/README.md` | S1 capacity、token-only falsification、controlled context routing |
| `ara/s2-translation/evidence/strategy_audit/` | S2 graph-builder bottleneck、vector collapse、probability containers |
| `ara/s2-translation/evidence/tmerge_diagnostic/` | t_merge / background-field diagnostics |
| `ara/s2-translation/evidence/world_model_long_20260617_180554/` | world-model night-run diagnostic，不是 positive WMT claim |
| `ara/m0-treeheap-math/evidence/treeheap_math_probe/` | minimal algebra probe |
| `ara/m0-treeheap-math/evidence/primitive_plus_probe/` | addressable plus 和 mod-fold toy |
| `ara/m0-treeheap-math/evidence/trainability_quiz/` | linear regression、XOR、modular addition learning checks |
| `ara/m0-treeheap-math/evidence/soft_plus_probe/` | Soft Plus gradient path、toy collapse、GLM feature-ablation audit |
| `ara/m0-treeheap-math/evidence/structural_c05_probe/` | C05 structural ablation：flat/path-only vs subheap/path-subheap relocation |
| `ara/m0-treeheap-math/evidence/treeheap_diff_algebra_probe/` | TreeHeap diff/distance/finite-difference learning signal |
| `ara/m0-treeheap-math/evidence/algebraic_decoder_probe/` | finite-field TreeHeap algebraic decoders |
| `ara/s1-echo/evidence/s1_wmt_echo_kernel_probe/` | WMT short BPE echo write/read kernel |
| `ara/s1-echo/evidence/s1_probabilistic_read_kernel_probe/` | probabilistic stop/left/right read collapse |

## 当前 Open Proof Queue

最高优先级开放实验：

1. `Write Mechanism Ablation`
   - Claims：C5-003..C5-005
   - 比较：

```text
A: naive soft memory write
B: encoder soft plus
C: kernel-guided soft plus
```

2. `Hard/Soft Consistency`
   - Claims：C5-002, C5-007
   - 测量 hard-soft output gap、collapse legality、route interpretability。

3. `Existence Proof Suite`
   - Claims：C4-002..C4-004
   - 测量 address extrapolation、subheap relocation、prefix compression。

4. `Real Context S1b`
   - Claims：C1-005, C1-006
   - controlled proof 必须面对 BoW、keyword、random-hash、real-corpus baselines。

5. `Internal Decoder Bridge`
   - Claims：M0-DEC-C01 与 S1-READ-C01 后续
   - 目标：把 algebraic decoder 接到 learned semantic decoder，验证 internal node state 的可读性。

## 当前降级或拒绝的主张

| Claim | Decision |
|---|---|
| Token-only routing encodes contextual semantics | rejected |
| Current 3-epoch TreeHeap vectors solve syntax energy | downgraded |
| Historical checkpoint proves TreeHeap syntax | downgraded |
| Naive soft memory write proves TreeHeap algebra is trainable | rejected as too weak |
| WMT / Transformer replacement claim | not allowed yet |

## Reviewer Notes

当前研究不允许声称：

```text
TreeHeap beats Transformer on WMT.
TreeHeap has learned syntax from the current historical checkpoint.
Soft TreeHeap training is already proved.
Kernel-guided soft plus has learned clean routing from raw TreeHeap geometry.
```

当前研究允许声称：

```text
S1 has strong path-hash capacity.
Token-only semantic routing failed.
Context-conditioned routing works in a controlled proof.
S2 fold/action signals are probe-predictable.
Current graph builder needs probability containers and better allocation.
TreeHeap math/toy probes support a narrow algebraic toolbox direction.
Soft Plus has a working gradient-path toy proof.
Kernel-guided clean route learning remains open.
Subheap structure carries unseen-depth relocation in the C05 structural toy.
Finite-field/path-addressed TreeHeap has algebraic decoders in M0 toy.
```

## 维护规则

当新的 SPR blog 改变 claim 时，必须在同一个 commit 中更新本文，或在长实验前补 follow-up commit。

每个新 claim 应该有：

```text
ID
source
claim sentence
status
evidence pointer
falsification condition
next experiment
```

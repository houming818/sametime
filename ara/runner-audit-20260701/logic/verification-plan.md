# Runner Verification Plan

Status: executing
Created: 2026-07-01
Owner: DeepSeek / Runner
Target: GPT Reviewer cross-review after completion

## Purpose

独立验证所有 ARA 声明的实验可复现性。不依赖博客叙事，只看 probe → evidence → 数字是否吻合。

## Verification Methodology

对于每个探针：

```
1. 读取 evidence/summary.json —— 记录 original 数字
2. 在 ni（16 个 NumPy 探针）或 io（5 个 torch 探针）重新运行
3. 比较重新运行后的输出与 original —— 判断是否吻合
4. 如果吻合：记录 ✅ + 多 seed 运行验证稳定性
5. 如果不吻合：记录 ❌ + 诊断差异来源
6. 提交到本目录下的 evidence/
```

对每个声明运行多 seed（至少 3 个不同的 seed）以确保结果并非单次运气。

## 按可靠性分级的验证范围

### 第一阶段：ni 上 NumPy 探针（16 个，优先级排序）

#### P0 — 已部分验证的

| # | 探针 | 声明 | 已做 | 还要做 |
|---|------|------|------|--------|
| 1 | deductive_inductive_kernel_probe | M0-SOFT-C09 | ✅ seed=13 | 多 seed: 42, 7, 123 |
| 2 | shallow_treeheap_s1_probe | S1-C30 | ✅ seed=17 | 多 seed: 42, 7, 123 |
| 3 | treeheap_diff_algebra_probe | M0-DIFF-C01 | ✅ seed=27 | 多 seed: 42, 7, 123 |

#### P1 — 高优先级未验证

| # | 探针 | 声明 | S1-echo |
|---|------|------|--------|
| 4 | structural_c05_probe | C5-008 (M0-SOFT-C07) | M0 |
| 5 | soft_plus_probe | M0-SOFT-C03/04 | M0 |
| 6 | kernel_convolution_ops_probe | M0-SOFT-C08 | M0 |
| 7 | kernel_parameter_learning_probe | S1-KERNEL-LEARN-C01 | S1 |
| 8 | mirror_kernel_symmetry_probe | S1-KERNEL-MIRROR-C01 | S1 |
| 9 | echo_encoder_decoder_probe | S1-ECHO-ED-C01 | S1 |
| 10 | algebraic_readout_probe | S1-READ-C02 | S1 |
| 11 | ordered_fold_kernel_probe | S1-FOLD-C01 | S1 |
| 12 | controllable_manifold_probe | S1-MANIFOLD-C01 | S1 |
| 13 | heap_state_relaxation_probe | S1-RELAX-C01 | S1 |
| 14 | probabilistic_read_kernel_probe | S1-READ-C01 | S1 |

#### P2 — 已覆盖的基础

| # | 探针 | 声明 |
|---|------|------|
| 15 | algebraic_decoder_probe | M0-DEC-C01 |
| 16 | primitive_plus_probe | M0-C03 |
| 17 | treeheap_math_probe | M0-C01 |

### 第二阶段：io 上 PyTorch 探针（5 个）

| # | 探针 | 声明 | 备注 |
|---|------|------|------|
| 18 | wmt_echo_kernel_probe | S1-WMT-ECHO-C01 | 需要 WMT17 data at /mnt/nas |
| 19 | wmt_multikernel_specialization_probe | S1-MK-C01 | 同上，2049+513 词表 |
| 20 | corpus_embedding_kernel_probe | S1-WM-C02 | 需要本地 SGNS 语料 |
| 21 | world_model_compound_probe | S1-WM-C01 | 需要 HuggingFace（io 不可达）— 只 audit evidence JSON |
| 22 | frame_probe (S2) | C2-001..C2-012 | S2 诊断探针 |

## 验证通过的评判标准

每项验证需满足**三者皆对**：

1. **pilot_pass 匹配**：重新运行后的 pilot_pass 与 evidence 中的值一致
2. **核心指标数字在容差内**：关键 metric（exact、KL、top1 等）在合理舍入误差内
3. **多 seed 稳定**：≥ 3 个不同的 seed 产生一致结果（定性判断，非定量）

## 本次验证的发现写入目标

```
ara/runner-audit-20260701/
├── logic/
│   ├── verification-plan.md          ← this file
│   └── verification-checklist.md     ← per-claim pass/fail table
├── evidence/
│   ├── {probe_name}_seed{nn}/         ← re-run outputs
│   │   └── summary.json
│   └── aggregate.json                 ← all results in one file
└── trace/
    └── decision_dag.yaml              ← decisions made during verification
```

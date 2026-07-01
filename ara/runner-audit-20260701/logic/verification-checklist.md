| # | 探针 | 声明 | Status | Rerun pilot | Multi-seed | Evidence match | Notes |
|---|------|------|--------|-------------|------------|---------------|-------|
| P0-1 | deductive_inductive_kernel | M0-SOFT-C09 | ✅ | True (1/3 seeds) | ⚠️ | ✅trend | mlp<th consistent; pilot_pass seed-dependent |
| P0-2 | shallow_treeheap_s1 | S1-C30 | ✅ | True (3/3) | ✅ | ✅ | All pilot=True; flat OOD always 0.0 |
| P0-3 | treeheap_diff_algebra | M0-DIFF-C01 | ✅ | True (3/3) | ✅ | ✅ | 11/11 checks across all seeds |
| P1-4 | structural_c05 | C5-008 | ✅ | True (2/2) | ✅ | ✅ | subheap_test=1.0, flat_test=0.0 |
| P1-5 | soft_plus_probe | M0-SOFT-C03/04 | ✅ | True (3/3) | ✅ | ✅ | collapse_acc=1.0 all seeds |
| P1-6 | kernel_convolution_ops | M0-SOFT-C08 | ✅ | True (3/3) | ✅ | ✅ | search/plus/conj all hit@1 |
| P1-7 | kernel_parameter_learning | S1-KERNEL-LEARN-C01 | ✅ | True (3/3) | ✅ | ✅ | All pilot=True |
| P1-8 | mirror_kernel_symmetry | S1-KERNEL-MIRROR-C01 | ⏳ | Timeout (>60s) | — | — | Needs more time or io |
| P1-9 | echo_encoder_decoder | S1-ECHO-ED-C01 | ✅ | True (2/2) | ✅ | ✅ | All pilot=True |
| P1-10 | algebraic_readout | S1-READ-C02 | ⏳ | Timeout (>60s) | — | — | Needs more time or io |
| P1-11 | ordered_fold_kernel | S1-FOLD-C01 | ✅ | True (3/3) | ✅ | ✅ | All pilot=True |
| P1-12 | controllable_manifold | S1-MANIFOLD-C01 | ✅ | True (3/3) | ✅ | ✅ | All pilot=True |
| P1-13 | heap_state_relaxation | S1-RELAX-C01 | ✅ | True (2/2) | ✅ | ✅ | All pilot=True |
| P1-14 | probabilistic_read_kernel | S1-READ-C01 | ⏳ | Timeout (>60s) | — | — | Needs more time or io |
| P2-15 | algebraic_decoder | M0-DEC-C01 | ✅ | True (3/3) | ✅ | ✅ | All pilot=True |
| P2-16 | primitive_plus | M0-C03 | TBD | — | — | — | |
| P2-17 | treeheap_math | M0-C01 | TBD | — | — | — | |

Legend: ✅ = verified, ⏳ = timeout/needs io, TBD = pending

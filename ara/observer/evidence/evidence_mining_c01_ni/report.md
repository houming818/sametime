# TreeHeap Observer C01 Report

Generated: `2026-08-31T14:29:34.119074+00:00`

## Inventory

- Source files: 969
- Source bytes: 82065971
- Parsed records: 123521
- Parse errors: 0
- Metrics: 70653
- Generations: 5461
- Findings: 685

## Metric Catalogue

| Metric | Records |
|---|---:|
| `teacher_nll` | 18288 |
| `single_residual_train_nll` | 6009 |
| `noresidual_forest_train_nll` | 6009 |
| `nll` | 3687 |
| `valid_nll` | 3057 |
| `loss` | 2690 |
| `ppl` | 2628 |
| `train_nll` | 2249 |
| `train_nll_window` | 2195 |
| `grad_norm` | 1714 |
| `slot_variance` | 1282 |
| `residual_forest_train_nll` | 1277 |
| `route_pair_overlap` | 1169 |
| `route_pair_cosine` | 1169 |
| `argmax_coverage` | 1169 |
| `owner_leaf_coverage` | 974 |
| `entropy` | 891 |
| `unigram_js` | 876 |
| `model_js` | 876 |
| `token_bleu4` | 759 |
| `exact` | 591 |
| `between_slot_variance` | 546 |
| `exact_match` | 408 |
| `nonempty_rate` | 365 |
| `exact_rate` | 334 |
| `damage_nll` | 324 |
| `gate_entropy` | 320 |
| `entropy_bits` | 294 |
| `exact_probability_mass` | 252 |
| `nonempty` | 222 |

## Findings

| Severity | Code | Source | Detail |
|---|---|---|---|
| error | `NONFINITE` | `s1-echo/evidence/s1_echo_entry_gate_probe/summary.json:$.models.no_task_single_kernel.metrics.train.gate_echo_identity_prob` | nan |
| error | `NONFINITE` | `s1-echo/evidence/s1_echo_entry_gate_probe/summary.json:$.models.no_task_single_kernel.metrics.train.gate_mirror_mirror_prob` | nan |
| error | `NONFINITE` | `s1-echo/evidence/s1_echo_entry_gate_probe/summary.json:$.models.no_task_single_kernel.metrics.test.gate_echo_identity_prob` | nan |
| error | `NONFINITE` | `s1-echo/evidence/s1_echo_entry_gate_probe/summary.json:$.models.no_task_single_kernel.metrics.test.gate_mirror_mirror_prob` | nan |
| error | `NONFINITE` | `s1-echo/evidence/s1_echo_entry_gate_probe/summary.json:$.models.no_task_single_kernel.metrics.ood.gate_echo_identity_prob` | nan |
| error | `NONFINITE` | `s1-echo/evidence/s1_echo_entry_gate_probe/summary.json:$.models.no_task_single_kernel.metrics.ood.gate_mirror_mirror_prob` | nan |
| error | `NONFINITE` | `s1-echo/evidence/s1_echo_inverse_gate_probe/summary.json:$.models.no_inverse_single_route.metrics.train.mirror_route_confidence` | nan |
| error | `NONFINITE` | `s1-echo/evidence/s1_echo_inverse_gate_probe/summary.json:$.models.no_inverse_single_route.metrics.train.identity_gate_identity_inverse_prob` | nan |
| error | `NONFINITE` | `s1-echo/evidence/s1_echo_inverse_gate_probe/summary.json:$.models.no_inverse_single_route.metrics.train.mirror_gate_mirror_inverse_prob` | nan |
| error | `NONFINITE` | `s1-echo/evidence/s1_echo_inverse_gate_probe/summary.json:$.models.no_inverse_single_route.metrics.test.mirror_route_confidence` | nan |
| error | `NONFINITE` | `s1-echo/evidence/s1_echo_inverse_gate_probe/summary.json:$.models.no_inverse_single_route.metrics.test.identity_gate_identity_inverse_prob` | nan |
| error | `NONFINITE` | `s1-echo/evidence/s1_echo_inverse_gate_probe/summary.json:$.models.no_inverse_single_route.metrics.test.mirror_gate_mirror_inverse_prob` | nan |
| error | `NONFINITE` | `s1-echo/evidence/s1_echo_inverse_gate_probe/summary.json:$.models.no_inverse_single_route.metrics.ood.mirror_route_confidence` | nan |
| error | `NONFINITE` | `s1-echo/evidence/s1_echo_inverse_gate_probe/summary.json:$.models.no_inverse_single_route.metrics.ood.identity_gate_identity_inverse_prob` | nan |
| error | `NONFINITE` | `s1-echo/evidence/s1_echo_inverse_gate_probe/summary.json:$.models.no_inverse_single_route.metrics.ood.mirror_gate_mirror_inverse_prob` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.valid.nll` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.valid.root_variance` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.valid.root_mean_pair_cosine` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.valid.gate_mean[0]` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.valid.gate_mean[1]` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.valid.gate_mean[2]` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.valid.gate_mean[3]` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.valid.gate_entropy` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.valid.head_pair_cosine` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.destroy_addresses.nll` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.destroy_addresses.root_variance` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.destroy_addresses.root_mean_pair_cosine` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.destroy_addresses.gate_mean[0]` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.destroy_addresses.gate_mean[1]` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.destroy_addresses.gate_mean[2]` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.destroy_addresses.gate_mean[3]` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.destroy_addresses.gate_entropy` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.destroy_addresses.head_pair_cosine` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.address_destroy_nll_increase` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[0].nll` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[0].root_variance` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[0].root_mean_pair_cosine` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[0].gate_mean[0]` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[0].gate_mean[1]` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[0].gate_mean[2]` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[0].gate_mean[3]` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[0].gate_entropy` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[0].head_pair_cosine` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[1].nll` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[1].root_variance` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[1].root_mean_pair_cosine` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[1].gate_mean[0]` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[1].gate_mean[1]` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[1].gate_mean[2]` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[1].gate_mean[3]` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[1].gate_entropy` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[1].head_pair_cosine` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[2].nll` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[2].root_variance` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[2].root_mean_pair_cosine` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[2].gate_mean[0]` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[2].gate_mean[1]` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[2].gate_mean[2]` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[2].gate_mean[3]` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[2].gate_entropy` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[2].head_pair_cosine` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[3].nll` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[3].root_variance` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[3].root_mean_pair_cosine` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[3].gate_mean[0]` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[3].gate_mean[1]` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[3].gate_mean[2]` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[3].gate_mean[3]` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[3].gate_entropy` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablations[3].head_pair_cosine` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablation_nll_increase[0]` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablation_nll_increase[1]` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablation_nll_increase[2]` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.head_ablation_nll_increase[3]` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.gradient_audit.loss` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.gradient_audit.embedding_grad_norm` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.gradient_audit.kernel_down_grad_norm` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.gradient_audit.kernel_up_grad_norm` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.models.residual_forest.gradient_audit.decoder_grad_norm` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.derived.nores_minus_residual_valid_nll` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.derived.residual_address_destroy_nll_increase` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/summary.json:$.derived.residual_max_head_ablation_nll_increase` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_residual_treeheap_forest/full/trace.jsonl:$.residual_forest_train_nll` | nan |
| error | `NONFINITE` | `s3-generation/evidence/s3_treeheap_multiscale_mask/smoke/trace_echo.jsonl:$.grad_norm` | inf |
| error | `NONFINITE` | `s3-generation/evidence/s3_treeheap_multiscale_mask/smoke/trace_multiscale_mask.jsonl:$.grad_norm` | inf |
| error | `NONFINITE` | `s3-generation/evidence/s3_treeheap_native_codec/smoke_v1/trace_seed_71411.jsonl:$.grad_norm` | inf |
| error | `NONFINITE` | `s3-generation/evidence/s3_treeheap_native_codec/smoke_v2/trace_seed_71411.jsonl:$.grad_norm` | inf |
| error | `NONFINITE` | `s3-generation/evidence/s3_treeheap_native_codec/smoke_v3/trace_seed_71411.jsonl:$.grad_norm` | inf |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_partial_state_protocol_transfer/smoke/PB/stage_b/examples.json:$` | same generation reused for 7 distinct sources: ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_partial_state_protocol_transfer/smoke/PB/stage_b/examples.json:$` | same generation reused for 3 distinct sources: ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_partial_state_protocol_transfer/smoke/PB/stage_b/summary.json:$` | same generation reused for 7 distinct sources: ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_partial_state_protocol_transfer/smoke/PB/stage_b/summary.json:$` | same generation reused for 3 distinct sources: ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_partial_state_protocol_transfer/smoke/PF/stage_b/examples.json:$` | same generation reused for 3 distinct sources: ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_partial_state_protocol_transfer/smoke/PF/stage_b/examples.json:$` | same generation reused for 4 distinct sources: The the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the  |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_partial_state_protocol_transfer/smoke/PF/stage_b/examples.json:$` | same generation reused for 5 distinct sources: The,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_partial_state_protocol_transfer/smoke/PF/stage_b/summary.json:$` | same generation reused for 3 distinct sources: ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_partial_state_protocol_transfer/smoke/PF/stage_b/summary.json:$` | same generation reused for 4 distinct sources: The the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the  |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_partial_state_protocol_transfer/smoke/PF/stage_b/summary.json:$` | same generation reused for 5 distinct sources: The,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_partial_state_protocol_transfer/smoke/SP/stage_b/examples.json:$` | same generation reused for 7 distinct sources: ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_partial_state_protocol_transfer/smoke/SP/stage_b/examples.json:$` | same generation reused for 3 distinct sources: ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_partial_state_protocol_transfer/smoke/SP/stage_b/examples.json:$` | same generation reused for 2 distinct sources: The,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_partial_state_protocol_transfer/smoke/SP/stage_b/examples.json:$` | same generation reused for 2 distinct sources: the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the  |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_partial_state_protocol_transfer/smoke/SP/stage_b/summary.json:$` | same generation reused for 7 distinct sources: ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_partial_state_protocol_transfer/smoke/SP/stage_b/summary.json:$` | same generation reused for 3 distinct sources: ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_partial_state_protocol_transfer/smoke/SP/stage_b/summary.json:$` | same generation reused for 2 distinct sources: The,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_partial_state_protocol_transfer/smoke/SP/stage_b/summary.json:$` | same generation reused for 2 distinct sources: the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the  |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_partial_state_protocol_transfer/smoke/summary.json:$` | same generation reused for 10 distinct sources: ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_partial_state_protocol_transfer/smoke/summary.json:$` | same generation reused for 3 distinct sources: ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_partial_state_protocol_transfer/smoke/summary.json:$` | same generation reused for 5 distinct sources: The the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the  |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_partial_state_protocol_transfer/smoke/summary.json:$` | same generation reused for 5 distinct sources: The,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_partial_state_protocol_transfer/smoke/summary.json:$` | same generation reused for 2 distinct sources: the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the  |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_pretrain_task_posterior_pipeline/pilot_seed10101/proof/generations.json:$` | same generation reused for 2 distinct sources: ,在北京市中共建国,共建国部,共建国部,共建国部, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_pretrain_task_posterior_pipeline/pilot_seed10101/proof/generations.json:$` | same generation reused for 2 distinct sources: ;zh-hk:意大利;zh-hk:意大利;zh-hk:意大利;zh- |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_pretrain_task_posterior_pipeline/pilot_seed10101/proof/generations.json:$` | same generation reused for 2 distinct sources: 年,他曾任国立台湾大学政治学系教授,曾任中华民国国民革命军官, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_pretrain_task_posterior_pipeline/pilot_seed10101/proof/generations.json:$` | same generation reused for 2 distinct sources: 日,在北京市中共中央、上海、上海、上海、上海、上海、上海、上海、上海 |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_pretrain_task_posterior_pipeline/pilot_seed10101/proof/generations.json:$` | same generation reused for 2 distinct sources: 日,在北京市中共中央省份,共计6个,共计2个,共计2 |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_pretrain_task_posterior_pipeline/pilot_seed10101/proof/generations.json:$` | same generation reused for 2 distinct sources: 日,在北京市中山市政府部门,在北京市中共建国,共建国部, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_pretrain_task_posterior_pipeline/smoke/proof/generations.json:$` | same generation reused for 16 distinct sources: ,,,,,,,,,,,,,,,,,,,,,,,, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_pretrain_task_posterior_pipeline/smoke/report.json:$` | same generation reused for 2 distinct sources: ,,,,,,,,,,,,,,,,,,,,,,,,,,,,, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_pretrain_task_posterior_pipeline/smoke/task/PT/summary.json:$` | same generation reused for 2 distinct sources: ,,,,,,,,,,,,,,,,,,,,,,,,,,,,, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_pretrain_task_posterior_pipeline/smoke_v2/proof/generations.json:$` | same generation reused for 2 distinct sources: ,,,,,,,,,,,,,,,,,,,,,,,, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_pretrain_task_posterior_pipeline/smoke_v2/proof/generations.json:$` | same generation reused for 2 distinct sources: 、、、、、、、、、、、、、、、、、、、、、、、、 |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_recursive_depth_pressure_protocol_d07/smoke_seed10701/summary.json:$` | same generation reused for 2 distinct sources:  ⁇ 湾,  ⁇ 湾,  ⁇ 湾,  ⁇ 湾,  ⁇ 湾,  ⁇ 湾,  ⁇ 湾湾,  ⁇ 湾湾湾,  ⁇ 湾湾湾,  ⁇ 湾湾湾,  ⁇ 湾湾湾,  ⁇ 湾湾湾湾湾,  ⁇ 州市 |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_recursive_depth_pressure_protocol_d07/smoke_seed10701/summary.json:$` | same generation reused for 2 distinct sources:  ⁇ 湾,  ⁇ 湾,  ⁇ 湾,  ⁇ 湾,  ⁇ 湾,  ⁇ 湾湾,  ⁇ 湾湾圆 ⁇ 湾,  ⁇ 湾湾圆湾湾湾开发区  ⁇ 具中心 |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_recursive_depth_pressure_protocol_d07/smoke_seed10701/summary.json:$` | same generation reused for 2 distinct sources:  ⁇ 湾,  ⁇ 湾,  ⁇ 湾,  ⁇ 湾,  ⁇ 湾,  ⁇ 湾湾,  ⁇ 湾湾湾,  ⁇ 湾湾湾,  ⁇ 湾湾湾湾湾,  ⁇ 湾湾湾湾湾湾湾开发区  ⁇ 具湾的圆铺,  |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_recursive_depth_pressure_protocol_d07r1/smoke_seed10711/summary.json:$` | same generation reused for 5 distinct sources:  ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇  |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_recursive_depth_pressure_protocol_d07r1/smoke_seed10711/summary.json:$` | same generation reused for 2 distinct sources: In fact, I will have a number of the free of the free of all the way of the Terms of Use. |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_recursive_depth_pressure_protocol_d07r1/smoke_seed10711/summary.json:$` | same generation reused for 2 distinct sources: In this domain, I can do so in the other cases I's what you want to be able to do you want to be able to do so you want to do so I want you are so you can do so |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_recursive_depth_pressure_protocol_d07r1/smoke_seed10711/summary.json:$` | same generation reused for 2 distinct sources: In this domain, I can do so in the other cases I's what you want to be able to do you want to be able to do so you want to do so I want you are so you do it so  |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_recursive_depth_pressure_protocol_d07r2/smoke_seed10721/summary.json:$` | same generation reused for 4 distinct sources:  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_recursive_depth_pressure_protocol_d07r2/smoke_seed10721/summary.json:$` | same generation reused for 2 distinct sources: The requested URL /images/s. |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_recursive_depth_pressure_protocol_d07r2/smoke_seed10721/summary.json:$` | same generation reused for 2 distinct sources: The requested page is not used to be found on the site. |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_stone2_high_confidence_c02/formal_seed14201/dreams/step-000062953195.txt:$` | same generation reused for 2 distinct sources: The key to the door was opened. |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_stone2_high_confidence_c02/smoke/dreams/step-000062953195.txt:$` | same generation reused for 2 distinct sources: The key to the door was opened. |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_stone2_high_confidence_c02/smoke/dreams/step-000062963179.txt:$` | same generation reused for 2 distinct sources: The key to the last week was the last. |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_stone2_integrated_c03/smoke_seed16101/proof/generations.json:$` | same generation reused for 27 distinct sources: ,但,但,但,但,但,但,但,但,但,但,但,但 |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_stone2_integrated_c03/smoke_seed16101/proof/generations.json:$` | same generation reused for 13 distinct sources: ,在,但,但,但,但,但,但,但,但,但,但,但 |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_stone2_integrated_c03/smoke_seed16101/proof/generations.json:$` | same generation reused for 3 distinct sources: ,在,在,但,但,但,但,但,但,但,但,但,但 |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_stone2_integrated_c03/smoke_seed16101/proof/generations.json:$` | same generation reused for 3 distinct sources: ,在,在,在,在,在,在,在,在,在,在,在,在 |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_stone2_integrated_c03/smoke_seed16101/proof/generations.json:$` | same generation reused for 2 distinct sources: ,在,在在,在在,在在,在在,在在,在在,在在, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_stone2_integrated_c03/smoke_seed16101/proof/generations.json:$` | same generation reused for 4 distinct sources: ,在在,在在,在在,在在,在在,在在,在在,在在 |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_protocol_full_pipeline_d10/full_seed11001/pretrain_stop_step160k/wakes.jsonl:$` | same generation reused for 2 distinct sources: ,在 ⁇ 山上, ⁇ 山 ⁇ , ⁇ 山, ⁇ 山, ⁇ 山, ⁇ 山, ⁇ 山, ⁇ 山, ⁇ 山, ⁇ 山, ⁇ 山, ⁇ 山, ⁇ 山, ⁇ 山, ⁇ 山, ⁇ 山, ⁇ 山, ⁇ 山, ⁇ 山, ⁇ 山, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d08/formal_seed10812/random/summary.json:$` | same generation reused for 2 distinct sources:  ⁇ 子  ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d08/formal_seed10812/subheap/summary.json:$` | same generation reused for 2 distinct sources:  ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇  |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d08/formal_seed10812/summary.json:$` | same generation reused for 2 distinct sources:  ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇  |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d08/formal_seed10812/summary.json:$` | same generation reused for 2 distinct sources:  ⁇ 子  ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d08/formal_seed10813/free/summary.json:$` | same generation reused for 2 distinct sources:  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d08/formal_seed10813/summary.json:$` | same generation reused for 2 distinct sources:  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d08/smoke_r1_seed10801/free/summary.json:$` | same generation reused for 4 distinct sources:  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d08/smoke_r1_seed10801/free/summary.json:$` | same generation reused for 2 distinct sources:  ⁇ 子,我喜欢你,你喜欢你。 |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d08/smoke_r1_seed10801/random/summary.json:$` | same generation reused for 3 distinct sources:  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d08/smoke_r1_seed10801/random/summary.json:$` | same generation reused for 2 distinct sources:  ⁇ 子,就 ⁇ 了,就好好好事了。 |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d08/smoke_r1_seed10801/random/summary.json:$` | same generation reused for 2 distinct sources: Hooray to the World |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d08/smoke_r1_seed10801/subheap/summary.json:$` | same generation reused for 2 distinct sources: Hooray to the World |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d08/smoke_r1_seed10801/subheap/summary.json:$` | same generation reused for 2 distinct sources: The hotel's recreational facilities, which is a private bathroom. |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d08/smoke_r1_seed10801/summary.json:$` | same generation reused for 4 distinct sources:  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d08/smoke_r1_seed10801/summary.json:$` | same generation reused for 2 distinct sources:  ⁇ 子,就 ⁇ 了,就好好好事了。 |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d08/smoke_r1_seed10801/summary.json:$` | same generation reused for 2 distinct sources:  ⁇ 子,我喜欢你,你喜欢你。 |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d08/smoke_r1_seed10801/summary.json:$` | same generation reused for 2 distinct sources: Hooray to the World |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d08/smoke_r1_seed10801/summary.json:$` | same generation reused for 2 distinct sources: The hotel's recreational facilities, which is a private bathroom. |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d08/smoke_seed10801/free/summary.json:$` | same generation reused for 4 distinct sources:  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d08/smoke_seed10801/free/summary.json:$` | same generation reused for 2 distinct sources:  ⁇ 子,我喜欢你,你喜欢你。 |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d08/smoke_seed10801/random/summary.json:$` | same generation reused for 4 distinct sources:  ⁇ 子,  ⁇ 子,  ⁇ 子 |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d08/smoke_seed10801/random/summary.json:$` | same generation reused for 2 distinct sources:  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子 |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d08/smoke_seed10801/random/summary.json:$` | same generation reused for 2 distinct sources: Hooray to the World |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d08/smoke_seed10801/subheap/summary.json:$` | same generation reused for 2 distinct sources: Hooray to the World |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d08/smoke_seed10801/subheap/summary.json:$` | same generation reused for 2 distinct sources: The hotel's recreational facilities, which is a private bathroom. |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d08/smoke_seed10801/summary.json:$` | same generation reused for 4 distinct sources:  ⁇ 子,  ⁇ 子,  ⁇ 子 |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d08/smoke_seed10801/summary.json:$` | same generation reused for 2 distinct sources:  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子 |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d08/smoke_seed10801/summary.json:$` | same generation reused for 4 distinct sources:  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子,  ⁇ 子, |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d08/smoke_seed10801/summary.json:$` | same generation reused for 2 distinct sources:  ⁇ 子,我喜欢你,你喜欢你。 |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d08/smoke_seed10801/summary.json:$` | same generation reused for 2 distinct sources: Hooray to the World |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d08/smoke_seed10801/summary.json:$` | same generation reused for 2 distinct sources: The hotel's recreational facilities, which is a private bathroom. |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d09_scale/formal_seed10901/wakes.jsonl:$` | same generation reused for 2 distinct sources:  ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇  |
| warning | `GENERATION_REUSE` | `s3-generation/evidence/s3_structural_slot_ownership_d09_scale/formal_seed10901/wakes.jsonl:$` | same generation reused for 2 distinct sources:  ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇  |
| warning | `HIGH_ADJACENT_REPETITION` | `data-quality/evidence/purification_scale_ladder/formal_seed14108/120000/generation.json:$.examples[1]/record:1` | rate=0.276: LHHLHHLHLHLHLHHLHHLHHLHHLHHLHH |
| warning | `HIGH_ADJACENT_REPETITION` | `data-quality/evidence/purification_scale_ladder/formal_seed14108/120000/summary.json:$.generation.examples[1]/record:1` | rate=0.276: LHHLHHLHLHLHLHHLHHLHHLHHLHHLHH |
| warning | `HIGH_ADJACENT_REPETITION` | `data-quality/evidence/purification_scale_ladder/formal_seed14108/160000/generation.json:$.examples[2]/record:1` | rate=0.542:  ⁇  : : :  ⁇ ··卢克·顿 : : : : : : : : : : : 冒险 : |
| warning | `HIGH_ADJACENT_REPETITION` | `data-quality/evidence/purification_scale_ladder/formal_seed14108/160000/generation.json:$.examples[10]/record:1` | rate=0.958: (JJJJJJJJJJJJJJJJJJJJJJJJ |
| warning | `HIGH_ADJACENT_REPETITION` | `data-quality/evidence/purification_scale_ladder/formal_seed14108/160000/summary.json:$.generation.examples[2]/record:1` | rate=0.542:  ⁇  : : :  ⁇ ··卢克·顿 : : : : : : : : : : : 冒险 : |
| warning | `HIGH_ADJACENT_REPETITION` | `data-quality/evidence/purification_scale_ladder/formal_seed14108/160000/summary.json:$.generation.examples[10]/record:1` | rate=0.958: (JJJJJJJJJJJJJJJJJJJJJJJJ |
| warning | `HIGH_ADJACENT_REPETITION` | `data-quality/evidence/purification_scale_ladder/formal_seed14108/300000/generation.json:$.examples[2]/record:1` | rate=1.000:  ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇  |
| warning | `HIGH_ADJACENT_REPETITION` | `data-quality/evidence/purification_scale_ladder/formal_seed14108/300000/generation.json:$.examples[9]/record:1` | rate=0.593:  ⁇   ⁇   ⁇   ⁇ 的严厉  ⁇   ⁇   ⁇   ⁇   ⁇   ⁇  拳 ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇ 抗结宇宙。 |
| warning | `HIGH_ADJACENT_REPETITION` | `data-quality/evidence/purification_scale_ladder/formal_seed14108/300000/summary.json:$.generation.examples[2]/record:1` | rate=1.000:  ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇  |
| warning | `HIGH_ADJACENT_REPETITION` | `data-quality/evidence/purification_scale_ladder/formal_seed14108/300000/summary.json:$.generation.examples[9]/record:1` | rate=0.593:  ⁇   ⁇   ⁇   ⁇ 的严厉  ⁇   ⁇   ⁇   ⁇   ⁇   ⁇  拳 ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇   ⁇ 抗结宇宙。 |
| warning | `HIGH_ADJACENT_REPETITION` | `data-quality/evidence/purification_scale_ladder/formal_seed14108/40000/generation.json:$.examples[12]/record:1` | rate=0.533: 在 众人两个两个顶顶顶顶顶顶顶顶顶顶顶顶顶顶顶顶峰顶峰顶峰,两两 |
| warning | `HIGH_ADJACENT_REPETITION` | `data-quality/evidence/purification_scale_ladder/formal_seed14108/40000/generation.json:$.examples[17]/record:1` | rate=0.640: 我是能希望它它它它它它它它它它它它它它它它它是什么。 |
| warning | `HIGH_ADJACENT_REPETITION` | `data-quality/evidence/purification_scale_ladder/formal_seed14108/40000/summary.json:$.generation.examples[12]/record:1` | rate=0.533: 在 众人两个两个顶顶顶顶顶顶顶顶顶顶顶顶顶顶顶顶峰顶峰顶峰,两两 |
| warning | `HIGH_ADJACENT_REPETITION` | `data-quality/evidence/purification_scale_ladder/formal_seed14108/40000/summary.json:$.generation.examples[17]/record:1` | rate=0.640: 我是能希望它它它它它它它它它它它它它它它它它是什么。 |
| warning | `HIGH_ADJACENT_REPETITION` | `data-quality/evidence/purified_training_ab/formal_seed14104/purified/generation.json:$.examples[12]/record:1` | rate=0.364: 我想在南世世世世世开的。 |
| warning | `HIGH_ADJACENT_REPETITION` | `data-quality/evidence/purified_training_ab/formal_seed14104/purified/summary.json:$.generation.examples[12]/record:1` | rate=0.364: 我想在南世世世世世开的。 |
| warning | `HIGH_ADJACENT_REPETITION` | `s3-generation/evidence/s3_bounded_annealing_fold_c13/formal_seed_13301/learned/generation.json:$.examples[4]/record:1` | rate=0.200:  ⁇  ⁇ , 离舞舞 Room (显示地图 ) |
| warning | `HIGH_ADJACENT_REPETITION` | `s3-generation/evidence/s3_bounded_annealing_fold_c13/formal_seed_13301/learned/generation.json:$.examples[7]/record:1` | rate=0.219:  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇ 人,人我们用人倒红红红的海了,学习多了多了多了多的海上 |
| warning | `HIGH_ADJACENT_REPETITION` | `s3-generation/evidence/s3_bounded_annealing_fold_c13/formal_seed_13301/learned/summary.json:$.generation.examples[4]/record:1` | rate=0.200:  ⁇  ⁇ , 离舞舞 Room (显示地图 ) |
| warning | `HIGH_ADJACENT_REPETITION` | `s3-generation/evidence/s3_bounded_annealing_fold_c13/formal_seed_13301/learned/summary.json:$.generation.examples[7]/record:1` | rate=0.219:  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇ 人,人我们用人倒红红红的海了,学习多了多了多了多的海上 |
| warning | `HIGH_ADJACENT_REPETITION` | `s3-generation/evidence/s3_bounded_annealing_fold_c13/formal_seed_13302/ref_zero/generation.json:$.examples[7]/record:1` | rate=0.250: 许 ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇   ⁇ 包 ⁇  Spain ⁇ 鱼的 ⁇ 鱼  ⁇ 包 ⁇ 时 ⁇ 包 ⁇ ! |
| warning | `HIGH_ADJACENT_REPETITION` | `s3-generation/evidence/s3_bounded_annealing_fold_c13/formal_seed_13302/ref_zero/summary.json:$.generation.examples[7]/record:1` | rate=0.250: 许 ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇   ⁇ 包 ⁇  Spain ⁇ 鱼的 ⁇ 鱼  ⁇ 包 ⁇ 时 ⁇ 包 ⁇ ! |
| warning | `HIGH_ADJACENT_REPETITION` | `s3-generation/evidence/s3_bounded_annealing_fold_c13/formal_seed_13303/learned/generation.json:$.examples[3]/record:1` | rate=0.564:  ⁇ 网站致网站,您您您您您您您您您您您您您您您您您您您您您您您烈您烈铁本协议页面_ |
| warning | `HIGH_ADJACENT_REPETITION` | `s3-generation/evidence/s3_bounded_annealing_fold_c13/formal_seed_13303/learned/generation.json:$.examples[4]/record:1` | rate=0.667:  ⁇ 烈 ⁇ 塔塔塔塔塔塔塔塔塔塔塔塔塔塔塔塔塔塔塔塔塔 ( 英国 (Mto |
| warning | `HIGH_ADJACENT_REPETITION` | `s3-generation/evidence/s3_bounded_annealing_fold_c13/formal_seed_13303/learned/generation.json:$.examples[9]/record:1` | rate=0.500: 贡献者子 巧克力烈烈烈烈烈烈烈烈烈。 |

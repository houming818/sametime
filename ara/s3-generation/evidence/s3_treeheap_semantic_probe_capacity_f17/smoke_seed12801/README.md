# S3-TREEHEAP-SEMANTIC-PROBE-CAPACITY-F17

- Cases: `480`
- Macro AUROC: `{'input_bow': 0.5266666666666666, 'fold_root': 0.5339930555555557, 'fold_leaf': 0.5240972222222223, 'read_facing_root': 0.5630555555555555, 'read_facing_leaf': 0.5240972222222223}`
- Gates: `{'O0_f16_manifest_contract': True, 'O1_finite': True, 'O2_frozen_checkpoint': True, 'C0_shuffled_control_rejected': True, 'P1_fold_root_semantic_recoverable': False, 'P2_read_root_semantic_recoverable': False, 'P3_fold_compression_loss_pattern': False, 'P3_read_compression_loss_pattern': False, 'P4_fold_root_probe_underfit': False, 'P4_read_root_probe_underfit': False}`
- Diagnosis: `strong probe remains weak at leaf and root; build semantic protocol before FOLD change`

# S3-TREEHEAP-CROSS-LANGUAGE-SEMANTIC-RECOVERABILITY-F16

- Cases: `480`
- Macro AUROC: `{'input_bow': 0.5327083333333333, 'fold_root': 0.5974305555555555, 'fold_leaf': 0.6003124999999999, 'read_facing_root': 0.5842013888888888, 'read_facing_leaf': 0.6003124999999999}`
- Gates: `{'O0_balanced_complete_family_holdout': True, 'O1_finite': True, 'O2_frozen_checkpoint': True, 'C0_shuffled_control_rejected': True, 'P1_fold_root_semantic_recoverable': False, 'P2_read_root_semantic_recoverable': False, 'P3_fold_compression_loss_pattern': False, 'P3_read_compression_loss_pattern': False}`
- Diagnosis: `no strong target-label direction even at leaf; compression is not isolated`

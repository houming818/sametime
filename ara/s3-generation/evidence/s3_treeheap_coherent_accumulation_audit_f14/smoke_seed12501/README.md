# S3-TREEHEAP-COHERENT-ACCUMULATION-F14

- Preregistration commit: `78ee32b`
- Width-contract fix: `d40e252`
- io taskd 407: failed before model execution because width 32 exceeded the checkpoint leaf limit.
- io taskd 408: completed after the preregistered fixed-width contract was corrected to width 16.
- GPU limit/peak: `270 W / 161.99 W`; peak temperature/memory: `53 C / 892 MiB`.
- Cases: `14`
- Gates: `{'O0_fixed_single_token_contract': True, 'O1_zero_parity_and_finite': True, 'O2_frozen_checkpoint': True, 'P1_native_root_count_response': False, 'P2_native_full_count_response': True, 'P3_normalized_sum_root_advantage': False, 'P4_root_read_causal_growth': False}`
- Native root margins: `{'0': -15.689596811930338, '1': -14.629613558451334, '2': -13.711588144302368, '4': -13.591166337331137, '8': -14.889808972676596}`
- Mean-FOLD root margins: `{'0': -13.134872436523438, '1': -12.420756816864014, '2': -11.963816245396933, '4': -11.435981591542562, '8': -11.02328872680664}`
- Native full margins: `{'0': -14.17781893412272, '1': -12.983168443044027, '2': -12.49176828066508, '4': -11.150639057159424, '8': -9.62138589223226}`
- Runtime seconds: `3.98`

This is a frozen synthetic mechanism audit, not a translation-quality result.

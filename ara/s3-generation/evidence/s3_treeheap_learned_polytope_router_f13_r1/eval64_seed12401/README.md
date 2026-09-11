# F13-R1 expanded evaluation

- Gates: {"O0_reload_and_finite": true, "O1_frozen_and_hash_stable": true, "P1_full_bleu_advantage": true, "P2_block_majority": false, "P3_generation_health": false}
- Grouped block wins: 2/8
- Runtime seconds: 198.727

## Evidence audit

- Pre-registration commit: `7f7b0a4`
- taskd: `406`, status `done`, exit code `0`
- Scope: 64 held-out pairs, 128 bidirectional rows, 8 fixed blocks
- Input hashes and frozen model state remained exact
- Full BLEU delta: `+0.498423`; grouped block wins: `2/8`
- Grouped minus scalarized test NLL: `+0.001015`
- P2 failed on block majority; P3 failed on repetition (`0.03451` vs `0.00494`)

# STONE-1 C04: Recursive Private-Protocol Growth Trajectory

Date: 2026-07-23
Status: supported path-sensitive root compression / recursive decoder growth rejected / single seed
Milestone: `STONE-1` (still incomplete)
Claim: `S3-STONE1-PROTOCOL-GROWTH-C04`
Predict: `P-S3-STONE1-PROTOCOL-GROWTH-04`

## Question

C03 showed that the 50.27M model was still improving at 15,625 updates, while
its decoder stopped at root. This does not decide whether the recursive private
protocol was undertrained or whether token cross-entropy converges toward a
root-only decoder. C04 observes the entire growth trajectory without changing
the model, loss, data, or optimizer contract.

## Frozen Run

Train one `D=320, H=512` model from scratch with seed `71902` for `62,500`
continuous AdamW updates on the frozen C02/C03 one-million-pair WMT split.
Do not reload the C03 checkpoint because it does not contain AdamW state.

Record normal validation NLL and route mass every 500 updates. At updates
`15,625`, `31,250`, `46,875`, and `62,500`, also record:

1. root-only versus full-depth NLL;
2. post-fold unfolded-child address-swap damage;
3. pre-fold left/right subtree mirror damage at every fold depth;
4. force-algebraic codec damage;
5. final non-teacher-forced generation metrics.

The pre-fold mirror is an evaluation intervention only. It swaps left/right
subheaps before the selected fold operation, recomputes root, and therefore
tests whether path handedness is encoded into root. The existing post-fold
swap leaves root unchanged and tests whether the decoder directly reads
unfolded children.

## Competing Predictions

Protocol-growth prediction:

```text
G1  NLL at 31,250 improves at least 0.08 over NLL at 15,625
G2  non-root route mass reaches 0.10 after 15,625 and remains >= 0.05 finally
G3  final root-only minus full-depth NLL >= 0.10
G4  final post-fold address-swap damage >= 0.10 NLL
G5  final maximum pre-fold mirror damage >= 0.10 NLL
```

Root-compression prediction:

```text
R1  held-out NLL improves while G2-G4 remain false
R2  pre-fold mirror damage is positive
```

`R1+R2` means recursive folding learned a path-sensitive compressed root, but
the decoder did not learn a recursive inverse/read protocol. If both post-fold
and pre-fold damage remain near zero, the model has converged toward a bag-like
root shortcut.

## Decision Boundary

One seed can reject a simple undertraining story but cannot establish stable
emergence. Run three seeds only if G2, G3, or G4 becomes positive at a later
milestone. Stop immediately on non-finite gradients, GPU loss, or rising
validation NLL across three consecutive milestones. C04 does not authorize a
92M model and does not alter STONE-1 product thresholds.

## Result

The formal run completed all `62,500` updates in `12,103` seconds on the io
RTX 3090. Gradients remained finite and peak allocated VRAM was `2.104 GiB`.
Validation NLL improved continuously:

| Update | Validation NLL | Perplexity | Non-root route mass | Root-only gain | Post-fold swap damage | Maximum pre-fold mirror damage |
|---:|---:|---:|---:|---:|---:|---:|
| 15,625 | 4.1879 | 65.88 | 0.0000 | 0.0000 | 0.0000 | 1.6279 |
| 31,250 | 3.8867 | 48.75 | 0.0000 | 0.0000 | 0.0000 | 0.6018 |
| 46,875 | 3.7331 | 41.81 | approximately 0 | 0.0000 | 0.0000 | 0.6860 |
| 62,500 | 3.6613 | 38.91 | approximately 0 | 0.0000 | 0.0000 | 0.6747 |

Final test NLL was `3.5818`, token BLEU-4 was `12.5107`, and severe
repetition rate was `0.0275`. The output was nonempty for every evaluated
example, but exact sentence recovery was only `0.004`; this remains far below
the STONE-1 product target.

Gates `G1` and `G5` passed. Gates `G2`, `G3`, and `G4` failed. More training
therefore improved prediction quality but did not make the decoder read lower
TreeHeap levels: final route mass was effectively all at root, root-only and
full-depth NLL were equal, and swapping unfolded child addresses after root
formation caused zero damage.

The positive pre-fold intervention is the important counterpoint. Mirroring
left/right subheaps before folding changed root formation and increased final
validation NLL at every tested depth. Damage ranged from `0.1106` to `0.6747`
NLL, with the largest damage at fold depth 4. Forcing the fixed algebraic codec
also caused `+1.3456` NLL damage. The learned root is therefore not a
bag-of-tokens shortcut: its state depends on ordered recursive folding and on
the learned codec.

## Decision

The simple hypothesis "the decoder reads only root because 15,625 updates were
insufficient" is rejected under this training contract. The supported
single-seed interpretation is:

```text
ordered leaves
  -> path-sensitive recursive fold
  -> compressed root state
  -> root-only surface decoder
```

This is a real TreeHeap encoder-path effect, but it is not yet the intended
recursive encoder-decoder private protocol. C04 does not show that root gives a
coarse outline while deeper levels add detail, does not establish stable
emergence across seeds, and does not complete STONE-1.

Evidence: `../evidence/s3_stone1_protocol_growth_trajectory/`.

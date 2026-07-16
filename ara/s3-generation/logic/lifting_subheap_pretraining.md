# Lifting Subheap Pretraining

Status: preregistered experiment

Claim ID: `S3-LIFT-SUBHEAP-PRETRAIN-C01`

## Question

Can unlabeled text train the current reversible lifting TreeHeap to write
multi-scale associations into `H_state`, while also training a decoder that
generates surface token sequences?

This is not another test of the old nonlinear compose encoder. The encoder is
the `learned_update` lifting system supported by
`S2-ADAPTIVE-LIFT-WMT-C01`:

\[
D = R - P_\theta(L), \qquad U = L + A_\phi(D).
\]

`FOLD` recursively applies this operation from leaves to root. `UNFOLD` is its
closed inverse, and recursive `READ` chooses a probability mass over root,
intermediate nodes, and leaves while generating each output token.

## Data

Use real, unlabeled Chinese documents from the NAS/local mirror:

- news2016zh;
- Chinese Wikipedia;
- webtext2019zh.

Tokenization uses the existing bilingual WMT Massive SentencePiece model. A
training example is a clean 64-token block. No syntax, category, QA, or
translation label is supplied.

## One Loss, Three Curricula

Both models minimize exactly the same missing-span generation cross-entropy:

\[
\mathcal{L}_{span}
= -\sum_{t=1}^{|X_v|+1}
  \log p_\Theta(x_t\mid H_{visible},x_{<t}),
\]

where `+1` is EOS and teacher forcing supplies only the preceding gold output
tokens. The original text supplies its own target, so this is self-supervised.

The only experimental variable is how the missing span is selected:

1. `token_only`: hide one leaf (`width=1`).
2. `random_span`: sample width uniformly from `{1, 2, 4, 8}`, then hide a
   contiguous span at an arbitrary leaf offset.
3. `subheap`: hide one complete, address-aligned subheap with width sampled
   uniformly from `{1, 2, 4, 8}`.

For example, a width-4 mask always starts on a multiple-of-four leaf address.
It therefore removes exactly one node's receptive field, rather than an
arbitrary four-token window.

`random_span` is the primary matched-supervision control: it has the same
target-width distribution and therefore approximately the same number of
predicted tokens per update as `subheap`, but its spans need not be one node's
exact receptive field. `token_only` remains a useful curriculum diagnostic,
not the sole structural baseline.

The mask is applied before `FOLD`. The pump must propagate the visible context
upward, and the decoder must generate the removed subheap from the resulting
TreeHeap state. There is no separate category or structure loss.

## Predict

All thresholds are evaluated on held-out documents and identical fixed-width
test suites.

- `P1`: both curricula remain finite and lower held-out NLL from their initial
  value.
- `P2`: on the common address-aligned width-4 and width-8 test suites,
  `subheap` improves mean token NLL by at least `0.03` over `random_span` and
  does not lose to `token_only` on both widths.
- `P3`: `subheap` produces non-empty greedy output, improves greedy token
  accuracy on at least three of the four widths, keeps adjacent repetition at
  or below `0.40`, and produces at least `10%` unique outputs. These extra
  checks prevent a repeated high-frequency phrase from passing as generation.
- `P4`: shuffling complete source TreeHeaps across the batch increases NLL by
  at least `0.10` on width-4 or width-8 recovery.
- `P5`: zeroing the root or shuffling at least two encoded detail depths
  increases NLL by at least `0.03`; generation is therefore not a decoder-only
  language-model shortcut.
- `P6`: swapping every adjacent leaf pair before `FOLD` without retraining
  increases NLL by at least `0.03`, showing that the kernel's left/right input
  slots participate. Reordering already unfolded nodes is not a valid address
  intervention because recursive READ is intentionally permutation-invariant
  within a level.
- `P7`: lifting closure remains below `1e-10` state MSE after training.

## Interpretation

Passing `P1` only proves ordinary conditional generation.

Passing `P2-P3` supports the narrower inductive claim that a curriculum made
of complete TreeHeap receptive fields teaches larger missing structures more
effectively than a target-length-matched arbitrary-span curriculum under equal
optimizer updates. Wall-clock and token throughput are recorded separately;
this smoke does not make an efficiency claim.

Passing `P4-P6` is required before saying that the generated text depends on
TreeHeap state, depth, and addresses. It does not prove semantic categories,
world knowledge, consciousness, or superiority over a scaled Transformer.

The resulting checkpoint is a pretraining checkpoint only if it passes these
causal gates. A later, separate transfer experiment must compare WMT
fine-tuning from this checkpoint against the same model trained from scratch.

## Result: 5K-Update Pilot

Executed on `io` with real news, Wikipedia, and web text, the WMT Massive 32K
tokenizer, 256-dimensional states, batch 64, and 5,000 optimizer updates per
curriculum. Evidence is in
`evidence/s3_lifting_subheap_pretrain_5k/`; 392 MiB of checkpoints are stored
at `/mnt/nas/ara/s3-generation/evidence/s3_lifting_subheap_pretrain_5k/`.

Held-out NLL on the common aligned test suites was:

| curriculum | width 1 | width 2 | width 4 | width 8 |
|---|---:|---:|---:|---:|
| token only | 6.5107 | 7.9457 | 9.1347 | 9.8977 |
| matched random span | 3.6174 | 4.6399 | 5.4216 | 5.9718 |
| aligned subheap | **3.6023** | **4.5774** | **5.3852** | **5.9179** |

The aligned curriculum improved width-4/8 NLL over the matched random-span
control by `0.0364/0.0540`. Width-8 free generation had token accuracy
`0.1310`, adjacent repetition `0.0854`, unique-output fraction `0.8262`, and
zero exact recovery. It learned non-empty conditional generation, but not
high-quality reconstruction.

The causal audit split the claim:

- shuffling complete source states cost `+0.1345` NLL;
- shuffling detail depths 0..5 cost
  `+0.5311/+2.1561/+0.8220/+2.8830/+5.5623/+9.6856`;
- removing all source state cost `+1.5645`;
- zeroing root cost only `+0.0042`;
- swapping every adjacent left/right leaf pair cost only `+0.0022`;
- lifting closure MSE remained below `1e-10`.

Therefore `P1/P2/P3/P4/P7` passed and `P5/P6` failed. The full claim is not
supported. The narrower evidence is that multi-scale subheap masking avoids
single-token curriculum collapse, gives a small gain over a matched arbitrary
span curriculum, and trains a source- and detail-dependent seq2seq generator.
It remains a detail-dominant protocol: root and left/right addresses are not
yet causal enough to call it a complete TreeHeap encoding protocol.

## Falsification

Reject the structural curriculum advantage if the target-length-matched
random-span control matches or beats subheap masking on the common width-4/8
suites. Reject structural dependence
if source/root/depth/left-right interventions have negligible effect. Reject
useful pretraining if generation stays empty or degenerate, even when NLL
falls. A positive smoke result must be repeated at larger scale and more than
one seed before promotion beyond `supported pilot`.

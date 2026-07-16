# S2 WMT over the TreeHeap Lifting Pump

## Claim

`S2-LIFT-WMT-C01`:

> The lifting information pump can serve as a real English-to-Chinese S2
> source representation.  Translation loss should learn a query-conditioned
> recursive READ that begins at root, unfolds addressed details only when
> needed, uses more than one resolution, and remains causally dependent on the
> source root, detail addresses, and recursive pairings.

This is the first downstream test after `S1-LIFT-PUMP-C01`.  It does not assume
that any depth has a predefined linguistic label.

## Source Encoder

English source tokens are written to leaves.  At every depth the same bounded
predictor is used:

\[
D=R-P_\theta(L),
\qquad
U=L+\frac12D.
\]

Only `U` recurses upward.  The final state is:

\[
H_{src}=(root,D^{(D)},\ldots,D^{(1)}).
\]

The inverse is exact:

\[
L=U-\frac12D,
\qquad
R=D+P_\theta(L).
\]

## Recursive READ

At Chinese generation step `t`, decoder state `q_t` starts with probability
mass 1 at source root.  At each visited node it predicts:

\[
p_{stop}=\sigma K_{stop}(q_t,H_v,depth(v)).
\]

Stopped mass contributes the current node to source context.  Remaining mass
UNFOLDs the node, then a local left/right softmax distributes that mass to its
children.  The process continues until all probability has stopped or reached
leaves.  The final context is a weighted sum over a query-dependent frontier.

There are no route, syntax, alignment, mask, echo, or internal-state labels.
Only normal WMT target-token cross-entropy is optimized:

\[
L_{S2}=-\sum_t\log p(y_t\mid y_{<t},H_{src}).
\]

## Models

1. `target_only`: Chinese decoder with no English source information;
2. `flat_seq`: GRU source sequence with ordinary attention;
3. `lifting_root`: decoder sees only the lifting root;
4. `lifting_full`: lifting state is fully UNFOLDed and decoder attends leaves;
5. `lifting_recursive`: registered TreeHeap model using root-first recursive
   probability READ.

`lifting_full` is an information upper control, not a TreeHeap efficiency
claim: full UNFOLD exposes the original source resolution.

## Predict

`P-S2-LIFT-WMT-01`:

1. recursive READ held-out NLL is at most `6.55`, improving over the historical
   root-exclusive TreeHeap NLL `6.6072`;
2. recursive READ beats `lifting_root` by at least `0.05` NLL;
3. recursive READ stays within `0.10` NLL of `lifting_full`;
4. cross-sample source-state shuffle raises recursive NLL by at least `0.20`;
5. root shuffle raises NLL by at least `0.05`, and at least one detail-level
   shuffle raises NLL by at least `0.02`;
6. recursive pair breaking raises NLL at two or more depths by at least `0.02`;
7. native READ assigns at least `0.05` stop mass to two different depths and
   does not send more than `0.90` mass to leaves;
8. lifting FOLD/UNFOLD state MSE remains below `1e-10`, gradients stay finite,
   and greedy output is non-empty.

Flat-sequence superiority is not preregistered.  Its result is reported as a
baseline battle rather than silently omitted.

## Interventions

- `source_shuffle`: replace complete source H_state across samples;
- `root_shuffle`: replace only root across samples;
- `detail_shuffle_d`: replace one addressed detail depth across samples;
- `pair_break_d`: cross-sample replace right subtrees before FOLD at one depth;
- `force_root`: force all READ mass to stop at root;
- `force_leaf`: force all READ mass to expand to leaves.

## Falsification and Boundary

If source shuffle is neutral, the system is only a Chinese language model.  If
full UNFOLD works but recursive READ fails, the blocker is route/collapse.  If
both fail, the blocker is cross-language learning or the target decoder.  If
recursive READ expands entirely to leaves, S2 may work but no adaptive
multiresolution claim is supported.

Even a full pilot pass establishes only a WMT mechanism result.  It does not
prove production translation quality, compression, compute advantage,
human-readable semantics, world knowledge, or consciousness.

## Results

The registered pilot (`5K/500/500`, five epochs, 192 dimensions) passed all
eight gates.  The near-full WMT run then used every length-filtered example
that fit the registered split: `27K/2K/2K`, ten epochs, 256 dimensions, on
`io`'s RTX 3090.  The best checkpoint was selected only by validation NLL.

| Model | Test NLL | PPL | Token BLEU-4 |
|---|---:|---:|---:|
| target only | 5.7189 | 304.57 | 0.117 |
| flat sequence | **4.8103** | **122.77** | **3.169** |
| lifting root | 5.4337 | 229.01 | 1.585 |
| lifting full UNFOLD | 5.1342 | 169.73 | 1.881 |
| lifting recursive READ | **5.0903** | **162.44** | **2.528** |

The recursive model improved over root-only by `0.3434` NLL and slightly
improved over full UNFOLD by `0.0439`.  It still lost to the flat sequence
baseline by `0.2800`, so this is not a translation-quality victory.

The causal tests were strong.  Complete source shuffle added `1.4450` NLL;
root shuffle added `1.7204`; detail shuffles from leaf-near to root-near added
`[0.1171, 0.1422, 0.2402, 0.4960, 0.4857]`; and pair breaking added
`[0.4807, 0.4414, 0.4107, 0.4354, 0.2105]`.  Native stop mass was distributed
across depths as `[0.3329, 0.0288, 0.1283, 0.0384, 0.0073, 0.4643]`, rather
than collapsing entirely to root or leaves.  FOLD/UNFOLD closure MSE was
`1.73e-14`.

**Decision:** `S2-LIFT-WMT-C01` is supported as a real-data mechanism claim.
Translation loss learned a source-causal, query-conditioned, multiresolution
TreeHeap READ.  The flat model's better NLL remains the next quality gap.  The
result does not establish compression or compute advantage because the current
implementation materializes all levels before applying probabilistic READ.

Evidence is in `evidence/s2_lifting_pump_wmt_full/`; exact checkpoints are
archived at `/mnt/nas/ara/s3-generation/evidence/s2_lifting_pump_wmt_full/`.

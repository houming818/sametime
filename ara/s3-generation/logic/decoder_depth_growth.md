# Variable-arity decoder depth growth

Claim ID: `S3-DECODER-DEPTH-GROWTH-C01`

Status: **open / preregistered overnight pilot**

## Question

Does a decoder become more capable as one shared READ kernel recursively reads
deeper TreeHeap frontiers, and is that growth better explained by learned
contiguous FOLD than by an equal-parameter random tree or flat multiresolution
memory?

The pilot uses 128 BPE-token contexts from held-out real Chinese documents and
predicts the following 32 tokens. Decoded character counts are recorded; the
human-facing target is approximately 150 Chinese characters, but BPE count is
the actual TreeHeap leaf count.

## Construction

One model supports maximum arity eight. The same ordered child-slot FOLD kernel
is reused at every depth and for every arity. Training samples only
`k in {2,4,8}`. Evaluation covers `k=2..8`.

Three exactly parameter-matched variants are trained:

1. `tree`: recursively FOLD adjacent ordered child groups;
2. `random`: apply a fixed nonlocal permutation before every FOLD level;
3. `flat`: derive equal-width frontiers directly from leaf average pools.

All variants retain the same modules and parameter count. Unused modules remain
allocated, so a parameter-count difference cannot explain the result.

At target step `t`, the shared reader begins from the decoder query and reads
frontiers cumulatively from root through dose `d`:

$$q_{j+1}=R_phi(q_j,S_j), 0 <= j <= d$$

No depth owns a private READ parameter table.

## Controls

- `random`: separately trained random-link hierarchy;
- `flat`: separately trained equal-frontier flat memory;
- `shuffled_links`: frozen tree model evaluated with random links;
- `repeated_root`: each deeper frontier is replaced by repeated root vectors.

The primary comparison uses compressed doses only and excludes the final leaf
frontier, where all methods can recover direct token embeddings.

## Pilot predictions

`P1 depth growth`: averaged over seen arities, tree best compressed NLL improves
over root-only NLL by at least `0.05`.

`P2 structural advantage`: tree mean compressed NLL beats the better of random
and flat by at least `0.02`.

`P3 link causality`: evaluating the frozen tree model with shuffled links raises
mean compressed NLL by at least `0.02`.

`P4 unseen arity`: at least two of `k in {3,5,6,7}` show a tree compressed-NLL
advantage of at least `0.02` over both trained controls.

`P5 long context`: median decoded source length is at least 100 Chinese
characters.

`P6 finite`: every train/evaluation metric and parameter remains finite.

This is a one-seed pilot. Passing gates does not upgrade the claim beyond
`pilot support`; it authorizes a three-seed run and explicit unseen-depth test.

## Falsification and boundaries

If only P1 passes, the experiment supports additional readable information, not
TreeHeap structure. If tree does not beat random/flat, or link shuffling is
neutral, the structural claim remains unsupported. Failure is retained.

This experiment does not prove semantic coarse/fine specialization, TreeHeap
superiority over Transformer, world knowledge, consciousness, or a universal
optimal branching factor.


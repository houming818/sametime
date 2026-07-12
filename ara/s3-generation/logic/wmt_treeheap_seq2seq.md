# Real WMT TreeHeap Seq2Seq

## Minimal Architecture

This experiment starts from raw WMT English and Chinese, not from a historical
TreeHeap checkpoint.  SentencePiece converts both sides to one joint BPE
vocabulary.  The encoder writes source token vectors into TreeHeap leaves.

For each adjacent left/right pair it applies an ordered local kernel:

\[
h_p = \operatorname{Compose}_\theta(h_l + e_L, h_r + e_R)
\]

until one root remains.  The decoder does not only read the root.  At each
Chinese generation step, its GRU state queries every valid leaf and internal
node:

\[
p(i\mid q_t,H)=\operatorname{softmax}_i K_\theta(q_t,h_i)
\]

\[
c_t=\sum_i p(i\mid q_t,H)h_i
\]

and uses `c_t` to predict the next BPE token.  This is a probability container
over TreeHeap subheaps; greedy decoding collapses it only to emit surface text.

## Training Objective

\[
L=\sum_t \operatorname{CE}(p_\theta(y_t\mid y_{<t},H(x)), y_t)
\]

There is no gold route, parse tree, category, code length, or old checkpoint.
The only supervision is the real Chinese reference sequence.

## Why The Baselines Matter

`bow` receives no source order.  `flat_seq` receives source order but no
recursive subheap states.  A TreeHeap result below flat GRU is still useful: it
locates the present weakness honestly.  A TreeHeap result above BoW but below
flat GRU shows basic translation feasibility without demonstrating structural
advantage.

## Structural Audit

After training, evaluate the same TreeHeap checkpoint with:

```text
full       leaves and internal nodes are readable
leaf_only  internal nodes hidden
root_only  only the final root is readable
```

If full generation is unchanged after both ablations, the tree is decorative.
If internal-node removal damages held-out generation while leaf-only/root-only
do not match full, internal subheaps are carrying usable translation state.

## Boundaries

This experiment cannot establish WMT quality, semantic Huffman coding, or a
general advantage over Transformers.  It is a real-data feasibility and
mechanism checkpoint: can a TreeHeap encoder participate in actual seq2seq
learning and decoding at all?

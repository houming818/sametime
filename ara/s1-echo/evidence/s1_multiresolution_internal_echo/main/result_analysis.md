# Result Analysis

Status: **partial support**.

The experiment closes the leaf bypass and demonstrates a strong learned
parent-only codec.  Multichannel FOLD reconstructs held-out 16-token blocks at
`0.999428` token top-1 and `0.991211` block exact.  Removing the closest parent
level or recursively mismatching right-child addresses raises NLL by
`74.425402` and `54.368653`.

The stronger multiresolution reading is rejected.  READ assigns `0.999786` of
its mass to the closest parent level.  Removing every higher level, zeroing the
root, or replacing the root across samples is numerically neutral.  The system
has learned eight addressed pair codes, not a hierarchy of causally useful
resolutions.

The algebraic channels help substantially against the symmetric mean-only
control, but only narrowly against a generic nonlinear binary FOLD: NLL gain
`0.002969`, below the registered `0.02` gate.  Therefore the evidence supports
nonlinear ordered FOLD, not a unique advantage for the selected
mean/diff/product/joint basis.

The next proof must make disjoint pair codes insufficient.  It should require a
held-out answer that depends on interactions spanning four, eight, or sixteen
tokens, while retaining no leaf access and separately preregistering causal
gates for levels above the closest parent.

# WMT TreeHeap Functional Equivalence

## Terminology

The structural objects are a `TreeHeap` and its addressed `subheap` states.

Functional equivalence is a relation between two same-address TreeHeap states:

\[
H_a \sim_f H_b
\quad\Longleftrightarrow\quad
D\left(p(y\mid H_a,C),p(y\mid H_b,C)\right)\text{ is small}
\]

where `C` is the remaining source frontier and the teacher-forced target
context.  This definition is operational: equivalent TreeHeaps may contain
different words, but the trained decoder uses them similarly in the measured
translation context.

## Claim

`S2-TREEHEAP-FUNCTIONAL-EQUIV-C01`:

> A translation-trained learned TreeHeap frontier should contain repeatable
> groups of functionally equivalent subheaps.  On held-out WMT examples,
> replacing one frontier subheap with a same-group donor should damage the
> target distribution less than replacing it with an equally distant
> different-group donor.  This separation should be stronger for the learned
> TreeHeap than for fixed-route and random-route TreeHeap controls.

This is deliberately narrower than claiming semantic categories, syntax,
world knowledge, or improved BLEU.  It asks whether the existing TreeHeap
states have reusable translation behavior.

## Why Raw Cosine Is Insufficient

If the same-group donor is simply closer in vector space, lower damage only
shows continuity.  The proof therefore matches each different-group donor to
the cosine distance of its same-group donor:

\[
\left|d(H,H_{same})-d(H,H_{different})\right| \rightarrow \min.
\]

Both donors also occupy the same one of the four frontier addresses.  The
comparison asks whether group membership explains causal behavior after these
two simple shortcuts are controlled.

## Protocol

1. Freeze an existing WMT frontier checkpoint.
2. Use only the validation split to fit `K` groups independently at each
   frontier address.
3. Assign held-out test subheaps to those groups without refitting.
4. For each test sentence, select one frontier address and construct:
   - a same-group donor;
   - a different-group donor matched to the same cosine distance;
   - a random same-address donor.
5. Replace only that TreeHeap state and run the frozen decoder with the same
   Chinese teacher-forcing context.
6. Measure target NLL increase, KL divergence from the unmodified prediction,
   and token argmax changes.
7. Repeat the entire audit for learned-, fixed-, and random-route TreeHeap
   checkpoints.

The validation set nominates the groups; the test set judges them.  No POS,
dependency, semantic class, or target label is used to create a group.

## Predictions

Primary registered predictions:

1. donor cosine-distance mismatch is at most `0.02` for same versus matched
   different groups, with at least 50% of test examples admitting such a
   matched pair;
2. for the learned TreeHeap, `NLL_delta(different) - NLL_delta(same) > 0.02`
   and its bootstrap 95% interval excludes zero;
3. the learned separation exceeds both fixed-route and random-route
   separations by at least `0.01` NLL.

The result is `partial` if only prediction 2 passes.  It is `not supported` if
same- and different-group exchanges are indistinguishable, or if controls show
the same separation.

## Falsification and Limits

The claim is rejected at this checkpoint if distance matching fails, if the
held-out causal separation is absent, or if fixed/random TreeHeaps match the
learned model.  Passing would establish only checkpoint-local functional
equivalence under WMT teacher forcing.  It would not prove that the groups are
human semantic categories, that they are stable across seeds, or that using
them improves translation quality.

## Result

The io audit completed on all 500 held-out WMT examples.  The strict matching
revision found a same-group and different-group donor at nearly identical
cosine distance for every learned-TreeHeap example:

| TreeHeap | Same-group NLL delta | Different-group NLL delta | Different - same | Bootstrap 95% |
|---|---:|---:|---:|---:|
| learned route | 0.01865 | 0.02053 | 0.00187 | [-0.00395, 0.00745] |
| fixed route | 0.00870 | 0.00584 | -0.00285 | [-0.00793, 0.00199] |
| random route | 0.01206 | 0.00635 | -0.00571 | [-0.01197, 0.00075] |

For the learned TreeHeap, same/different donor cosine distances were
`0.620170/0.620222`, an error of only `0.000052`.  Therefore distance mismatch
cannot explain the null result.  All three registered gates evaluate to
`true/false/false`; the claim is **not supported at this checkpoint**.

The useful positive observation is narrower: replacing one of four TreeHeap
frontier states changes target predictions, so the decoder does read these
states.  What failed is the stronger idea that unsupervised geometric groups in
the current states already behave as reusable translation-equivalent classes.
The next architecture should train reuse explicitly instead of naming clusters
after the fact.

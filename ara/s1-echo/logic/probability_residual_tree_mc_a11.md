# A11 Probability-Residual Tree Monte Carlo Search

Date: 2026-09-23

Status: preregistered smoke; no result at registration time.

## Question

Can a TreeHeap embedding constructor discover a useful F function from an
accumulated corpus context field without using next-token targets or allowing
one sentence gradient to rewrite the tree directly?

## Data Contract

Reuse the frozen real-WMT count artifact from
`s1_real_corpus_annealed_token_space/formal_200k_seed19101/context_counts.pt`.
It contains 512 target-token rows by 1,024 context dimensions and a previously
sealed WMT test count matrix.

The existing train count matrix is split once by seeded binomial thinning:

```text
fit_counts + search_dev_counts = original_train_counts
```

Monte Carlo search may use only `fit_counts` and `search_dev_counts`. The
original WMT test matrix is evaluated only after the globally best search state
has been selected.

## State And Operator

The search state is a complete depth-5 binary TreeHeap. Each internal node
stores a unit observation axis and a scalar split threshold. A token begins as
the square-root probability coordinate of its fit-context distribution and
follows the node decisions to one of 32 leaves.

For child masses `m_L, m_R` and raw context prototypes `mu_L, mu_R`, FOLD is:

```text
m_parent = m_L + m_R
mu_parent = (m_L * mu_L + m_R * mu_R) / m_parent
```

The child residual is `delta_child = mu_child - mu_parent`. Along an occupied
root-to-leaf path:

```text
mu_leaf = mu_root + sum(delta_path)
```

This gives the proposed F a probability-field interpretation. No decoder,
translation label, next-token target, or recursive READ is involved.

## Search

Start from a deterministic two-pole recursive partition. At each Monte Carlo
iteration:

1. choose one occupied internal node;
2. propose either a token-pair axis or a bounded perturbation of its current
   axis;
3. choose a threshold from the local projected token distribution;
4. reroute all token rows and rebuild leaf fit prototypes;
5. score held-out `search_dev_counts` by context NLL;
6. accept improvements and occasionally accept worse states according to a
   decaying Metropolis temperature;
7. retain a separate global-best state.

There is no balance quota or minimum branch traffic in the objective. Empty
or chain-like structures are allowed and must survive on predictive evidence,
not by receiving artificial flow.

## Claim

`S1-F-MC-A11-C01`: On the frozen real-WMT context field, Monte Carlo search
over local TreeHeap split rules can improve dev context reconstruction over
the deterministic initial tree while preserving the probability FOLD and
path-residual identities.

## Preregistered Gates

Mechanical gates:

```text
all scores finite
the sealed WMT test matrix is absent from proposal scoring
at least one proposal accepted
FOLD conservation max abs <= 1e-10
occupied-path residual closure max abs <= 1e-10
```

Empirical smoke gates:

```text
best dev NLL <= initial dev NLL - 0.005
best sealed-test NLL <= initial sealed-test NLL + 0.01
best sealed-test NLL < random-route sealed-test NLL
best leaf utilization >= 0.50
```

The claim is supported only if all mechanical and empirical gates pass. A
failure remains evidence about the proposed search grammar; it must not be
reinterpreted as a decoder or READ failure.

## Boundaries

This experiment does not establish semantic categories, polysemy resolution,
translation, generation, a universal F, dynamic tree shape, or TreeHeap
superiority over SGNS/Transformer. It tests only whether a globally scored,
probability-conserving F search can improve one fixed corpus background field.

## Result

The preregistered run completed on `io` as task `561` with seed `20260924`.
All eight gates passed:

```text
initial dev NLL                 5.3734355897
best dev NLL                    5.3573660767
dev gain                        0.0160695129

initial sealed-test NLL         5.3963719738
best sealed-test NLL            5.3780405021
random sealed-test NLL          5.4578879603
best - initial test NLL        -0.0183314717
best - random test NLL         -0.0798474582

accepted proposals              175 / 256
leaf utilization                1.0
occupancy entropy               0.9130750036
FOLD conservation max abs       1.3877787808e-17
path residual closure max abs   3.4694469520e-18
```

Status: `S1-F-MC-A11-C01` is supported at smoke-evidence level for this frozen
corpus field, seed, depth, and search budget. The sealed-test result did not
reverse the dev improvement, and the searched tree also beat the random-route
control. This supports continuing the probability-residual F search line.

It does not yet show that the discovered routing is semantically interpretable
or useful to TreeHeap READ/decoder training. Replicated seeds and a controlled
depth ladder are required before promoting the structure into the main model.

# Real-Corpus Annealed Token Space

## Claim

`S1-ANNEAL-SPACE-REAL-C01`

On real WMT text, token-context observations contain enough non-frequency
structure for deterministic TreeHeap FOLD/UNFOLD annealing to produce a stable
hierarchical token coordinate.  The claim is narrower than natural-language
understanding: it requires held-out context prediction and reproducible
neighborhoods, not a human semantic label.

## Corpus Contract

```text
source: WMT massive parallel TSV, English side only
tokenizer: existing 32K shared SentencePiece model
scan: first 200,000 valid lines
test split: deterministic line_index mod 10
target tokens: 512 frequent readable English word-start pieces
context tokens: 1,024 frequent non-special pieces
window: 4 tokens on each side
tree depth: 5 (32 leaves)
formal bootstrap seeds: 8
```

The script records a SHA-256 digest over the exact scanned TSV bytes and the
SentencePiece model.  The test count matrix is never used to form routes.

## Arms

```text
tree_structured:
  deterministic annealed FOLD/UNFOLD over bootstrap train counts

tree_frequency_shuffle:
  permute context rows only among tokens in adjacent frequency bins, preserving
  the global row multiset and approximately matching per-token frequency

random_route:
  random depth-5 routes over the original observations

flat_kmeans:
  deterministic flat K-means with 32 clusters over the same Hellinger features

expected_sgns:
  expected negative-sampling objective with 0.75 noise distribution; token
  coordinates are learned only for this baseline
```

## Metrics

```text
heldout_context_nll
co-cluster stability across bootstrap seeds
leaf utilization and normalized occupancy entropy
FOLD parent/children conservation
top-10 neighbor overlap between TreeHeap routes and SGNS coordinates
frequency R^2 explained by leaf assignment
readable nearest-neighbor examples (qualitative only)
```

The token-specific empirical model and global context unigram are reported as
upper- and lower-information references.

## Pre-registered Decision

The narrow real-corpus claim is supported only if:

```text
tree NLL < global unigram NLL
tree NLL < frequency-shuffled TreeHeap NLL
tree NLL < random-route NLL
tree stability > frequency-shuffled stability
tree stability > random-route stability
TreeHeap/SGNS top-10 overlap exceeds random-route/SGNS overlap by >= 0.02
tree NLL - flat K-means NLL <= 0.15 nat
leaf utilization >= 0.75
occupancy entropy >= 0.75
FOLD conservation max abs <= 1e-6
```

If TreeHeap predicts real held-out contexts but flat K-means is materially
better, record a corpus-structure positive and TreeHeap-shape negative.  If
TreeHeap agrees with SGNS only because of token frequency, downgrade the claim.

## Boundaries

This does not prove bilingual alignment, polysemy resolution, generation,
translation, or replacement of the current model embedding.  A token still
has one corpus-level route in this pilot; context-conditioned routes remain a
future gate.

## Formal Result

Executed on `io.grepcode.cn` as task `449` on 2026-09-16.  The run scanned
200,000 WMT massive English-side lines, containing 4,283,059 SentencePiece
tokens.  It formed 6,599,588 train and 735,369 deterministic held-out context
pairs for 512 target and 1,024 context pieces.  Eight bootstrap seeds completed.

| Model / control | Held-out NLL |
|---|---:|
| TreeHeap depth-5 route | 5.35895 |
| frequency-matched shuffle | 5.82061 |
| random route | 5.46101 |
| flat K-means, 32 clusters | 5.36257 |
| expected SGNS, 32D | 5.20325 |
| token empirical reference | 5.20890 |
| global unigram reference | 5.56978 |

TreeHeap/shuffle/random/flat co-cluster stability was
`0.35096/0.08583/0.03122/0.59186`.  TreeHeap/SGNS top-10 neighbor overlap was
`0.17148`, versus random/SGNS `0.02014` and flat/SGNS `0.12593`.  Mean TreeHeap
leaf utilization and occupancy entropy were `0.91406` and `0.77141`; FOLD
conservation max-abs never exceeded `1.11e-16`.

All aggregate registered gates passed.  However, the result does not establish
TreeHeap superiority: flat K-means is tied on NLL and more stable, and SGNS is
better on NLL.  Seed `19105` partially collapsed (`0.71875` utilization,
`0.50932` occupancy entropy), exposing a real annealing/split robustness issue.

Evidence: `evidence/s1_real_corpus_annealed_token_space/formal_200k_seed19101/`.

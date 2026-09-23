# S1-F-1/F-2 Matched Parent-Read Bridge

Remote task: `io` task `560`  
Seed: `20260923`  
GPU limit observed before submission: `270 W`

## Contract

This is a short WMT English-side next-token probe:

```text
four observed tokens -> fixed-capacity candidate FOLD -> parent state -> token five
```

Candidate arms are `left_deep`, `balanced`, `right_deep`, and `bag_3step`.
Every encoder has exactly 11,360 parameters, including three local F transform
modules. The three tree arms use those modules at binary internal nodes. The
bag control applies the same number of modules after a leaf mean but has no
hierarchical child aggregation.

Each encoder trains with its own pretraining decoder for 1,500 updates and is
then frozen. A newly initialized linear head receives only the frozen parent
vector for 750 updates. A second new linear head receives only the mean of
frozen leaves. Neither fresh head gets a TreeHeap READ route, node address, or
the original leaves through the parent interface.

## Result

`summary.json` is the machine-readable result. The best frozen parent is
`left_deep`: fresh parent NLL `2.999877` versus its mean-leaf control
`3.278185`, a `-0.278307` NLL difference. Balanced also improves over its
mean-leaf control; right-deep does not.

This is one-seed, short-window evidence that FOLD topology changes retained
next-token information and that a parent-only linear readout can use the
left-deep parent representation. It is not evidence of learned routing,
stop/left/right READ, translation, sentence generation, or a universal
language tree.

## Command

```text
python3 ara/s1-echo/src/s1_f_function_parent_read_f21.py --out ara/s1-echo/evidence/s1_f_function_parent_read_f21/formal_seed20260923 --seed 20260923
```

The integration record and next replication gate are in:
`../../../logic/f_function_claim_integration_f21.md`.

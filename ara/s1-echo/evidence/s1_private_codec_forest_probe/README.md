# S1 Private Codec Forest Probe

Claim: `S1-PRIVATE-CODEC-C01`
Predict: `P-S1-PRIVATE-CODEC01`
Host: `io.grepcode.cn`

## Result

pilot_pass: `True`

```text
source_food_ce = 1.386365
heldout_serial_mean_ce = 0.231170
constant_baseline_mean_ce = 0.713971
untrained_composition_mean_ce = 2.013147
heldout_support_correct = True
```

## Meaning

`Theta_food` and `Theta_filter` are separate parameter TreeHeaps. The direct
held-out composition loss is not used during training:

```text
direct_heldout_loss_used = False
```

The test asks whether:

```text
K_filter(K_food(H0; Theta_food); Theta_filter)
```

recovers the fruit bucket from the learned food bucket.

## Boundary

This is a toy learning-mechanism proof. It is not natural-language semantics,
WMT translation, or unsupervised world-model discovery.

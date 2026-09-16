# Fixed-Seed Route Audit

Task 468 is a read-only audit of seed `19301` checkpoints. It does not train or
alter any parameter. For every round and level it records hard-prefix
utilization, hard-prefix entropy, largest hard bucket, mass-weighted soft
branch entropy, decision saturation, and route parameter magnitudes.

The main result is that hard route use contracts while soft branch entropy
stays between `0.9971` and `1.0000` bits. No routed mass has branch probability
below `0.05` or above `0.95`. Hard argmax therefore amplifies tiny deviations
around `0.5`; the checkpoints have not learned confident hierarchical splits.

Raw evidence: `fixed_seed_route_profile.json`.

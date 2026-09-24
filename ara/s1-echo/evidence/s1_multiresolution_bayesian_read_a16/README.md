# A16 evidence index

Claim: `S1-BAYES-READ-A16-C01`

- `smoke/summary.json`: 2,000-example D3 smoke, task 603.
- `formal/summary.json`: original preregistered two-candidate formal run, task
  604, SHA-256
  `4d819214f506e27d193a395a3b6691948a6d034ec01c69b13fa69e2625d75cde`.
- `formal_paired/summary.json`: four-candidate paired depth control, task 605,
  SHA-256
  `6ac2fa0b1e216516a34524cdbd9acd37f2be146d160b75af6d19dd5a000eb6c9`.

The paired artifact repeats the original arms and adds the complementary
`1M/D5` and `200K/D3` checkpoints. All four use the same evaluation tensor hash,
Bayesian equation, shuffle seeds, and mechanical gates.

Every machine-readable command argument, source hash, checkpoint hash, count
artifact hash, metric, structure audit, and control result is embedded in the
corresponding `summary.json`.


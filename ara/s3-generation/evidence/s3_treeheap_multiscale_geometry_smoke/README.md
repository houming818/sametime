# S3 TreeHeap Multiscale Geometry Smoke

Claim: `S3-TREEHEAP-GEOMETRY-C01`

Host: `io` (`NVIDIA GeForce RTX 3090`)

Command:

```bash
cd /home/nio/log/holds/SameTime
python3 ara/s3-generation/src/s3_treeheap_multiscale_geometry.py \
  --output ara/s3-generation/evidence/s3_treeheap_multiscale_geometry_smoke
```

The run trained on 200,000 real Chinese 64-token blocks and evaluated 512
held-out blocks. P1, P5, and P6 passed; P2, P3, and P4 failed. Bag geometry was
readable and gave positive same-depth retrieval gains, while directed
adjacency remained near zero. The full claim is not supported by this probe.

Machine-readable metrics and the preregistered gate decision are in
`summary.json`. Training checkpoints are evidence artifacts, not endorsed
language-model checkpoints.

# D10 evidence index

Claim: `S3-STRUCTURAL-PROTOCOL-FULL-PIPELINE-D10`

- `smoke_seed11001/`: taskd #340 and reload/contract audit #341. Large checkpoint files stay on io.
- `full_seed11001/pretrain_stop_step160k/`: taskd #342 plateau-stop evidence; Stage B was blocked.
- `full_seed11001/`: taskd #343 continues from the latest Stage A checkpoint to observe the complete curve.
- `full_seed11001/task.log`: taskd #343 Stage B crash log.  Training reached
  step 299000 before a PyTorch `CUDACachingAllocator` internal assertion; the
  latest durable checkpoint and wake are at step 275000.
- `full_seed11001/task/`: last durable wake and trace copied from `io`.  Large
  checkpoint files remain on `io`.
- taskd #339 is the preserved pre-training interface failure described in the ARA contract.

The formal result is not claimed.  Stage B did not finish and proof/reload
gates did not run.  Automatic continuation is blocked until the allocator
failure is audited.

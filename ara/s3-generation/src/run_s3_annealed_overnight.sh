#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/nio/log/holds/SameTime

toy=ara/s3-generation/evidence/s3_annealed_contraction_toy
real=ara/s3-generation/evidence/s3_annealed_frontier_pretrain
nas=/mnt/nas/ara/s3-generation/evidence
master=ara/s3-generation/evidence/s3_annealed_overnight.log

rm -rf "$toy" "$real"
mkdir -p "$toy" "$real"
: > "$master"

notify() {
  status=$?
  if [[ $status -eq 0 ]]; then
    sendme -s "TreeHeap annealed contraction complete" \
      "Toy and real-corpus annealed contraction proofs completed on io. Evidence: $toy and $real"
  else
    tail -n 120 "$master" | sendme -s "TreeHeap annealed contraction failed (exit $status)"
  fi
}
trap notify EXIT

printf '%s\n' \
  'python3 ara/s3-generation/src/s3_annealed_contraction_toy.py --evidence-dir ara/s3-generation/evidence/s3_annealed_contraction_toy --seeds 72001 72002 72003 --train-samples 24000 --test-samples 4000 --dim 64 --batch 256 --steps 3000 --log-every 500' \
  > "$toy/command.txt"

python3 ara/s3-generation/src/s3_annealed_contraction_toy.py \
  --evidence-dir "$toy" \
  --seeds 72001 72002 72003 \
  --train-samples 24000 --test-samples 4000 \
  --dim 64 --batch 256 --steps 3000 --log-every 500 \
  2>&1 | tee "$toy/stdout.log" | tee -a "$master"

printf '%s\n' \
  'python3 ara/s3-generation/src/s3_annealed_frontier_pretrain.py --evidence-dir ara/s3-generation/evidence/s3_annealed_frontier_pretrain --dim 256 --hidden 256 --batch 64 --steps 20000 --valid-every 2000 --eval-batches 32 --num-workers 0' \
  > "$real/command.txt"

python3 ara/s3-generation/src/s3_annealed_frontier_pretrain.py \
  --evidence-dir "$real" \
  --dim 256 --hidden 256 --batch 64 \
  --steps 20000 --valid-every 2000 --eval-batches 32 \
  --num-workers 0 \
  2>&1 | tee "$real/stdout.log" | tee -a "$master"

sudo mkdir -p "$nas/s3_annealed_contraction_toy" "$nas/s3_annealed_frontier_pretrain"
for checkpoint in "$toy"/checkpoint_*.pt; do
  sudo mv "$checkpoint" "$nas/s3_annealed_contraction_toy/"
done
for checkpoint in "$real"/checkpoint_*.pt; do
  sudo mv "$checkpoint" "$nas/s3_annealed_frontier_pretrain/"
done
sudo chown -R nio:nio "$nas/s3_annealed_contraction_toy" "$nas/s3_annealed_frontier_pretrain" || true
sha256sum "$nas/s3_annealed_contraction_toy"/checkpoint_*.pt > "$toy/checkpoints.sha256"
sha256sum "$nas/s3_annealed_frontier_pretrain"/checkpoint_*.pt > "$real/checkpoints.sha256"
printf '\nCheckpoints: %s\n' "$nas/s3_annealed_contraction_toy" >> "$toy/README.md"
printf '\nCheckpoints: %s\n' "$nas/s3_annealed_frontier_pretrain" >> "$real/README.md"

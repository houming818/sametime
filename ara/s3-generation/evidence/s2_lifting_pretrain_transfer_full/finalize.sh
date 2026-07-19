#!/usr/bin/env bash
set -u

pid="${1:?main training pid required}"
evidence="/home/nio/log/holds/SameTime/ara/s3-generation/evidence/s2_lifting_pretrain_transfer_full"
nas="/mnt/nas/ara/s3-generation/evidence/s2_lifting_pretrain_transfer_full"

while kill -0 "$pid" 2>/dev/null; do
  sleep 60
done

sudo mkdir -p "$nas"
sudo rsync -a "$evidence/" "$nas/"

if command -v sendme >/dev/null 2>&1; then
  if [[ -f "$evidence/summary.json" ]]; then
    sendme -s "TreeHeap pretrain transfer completed" \
      "Formal pretrain and WMT transfer run completed on io. Evidence: $evidence"
  else
    sendme -s "TreeHeap pretrain transfer failed" \
      "Formal run exited without summary.json. Inspect $evidence/stderr.log"
  fi
fi

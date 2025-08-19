#!/usr/bin/env bash

TRAINABLE=(
  "airplane"
  "bed"
  "bottle"
  "cap"
  "car"
  "chair"
  "guitar"
  "helmet"
  "knife"
  "motorbike"
  "mug"
  "table"
  "vessel"
)
sleep 1m


SCRIPT="${1:-train_kpd.py}"
shift

for category in "${TRAINABLE[@]}"; do
    echo "Training category: $category"
    python3 $SCRIPT --category="$category" --db=/mnt/slow/results.db --ckpt-dir=/mnt/slow2/skeleton_merger
done

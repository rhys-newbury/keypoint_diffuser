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

for category in "${TRAINABLE[@]}"; do
    echo "Training category: $category"
    python3 train_kpd.py --category="$category" --db=/mnt/slow/results.db --ckpt-dir=/mnt/slow2/skeleton_merger
done

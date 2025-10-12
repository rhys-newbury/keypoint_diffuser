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
    config_path="configs/partial-training/${category}-10kpt-partial.yaml"
    echo "Training category: ${category}"
    if ! python scripts/train/train_keypoint_diffuser_partial.py \
        -c "${config_path}"
    then
      echo "WARN: get_das run failed for category '${category}'" >&2
      # continue to the next class even if this one fails
      continue
done

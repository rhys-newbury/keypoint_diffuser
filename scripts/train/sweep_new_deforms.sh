#!/usr/bin/env bash

TRAINABLE=(
  "airplane"
  # "bed"
  # "bottle"
  # "cap"
  # "car"
  # "chair"
  # "guitar"
  # "helmet"
  # "knife"
  # "motorbike"
  # "mug"
  # "table"
  # "vessel"
)

SCALE=(1.3 2.0 3.0)
TRANSLATE=(0.3 0.8 2.0)
COMBOS=(
  "0 0"
  "0 1"
  "0 2"
  "1 0"
  "1 1"
  "1 2"
  "2 0"
  "2 1"
  "2 2"
)

i=1
for combo in "${COMBOS[@]}"; do
    read si ti <<< "$combo"
    SC=${SCALE[$si]}
    TR=${TRANSLATE[$ti]}

    echo "=== Run $i ==="
    echo "Scale: $SC, Translate: $TR"
    config_path="configs/airplane-10kpt-sweep.yaml"
    echo "Training category: airplane"
    if ! python scripts/train/train_keypoint_diffuser.py \
        -c "${config_path}" \
        --max_scaling_factor "$SC" \
        --max_translation_offset "$TR" \
        
    then
      echo "WARN: training run failed for category '${category}'" >&2
      # continue to the next class even if this one fails
      continue
    fi

    i=$((i+1))
done

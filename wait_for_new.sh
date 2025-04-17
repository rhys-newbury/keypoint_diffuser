#!/bin/bash

SCRIPT=$1
FOLDER=$2
LATENT_DIM=$3
shift 3  # "$@" now contains only the extra args

CHECKPOINT_DIR="/mnt/slow/keypoints/logs/$FOLDER/checkpoints"

# Initialize seen files with all current .pth files
SEEN_FILES=($(ls "$CHECKPOINT_DIR"/*.pth 2>/dev/null))

echo "Initialized with ${#SEEN_FILES[@]} existing checkpoints."
echo "Watching for new checkpoints in $CHECKPOINT_DIR"

while true; do
  sleep 10  # Polling interval

  # Get the current list of checkpoint files
  files=($(ls -t "$CHECKPOINT_DIR"/*.pth 2>/dev/null))

  for file in "${files[@]}"; do
    if [[ ! " ${SEEN_FILES[*]} " =~ " $file " ]]; then
      echo "New checkpoint detected: $file"
      SEEN_FILES+=("$file")

      if [[ "$SCRIPT" == "main.py" ]]; then
        python scripts/"$SCRIPT" --n_keypoints="$LATENT_DIM" --ckpt="$file" -c configs/airplane-8kpt.yaml -t configs/test.yaml "$@"
      elif [[ "$SCRIPT" == "train_ae.py" ]]; then
        python scripts/"$SCRIPT" --latent_dim="$LATENT_DIM" --ckpt="$file" -c configs/airplane-8kpt.yaml -t configs/test.yaml "$@"
      else
        echo "Unknown script: $SCRIPT"
        exit 1
      fi
    fi
  done
done

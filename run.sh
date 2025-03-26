#!/bin/bash

SCRIPT=$1
FOLDER=$2
LATENT_DIM=$3

for file in /mnt/storage/keypoints/logs/"$FOLDER"/checkpoints/*.pth; do
  echo "Processing $file"

  if [[ "$SCRIPT" == "main.py" ]]; then
    python scripts/"$SCRIPT" --n_keypoints="$LATENT_DIM" --ckpt="$file" -c configs/airplane-8kpt.yaml -t configs/test.yaml
  elif [[ "$SCRIPT" == "train_ae.py" ]]; then
    python scripts/"$SCRIPT" --latent_dim="$LATENT_DIM" --ckpt="$file" -c configs/airplane-8kpt.yaml -t configs/test.yaml
  else
    echo "Unknown script: $SCRIPT"
    exit 1
  fi
done

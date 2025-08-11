#!/bin/bash

CHECKPOINT_DIR="."
OUTPUT_DIR="eval_logs"

mkdir -p "$OUTPUT_DIR"

for CKPT in $(ls -t "$CHECKPOINT_DIR"/txt_cap*.pth); do
    CKPT_NAME=$(basename "$CKPT")
    CKPT_BASENAME="${CKPT_NAME%.*}"  # Remove .pth extension

    echo "Running evaluation for $CKPT_NAME..."
    
    python eval.py \
        --checkpoint-path "$CKPT" \
        > "$OUTPUT_DIR/${CKPT_BASENAME}.txt"

    echo "Saved results to $OUTPUT_DIR/${CKPT_BASENAME}.txt"
done

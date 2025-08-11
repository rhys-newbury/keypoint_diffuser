CHECKPOINT_DIR="/app/data2/keypoints/logs/devout-sunset-241/checkpoints"

OUTPUT_DIR="eval_logs"

mkdir -p "$OUTPUT_DIR"

for ckpt in $(ls -t "$CHECKPOINT_DIR"/*.pth); do

  CKPT_NAME=$(basename "$ckpt")
  CKPT_BASENAME="${CKPT_NAME%.*}"  # Remove .pth extension


  echo "Running with checkpoint: $ckpt"
  python3 /app/scripts/test_das.py \
    --latent_dim=10 \
    --n_iterations=20000 \
    -c=../configs/object.yaml \
    -t=../configs/test.yaml \
    --category=03001627 \
    --extra_latent=5 \
    --use_edm=True \
    --ckpt="$ckpt" \
     > "$OUTPUT_DIR/das_${CKPT_BASENAME}.txt" 2>&1


  python3 /app/scripts/sample.py \
    --latent_dim=10 \
    --n_iterations=20000 \
    -c=../configs/object.yaml \
    -t=../configs/test.yaml \
    --category=03001627 \
    --extra_latent=5 \
    --use_edm=True \
    --ckpt="$ckpt" \
     > "$OUTPUT_DIR/${CKPT_BASENAME}.txt" 2>&1
done

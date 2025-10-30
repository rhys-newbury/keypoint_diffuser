#!/usr/bin/env bash
set -Euo pipefail

classes=(
  airplane
  bed
  bottle
  cap
  car
  chair
  guitar
  helmet
  knife
  motorbike
  mug
  table
  vessel
)

for cls in "${classes[@]}"; do
  db_path="db/train/original-300/${cls}-10kpt-300-train-results.db"
  eval_db_path="db/test_db/original-300/${cls}-300-das-results.db"
  keypoints_out_dir="kps_out/original-300/"
  echo ">>> Running for category: ${cls}"
  if ! python scripts/test/test_db.py \
      --db "${db_path}" \
      --script scripts/test/get_das.py \
      --only-category "${cls}" \
      --keep-going \
      --eval-db "${eval_db_path}" \
      --output-dir "${keypoints_out_dir}" \
      --epoch-interval 100
  then
    echo "WARN: get_das run failed for category '${cls}'" >&2
    # continue to the next class even if this one fails
    continue
  fi
done

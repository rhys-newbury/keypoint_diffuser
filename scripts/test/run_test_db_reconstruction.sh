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
  db_path="db/train/${cls}-8kpt-train-results.db"
  eval_db_path="db/test_db/${cls}-recon-results.db"
  echo ">>> Running for category: ${cls}"
  if ! python scripts/test/test_db.py \
      --db "${db_path}" \
      --script scripts/test/get_reconstruction.py \
      --only-category "${cls}" \
      --keep-going \
      --eval-db "${eval_db_path}"
  then
    echo "WARN: get_reconstruction run failed for category '${cls}'" >&2
    # continue to the next class even if this one fails
    continue
  fi
done

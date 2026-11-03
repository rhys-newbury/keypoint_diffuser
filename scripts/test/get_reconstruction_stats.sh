#!/usr/bin/env bash
# Runs get_reconstruction.py multiple times with the same arguments and reports
# the mean/std of CD, EMD, KVC, and the slot-swap gap across runs. Useful since
# reconstruction involves stochastic diffusion sampling and (for partial input
# types) random view selection, so a single run is noisy.
set -Euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") [--runs N] -- <args passed through to get_reconstruction.py>

  --runs N   Number of times to run get_reconstruction.py (default: 5)

Example:
  $(basename "$0") --runs 5 -- Ours --ckpt path/to/ckpt.pth --category chair \\
      --db-path results.db --output-dir recons_out --input-type full --no-save
EOF
}

RUNS=5
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs)
      RUNS="$2"
      shift 2
      ;;
    --runs=*)
      RUNS="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      ARGS=("$@")
      break
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#ARGS[@]} -eq 0 ]]; then
  echo "ERROR: no arguments to pass through to get_reconstruction.py" >&2
  usage
  exit 1
fi

if ! [[ "$RUNS" =~ ^[0-9]+$ ]] || [[ "$RUNS" -lt 1 ]]; then
  echo "ERROR: --runs must be a positive integer (got '${RUNS}')" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GET_RECON="${SCRIPT_DIR}/get_reconstruction.py"

RESULTS_FILE="$(mktemp)"
LOG_FILE="$(mktemp)"
trap 'rm -f "$RESULTS_FILE" "$LOG_FILE"' EXIT

for i in $(seq 1 "$RUNS"); do
  echo ""
  echo ">>> Run ${i}/${RUNS}: python ${GET_RECON} ${ARGS[*]}"

  if ! python "$GET_RECON" "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"; then
    echo "WARN: run ${i} failed; skipping it" >&2
    continue
  fi

  cd_val=$(grep -oP 'Chamfer Distance \(mean\):\s*\K[-0-9.eE]+' "$LOG_FILE" | tail -1)
  emd_val=$(grep -oP 'EMD \(Sinkhorn\)\s*\(mean\):\s*\K[-0-9.eE]+' "$LOG_FILE" | tail -1)
  kvc_val=$(grep -oP 'KVC \(ordered, mean\):\s*\K[-0-9.eE]+' "$LOG_FILE" | tail -1)
  swap_val=$(grep -oP 'Slot-swap gap:\s*\K[-0-9.eE]+' "$LOG_FILE" | tail -1)

  if [[ -z "$cd_val" || -z "$emd_val" || -z "$kvc_val" || -z "$swap_val" ]]; then
    echo "WARN: could not parse metrics from run ${i}'s output; skipping it" >&2
    continue
  fi

  echo "${cd_val},${emd_val},${kvc_val},${swap_val}" >> "$RESULTS_FILE"
done

echo ""
python3 - "$RESULTS_FILE" <<'PYEOF'
import statistics
import sys

path = sys.argv[1]
rows = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rows.append([float(x) for x in line.split(",")])

if not rows:
    print("No successful runs to summarize.", file=sys.stderr)
    sys.exit(1)

names = ["CD", "EMD", "KVC", "Slot-swap gap"]
cols = list(zip(*rows))

print(f"=== Summary over {len(rows)} successful run(s) ===")
for name, col in zip(names, cols):
    mean = statistics.mean(col)
    std = statistics.stdev(col) if len(col) > 1 else 0.0
    print(f"{name:15s}: mean= {mean:.8f}  std= {std:.8f}")
PYEOF

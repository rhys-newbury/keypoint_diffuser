#!/usr/bin/env bash
# Runs get_mmd.py multiple times with the same arguments and reports the
# mean/std of MMD-CD, COV-CD (and, when the corresponding flags are passed
# through, MMD-EMD/COV-EMD and 1-NNA-CD/1-NNA-EMD) across runs. Useful since
# the density fit + sampling + decode pipeline is stochastic, so a single run
# is noisy.
set -Euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") [--runs N] -- <args passed through to get_mmd.py>

  --runs N   Number of times to run get_mmd.py (default: 5)

Pass --emd and/or --nna through to get_mmd.py (after the --) to also collect
stats for MMD-EMD/COV-EMD and 1-NNA-CD/1-NNA-EMD.

Example:
  $(basename "$0") --runs 5 -- Ours --ckpt path/to/ckpt.pth --category chair \\
      --fit-split train --ref-split val --num-gen 256 --db-path db/mmd.db --no-save
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
  echo "ERROR: no arguments to pass through to get_mmd.py" >&2
  usage
  exit 1
fi

if ! [[ "$RUNS" =~ ^[0-9]+$ ]] || [[ "$RUNS" -lt 1 ]]; then
  echo "ERROR: --runs must be a positive integer (got '${RUNS}')" >&2
  exit 1
fi

# get_mmd.py only prints MMD-EMD/COV-EMD when --emd was passed, and only
# prints 1-NNA-* when --nna was passed (1-NNA-EMD further requires --emd) -
# mirror that here so we only look for metrics that will actually be printed.
HAS_EMD=false
HAS_NNA=false
for a in "${ARGS[@]}"; do
  [[ "$a" == "--emd" ]] && HAS_EMD=true
  [[ "$a" == "--nna" ]] && HAS_NNA=true
done

METRIC_NAMES=(MMD-CD COV-CD)
METRIC_PATTERNS=('MMD-CD\s*:\s*\K[-0-9.eE+]+' 'COV-CD\s*:\s*\K[-0-9.eE+]+')
if $HAS_EMD; then
  METRIC_NAMES+=(MMD-EMD COV-EMD)
  METRIC_PATTERNS+=('MMD-EMD\s*:\s*\K[-0-9.eE+]+' 'COV-EMD\s*:\s*\K[-0-9.eE+]+')
fi
if $HAS_NNA; then
  METRIC_NAMES+=(1-NNA-CD)
  METRIC_PATTERNS+=('1-NNA-CD\s*:\s*\K[-0-9.eE+]+')
  if $HAS_EMD; then
    METRIC_NAMES+=(1-NNA-EMD)
    METRIC_PATTERNS+=('1-NNA-EMD\s*:\s*\K[-0-9.eE+]+')
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# GET_MMD_SCRIPT swaps in a drop-in variant with the same output format (e.g. get_mmd_cross.py)
GET_MMD="${GET_MMD_SCRIPT:-${SCRIPT_DIR}/get_mmd.py}"

RESULTS_FILE="$(mktemp)"
LOG_FILE="$(mktemp)"
trap 'rm -f "$RESULTS_FILE" "$LOG_FILE"' EXIT

for i in $(seq 1 "$RUNS"); do
  echo ""
  echo ">>> Run ${i}/${RUNS}: python ${GET_MMD} ${ARGS[*]}"

  if ! python "$GET_MMD" "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"; then
    echo "WARN: run ${i} failed; skipping it" >&2
    continue
  fi

  vals=()
  ok=true
  for pattern in "${METRIC_PATTERNS[@]}"; do
    v=$(grep -oP "$pattern" "$LOG_FILE" | tail -1)
    if [[ -z "$v" ]]; then
      ok=false
      break
    fi
    vals+=("$v")
  done

  if ! $ok; then
    echo "WARN: could not parse metrics from run ${i}'s output; skipping it" >&2
    continue
  fi

  (IFS=,; echo "${vals[*]}") >> "$RESULTS_FILE"
done

echo ""
python3 - "$RESULTS_FILE" "${METRIC_NAMES[@]}" <<'PYEOF'
import statistics
import sys

path = sys.argv[1]
names = sys.argv[2:]

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

cols = list(zip(*rows))

print(f"=== Summary over {len(rows)} successful run(s) ===")
for name, col in zip(names, cols):
    mean = statistics.mean(col)
    std = statistics.stdev(col) if len(col) > 1 else 0.0
    print(f"{name:15s}: mean= {mean:.8f}  std= {std:.8f}")
PYEOF

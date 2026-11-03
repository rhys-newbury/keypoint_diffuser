#!/usr/bin/env bash
# Runs get_reconstruction_stats.sh (mean/std over several repeats of
# get_reconstruction.py) for a manually-listed set of trained checkpoint
# folders, once per --input-type in INPUT_TYPES, and writes one CSV row per
# (run, input type) combination.
#
# Edit the "EDIT THESE" block below, then just run this script with no args:
#   bash scripts/test/reconstruction_checkpoint_sweep.sh
#
# Each entry in RUN_NAMES is "run_folder:category[:key_points]", e.g.
# "kind-lake-65:airplane". run_folder is a checkpoint folder name under
# CKPT_ROOT (typically a wandb run name, which carries no category or
# kpt-count info of its own, hence the explicit ":category" field).
# key_points is optional - it's normally parsed straight off the checkpoint
# filenames (see below), so only add it if a run has nothing but
# net_final.pth and no numbered *.pth to infer it from.
#
# Each folder is expected to contain its checkpoints directly, or under a
# "checkpoints/" subfolder (both layouts are checked, matching db_utils.py's
# find_checkpoint_from_db_dir convention). Checkpoint filenames follow the
# "<K>kp_<epoch>.pth" convention (e.g. "10kp_500004.pth"), plus optionally a
# "net_final.pth" with no K or epoch in its name.
#
# "net_final.pth" is preferred for --ckpt when present; otherwise the *.pth
# with the largest epoch is used. The "iters" column is that epoch number,
# or (when net_final.pth was used) the largest epoch found among its sibling
# numbered checkpoints, or "final" if none exist.
set -Euo pipefail

# ==================== EDIT THESE ====================

# Directory that all the run folders below share.
# CKPT_ROOT="./logs/original-200-domain_randomization"
CKPT_ROOT="./logs/partial-domain_randomization"

# "run_folder:category[:key_points]" entries under CKPT_ROOT.
RUN_NAMES=(
  # "kind-lake-65:airplane"
  # "charmed-salad-13:bed"
  # "comfy-night-12:bottle"
  # "whole-valley-13:cap"
  # "worldly-voice-13:car"
  # "clean-flower-13:chair"
  # "comic-grass-11:guitar"
  # "charmed-pyramid-9:helmet"
  # "legendary-grass-13:knife"
  # "firm-plasma-10:motorbike"
  # "jumping-donkey-44:mug"
  # "leafy-blaze-14:table"
  # "electric-terrain-13:vessel"
# )
# partial
# RUN_NAMES=(
  # "northern-glitter-66:airplane"
  # "solar-pyramid-14:bed"
  # "swift-voice-6:bottle"
  # "curious-microwave-19:cap"
  # "snowy-cloud-13:car"
  # "eager-waterfall-11:chair"
  # "expert-bee-8:guitar"
  "exalted-wildflower-7:helmet"
  # "valiant-voice-8:knife"
  # "ruby-shadow-8:motorbike"
  # "helpful-morning-74:mug"      # this was best but wasn't enumerating
  # "dazzling-brook-9:table"
  # "young-wildflower-7:vessel"
)

# get_reconstruction.py --input-type values to test each run against.
# "full" -> test obj "full"; anything else -> test obj "partial".
INPUT_TYPES=(full uniform)

# Repeats passed to get_reconstruction_stats.sh's --runs (mean/std sample size).
REPEATS=5

# Model name (first positional arg to get_reconstruction.py).
MODEL="Ours"

# Where per-combination reconstruction outputs (coverage plots, etc.) go.
OUT_ROOT="recons_out/checkpoint_sweep"

# Final results table.
CSV_OUT="recon_checkpoint_sweep.csv"

# Any extra args forwarded to get_reconstruction.py for every run, e.g.:
#   EXTRA_ARGS=(--batch-size 8 --no-coverage-plot)
EXTRA_ARGS=(--no-save --split val)

# =====================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATS_SCRIPT="${SCRIPT_DIR}/get_reconstruction_stats.sh"

# "train obj" column: full-cloud-only training lives under logs/original-*,
# partial-view training under logs/partial-* (see CLAUDE.md's Setup & commands
# section - configs/original-training* vs configs/partial-training*).
ckpt_root_base="$(basename "$CKPT_ROOT")"
case "$ckpt_root_base" in
  original*) train_obj="full" ;;
  partial*) train_obj="partial" ;;
  *)
    echo "WARN: could not tell train obj from CKPT_ROOT basename '${ckpt_root_base}' (expected it to start with 'original' or 'partial'); recording 'unknown'" >&2
    train_obj="unknown"
    ;;
esac

resolve_checkpoint() {
  # Prints "ckpt_path<TAB>key_points<TAB>iters" on success (key_points is
  # empty if it couldn't be inferred from any checkpoint filename), or writes
  # an error to stderr and returns non-zero on failure.
  local run_dir="$1"
  python3 - "$run_dir" <<'PYEOF'
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])

search_dirs = [run_dir / "checkpoints", run_dir]
pth_files = []
for d in search_dirs:
    if d.is_dir():
        pth_files = list(d.glob("*.pth"))
        if pth_files:
            break

if not pth_files:
    print(f"no .pth files found under {run_dir}/checkpoints or {run_dir}", file=sys.stderr)
    sys.exit(1)


def kp_epoch_of(p):
    m = re.match(r"(\d+)kp_(\d+)\.pth$", p.name)
    return (int(m.group(1)), int(m.group(2))) if m else None


kp_epoch_pairs = [pr for pr in (kp_epoch_of(p) for p in pth_files) if pr is not None]
key_points_seen = {k for k, _ in kp_epoch_pairs}
max_epoch = max((e for _, e in kp_epoch_pairs), default=None)

if len(key_points_seen) > 1:
    print(f"multiple key-point counts found under {run_dir}: {sorted(key_points_seen)}; "
          f"add an explicit ':key_points' override for this run", file=sys.stderr)
    sys.exit(1)
key_points = str(next(iter(key_points_seen))) if key_points_seen else ""

final = next((p for p in pth_files if p.name == "net_final.pth"), None)
if final is not None:
    ckpt = final
    iters = str(max_epoch) if max_epoch is not None else "final"
elif max_epoch is not None:
    ckpt = next(p for p, pr in zip(pth_files, (kp_epoch_of(p) for p in pth_files)) if pr and pr[1] == max_epoch)
    iters = str(max_epoch)
else:
    print(f"no net_final.pth and no '<K>kp_<epoch>.pth' files under {run_dir}", file=sys.stderr)
    sys.exit(1)

print(f"{ckpt}\t{key_points}\t{iters}")
PYEOF
}

echo "run name,object class,train obj,test obj,iters,chamfer distance,cd (std),emd,emd (std),KVC,KVC (std),swap gap,gap (std)" > "$CSV_OUT"

for entry in "${RUN_NAMES[@]}"; do
  IFS=':' read -r run_name category kp_override <<< "$entry"
  if [[ -z "$run_name" || -z "$category" ]]; then
    echo "WARN: skipping malformed RUN_NAMES entry '${entry}' (expected 'run_folder:category[:key_points]')" >&2
    continue
  fi
  run_dir="${CKPT_ROOT}/${run_name}"

  resolved="$(resolve_checkpoint "$run_dir")" || {
    echo "WARN: skipping '${run_name}': checkpoint resolution failed" >&2
    continue
  }
  IFS=$'\t' read -r ckpt key_points_parsed iters <<< "$resolved"

  key_points="${kp_override:-$key_points_parsed}"
  if [[ -z "$key_points" ]]; then
    echo "WARN: skipping '${run_name}': could not infer key_points from checkpoint filenames; add ':key_points' to its RUN_NAMES entry" >&2
    continue
  fi

  for input_type in "${INPUT_TYPES[@]}"; do
    if [[ "$input_type" == "full" ]]; then
      test_obj="full"
    else
      test_obj="partial"
    fi

    out_dir="${OUT_ROOT}/${run_name}/${input_type}"
    echo ""
    echo ">>> ${run_name} (${category}, ${key_points}kpt, iters=${iters}) / input-type=${input_type}"

    if ! summary="$(bash "$STATS_SCRIPT" --runs "$REPEATS" -- "$MODEL" \
        --ckpt "$ckpt" \
        --category "$category" \
        --key-points "$key_points" \
        --output-dir "$out_dir" \
        --input-type "$input_type" \
        "${EXTRA_ARGS[@]}" 2>&1 | tee /dev/stderr)"; then
      echo "WARN: run failed for '${run_name}' / input-type '${input_type}'; skipping" >&2
      continue
    fi

    cd_mean=$(grep -oP '^CD\s*:\s*mean=\s*\K[-0-9.eE+]+' <<< "$summary" | tail -1)
    cd_std=$(grep -oP '^CD\s*:\s*mean=\s*[-0-9.eE+]+\s*std=\s*\K[-0-9.eE+]+' <<< "$summary" | tail -1)
    emd_mean=$(grep -oP '^EMD\s*:\s*mean=\s*\K[-0-9.eE+]+' <<< "$summary" | tail -1)
    emd_std=$(grep -oP '^EMD\s*:\s*mean=\s*[-0-9.eE+]+\s*std=\s*\K[-0-9.eE+]+' <<< "$summary" | tail -1)
    kvc_mean=$(grep -oP '^KVC\s*:\s*mean=\s*\K[-0-9.eE+]+' <<< "$summary" | tail -1)
    kvc_std=$(grep -oP '^KVC\s*:\s*mean=\s*[-0-9.eE+]+\s*std=\s*\K[-0-9.eE+]+' <<< "$summary" | tail -1)
    swap_mean=$(grep -oP '^Slot-swap gap\s*:\s*mean=\s*\K[-0-9.eE+]+' <<< "$summary" | tail -1)
    swap_std=$(grep -oP '^Slot-swap gap\s*:\s*mean=\s*[-0-9.eE+]+\s*std=\s*\K[-0-9.eE+]+' <<< "$summary" | tail -1)

    if [[ -z "$cd_mean" || -z "$cd_std" || -z "$emd_mean" || -z "$emd_std" \
          || -z "$kvc_mean" || -z "$kvc_std" || -z "$swap_mean" || -z "$swap_std" ]]; then
      echo "WARN: could not parse summary for '${run_name}' / input-type '${input_type}'; skipping" >&2
      continue
    fi

    echo "${run_name},${category},${train_obj},${test_obj},${iters},${cd_mean},${cd_std},${emd_mean},${emd_std},${kvc_mean},${kvc_std},${swap_mean},${swap_std}" >> "$CSV_OUT"
  done
done

echo ""
echo "[✓] Wrote ${CSV_OUT}"

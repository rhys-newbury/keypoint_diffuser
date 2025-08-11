#!/bin/bash
# sweep_grid.sh
SEED=42
PHASE="train"
CATEGORY="lamp"
BATCH_SIZE=16
N_ITERATIONS=100000
PROJECT="diffuse_keypoints_sweep"

# Define discrete values
STRETCH=(1.0 1.15 1.3)
BEND=(0.0 0.25 0.5)
TWIST=(0.0 0.2 0.4)
TAPER=(0.0 0.15 0.3)
ROT=(0.0 0.1309 0.2618)  # radians: 0, pi/24, pi/12

# Pre-selected 10 combinations (low/mid/high coverage)
COMBOS=(
    "0 0 0 0 0"
    "2 0 0 0 0"
    "1 1 0 0 0"
    "1 0 1 0 0"
    "1 0 0 1 0"
    "1 0 0 0 1"
    "2 2 1 1 0"
    "2 1 2 0 1"
    "0 2 2 2 0"
    "2 2 2 2 2"
)

i=1
for combo in "${COMBOS[@]}"; do
    read si bi ti tai ri <<< "$combo"
    ST=${STRETCH[$si]}
    BE=${BEND[$bi]}
    TW=${TWIST[$ti]}
    TA=${TAPER[$tai]}
    RO=${ROT[$ri]}

    echo "=== Run $i ==="
    echo "Stretch: $ST, Bend: $BE, Twist: $TW, Taper: $TA, Rot: $RO"

    python -u train_ae.py \
        --latent_dim=10 \
        --n_iterations=20000 \
        -c=../configs/object.yaml \
        --category=airplane \
        --extra_latent=5 \
        --use_edm=True \
        --max_stretch_factor "$ST" \
        --max_bending_factor "$BE" \
        --max_twist_factor "$TW" \
        --max_taper_factor "$TA" \
        --max_rotation_angle "$RO" \

    i=$((i+1))
done

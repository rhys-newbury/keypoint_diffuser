#!/bin/bash

SCRIPT=$1
FOLDER=$2
LATENT_DIM=$3
CONFIG=$4
shift 4  # shift the first 3 args so "$@" contains only the extra args

if [ -d "/mnt/storage/keypoints/logs/$FOLDER/checkpoints" ]; then
    BASE_PATH="/mnt/storage/keypoints/logs/$FOLDER/checkpoints"
elif [ -d "/mnt/slow/keypoints/logs/$FOLDER/checkpoints" ]; then
    BASE_PATH="/mnt/slow/keypoints/logs/$FOLDER/checkpoints"
else
    echo "Checkpoint directory not found in /mnt/storage or /mnt/slow for folder: $FOLDER"
    exit 1
fi


# Get sorted list of net_*.pth files by modification time (most recent first)
for file in $(ls -t "$BASE_PATH"/net_*.pth 2>/dev/null); do
    filename=$(basename "$file")

    # Extract the numeric part after the last underscore and before .pth
    n=$(echo "$filename" | awk -F'[_\.]' '{print $(NF-1)}')

    if [[ "$n" != "final" ]]; then
        if [[ "$n" =~ ^[0-9]+$ ]]; then
            if (( n <= 20000 )); then
                echo "Processing $file"

                while true; do
                    if [[ "$SCRIPT" == "main.py" ]]; then
                        python -u "$SCRIPT" --n_keypoints="$LATENT_DIM" --ckpt="$file" -c "$CONFIG" -t configs/test.yaml "$@"
                    elif [[ "$SCRIPT" == "train_ae.py" ]]; then
                        python -u "$SCRIPT" --latent_dim="$LATENT_DIM" --ckpt="$file" -c "$CONFIG" -t configs/test.yaml "$@"
                    elif [[ "$SCRIPT" == "train_dpm.py" ]]; then
                        python -u "$SCRIPT" --latent_dim="$LATENT_DIM" --ckpt="$file" -c "$CONFIG" -t configs/test.yaml "$@"
                    else
                        echo "Unknown script: $SCRIPT"
                        exit 1
                    fi

                    if [[ $? -eq 0 ]]; then
                        echo "Successfully processed $file"
                        break
                    else
                        echo "Python script failed, retrying for $file..."
                        sleep 1  # optional: short delay before retry
                    fi
                done

            fi
        fi
    fi
done

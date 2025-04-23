import matplotlib.pyplot as plt
import pandas as pd
import tqdm

import wandb


# Initialize wandb API
api = wandb.Api()

# Replace with your project path
runs = api.runs("rhys-newbury/diffuse_keypoints_test")

data = []

# Collect data from runs
for run in tqdm.tqdm(runs, total=1176):
    # Get config values
    try:
        x = run.history()
        n_keypoints = int(x["n_keypoints"][0])
        run_type = x["type"][0]

        # Skip if necessary config values are missing
        # if n_keypoints is None or run_type is None:

        # Get metric history

        # if "average_correlation_per_keypoint" in history.columns:
        max_corr = x["average_correlation_per_keypoint"][1]
        data.append(
            {
                "n_keypoints": n_keypoints,
                "type": run_type,
                "average_correlation_per_keypoint": max_corr,
            }
        )
    except Exception as e:
        print(e)

# Convert to DataFrame
df = pd.DataFrame(data)

# Group and aggregate by max
grouped = df.groupby(["n_keypoints", "type"]).max().reset_index()

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
for key, grp in grouped.groupby("type"):
    ax.bar(
        grp["n_keypoints"].astype(str) + ", type: " + key,
        grp["average_correlation_per_keypoint"],
        label=key,
    )

ax.set_ylabel("average_correlation_per_keypoint")
ax.set_xlabel("n_keypoints, type")
ax.set_title("Average Correlation per Keypoint (Max Aggregated)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output.png")

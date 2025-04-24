from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import plotly.express as px
import tqdm

import wandb


# Initialize wandb API
api = wandb.Api()

# Replace with your project path
runs = api.runs("rhys-newbury/diffuse_keypoints_test_fr")

data = []


def get_latest_valid(history, key):
    series = history[key].dropna()
    return series.iloc[-1] if not series.empty else None


def fetch_run_data(run):
    try:
        x = run.history()
        return {
            "name": run.name,
            "n_keypoints": int(get_latest_valid(x, "n_keypoints")),
            "type": get_latest_valid(x, "type"),
            "category": get_latest_valid(x, "category"),
            "average_correlation_per_keypoint": get_latest_valid(
                x, "average_correlation_per_keypoint"
            ),
            "MMD-EMD": get_latest_valid(x, "MMD-EMD"),
            "MMD-CD": get_latest_valid(x, "MMD-CD"),
        }
    except Exception as e:
        print(f"Error in run {run.name}: {e}")
        return None


with ThreadPoolExecutor(max_workers=16) as executor:
    print(runs)
    futures = [
        executor.submit(fetch_run_data, run) for run in tqdm.tqdm(runs, total=770)
    ]
    print(len(futures))

    # Show tqdm progress bar as tasks complete
    data = []
    for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
        result = future.result()
        if result is not None:
            data.append(result)

# Convert to DataFrame
df = pd.DataFrame(data)

best_per_group = df.loc[
    df.groupby(["n_keypoints", "type", "category"])[
        "average_correlation_per_keypoint"
    ].idxmax()
].reset_index(drop=True)

# Sort by category to group visually
df_sorted = best_per_group.sort_values(by=["category", "type", "n_keypoints"])

# Build rows + manual gaps
rows_with_gaps = []
last_category = None

for _, row in df_sorted.iterrows():
    if last_category is not None and row["category"] != last_category:
        # Insert gap row
        rows_with_gaps.append(pd.Series({col: None for col in df_sorted.columns}))
    rows_with_gaps.append(row)
    last_category = row["category"]

# Create DataFrame with gaps
df_with_gaps = pd.DataFrame(rows_with_gaps)

# Assume df_with_gaps and labels are built the same way as before
df_with_gaps["label"] = df_with_gaps.apply(
    lambda row: (
        f"category: {row['category']}, type: {row['type']}, n_keypoints: {row['n_keypoints']}"
        if pd.notna(row["category"])
        else ""
    ),
    axis=1,
)

# Drop gap rows for plotting
plot_df = df_with_gaps[df_with_gaps["label"] != ""]

# Define metrics and nice titles
metrics = {
    "average_correlation_per_keypoint": "Average Correlation per Keypoint",
    "MMD-EMD": "MMD-EMD",
    "MMD-CD": "MMD-CD",
}

# Plot each metric in its own chart
for metric_key, metric_title in metrics.items():
    fig = px.bar(
        plot_df,
        y="label",
        x=metric_key,
        orientation="h",
        hover_data=["category", "type", "n_keypoints", metric_key],
        title=metric_title,
        labels={metric_key: metric_title},
    )

    fig.update_layout(
        yaxis={"autorange": "reversed"},
        height=600,
    )

    fig.show()

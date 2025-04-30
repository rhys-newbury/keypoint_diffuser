import json

import pandas as pd
import requests


category_mapping = {
    "02691156": "Airplane",
    "02773838": "Bag",
    "02954340": "Cap",
    "02958343": "Car",
    "03001627": "Chair",
    "03261776": "Earphone",
    "03467517": "Guitar",
    "03624134": "Knife",
    "03636649": "Lamp",
    "03642806": "Laptop",
    "03790512": "Motorbike",
    "03797390": "Mug",
    "03948459": "Pistol",
    "04099429": "Rocket",
    "04225987": "Skateboard",
    "04379243": "Table",
}


WANDB_API_KEY = (
    "db09fabd9a9cd7887ace1f168b3701bfa094f12b"  # Replace with your actual API key
)
entity = "rhys-newbury"
project = "diffuse_keypoints_test_fr"

url = "https://api.wandb.ai/graphql"

headers = {
    "Authorization": f"Bearer {WANDB_API_KEY}",
    "Content-Type": "application/json",
}

query = """
query Runs($entity: String!, $project: String!, $first: Int, $after: String) {
  project(entityName: $entity, name: $project) {
    runs(first: $first, after: $after) {
      pageInfo {
        endCursor
        hasNextPage
      }
      edges {
        node {
          displayName
          summaryMetrics
        }
      }
    }
  }
}
"""

summary_metrics = []
run_display_names = []

after_cursor = None

while True:
    variables = {
        "entity": entity,
        "project": project,
        "first": 1000,
        "after": after_cursor,
    }

    response = requests.post(
        url, headers=headers, json={"query": query, "variables": variables}, timeout=60
    )
    response.raise_for_status()
    data = response.json()

    edges = data["data"]["project"]["runs"]["edges"]
    for edge in edges:
        summary_metrics.append(edge["node"]["summaryMetrics"])
        run_display_names.append(edge["node"]["displayName"])

    page_info = data["data"]["project"]["runs"]["pageInfo"]
    if page_info["hasNextPage"]:
        after_cursor = page_info["endCursor"]
    else:
        break

# Done: All runs fetched!
data = []
# Example print
for summary_metric, run_display_name in zip(
    summary_metrics, run_display_names, strict=True
):
    summary_metric_ = json.loads(summary_metric)
    try:
        x = {
            "name": run_display_name,
            "n_keypoints": summary_metric_["n_keypoints"],
            "type": summary_metric_["type"],
            "category": summary_metric_["category"],
            "average_correlation_per_keypoint": summary_metric_.get(
                "average_correlation_per_keypoint", -1
            ),
            "MMD-EMD": summary_metric_["MMD-EMD"],
            "MMD-CD": summary_metric_["MMD-CD"],
        }
        data.append(x)
    except KeyError:
        pass

print(f"Total runs collected: {len(summary_metrics)}")

df = pd.DataFrame(data).dropna()

df = df.replace(-1, pd.NA)
print(df.columns.tolist(), data)
df["MMD-CD"] = pd.to_numeric(df["MMD-CD"], errors="coerce")
df["average_correlation_per_keypoint"] = pd.to_numeric(
    df["average_correlation_per_keypoint"], errors="coerce"
)


df.to_csv("output.csv", index=False)

df_filtered = df.dropna(
    subset=["average_correlation_per_keypoint", "MMD-CD"], how="all"
)


# Define a custom selection function
def select_best(group):
    if group["average_correlation_per_keypoint"].notna().any():
        # If any non-NaN correlations, pick the highest correlation
        best_idx = group["average_correlation_per_keypoint"].idxmax()
    else:
        # Otherwise, pick the lowest MMD-CD
        best_idx = group["MMD-CD"].idxmin()
    return best_idx


# Apply per group
best_indices = df_filtered.groupby(["n_keypoints", "type", "category"]).apply(
    select_best
)

# Select the rows
best_per_group = df_filtered.loc[best_indices].reset_index(drop=True)


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


# Extract rows
# Only keep rows where n_keypoints == 8
list_view = plot_df[plot_df["n_keypoints"] == 8][
    ["category", "type", "n_keypoints", *list(metrics.keys())]
]

latex_tables = {
    "Average Correlation per Keypoint": [],
    "MMD-CD": [],
    "MMD-EMD": [],
}

# Extract rows into LaTeX structure
for _idx, row in list_view.iterrows():
    category = row["category"]
    type_ = row["type"]

    latex_tables["Average Correlation per Keypoint"].append(
        (type_, category, row["average_correlation_per_keypoint"])
    )
    latex_tables["MMD-CD"].append((type_, category, row["MMD-CD"]))
    latex_tables["MMD-EMD"].append((type_, category, row["MMD-EMD"]))


def generate_latex_table(metric_name, data, maximize=True, scientific=False):
    categories = sorted({x[1] for x in data})
    types = sorted({x[0] for x in data})

    lookup = {(t, c): v for (t, c, v) in data}

    # Precompute best per category
    best_values = {}
    for cat in categories:
        values = [(t, lookup.get((t, cat))) for t in types]
        values = [(t, v) for (t, v) in values if v is not None]
        if not values:
            continue
        best_value = (
            max(v for t, v in values) if maximize else min(v for t, v in values)
        )
        best_values[cat] = best_value

    # Precompute averages
    averages = {}
    for type_ in types:
        values = [lookup.get((type_, cat)) for cat in categories]
        values = [v for v in values if v is not None]
        if values:
            averages[type_] = sum(values) / len(values)
        else:
            averages[type_] = None

    # Find the best average
    avg_values = [(t, v) for (t, v) in averages.items() if v is not None]
    if maximize:
        best_avg_value = max(avg_values, key=lambda x: x[1])[1]
    else:
        best_avg_value = min(avg_values, key=lambda x: x[1])[1]

    mapped_categories = [category_mapping.get(c, c) for c in categories]

    table = (
        "\\begin{table*}[h]\n\\centering\n\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}{l|"
        + "c" * (len(categories))
        + "|c}\n"
    )
    table += "\\toprule\n"
    table += "Type & " + " & ".join(mapped_categories) + " & Average \\\\\n"
    table += "\\midrule\n"

    for type_ in types:
        row_entries = []
        for cat in categories:
            value = lookup.get((type_, cat), None)
            if value is None:
                row_entries.append("-")
            else:
                formatted = f"{value:.2e}" if scientific else f"{value:.4f}"
                if value == best_values.get(cat):
                    formatted = f"\\textbf{{{formatted}}}"
                row_entries.append(formatted)

        # Add average
        avg_value = averages.get(type_)
        if avg_value is None:
            avg_formatted = "-"
        else:
            avg_formatted = f"{avg_value:.2e}" if scientific else f"{avg_value:.4f}"

            if avg_value == best_avg_value:
                avg_formatted = f"\\textbf{{{avg_formatted}}}"

        row_entries.append(avg_formatted)

        table += f"{type_} & " + " & ".join(row_entries) + " \\\\\n"

    table += "\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n"
    table += f"\\caption{{{metric_name}}}\n\\end{{table*}}\n"
    return table


# Generate and print LaTeX tables
for metric_name, data in latex_tables.items():
    if metric_name == "Average Correlation per Keypoint":
        maximize = True
        scientific = False
    elif metric_name == "MMD-EMD":
        maximize = False
        scientific = True
    else:
        maximize = False
        scientific = False

    latex_code = generate_latex_table(
        metric_name, data, maximize=maximize, scientific=scientific
    )
    print(latex_code)
    print("\n\n")

#!/usr/bin/env python3
import argparse
import sqlite3
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


# Added recon metrics
VALID_METRICS = {"das", "fwd", "bwd", "miou_at_0_1", "correlation", "recon_cd", "recon_emd"}


# ---------------- DB helpers ----------------
def table_exists(db_path: Path, table: str) -> bool:
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table,)
        )
        return cur.fetchone() is not None
    finally:
        con.close()


# ---------------- DB fetchers ----------------
def fetch_rows_runs(db_path: Path, model_filter: str | None) -> list[tuple]:
    if not table_exists(db_path, "runs"):
        return []
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    q = (
        "SELECT id, model, ckpt, annotation_json, pcd_path, batch_size, key_points, "
        "category, fwd, bwd, das, miou_at_0_1 FROM runs"
    )
    params: tuple = ()
    if model_filter:
        q += " WHERE model = ?"
        params = (model_filter,)
    rows = cur.execute(q, params).fetchall()
    con.close()
    return rows


def fetch_rows_corr(db_path: Path, model_filter: str | None) -> list[tuple]:
    """
    Fetch correlation results:
      - Always include 'Ours' from runs_correlation2.
      - Include other models from runs_correlation.
      - If model_filter is set, limit results accordingly.
    """
    rows: list[tuple] = []
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    def safe_fetch(table: str, query: str, params: tuple = ()):
        if not table_exists(db_path, table):
            return []
        return cur.execute(query, params).fetchall()

    if model_filter:
        if model_filter == "Ours":
            q = (
                "SELECT id, model, ckpt, annotation_json, pcd_path, batch_size, "
                "key_points, category, correlation FROM runs_correlation2 WHERE model = ?"
            )
            rows.extend(safe_fetch("runs_correlation2", q, (model_filter,)))
        else:
            q = (
                "SELECT id, model, ckpt, annotation_json, pcd_path, batch_size, "
                "key_points, category, correlation FROM runs_correlation WHERE model = ?"
            )
            rows.extend(safe_fetch("runs_correlation", q, (model_filter,)))
    else:
        q_ours = (
            "SELECT id, model, ckpt, annotation_json, pcd_path, batch_size, "
            "key_points, category, correlation FROM runs_correlation2 WHERE model = 'Ours'"
        )
        q_others = (
            "SELECT id, model, ckpt, annotation_json, pcd_path, batch_size, "
            "key_points, category, correlation FROM runs_correlation WHERE model != 'Ours'"
        )
        rows.extend(safe_fetch("runs_correlation2", q_ours))
        rows.extend(safe_fetch("runs_correlation", q_others))

    con.close()
    return rows


def fetch_rows_recon(db_path: Path, model_filter: str | None) -> list[tuple]:
    """
    Fetch reconstruction metrics from reconstruction_results_low:
      columns: id, model, ckpt, category, batch_size, cd_recon, emd_recon
    """
    if not table_exists(db_path, "reconstruction_results_low"):
        return []
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    q = (
        "SELECT id, model, ckpt, category, batch_size, cd_recon, emd_recon "
        "FROM reconstruction_results_low"
    )
    params: tuple = ()
    if model_filter:
        q += " WHERE model = ?"
        params = (model_filter,)
    rows = cur.execute(q, params).fetchall()
    con.close()
    return rows


# ---------------- Metric helpers ----------------
def metric_title(metric: str) -> str:
    if metric == "correlation":
        return "Correlation $\\uparrow$"
    if metric == "recon_cd":
        return "Chamfer ($\\downarrow$)"
    if metric == "recon_emd":
        return "EMD ($\\downarrow$)"
    return metric


def higher_is_better(metric: str) -> bool:
    # Reconstruction distances are LOWER-is-better
    return metric in {"das", "fwd", "miou_at_0_1", "correlation"}


# ---------------- Per-DB best maps ----------------
def best_map_from_rows(rows: list[tuple], metric: str, table_name: str) -> dict:
    """
    Best per (category, algo) inside a SINGLE DB for the chosen metric.
    Chooses max for higher-is-better, min for lower-is-better.
    """
    # Column index per table
    if table_name == "runs":
        idx = {"fwd": 8, "bwd": 9, "das": 10, "miou_at_0_1": 11}[metric]
        cat_idx, algo_idx = 7, 1
    elif table_name == "runs_correlation":
        idx = 8  # correlation
        cat_idx, algo_idx = 7, 1
    elif table_name == "reconstruction_results_low":
        # rows: id(0), model(1), ckpt(2), category(3), batch_size(4), cd(5), emd(6)
        idx = 5 if metric == "recon_cd" else 6
        cat_idx, algo_idx = 3, 1
    else:
        raise ValueError(f"Unknown table: {table_name}")

    want_max = higher_is_better(metric)

    best: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        cat = str(r[cat_idx]).capitalize()
        algo = r[algo_idx]
        val = r[idx]
        if val is None:
            continue
        v = float(val)
        if cat not in best:
            best[cat] = {}
        if algo not in best[cat]:
            best[cat][algo] = {"metric": v}
        else:
            if want_max:
                if v > best[cat][algo]["metric"]:
                    best[cat][algo]["metric"] = v
            else:
                if v < best[cat][algo]["metric"]:
                    best[cat][algo]["metric"] = v
    return best


# ---------------- Aggregate across DBs ----------------
def aggregate_across_dbs(best_maps: list[dict], metric: str) -> dict:
    """
    Aggregate per (category, algo) across DBs.
    Returns agg[cat][algo] = {"values": np.array, "mean": float, "std": float, "best": float, "count": int}
    where "best" is max if higher-is-better else min (computed over per-DB bests).
    """
    cats = sorted({c for bm in best_maps for c in bm})
    algos = sorted({a for bm in best_maps for c in bm for a in bm[c]})
    want_max = higher_is_better(metric)

    agg: dict[str, dict[str, dict[str, Any]]] = {}
    for c in cats:
        agg[c] = {}
        for a in algos:
            vals = []
            for bm in best_maps:
                if c in bm and a in bm[c]:
                    vals.append(float(bm[c][a]["metric"]))
                else:
                    vals.append(np.nan)
            arr = np.array(vals, dtype=float)
            if np.all(np.isnan(arr)):
                continue
            mask = ~np.isnan(arr)
            finite = arr[mask]
            best_val = float(np.max(finite)) if want_max else float(np.min(finite))
            agg[c][a] = {
                "values": arr,
                "mean": float(np.nanmean(arr)),
                "std": float(np.nanstd(arr, ddof=0)),
                "best": best_val,
                "count": int(mask.sum()),
            }
    return agg


# ---------------- Plotting ----------------
def plot_grouped_with_errbars(agg_map: dict, metric: str, out_path: Path | None):
    """
    Bars = best across DBs; error bars = std across DBs (from per-DB bests).
    Works for both higher- and lower-is-better metrics.
    """
    cats = sorted(agg_map.keys())
    algos = sorted({a for c in agg_map.values() for a in c})
    if not cats or not algos:
        print(f"[{metric}] Nothing to plot.")
        return

    x = np.arange(len(cats))
    width = min(0.8 / max(len(algos), 1), 0.25)
    plt.figure(figsize=(max(8, 1.2 * len(cats)), 5))

    for i, algo in enumerate(algos):
        y_best = [
            agg_map[c][algo]["best"] if algo in agg_map[c] else np.nan for c in cats
        ]
        y_std = [
            agg_map[c][algo]["std"] if algo in agg_map[c] else np.nan for c in cats
        ]
        y = np.array(y_best, dtype=float)
        yerr = np.array([0.0 if np.isnan(s) else s for s in y_std], dtype=float)

        offset = (i - (len(algos) - 1) / 2) * width
        bars = plt.bar(x + offset, y, width=width, label=algo, yerr=yerr, capsize=3)

        # annotate bars with "best" value (2 sig figs)
        for bar, v in zip(bars, y):
            if not np.isnan(v):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{v:.2g}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    plt.title(f"Best {metric_title(metric)} per category (error bars: std across DBs)")
    plt.ylabel(metric_title(metric))
    plt.xticks(x, cats, rotation=45, ha="right")
    plt.legend(title="Algorithm", fontsize=8)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200)
        print(f"[✓] saved plot -> {out_path}")
    else:
        plt.show()


# ---------------- LaTeX helpers ----------------
def fmt_sig2(x: float) -> str:
    if np.isnan(x):
        return "--"
    return f"{x:.2g}"


def build_latex_table_multi_metrics(
    agg_by_metric: dict[str, dict],
    metrics: list[str],
    algo_order: list[str] | None = None,
    category_order: list[str] | None = None,
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """
    One big table:
    Category | [metric1 block: algos...] | [metric2 block: algos...] | ...
    Each cell: mean ± std (2 sig figs), best per row per metric is bolded.
    Summary row ("Average ± Std. Dev.") per metric block.
    """
    # Collect union of categories across all metrics
    all_cats = set()
    for m in metrics:
        all_cats |= set(agg_by_metric[m].keys())
    cats = category_order if category_order else sorted(all_cats)

    # For each metric, collect algos available; or force algo_order if provided
    algos_per_metric = {}
    for m in metrics:
        agg = agg_by_metric[m]
        if algo_order:
            algos = algo_order
        else:
            algos = sorted({a for c in agg.values() for a in c})
            if "Ours" in algos:
                algos = [a for a in algos if a != "Ours"] + ["Ours"]
        algos_per_metric[m] = algos

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\begin{adjustbox}{max width=\\textwidth}")
    lines.append(
        f"\\begin{{tabular}}{{l|{'|'.join(['c'*len(algos_per_metric[m]) for m in metrics])}}}"
    )
    lines.append("\\toprule")

    # First header line with metric blocks
    header_blocks = []
    for m in metrics:
        header_blocks.append(
            f"\\multicolumn{{{len(algos_per_metric[m])}}}{{c}}{{{metric_title(m)}}}"
        )
    lines.append("\\multirow{2}{*}{Category} & " + " & ".join(header_blocks) + " \\\\")

    # Second header line with algo names
    header_algos = []
    for m in metrics:
        header_algos.extend(algos_per_metric[m])
    lines.append("& " + " & ".join(header_algos) + " \\\\")
    lines.append("\\midrule")

    # Body rows
    for c in cats:
        row = [c]
        for m in metrics:
            agg = agg_by_metric[m]
            algos = algos_per_metric[m]

            # collect values for bolding (based on MEAN)
            means_for_bolding = []
            cell_texts = []
            for a in algos:
                cell = agg.get(c, {}).get(a)
                if not cell:
                    cell_texts.append("--")
                    means_for_bolding.append(np.nan)
                    continue
                mean = cell["mean"]
                std = cell["std"]
                cell_texts.append(f"{fmt_sig2(mean)} $\\pm$ {fmt_sig2(std)}")
                means_for_bolding.append(mean)

            # bold best within metric block for the category
            means_arr = np.array(means_for_bolding, dtype=float)
            if np.isfinite(means_arr).any():
                if higher_is_better(m):
                    best_idx = int(np.nanargmax(means_arr))
                else:
                    best_idx = int(np.nanargmin(means_arr))
                if not np.isnan(means_arr[best_idx]):
                    cell_texts[best_idx] = f"\\textbf{{{cell_texts[best_idx]}}}"

            row.extend(cell_texts)

        lines.append(" & ".join(row) + " \\\\")

    # Summary row
    lines.append("\\midrule")
    row_sum = ["\\textbf{Average $\\pm$ Std. Dev.}"]
    for m in metrics:
        agg = agg_by_metric[m]
        algos = algos_per_metric[m]

        per_algo_means = []
        cells = []
        for a in algos:
            vals = [agg.get(c, {}).get(a, {}).get("mean", np.nan) for c in cats]
            arr = np.array(vals, dtype=float)
            m_mean = np.nanmean(arr) if np.isfinite(arr).any() else np.nan
            m_std = np.nanstd(arr, ddof=0) if np.isfinite(arr).any() else np.nan
            per_algo_means.append(m_mean)
            cells.append(f"{fmt_sig2(m_mean)} $\\pm$ {fmt_sig2(m_std)}")

        means_arr = np.array(per_algo_means, dtype=float)
        if np.isfinite(means_arr).any():
            if higher_is_better(m):
                best_idx = int(np.nanargmax(means_arr))
            else:
                best_idx = int(np.nanargmin(means_arr))
            if not np.isnan(means_arr[best_idx]):
                cells[best_idx] = f"\\textbf{{{cells[best_idx]}}}"

        row_sum.extend(cells)

    lines.append(" & ".join(row_sum) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    lines.append("\\caption{Per-category performance (mean $\\pm$ std across DBs) for multiple metrics.}")
    lines.append("\\label{tab:multi_metrics_mean_std}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


# ---------------- Orchestrator ----------------
def main():
    ap = argparse.ArgumentParser(
        description="Plot grouped bars and emit a LaTeX table (mean ± std) across multiple metrics, including reconstruction."
    )
    ap.add_argument(
        "--db",
        type=Path,
        nargs="+",
        required=True,
        help="One or more DBs (e.g., results.db results2.db results3.db)",
    )
    ap.add_argument(
        "--metrics",
        choices=sorted(VALID_METRICS),
        nargs="+",
        required=True,
        help="e.g., --metrics das correlation miou_at_0_1 recon_cd recon_emd",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional algorithm filter (only include this model).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("grouped_best_std.png"),
        help="Plot output base path; per-metric plots saved as <stem>_<metric><suffix>. Use '-' to show instead.",
    )
    ap.add_argument(
        "--algo-order",
        type=str,
        nargs="+",
        default=None,
        help="Optional fixed column order for algos, e.g. --algo-order KPD DPM Ours",
    )
    ap.add_argument(
        "--category-order",
        type=str,
        nargs="+",
        default=None,
        help="Optional fixed row order for categories.",
    )
    args = ap.parse_args()

    agg_by_metric: dict[str, dict] = {}
    for metric in args.metrics:
        best_maps_per_db = []
        for db in args.db:
            if metric == "correlation":
                rows = fetch_rows_corr(db, args.model)
                table_name = "runs_correlation"
            elif metric in {"recon_cd", "recon_emd"}:
                rows = fetch_rows_recon(db, args.model)
                table_name = "reconstruction_results_low"
            else:
                rows = fetch_rows_runs(db, args.model)
                table_name = "runs"
            best_maps_per_db.append(
                best_map_from_rows(rows, metric, table_name) if rows else {}
            )

        agg = aggregate_across_dbs(best_maps_per_db, metric)
        if not agg:
            print(f"[{metric}] No data found across the provided DBs. Skipping this metric.")
            continue
        agg_by_metric[metric] = agg

        # Plot per metric
        if str(args.out) == "-":
            plot_grouped_with_errbars(agg, metric, out_path=None)
        else:
            out_path = args.out.with_name(f"{args.out.stem}_{metric}{args.out.suffix}")
            plot_grouped_with_errbars(agg, metric, out_path)

    if not agg_by_metric:
        print("No metrics produced any data; nothing to output.")
        return

    # Build one big LaTeX table across the metrics that produced data
    metrics_in_table = [m for m in args.metrics if m in agg_by_metric]
    latex = build_latex_table_multi_metrics(
        agg_by_metric=agg_by_metric,
        metrics=metrics_in_table,
        algo_order=args.algo_order,
        category_order=args.category_order,
        caption="Per-category performance (mean $\\pm$ std across DBs) for multiple metrics.",
        label="tab:multi_metrics_mean_std",
    )
    print("\n" + "=" * 80)
    print(latex)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

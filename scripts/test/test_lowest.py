#!/usr/bin/env python3
import argparse
import re
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
from classes import MODEL_CLASSES

import wandb


# -------------------- config / patterns --------------------
STEP_IN_NAME_RE = re.compile(
    r"(?:^|[_\-])(?:step|global_step)[^\d]*(\d+)", re.IGNORECASE
)


# -------------------- DB helpers --------------------
def table_exists(con: sqlite3.Connection, table: str) -> bool:
    cur = con.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table,)
    )
    return cur.fetchone() is not None


def already_evaluated(
    eval_db: Path, ckpt: Path, model: str, category: str, table_name: str
) -> bool:
    con = sqlite3.connect(str(eval_db))
    try:
        if not table_exists(con, table_name):
            return False
        cur = con.cursor()
        cur.execute(
            f"SELECT 1 FROM {table_name} WHERE ckpt = ? AND model = ? AND category = ? LIMIT 1;",
            (str(ckpt), model, category),
        )
        return cur.fetchone() is not None
    finally:
        con.close()


# -------------------- W&B helpers --------------------
def wandb_run_by_display_name(entity: str, project: str, run_name: str):
    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}", {"display_name": run_name}))
    if not runs:
        raise RuntimeError(f"W&B run not found: {entity}/{project}:{run_name}")
    assert len(runs) == 1
    return runs[0]


def wandb_series(run) -> dict[str, dict[str, np.ndarray]]:
    """
    Fetch ALL numeric metrics from W&B history with steps.
    Returns:
        { metric_name: { "steps": np.ndarray, "values": np.ndarray } }
    """
    # Fetch full history
    df = run.history(pandas=True)
    if df is None or len(df) == 0 or "_step" not in df.columns:
        raise RuntimeError("W&B history empty or missing _step column.")

    # Remove NaN rows and fill missing numeric
    df = df.dropna(subset=["_step"])
    steps = df["_step"].to_numpy(dtype=float, copy=False)

    # W&B internal/system columns to ignore
    exclude_cols = {
        "_runtime",
        "_timestamp",
        "_wandb",
        "_step",
    }

    out = {}
    for col in df.columns:
        if col in exclude_cols:
            continue
        if col.startswith("_"):  # W&B internal
            continue
        # Only keep numeric columns
        if not np.issubdtype(df[col].dropna().dtype, np.number):
            continue

        vals = df[col].to_numpy(dtype=float, copy=False)
        out[col] = {"steps": steps, "values": vals}

    if not out:
        raise RuntimeError("No numeric metrics found in W&B history.")

    return out


def ema(x: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


def pick_lowest_in_tail(
    steps: np.ndarray, vals: np.ndarray, tail_frac: float = 0.2
) -> int:
    order = np.argsort(steps)
    steps, vals = steps[order], vals[order]
    n = len(steps)
    start = max(0, int((1.0 - tail_frac) * n))
    smoothed = ema(vals[start:], alpha=0.1)
    j = int(np.nanargmin(smoothed))
    return start + j


def choose_target_step_from_wandb(
    run_name: str, entity: str, project: str, tail_frac: float = 0.2
) -> float:
    run = wandb_run_by_display_name(entity, project, run_name)

    series = wandb_series(run, keys=("diffusion_loss", "chamfer_loss", "mse_loss"))

    for metric in ("diffusion_loss", "chamfer_loss", "mse_loss"):
        if metric in series:
            steps = series[metric]["steps"]
            vals = series[metric]["values"]
            idx = pick_lowest_in_tail(steps, vals, tail_frac=tail_frac)
            return float(steps[idx])

    raise RuntimeError(
        "No usable metrics found in W&B history (diffusion/chamfer/mse)."
    )


# -------------------- checkpoint resolution --------------------
def find_ckpt_for_step(run_dir: Path, target_step: float) -> Path:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    step_int = int(round(target_step))

    # 1) direct pattern matches that include the step
    patterns = [f"*step*{step_int}*.pth", f"*{step_int}*.pth"]
    for pat in patterns:
        matches = list(run_dir.rglob(pat))
        if matches:
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return matches[0]

    # 2) nearest step parsed from filenames across directory
    all_pths = list(run_dir.rglob("*.pth"))
    if not all_pths:
        raise FileNotFoundError(f"No .pth files under {run_dir}")

    best, best_abs = None, None
    for p in all_pths:
        m = STEP_IN_NAME_RE.search(p.name)
        if not m:
            continue
        s = float(m.group(1))
        diff = abs(s - target_step)
        if best is None or diff < best_abs:
            best, best_abs = p, diff

    if best is not None:
        return best

    # 3) as a last resort (still simple, no env fallbacks): most recent .pth
    all_pths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return all_pths[0]


# -------------------- main orchestration --------------------
def main():
    ap = argparse.ArgumentParser(
        description="Evaluate a single W&B-selected checkpoint per train_runs row."
    )
    ap.add_argument(
        "--db", type=Path, required=True, help="Path to results.db (has train_runs)"
    )
    ap.add_argument("--wandb-entity", type=str, default="rhys-newbury")

    ap.add_argument(
        "--ckpt-root",
        type=Path,
        default=Path("/mnt/slow2"),
        help="Root directory containing subdirs named by W&B run display_name.",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--only-algo")
    ap.add_argument("--only-category")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument(
        "--tail-frac",
        type=float,
        default=0.2,
        help="Use last tail_frac of steps when picking best metric (default 0.2).",
    )

    ap.add_argument("--reconstruct-script", type=Path, default=Path("get_mmd.py"))

    args = ap.parse_args()
    eval_db_path = args.db

    con = sqlite3.connect(str(args.db))
    cur = con.cursor()

    q = "SELECT id, algo, category, ckpt_dir, key_points FROM train_runs"
    clauses, params = [], []
    if args.only_algo:
        clauses.append("algo = ?")
        params.append(args.only_algo)
    if args.only_category:
        clauses.append("category = ?")
        params.append(args.only_category)
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    q += " ORDER BY id ASC"

    rows = list(cur.execute(q, params))
    con.close()

    if not rows:
        print("No matching rows.")
        return

    def run_cmd(cmd: list[str], rid: int, label: str) -> int:
        print("→", " ".join(cmd))
        if args.dry_run:
            print(f"[run_id={rid}] (dry-run) {label} OK")
            return 0
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            msg = f"[run_id={rid}] {label} failed code {rc}"
            if args.keep_going:
                print("!!", msg)
                return rc
            raise SystemExit(msg)
        print(f"[run_id={rid}] ✓ {label} done")
        return rc

    for rid, algo, category, ckpt_dir, key_points in rows:
        if algo == "SC3K":
            continue

        # normalize algo for child scripts
        algo_ = "Ours" if algo == "Ours2" else algo

        name_map = {"SM": "skeleton_merger"}

        try:
            final_dir = Path(ckpt_dir).name  # W&B display_name
            wandb_project = f"{name_map.get(algo_, algo_)}_{category}_train"

            run = wandb_run_by_display_name(args.wandb_entity, wandb_project, final_dir)
            model_cls = MODEL_CLASSES[algo_]

            run_dir = args.ckpt_root / "skeleton_merger" / final_dir

            run_dirs = [f"/mnt/slow3/{final_dir}", run_dir]

            ckpt_path = model_cls.choose_model(wandb_series(run), run_dirs)["ckpt"]

            # Reconstruction
            if not already_evaluated(
                eval_db_path, ckpt_path, algo_, category, "mmd_results"
            ):
                cmd = [
                    "python3",
                    str(args.reconstruct_script),
                    algo_,
                    "--ckpt",
                    str(ckpt_path),
                    "--batch-size",
                    str(args.batch_size),
                    "--key_point",
                    str(key_points),
                    "--category",
                    str(category),
                    "--db-path",
                    str(eval_db_path),
                ]
                run_cmd(cmd, rid, "get_reconstruction")

        except Exception as e:
            if args.keep_going:
                print(f"[run_id={rid}] !! {e}")
                continue
            raise


if __name__ == "__main__":
    main()

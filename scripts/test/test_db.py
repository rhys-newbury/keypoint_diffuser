#!/usr/bin/env python3
import argparse
import re
import sqlite3
import subprocess
from pathlib import Path


EPOCH_RE = re.compile(r"(?P<k>\d+)kp_(?P<epoch>\d+)\.pth$", re.IGNORECASE)


def list_ckpts(path_str: str) -> list[Path]:
    """
    If path is a file -> [file].
    If path is a dir  -> all *.pth files inside (recursively), sorted by epoch if parseable,
                         else by mtime ascending.
    """
    p = Path(path_str)
    if p.is_file() and p.suffix == ".pth":
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"Checkpoint path is neither .pth nor directory: {p}")

    pths = list(p.glob("**/*.pth"))
    if not pths:
        raise FileNotFoundError(f"No .pth files found under {p}")

    # Try epoch-aware sort first
    with_epochs, others = [], []
    for f in pths:
        m = EPOCH_RE.search(f.name)
        if m:
            with_epochs.append((int(m.group("epoch")), f))
        else:
            others.append(f)

    result = []
    if with_epochs:
        with_epochs.sort(key=lambda x: x[0])  # earliest -> latest
        result.extend([f for _, f in with_epochs])
    result.extend(sorted(others, key=lambda f: f.stat().st_mtime))
    return result


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    c = con.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table,))
    return c.fetchone() is not None


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


def main():
    ap = argparse.ArgumentParser(
        description="Loop train_runs and evaluate checkpoints (DAS and/or Correlation)."
    )
    ap.add_argument(
        "--db", type=Path, required=True, help="Path to results.db (has train_runs)"
    )
    ap.add_argument("--pcd-path", type=Path, default=Path("/app/pcds"))
    ap.add_argument("--label-path", type=Path, default=Path("/mnt/slow2/shape-data"))

    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--only-algo")
    ap.add_argument("--only-category")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-going", action="store_true")

    # What to run
    ap.add_argument(
        "--mode",
        choices=["das", "corr", "both"],
        default="both",
        help="Which evaluation(s) to run per checkpoint.",
    )

    # Scripts
    ap.add_argument(
        "--das-script", type=Path, default=Path("get_das.py"), help="Path to DAS script"
    )
    ap.add_argument(
        "--corr-script",
        type=Path,
        default=Path("get_correspondence.py"),
        help="Path to Correlation script",
    )

    # DB passed through to the child script(s)
    ap.add_argument(
        "--eval-db-path",
        type=Path,
        default=None,
        help="DB path passed to child scripts via --db-path (default: same as --db)",
    )

    # Table names (for existence checks / child overrides)
    ap.add_argument(
        "--das-table",
        type=str,
        default="runs",
        help="Table name to check/populate for DAS (default: runs)",
    )
    ap.add_argument(
        "--corr-table",
        type=str,
        default="runs_correlation",
        help="Table name to check/populate for correlation (default: runs_correlation)",
    )

    # Optional limits/filters
    ap.add_argument(
        "--start-epoch",
        type=int,
        default=None,
        help="If filenames have '<K>kp_<E>.pth', skip checkpoints with epoch < E",
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="Process at most N checkpoints per row"
    )
    args = ap.parse_args()

    eval_db_path = args.eval_db_path or args.db

    con = sqlite3.connect(str(args.db))
    cur = con.cursor()
    # Expect columns: id, algo, category, ckpt_dir, key_points
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

    def run_cmd(cmd, rid, label):
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
        try:
            ckpt_list = list_ckpts(ckpt_dir)

            # apply optional epoch filter
            if args.start_epoch is not None:
                filtered = []
                for f in ckpt_list:
                    m = EPOCH_RE.search(f.name)
                    if m and int(m.group("epoch")) < args.start_epoch:
                        continue
                    filtered.append(f)
                ckpt_list = filtered

            if args.limit is not None:
                ckpt_list = ckpt_list[: args.limit]

            if not ckpt_list:
                print(f"[run_id={rid}] !! no checkpoints to process after filtering")
                continue

            for ckpt_path in ckpt_list:
                # DAS
                if args.mode in ("das", "both") and not already_evaluated(
                    eval_db_path, ckpt_path, algo, category, args.das_table
                ):
                    annotation_json = Path("/app/annotations") / f"{category}.json"
                    cmd_das = [
                        "python3",
                        str(args.das_script),
                        algo,
                        "--ckpt",
                        str(ckpt_path),
                        "--annotation-json",
                        str(annotation_json),
                        "--pcd-path",
                        str(args.pcd_path),
                        "--batch-size",
                        str(args.batch_size),
                        "--key-points",
                        str(key_points),
                        "--category",
                        str(category),
                        "--db-path",
                        str(eval_db_path),
                        "--table-name",
                        args.das_table,
                    ]
                    run_cmd(cmd_das, rid, "get_das")
                    # else: already done; skip

                # Correlation
                if args.mode in ("corr", "both") and not already_evaluated(
                    eval_db_path, ckpt_path, algo, category, args.corr_table
                ):
                    annotation_json = Path("/app/annotations") / f"{category}.json"
                    cmd_corr = [
                        "python3",
                        str(args.corr_script),
                        algo,
                        "--ckpt",
                        str(ckpt_path),
                        "--annotation-json",
                        str(annotation_json),
                        "--pcd-path",
                        str(args.pcd_path),
                        "--batch-size",
                        str(args.batch_size),
                        "--key-points",
                        str(key_points),
                        "--category",
                        str(category),
                        "--db-path",
                        str(eval_db_path),
                        "--label-path",
                        str(args.label_path),
                    ]
                    run_cmd(cmd_corr, rid, "get_correspondence")
                    # else: already done; skip

        except Exception as e:
            if args.keep_going:
                print(f"[run_id={rid}] !! {e}")
                continue
            raise


if __name__ == "__main__":
    main()

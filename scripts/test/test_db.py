#!/usr/bin/env python3
import argparse
import re
import sqlite3
import subprocess
from pathlib import Path

from .get_das import MODEL_CLASSES


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
    with_epochs = []
    others = []
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
    # Append the rest ordered by mtime
    result.extend(sorted(others, key=lambda f: f.stat().st_mtime))
    return result


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    c = con.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table,))
    return c.fetchone() is not None


def already_evaluated(eval_db: Path, ckpt: Path, model: str, category: str) -> bool:
    con = sqlite3.connect(str(eval_db))
    try:
        if not table_exists(con, "runs"):
            return False
        cur = con.cursor()
        cur.execute(
            "SELECT 1 FROM runs WHERE ckpt = ? AND model = ? AND category = ? LIMIT 1;",
            (str(ckpt), model, category),
        )
        return cur.fetchone() is not None
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser(
        description="Loop train_runs and call get_das for all checkpoints."
    )
    ap.add_argument(
        "--db", type=Path, required=True, help="Path to results.db (has train_runs)"
    )
    ap.add_argument(
        "--get-das", type=Path, default=Path("get_das.py"), help="Path to get_das.py"
    )
    ap.add_argument("--pcd-path", type=Path, default=Path("/app/pcds"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--only-algo", choices=MODEL_CLASSES.keys())
    ap.add_argument("--only-category")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    # DB that get_das writes into (defaults to same file as --db)
    ap.add_argument(
        "--eval-db-path",
        type=Path,
        default=None,
        help="DB passed to get_das --db-path (default: same as --db)",
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
                # if already_evaluated(eval_db_path, ckpt_path, algo, category):

                annotation_json = Path("/app/annotations") / f"{category}.json"
                cmd = [
                    "python3",
                    str(args.get_das),
                    algo,  # subcommand: SC3K / SM
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
                ]

                print("→", " ".join(cmd))
                if args.dry_run:
                    print(f"[run_id={rid}] (dry-run) OK")
                    continue

                rc = subprocess.run(cmd).returncode
                if rc != 0:
                    msg = f"[run_id={rid}] get_das failed ({ckpt_path}) code {rc}"
                    if args.keep_going:
                        print("!!", msg)
                        continue
                    raise SystemExit(msg)

                print(f"[run_id={rid}] ✓ done {ckpt_path}")

        except Exception as e:
            if args.keep_going:
                print(f"[run_id={rid}] !! {e}")
                continue
            raise


if __name__ == "__main__":
    main()

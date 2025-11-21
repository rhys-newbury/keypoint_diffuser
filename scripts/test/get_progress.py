#!/usr/bin/env python3
import argparse
import re
import sqlite3
from pathlib import Path

from tqdm import tqdm


EPOCH_RE = re.compile(r"(?P<k>\d+)kp_(?P<epoch>\d+)\.pth$", re.IGNORECASE)


def list_ckpts_fallback(path_str: str):
    base = Path(path_str)

    try:
        return list_ckpts(str(base))
    except FileNotFoundError:
        pass

    fallback = Path("/mnt/slow3") / base.name
    try:
        return list_ckpts(str(fallback))
    except FileNotFoundError:
        return []


def list_ckpts(path_str: str):
    p = Path(path_str)
    if p.is_file() and p.suffix == ".pth":
        return [p]

    if not p.is_dir():
        raise FileNotFoundError(f"Checkpoint path is neither .pth nor directory: {p}")

    files = list(p.glob("**/*.pth"))
    if not files:
        raise FileNotFoundError(f"No .pth files in {p}")

    with_epochs, others = [], []
    for f in files:
        m = EPOCH_RE.search(f.name)
        if m:
            with_epochs.append((int(m.group("epoch")), f))
        else:
            others.append(f)

    result = []
    if with_epochs:
        with_epochs.sort(key=lambda x: x[0])
        result.extend(f for _, f in with_epochs)

    result.extend(sorted(others, key=lambda f: f.stat().st_mtime))
    return result


def table_exists(con, table):
    c = con.cursor()
    c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?;", (table,))
    return c.fetchone() is not None


def already_evaluated(db, ckpt, model, category, table):
    con = sqlite3.connect(str(db))
    try:
        if not table_exists(con, table):
            return False

        cur = con.cursor()
        cur.execute(
            f"SELECT 1 FROM {table} WHERE ckpt=? AND model=? AND category=? LIMIT 1;",
            (str(ckpt), model, category),
        )
        return cur.fetchone() is not None
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser(description="Compute final progress only.")
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--only-algo")
    ap.add_argument("--only-category")
    ap.add_argument("--start-epoch", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--eval-db-path", type=Path, default=None)
    ap.add_argument("--das-table", type=str, default="runs")
    ap.add_argument("--corr-table", type=str, default="runs_correlation2")
    ap.add_argument("--mode", choices=["das", "corr", "all"], default="all")
    args = ap.parse_args()

    eval_db = args.eval_db_path or args.db

    # Load train_runs rows
    con = sqlite3.connect(str(args.db))
    cur = con.cursor()

    query = "SELECT id, algo, category, ckpt_dir, key_points FROM train_runs"
    clauses, params = [], []

    if args.only_algo:
        clauses.append("algo = ?")
        params.append(args.only_algo)

    if args.only_category:
        clauses.append("category = ?")
        params.append(args.only_category)

    if clauses:
        query += " WHERE " + " AND ".join(clauses)

    rows = list(cur.execute(query, params))
    con.close()

    if not rows:
        print("No matching rows.")
        return

    # ----- Progress counters -----
    total_ckpts = 0
    das_done = 0
    corr_done = 0

    # ----- Loop (no eval execution) -----
    for rid, algo, category, ckpt_dir, _key_points in tqdm(rows, total=len(rows)):
        if algo == "DPM":
            continue

        try:
            ckpt_list = list_ckpts_fallback(ckpt_dir)

            # Epoch filter
            if args.start_epoch is not None:
                ckpt_list = [
                    f
                    for f in ckpt_list
                    if (m := EPOCH_RE.search(f.name)) is None
                    or int(m.group("epoch")) >= args.start_epoch
                ]

            # Limit
            if args.limit is not None:
                ckpt_list = ckpt_list[: args.limit]

            if not ckpt_list:
                continue

            total_ckpts += len(ckpt_list)

            for ckpt_path in ckpt_list:
                if args.mode in ("das", "all") and already_evaluated(
                    eval_db, ckpt_path, algo, category, args.das_table
                ):
                    das_done += 1

                if args.mode in ("corr", "all") and already_evaluated(
                    eval_db, ckpt_path, algo, category, args.corr_table
                ):
                    corr_done += 1

        except Exception as e:
            print(f"[run_id={rid}] !! {e}")
            continue

    # ----- Final summary -----
    print("\n====== PROGRESS SUMMARY ======")
    print(f"Total checkpoints: {total_ckpts}")

    if args.mode in ("das", "all"):
        print(
            f"DAS:  {das_done} / {total_ckpts}  ({das_done / total_ckpts * 100:.1f}%)"
        )

    if args.mode in ("corr", "all"):
        print(
            f"CORR: {corr_done} / {total_ckpts}  ({corr_done / total_ckpts * 100:.1f}%)"
        )

    print("==============================\n")


if __name__ == "__main__":
    main()

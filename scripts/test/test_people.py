#!/usr/bin/env python3
import argparse
import random
import re
import sqlite3
import subprocess
from pathlib import Path
from typing import List


EPOCH_RE = re.compile(r"(?P<k>\d+)kp_(?P<epoch>\d+)\.pth$", re.IGNORECASE)


def list_ckpts(path_str: str) -> List[Path]:
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

    result: List[Path] = []
    if with_epochs:
        with_epochs.sort(key=lambda x: x[0])  # earliest -> latest
        result.extend([f for _, f in with_epochs])
    # fallback ordering for non-matching filenames
    result.extend(sorted(others, key=lambda f: f.stat().st_mtime))
    return result


def list_ckpts_fallback(path_str: str) -> List[Path]:
    """
    Try original ckpt_dir; if missing, try same final dir name under /mnt/slow3.
    """
    base = Path(path_str)

    # First try original path
    try:
        return list_ckpts(str(base))
    except FileNotFoundError:
        pass

    # Fallback: /mnt/slow3/<final_dir_name>
    final_dir = base.name
    fallback = Path("/mnt/slow3") / final_dir

    try:
        return list_ckpts(str(fallback))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No checkpoints found in either:\n  {base}\n  {fallback}"
        )


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    c = con.cursor()
    c.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
        (table,),
    )
    return c.fetchone() is not None


def already_evaluated(
    eval_db: Path,
    ckpt: Path,
    model: str,
    category: str,
    table_name: str,
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
        is_evaled = cur.fetchone() is not None
        if is_evaled:
            print(f"[skip] {ckpt} for algo={model}, category={category} already evaled")
        return is_evaled
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser(
        description="Loop train_runs and evaluate people checkpoints with eval_people script."
    )
    ap.add_argument(
        "--db",
        type=Path,
        required=True,
        help="Path to train results DB (must have train_runs table).",
    )
    ap.add_argument("--only-algo", type=str, default=None)
    ap.add_argument("--only-category", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-going", action="store_true")

    ap.add_argument(
        "--eval-script",
        type=Path,
        default=Path("eval_people.py"),
        help="Path to the people evaluation script.",
    )
    ap.add_argument(
        "--eval-table",
        type=str,
        default="runs_correlation_people",
        help="Table name where eval_people_corr.py writes (for already_evaluated checks).",
    )

    ap.add_argument(
        "--start-epoch",
        type=int,
        default=None,
        help="If filenames have '<K>kp_<E>.pth', skip checkpoints with epoch < E.",
    )
    args = ap.parse_args()

    eval_db_path = args.db
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
        print("No matching rows in train_runs.")
        return

    random.shuffle(rows)
    print(f"[info] {len(rows)} train_runs rows to process")

    def run_cmd(cmd, rid):
        print("→", " ".join(cmd))
        if args.dry_run:
            print(f"[run_id={rid}] (dry-run) OK")
            return 0
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            msg = f"[run_id={rid}] eval failed with code {rc}"
            if args.keep_going:
                print("!!", msg)
                return rc
            raise SystemExit(msg)
        print(f"[run_id={rid}] ✓ eval done")
        return rc

    for rid, algo, category, ckpt_dir, key_points in rows:
        print(f"\n[run_id={rid}] algo={algo}, category={category}")

        # Skip some algos if desired
        if algo == "DPM":
            print("[skip] DPM")
            continue

        # Map "Ours2" to "Ours" if you want consistency with older code
        algo_for_eval = "Ours" if algo == "Ours2" else algo

        try:
            ckpt_list = list_ckpts_fallback(ckpt_dir)

            # optional epoch filter
            if args.start_epoch is not None:
                filtered = []
                for f in ckpt_list:
                    m = EPOCH_RE.search(f.name)
                    if m and int(m.group("epoch")) < args.start_epoch:
                        continue
                    filtered.append(f)
                ckpt_list = filtered

            if not ckpt_list:
                print(f"[run_id={rid}] !! no checkpoints to process after filtering")
                continue

            random.shuffle(ckpt_list)

            for ckpt_path in ckpt_list:
                if already_evaluated(
                    eval_db_path,
                    ckpt_path,
                    algo_for_eval,
                    category,
                    args.eval_table,
                ):
                    continue

                cmd = [
                    "python3",
                    str(args.eval_script),
                    algo_for_eval,
                    "--ckpt",
                    str(ckpt_path),
                    "--key_point",
                    str(key_points),
                    "--db-path",
                    str(eval_db_path),
                ]
                run_cmd(cmd, rid)

        except Exception as e:
            msg = f"[run_id={rid}] !! {e}"
            if args.keep_going:
                print(msg)
                continue
            raise


if __name__ == "__main__":
    main()

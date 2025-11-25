#!/usr/bin/env python3
import argparse
import re
import sqlite3
import subprocess
from pathlib import Path

from classes import MODEL_CLASSES


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
        description="Loop train_runs and call get_das for all checkpoints."
    )
    ap.add_argument(
        "--db", type=Path, required=True, help="Path to training results.db (has train_runs)"
    )
    ap.add_argument(
        "--script", type=Path, default=Path("get_das.py"), help="Path to get_das.py"
    )
    ap.add_argument("--pcd-path", type=Path, default=Path("/mnt/shape-data/newdata/pcds"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--only-algo", choices=MODEL_CLASSES.keys())
    ap.add_argument("--only-category")
    ap.add_argument("--input-type", type=str, default="full")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=None)
    # DB that get_das writes into (defaults to same file as --db)
    ap.add_argument(
        "--eval-db-path",
        type=Path,
        default=None,
        help="path to output evaluation db (default: same as --db)",
    )
    ap.add_argument(
        "--table-name",
        type=str,
        default="runs",
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
    ap.add_argument("--epoch-interval", type=int, default=None)
    ap.add_argument("--only-epoch", type=int, default=None)
    
    args = ap.parse_args()
    eval_db_path = args.eval_db_path or args.db
    eval_db_path.parent.mkdir(parents=True, exist_ok=True)
    
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
        if algo == "Ours2": # hack
            algo = "Ours"
        try:
            # TODO: clean up logic ##################################################################################################################
            ckpt_list = list_ckpts(ckpt_dir)
            for f in ckpt_list:
                if "net_final" in str(f):
                    final_path = f
            # apply optional epoch filters
            epoch_num_list = []
            
            if args.only_epoch is not None:
                filtered_f = []
                for f in ckpt_list:
                    m = EPOCH_RE.search(f.name)
                    if m and (int(m.group("epoch")) == args.only_epoch):
                        filtered_f.append(f)
                        epoch_num_list.append(int(m.group("epoch")))
                        break
                    elif "net_final.pth" in f.name:
                        final_f = f
                # if the only_epoch was not found, use the final epoch
                if not filtered_f:
                    filtered_f.append(final_f)
                    epoch_num_list.append(args.only_epoch)
                ckpt_list = filtered_f
            elif arg.start_epoch is not None:
                filtered = []
                for f in ckpt_list:
                    m = EPOCH_RE.search(f.name)
                    if m and int(m.group("epoch")) < args.start_epoch:
                        continue
                    filtered.append(f)
                ckpt_list = filtered
                
                all_ms = []

                if args.epoch_interval is not None:
                    filtered_int = []
                    for f in ckpt_list:
                        # print(f"f: {f.name}")
                        m = EPOCH_RE.search(f.name)
                        if m: 
                            all_ms.append(int(m.group("epoch")))
                            mint = int(m.group("epoch"))
                        elif "net_final.pth" in f.name:
                            ms_interval = all_ms[-1] - all_ms[-2]
                            mint = all_ms[-1] + ms_interval
                        else:
                            print(f"what is this f: {f.name}")
                        # print(f"mint: {mint}")
                        if mint % args.epoch_interval == 0:
                            filtered_int.append(f)
                            epoch_num_list.append(mint)
                        else:
                            continue
                    ckpt_list = filtered_int
                    # failsafe adding the final epoch number (only applies if the ckpt_list length does not match the epoch_num_list length)
                    if epoch_num_list and (len(ckpt_list) != len(epoch_num_list)):
                        epoch_num_list.append(epoch_num_list[-1] + args.epoch_interval)


            if args.limit is not None:
                ckpt_list = ckpt_list[: args.limit]

            # append the final model net_final as well
            # ckpt_list.append(final_path)
            ###################################################################################################################

            if not ckpt_list:
                print(f"[run_id={rid}] !! no checkpoints to process after filtering")
                continue

            for ckpt_path in ckpt_list:
                if already_evaluated(
                    eval_db_path, ckpt_path, algo, category, args.table_name
                ):
                    print(f"{eval_db_path} already exists. Skipping...")
                    continue

                annotation_json = Path("/mnt/shape-data/newdata/annotations") / f"{category}.json"
                if "get_das" in str(args.script):
                    cmd = [
                        "python3",
                        str(args.script),
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
                        "--input-type",
                        str(args.input_type), 
                        "--source-points-dir",
                        "/mnt/slow/shapenetcorev2-source", 
                        # "--output-dir",
                        # str(args.output_dir / f"epoch_{epoch_num_list.pop(0)}"), 
                        # "--save", 
                    ]
                elif "get_reconstruction" in str(args.script):
                    cmd = [
                        "python3",
                        str(args.script),
                        algo,  # subcommand: SC3K / SM
                        "--ckpt",
                        str(ckpt_path),
                        "--batch-size",
                        str(args.batch_size),
                        "--key-points",
                        str(key_points),
                        "--category",
                        str(category),
                        "--db-path",
                        str(eval_db_path),
                        "--output-dir",
                        str(args.output_dir / f"epoch_{epoch_num_list.pop(0)}"),
                        "--input-type",
                        str(args.input_type),
                    ]
                else:
                    raise SystemExit(f"Unknown script: {args.script}")

                print("→", " ".join(cmd))
                if args.dry_run:
                    print(f"[run_id={rid}] (dry-run) OK")
                    continue

                rc = subprocess.run(cmd).returncode
                if rc != 0:
                    msg = f"[run_id={rid}] script failed ({ckpt_path}) code {rc}"
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

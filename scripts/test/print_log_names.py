#!/usr/bin/env python3
"""
List ckpt_dir values from 'train_runs' table across many SQLite DB files.

Usage:
  python list_ckpt_dirs.py --dir /path/to/db_folder
  python list_ckpt_dirs.py --dir /path/to/db_folder --recursive
  python list_ckpt_dirs.py --dir /path/to/db_folder --unique

Assumes each DB has a table named 'train_runs' with a column 'ckpt_dir'.
"""

import argparse
import sqlite3
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Print train_runs.ckpt_dir from DBs in a folder.")
    p.add_argument("--dir", type=Path, required=True, help="Directory containing .db files")
    p.add_argument("--recursive", action="store_true", help="Search subdirectories for .db files")
    p.add_argument("--unique", action="store_true", help="Print unique ckpt_dir values per folder (deduplicated across DBs)")
    return p.parse_args()


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table,))
    return cur.fetchone() is not None


def column_exists(con: sqlite3.Connection, table: str, column: str) -> bool:
    cur = con.cursor()
    cur.execute(f"PRAGMA table_info({table});")
    cols = [row[1] for row in cur.fetchall()]  # row[1] = column name
    return column in cols


def list_db_files(root: Path, recursive: bool):
    pattern = "**/*.db" if recursive else "*.db"
    return sorted([p for p in root.glob(pattern) if p.is_file()])


def main():
    args = parse_args()

    if not args.dir.is_dir():
        raise SystemExit(f"Not a directory: {args.dir}")

    db_files = list_db_files(args.dir, args.recursive)
    if not db_files:
        raise SystemExit("No .db files found.")

    all_ckpts = set() if args.unique else None

    for db in db_files:
        try:
            con = sqlite3.connect(str(db))
        except Exception as e:
            print(f"[WARN] Cannot open {db}: {e}")
            continue

        try:
            if not table_exists(con, "train_runs"):
                print(f"[INFO] {db}: no 'train_runs' table; skipping.")
                con.close()
                continue
            if not column_exists(con, "train_runs", "ckpt_dir"):
                print(f"[INFO] {db}: 'train_runs' lacks 'ckpt_dir' column; skipping.")
                con.close()
                continue

            cur = con.cursor()
            cur.execute("SELECT ckpt_dir FROM train_runs;")
            rows = cur.fetchall()

            if args.unique:
                for (ckpt_dir,) in rows:
                    if ckpt_dir is not None:
                        all_ckpts.add(str(ckpt_dir))
            else:
                print(f"\n# {db}")
                if not rows:
                    print("(no rows)")
                else:
                    for (ckpt_dir,) in rows:
                        print(str(ckpt_dir))
        except Exception as e:
            print(f"[WARN] Failed to read {db}: {e}")
        finally:
            con.close()

    if args.unique:
        print("\n# UNIQUE ckpt_dir values")
        if not all_ckpts:
            print("(none)")
        else:
            for ck in sorted(all_ckpts):
                print(ck)


if __name__ == "__main__":
    main()

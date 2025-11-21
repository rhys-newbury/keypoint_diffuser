#!/usr/bin/env python3
import argparse
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path


VALID_METRICS = {"das", "fwd", "bwd", "miou_at_0_1"}


def backup_db(db_path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = db_path.with_suffix(db_path.suffix + f".bak.{ts}")
    shutil.copy2(db_path, bak)
    return bak


def delete_dominated_rows(
    con: sqlite3.Connection,
    metric: str,
    per_category: bool,
    model_filter: str | None,
    dry_run: bool,
) -> int:
    """
    First pass:
    Delete rows that are dominated by another row in the same group
    with a strictly larger metric, or equal metric but larger id.
    Also deletes rows whose metric is NULL if any competitor has a non-NULL value.
    """
    col = metric
    where_same = "r2.ckpt = r.ckpt"
    if per_category:
        where_same += " AND r2.category = r.category"
    if model_filter:
        where_same += " AND r2.model = r.model"

    model_clause = "AND r.model = :model" if model_filter else ""

    # Rows to delete:
    #  - r.col is NULL and there exists a competitor with non-NULL
    #  - OR competitor has (col > r.col) or (col == r.col AND id > r.id)
    cond = f"""
        (
            r.{col} IS NULL
            OR EXISTS (
                SELECT 1
                FROM runs AS r2
                WHERE {where_same}
                  AND r2.{col} IS NOT NULL
                  AND (
                        r.{col} IS NULL
                        OR r2.{col} > r.{col}
                        OR (r2.{col} = r.{col} AND r2.id > r.id)
                      )
            )
        )
    """

    params = {"model": model_filter} if model_filter else {}

    if dry_run:
        sql = f"""
            SELECT r.id
            FROM runs AS r
            WHERE 1=1 {model_clause and " " + model_clause}
              AND {cond}
        """
        cur = con.execute(sql, params)
        ids = [row[0] for row in cur.fetchall()]
        return len(ids)

    sql = f"""
        DELETE FROM runs AS r
        WHERE 1=1 {model_clause and " " + model_clause}
          AND {cond}
    """
    cur = con.execute(sql, params)
    return cur.rowcount if cur.rowcount != -1 else 0


def delete_null_ties(
    con: sqlite3.Connection,
    metric: str,
    per_category: bool,
    model_filter: str | None,
    dry_run: bool,
) -> int:
    """
    Second pass (only affects groups where all competitors had NULL metric):
    Keep the largest id per (group), delete the rest.
    """
    col = metric
    where_same = "r2.ckpt = r.ckpt"
    if per_category:
        where_same += " AND r2.category = r.category"
    if model_filter:
        where_same += " AND r2.model = r.model"

    model_clause = "AND r.model = :model" if model_filter else ""
    cond = f"""
        r.{col} IS NULL
        AND EXISTS (
            SELECT 1
            FROM runs AS r2
            WHERE {where_same}
              AND r2.{col} IS NULL
              AND r2.id > r.id
        )
    """
    params = {"model": model_filter} if model_filter else {}

    if dry_run:
        sql = f"""
            SELECT r.id
            FROM runs AS r
            WHERE 1=1 {model_clause and " " + model_clause}
              AND {cond}
        """
        cur = con.execute(sql, params)
        ids = [row[0] for row in cur.fetchall()]
        return len(ids)

    sql = f"""
        DELETE FROM runs AS r
        WHERE 1=1 {model_clause and " " + model_clause}
          AND {cond}
    """
    cur = con.execute(sql, params)
    return cur.rowcount if cur.rowcount != -1 else 0


def main():
    ap = argparse.ArgumentParser(
        description="Deduplicate SQLite 'runs' table by ckpt (keep the larger metric)."
    )
    ap.add_argument("--db", type=Path, required=True, help="Path to results.db")
    ap.add_argument(
        "--metric",
        choices=sorted(VALID_METRICS),
        required=True,
        help="Metric column to compare",
    )
    ap.add_argument(
        "--model",
        choices=["SC3K", "SM"],
        default=None,
        help="Optional: restrict dedupe inside each model",
    )
    ap.add_argument(
        "--global-ckpt",
        action="store_true",
        help="Dedupe by ckpt globally (ignore category). Default is per-category.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Show how many rows would be deleted, without changing the DB.",
    )
    ap.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating a backup .bak.<timestamp> copy before modifying the DB.",
    )
    ap.add_argument(
        "--vacuum",
        action="store_true",
        help="Run VACUUM after deletion to reclaim space.",
    )
    args = ap.parse_args()

    if not args.db.exists():
        raise SystemExit(f"DB not found: {args.db}")

    per_category = not args.global_ckpt

    if not args.dry_run and not args.no_backup:
        bak = backup_db(args.db)
        print(f"[backup] -> {bak}")

    con = sqlite3.connect(str(args.db))
    con.execute("PRAGMA foreign_keys=ON;")

    try:
        con.execute("BEGIN")
        n1 = delete_dominated_rows(
            con, args.metric, per_category, args.model, dry_run=args.dry_run
        )
        n2 = delete_null_ties(
            con, args.metric, per_category, args.model, dry_run=args.dry_run
        )

        if args.dry_run:
            con.rollback()
            scope = "ckpt & category" if per_category else "ckpt (global)"
            if args.model:
                scope += f", model={args.model}"
            print(
                f"[dry-run] would delete {n1 + n2} rows "
                f"(pass1: {n1}, pass2: {n2}) | scope={scope} | metric={args.metric}"
            )
        else:
            con.commit()
            print(f"[✓] deleted {n1 + n2} rows (pass1: {n1}, pass2: {n2})")
            if args.vacuum:
                con.execute("VACUUM")
                print("[✓] VACUUM done")
    finally:
        con.close()


if __name__ == "__main__":
    main()

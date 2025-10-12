import sqlite3
from pathlib import Path
import re


def init_db(db_path: Path):
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    # training info table (what you asked for here)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS train_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            algo TEXT,
            category TEXT,
            ckpt_dir TEXT,
            key_points INTEGER
        )
        """
    )

    con.commit()
    return con


def save_train_run(
    db_path: Path, algo: str, category: str, ckpt_dir: Path, key_points: int
) -> int:
    con = init_db(db_path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO train_runs (algo, category, ckpt_dir, key_points)
        VALUES (?, ?, ?, ?)
    """,
        (str(algo), str(category), str(ckpt_dir), key_points),
    )
    con.commit()
    rid = cur.lastrowid
    con.close()
    return rid


def find_checkpoint_from_db_dir(db_root, category, epoch, also_filter_table_by_category=True):
    """
    Look through .db files in `db_root` whose *filename* contains `category`;
    read `train_runs.ckpt_dir` from each; then verify that directory contains
    a checkpoint whose filename's *last integer* equals `epoch`.
    Return the first matching ckpt_dir (as a Path), or None if nothing matches.

    Assumptions:
      - Table name: train_runs
      - Column: ckpt_dir (text path to a run directory, e.g. .../logs/<run_name>)
      - Checkpoints live in either <ckpt_dir>/checkpoints/*.pth or directly under <ckpt_dir>/*.pth
      - Epoch is parsed as the LAST integer in each checkpoint filename, e.g.:
          net_12.pth     -> epoch 12
          epoch-0010.pth -> epoch 10
          best_final.pth -> (no integer; ignored)

    Params
    ------
    db_root : str | Path
        Directory containing .db files.
    category : str
        Class/category string that must appear in the DB *filename* (case-insensitive).
    epoch : int
        Target epoch number to look for in checkpoint filenames.
    also_filter_table_by_category : bool
        If True, also filter rows by 'category' in the train_runs table when present.

    Returns
    -------
    Path | None
        The ckpt_dir that contains a checkpoint for the requested epoch, or None.
    """
    # import pdb; pdb.set_trace()
    db_root = Path(db_root)
    if not db_root.is_dir():
        raise NotADirectoryError(f"{db_root} is not a directory")

    pattern = "*.db"
    cat_lower = category.lower()

    # 1) candidate DBs: filename contains category token (case-insensitive)
    candidate_dbs = (
        p for p in db_root.glob(pattern)
        if p.is_file() and cat_lower in p.name.lower()
    )

    def table_exists(con: sqlite3.Connection, name: str) -> bool:
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (name,))
        return cur.fetchone() is not None

    def column_exists(con: sqlite3.Connection, table: str, column: str) -> bool:
        cur = con.cursor()
        cur.execute(f"PRAGMA table_info({table});")
        return any(row[1] == column for row in cur.fetchall())

    def extract_last_int(s: str) -> int | None:
        nums = re.findall(r"(\d+)", s)
        return int(nums[-1]) if nums else None

    def scan_ckpt_dir_for_best(ckpt_dir: Path, target_epoch: int):
        """
        Scan ckpt_dir (and ckpt_dir/checkpoints) for *.pth files, pick the one with epoch
        closest to target_epoch. Tie-breaker: prefer the larger epoch.
        """
        if not ckpt_dir.exists():
            return None

        # where to look
        roots = [ckpt_dir, ckpt_dir / "checkpoints"]
        candidates: list[tuple[int, Path]] = []

        for root in roots:
            if not root.exists():
                continue
            for pth in root.glob("*.pth"):
                e = extract_last_int(pth.name)
                if e is not None:
                    candidates.append((e, pth))

        if not candidates:
            return None

        # exact match first
        for e, p in candidates:
            if e == target_epoch:
                return p

        # otherwise pick closest; on tie choose larger epoch
        best_epoch, best_path = min(
            candidates,
            key=lambda ev: (abs(ev[0] - target_epoch), -ev[0])  # tie-breaker: larger epoch
        )
        return best_path

    # 2) iterate DBs → read ckpt_dir rows → pick best checkpoint
    for db in candidate_dbs:
        try:
            con = sqlite3.connect(str(db))
        except Exception:
            continue
        try:
            if not table_exists(con, "train_runs") or not column_exists(con, "train_runs", "ckpt_dir"):
                continue

            cur = con.cursor()
            if also_filter_table_by_category and column_exists(con, "train_runs", "category"):
                cur.execute("SELECT ckpt_dir FROM train_runs WHERE category=?;", (category,))
            else:
                cur.execute("SELECT ckpt_dir FROM train_runs;")

            for (ckpt_dir_str,) in cur.fetchall():
                if not ckpt_dir_str:
                    continue
                ckpt_dir = Path(ckpt_dir_str)
                best = scan_ckpt_dir_for_best(ckpt_dir, epoch)
                if best is not None:
                    return best
        finally:
            con.close()

    # nothing matched
    return None
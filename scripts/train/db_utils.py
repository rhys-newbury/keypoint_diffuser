import sqlite3
from pathlib import Path


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

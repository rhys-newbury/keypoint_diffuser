import os
import re
import sqlite3
import subprocess
import time
import uuid


DB_PATH = "/mnt/slow/jobs.db"
SLEEP_INTERVAL = 10
GPU_ENV = os.environ.get("GPU_TYPE", "3090")


def extract_wandb_name(output):
    match = re.search(r"Run name[:\s]+([a-zA-Z0-9_\-]+)", output)
    return match.group(1) if match else None


def claim_job():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("SELECT * FROM jobs")
        rows = cur.fetchall()

        for row in rows:
            if row["status"] != "pending":
                continue

            gpu_list = row["gpu_type"].split(",")
            if GPU_ENV not in gpu_list:
                continue

            if row["type"] == "test":
                # Only run test if training completed & wandb_name exists
                cur.execute(
                    "SELECT * FROM jobs WHERE id=? AND type='train'", (row["id"],)
                )
                train = cur.fetchone()
                if not train or train["status"] != "done" or not train["wandb_name"]:
                    continue

            # Claim job atomically
            worker_id = str(uuid.uuid4())
            cur.execute(
                "UPDATE jobs SET status='running', worker_id=? WHERE id=? AND type=? AND status='pending'",
                (worker_id, row["id"], row["type"]),
            )
            if cur.rowcount > 0:
                conn.commit()
                row = dict(row)
                row["worker_id"] = worker_id
                return row

    return None


def run_command(cmd):
    try:
        print(f"Running: {cmd}")
        result = subprocess.run(
            cmd, shell=True, text=True, capture_output=True, check=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return False, None


def complete_job(job, success, output):
    with sqlite3.connect(DB_PATH) as conn:
        if success and job["type"] == "train":
            wandb_name = extract_wandb_name(output)
            conn.execute(
                "UPDATE jobs SET status='done', wandb_name=? WHERE id=? AND type=?",
                (wandb_name, job["id"], job["type"]),
            )
        else:
            conn.execute(
                "UPDATE jobs SET status=? WHERE id=? AND type=?",
                ("done" if success else "failed", job["id"], job["type"]),
            )
        conn.commit()


def main():
    while True:
        job = claim_job()
        if job:
            cmd = job["command"]
            if job["type"] == "test" and "$WANDB_NAME" in cmd:
                with sqlite3.connect(DB_PATH) as conn:
                    conn.row_factory = sqlite3.Row
                    train = conn.execute(
                        "SELECT wandb_name FROM jobs WHERE id=? AND type='train'",
                        (job["id"],),
                    ).fetchone()
                    if train:
                        cmd = cmd.replace("$WANDB_NAME", train["wandb_name"])

            success, output = run_command(cmd)
            complete_job(job, success, output)
        else:
            print("No eligible jobs. Sleeping...")
            time.sleep(SLEEP_INTERVAL)


if __name__ == "__main__":
    main()

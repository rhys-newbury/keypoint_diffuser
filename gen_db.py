# Now populate the SQLite database from the recreated JSONL file
import json
import sqlite3


jobs_path = (
    "/run/user/1000/gvfs/smb-share:server=130.194.128.238,share=slow/job_list.jsonl"
)
db_path = "/run/user/1000/gvfs/smb-share:server=130.194.128.238,share=slow/jobs.db"

jobs = []
with open(jobs_path) as f:
    for line in f:
        jobs.append(json.loads(line))


# Initialize SQLite database and insert jobs
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS jobs")

cursor.execute(
    """
CREATE TABLE jobs (
    id TEXT,
    type TEXT,
    status TEXT,
    command TEXT,
    wandb_name TEXT,
    gpu_type TEXT,
    worker_id TEXT,
    PRIMARY KEY (id, type)
)
"""
)

for job in jobs:
    cursor.execute(
        """
        INSERT INTO jobs (id, type, status, command, wandb_name, gpu_type, worker_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        (
            job["id"],
            job["type"],
            job["status"],
            job["command"],
            job.get("wandb_name"),
            ",".join(job["gpu_type"]),
            None,
        ),
    )

conn.commit()
conn.close()

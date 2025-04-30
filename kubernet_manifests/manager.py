import argparse
import fcntl
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

import torch
import yaml


parser = argparse.ArgumentParser()
parser.add_argument(
    "--jobs_path",
    type=str,
    default="/mnt/slow/job_list.yaml",
    help="Path to the job list YAML file.",
)
args = parser.parse_args()

JOBS_PATH = Path(args.jobs_path)
LOCK_PATH = JOBS_PATH.with_suffix(".lock")

SLEEP_INTERVAL = 10


def detect_gpu_type():
    if not torch.cuda.is_available():
        return "CPU"
    raw = torch.cuda.get_device_name(0).strip()
    if not raw.startswith("NVIDIA"):
        raw = "NVIDIA " + raw
    return re.sub(r"\s+", "-", raw)


GPU_ENV = detect_gpu_type()
print(f"[INFO] Detected GPU type: {GPU_ENV}")


def extract_wandb_name(output):
    match = re.search(r"Run name[:\s]+([a-zA-Z0-9_\-]+)", output)
    return match.group(1) if match else None


def load_jobs():
    with open(LOCK_PATH, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not os.path.exists(JOBS_PATH):
            return []
        with open(JOBS_PATH) as f:
            jobs = yaml.safe_load(f) or []
        return jobs


def save_jobs(jobs):
    with open(LOCK_PATH, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with open(JOBS_PATH, "w") as f:
            yaml.dump(jobs, f)


def claim_job():
    jobs = load_jobs()
    for job in jobs:
        if job["status"] != "pending":
            continue
        if GPU_ENV not in job["gpu_type"]:
            continue
        if job["type"] == "test":
            # Find matching train job
            train_job = next(
                (j for j in jobs if j["id"] == job["id"] and j["type"] == "train"), None
            )
            if (
                not train_job
                or train_job["status"] != "done"
                or not train_job.get("wandb_name")
            ):
                continue
        job["status"] = "running"
        job["worker_id"] = str(uuid.uuid4())
        save_jobs(jobs)
        return job
    return None


def complete_job(job, success, output):
    jobs = load_jobs()
    for j in jobs:
        if j["id"] == job["id"] and j["type"] == job["type"]:
            j["status"] = "done" if success else "failed"
            if success and j["type"] == "train":
                wandb_name = extract_wandb_name(output)
                j["wandb_name"] = wandb_name
                print(
                    f"[✓] Training job {j['id']} complete. WANDB run name: {wandb_name or 'N/A'}"
                )
                # propagate to test job
                for t in jobs:
                    if t["id"] == j["id"] and t["type"] == "test":
                        t["wandb_name"] = wandb_name
            break
    save_jobs(jobs)
    print(
        f"[{'✓' if success else '✗'}] {job['type'].capitalize()} job {job['id']} {'completed' if success else 'failed'}."
    )


def run_command(cmd):
    print(f"Running: {cmd}")
    output = b""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # with open(log_path, "wb") as f:
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
    )
    for c in iter(lambda: process.stdout.read(1), b""):
        sys.stdout.buffer.write(c)
        output += c
    process.stdout.close()
    process.wait()
    success = process.returncode == 0
    return success, output.decode(errors="replace")


def main():
    while True:
        job = claim_job()
        if job:
            cmd = job["command"]
            if job["type"] == "test" and "$WANDB_NAME" in cmd and job.get("wandb_name"):
                cmd = cmd.replace("$WANDB_NAME", job["wandb_name"])
            success, output = run_command(cmd)
            complete_job(job, success, output)
        else:
            print("No eligible jobs. Sleeping...")
            time.sleep(SLEEP_INTERVAL)


if __name__ == "__main__":
    main()

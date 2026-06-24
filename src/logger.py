"""
Structured experiment logging.

Every experiment run produces a log entry in JSONL format with:
  - Timestamp, git commit hash, hostname
  - Full experiment configuration
  - Runtime stats (duration, records processed, GPU info)
  - Any errors or warnings

This makes it easy to trace any result back to its exact configuration.

Usage:
    from src.logger import ExperimentLogger
    
    log = ExperimentLogger("judge_inference")
    log.start(config={"model": "aya-expanse-32b", "dataset": "flores200"})
    # ... do work ...
    log.finish(records_processed=500)
"""
import json
import os
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from configs.config import LOGS


class ExperimentLogger:
    """Structured experiment logger with JSONL output."""

    def __init__(self, experiment_name: str, log_dir: Path = None):
        self.name = experiment_name
        self.log_dir = log_dir or LOGS
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{experiment_name}.jsonl"
        self.entry = {}
        self._start_time = None

    def start(self, config: dict = None):
        """Begin logging an experiment run."""
        self._start_time = time.time()
        self.entry = {
            "experiment": self.name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
            "git_commit": self._get_git_commit(),
            "config": config or {},
            "warnings": [],
            "errors": [],
        }

    def warn(self, message: str):
        """Log a warning."""
        self.entry.setdefault("warnings", []).append(message)
        print(f"  ⚠ {message}")

    def error(self, message: str):
        """Log an error."""
        self.entry.setdefault("errors", []).append(message)
        print(f"  ❌ {message}")

    def finish(self, **stats):
        """Finalize and write the log entry."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        self.entry["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.entry["duration_s"] = round(elapsed, 2)
        self.entry["stats"] = stats
        self.entry["gpu_info"] = self._get_gpu_info()

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.entry, ensure_ascii=False, default=str) + "\n")

        print(f"\n📝 Log entry written to {self.log_file}")
        return self.entry

    def _get_git_commit(self) -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"

    def _get_gpu_info(self) -> list:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                return [line.strip() for line in result.stdout.strip().split("\n")]
        except Exception:
            pass
        return []

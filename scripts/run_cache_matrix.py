#!/usr/bin/env python3
"""
Batch driver for cache size × eviction policy experiments.

Features
--------
- Reads a declarative plan (JSON) describing policies and cache sizes.
- Generates per-run configuration files from policy templates.
- Persists run status in a JSON checkpoint file to support resume after failures.
- Emits concise structured logs for quick diagnosis.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_PLAN = Path("runs/eval_evict/experiment_plan.json")
DEFAULT_STATUS = Path("runs/eval_evict/status.json")
DEFAULT_LOG_DIR = Path("runs/eval_evict/logs")
DEFAULT_TMP_DIR = Path("runs/eval_evict/tmp")


def load_json(path: Path) -> dict:
    with path.open() as fp:
        return json.load(fp)


def dump_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as tmp:
        json.dump(payload, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


@dataclass(frozen=True)
class TaskKey:
    policy_id: str
    size_gb: float

    def to_key(self) -> str:
        return f"{self.policy_id}::{self.size_gb}"


def build_tasks(plan: dict) -> List[Tuple[TaskKey, dict]]:
    default_sizes = plan.get("cache_sizes_gb", [])
    if not isinstance(default_sizes, list) or not default_sizes:
        raise ValueError("experiment plan must define a non-empty \"cache_sizes_gb\" list")
    tasks: List[Tuple[TaskKey, dict]] = []
    for policy in plan.get("policies", []):
        policy_id = policy["id"]
        policy_sizes = policy.get("sizes_gb", default_sizes)
        if not policy_sizes:
            raise ValueError(f"policy {policy_id} must have cache sizes")
        for size in policy_sizes:
            key = TaskKey(policy_id=policy_id, size_gb=float(size))
            task_payload = {
                "policy": policy,
                "size_gb": float(size),
            }
            tasks.append((key, task_payload))
    return tasks


def load_status(status_path: Path) -> Dict[str, dict]:
    if not status_path.exists():
        return {}
    data = load_json(status_path)
    return {entry["key"]: entry for entry in data.get("entries", [])}


def persist_status(status_path: Path, entries: Dict[str, dict]) -> None:
    dump_json_atomic(status_path, {"entries": list(entries.values())})


def setup_logger(log_dir: Path, verbose: bool = False) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run_cache_matrix.log"
    level = logging.DEBUG if verbose else logging.INFO
    logger = logging.getLogger("cache-matrix")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    logger.debug("Logger initialised. Output file: %s", log_path)
    return logger


def format_size_dir(size: float) -> str:
    if float(size).is_integer():
        return f"size_{int(size)}"
    return f"size_{size}"


def merge_config(template_path: Path, overrides: dict, size_gb: float, output_dir: Path) -> dict:
    config = load_json(template_path)
    config.update(overrides or {})
    config["size_gb"] = size_gb
    config["output_dir"] = str(output_dir)
    # Always enforce admission/prefetch constraints expected by the assignment.
    config["ap"] = "acceptall"
    config["prefetch_when"] = "never"
    return config


def ensure_directories(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def run_single_task(
    logger: logging.Logger,
    project_root: Path,
    tmp_dir: Path,
    task_key: TaskKey,
    task_payload: dict,
    status_entry: dict,
) -> Tuple[bool, dict]:
    policy = task_payload["policy"]
    size_gb = task_payload["size_gb"]
    template_path = project_root / policy["config"]
    output_base = Path(policy["output_dir"])
    overrides = policy.get("overrides", {})

    size_dir_name = format_size_dir(size_gb)
    run_output_dir = output_base / size_dir_name
    ensure_directories(project_root / run_output_dir, tmp_dir)

    config_data = merge_config(template_path, overrides, size_gb, project_root / run_output_dir)
    tmp_cfg_path = tmp_dir / f"{task_key.policy_id}_{size_dir_name}.json"
    dump_json_atomic(tmp_cfg_path, config_data)

    log_file = project_root / "runs" / "eval_evict" / "logs" / f"{task_key.policy_id}_{size_dir_name}.log"
    with log_file.open("w", encoding="utf-8") as log_fp:
        log_fp.write(f"# Command executed at {datetime.utcnow().isoformat()}Z\n")
        log_fp.flush()

    cmd = [
        sys.executable,
        "-m",
        "BCacheSim.cachesim.simulate_ap",
        "--config",
        str(tmp_cfg_path),
        "--ignore-existing",
    ]
    logger.info("Starting %s (size %.2f GiB)", task_key.policy_id, size_gb)
    logger.debug("Command: %s", " ".join(cmd))

    started_at = datetime.utcnow().isoformat() + "Z"
    status_entry.update(
        status="in_progress",
        started_at=started_at,
        finished_at=None,
        output_dir=str(run_output_dir),
        command=" ".join(cmd),
        log_file=str(log_file),
    )

    with log_file.open("a", encoding="utf-8") as log_fp:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
        )

    finished_at = datetime.utcnow().isoformat() + "Z"
    status_entry["finished_at"] = finished_at
    status_entry["returncode"] = result.returncode

    if result.returncode == 0:
        status_entry["status"] = "completed"
        status_entry["error"] = None
        logger.info("Completed %s (size %.2f GiB)", task_key.policy_id, size_gb)
        success = True
    else:
        status_entry["status"] = "failed"
        status_entry["error"] = f"return code {result.returncode}"
        logger.error(
            "FAILED %s (size %.2f GiB) – see %s",
            task_key.policy_id,
            size_gb,
            log_file,
        )
        success = False

    # Remove temporary config only if run finished successfully to aid debugging.
    if success:
        try:
            tmp_cfg_path.unlink()
        except OSError:
            logger.debug("Failed to delete temp config %s", tmp_cfg_path)

    return success, status_entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch runner for cache policy experiments.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN, help="Path to experiment plan JSON.")
    parser.add_argument(
        "--status-file",
        type=Path,
        default=DEFAULT_STATUS,
        help="Checkpoint file used for resume.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Do not stop after a failed run; attempt remaining combinations.",
    )
    parser.add_argument(
        "--force-retry",
        action="store_true",
        help="Re-run combinations even if marked completed.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit debug logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    tmp_dir = DEFAULT_TMP_DIR
    ensure_directories(DEFAULT_LOG_DIR, tmp_dir)
    logger = setup_logger(DEFAULT_LOG_DIR, verbose=args.verbose)

    plan = load_json(args.plan)
    tasks = build_tasks(plan)
    status_entries = load_status(args.status_file)

    logger.info("Loaded plan from %s (%d combinations)", args.plan, len(tasks))

    for task_key, task_payload in tasks:
        entry_key = task_key.to_key()
        entry = status_entries.get(entry_key, {})
        previous_status = entry.get("status")

        if previous_status == "completed" and not args.force_retry:
            logger.info("Skipping %s (size %.2f GiB) – already completed", task_key.policy_id, task_key.size_gb)
            continue

        entry.update(
            key=entry_key,
            policy_id=task_key.policy_id,
            size_gb=task_key.size_gb,
        )
        status_entries[entry_key] = entry
        persist_status(args.status_file, status_entries)

        success, updated_entry = run_single_task(
            logger=logger,
            project_root=project_root,
            tmp_dir=tmp_dir,
            task_key=task_key,
            task_payload=task_payload,
            status_entry=entry,
        )
        status_entries[entry_key] = updated_entry
        persist_status(args.status_file, status_entries)

        if not success and not args.continue_on_error:
            logger.error("Halting due to failure (use --continue-on-error to proceed with remaining runs)")
            sys.exit(1)

    logger.info("All requested combinations processed.")


if __name__ == "__main__":
    main()

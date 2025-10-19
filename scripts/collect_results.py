#!/usr/bin/env python3
"""
Aggregate simulator outputs into a single CSV for plotting.

Scans runs/eval_evict/results/**/size_*/..._cache_perf.txt.lzma and emits
summaries containing peak service time, hit rate, flash write metrics, etc.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import compress_json


DEFAULT_ROOT = Path("runs/eval_evict/results")
DEFAULT_OUTPUT = Path("runs/eval_evict/summary.csv")


@dataclass
class Record:
    policy: str
    size_gb: float
    trace: str
    peak_dt_total: float
    peak_dt_get: float
    p99_dt_total: float
    p99_dt_get: float
    hit_rate: float
    flash_write_rate: float
    flash_write_total: float
    client_bandwidth: float
    backend_bandwidth: float
    cache_elems: int
    source_file: str


def extract_metrics(path: Path, policy: str, size_gb: float) -> Optional[Record]:
    data = compress_json.load(str(path))
    results = data.get("results", {})
    options = data.get("options", {})
    trace = data.get("trace_kwargs", {}).get("region", data.get("trace_kwargs", {}).get("trace", ""))

    def grab(key: str, default=None):
        return results.get(key, default)

    record = Record(
        policy=policy,
        size_gb=size_gb,
        trace=str(trace),
        peak_dt_total=grab("PeakServiceTimeUsedWithPutUtil1", 0.0),
        peak_dt_get=grab("PeakServiceTimeUtil1", 0.0),
        p99_dt_total=grab("P99ServiceTimeWithPutUtil1", 0.0),
        p99_dt_get=grab("P99ServiceTimeUtil1", 0.0),
        hit_rate=grab("ChunkHitRatio", 0.0),
        flash_write_rate=grab("FlashWriteRate", 0.0),
        flash_write_total=grab("FlashChunkWritten", 0.0),
        client_bandwidth=grab("ClientBandwidth", 0.0),
        backend_bandwidth=grab("BackendBandwidth", 0.0),
        cache_elems=results.get("NumCacheElems", options.get("cache_elems", 0)),
        source_file=str(path),
    )
    return record


def find_result_files(root: Path) -> Iterable[Path]:
    pattern = "*/*/*/*_cache_perf.txt.lzma"
    for path in root.glob(pattern):
        yield path


def parse_size_from_path(path: Path) -> float:
    try:
        size_dir = path.parent.parent.name  # size_256
        return float(size_dir.split("_", 1)[1])
    except Exception:
        return 0.0


def parse_policy_from_path(path: Path) -> str:
    return path.parent.parent.parent.name


def collect_records(root: Path) -> List[Record]:
    records: List[Record] = []
    for file_path in find_result_files(root):
        policy = parse_policy_from_path(file_path)
        size = parse_size_from_path(file_path)
        record = extract_metrics(file_path, policy, size)
        if record:
            records.append(record)
    records.sort(key=lambda r: (r.policy, r.size_gb))
    return records


def write_csv(records: List[Record], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(asdict(records[0]).keys()) if records else [])
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate cache simulator results into CSV.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Root directory containing results.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="CSV output path.")
    args = parser.parse_args()

    records = collect_records(args.root)
    if not records:
        print(f"No results found under {args.root}")
        return
    write_csv(records, args.output)
    print(f"Wrote {len(records)} rows to {args.output}")


if __name__ == "__main__":
    main()

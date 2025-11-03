#!/usr/bin/env python3
"""
Sweep the EDE atti (ede_alpha) parameter, run simulations, aggregate Peak DT,
and produce plots for the ablation study.

Default sweep values: 0.1, 0.3, 0.5, 0.7, 0.9.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List

import compress_json
import matplotlib.pyplot as plt


DEFAULT_ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]
BASE_CONFIG = Path("runs/eval_evict/configs/e2_ede.json")
RESULT_ROOT = Path("runs/eval_evict/results/e2_alpha")
LOG_ROOT = Path("runs/eval_evict/logs")
TMP_ROOT = Path("runs/eval_evict/tmp")
FIG_ROOT = Path("runs/eval_evict/figures")
SUMMARY_PATH = Path("runs/eval_evict/alpha_summary.csv")


@dataclass
class RunRecord:
    ede_alpha: float
    size_gb: float
    peak_dt_total: float
    peak_dt_get: float
    hit_rate: float
    flash_write_rate: float
    result_file: str


def load_config(path: Path) -> dict:
    return compress_json.load(str(path)) if path.suffix in {".lzma", ".bz2", ".xz"} else path.read_text()


def load_json(path: Path) -> dict:
    with path.open() as fp:
        import json
        return json.load(fp)


def save_json(payload: dict, path: Path) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        json.dump(payload, fp, indent=2)


def run_simulation(config: dict, alpha: float, size_gb: float, *, log_file: Path) -> Path:
    label = f"alpha_{alpha:.1f}"
    output_dir = RESULT_ROOT / label
    output_dir.mkdir(parents=True, exist_ok=True)

    config = dict(config)
    config["ede_alpha"] = alpha
    config["size_gb"] = size_gb
    config["output_dir"] = str(output_dir)

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", suffix=f"_{label}.json", dir=TMP_ROOT, delete=False) as tmp:
        save_json(config, Path(tmp.name))
        tmp_path = Path(tmp.name)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "BCacheSim.cachesim.simulate_ap",
        "--config",
        str(tmp_path),
        "--ignore-existing",
    ]

    with log_file.open("w") as lf:
        lf.write(f"# Command: {' '.join(cmd)}\n")
    with log_file.open("a") as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=True)

    tmp_path.unlink(missing_ok=True)

    result_files = sorted(output_dir.glob("**/*_cache_perf.txt.lzma"))
    if not result_files:
        raise FileNotFoundError(f"No result file found in {output_dir}")
    return result_files[0]


def collect_metrics(result_path: Path, alpha: float, size_gb: float) -> RunRecord:
    data = compress_json.load(str(result_path))
    results = data["results"]
    return RunRecord(
        ede_alpha=alpha,
        size_gb=size_gb,
        peak_dt_total=results.get("PeakServiceTimeUsedWithPutUtil1"),
        peak_dt_get=results.get("PeakServiceTimeUtil1"),
        hit_rate=results.get("ChunkHitRatio"),
        flash_write_rate=results.get("FlashWriteRate"),
        result_file=str(result_path),
    )


def write_summary(records: List[RunRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def plot_records(records: List[RunRecord], alpha_label: str, fig_root: Path) -> None:
    fig_root.mkdir(parents=True, exist_ok=True)
    alphas = [r.ede_alpha for r in records]
    peak_dt = [r.peak_dt_total for r in records]
    hit_rate = [r.hit_rate for r in records]

    plt.figure(figsize=(6, 4))
    plt.plot(alphas, peak_dt, marker="o")
    plt.title("Peak DT vs atti (EDE)")
    plt.xlabel("atti (ede_alpha)")
    plt.ylabel("Peak DT (total, % utilisation)")
    plt.grid(True, linestyle="--", alpha=0.4)
    peak_fig = fig_root / "peak_dt_vs_alpha.png"
    plt.tight_layout()
    plt.savefig(peak_fig, dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(alphas, hit_rate, marker="o")
    plt.title("Chunk Hit Rate vs atti (EDE)")
    plt.xlabel("atti (ede_alpha)")
    plt.ylabel("Chunk Hit Rate")
    plt.grid(True, linestyle="--", alpha=0.4)
    hit_fig = fig_root / "hit_rate_vs_alpha.png"
    plt.tight_layout()
    plt.savefig(hit_fig, dpi=200)
    plt.close()

    print(f"Saved figures:\n - {peak_fig}\n - {hit_fig}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep ede_alpha values for EDE.")
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS, help="Values of ede_alpha to test.")
    parser.add_argument("--size", type=float, default=400.0, help="Cache size (GB) used in each run.")
    args = parser.parse_args()

    base_config = load_json(BASE_CONFIG)
    records: List[RunRecord] = []

    for alpha in args.alphas:
        label = f"alpha_{alpha:.1f}"
        log_file = LOG_ROOT / f"e2_alpha_{label}.log"
        print(f"[Sweep] ede_alpha={alpha:.2f}, log -> {log_file}")
        result_path = run_simulation(base_config, alpha, args.size, log_file=log_file)
        records.append(collect_metrics(result_path, alpha, args.size))

    records.sort(key=lambda r: r.ede_alpha)
    write_summary(records, SUMMARY_PATH)
    plot_records(records, "ede_alpha", FIG_ROOT)
    print(f"Wrote summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

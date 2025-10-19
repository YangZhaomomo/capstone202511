#!/usr/bin/env python3
"""
Plot summary metrics from runs/eval_evict/summary.csv.

Generates line charts for:
1. Peak DT (total) vs cache size.
2. Chunk hit rate vs cache size.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_SUMMARY = Path("runs/eval_evict/summary.csv")
DEFAULT_OUTPUT_DIR = Path("runs/eval_evict/figures")


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_peak_dt(df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    for policy, group in df.groupby("policy"):
        group = group.sort_values("size_gb")
        ax.plot(group["size_gb"], group["peak_dt_total"], marker="o", label=policy)
    ax.set_title("Peak DT vs Cache Size")
    ax.set_xlabel("Cache Size (GB)")
    ax.set_ylabel("Peak DT (total, % utilisation)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    output_path = output_dir / "peak_dt_vs_cache_size.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_hit_rate(df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    for policy, group in df.groupby("policy"):
        group = group.sort_values("size_gb")
        ax.plot(group["size_gb"], group["hit_rate"], marker="o", label=policy)
    ax.set_title("Chunk Hit Rate vs Cache Size")
    ax.set_xlabel("Cache Size (GB)")
    ax.set_ylabel("Chunk Hit Rate")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    output_path = output_dir / "hit_rate_vs_cache_size.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot evaluation metrics from summary CSV.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="Path to summary CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for output figures.")
    args = parser.parse_args()

    ensure_output_dir(args.output_dir)
    df = pd.read_csv(args.summary)
    if df.empty:
        raise SystemExit(f"No rows found in {args.summary}. Run collect_results.py first.")

    outputs = []
    outputs.append(plot_peak_dt(df, args.output_dir))
    outputs.append(plot_hit_rate(df, args.output_dir))

    print("Generated figures:")
    for path in outputs:
        print(f" - {path}")


if __name__ == "__main__":
    main()

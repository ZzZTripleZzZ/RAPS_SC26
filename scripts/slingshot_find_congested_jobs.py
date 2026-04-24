"""
Crawl Frontier slingshot telemetry directories and rank jobs by congestion intensity.

Usage:
    python find_congested_jobs.py /path/to/slingshot_data
    python find_congested_jobs.py /path/to/slingshot_data --date 2025_08_23
    python find_congested_jobs.py /path/to/slingshot_data --top 20 --out results.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


METRICS = {
    "rxCongestion": ("slingshot_rxCongestion", "rxCongestion_cassini.parquet"),
    "txBW":         ("slingshot_txBW",          "txBW_cassini.parquet"),
    "rxBW":         ("slingshot_rxBW",           "rxBW_cassini.parquet"),
    "idle":         ("slingshot_idle",            "idle_cassini.parquet"),
}


def job_id_from_dir(d: Path) -> str:
    """Extract numeric job ID from directory name like '3691196_AGMNPOEJMA'."""
    return d.name.split("_")[0]


def congestion_delta(df: pd.DataFrame) -> dict:
    """
    For a rxCongestion parquet, compute per-port total counter increment
    (last non-NaN minus first non-NaN).  Returns summary stats across all ports.
    """
    node_cols = [c for c in df.columns if c != "Timestamp"]
    if not node_cols:
        return {"max_delta": 0.0, "sum_delta": 0.0, "n_ports": 0, "n_nodes": 0}

    deltas = []
    for col in node_cols:
        series = df[col].dropna()
        if len(series) >= 2:
            delta = series.iloc[-1] - series.iloc[0]
            if delta > 0:
                deltas.append(delta)

    n_nodes = len(set(c[:-2] for c in node_cols))  # strip hX suffix
    return {
        "max_delta":  float(max(deltas)) if deltas else 0.0,
        "sum_delta":  float(sum(deltas)) if deltas else 0.0,
        "n_congested_ports": len(deltas),
        "n_ports":    len(node_cols),
        "n_nodes":    n_nodes,
    }


def bandwidth_stats(df: pd.DataFrame) -> dict:
    """
    For a txBW or rxBW parquet, compute mean and peak per-node bandwidth
    (bytes/s, treating NaN as zero).
    """
    node_cols = [c for c in df.columns if c != "Timestamp"]
    if not node_cols:
        return {"mean_bw": 0.0, "peak_bw": 0.0, "n_nodes": 0}

    vals = df[node_cols].fillna(0).values
    n_nodes = len(set(c[:-2] for c in node_cols))
    return {
        "mean_bw": float(np.nanmean(vals)) if vals.size else 0.0,
        "peak_bw": float(np.nanmax(vals)) if vals.size else 0.0,
        "n_nodes": n_nodes,
    }


def scan_date_dir(date_dir: Path, metric_name: str, parquet_name: str) -> list[dict]:
    """Scan all job subdirectories under a single date directory."""
    rows = []
    job_dirs = sorted(date_dir.iterdir())
    for job_dir in job_dirs:
        if not job_dir.is_dir():
            continue
        parquet_path = job_dir / parquet_name
        if not parquet_path.exists():
            continue

        job_id = job_id_from_dir(job_dir)
        job_name = "_".join(job_dir.name.split("_")[1:])

        try:
            df = pd.read_parquet(parquet_path, engine="pyarrow")
            if "Timestamp" not in df.columns:
                df = df.reset_index()

            n_timestamps = len(df)
            t_start = str(df["Timestamp"].min())[:19] if n_timestamps else ""
            t_end   = str(df["Timestamp"].max())[:19] if n_timestamps else ""

            if metric_name == "rxCongestion":
                stats = congestion_delta(df)
            else:
                stats = bandwidth_stats(df)

            rows.append({
                "job_id":    job_id,
                "job_name":  job_name,
                "date":      date_dir.name,
                "t_start":   t_start,
                "t_end":     t_end,
                "n_timestamps": n_timestamps,
                **stats,
            })
        except Exception as e:
            rows.append({
                "job_id":   job_id,
                "job_name": job_name,
                "date":     date_dir.name,
                "error":    str(e),
            })

    return rows


def find_date_dirs(data_root: Path, metric_dir: str, date_filter: str | None) -> list[Path]:
    metric_path = data_root / metric_dir
    if not metric_path.exists():
        print(f"[WARN] {metric_path} not found", file=sys.stderr)
        return []

    date_dirs = []
    for month_dir in sorted(metric_path.iterdir()):
        if not month_dir.is_dir():
            continue
        for date_dir in sorted(month_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            if date_filter and date_dir.name != date_filter:
                continue
            date_dirs.append(date_dir)
    return date_dirs


def main():
    parser = argparse.ArgumentParser(description="Find congested/high-BW Frontier jobs")
    parser.add_argument("data_root", help="Root directory containing slingshot_* folders")
    parser.add_argument("--date",    help="Filter to specific date, e.g. 2025_08_23")
    parser.add_argument("--metric",  default="rxCongestion",
                        choices=list(METRICS.keys()),
                        help="Metric to rank by (default: rxCongestion)")
    parser.add_argument("--top",     type=int, default=20, help="Show top N jobs")
    parser.add_argument("--out",     help="Save full results to CSV path")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    metric_dir, parquet_name = METRICS[args.metric]

    date_dirs = find_date_dirs(data_root, metric_dir, args.date)
    if not date_dirs:
        print("No matching date directories found.")
        sys.exit(1)

    all_rows = []
    for date_dir in tqdm(date_dirs, desc="Scanning dates"):
        rows = scan_date_dir(date_dir, args.metric, parquet_name)
        all_rows.extend(rows)

    if not all_rows:
        print("No data found.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)

    # Sort by the primary signal for each metric
    sort_col = "max_delta" if args.metric == "rxCongestion" else "peak_bw"
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False)

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Full results saved to {args.out}")

    # Print top N
    top = df.head(args.top)
    print(f"\nTop {args.top} jobs by {sort_col} ({args.metric}):\n")
    display_cols = ["job_id", "job_name", "date", "t_start", "t_end"]
    if args.metric == "rxCongestion":
        display_cols += ["max_delta", "sum_delta", "n_congested_ports", "n_ports", "n_nodes"]
    else:
        display_cols += ["peak_bw", "mean_bw", "n_nodes"]
    display_cols = [c for c in display_cols if c in top.columns]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.3e}".format)
    print(top[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()

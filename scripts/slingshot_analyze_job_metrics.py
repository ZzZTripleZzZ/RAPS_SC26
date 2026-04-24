"""
Load all four Slingshot metrics for a single job and produce validation plots or a CSV summary.

Usage:
    DATA=/lustre/orion/stf218/proj-shared/data/lake/frontier-data-campaign-2026/frontier-interconnect-fabric-telemetry

    # Full 4-subplot figure (saved to file)
    python scripts/analyze_job_metrics.py $DATA --job-id 3691034 --date 2025_08_23 --out results/

    # One-line CSV summary (no plot) — good for comparing many jobs
    python scripts/analyze_job_metrics.py $DATA --job-id 3691034 --date 2025_08_23 --summary

    # Append CSV rows from multiple jobs into a single file
    for jid in 3691034 3688454 3691160 3688392; do
        python scripts/analyze_job_metrics.py $DATA --job-id $jid --date 2025_08_23 --csv results_congestion/summary.csv
    done
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = {
    "rxCongestion": ("slingshot_rxCongestion", "rxCongestion_cassini.parquet"),
    "txBW":         ("slingshot_txBW",          "txBW_cassini.parquet"),
    "rxBW":         ("slingshot_rxBW",           "rxBW_cassini.parquet"),
    "idle":         ("slingshot_idle",            "idle_cassini.parquet"),
}


def find_job_dir(data_root: Path, metric_dir: str, job_id: str, date_filter: str | None) -> Path | None:
    """Find the job directory for a given job_id within a metric tree."""
    metric_path = data_root / metric_dir
    if not metric_path.exists():
        return None
    for month_dir in sorted(metric_path.iterdir()):
        if not month_dir.is_dir():
            continue
        for date_dir in sorted(month_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            if date_filter and date_dir.name != date_filter:
                continue
            for job_dir in date_dir.iterdir():
                if job_dir.is_dir() and job_dir.name.split("_")[0] == job_id:
                    return job_dir
    return None


def load_parquet(job_dir: Path, parquet_name: str) -> pd.DataFrame:
    df = pd.read_parquet(job_dir / parquet_name, engine="pyarrow")
    if "Timestamp" not in df.columns:
        df = df.reset_index()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df.sort_values("Timestamp").reset_index(drop=True)


def node_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "Timestamp"]


def aggregate(df: pd.DataFrame, metric: str) -> pd.Series:
    """Reduce per-port dataframe to a single per-timestamp scalar."""
    cols = node_cols(df)
    if metric == "idle":
        # mean idle % across all ports (NaN = port not reporting, skip)
        return df[cols].mean(axis=1, skipna=True)
    else:
        # sum bandwidth ports, NaN -> 0
        return df[cols].fillna(0).sum(axis=1)


def compute_interval(timestamps: pd.Series) -> pd.Series:
    """Return per-row interval in seconds (forward difference, last row copies prev)."""
    dt = timestamps.diff().shift(-1).dt.total_seconds()
    dt = dt.copy()
    dt.iloc[-1] = dt.iloc[-2] if len(dt) > 1 else 60.0
    return dt.clip(lower=1.0)


def main():
    parser = argparse.ArgumentParser(description="Analyze all Slingshot metrics for one job")
    parser.add_argument("data_root", help="Root directory containing slingshot_* folders")
    parser.add_argument("--job-id",  required=True, help="Numeric job ID, e.g. 3691034")
    parser.add_argument("--date",    help="Filter to specific date dir, e.g. 2025_08_23")
    parser.add_argument("--out",     help="Directory to save plots (default: show interactively)")
    parser.add_argument("--summary", action="store_true",
                        help="Print one CSV row of peak/mean stats and skip plots. "
                             "Run on multiple jobs and concatenate for a comparison table.")
    parser.add_argument("--csv",     help="Append the CSV summary row to this file (implies --summary)")
    args = parser.parse_args()
    if args.csv:
        args.summary = True

    data_root = Path(args.data_root)
    out_dir   = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load all four metrics ---
    raw: dict[str, pd.DataFrame] = {}
    for name, (metric_dir, parquet_name) in METRICS.items():
        job_dir = find_job_dir(data_root, metric_dir, args.job_id, args.date)
        if job_dir is None:
            print(f"[WARN] {name}: job directory not found", file=sys.stderr)
            continue
        parquet_path = job_dir / parquet_name
        if not parquet_path.exists():
            print(f"[WARN] {name}: parquet not found at {parquet_path}", file=sys.stderr)
            continue
        raw[name] = load_parquet(job_dir, parquet_name)
        n_nodes = len(set(c[:-2] for c in node_cols(raw[name])))
        print(f"  {name:15s}: {len(raw[name])} timestamps, {len(node_cols(raw[name]))} ports, {n_nodes} nodes")

    if len(raw) < 2:
        print("Not enough metrics loaded — check data_root and job-id.", file=sys.stderr)
        sys.exit(1)

    # --- Aggregate each metric to a per-timestamp scalar series ---
    agg: dict[str, pd.Series] = {}
    for name, df in raw.items():
        agg[name] = aggregate(df, name).rename(name)

    # --- Build aligned dataframe (inner join on Timestamp) ---
    ts_frames = [raw[n][["Timestamp"]].assign(**{n: agg[n]}) for n in agg]
    merged = ts_frames[0]
    for f in ts_frames[1:]:
        merged = merged.merge(f, on="Timestamp", how="inner")
    merged = merged.sort_values("Timestamp").reset_index(drop=True)
    print(f"\n  {len(merged)} timestamps after aligning all metrics")

    if len(merged) < 3:
        print("Too few aligned timestamps for analysis.", file=sys.stderr)
        sys.exit(1)

    # --- Derived quantities ---
    interval = compute_interval(merged["Timestamp"])  # seconds per sample

    # Utilization: 1 - idle_fraction  (idle is in %)
    if "idle" in merged.columns:
        merged["utilization"] = (100.0 - merged["idle"]) / 100.0

    # Congestion rate: diff of cumulative counter / interval  (bytes/s of congestion)
    if "rxCongestion" in merged.columns:
        cong_diff = merged["rxCongestion"].diff().clip(lower=0)  # drop apparent decreases
        merged["cong_rate"] = cong_diff / interval

        # Congestion ratio: congestion_rate / rxBW  (dimensionless, ~ stall_ratio in RAPS)
        if "rxBW" in merged.columns:
            merged["cong_ratio"] = np.where(
                merged["rxBW"] > 0,
                merged["cong_rate"] / merged["rxBW"],
                np.nan,
            )

    t0 = merged["Timestamp"].iloc[0]
    merged["elapsed_min"] = (merged["Timestamp"] - t0).dt.total_seconds() / 60.0

    # --- Compute per-job summary stats ---
    n_nodes_loaded = max(
        len(set(c[:-2] for c in node_cols(df))) for df in raw.values()
    )
    row: dict = {
        "job_id":        args.job_id,
        "n_nodes":       n_nodes_loaded,
        "n_timestamps":  len(merged),
        "duration_min":  round(merged["elapsed_min"].max(), 1),
    }
    if "utilization" in merged.columns:
        row["peak_util"]  = merged["utilization"].max()
        row["mean_util"]  = merged["utilization"].mean()
    if "rxBW" in merged.columns:
        row["peak_rxBW_GBs_per_node"] = merged["rxBW"].max() / 1e9 / n_nodes_loaded
        row["mean_rxBW_GBs_per_node"] = merged["rxBW"].mean() / 1e9 / n_nodes_loaded
    if "txBW" in merged.columns:
        row["peak_txBW_GBs_per_node"] = merged["txBW"].max() / 1e9 / n_nodes_loaded
    if "cong_rate" in merged.columns:
        row["peak_cong_rate_GBs_per_node"] = merged["cong_rate"].max() / 1e9 / n_nodes_loaded
        congested = merged["cong_rate"] > 0
        row["frac_congested"] = round(congested.mean(), 3)
        row["cong_onset_util"] = (
            merged.loc[congested, "utilization"].mean()
            if congested.any() and "utilization" in merged.columns
            else float("nan")
        )

    if args.summary:
        # One CSV line per job — pipe multiple jobs together for a comparison table
        import csv, io
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
        lines = buf.getvalue().splitlines()
        print(lines[0])   # header
        print(lines[1])   # data row

        if args.csv:
            csv_path = Path(args.csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not csv_path.exists() or csv_path.stat().st_size == 0
            with csv_path.open("a", newline="") as fh:
                writer2 = csv.DictWriter(fh, fieldnames=list(row.keys()))
                if write_header:
                    writer2.writeheader()
                writer2.writerow(row)
            print(f"  → appended to {csv_path}", file=sys.stderr)
    else:
        # Verbose describe() table (original behaviour)
        print(f"\nJob {args.job_id} summary:")
        summary_cols = ["elapsed_min"]
        for col in ("utilization", "rxBW", "txBW", "cong_rate", "cong_ratio"):
            if col in merged.columns:
                summary_cols.append(col)
        pd.set_option("display.float_format", "{:.3e}".format)
        print(merged[summary_cols].describe().to_string())

    if args.summary:
        return

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Job {args.job_id}  —  Slingshot multi-metric analysis", fontsize=13)

    # 1. Time series: utilization
    ax = axes[0, 0]
    if "utilization" in merged.columns:
        ax.plot(merged["elapsed_min"], merged["utilization"], color="steelblue")
        ax.set_ylabel("Link utilization (1 - idle)")
        ax.set_ylim(0, 1)
    ax.set_xlabel("Elapsed (min)")
    ax.set_title("Utilization over time")
    ax.grid(True, alpha=0.3)

    # 2. Time series: rx/tx bandwidth
    ax = axes[0, 1]
    GB = 1e9
    if "rxBW" in merged.columns:
        ax.plot(merged["elapsed_min"], merged["rxBW"] / GB, label="rxBW", color="steelblue")
    if "txBW" in merged.columns:
        ax.plot(merged["elapsed_min"], merged["txBW"] / GB, label="txBW", color="darkorange", linestyle="--")
    ax.set_ylabel("Bandwidth (GB/s, sum all ports)")
    ax.set_xlabel("Elapsed (min)")
    ax.set_title("Bandwidth over time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Time series: congestion rate
    ax = axes[1, 0]
    if "cong_rate" in merged.columns:
        ax.plot(merged["elapsed_min"], merged["cong_rate"] / GB, color="crimson")
        ax.set_ylabel("Congestion rate (GB/s increment)")
    ax.set_xlabel("Elapsed (min)")
    ax.set_title("rxCongestion rate over time")
    ax.grid(True, alpha=0.3)

    # 4. Scatter: utilization vs congestion_ratio (the key RAPS validation plot)
    ax = axes[1, 1]
    if "utilization" in merged.columns and "cong_ratio" in merged.columns:
        valid = merged.dropna(subset=["utilization", "cong_ratio"])
        sc = ax.scatter(
            valid["utilization"],
            valid["cong_ratio"],
            c=valid["rxBW"] / GB if "rxBW" in valid.columns else "steelblue",
            cmap="plasma",
            alpha=0.7,
            s=40,
        )
        if "rxBW" in valid.columns:
            cb = fig.colorbar(sc, ax=ax)
            cb.set_label("rxBW (GB/s)")
        ax.set_xlabel("Link utilization")
        ax.set_ylabel("Congestion ratio (cong_rate / rxBW)")
        ax.set_title("Utilization vs congestion ratio\n(compare to RAPS stall_ratio)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if out_dir:
        out_path = out_dir / f"job_{args.job_id}_metrics.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

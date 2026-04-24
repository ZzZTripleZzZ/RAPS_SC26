#!/usr/bin/env python3
"""
Extract Blue Waters job schedule + network traffic for Layer 3 validation.

Parses torque_logs + cray_system_sampler for a given day:
  1. Extract all completed jobs with node allocation (exec_host)
  2. For each job, compute total tx/rx bytes from Cray sampler
  3. Compute system-wide concurrency (how many nodes busy at each time)
  4. Output: per-job CSV with timing, nodes, traffic, concurrency

Usage:
    cd /lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26
    .venv/bin/python3 Baseline/hardware/extract_bw_jobs.py --day 20170328
"""

import argparse
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

BW_DATA = "/lustre/orion/gen053/scratch/zhangzifan/bluewaters_data"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Default target apps for slowdown validation
# Use --apps flag to override (see argparse below)
TARGET_SPECS = [
    ("NAMD", 64),
    ("NAMD", 32),
    ("NAMD", 128),
    ("scsw_xy_2x", 160),
    ("scsw_xy", 80),   # ~658 runs in May 2017 — more statistics
    ("scsw", 80),      # ~661 runs in May 2017 — neighbor-heavy stencil
]

# Name normalization: raw job name prefix → canonical name used in output.
# Jobs matching a prefix are renamed so all variants appear as one app.
NAME_NORMALIZE = {
    "lmp_f_": ("LAMMPS", 96),   # lmp_f_00_*, lmp_f_30_* → LAMMPS n=96
}

BW_TOTAL_XE_NODES = 22640  # Blue Waters XE nodes


# ── Torque log parser ────────────────────────────────────────────
def parse_torque_day(day: str):
    """Parse one day of torque logs, return list of completed job dicts."""
    fp = os.path.join(BW_DATA, "torque_logs", day)
    if not os.path.exists(fp):
        raise FileNotFoundError(f"No torque log for {day}: {fp}")

    pats = {
        "id": re.compile(r";E;([^;]+);"),
        "name": re.compile(r"\bjobname=(\S+)"),
        "nodes_required": re.compile(r"\bunique_node_count=(\d+)"),
        "exec_host": re.compile(r"\bexec_host=(\S+)"),
        "start": re.compile(r"\bstart=(\d+)"),
        "end": re.compile(r"\bend=(\d+)"),
        "submit": re.compile(r"\bqtime=(\d+)"),
        "wall_req": re.compile(r"Resource_List\.walltime=(\d+:\d{2}:\d{2})"),
        "wall_used": re.compile(r"resources_used\.walltime=(\d+:\d{2}:\d{2})"),
        "queue": re.compile(r"\bqueue=(\S+)"),
        "exit_status": re.compile(r"\bExit_status=(\d+)"),
    }

    jobs = []
    with open(fp) as f:
        for line in f:
            if ";E;" not in line:
                continue
            rec = {}
            for key, pat in pats.items():
                m = pat.search(line)
                if m:
                    rec[key] = m.group(1)

            if not (rec.get("start") and rec.get("end") and rec.get("exec_host")):
                continue

            # Extract node IDs from exec_host
            node_ids = set()
            for token in rec["exec_host"].split("+"):
                nid = token.split("/")[0]
                try:
                    node_ids.add(int(nid))
                except ValueError:
                    pass

            if not node_ids:
                continue

            start = int(rec["start"])
            end = int(rec["end"])
            duration = end - start
            if duration <= 0:
                continue

            jobs.append({
                "job_id": rec.get("id", ""),
                "name": rec.get("name", ""),
                "queue": rec.get("queue", ""),
                "n_nodes": len(node_ids),
                "node_ids": sorted(node_ids),
                "start": start,
                "end": end,
                "duration_s": duration,
                "submit": int(rec.get("submit", start)),
                "wall_req": rec.get("wall_req", ""),
                "wall_req_s": parse_walltime(rec.get("wall_req", "")),
                "exit_status": int(rec.get("exit_status", -1)),
            })

    return jobs


# ── Wall time parser ─────────────────────────────────────────────
def parse_walltime(s):
    """Convert HH:MM:SS or H:MM:SS to seconds."""
    if not s:
        return 0
    parts = s.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
    except ValueError:
        pass
    return 0


# ── Cray sampler loader ──────────────────────────────────────────
def load_sampler(day: str, node_set: set):
    """Load Cray system sampler data for a given day and set of nodes."""
    fp = os.path.join(BW_DATA, "cray_system_sampler", day)
    if not os.path.exists(fp):
        return None

    print(f"  Loading sampler for {day} ({len(node_set)} nodes)...")
    df = pd.read_csv(fp, header=None,
                     names=["timestamp", "node_id", "tx_bytes", "rx_bytes"])
    # Filter to relevant nodes
    df = df[df["node_id"].isin(node_set)]
    print(f"  Loaded {len(df):,} rows for {df.node_id.nunique()} nodes")
    return df


def compute_job_traffic(sampler_df, job):
    """Compute total tx/rx bytes for a job from sampler data."""
    if sampler_df is None or sampler_df.empty:
        return 0, 0

    node_ids = set(job["node_ids"])
    t_start = job["start"]
    t_end = job["end"]

    # Filter to job's nodes and time window
    mask = (sampler_df["node_id"].isin(node_ids) &
            (sampler_df["timestamp"] >= t_start - 60) &
            (sampler_df["timestamp"] <= t_end + 60))
    dfj = sampler_df[mask].copy()

    if dfj.empty or len(dfj) < 2:
        return 0, 0

    # Compute deltas (sampler stores cumulative counters)
    dfj = dfj.sort_values(["node_id", "timestamp"])
    dtx = dfj.groupby("node_id")["tx_bytes"].diff().clip(lower=0)
    drx = dfj.groupby("node_id")["rx_bytes"].diff().clip(lower=0)

    return int(dtx.sum()), int(drx.sum())


# ── Concurrency analysis ─────────────────────────────────────────
def compute_concurrency_at_job(job, all_jobs):
    """How many OTHER nodes were active during this job's lifetime?"""
    t_start = job["start"]
    t_end = job["end"]
    my_nodes = set(job["node_ids"])

    concurrent_nodes = set()
    for other in all_jobs:
        if other["job_id"] == job["job_id"]:
            continue
        # Check temporal overlap
        if other["end"] <= t_start or other["start"] >= t_end:
            continue
        concurrent_nodes.update(other["node_ids"])

    # Remove own nodes
    concurrent_nodes -= my_nodes
    return len(concurrent_nodes)


def compute_system_load_timeseries(all_jobs, bin_s=300):
    """Compute system load (fraction of nodes busy) over time."""
    if not all_jobs:
        return [], []

    t_min = min(j["start"] for j in all_jobs)
    t_max = max(j["end"] for j in all_jobs)
    n_bins = int((t_max - t_min) / bin_s) + 1

    node_counts = np.zeros(n_bins)

    for job in all_jobs:
        i_start = int((job["start"] - t_min) / bin_s)
        i_end = int((job["end"] - t_min) / bin_s)
        for i in range(max(0, i_start), min(n_bins, i_end + 1)):
            node_counts[i] += job["n_nodes"]

    timestamps = [t_min + i * bin_s for i in range(n_bins)]
    return timestamps, node_counts


# ── Target app extraction ─────────────────────────────────────────
def extract_target_apps(days, target_specs=None, use_sampler=True, output_tag="jan2017"):
    """Extract target app runs + all concurrent jobs across multiple days.

    Saves:
      bw_target_apps_{tag}.csv  — target job runs with full node_ids + sampler data
      bw_all_jobs_{tag}.csv     — all jobs in period (for concurrent lookup)
    """
    if target_specs is None:
        target_specs = TARGET_SPECS
    target_spec_set = {(name, n) for name, n in target_specs}

    def _normalize(raw_name, n_nodes):
        """Apply NAME_NORMALIZE: return (canonical_name, n_nodes) or None."""
        for prefix, (canon_name, canon_n) in NAME_NORMALIZE.items():
            if raw_name.startswith(prefix) and n_nodes == canon_n:
                return canon_name, canon_n
        if (raw_name, n_nodes) in target_spec_set:
            return raw_name, n_nodes
        return None

    os.makedirs(OUT_DIR, exist_ok=True)
    all_target_rows = []
    all_job_rows = []

    for day in days:
        print(f"\n{'='*60}")
        print(f"  Day {day}...")

        try:
            jobs = parse_torque_day(day)
        except FileNotFoundError:
            print(f"  Skipping {day} (no torque log)")
            continue

        if not jobs:
            print(f"  No jobs on {day}")
            continue

        target_jobs = [
            j for j in jobs
            if _normalize(j["name"], j["n_nodes"]) is not None
        ]
        print(f"  {len(jobs)} total jobs, {len(target_jobs)} target jobs")

        # Load sampler data for target job nodes (if preprocessed)
        sampler_df = None
        if use_sampler and target_jobs:
            sampler_fp = os.path.join(BW_DATA, "cray_system_sampler", day)
            if os.path.exists(sampler_fp):
                target_nodes = set()
                for j in target_jobs:
                    target_nodes.update(j["node_ids"])
                sampler_df = load_sampler(day, target_nodes)

        # Process each target job
        for job in target_jobs:
            canon = _normalize(job["name"], job["n_nodes"])
            if canon is None:
                continue
            canon_name, _ = canon

            tx_bytes, rx_bytes = 0, 0
            if sampler_df is not None:
                tx_bytes, rx_bytes = compute_job_traffic(sampler_df, job)

            # Concurrent jobs during this job's lifetime
            concurrent = [
                other for other in jobs
                if other["job_id"] != job["job_id"]
                and other["end"] > job["start"]
                and other["start"] < job["end"]
            ]
            concurrent_nodes = sum(c["n_nodes"] for c in concurrent)
            tx_rate = (tx_bytes / (job["n_nodes"] * job["duration_s"])
                       if job["n_nodes"] > 0 and job["duration_s"] > 0 else 0)

            all_target_rows.append({
                "day": day,
                "job_id": job["job_id"],
                "name": canon_name,
                "queue": job.get("queue", ""),
                "n_nodes": job["n_nodes"],
                "start_epoch": job["start"],
                "end_epoch": job["end"],
                "duration_s": job["duration_s"],
                "wall_req_s": job.get("wall_req_s", 0),
                "exit_status": job.get("exit_status", -1),
                "tx_bytes": tx_bytes,
                "rx_bytes": rx_bytes,
                "tx_rate_per_node_bps": tx_rate,
                "concurrent_other_nodes": concurrent_nodes,
                "concurrent_fraction": concurrent_nodes / BW_TOTAL_XE_NODES,
                "n_concurrent_jobs": len(concurrent),
                # Full node list for DOR routing computation
                "node_ids_str": " ".join(str(n) for n in job["node_ids"]),
                "concurrent_job_ids": " ".join(c["job_id"] for c in concurrent),
            })

        # Save ALL jobs from this day for concurrent lookup
        for job in jobs:
            all_job_rows.append({
                "day": day,
                "job_id": job["job_id"],
                "name": job["name"],
                "n_nodes": job["n_nodes"],
                "start_epoch": job["start"],
                "end_epoch": job["end"],
                "duration_s": job["duration_s"],
                "node_ids_str": " ".join(str(n) for n in job["node_ids"]),
            })

    print(f"\n{'='*60}")
    print(f"  Summary across {len(days)} days:")

    if all_target_rows:
        df = pd.DataFrame(all_target_rows)
        csv_path = os.path.join(OUT_DIR, f"bw_target_apps_{output_tag}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Target apps: {csv_path} ({len(df)} jobs)")
        for name, n in target_specs:
            subset = df[(df.name == name) & (df.n_nodes == n)]
            if len(subset) > 0:
                d_min = subset.duration_s.min() / 60
                d_max = subset.duration_s.max() / 60
                has_tx = (subset.tx_bytes > 0).sum()
                print(f"    {name}(n={n}): {len(subset)} runs, "
                      f"dur {d_min:.1f}-{d_max:.1f} min, "
                      f"{has_tx} with real tx_bytes")
    else:
        print("  No target jobs found")

    if all_job_rows:
        df_all = pd.DataFrame(all_job_rows)
        csv_path_all = os.path.join(OUT_DIR, f"bw_all_jobs_{output_tag}.csv")
        df_all.to_csv(csv_path_all, index=False)
        print(f"  All jobs: {csv_path_all} ({len(df_all)} jobs)")

    return all_target_rows


# ── Main ──────────────────────────────────────────────────────────
def extract_day(day: str, use_sampler: bool = True):
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Blue Waters Job Extraction: {day}")
    print(f"{'='*60}")

    # 1. Parse torque logs
    jobs = parse_torque_day(day)
    print(f"  {len(jobs)} completed jobs")
    if not jobs:
        return

    total_nodes = sum(j["n_nodes"] for j in jobs)
    print(f"  Total node-hours: {sum(j['n_nodes'] * j['duration_s'] / 3600 for j in jobs):.0f}")

    # 2. Load sampler data
    sampler_df = None
    if use_sampler:
        all_nodes = set()
        for j in jobs:
            all_nodes.update(j["node_ids"])
        sampler_fp = os.path.join(BW_DATA, "cray_system_sampler", day)
        if os.path.exists(sampler_fp):
            sampler_df = load_sampler(day, all_nodes)

    # 3. Compute per-job metrics
    rows = []
    for i, job in enumerate(jobs):
        if (i + 1) % 50 == 0:
            print(f"  Processing job {i+1}/{len(jobs)}...")

        # Traffic from sampler
        tx_bytes, rx_bytes = 0, 0
        if sampler_df is not None:
            tx_bytes, rx_bytes = compute_job_traffic(sampler_df, job)

        # Concurrency
        concurrent_nodes = compute_concurrency_at_job(job, jobs)

        # Compute tx rate (bytes/sec per node)
        tx_rate_per_node = tx_bytes / (job["n_nodes"] * job["duration_s"]) \
            if job["n_nodes"] > 0 and job["duration_s"] > 0 else 0

        rows.append({
            "job_id": job["job_id"],
            "name": job["name"],
            "queue": job["queue"],
            "n_nodes": job["n_nodes"],
            "start_epoch": job["start"],
            "end_epoch": job["end"],
            "duration_s": job["duration_s"],
            "tx_bytes": tx_bytes,
            "rx_bytes": rx_bytes,
            "tx_rate_per_node_bps": tx_rate_per_node,
            "concurrent_other_nodes": concurrent_nodes,
            "concurrent_fraction": concurrent_nodes / 22640,  # BW has ~22640 XE nodes
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, f"bw_jobs_{day}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path} ({len(df)} jobs)")

    # 4. System load timeseries
    ts, loads = compute_system_load_timeseries(jobs)
    ts_df = pd.DataFrame({"timestamp": ts, "active_nodes": loads})
    ts_path = os.path.join(OUT_DIR, f"bw_load_{day}.csv")
    ts_df.to_csv(ts_path, index=False)
    print(f"  Saved: {ts_path}")

    # Summary
    print(f"\n  Summary for {day}:")
    print(f"    Jobs: {len(df)}")
    has_tx = df[df.tx_bytes > 0]
    print(f"    Jobs with sampler data: {len(has_tx)}")
    if len(has_tx) > 0:
        print(f"    Avg tx_rate/node: {has_tx.tx_rate_per_node_bps.mean()/1e6:.1f} MB/s")
        print(f"    Max tx_rate/node: {has_tx.tx_rate_per_node_bps.max()/1e6:.1f} MB/s")
    print(f"    Avg concurrent nodes: {df.concurrent_other_nodes.mean():.0f}")
    print(f"    Max concurrent nodes: {df.concurrent_other_nodes.max()}")


def extract_multi_day(days, use_sampler=False):
    """Extract jobs from multiple days and merge into one CSV."""
    os.makedirs(OUT_DIR, exist_ok=True)
    all_rows = []

    for day in days:
        print(f"\n{'='*60}")
        print(f"  Extracting {day}...")

        try:
            jobs = parse_torque_day(day)
        except FileNotFoundError:
            print(f"  Skipping {day} (no torque log)")
            continue

        # Filter: only jobs with node allocation and >= 4 nodes
        jobs = [j for j in jobs if j["n_nodes"] >= 4 and j["node_ids"]]
        if not jobs:
            continue

        for job in jobs:
            concurrent_nodes = compute_concurrency_at_job(job, jobs)
            all_rows.append({
                "day": day,
                "job_id": job["job_id"],
                "name": job["name"],
                "queue": job["queue"],
                "n_nodes": job["n_nodes"],
                "start_epoch": job["start"],
                "end_epoch": job["end"],
                "duration_s": job["duration_s"],
                "tx_bytes": 0,  # no sampler for multi-day
                "rx_bytes": 0,
                "tx_rate_per_node_bps": 0,
                "concurrent_other_nodes": concurrent_nodes,
                "concurrent_fraction": concurrent_nodes / 22640,
                "node_ids_str": " ".join(str(n) for n in job["node_ids"][:100]),
            })

        print(f"  {len(jobs)} jobs extracted")

    if all_rows:
        df = pd.DataFrame(all_rows)
        csv_path = os.path.join(OUT_DIR, f"bw_jobs_multiday.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path} ({len(df)} total jobs)")
        return df
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Blue Waters job data")
    parser.add_argument("--day", default="20170328",
                        help="Start day (YYYYMMDD)")
    parser.add_argument("--days", type=int, default=1,
                        help="Number of consecutive days to process (default: 1)")
    parser.add_argument("--multi", action="store_true",
                        help="Extract 7 days of multiday data (--day is start)")
    parser.add_argument("--target-apps", action="store_true",
                        help="Extract target app runs (NAMD, scsw_xy_2x) + all concurrent jobs")
    parser.add_argument("--tag", default=None,
                        help="Output file tag (default: YYYYMMDD or jan2017)")
    parser.add_argument("--no-sampler", action="store_true",
                        help="Skip Cray sampler loading (faster)")
    parser.add_argument("--apps", nargs="+", default=None,
                        help="Override target apps as 'name:n_nodes' pairs, "
                             "e.g. --apps NAMD:64 scsw_xy_2x:160")
    args = parser.parse_args()

    start_dt = datetime.strptime(args.day, "%Y%m%d")

    if args.target_apps:
        n_days = args.days if args.days > 1 else 31  # default to 31 days for target apps
        days = [(start_dt + timedelta(days=i)).strftime("%Y%m%d") for i in range(n_days)]
        tag = args.tag or f"{args.day[:6]}"  # e.g., "201701"
        # Override target specs from CLI if provided
        cli_specs = None
        if args.apps:
            cli_specs = []
            for spec in args.apps:
                parts = spec.split(":")
                if len(parts) == 2:
                    cli_specs.append((parts[0], int(parts[1])))
        extract_target_apps(days, target_specs=cli_specs,
                            use_sampler=not args.no_sampler, output_tag=tag)
    elif args.multi:
        days = [(start_dt + timedelta(days=i)).strftime("%Y%m%d") for i in range(7)]
        extract_multi_day(days, use_sampler=not args.no_sampler)
    elif args.days > 1:
        days = [(start_dt + timedelta(days=i)).strftime("%Y%m%d") for i in range(args.days)]
        extract_multi_day(days, use_sampler=not args.no_sampler)
    else:
        extract_day(args.day, use_sampler=not args.no_sampler)

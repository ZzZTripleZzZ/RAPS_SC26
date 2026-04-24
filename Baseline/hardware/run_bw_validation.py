#!/usr/bin/env python3
"""
Blue Waters Layer 3 Validation: RAPS torus3d replay vs real hardware.

Takes the extracted job data (from extract_bw_jobs.py) and:
  1. Replays jobs through RAPS torus3d network model
  2. Computes RAPS-predicted congestion for each job
  3. Compares with real-world metrics (runtime, tx_rate, concurrency)
  4. Outputs correlation analysis + validation CSV

The key insight from Jha et al. NSDI 2020:
  - On Blue Waters, max packet time in congestion region correlates 0.89
    with execution time
  - RAPS should capture the same trend: higher system load → higher
    congestion → longer job runtime

Usage:
    cd /lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26
    .venv/bin/python3 Baseline/hardware/run_bw_validation.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from raps.network.torus3d import (
    build_torus3d, link_loads_for_job_torus, halo3d_26_pairs,
    torus_host_from_real_index,
)
from raps.job import CommunicationPattern

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Blue Waters: 24×24×24 torus, 2 hosts/router = 27648 hosts
BW_TORUS_DIMS = (24, 24, 24)
BW_HOSTS_PER_ROUTER = 2
BW_LINK_BW = 9.6e9        # Gemini NIC bandwidth (bytes/s)
BW_TRACE_QUANTA = 15       # seconds per trace bin (from bluewaters.yaml)
BW_MAX_THROUGHPUT = BW_LINK_BW * BW_TRACE_QUANTA  # bytes per quanta

# Synthetic tx rate for jobs without sampler data (50 MB/s/node is typical for BW)
SYNTHETIC_TX_RATE = 50e6  # bytes/second per node


def build_bw_topology():
    """Build Blue Waters torus3d topology."""
    print("  Building torus3d (24×24×24 × 2h/r)...")
    t0 = time.time()
    G, meta = build_torus3d(BW_TORUS_DIMS, hosts_per_router=BW_HOSTS_PER_ROUTER)
    hosts = sorted([n for n in G.nodes() if str(n).startswith("h_")])
    print(f"  Built: {len(hosts)} hosts, {G.number_of_edges()} edges "
          f"in {time.time()-t0:.1f}s")
    return G, meta, hosts


def real_nodes_to_torus_hosts(real_node_ids):
    """Map real BW node IDs to torus host names using modular indexing.

    Uses torus_host_from_real_index which preserves locality:
    nearby node IDs → nearby torus positions.
    """
    X, Y, Z = BW_TORUS_DIMS
    return [
        torus_host_from_real_index(nid, X, Y, Z, BW_HOSTS_PER_ROUTER)
        for nid in real_node_ids
    ]


def map_real_nodes_to_torus(real_node_ids, all_hosts):
    """Map real BW node IDs to torus host IDs.

    Blue Waters node IDs are physical (0-26000+). We map them to our
    torus model hosts using modular indexing. This preserves locality
    for nodes with nearby IDs (which are often on nearby torus positions).
    """
    n_hosts = len(all_hosts)
    mapped = []
    for nid in real_node_ids:
        idx = nid % n_hosts
        mapped.append(all_hosts[idx])
    return mapped


def compute_raps_congestion(G, meta, hosts, job_hosts, tx_bytes_per_node):
    """Compute RAPS congestion metric for a job on the torus.

    Returns: (max_link_utilization, avg_link_utilization, n_loaded_links)
    """
    if len(job_hosts) < 2 or tx_bytes_per_node <= 0:
        return 0.0, 0.0, 0

    # Compute link loads using torus dimension-order routing
    loads = link_loads_for_job_torus(
        G, meta, job_hosts, tx_bytes_per_node,
        comm_pattern=CommunicationPattern.STENCIL_3D,
    )

    if not loads:
        return 0.0, 0.0, 0

    max_load = max(loads.values())
    avg_load = np.mean(list(loads.values()))

    # Normalize by max throughput
    max_util = max_load / BW_MAX_THROUGHPUT if BW_MAX_THROUGHPUT > 0 else 0
    avg_util = avg_load / BW_MAX_THROUGHPUT if BW_MAX_THROUGHPUT > 0 else 0

    return max_util, avg_util, len(loads)


def compute_multi_job_congestion(G, meta, hosts, all_active_jobs, target_job_hosts):
    """Compute congestion from ALL active jobs, then measure impact on target.

    Returns: max link utilization on links used by the target job.
    """
    # Compute loads from all jobs combined
    combined_loads = {}
    for job_hosts, tx_per_node in all_active_jobs:
        if len(job_hosts) < 2 or tx_per_node <= 0:
            continue
        loads = link_loads_for_job_torus(
            G, meta, job_hosts, tx_per_node,
            comm_pattern=CommunicationPattern.STENCIL_3D,
        )
        for edge, load in loads.items():
            combined_loads[edge] = combined_loads.get(edge, 0) + load

    if not combined_loads:
        return 0.0

    # Compute target job's loads to find which links it uses
    if len(target_job_hosts) < 2:
        return 0.0
    target_loads = link_loads_for_job_torus(
        G, meta, target_job_hosts, 1.0,  # dummy tx
        comm_pattern=CommunicationPattern.STENCIL_3D,
    )

    if not target_loads:
        return max(combined_loads.values()) / BW_MAX_THROUGHPUT

    # Max combined load on links used by target
    max_on_target = 0
    for edge in target_loads:
        if edge in combined_loads:
            max_on_target = max(max_on_target, combined_loads[edge])

    return max_on_target / BW_MAX_THROUGHPUT if BW_MAX_THROUGHPUT > 0 else 0


# ── Actual slowdown computation ───────────────────────────────────
def compute_actual_slowdown(df, min_dur_s=300, wall_frac=0.95):
    """Compute actual slowdown relative to per-app baseline duration.

    Filters out:
      - duration < min_dur_s (failed/crashed jobs)
      - duration > wall_frac * wall_req_s (hit wall time limit)

    Baseline = 10th percentile of surviving durations per (name, n_nodes).
    slowdown = duration / baseline

    Returns df with added columns: filtered, baseline_s, actual_slowdown.
    """
    df = df.copy()
    df["filtered"] = False
    df["baseline_s"] = np.nan
    df["actual_slowdown"] = np.nan

    for (name, n_nodes), grp in df.groupby(["name", "n_nodes"]):
        idx = grp.index

        # Filter failed (too short) or wall-time-hit (too long)
        too_short = grp["duration_s"] < min_dur_s
        wall_req = grp.get("wall_req_s", pd.Series(0, index=idx))
        too_long = (wall_req > 0) & (grp["duration_s"] > wall_frac * wall_req)
        bad = too_short | too_long
        df.loc[idx[bad], "filtered"] = True

        good = grp.loc[~bad]
        if len(good) < 3:
            # Not enough data for a baseline
            df.loc[idx, "filtered"] = True
            continue

        baseline = np.percentile(good["duration_s"], 10)
        df.loc[idx, "baseline_s"] = baseline
        df.loc[idx, "actual_slowdown"] = df.loc[idx, "duration_s"] / baseline

    return df


# ── Inter-job congestion computation ──────────────────────────────
def compute_all_interjob_congestion(df_target, df_all, G, meta):
    """Compute RAPS-predicted inter-job congestion for each target job.

    Algorithm:
    1. Pre-compute link loads for all unique jobs (keyed by job_id)
    2. For each target job:
       a. Find all temporally overlapping jobs
       b. Sum their link loads on links used by the target job
       c. Compute max combined utilization → raps_interjob_max_util
       d. Apply M/D/1 model → raps_interjob_md1_slowdown

    Returns DataFrame with congestion metrics added.
    """
    X, Y, Z = BW_TORUS_DIMS

    # ---- Pre-compute link loads for all jobs ----
    print(f"  Pre-computing link loads for {len(df_all)} total jobs...")
    t0 = time.time()

    job_loads_cache = {}  # job_id → {edge: bytes}

    def _tx_per_quanta(row):
        """tx bytes per node per trace quanta."""
        dur = max(row["duration_s"], 1)
        n = max(row["n_nodes"], 1)
        n_q = max(1, dur / BW_TRACE_QUANTA)
        if row.get("tx_bytes", 0) > 0:
            total = row["tx_bytes"]
        else:
            total = SYNTHETIC_TX_RATE * dur * n  # synthetic estimate
        return total / (n * n_q)

    for i, (_, row) in enumerate(df_all.iterrows()):
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(df_all)} jobs pre-computed...", flush=True)

        jid = row["job_id"]
        n_nodes = int(row["n_nodes"])
        node_ids_str = str(row.get("node_ids_str", ""))

        if node_ids_str and node_ids_str != "nan" and node_ids_str.strip():
            real_ids = [int(x) for x in node_ids_str.split()]
        else:
            # No node info: skip (will contribute no load)
            job_loads_cache[jid] = {}
            continue

        # Map to torus hosts
        job_hosts = real_nodes_to_torus_hosts(real_ids)

        # Tx per node per quanta
        tx_pq = _tx_per_quanta(row)

        # Compute DOR link loads (STENCIL_3D approximates most HPC apps)
        try:
            loads = link_loads_for_job_torus(
                G, meta, job_hosts, tx_pq,
                comm_pattern=CommunicationPattern.STENCIL_3D,
            )
            job_loads_cache[jid] = loads if loads else {}
        except Exception:
            job_loads_cache[jid] = {}

    dt = time.time() - t0
    print(f"  Pre-computed in {dt:.1f}s")

    # ---- Compute per-target inter-job congestion ----
    # Build time-indexed lookup for fast concurrent job search
    all_jobs_list = df_all.to_dict("records")

    results = []
    for i, (_, trow) in enumerate(df_target.iterrows()):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Target job {i+1}/{len(df_target)}: "
                  f"{trow['name']} n={trow['n_nodes']}...", flush=True)

        t_start = trow["start_epoch"]
        t_end = trow["end_epoch"]
        tid = trow["job_id"]

        # Find concurrent jobs (overlap with target's time window)
        concurrent_job_ids = set()
        if "concurrent_job_ids" in trow and str(trow["concurrent_job_ids"]) not in ("nan", ""):
            concurrent_job_ids = set(str(trow["concurrent_job_ids"]).split())

        # Get target job's link loads (already pre-computed)
        target_loads = job_loads_cache.get(tid, {})

        if not target_loads:
            # Recompute for target using its own node_ids_str
            n_nodes = int(trow["n_nodes"])
            node_ids_str = str(trow.get("node_ids_str", ""))
            if node_ids_str and node_ids_str != "nan":
                real_ids = [int(x) for x in node_ids_str.split()]
                job_hosts = real_nodes_to_torus_hosts(real_ids)
                tx_pq = _tx_per_quanta(trow)
                try:
                    target_loads = link_loads_for_job_torus(
                        G, meta, job_hosts, tx_pq,
                        comm_pattern=CommunicationPattern.STENCIL_3D,
                    ) or {}
                except Exception:
                    target_loads = {}

        # Sum loads from concurrent jobs on target's links
        combined_loads = dict(target_loads)  # start with target's own load

        n_concurrent_with_loads = 0
        for cid in concurrent_job_ids:
            c_loads = job_loads_cache.get(cid, {})
            if not c_loads:
                continue
            n_concurrent_with_loads += 1
            for edge, load in c_loads.items():
                if edge in combined_loads:
                    combined_loads[edge] = combined_loads.get(edge, 0) + load

        # Only consider links that the TARGET job uses
        target_combined_max = max(
            (combined_loads.get(e, 0) for e in target_loads),
            default=0.0
        )
        interjob_max_util = (target_combined_max / BW_MAX_THROUGHPUT
                             if BW_MAX_THROUGHPUT > 0 else 0.0)

        # Self-only congestion (target job alone)
        self_max = max(target_loads.values(), default=0.0)
        self_max_util = self_max / BW_MAX_THROUGHPUT if BW_MAX_THROUGHPUT > 0 else 0.0

        # M/D/1 slowdown
        rho = min(interjob_max_util, 0.99)
        if 0.05 < rho < 1.0:
            md1_slowdown = 1.0 + rho ** 2 / (2.0 * (1.0 - rho))
        elif rho >= 1.0:
            md1_slowdown = 100.0
        else:
            md1_slowdown = 1.0

        results.append({
            "job_id": tid,
            "name": trow["name"],
            "n_nodes": trow["n_nodes"],
            "day": trow.get("day", ""),
            "duration_s": trow["duration_s"],
            "wall_req_s": trow.get("wall_req_s", 0),
            "exit_status": trow.get("exit_status", -1),
            "tx_bytes": trow.get("tx_bytes", 0),
            "tx_rate_per_node_bps": trow.get("tx_rate_per_node_bps", 0),
            "concurrent_other_nodes": trow.get("concurrent_other_nodes", 0),
            "concurrent_fraction": trow.get("concurrent_fraction", 0),
            "n_concurrent_jobs": trow.get("n_concurrent_jobs", 0),
            "n_concurrent_with_loads": n_concurrent_with_loads,
            "raps_self_max_util": self_max_util,
            "raps_interjob_max_util": interjob_max_util,
            "raps_interjob_stall_pct": 100.0 * interjob_max_util,
            "raps_interjob_md1_slowdown": md1_slowdown,
        })

    return pd.DataFrame(results)


def run_slowdown_interjob_validation(tag="201701"):
    """Full pipeline: load target apps, compute slowdown + inter-job congestion."""
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load target apps
    target_csv = os.path.join(OUT_DIR, f"bw_target_apps_{tag}.csv")
    if not os.path.exists(target_csv):
        print(f"ERROR: {target_csv} not found. Run extract_bw_jobs.py --target-apps first.")
        return

    df_target = pd.read_csv(target_csv)
    print(f"\nLoaded {len(df_target)} target job runs from {target_csv}")

    # Load all jobs (for concurrent lookup)
    all_jobs_csv = os.path.join(OUT_DIR, f"bw_all_jobs_{tag}.csv")
    if not os.path.exists(all_jobs_csv):
        print(f"WARNING: {all_jobs_csv} not found. Skipping inter-job congestion.")
        df_all = df_target.copy()
    else:
        df_all = pd.read_csv(all_jobs_csv)
        print(f"Loaded {len(df_all)} total jobs from {all_jobs_csv}")

    # Compute actual slowdown
    print("\nComputing actual slowdown...")
    df_target = compute_actual_slowdown(df_target)
    valid = df_target[~df_target["filtered"]]
    print(f"  Valid runs (not filtered): {len(valid)}/{len(df_target)}")
    for (name, n), grp in valid.groupby(["name", "n_nodes"]):
        b = grp["baseline_s"].iloc[0] if len(grp) > 0 else 0
        sd_max = grp["actual_slowdown"].max()
        print(f"    {name}(n={n}): {len(grp)} runs, "
              f"baseline={b/60:.1f} min, max_slowdown={sd_max:.2f}x")

    # Build topology
    G, meta, all_hosts = build_bw_topology()

    # Compute inter-job congestion
    print(f"\nComputing inter-job congestion for {len(df_target)} target jobs...")
    t0 = time.time()
    df_result = compute_all_interjob_congestion(df_target, df_all, G, meta)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Merge slowdown info back
    sd_cols = ["job_id", "filtered", "baseline_s", "actual_slowdown"]
    df_sd = df_target[sd_cols]
    df_result = df_result.merge(df_sd, on="job_id", how="left")

    # Save
    out_csv = os.path.join(OUT_DIR, f"bw_slowdown_validation_{tag}.csv")
    df_result.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv} ({len(df_result)} rows)")

    # Correlation analysis
    valid_res = df_result[~df_result["filtered"].fillna(True)]
    print(f"\n{'='*60}")
    print(f"  Correlation Analysis ({len(valid_res)} valid runs)")
    print(f"{'='*60}")

    if len(valid_res) >= 5:
        for col, label in [
            ("raps_interjob_stall_pct", "RAPS inter-job stall %"),
            ("raps_interjob_max_util", "RAPS max link util"),
            ("raps_interjob_md1_slowdown", "RAPS M/D/1 slowdown"),
            ("concurrent_fraction", "System concurrent fraction"),
        ]:
            if col in valid_res.columns:
                x = valid_res[col].dropna()
                y = valid_res["actual_slowdown"].loc[x.index].dropna()
                shared = x.index.intersection(y.index)
                if len(shared) >= 3:
                    r, p = scipy_stats.pearsonr(x.loc[shared], y.loc[shared])
                    print(f"  {label:35s}: r = {r:.3f}  (p={p:.3f}, n={len(shared)})")

    # Per-app breakdown
    print()
    for (name, n), grp in valid_res.groupby(["name", "n_nodes"]):
        if len(grp) >= 3:
            x = grp["raps_interjob_stall_pct"].dropna()
            y = grp["actual_slowdown"].loc[x.index].dropna()
            shared = x.index.intersection(y.index)
            if len(shared) >= 3:
                r, p = scipy_stats.pearsonr(x.loc[shared], y.loc[shared])
                print(f"  {name}(n={n}): r = {r:.3f}, n = {len(shared)}, "
                      f"slowdown {y.min():.2f}-{y.max():.2f}x")

    return df_result


def run_validation(day="20170328", multi=False):
    """Run RAPS validation against real Blue Waters data."""
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load extracted job data
    if multi:
        csv_path = os.path.join(OUT_DIR, "bw_jobs_multiday.csv")
    else:
        csv_path = os.path.join(OUT_DIR, f"bw_jobs_{day}.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: Run extract_bw_jobs.py first")
        return

    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} jobs from {csv_path}")

    # Filter: jobs with >= 4 nodes and reasonable duration
    df = df[df["n_nodes"] >= 4].copy()
    df = df[df["duration_s"] >= 60].copy()  # at least 1 minute
    print(f"After filtering (nodes>=4, dur>=60s): {len(df)} jobs")

    if len(df) == 0:
        print("No jobs to analyze")
        return

    # Build topology
    G, meta, all_hosts = build_bw_topology()

    # Compute RAPS congestion for each job
    print(f"\nComputing RAPS congestion for {len(df)} jobs...")
    results = []
    t0 = time.time()

    for i, (_, row) in enumerate(df.iterrows()):
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Job {i+1}/{len(df)}: {row['name']}, "
                  f"{row['n_nodes']} nodes...")

        # Map real node IDs to torus hosts
        n_nodes = int(row["n_nodes"])
        node_ids_str = str(row.get("node_ids_str", ""))
        if node_ids_str and node_ids_str != "nan":
            real_ids = [int(x) for x in node_ids_str.split()[:n_nodes]]
            job_hosts = map_real_nodes_to_torus(real_ids, all_hosts)
            # Pad if fewer IDs than n_nodes (CSV truncates to 20)
            if len(job_hosts) < n_nodes:
                start_idx = hash(row["job_id"]) % max(1, len(all_hosts) - n_nodes)
                extra = all_hosts[start_idx:start_idx + n_nodes - len(job_hosts)]
                job_hosts.extend(extra)
        else:
            start_idx = hash(row["job_id"]) % max(1, len(all_hosts) - n_nodes)
            start_idx = max(0, min(start_idx, len(all_hosts) - n_nodes))
            job_hosts = all_hosts[start_idx:start_idx + n_nodes]

        # Tx bytes per node per trace quanta
        # If we have real sampler data, use it; otherwise estimate from
        # a typical BW network load (~1 GB/s per node for comm-heavy apps)
        if row.get("tx_bytes", 0) > 0:
            tx_per_node = row["tx_bytes"] / n_nodes
        else:
            # Estimate: assume stencil-like comm pattern
            # ~100 MB/s per node is typical for HPC apps
            tx_per_node = 100e6 * row["duration_s"]
        n_quanta = max(1, row["duration_s"] / BW_TRACE_QUANTA)
        tx_per_quanta = tx_per_node / n_quanta

        # Single-job congestion
        max_util, avg_util, n_links = compute_raps_congestion(
            G, meta, all_hosts, job_hosts, tx_per_quanta)

        # M/D/1 slowdown prediction
        rho = min(max_util, 0.99)
        if 0.05 < rho < 1.0:
            md1_slowdown = 1 + rho**2 / (2 * (1 - rho))
        elif rho >= 1.0:
            md1_slowdown = 100.0
        else:
            md1_slowdown = 1.0

        results.append({
            "job_id": row["job_id"],
            "name": row["name"],
            "n_nodes": n_nodes,
            "duration_s": row["duration_s"],
            "tx_bytes": row["tx_bytes"],
            "tx_rate_per_node_bps": row["tx_rate_per_node_bps"],
            "concurrent_other_nodes": row["concurrent_other_nodes"],
            "concurrent_fraction": row["concurrent_fraction"],
            "raps_max_util": max_util,
            "raps_avg_util": avg_util,
            "raps_n_loaded_links": n_links,
            "raps_md1_slowdown": md1_slowdown,
        })

    dt = time.time() - t0
    print(f"  Done in {dt:.1f}s")

    # Save results
    res_df = pd.DataFrame(results)
    suffix = "multiday" if multi else day
    out_csv = os.path.join(OUT_DIR, f"bw_validation_{suffix}.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Correlation analysis
    print(f"\n{'='*60}")
    print("  Correlation Analysis")
    print(f"{'='*60}")

    # 1. Concurrent nodes vs RAPS congestion
    if len(res_df) > 3:
        corr_conc_util = res_df["concurrent_fraction"].corr(
            res_df["raps_max_util"])
        print(f"\n  concurrent_fraction vs raps_max_util: r = {corr_conc_util:.3f}")

        # 2. TX rate vs RAPS congestion
        corr_tx_util = res_df["tx_rate_per_node_bps"].corr(
            res_df["raps_max_util"])
        print(f"  tx_rate_per_node vs raps_max_util:    r = {corr_tx_util:.3f}")

        # 3. Concurrent nodes vs tx_rate (real data internal consistency)
        corr_conc_tx = res_df["concurrent_fraction"].corr(
            res_df["tx_rate_per_node_bps"])
        print(f"  concurrent_fraction vs tx_rate:       r = {corr_conc_tx:.3f}")

        # 4. Summary stats
        print(f"\n  RAPS max_util: mean={res_df.raps_max_util.mean():.4f}, "
              f"max={res_df.raps_max_util.max():.4f}")
        print(f"  RAPS md1_slowdown: mean={res_df.raps_md1_slowdown.mean():.3f}, "
              f"max={res_df.raps_md1_slowdown.max():.3f}")
        print(f"  Real concurrent_frac: mean={res_df.concurrent_fraction.mean():.3f}, "
              f"max={res_df.concurrent_fraction.max():.3f}")

    return res_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", default="20170328",
                        help="Day for single-day validation")
    parser.add_argument("--multi", action="store_true",
                        help="Use multiday extracted data (bw_jobs_multiday.csv)")
    parser.add_argument("--slowdown", action="store_true",
                        help="Run target-app slowdown + inter-job congestion validation")
    parser.add_argument("--interjob", action="store_true",
                        help="Include inter-job congestion (alias for --slowdown)")
    parser.add_argument("--tag", default="201701",
                        help="Data tag for --slowdown mode (default: 201701)")
    args = parser.parse_args()

    if args.slowdown or args.interjob:
        run_slowdown_interjob_validation(tag=args.tag)
    else:
        run_validation(args.day, multi=args.multi)

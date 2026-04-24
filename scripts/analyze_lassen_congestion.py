#!/usr/bin/env python3
"""
Lassen Inter-Job Network Congestion Analyzer
=============================================

Overview
--------
This script detects inter-job network interference in the Lassen Supercomputer
Job Dataset by replaying measured InfiniBand (IB) traffic through a simulated
fat-tree network topology. Rather than using synthetic workloads, it uses the
*actual* IB TX/RX counters recorded by IBM's Cluster Status Monitor (CSM) for
every job that ran on Lassen.

The central idea is straightforward: jobs that ran concurrently may have been
fighting over the same fat-tree links. By routing each job's measured IB traffic
through the simulated topology at each time snapshot, we can estimate the worst-
case link utilization. When that utilization exceeds 1.0, the fabric was
over-subscribed and jobs were almost certainly throttling each other.

Methodology
-----------
1.  **Load and filter**: Read `final_csm_allocation_history_hashed.csv` and
    `final_csm_allocation_node_history.csv` for the user-supplied time window.

2.  **Convert IB counters to rates**: The CSM records cumulative octets in
    "octets / quarter-byte" units (multiply ×4 to get bytes). Dividing by
    wall-time and the number of nodes gives a per-node byte rate. We then
    express this as bytes-per-trace-quanta (20 s by default) so it matches the
    unit used internally by the RAPS network model.

3.  **Snapshot sweep**: At each sample point (configurable interval, default
    1 hour), identify all jobs that were active (begin_time ≤ t < end_time).
    Only jobs with at least 2 nodes and non-zero IB traffic are included in the
    network simulation; single-node or silent jobs are tracked but excluded from
    the routing calculation.

4.  **Fat-tree simulation**: For each snapshot, build lightweight job objects
    and call ``simulate_inter_job_congestion`` from ``raps.network``. This
    routes each job's traffic through the fat-tree graph (k=32 for Lassen,
    100 Gbps EDR InfiniBand) and accumulates per-link byte loads. The result
    is a dict of statistics including the maximum link utilization across all
    edges.

5.  **Communication patterns**: The script supports two patterns selectable via
    ``--comm-pattern``:

    * ``all-to-all`` (default): every node in the job sends ``tx_rate`` bytes
      to each of its (N−1) peers. This is the upper bound — it assumes maximum
      fan-out traffic and produces the highest simulated congestion.
    * ``stencil-3d``: each node sends only to its 6 nearest neighbors in a
      virtual 3-D grid. This is closer to stencil / finite-difference workloads
      and gives a lower-bound estimate.

    Running the script twice with both patterns brackets the likely real-world
    congestion between best and worst case.

6.  **Output**: Three CSV files are produced:

    * ``<output>_snapshots.csv`` — one row per time snapshot with timestamp,
      number of active/simulated jobs, nodes occupied, and link utilization
      statistics (max, mean, std dev).
    * ``<output>_jobs.csv`` — one row per job with its peak congestion exposure
      (the maximum ``max_link_util`` across all snapshots the job appeared in),
      total IB bytes, node count, and whether it was identified as a likely
      "bully" (high sender) or "victim" (active during congested snapshots).
    * ``<output>_top_congested.csv`` — the N most congested snapshots, with the
      list of job IDs active at each.

Units and thresholds
--------------------
* **Link utilization** is expressed as a fraction of the per-link capacity
  (12.5 GB/s = 100 Gbps EDR IB). Values >1.0 indicate over-subscription.
* Jobs are flagged as "high congestion exposure" if the max link utilization
  during any snapshot they appear in exceeds ``--congestion-threshold``
  (default 0.5).
* Jobs are flagged as "bully" if their per-node IB TX rate is in the top
  ``--bully-percentile`` of all simulated jobs (default top 10 %).

Performance notes
-----------------
The routing computation scales as O(J × N²) per snapshot, where J is the number
of active network-intensive jobs and N is the average job node count. Lassen can
have hundreds of concurrent multi-node jobs. To keep runtime manageable:

* Use ``--min-nodes N`` to skip small jobs (default: 2).
* Use ``--min-ib-rate R`` to skip jobs with very low IB activity (default 0).
* Use ``--interval 3600`` (1 h) for a week-long window; use 600 (10 min) for a
  day-long window to get finer resolution.
* The ``--limit-concurrent`` option caps how many jobs are included per snapshot
  (keeps the top senders by IB rate) to bound computation time.

Usage examples
--------------
::

    # Full-week scan with ALL_TO_ALL (upper bound), hourly snapshots
    python scripts/analyze_lassen_congestion.py \\
        --data /opt/data/lassen/Lassen-Supercomputer-Job-Dataset \\
        --config config/lassen.yaml \\
        --start 2019-08-22 --end 2019-08-29 \\
        --comm-pattern all-to-all \\
        --output results/lassen_a2a

    # Repeat with stencil-3d (lower bound) for comparison
    python scripts/analyze_lassen_congestion.py \\
        --data /opt/data/lassen/Lassen-Supercomputer-Job-Dataset \\
        --config config/lassen.yaml \\
        --start 2019-08-22 --end 2019-08-29 \\
        --comm-pattern stencil-3d \\
        --output results/lassen_stencil

    # Single busy day, 10-minute snapshots, verbose
    python scripts/analyze_lassen_congestion.py \\
        --data /opt/data/lassen/Lassen-Supercomputer-Job-Dataset \\
        --config config/lassen.yaml \\
        --start 2019-08-22 --end 2019-08-23 \\
        --interval 600 \\
        --comm-pattern all-to-all \\
        --verbose --output results/lassen_day1

References
----------
* Patki et al., "Monitoring Large Scale Supercomputers: A Case Study with the
  Lassen Supercomputer", IEEE Cluster 2021.
* LLNL Lassen Dataset: https://github.com/LLNL/LAST
* RAPS fat-tree model: raps/network/fat_tree.py, raps/network/base.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from raps.job import CommunicationPattern, normalize_comm_pattern
from raps.network import NetworkModel, simulate_inter_job_congestion
from raps.network.fat_tree import node_id_to_host_name
from raps.system_config import get_system_config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CSM reports IB counters in "octets / quarter-byte"; multiply by 4 → real bytes
IB_UNIT_SCALE = 4

# Matches Lassen config: scheduler.trace_quanta = 20 s
DEFAULT_TRACE_QUANTA = 20

# Congestion threshold: link utilisation above this is flagged as interference
DEFAULT_CONGESTION_THRESHOLD = 0.5

# Default snapshot interval in seconds (1 hour)
DEFAULT_INTERVAL = 3600


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class JobRecord:
    """
    Lightweight record derived from Lassen CSM telemetry for one job allocation.

    Attributes
    ----------
    job_id:
        ``primary_job_id`` from the allocation history CSV.
    allocation_id:
        Internal CSM allocation identifier (links the two CSVs).
    begin_ts, end_ts:
        Parsed start and end timestamps for the allocation.
    wall_time:
        Duration in seconds (end_ts − begin_ts). Zero-duration jobs are skipped.
    num_nodes:
        Number of nodes allocated to this job.
    scheduled_nodes:
        List of integer node IDs (parsed from "lassenXXX" strings).
    ib_tx_rate_per_node:
        Bytes transmitted per node per trace-quanta interval. Computed as::

            4 × sum(ib_tx across nodes) / num_nodes / wall_time × trace_quanta

        This is the unit consumed by ``get_current_utilization`` inside the
        RAPS network model.
    ib_rx_rate_per_node:
        Same as above for received bytes.
    total_ib_bytes:
        Total IB bytes (TX+RX) over the job lifetime, summed across all nodes.
        Used to rank "bully" jobs.
    congestion_exposure:
        Updated during analysis: the maximum ``max_link_util`` observed across
        all snapshots this job appeared in.
    n_congested_snapshots:
        Number of snapshots where this job was active and ``max_link_util``
        exceeded the congestion threshold.
    """
    job_id: int
    allocation_id: int
    begin_ts: pd.Timestamp
    end_ts: pd.Timestamp
    wall_time: float
    num_nodes: int
    scheduled_nodes: list
    ib_tx_rate_per_node: float
    ib_rx_rate_per_node: float
    total_ib_bytes: float
    # Filled in during analysis
    congestion_exposure: float = field(default=0.0)
    n_congested_snapshots: int = field(default=0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(
    data_path: Path,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    trace_quanta: int,
    min_nodes: int,
    min_ib_rate: float,
    verbose: bool,
) -> list[JobRecord]:
    """
    Load and filter Lassen CSM records for the given time window.

    Reads ``final_csm_allocation_history_hashed.csv`` and
    ``final_csm_allocation_node_history.csv``, joins them on ``allocation_id``,
    and returns a list of :class:`JobRecord` objects covering jobs that overlapped
    with [start_ts, end_ts].

    Parameters
    ----------
    data_path:
        Directory containing the two CSV files.
    start_ts, end_ts:
        UTC-aware pandas Timestamps bounding the analysis window.
    trace_quanta:
        Seconds per trace interval; used to convert byte totals to rates.
    min_nodes:
        Skip jobs with fewer than this many allocated nodes.
    min_ib_rate:
        Skip jobs whose per-node TX rate (bytes/quanta) is below this value.
    verbose:
        Print extra progress information.
    """
    alloc_path = data_path / "final_csm_allocation_history_hashed.csv"
    node_path  = data_path / "final_csm_allocation_node_history.csv"

    print(f"[load] Reading allocation history: {alloc_path}")
    alloc_df = pd.read_csv(alloc_path, low_memory=False)

    print(f"[load] Reading node history: {node_path}")
    node_df = pd.read_csv(node_path, low_memory=False)

    # Parse timestamps
    alloc_df["begin_ts"] = pd.to_datetime(alloc_df["begin_time"], format="mixed", errors="coerce", utc=True)
    alloc_df["end_ts"]   = pd.to_datetime(alloc_df["end_time"],   format="mixed", errors="coerce", utc=True)

    # Drop rows with unparseable timestamps
    alloc_df = alloc_df.dropna(subset=["begin_ts", "end_ts"])

    # Keep only jobs that overlapped with the analysis window:
    #   job ended after window start AND job began before window end
    mask = (alloc_df["end_ts"] >= start_ts) & (alloc_df["begin_ts"] < end_ts)
    alloc_df = alloc_df[mask].copy()
    print(f"[load] {len(alloc_df)} allocations overlap the analysis window")

    # Pre-group node history by allocation_id for fast lookup
    node_by_alloc = node_df.groupby("allocation_id")

    records: list[JobRecord] = []
    skipped_small = skipped_no_ib = skipped_zero_wall = 0

    for _, row in tqdm(alloc_df.iterrows(), total=len(alloc_df), desc="Building job records"):
        num_nodes = int(row["num_nodes"])
        if num_nodes < min_nodes:
            skipped_small += 1
            continue

        begin_ts = row["begin_ts"]
        end_ts_job = row["end_ts"]
        wall_time = (end_ts_job - begin_ts).total_seconds()
        if wall_time <= 0:
            skipped_zero_wall += 1
            continue

        alloc_id = row["allocation_id"]

        # Get node-level data for this allocation
        if alloc_id in node_by_alloc.groups:
            node_data = node_by_alloc.get_group(alloc_id)
        else:
            node_data = pd.DataFrame()

        # Parse scheduled node IDs from "lassenXXX" strings
        scheduled_nodes: list[int] = []
        if not node_data.empty and "node_name" in node_data.columns:
            for name in node_data["node_name"].dropna():
                try:
                    scheduled_nodes.append(int(str(name).split("lassen")[-1]))
                except ValueError:
                    pass

        # Compute per-node IB TX/RX rates (bytes per trace_quanta interval)
        if not node_data.empty and "ib_tx" in node_data.columns:
            raw_tx = node_data["ib_tx"].fillna(0).clip(lower=0).sum()
            raw_rx = node_data["ib_rx"].fillna(0).clip(lower=0).sum() if "ib_rx" in node_data.columns else 0.0
            total_ib_bytes = IB_UNIT_SCALE * (raw_tx + raw_rx)
            # bytes/node/wall_time * trace_quanta = bytes per node per quanta interval
            tx_rate = IB_UNIT_SCALE * raw_tx / num_nodes / wall_time * trace_quanta
            rx_rate = IB_UNIT_SCALE * raw_rx / num_nodes / wall_time * trace_quanta
        else:
            tx_rate = rx_rate = total_ib_bytes = 0.0

        if tx_rate < min_ib_rate:
            skipped_no_ib += 1
            continue

        records.append(JobRecord(
            job_id=int(row["primary_job_id"]),
            allocation_id=int(alloc_id),
            begin_ts=begin_ts,
            end_ts=end_ts_job,
            wall_time=wall_time,
            num_nodes=num_nodes,
            scheduled_nodes=scheduled_nodes,
            ib_tx_rate_per_node=tx_rate,
            ib_rx_rate_per_node=rx_rate,
            total_ib_bytes=total_ib_bytes,
        ))

    print(
        f"[load] Built {len(records)} job records "
        f"(skipped: {skipped_small} too small, "
        f"{skipped_no_ib} no IB traffic, "
        f"{skipped_zero_wall} zero wall-time)"
    )
    return records


# ---------------------------------------------------------------------------
# Pre-screening pass
# ---------------------------------------------------------------------------

def prescreen_dataset(
    data_path: Path,
    min_nodes: int,
    window_days: int,
    top_n: int,
    use_ib: bool,
) -> None:
    """
    Fast full-dataset scan to identify the busiest windows for congestion.

    This runs entirely from ``final_csm_allocation_history_hashed.csv`` (and
    optionally ``final_csm_allocation_node_history.csv``) without building the
    fat-tree graph or running any routing.  It is designed to be run once to
    choose candidate windows before committing to the heavier full analysis.

    Algorithm
    ---------
    1.  Load the allocation history and filter to multi-node jobs
        (``num_nodes >= min_nodes``).

    2.  Build a **sweep-line** of concurrent network load using vectorised
        pandas operations:

        * For each job create two events: ``+weight`` at ``begin_ts`` and
          ``-weight`` at ``end_ts``.
        * Sort all events by timestamp and take the cumulative sum to get a
          running load signal sampled at event boundaries.
        * Re-sample onto a regular hourly grid with ``pd.merge_asof``.

        *Weight* is ``num_nodes`` by default.  With ``--prescreen-ib`` it is
        replaced by ``num_nodes × normalised_ib_tx_rate``, giving a score
        that rewards windows where many nodes are simultaneously doing heavy
        InfiniBand traffic rather than just occupying nodes idly.

    3.  Compute a **rolling mean** of the hourly load signal over
        ``window_days × 24`` hours.  The peak of this rolling mean marks the
        centre of the busiest sustained window.

    4.  Select the top-N **non-overlapping** windows using a greedy peak-
        picking pass: after choosing a window, suppress all hours within
        ``window_days`` of it before looking for the next candidate.

    5.  Print a ranked table and ready-to-paste CLI commands.

    Parameters
    ----------
    data_path:
        Directory containing the Lassen CSV files.
    min_nodes:
        Minimum job size to include in the load signal.
    window_days:
        Candidate window length in days.
    top_n:
        Number of top non-overlapping windows to report.
    use_ib:
        If True, weight the load signal by per-job IB TX rate (requires
        loading ``final_csm_allocation_node_history.csv``).
    """
    alloc_path = data_path / "final_csm_allocation_history_hashed.csv"
    print(f"[prescreen] Loading allocation history …")
    alloc_df = pd.read_csv(alloc_path, low_memory=False)

    alloc_df["begin_ts"] = pd.to_datetime(alloc_df["begin_time"], format="mixed", errors="coerce", utc=True)
    alloc_df["end_ts"]   = pd.to_datetime(alloc_df["end_time"],   format="mixed", errors="coerce", utc=True)
    alloc_df["wall_time"] = (alloc_df["end_ts"] - alloc_df["begin_ts"]).dt.total_seconds()

    filt = alloc_df[
        (alloc_df["num_nodes"] >= min_nodes) &
        alloc_df["begin_ts"].notna() &
        alloc_df["end_ts"].notna() &
        (alloc_df["wall_time"] > 0)
    ].copy()
    print(f"[prescreen] {len(filt):,} multi-node jobs (≥{min_nodes} nodes) across full dataset")
    print(f"[prescreen] Dataset spans {alloc_df['begin_ts'].min().date()} → {alloc_df['end_ts'].max().date()}")

    # --- Compute per-job weight -------------------------------------------
    if use_ib:
        node_path = data_path / "final_csm_allocation_node_history.csv"
        print(f"[prescreen] Loading node history for IB weighting (slow) …")
        node_df = pd.read_csv(node_path, low_memory=False, usecols=["allocation_id", "ib_tx"])
        ib_totals = (
            node_df.groupby("allocation_id")["ib_tx"]
            .sum()
            .rename("raw_ib_tx_sum")
        )
        filt = filt.join(ib_totals, on="allocation_id", how="left")
        filt["raw_ib_tx_sum"] = filt["raw_ib_tx_sum"].fillna(0).clip(lower=0)
        # bytes/s/node (before ×4 scale — we only need relative ranking)
        filt["ib_rate"] = filt["raw_ib_tx_sum"] / filt["num_nodes"] / filt["wall_time"]
        # Normalise to [0,1] so weight ≈ num_nodes when IB is at max
        max_rate = max(float(filt["ib_rate"].quantile(0.99)), 1.0)
        filt["weight"] = filt["num_nodes"] * (filt["ib_rate"] / max_rate).clip(upper=1.0)
        score_label = "nodes × norm_IB_rate"
    else:
        filt["weight"] = filt["num_nodes"].astype(float)
        score_label = "nodes in use"

    # --- Vectorised sweep-line onto hourly grid ---------------------------
    starts = filt[["begin_ts", "weight"]].rename(columns={"begin_ts": "ts", "weight": "delta"})
    ends   = filt[["end_ts",   "weight"]].rename(columns={"end_ts":   "ts", "weight": "delta"})
    ends   = ends.copy()
    ends["delta"] = -ends["delta"]

    events = (
        pd.concat([starts, ends], ignore_index=True)
        .sort_values("ts")
        .reset_index(drop=True)
    )
    events["running"] = events["delta"].cumsum()

    # Hourly grid over full dataset lifetime
    grid_start = filt["begin_ts"].min().floor("h")
    grid_end   = filt["end_ts"].max().ceil("h")
    hourly_grid = pd.DataFrame({"ts": pd.date_range(grid_start, grid_end, freq="h")})

    hourly = pd.merge_asof(hourly_grid, events[["ts", "running"]], on="ts", direction="backward")
    hourly["running"] = hourly["running"].fillna(0).clip(lower=0)

    # Also track concurrent job count (unweighted) for the table
    starts_j = filt[["begin_ts"]].assign(delta=1).rename(columns={"begin_ts": "ts"})
    ends_j   = filt[["end_ts"]].assign(delta=-1).rename(columns={"end_ts": "ts"})
    events_j = pd.concat([starts_j, ends_j]).sort_values("ts").reset_index(drop=True)
    events_j["n_jobs"] = events_j["delta"].cumsum()
    hourly = pd.merge_asof(hourly, events_j[["ts", "n_jobs"]], on="ts", direction="backward")
    hourly["n_jobs"] = hourly["n_jobs"].fillna(0).clip(lower=0).astype(int)

    # --- Rolling mean to find busiest sustained windows -------------------
    window_hours = window_days * 24
    hourly["rolling_score"] = hourly["running"].rolling(window_hours, min_periods=1).mean()

    # --- Greedy non-overlapping peak selection ----------------------------
    scored = hourly.copy()
    candidates = []
    suppressed = pd.Series(False, index=scored.index)

    for _ in range(top_n):
        available = scored[~suppressed]
        if available.empty:
            break
        peak_idx = available["rolling_score"].idxmax()
        peak_row  = scored.loc[peak_idx]

        # The window ends at this hour (rolling mean looks back window_hours)
        win_end   = peak_row["ts"]
        win_start = win_end - pd.Timedelta(hours=window_hours - 1)

        # Stats for the window
        mask = (hourly["ts"] >= win_start) & (hourly["ts"] <= win_end)
        win_data = hourly[mask]

        candidates.append({
            "rank":         len(candidates) + 1,
            "start":        win_start.date(),
            "end":          (win_end + pd.Timedelta(hours=1)).date(),
            "avg_score":    win_data["running"].mean(),
            "peak_score":   win_data["running"].max(),
            "avg_jobs":     win_data["n_jobs"].mean(),
            "peak_jobs":    win_data["n_jobs"].max(),
        })

        # Suppress all hours within one window of this peak
        suppress_mask = (
            (scored["ts"] >= win_start - pd.Timedelta(hours=window_hours)) &
            (scored["ts"] <= win_end   + pd.Timedelta(hours=window_hours))
        )
        suppressed = suppressed | suppress_mask

    # --- Print results ----------------------------------------------------
    print(f"\n{'=' * 72}")
    print(f"  TOP {len(candidates)} BUSIEST {window_days}-DAY WINDOWS  (score = {score_label})")
    print(f"{'=' * 72}")
    print(f"  {'Rank':<5} {'Start':<12} {'End':<12} {'Avg score':>10} {'Peak score':>11} {'Avg jobs':>9} {'Peak jobs':>10}")
    print(f"  {'-'*5} {'-'*12} {'-'*12} {'-'*10} {'-'*11} {'-'*9} {'-'*10}")
    for c in candidates:
        print(
            f"  {c['rank']:<5} {str(c['start']):<12} {str(c['end']):<12} "
            f"{c['avg_score']:>10.1f} {c['peak_score']:>11.1f} "
            f"{c['avg_jobs']:>9.1f} {c['peak_jobs']:>10.0f}"
        )

    print(f"\n  Suggested --start / --end commands:")
    for c in candidates:
        print(f"    #{c['rank']:>2}  --start {c['start']} --end {c['end']}")
    print(f"{'=' * 72}\n")


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def build_sim_job(
    rec: JobRecord,
    comm_pattern: CommunicationPattern,
    valid_nodes: set[int],
    trace_quanta: int,
) -> Optional[SimpleNamespace]:
    """
    Create a minimal simulation-compatible job object from a :class:`JobRecord`.

    The returned ``SimpleNamespace`` has exactly the attributes required by
    :func:`raps.network.simulate_inter_job_congestion`:

    * ``id``, ``scheduled_nodes``, ``nodes_required``
    * ``ntx_trace`` — single-element list ``[tx_rate_bytes_per_quanta]``
    * ``nrx_trace`` — single-element list ``[rx_rate_bytes_per_quanta]``
    * ``trace_quanta``, ``current_run_time``, ``trace_start_time``
    * ``comm_pattern`` — :class:`~raps.job.CommunicationPattern` enum
    * ``message_size``, ``message_overhead_bytes`` — both ``None`` to use the
      raw-bandwidth model (no packet framing overhead)

    Returns ``None`` if the job has fewer than 2 valid fat-tree nodes after
    filtering out node IDs not present in the topology graph.

    Parameters
    ----------
    rec:
        Source telemetry record.
    comm_pattern:
        Communication pattern to assume for routing.
    valid_nodes:
        Set of node integer IDs that exist in the fat-tree graph. Node IDs from
        the Lassen dataset that are out of range are silently dropped.
    trace_quanta:
        Seconds per trace interval (must match the NetworkModel config).
    """
    nodes = [n for n in rec.scheduled_nodes if n in valid_nodes]
    if len(nodes) < 2:
        return None

    job = SimpleNamespace()
    job.id                   = rec.job_id
    job.scheduled_nodes      = nodes
    job.nodes_required       = len(nodes)
    job.ntx_trace            = [rec.ib_tx_rate_per_node]
    job.nrx_trace            = [rec.ib_rx_rate_per_node]
    job.trace_quanta         = trace_quanta
    job.current_run_time     = 0
    job.trace_start_time     = 0
    job.comm_pattern         = comm_pattern
    job.message_size         = None   # raw bandwidth — no packet overhead
    job.message_size_bytes   = None
    job.message_overhead_bytes = None
    return job


def compute_valid_nodes(net_graph, k: int) -> set[int]:
    """
    Return the set of integer node IDs whose fat-tree host name exists in the
    network graph.

    Lassen node IDs come from parsing "lassenXXX" → XXX. The mapping
    ``node_id_to_host_name(n, k)`` converts these to "h_{pod}_{edge}_{host}"
    names in the fat-tree.

    Note: ``build_fattree(k, total_nodes)`` always builds the **full** k³/4
    host topology (total_nodes is only a bounds-check). For Lassen with k=32
    this means all 8192 fat-tree positions are present in the graph, so all
    real node IDs in range [0, k³/4) are valid. The returned set is used as a
    filter when constructing simulation job objects; in practice, for Lassen it
    is a no-op since all dataset node IDs fall within the fat-tree's address
    space.
    """
    host_set = set(net_graph.nodes())
    valid = set()
    # Only check IDs in the addressable fat-tree range (0 to k^3/4 - 1)
    max_possible = (k ** 3) // 4
    for nid in range(max_possible):
        if node_id_to_host_name(nid, k) in host_set:
            valid.add(nid)
    return valid


# ---------------------------------------------------------------------------
# Core analysis loop
# ---------------------------------------------------------------------------

def run_analysis(
    records: list[JobRecord],
    net: NetworkModel,
    legacy_cfg: dict,
    comm_pattern: CommunicationPattern,
    valid_nodes: set[int],
    trace_quanta: int,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    interval: int,
    congestion_threshold: float,
    limit_concurrent: Optional[int],
    verbose: bool,
) -> list[dict]:
    """
    Sweep through the time window in ``interval``-second steps, simulate
    inter-job congestion at each snapshot, and return snapshot statistics.

    For each snapshot timestamp ``t``:

    1. Find all :class:`JobRecord` objects whose ``[begin_ts, end_ts)`` contains
       ``t``.
    2. Convert them to lightweight simulation job objects via
       :func:`build_sim_job`.
    3. Optionally cap the concurrent set to the top ``limit_concurrent`` senders
       by IB TX rate (reduces computation for large snapshots).
    4. Call :func:`~raps.network.simulate_inter_job_congestion` to get link
       utilization statistics.
    5. Update each job's ``congestion_exposure`` and ``n_congested_snapshots``.

    Returns a list of dicts suitable for writing to ``_snapshots.csv``.

    Parameters
    ----------
    records:
        All :class:`JobRecord` objects in the analysis window.
    net:
        Initialized :class:`~raps.network.NetworkModel` (fat-tree).
    legacy_cfg:
        Legacy config dict as returned by ``sys_cfg.get_legacy()``.
    comm_pattern:
        Communication pattern assumed for all jobs.
    valid_nodes:
        Node IDs present in the fat-tree graph.
    trace_quanta:
        Seconds per trace interval.
    window_start, window_end:
        UTC-aware Timestamps bounding the sweep.
    interval:
        Seconds between snapshot sample points.
    congestion_threshold:
        Link utilisation above which a snapshot is considered congested.
    limit_concurrent:
        If set, each snapshot is limited to this many jobs (top IB senders).
    verbose:
        Print per-snapshot details.
    """
    # Build a sorted list of (begin_ts, end_ts, record) for the sweep
    sorted_records = sorted(records, key=lambda r: r.begin_ts)

    # Generate snapshot timestamps at regular intervals
    snapshot_times: list[pd.Timestamp] = []
    t = window_start
    while t < window_end:
        snapshot_times.append(t)
        t += pd.Timedelta(seconds=interval)

    snapshot_rows: list[dict] = []
    prev_job_ids: set[int] = set()

    for t in tqdm(snapshot_times, desc="Simulating snapshots"):
        # Find all active records at time t
        active = [
            r for r in sorted_records
            if r.begin_ts <= t < r.end_ts
        ]

        # Track all active job IDs (including single-node / no-IB jobs)
        active_job_ids = {r.job_id for r in active}
        n_active_total = len(active)
        n_nodes_busy = sum(r.num_nodes for r in active)

        # Build sim job objects (filters out <2 nodes, no valid nodes)
        sim_jobs_all = [
            j for r in active
            for j in [build_sim_job(r, comm_pattern, valid_nodes, trace_quanta)]
            if j is not None
        ]

        # Optionally cap to top senders to bound computation time
        if limit_concurrent and len(sim_jobs_all) > limit_concurrent:
            sim_jobs_all.sort(key=lambda j: j.ntx_trace[0], reverse=True)
            sim_jobs_all = sim_jobs_all[:limit_concurrent]

        n_sim = len(sim_jobs_all)

        row: dict = {
            "timestamp":        t.isoformat(),
            "n_active_jobs":    n_active_total,
            "n_sim_jobs":       n_sim,
            "n_nodes_busy":     n_nodes_busy,
            "max_link_util":    0.0,
            "mean_link_util":   0.0,
            "std_link_util":    0.0,
            "congested":        False,
            "active_job_ids":   ",".join(str(j) for j in sorted(active_job_ids)),
        }

        if n_sim >= 2:
            # Skip if the active job set is unchanged (optimization)
            sim_job_ids = {j.id for j in sim_jobs_all}
            if sim_job_ids == prev_job_ids and snapshot_rows:
                # Copy stats from previous snapshot (job set hasn't changed)
                prev = snapshot_rows[-1]
                row["max_link_util"]  = prev["max_link_util"]
                row["mean_link_util"] = prev["mean_link_util"]
                row["std_link_util"]  = prev["std_link_util"]
                row["congested"]      = prev["congested"]
            else:
                prev_job_ids = sim_job_ids
                stats = simulate_inter_job_congestion(net, sim_jobs_all, legacy_cfg, debug=False)
                max_util  = float(stats.get("max",     0.0))
                mean_util = float(stats.get("mean",    0.0))
                std_util  = float(stats.get("std_dev", 0.0))

                row["max_link_util"]  = max_util
                row["mean_link_util"] = mean_util
                row["std_link_util"]  = std_util
                row["congested"]      = max_util >= congestion_threshold

                # Update per-job exposure
                is_congested = max_util >= congestion_threshold
                for r in active:
                    if max_util > r.congestion_exposure:
                        r.congestion_exposure = max_util
                    if is_congested:
                        r.n_congested_snapshots += 1

            if verbose:
                flag = "*** CONGESTED ***" if row["congested"] else ""
                print(
                    f"  {t.isoformat()}  jobs={n_sim:3d}  nodes={n_nodes_busy:4d}  "
                    f"max_util={row['max_link_util']:.3f}  mean={row['mean_link_util']:.4f}  {flag}"
                )

        snapshot_rows.append(row)

    return snapshot_rows


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_snapshots_csv(snapshot_rows: list[dict], path: Path) -> None:
    """Write the per-snapshot statistics table to a CSV file."""
    if not snapshot_rows:
        return
    fieldnames = list(snapshot_rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(snapshot_rows)
    print(f"[output] Snapshots → {path}")


def write_jobs_csv(
    records: list[JobRecord],
    bully_threshold: float,
    congestion_threshold: float,
    path: Path,
) -> None:
    """
    Write the per-job summary to a CSV file.

    Each row includes the job's measured IB rate, simulated congestion exposure,
    and boolean flags for "bully" (high sender) and "victim" (high exposure).
    """
    fieldnames = [
        "job_id", "allocation_id", "begin_ts", "end_ts", "wall_time_s",
        "num_nodes", "n_scheduled_nodes",
        "ib_tx_rate_bytes_per_quanta_per_node",
        "ib_rx_rate_bytes_per_quanta_per_node",
        "total_ib_bytes",
        "congestion_exposure_max_link_util",
        "n_congested_snapshots",
        "is_bully", "is_victim",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(records, key=lambda x: x.congestion_exposure, reverse=True):
            writer.writerow({
                "job_id":                                   r.job_id,
                "allocation_id":                            r.allocation_id,
                "begin_ts":                                 r.begin_ts.isoformat(),
                "end_ts":                                   r.end_ts.isoformat(),
                "wall_time_s":                              f"{r.wall_time:.1f}",
                "num_nodes":                                r.num_nodes,
                "n_scheduled_nodes":                        len(r.scheduled_nodes),
                "ib_tx_rate_bytes_per_quanta_per_node":     f"{r.ib_tx_rate_per_node:.2f}",
                "ib_rx_rate_bytes_per_quanta_per_node":     f"{r.ib_rx_rate_per_node:.2f}",
                "total_ib_bytes":                           f"{r.total_ib_bytes:.0f}",
                "congestion_exposure_max_link_util":        f"{r.congestion_exposure:.4f}",
                "n_congested_snapshots":                    r.n_congested_snapshots,
                "is_bully":                                 r.ib_tx_rate_per_node >= bully_threshold,
                "is_victim":                                r.congestion_exposure >= congestion_threshold,
            })
    print(f"[output] Jobs      → {path}")


def write_top_congested_csv(
    snapshot_rows: list[dict],
    top_n: int,
    path: Path,
) -> None:
    """Write the top-N most congested snapshots to a CSV file."""
    sorted_rows = sorted(snapshot_rows, key=lambda r: r["max_link_util"], reverse=True)
    top = sorted_rows[:top_n]
    if not top:
        return
    fieldnames = list(top[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(top)
    print(f"[output] Top-{top_n}   → {path}")


def print_summary(records: list[JobRecord], snapshot_rows: list[dict], congestion_threshold: float) -> None:
    """Print a brief human-readable summary to stdout."""
    congested = [r for r in snapshot_rows if r["max_link_util"] >= congestion_threshold]
    all_utils  = [r["max_link_util"] for r in snapshot_rows if r["n_sim_jobs"] >= 2]

    print("\n" + "=" * 60)
    print("  CONGESTION ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"  Total snapshots analysed : {len(snapshot_rows)}")
    print(f"  Snapshots with ≥2 sim jobs: {sum(1 for r in snapshot_rows if r['n_sim_jobs'] >= 2)}")
    print(f"  Congested snapshots       : {len(congested)}  (max_link_util ≥ {congestion_threshold})")
    if all_utils:
        print(f"  Max link util (all time)  : {max(all_utils):.4f}")
        print(f"  Mean link util (all time) : {np.mean(all_utils):.4f}")

    victims = [r for r in records if r.congestion_exposure >= congestion_threshold]
    bullies_tx = sorted(records, key=lambda r: r.ib_tx_rate_per_node, reverse=True)[:10]

    print(f"\n  Jobs with high congestion exposure ({len(victims)} total):")
    for r in sorted(victims, key=lambda x: x.congestion_exposure, reverse=True)[:10]:
        print(
            f"    job {r.job_id:>8d}  nodes={r.num_nodes:4d}  "
            f"exposure={r.congestion_exposure:.3f}  "
            f"congested_in={r.n_congested_snapshots} snapshots"
        )

    print(f"\n  Top 10 IB senders (potential 'bullies'):")
    for r in bullies_tx:
        print(
            f"    job {r.job_id:>8d}  nodes={r.num_nodes:4d}  "
            f"tx_rate={r.ib_tx_rate_per_node:.1f} B/quanta/node  "
            f"exposure={r.congestion_exposure:.3f}"
        )
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Detect inter-job network congestion in the Lassen dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data", required=True, type=Path,
        help="Path to the Lassen-Supercomputer-Job-Dataset directory.",
    )
    p.add_argument(
        "--config", required=True,
        help="Path to system YAML config (e.g. config/lassen.yaml).",
    )
    p.add_argument(
        "--start", default="2019-08-22",
        help="Analysis window start date (ISO 8601, UTC assumed). Default: 2019-08-22.",
    )
    p.add_argument(
        "--end", default="2019-08-29",
        help="Analysis window end date (ISO 8601, UTC assumed). Default: 2019-08-29.",
    )
    p.add_argument(
        "--interval", type=int, default=DEFAULT_INTERVAL,
        help=f"Seconds between snapshot sample points. Default: {DEFAULT_INTERVAL} (1 hour).",
    )
    p.add_argument(
        "--comm-pattern",
        choices=("all-to-all", "stencil-3d"),
        default="all-to-all",
        help=(
            "Communication pattern assumed for all jobs. "
            "'all-to-all' gives an upper bound on congestion; "
            "'stencil-3d' gives a lower bound. Default: all-to-all."
        ),
    )
    p.add_argument(
        "--min-nodes", type=int, default=2,
        help="Skip jobs with fewer than this many nodes. Default: 2.",
    )
    p.add_argument(
        "--min-ib-rate", type=float, default=0.0,
        help="Skip jobs whose per-node IB TX rate (bytes/quanta) is below this. Default: 0.",
    )
    p.add_argument(
        "--congestion-threshold", type=float, default=DEFAULT_CONGESTION_THRESHOLD,
        help=(
            f"Link utilisation fraction above which a snapshot is marked congested "
            f"and jobs are flagged as 'victims'. Default: {DEFAULT_CONGESTION_THRESHOLD}."
        ),
    )
    p.add_argument(
        "--bully-percentile", type=float, default=90.0,
        help=(
            "Jobs with IB TX rate above this percentile of all simulated jobs "
            "are flagged as potential 'bullies'. Default: 90 (top 10%%)."
        ),
    )
    p.add_argument(
        "--limit-concurrent", type=int, default=None,
        help=(
            "Cap the number of concurrent jobs per snapshot to this many "
            "(selects the top senders). Reduces computation for large snapshots. "
            "Default: no limit."
        ),
    )
    p.add_argument(
        "--top-n", type=int, default=20,
        help="Number of most congested snapshots to include in the top-congested CSV. Default: 20.",
    )
    p.add_argument(
        "--output", default="lassen_congestion",
        help=(
            "Output file prefix. Three files are written: "
            "<prefix>_snapshots.csv, <prefix>_jobs.csv, <prefix>_top_congested.csv. "
            "Default: lassen_congestion."
        ),
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-snapshot details during the sweep.",
    )

    # --- Pre-screening options ---
    prescreen = p.add_argument_group("pre-screening options")
    prescreen.add_argument(
        "--prescreen", action="store_true",
        help=(
            "Scan the full dataset to find the busiest candidate windows, "
            "print a ranked table with ready-to-paste CLI commands, and exit. "
            "Does not require --start/--end and skips the fat-tree simulation entirely."
        ),
    )
    prescreen.add_argument(
        "--prescreen-window", type=int, default=7, metavar="DAYS",
        help="Candidate window length in days to rank. Default: 7.",
    )
    prescreen.add_argument(
        "--prescreen-top", type=int, default=10, metavar="N",
        help="Number of top non-overlapping windows to report. Default: 10.",
    )
    prescreen.add_argument(
        "--prescreen-ib", action="store_true",
        help=(
            "Weight the load signal by per-job IB TX rate instead of just node count. "
            "More accurate but requires loading the full node history CSV (~1 min extra)."
        ),
    )

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # --- Pre-screening mode (fast, exits before fat-tree build) -----------
    if args.prescreen:
        prescreen_dataset(
            data_path=args.data,
            min_nodes=args.min_nodes,
            window_days=args.prescreen_window,
            top_n=args.prescreen_top,
            use_ib=args.prescreen_ib,
        )
        sys.exit(0)

    # --- Time window ---
    window_start = pd.Timestamp(args.start, tz="UTC")
    window_end   = pd.Timestamp(args.end,   tz="UTC")
    print(f"[config] Window: {window_start.date()} → {window_end.date()}")
    print(f"[config] Comm pattern: {args.comm_pattern}")
    print(f"[config] Interval: {args.interval} s  |  Congestion threshold: {args.congestion_threshold}")

    comm_pattern = normalize_comm_pattern(args.comm_pattern)

    # --- Load system config ---
    sys_cfg    = get_system_config(args.config)
    legacy_cfg = sys_cfg.get_legacy()
    topology   = legacy_cfg.get("TOPOLOGY", "").lower()
    if topology != "fat-tree":
        print(f"[warn] Expected fat-tree topology, got '{topology}'. Proceeding anyway.")

    trace_quanta = legacy_cfg.get("TRACE_QUANTA", DEFAULT_TRACE_QUANTA)
    total_nodes  = legacy_cfg.get("TOTAL_NODES", 4626)
    k            = legacy_cfg.get("FATTREE_K", 32)
    print(f"[config] Fat-tree k={k}, total_nodes={total_nodes}, trace_quanta={trace_quanta} s")

    # --- Initialize network model ---
    print("[net] Building fat-tree network graph...")
    net = NetworkModel(
        config=legacy_cfg,
        available_nodes=list(range(total_nodes)),
    )

    # Compute the set of valid node IDs in the fat-tree graph
    valid_nodes = compute_valid_nodes(net.net_graph, k)
    print(f"[net] Fat-tree contains {len(valid_nodes)} valid host nodes")

    # --- Load telemetry records ---
    records = load_records(
        data_path=args.data,
        start_ts=window_start,
        end_ts=window_end,
        trace_quanta=trace_quanta,
        min_nodes=args.min_nodes,
        min_ib_rate=args.min_ib_rate,
        verbose=args.verbose,
    )

    if not records:
        print("[error] No job records found for this window. Exiting.")
        sys.exit(1)

    # --- Compute bully TX threshold ---
    tx_rates = [r.ib_tx_rate_per_node for r in records]
    bully_threshold = float(np.percentile(tx_rates, args.bully_percentile))
    print(f"[stats] IB TX rate percentile {args.bully_percentile}th = {bully_threshold:.1f} B/quanta/node")

    # --- Run the analysis sweep ---
    snapshot_rows = run_analysis(
        records=records,
        net=net,
        legacy_cfg=legacy_cfg,
        comm_pattern=comm_pattern,
        valid_nodes=valid_nodes,
        trace_quanta=trace_quanta,
        window_start=window_start,
        window_end=window_end,
        interval=args.interval,
        congestion_threshold=args.congestion_threshold,
        limit_concurrent=args.limit_concurrent,
        verbose=args.verbose,
    )

    # --- Write outputs ---
    out_prefix = Path(args.output)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    write_snapshots_csv(snapshot_rows, Path(f"{args.output}_snapshots.csv"))
    write_jobs_csv(
        records,
        bully_threshold=bully_threshold,
        congestion_threshold=args.congestion_threshold,
        path=Path(f"{args.output}_jobs.csv"),
    )
    write_top_congested_csv(snapshot_rows, args.top_n, Path(f"{args.output}_top_congested.csv"))

    # --- Print summary ---
    print_summary(records, snapshot_rows, args.congestion_threshold)


if __name__ == "__main__":
    main()

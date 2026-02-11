#!/usr/bin/env python3
"""
Frontier Scaling Experiments
=============================
Parallel benchmark runner for RAPS network simulation on OLCF Frontier.

Sweeps over:
  - System traces:   lassen (fat-tree), frontier (dragonfly)
  - Node counts:     100, 1_000, 10_000, 100_000
  - Time quanta:     0.1s, 1s, 10s, 60s
  - Repeats:         3 (for statistical averaging)

Each combination is an independent subprocess, launched in parallel up to
a configurable worker limit (default: number of CPUs).

Simulation duration: 12 hours (simulated time).

Output
------
  output/frontier_scaling/results.csv          — combined metrics
  output/frontier_scaling/<name>/              — per-experiment RAPS output
"""
import sys
import os
import csv
import time
import math
import itertools
import argparse
import multiprocessing as mp
from pathlib import Path
from datetime import timedelta

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raps.engine import Engine
from raps.sim_config import SingleSimConfig
from raps.system_config import SystemConfig, get_system_config
from raps.stats import get_engine_stats, get_network_stats
from raps.job import CommunicationPattern


# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------
SYSTEMS = ["lassen", "frontier"]

NODE_COUNTS = [100, 1_000, 10_000, 100_000]

TIME_QUANTA = [0.1, 1, 10, 60]   # seconds (Δt)

SIM_DURATION_HOURS = 12

NUM_REPEATS = 3

# Dragonfly params: p (hosts/router) * d (routers/group) * (a+1) (groups) >= nodes
# We pre-compute sensible (d, a, p) combos per node count.
DRAGONFLY_PARAMS = {
    100:     {"d": 10,  "a": 10,  "p": 1},   # 10*11*1 = 110 >= 100
    1_000:   {"d": 10,  "a": 10,  "p": 10},  # 10*11*10 = 1100 >= 1000
    10_000:  {"d": 24,  "a": 24,  "p": 16},  # 24*25*16 = 9600; bump a → 26 => 24*27*16 = 10368
    100_000: {"d": 48,  "a": 48,  "p": 48},  # 48*49*48 = 112896 >= 100000
}

# Fat-tree: k^3/4 hosts.  k must be even.
FATTREE_K = {
    100:     8,      # 128 hosts
    1_000:   14,     # 686 -> k=14 => 686; k=16 => 1024
    10_000:  28,     # 5488 -> k=28 => 5488; k=30 => 6750; k=32 => 8192
    100_000: 74,     # 101306 hosts
}


def _fattree_k_for_nodes(n: int) -> int:
    """Find smallest even k such that k^3/4 >= n."""
    k = 2
    while (k ** 3) // 4 < n:
        k += 2
    return k


def _dragonfly_params_for_nodes(n: int) -> dict:
    """Find sensible (d, a, p) such that d*(a+1)*p >= n, keeping values balanced."""
    # Try p from 1..64, pick the most cubic combo
    best = None
    for p in range(1, 65):
        # d*(a+1) >= ceil(n/p)
        needed = math.ceil(n / p)
        # d ~= a, so d*(d+1) >= needed => d ~= sqrt(needed)
        d = max(2, int(math.sqrt(needed)))
        while d * (d + 1) * p < n:
            d += 1
        a = d  # keep symmetric
        total = d * (a + 1) * p
        if total >= n:
            waste = total - n
            if best is None or waste < best[0]:
                best = (waste, d, a, p)
    _, d, a, p = best
    return {"d": d, "a": a, "p": p}


# Recompute to make sure our tables are correct
for n in NODE_COUNTS:
    FATTREE_K[n] = _fattree_k_for_nodes(n)
    DRAGONFLY_PARAMS[n] = _dragonfly_params_for_nodes(n)


# ---------------------------------------------------------------------------
# Single experiment runner  (runs in a worker process)
# ---------------------------------------------------------------------------

class _SimConfig(SingleSimConfig):
    """Thin wrapper to allow programmatic construction."""
    pass


def inject_network_traces(jobs, trace_quanta=15):
    """Inject synthetic network traces into jobs that lack them.

    Assigns a mix of communication patterns (ALL_TO_ALL and STENCIL_3D)
    and generates traffic proportional to job size.
    """
    import random as _rng
    for job in jobs:
        if job.ntx_trace is None or (hasattr(job.ntx_trace, '__len__') and len(job.ntx_trace) == 0):
            trace_len = max(1, int(job.expected_run_time / trace_quanta))
            nodes = max(1, job.nodes_required)

            # Alternate patterns based on job id
            if hash(job.id) % 2 == 0:
                job.comm_pattern = CommunicationPattern.STENCIL_3D
                traffic = nodes * 6 * 150.0  # 6 neighbors, ~150 bytes each
            else:
                job.comm_pattern = CommunicationPattern.ALL_TO_ALL
                traffic = nodes * (nodes - 1) * 50.0

            # Constant traffic with slight random variation
            job.ntx_trace = [traffic * _rng.uniform(0.8, 1.2) for _ in range(trace_len)]
            job.nrx_trace = [traffic * _rng.uniform(0.8, 1.2) for _ in range(trace_len)]


def _override_system_config(system_name: str, node_count: int) -> SystemConfig:
    """Load a base system config and override node count / network params."""
    base = get_system_config(system_name)
    data = base.model_dump(mode="json")

    # Adjust node grid so total_nodes >= node_count
    # Keep nodes_per_rack, adjust num_cdus * racks_per_cdu
    nodes_per_rack = data["system"]["nodes_per_rack"]
    racks_per_cdu = data["system"]["racks_per_cdu"]
    needed_racks = math.ceil(node_count / nodes_per_rack)
    needed_cdus = math.ceil(needed_racks / racks_per_cdu)
    data["system"]["num_cdus"] = needed_cdus
    data["system"]["missing_racks"] = []   # no missing racks in synthetic run
    data["system"]["down_nodes"] = []

    # Override network topology params for the new node count
    if data.get("network"):
        topo = data["network"]["topology"]
        if topo == "fat-tree":
            k = FATTREE_K[node_count]
            data["network"]["fattree_k"] = k
        elif topo == "dragonfly":
            params = DRAGONFLY_PARAMS[node_count]
            data["network"]["dragonfly_d"] = params["d"]
            data["network"]["dragonfly_a"] = params["a"]
            data["network"]["dragonfly_p"] = params["p"]

    return SystemConfig.model_validate(data)


def run_single_experiment(args_tuple):
    """
    Run one experiment configuration.

    Parameters are passed as a tuple for multiprocessing.
    Returns a dict of results.
    """
    system_name, node_count, delta_t, repeat_idx, output_root, sim_hours = args_tuple

    dt_str = f"{delta_t:g}"  # compact float format (0.1, 1, 10, 60)
    label = f"{system_name}_n{node_count}_dt{dt_str}_r{repeat_idx}"
    exp_output = Path(output_root) / label

    result = {
        "system": system_name,
        "node_count": node_count,
        "delta_t": delta_t,
        "repeat": repeat_idx,
        "label": label,
        "status": "FAILED",
    }

    try:
        # Build overridden SystemConfig
        sys_config = _override_system_config(system_name, node_count)

        # Build SimConfig
        sim_dict = {
            "system": system_name,
            "time": timedelta(hours=sim_hours),
            "time_delta": timedelta(seconds=delta_t),
            "simulate_network": True,
            "cooling": False,
            "uncertainties": False,
            "weather": False,
            "output": str(exp_output),
            "noui": True,
            "verbose": False,
            "debug": False,
            "workload": "synthetic",
            "policy": "fcfs",
            "arrival": "poisson",
            "numjobs": max(50, node_count // 20),
            "seed": 42 + repeat_idx,
            # Explicit distributions to avoid None errors in workload generator
            "jobsize_distribution": ["uniform"],
            "walltime_distribution": ["uniform"],
        }

        sim_config = _SimConfig(**sim_dict)
        # Inject our overridden SystemConfig
        sim_config._system_configs = [sys_config]

        # --- Run ---
        t_engine_start = time.perf_counter()
        engine = Engine(sim_config)

        # Inject network traces for synthetic jobs
        trace_quanta = sys_config.scheduler.trace_quanta
        inject_network_traces(engine.jobs, trace_quanta=trace_quanta)

        t_engine_ready = time.perf_counter()

        tick_count = 0
        for _ in engine.run_simulation():
            tick_count += 1

        t_sim_end = time.perf_counter()

        engine_init_time = t_engine_ready - t_engine_start
        sim_wall_time = t_sim_end - t_engine_ready
        total_wall_time = t_sim_end - t_engine_start
        simulated_seconds = sim_hours * 3600
        speedup = simulated_seconds / sim_wall_time if sim_wall_time > 0 else float("inf")
        per_tick = sim_wall_time / tick_count if tick_count > 0 else 0

        # Collect stats
        net_stats = get_network_stats(engine)

        result.update({
            "status": "OK",
            "ticks": tick_count,
            "engine_init_s": round(engine_init_time, 3),
            "sim_wall_s": round(sim_wall_time, 3),
            "total_wall_s": round(total_wall_time, 3),
            "per_tick_ms": round(per_tick * 1000, 3),
            "speedup": round(speedup, 1),
            "jobs_total": len(engine.jobs),
            "jobs_completed": engine.jobs_completed,
            "avg_net_util_pct": round(net_stats.get("avg_network_util", 0), 4),
            "avg_slowdown": round(net_stats.get("avg_per_job_slowdown", 1.0), 4),
            "max_slowdown": round(net_stats.get("max_per_job_slowdown", 1.0), 4),
            "avg_congestion": round(net_stats.get("avg_inter_job_congestion", 0), 6),
            "max_congestion": round(net_stats.get("max_inter_job_congestion", 0), 6),
        })

        print(f"[DONE] {label}  wall={total_wall_time:.1f}s  "
              f"speedup={speedup:.0f}x  tick={per_tick*1000:.2f}ms",
              flush=True)

    except Exception as e:
        result["error"] = str(e)
        print(f"[FAIL] {label}: {e}", file=sys.stderr, flush=True)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_experiment_grid(systems=None, node_counts=None, time_quanta=None,
                          num_repeats=None):
    """Build the full Cartesian product of experiment parameters."""
    systems = systems or SYSTEMS
    node_counts = node_counts or NODE_COUNTS
    time_quanta = time_quanta or TIME_QUANTA
    num_repeats = num_repeats or NUM_REPEATS

    grid = list(itertools.product(systems, node_counts, time_quanta,
                                  range(num_repeats)))
    return grid


def main():
    parser = argparse.ArgumentParser(
        description="RAPS Frontier Scaling Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--workers", "-j", type=int, default=None,
                        help="Max parallel workers (default: CPU count)")
    parser.add_argument("--output", "-o", type=str,
                        default="output/frontier_scaling",
                        help="Output directory (default: output/frontier_scaling)")
    parser.add_argument("--systems", nargs="+", default=None,
                        choices=SYSTEMS,
                        help="Systems to benchmark (default: all)")
    parser.add_argument("--nodes", nargs="+", type=int, default=None,
                        help="Node counts to sweep (default: 100 1000 10000 100000)")
    parser.add_argument("--dt", nargs="+", type=float, default=None,
                        help="Time quanta to sweep in seconds (default: 0.1 1 10 60)")
    parser.add_argument("--repeats", type=int, default=NUM_REPEATS,
                        help=f"Number of repeats (default: {NUM_REPEATS})")
    parser.add_argument("--duration", type=float, default=SIM_DURATION_HOURS,
                        help=f"Simulated hours per experiment (default: {SIM_DURATION_HOURS})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print experiment grid without running")

    args = parser.parse_args()

    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    grid = build_experiment_grid(
        systems=args.systems,
        node_counts=args.nodes,
        time_quanta=args.dt,
        num_repeats=args.repeats,
    )

    print("=" * 70)
    print("RAPS Frontier Scaling Experiments")
    print("=" * 70)
    print(f"  Systems:       {args.systems or SYSTEMS}")
    print(f"  Node counts:   {args.nodes or NODE_COUNTS}")
    print(f"  Time quanta:   {args.dt or TIME_QUANTA}")
    print(f"  Repeats:       {args.repeats}")
    print(f"  Total configs: {len(grid)}")
    print(f"  Sim duration:  {args.duration}h each")
    print(f"  Output:        {output_root}")
    print()

    # Print topology mapping
    print("  Fat-tree k values:    ", {n: FATTREE_K[n] for n in (args.nodes or NODE_COUNTS)})
    print("  Dragonfly (d,a,p):    ", {n: DRAGONFLY_PARAMS[n] for n in (args.nodes or NODE_COUNTS)})
    print()

    if args.dry_run:
        print("Experiments (dry run):")
        for i, (sys, nc, dt, rep) in enumerate(grid):
            print(f"  [{i+1:3d}] {sys:>10s}  nodes={nc:>7d}  dt={dt:>5.1f}s  repeat={rep}")
        print(f"\nTotal: {len(grid)} experiments")
        return

    workers = args.workers or mp.cpu_count()
    print(f"Launching {len(grid)} experiments with {workers} parallel workers...\n")

    sim_hours = args.duration

    # Build argument tuples
    tasks = [
        (sys_name, nc, dt, rep, str(output_root), sim_hours)
        for sys_name, nc, dt, rep in grid
    ]

    t_start = time.perf_counter()

    # Run experiments
    if workers == 1:
        # Sequential mode — avoids multiprocessing overhead
        results = [run_single_experiment(t) for t in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            results = pool.map(run_single_experiment, tasks)

    t_total = time.perf_counter() - t_start

    # Write CSV
    csv_path = output_root / "results.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    # Summary
    ok = sum(1 for r in results if r["status"] == "OK")
    fail = len(results) - ok

    print()
    print("=" * 70)
    print(f"All done in {t_total:.1f}s")
    print(f"  Succeeded: {ok}/{len(results)}")
    if fail:
        print(f"  Failed:    {fail}")
        for r in results:
            if r["status"] != "OK":
                print(f"    - {r['label']}: {r.get('error', 'unknown')}")
    print(f"  Results:   {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

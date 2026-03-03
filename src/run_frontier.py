#!/usr/bin/env python3
"""
Frontier Scaling Experiments
=============================
Benchmark runner for RAPS network simulation on OLCF Frontier.

Sweeps over:
  - System traces:   lassen (fat-tree), frontier (dragonfly)
  - Node counts:     100, 1_000, 10_000
  - Time quanta:     0.1s, 1s, 10s, 60s
  - Repeats:         3 (for statistical averaging)

Features:
  - Incremental CSV saving: results written after each experiment
  - Resume support: skips already-completed experiments on restart
  - Sorted execution: fast experiments first to maximize progress per job
  - Lassen uses real telemetry data; Frontier uses synthetic workloads

Simulation duration: 12 hours (simulated time).

Output
------
  output/frontier_scaling/results.csv          — combined metrics (incremental)
  output/frontier_scaling/<name>/              — per-experiment RAPS output
"""
import sys
import os
import csv
import time
import math
import signal
import itertools
import argparse
import filelock
import multiprocessing as mp
from pathlib import Path
from datetime import timedelta

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raps.engine import Engine
from raps.sim_config import SingleSimConfig
from raps.system_config import SystemConfig, get_system_config
from raps.stats import get_engine_stats, get_network_stats
from raps.job import CommunicationPattern, Job, job_dict
from raps.telemetry import Telemetry


# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------
SYSTEMS = ["lassen", "frontier"]

NODE_COUNTS = [100, 1_000, 10_000]

TIME_QUANTA = [0.1, 1, 10, 60]   # seconds (Δt)

SIM_DURATION_HOURS = 12

NUM_REPEATS = 3

# Estimated wall-time per experiment (seconds) for sorting fast-first.
# Based on observed data from previous runs.
ESTIMATED_WALL_TIME = {
    # (system, node_count, delta_t) -> estimated seconds
    ("frontier", 100, 60): 1,
    ("frontier", 100, 10): 2,
    ("frontier", 100, 1): 18,
    ("frontier", 100, 0.1): 160,
    ("frontier", 1000, 60): 1500,
    ("frontier", 1000, 10): 3,
    ("frontier", 1000, 1): 20,
    ("frontier", 1000, 0.1): 180,
    ("frontier", 10000, 60): 3600,
    ("frontier", 10000, 10): 600,
    ("frontier", 10000, 1): 1200,
    ("frontier", 10000, 0.1): 3600,
    ("lassen", 100, 60): 65,
    ("lassen", 100, 10): 65,
    ("lassen", 100, 1): 85,
    ("lassen", 100, 0.1): 200,
    ("lassen", 1000, 60): 65,
    ("lassen", 1000, 10): 70,
    ("lassen", 1000, 1): 85,
    ("lassen", 1000, 0.1): 285,
    ("lassen", 10000, 60): 600,
    ("lassen", 10000, 10): 600,
    ("lassen", 10000, 1): 900,
    ("lassen", 10000, 0.1): 3600,
}

# Dragonfly params: p (hosts/router) * d (routers/group) * (a+1) (groups) >= nodes
# We pre-compute sensible (d, a, p) combos per node count.
DRAGONFLY_PARAMS = {
    100:     {"d": 10,  "a": 10,  "p": 1},   # 10*11*1 = 110 >= 100
    1_000:   {"d": 10,  "a": 10,  "p": 10},  # 10*11*10 = 1100 >= 1000
    10_000:  {"d": 24,  "a": 24,  "p": 16},  # 24*25*16 = 9600
}

# Fat-tree: k^3/4 hosts.  k must be even.
FATTREE_K = {
    100:     8,      # 128 hosts
    1_000:   14,     # 686 -> k=16 => 1024
    10_000:  28,     # 5488 -> k=36 => 11664
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

            # Use STENCIL_3D for all synthetic jobs (O(N) vs O(N²) for ALL_TO_ALL)
            job.comm_pattern = CommunicationPattern.STENCIL_3D

            # Realistic HPC traffic: ~1.25 GB/s per node on 100 Gbps NICs
            # tx_volume = per-node send volume per trace_quanta; the coefficient
            # functions already iterate over all N nodes, so do NOT multiply by N.
            NIC_BW_BYTES_PER_S = 1.25e9  # 10 Gbps effective per node
            traffic = NIC_BW_BYTES_PER_S * trace_quanta

            # Constant traffic with slight random variation
            job.ntx_trace = [traffic * _rng.uniform(0.8, 1.2) for _ in range(trace_len)]
            job.nrx_trace = [traffic * _rng.uniform(0.8, 1.2) for _ in range(trace_len)]


def replicate_jobs_for_scale(jobs, target_nodes, sim_duration_sec, seed):
    """Replicate job list to fill a larger system over simulation time.
    
    Args:
        jobs: Original list of Job objects from telemetry
        target_nodes: Target number of nodes for the scaled system
        sim_duration_sec: Simulation duration in seconds (to spread job arrivals)
        seed: Random seed for deterministic scaling
    
    Returns:
        List of scaled Job objects with adjusted submit times
    """
    import random
    random.seed(seed)
    
    if not jobs:
        return []
    
    # Calculate how many times to repeat the job list
    avg_nodes_per_job = sum(j.nodes_required for j in jobs) / len(jobs)
    estimated_concurrent = target_nodes / avg_nodes_per_job
    replication_factor = max(1, int(estimated_concurrent / len(jobs) * 10))
    
    scaled_jobs = []
    job_id_offset = 0
    
    for rep in range(replication_factor):
        for orig_job in jobs:
            # Deep copy job attributes including trace metadata
            new_job = Job(job_dict(
                nodes_required=orig_job.nodes_required,
                name=orig_job.name,
                account=orig_job.account,
                id=job_id_offset + orig_job.id,
                priority=getattr(orig_job, 'priority', 0),
                cpu_trace=orig_job.cpu_trace,
                gpu_trace=orig_job.gpu_trace,
                ntx_trace=orig_job.ntx_trace,
                nrx_trace=orig_job.nrx_trace,
                submit_time=0,  # Will adjust below
                time_limit=orig_job.time_limit,
                expected_run_time=orig_job.expected_run_time,
                trace_quanta=getattr(orig_job, 'trace_quanta', 15),
                trace_time=getattr(orig_job, 'trace_time', None),
                trace_start_time=getattr(orig_job, 'trace_start_time', 0),
                trace_end_time=getattr(orig_job, 'trace_end_time', 0),
                comm_pattern=getattr(orig_job, 'comm_pattern', CommunicationPattern.ALL_TO_ALL),
            ))
            
            # Spread submit times across simulation duration
            new_submit = (rep * len(jobs) + len(scaled_jobs)) * (sim_duration_sec / (replication_factor * len(jobs)))
            new_job.submit_time = int(new_submit + random.uniform(-60, 60))
            
            if new_job.submit_time < 0:
                new_job.submit_time = 0
            
            scaled_jobs.append(new_job)
        
        job_id_offset += 10000
    
    # Trim to simulation duration
    scaled_jobs = [j for j in scaled_jobs if j.submit_time < sim_duration_sec]
    
    return scaled_jobs


def load_and_scale_lassen_jobs(node_count, sys_config, sim_hours, repeat_idx):
    """Load real lassen telemetry and scale to target node count.
    
    Args:
        node_count: Target number of nodes
        sys_config: SystemConfig object
        sim_hours: Simulation duration in hours
        repeat_idx: Repeat index for seeding
    
    Returns:
        List of Job objects (original or scaled)
    """
    # Try multiple possible data locations
    possible_paths = [
        Path("/opt/data/lassen/Lassen-Supercomputer-Job-Dataset"),
        Path("/lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26/data/lassen/Lassen-Supercomputer-Job-Dataset"),
        Path(__file__).parent.parent / "data" / "lassen" / "Lassen-Supercomputer-Job-Dataset",
    ]
    
    lassen_data_path = None
    for path in possible_paths:
        if path.exists():
            lassen_data_path = path
            break
    
    if lassen_data_path is None:
        raise FileNotFoundError(
            f"Lassen data not found. Tried:\n" +
            "\n".join(f"  - {p}" for p in possible_paths) +
            f"\n\nPlease download data with: bash scripts/download_lassen_data.sh"
        )
    
    telemetry_args = {
        'system': 'lassen',
        'config': sys_config.get_legacy(),
        'time': sim_hours * 3600,
        'start': None,  # Use dataset start
        'arrival': 'prescribed',  # Use original submit times
    }
    td = Telemetry(**telemetry_args)
    wd = td.load_from_files([lassen_data_path])
    original_jobs = wd.jobs
    
    print(f"  Loaded {len(original_jobs)} jobs from lassen telemetry ({lassen_data_path})", flush=True)
    
    # Always use replicate_jobs_for_scale to ensure adequate job count.
    # A simple node-count filter leaves almost no jobs for small node_count
    # (e.g., n=100/1000 vs Lassen's 4626-node jobs).
    print(f"  Scaling jobs to fill {node_count} nodes...", flush=True)
    scaled_jobs = replicate_jobs_for_scale(
        original_jobs,
        target_nodes=node_count,
        sim_duration_sec=sim_hours * 3600,
        seed=42 + repeat_idx
    )
    print(f"  Scaled to {len(scaled_jobs)} jobs", flush=True)
    
    return scaled_jobs


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

    # Cap max_nodes_per_job to actual node count so synthetic jobs can be scheduled
    if data.get("scheduler"):
        data["scheduler"]["max_nodes_per_job"] = min(
            data["scheduler"].get("max_nodes_per_job", node_count),
            node_count
        )

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
            # Use minimal routing for scaling benchmarks — UGAL is O(N^2 * groups)
            # per tick and cannot be cached, making it orders of magnitude slower.
            # UC1 still evaluates adaptive vs minimal routing separately.
            data["network"]["routing_algorithm"] = "minimal"

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
        
        # Common SimConfig fields for both lassen and frontier
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
            "numjobs": max(200, node_count // 5),
            "seed": 42 + repeat_idx,
            "jobsize_distribution": ["uniform"],
            "walltime_distribution": ["uniform"],
        }
        
        sim_config = _SimConfig(**sim_dict)
        sim_config._system_configs = [sys_config]
        
        t_engine_start = time.perf_counter()
        engine = Engine(sim_config)
        
        # Branch: Lassen overrides synthetic jobs with real telemetry data
        if system_name == "lassen":
            jobs = load_and_scale_lassen_jobs(node_count, sys_config, sim_hours, repeat_idx)
            engine.jobs = jobs
            engine.total_initial_jobs = len(jobs)
        
        # Inject network traces for all jobs
        trace_quanta = sys_config.scheduler.trace_quanta
        inject_network_traces(engine.jobs, trace_quanta=trace_quanta)

        # Reduce congestion computation frequency for large systems
        if node_count >= 10000:
            engine._congestion_interval = 100  # every 100 ticks instead of 10

        # --- Checkpoint / Resume ---
        ckpt_dir = exp_output / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "engine.ckpt"
        resuming = False

        if ckpt_path.exists():
            try:
                engine.load_checkpoint(ckpt_path)
                resuming = True
                pct = ((engine.current_timestep - engine.timestep_start) /
                       max(1, engine.timestep_end - engine.timestep_start) * 100)
                print(f"[RESUME] {label}  from timestep {engine.current_timestep} "
                      f"({pct:.1f}% done)", flush=True)
            except Exception as ckpt_err:
                print(f"[WARN] {label}: checkpoint load failed ({ckpt_err}), "
                      f"starting fresh", flush=True)
                resuming = False

        t_engine_ready = time.perf_counter()

        # Checkpoint every CKPT_INTERVAL_S seconds of wall time
        CKPT_INTERVAL_S = 300  # 5 minutes
        last_ckpt_wall = time.perf_counter()

        # Graceful shutdown flag — set by signal handler
        _shutdown_requested = False
        _prev_sigusr1 = signal.getsignal(signal.SIGUSR1)
        _prev_sigterm = signal.getsignal(signal.SIGTERM)

        def _request_shutdown(signum, frame):
            nonlocal _shutdown_requested
            _shutdown_requested = True
            print(f"\n[SIGNAL] {label}: received signal {signum}, "
                  f"will checkpoint and exit after current tick", flush=True)

        # Install signal handlers (only in main process, not in pool workers)
        if mp.current_process().name == 'MainProcess' or os.environ.get('RAPS_SINGLE_WORKER'):
            signal.signal(signal.SIGUSR1, _request_shutdown)
            signal.signal(signal.SIGTERM, _request_shutdown)

        tick_count = 0
        interrupted = False
        for _ in engine.run_simulation(resume=resuming):
            tick_count += 1

            # Periodic checkpoint
            now = time.perf_counter()
            if now - last_ckpt_wall >= CKPT_INTERVAL_S:
                engine.save_checkpoint(ckpt_path)
                last_ckpt_wall = now

            # Check for graceful shutdown
            if _shutdown_requested:
                print(f"[CKPT] {label}: saving checkpoint at tick {tick_count} "
                      f"(timestep {engine.current_timestep})...", flush=True)
                engine.save_checkpoint(ckpt_path)
                interrupted = True
                break

        # Restore original signal handlers
        if mp.current_process().name == 'MainProcess' or os.environ.get('RAPS_SINGLE_WORKER'):
            signal.signal(signal.SIGUSR1, _prev_sigusr1)
            signal.signal(signal.SIGTERM, _prev_sigterm)

        t_sim_end = time.perf_counter()

        engine_init_time = t_engine_ready - t_engine_start
        sim_wall_time = t_sim_end - t_engine_ready
        total_wall_time = t_sim_end - t_engine_start
        simulated_seconds = sim_hours * 3600
        speedup = simulated_seconds / sim_wall_time if sim_wall_time > 0 else float("inf")
        per_tick = sim_wall_time / tick_count if tick_count > 0 else 0

        # Collect stats
        net_stats = get_network_stats(engine)

        if interrupted:
            pct = ((engine.current_timestep - engine.timestep_start) /
                   max(1, engine.timestep_end - engine.timestep_start) * 100)
            result.update({
                "status": "INTERRUPTED",
                "error": f"Checkpoint saved at {pct:.1f}% ({tick_count} ticks)",
            })
            print(f"[INTERRUPTED] {label}  checkpoint at {pct:.1f}% "
                  f"({tick_count} ticks, wall={total_wall_time:.1f}s)",
                  flush=True)
        else:
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

            # Clean up checkpoint on success
            if ckpt_path.exists():
                ckpt_path.unlink()

            print(f"[DONE] {label}  wall={total_wall_time:.1f}s  "
                  f"speedup={speedup:.0f}x  tick={per_tick*1000:.2f}ms",
                  flush=True)

    except Exception as e:
        # Truncate error to avoid CSV corruption from huge tracebacks
        err_msg = str(e).split('\n')[0][:200]
        result["error"] = err_msg
        print(f"[FAIL] {label}: {err_msg}", file=sys.stderr, flush=True)

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


def sort_grid_by_estimated_time(grid):
    """Sort experiments from fastest to slowest based on estimated wall time."""
    def est_time(item):
        sys_name, nc, dt, rep = item
        return ESTIMATED_WALL_TIME.get((sys_name, nc, dt), 9999)
    return sorted(grid, key=est_time)


# ---------------------------------------------------------------------------
# Incremental CSV helpers
# ---------------------------------------------------------------------------
CSV_FIELDNAMES = [
    "system", "node_count", "delta_t", "repeat", "label", "status",
    "ticks", "engine_init_s", "sim_wall_s", "total_wall_s",
    "per_tick_ms", "speedup", "jobs_total", "jobs_completed",
    "avg_net_util_pct", "avg_slowdown", "max_slowdown",
    "avg_congestion", "max_congestion", "error",
]


def load_completed_labels(csv_path: Path) -> set:
    """Load labels of already-completed experiments from results CSV."""
    completed = set()
    if csv_path.exists():
        try:
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("status") == "OK":
                        completed.add(row["label"])
        except Exception:
            pass  # Corrupted CSV, start fresh
    return completed


def ensure_csv_header(csv_path: Path):
    """Create CSV with header if it doesn't exist yet."""
    lock_path = csv_path.with_suffix(".csv.lock")
    with filelock.FileLock(lock_path, timeout=30):
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                writer.writeheader()


def append_result_to_csv(csv_path: Path, result: dict):
    """Append a single result row to the CSV file (thread/process safe)."""
    lock_path = csv_path.with_suffix(".csv.lock")
    with filelock.FileLock(lock_path, timeout=30):
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES,
                                    extrasaction="ignore")
            writer.writerow(result)


def run_and_save(args_tuple_with_csv):
    """Run a single experiment and immediately save the result to CSV.

    INTERRUPTED results are NOT written to CSV so they will be retried
    on the next SLURM job (checkpoint is preserved on disk).
    """
    *task_args, csv_path_str = args_tuple_with_csv
    csv_path = Path(csv_path_str)
    result = run_single_experiment(tuple(task_args))
    if result["status"] != "INTERRUPTED":
        append_result_to_csv(csv_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="RAPS Frontier Scaling Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--workers", "-j", type=int, default=1,
                        help="Max parallel workers (default: 1 for sequential)")
    parser.add_argument("--output", "-o", type=str,
                        default="output/frontier_scaling",
                        help="Output directory (default: output/frontier_scaling)")
    parser.add_argument("--systems", nargs="+", default=None,
                        choices=SYSTEMS,
                        help="Systems to benchmark (default: all)")
    parser.add_argument("--nodes", nargs="+", type=int, default=None,
                        help="Node counts to sweep (default: 100 1000 10000)")
    parser.add_argument("--dt", nargs="+", type=float, default=None,
                        help="Time quanta to sweep in seconds (default: 0.1 1 10 60)")
    parser.add_argument("--repeats", type=int, default=NUM_REPEATS,
                        help=f"Number of repeats (default: {NUM_REPEATS})")
    parser.add_argument("--duration", type=float, default=SIM_DURATION_HOURS,
                        help=f"Simulated hours per experiment (default: {SIM_DURATION_HOURS})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print experiment grid without running")
    parser.add_argument("--no-resume", action="store_true",
                        help="Don't skip already-completed experiments")

    args = parser.parse_args()

    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "results.csv"

    grid = build_experiment_grid(
        systems=args.systems,
        node_counts=args.nodes,
        time_quanta=args.dt,
        num_repeats=args.repeats,
    )

    # Sort by estimated speed (fast first)
    grid = sort_grid_by_estimated_time(grid)

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
    print(f"  Results CSV:   {csv_path}")
    print()

    # Print topology mapping
    nc_list = args.nodes or NODE_COUNTS
    print("  Fat-tree k values:    ", {n: FATTREE_K[n] for n in nc_list if n in FATTREE_K})
    print("  Dragonfly (d,a,p):    ", {n: DRAGONFLY_PARAMS[n] for n in nc_list if n in DRAGONFLY_PARAMS})
    print()

    # Check for already-completed experiments (resume support)
    if not args.no_resume:
        completed = load_completed_labels(csv_path)
        if completed:
            print(f"  Resume mode: {len(completed)} experiments already completed, skipping them.")
            print()
    else:
        completed = set()

    # Filter out completed experiments
    pending_grid = []
    for sys_name, nc, dt, rep in grid:
        dt_str = f"{dt:g}"
        label = f"{sys_name}_n{nc}_dt{dt_str}_r{rep}"
        if label not in completed:
            pending_grid.append((sys_name, nc, dt, rep))

    if not pending_grid:
        print("All experiments already completed! Nothing to do.")
        print(f"Results: {csv_path}")
        return

    if args.dry_run:
        print(f"Experiments to run ({len(pending_grid)} pending, {len(completed)} completed):")
        for i, (sys, nc, dt, rep) in enumerate(pending_grid):
            est = ESTIMATED_WALL_TIME.get((sys, nc, dt), "?")
            print(f"  [{i+1:3d}] {sys:>10s}  nodes={nc:>7d}  dt={dt:>5.1f}s  repeat={rep}  (~{est}s)")
        total_est = sum(ESTIMATED_WALL_TIME.get((s, n, d), 600) for s, n, d, _ in pending_grid)
        print(f"\nTotal: {len(pending_grid)} experiments, estimated {total_est/3600:.1f}h sequential")
        return

    workers = args.workers
    sim_hours = args.duration

    # Ensure CSV header exists before workers start writing
    ensure_csv_header(csv_path)

    print(f"Running {len(pending_grid)} experiments ({len(completed)} already done) "
          f"with {workers} worker(s)...\n")

    # Build argument tuples
    tasks = [
        (sys_name, nc, dt, rep, str(output_root), sim_hours, str(csv_path))
        for sys_name, nc, dt, rep in pending_grid
    ]

    t_start = time.perf_counter()
    ok_count = 0
    fail_count = 0

    if workers == 1:
        # Sequential mode — simple, no multiprocessing overhead
        for i, task in enumerate(tasks):
            sys_name, nc, dt, rep = task[0], task[1], task[2], task[3]
            dt_str = f"{dt:g}"
            label = f"{sys_name}_n{nc}_dt{dt_str}_r{rep}"
            print(f"[{i+1}/{len(tasks)}] Starting {label}...", flush=True)
            result = run_and_save(task)
            if result["status"] == "OK":
                ok_count += 1
            else:
                fail_count += 1
            elapsed = time.perf_counter() - t_start
            print(f"  Progress: {ok_count+fail_count}/{len(tasks)} done, "
                  f"{ok_count} OK, {fail_count} failed, "
                  f"elapsed {elapsed/60:.1f}min\n", flush=True)
            # If the experiment was interrupted (checkpoint saved), exit now so
            # the SLURM shell handler can resubmit.  Remaining tasks will be
            # picked up automatically on the next job via resume logic.
            if result["status"] == "INTERRUPTED":
                sys.exit(99)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for result in pool.imap_unordered(run_and_save, tasks):
                if result["status"] == "OK":
                    ok_count += 1
                else:
                    fail_count += 1
                elapsed = time.perf_counter() - t_start
                print(f"  Progress: {ok_count+fail_count}/{len(tasks)} done, "
                      f"{ok_count} OK, {fail_count} failed, "
                      f"elapsed {elapsed/60:.1f}min", flush=True)

    t_total = time.perf_counter() - t_start

    # Summary
    total_completed = len(completed) + ok_count
    total_target = len(grid)

    print()
    print("=" * 70)
    print(f"Session done in {t_total:.1f}s ({t_total/60:.1f}min)")
    print(f"  This session:  {ok_count} succeeded, {fail_count} failed")
    print(f"  Overall:       {total_completed}/{total_target} experiments completed")
    if total_completed < total_target:
        remaining = total_target - total_completed
        print(f"  Remaining:     {remaining} experiments — resubmit to continue")
    else:
        print(f"  ALL DONE!")
    print(f"  Results:       {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

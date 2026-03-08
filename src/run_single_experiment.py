#!/usr/bin/env python3
"""
Run a single experiment with checkpoint/resume support.
Optimized for large dt=0.1s experiments.
"""
import sys
import os
import math
import time
import signal
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raps.engine import Engine
from raps.sim_config import SingleSimConfig
from raps.system_config import SystemConfig, get_system_config
from raps.stats import get_engine_stats, get_network_stats
from raps.job import CommunicationPattern
import filelock
import csv
import random as _rng


# ---------------------------------------------------------------------------
# Topology parameter tables (must match run_frontier.py)
# ---------------------------------------------------------------------------
CIRCULANT_PARAMS = {
    100:     {"groups": 10,  "d": 5,   "p": 2,  "inter": 4},   # 10*5*2  = 100,  port=10
    1_000:   {"groups": 32,  "d": 16,  "p": 2,  "inter": 14},  # 32*16*2 = 1024, port=31
    10_000:  {"groups": 74,  "d": 32,  "p": 5,  "inter": 27},  # 74*32*5 = 11840,port=63
}

FATTREE_K = {
    100:     8,
    1_000:   16,
    10_000:  36,
}

NODE_COUNTS = [100, 1_000, 10_000]


def _fattree_k_for_nodes(n: int) -> int:
    k = 2
    while (k ** 3) // 4 < n:
        k += 2
    return k


def _circulant_params_for_nodes(n: int) -> dict:
    for P in [2, 3, 4, 5, 6]:
        needed = math.ceil(n / P)
        R = max(4, int(math.sqrt(needed / 2)))
        G = math.ceil(needed / R)
        H = min(R - 1, 64 - P - (R - 1))
        if H < 2:
            continue
        if G * R * P >= n and P + (R - 1) + H <= 64:
            return {"groups": G, "d": R, "p": P, "inter": H}


for _n in NODE_COUNTS:
    FATTREE_K[_n] = _fattree_k_for_nodes(_n)
    CIRCULANT_PARAMS[_n] = _circulant_params_for_nodes(_n)


def _override_system_config(system_name: str, node_count: int) -> SystemConfig:
    """Load base config and override nodes, topology params, routing."""
    base = get_system_config(system_name)
    data = base.model_dump(mode="json")

    nodes_per_rack = data["system"]["nodes_per_rack"]
    racks_per_cdu = data["system"]["racks_per_cdu"]
    needed_racks = math.ceil(node_count / nodes_per_rack)
    needed_cdus = math.ceil(needed_racks / racks_per_cdu)
    data["system"]["num_cdus"] = needed_cdus
    data["system"]["missing_racks"] = []
    data["system"]["down_nodes"] = []

    if data.get("scheduler"):
        data["scheduler"]["max_nodes_per_job"] = min(
            data["scheduler"].get("max_nodes_per_job", node_count),
            node_count,
        )

    if data.get("network"):
        topo = data["network"]["topology"]
        if topo == "fat-tree":
            k = FATTREE_K.get(node_count) or _fattree_k_for_nodes(node_count)
            data["network"]["fattree_k"] = k
        elif topo == "dragonfly":
            params = CIRCULANT_PARAMS.get(node_count) or _circulant_params_for_nodes(node_count)
            data["network"]["dragonfly_groups"] = params["groups"]
            data["network"]["dragonfly_d"] = params["d"]
            data["network"]["dragonfly_p"] = params["p"]
            data["network"]["dragonfly_inter"] = params["inter"]
            data["network"].pop("dragonfly_a", None)
            data["network"]["routing_algorithm"] = "minimal"

    return SystemConfig.model_validate(data)

_shutdown_requested = False

def signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print(f"\n[SIGNAL] Received signal {signum}, will checkpoint and exit", flush=True)

signal.signal(signal.SIGUSR1, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def main():
    parser = argparse.ArgumentParser(description="Run single RAPS experiment")
    parser.add_argument("--system", required=True, choices=["lassen", "frontier"])
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--duration", type=float, required=True, help="Simulation hours")
    parser.add_argument("--output", type=Path, required=True)
    
    args = parser.parse_args()
    
    label = f"{args.system}_n{args.nodes}_dt{args.dt:g}_r{args.repeat}"
    exp_output = args.output / label
    exp_output.mkdir(parents=True, exist_ok=True)
    
    print(f"=" * 70)
    print(f"Running: {label}")
    print(f"=" * 70)
    print(f"System: {args.system}")
    print(f"Nodes: {args.nodes}")
    print(f"Time quantum (dt): {args.dt}s")
    print(f"Repeat: {args.repeat}")
    print(f"Sim duration: {args.duration}h")
    print(f"Output: {exp_output}")
    print()
    
    # Build config
    if args.dt == 0.1:
        time_delta_s = 0.1
        time_unit_s = 0.1
    else:
        time_delta_s = args.dt
        time_unit_s = 1.0

    num_jobs = max(200, args.nodes // 5)

    sim_config = SingleSimConfig(
        system=args.system,
        time=args.duration * 3600,
        time_delta=time_delta_s,
        time_unit=time_unit_s,
        numjobs=num_jobs,
        seed=42 + args.repeat,
        output=exp_output,
        simulate_network=True,
        noui=True,
        workload="synthetic",
        policy="fcfs",
        arrival="poisson",
        allocation="contiguous",
        jobsize_distribution=["uniform"],
        walltime_distribution=["uniform"],
    )

    # Override system config: correct node count, topology params, routing, etc.
    sys_config = _override_system_config(args.system, args.nodes)
    sim_config._system_configs = [sys_config]

    engine = Engine(sim_config)

    # Inject network traces for synthetic jobs (basic.py sets ntx_trace=None)
    trace_quanta = sys_config.scheduler.trace_quanta if hasattr(sys_config, 'scheduler') and hasattr(sys_config.scheduler, 'trace_quanta') else 15
    NIC_BW_BYTES_PER_S = 1.25e9
    for job in engine.jobs:
        if job.ntx_trace is None or (hasattr(job.ntx_trace, '__len__') and len(job.ntx_trace) == 0):
            trace_len = max(1, int(job.expected_run_time / trace_quanta))
            job.comm_pattern = CommunicationPattern.STENCIL_3D
            traffic = NIC_BW_BYTES_PER_S * trace_quanta
            job.ntx_trace = [traffic * _rng.uniform(0.8, 1.2) for _ in range(trace_len)]
            job.nrx_trace = [traffic * _rng.uniform(0.8, 1.2) for _ in range(trace_len)]

    if args.nodes >= 10000:
        engine._congestion_interval = 100

    # Checkpoint setup
    ckpt_dir = exp_output / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "engine.ckpt"
    
    resuming = False
    if ckpt_path.exists():
        try:
            engine.load_checkpoint(ckpt_path)
            resuming = True
            pct = 100.0 * engine.current_timestep / engine.sim_config.time_int
            print(f"[RESUME] from timestep {engine.current_timestep} ({pct:.1f}% done)")
            print()
        except Exception as e:
            print(f"[WARN] Checkpoint load failed: {e}, starting fresh")
            print()
    
    # Run simulation
    CKPT_INTERVAL_S = 600  # 10 minutes (increased from 5)
    last_ckpt_wall = time.perf_counter()
    tick_count = 0
    wall_start = time.perf_counter()
    
    print("Starting simulation...")
    print(f"Checkpoint interval: {CKPT_INTERVAL_S}s")
    print()
    
    try:
        for tick_data in engine.run_simulation(resume=resuming):
            tick_count += 1
            
            # Checkpoint periodically or on shutdown signal
            if time.perf_counter() - last_ckpt_wall >= CKPT_INTERVAL_S or _shutdown_requested:
                engine.save_checkpoint(ckpt_path)
                last_ckpt_wall = time.perf_counter()
                
                pct = 100.0 * engine.current_timestep / engine.sim_config.time_int
                elapsed = time.perf_counter() - wall_start
                print(f"[CKPT] at tick {tick_count}, {pct:.1f}% done, {elapsed:.1f}s elapsed")

                if _shutdown_requested:
                    print(f"[SHUTDOWN] Checkpointed at {pct:.1f}%, exiting...")
                    sys.exit(99)

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving checkpoint...")
        engine.save_checkpoint(ckpt_path)
        pct = 100.0 * engine.current_timestep / engine.sim_config.time_int
        print(f"[INTERRUPTED] Checkpoint saved at {pct:.1f}%")
        sys.exit(99)
    
    # Simulation complete
    wall_time = time.perf_counter() - wall_start
    sim_seconds = engine.current_timestep * engine.sim_config.time_unit.total_seconds()
    speedup = sim_seconds / wall_time if wall_time > 0 else 0
    
    print()
    print(f"[DONE] {label}")
    print(f"  Wall time: {wall_time:.1f}s")
    print(f"  Simulated: {sim_seconds:.1f}s")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Ticks: {tick_count}")
    print()
    
    # Save results
    results_csv = args.output / "results.csv"
    lock_path = args.output / "results.csv.lock"
    
    # Collect stats
    engine_stats = get_engine_stats(engine)
    net_stats = get_network_stats(engine) if engine.network_model else {}
    
    row = {
        "system": args.system,
        "node_count": args.nodes,
        "delta_t": args.dt,
        "repeat": args.repeat,
        "label": label,
        "status": "OK",
        "ticks": tick_count,
        "network_init_s": engine_stats.get("network_init_s", 0),
        "sim_wall_s": engine_stats.get("sim_wall_s", wall_time),
        "total_wall_s": engine_stats.get("total_wall_s", wall_time),
        "per_tick_ms": engine_stats.get("per_tick_ms", wall_time * 1000 / tick_count),
        "speedup": speedup,
        "num_jobs": num_jobs,
        "jobs_completed": engine_stats.get("jobs_completed", 0),
        "init_overhead_s": engine_stats.get("init_overhead_s", 0),
        "dilated_pct": net_stats.get("dilated_pct", 0),
        "avg_slowdown": net_stats.get("avg_slowdown", 1.0),
        "max_slowdown": net_stats.get("max_slowdown", 1.0),
        "avg_congestion": net_stats.get("avg_congestion", 0),
        "max_congestion": net_stats.get("max_congestion", 0),
    }
    
    # Write to CSV
    with filelock.FileLock(str(lock_path), timeout=30):
        write_header = not results_csv.exists()
        with open(results_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    
    print(f"Results saved to {results_csv}")
    
    # Clean up checkpoint on success
    if ckpt_path.exists():
        ckpt_path.unlink()
        print(f"Checkpoint removed (experiment complete)")
    
    print()
    print(f"[SUCCESS] {label}")


if __name__ == "__main__":
    main()

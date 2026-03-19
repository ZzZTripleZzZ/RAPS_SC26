#!/usr/bin/env python3
"""
NRAPS Bully-Victim Interference Study
======================================

Validates NRAPS inter-job interference modeling by running controlled experiments
where a bandwidth-heavy "bully" job (ALL_TO_ALL) co-schedules with "victim" jobs
(STENCIL_3D) and we measure victim slowdown as a function of bully size.

This generates the E2 validation data for the SC26 paper.

Usage:
    python src/run_interference_study.py --system frontier
    python src/run_interference_study.py --system lassen
    python src/run_interference_study.py --system frontier lassen
    python src/run_interference_study.py --system frontier --bully-sizes 64 128 256 512
    python src/run_interference_study.py --system frontier --victim-count 4 --victim-nodes 64
    python src/run_interference_study.py --system frontier --baseline  # bully_nodes=0 sanity check
"""

import sys
import time
import json
import math
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from raps.engine import Engine
from raps.sim_config import SingleSimConfig
from raps.system_config import SystemConfig, get_system_config
from raps.job import Job, job_dict, CommunicationPattern
from raps.stats import get_network_stats
from raps.workloads.bully_victim import generate_bully_victim

# Reuse helpers from run_use_cases
sys.path.insert(0, str(Path(__file__).parent))
from run_use_cases import (
    clone_jobs, _override_system_config_uc, override_system_routing,
    _get_topology, _all_routing_algos_for_system, UCSimConfig,
    CIRCULANT_PARAMS, FATTREE_K, _circulant_params_for_nodes, _fattree_k_for_nodes,
)


OUTPUT_DIR = PROJECT_ROOT / "output" / "interference"

# Default bully sizes to sweep (in nodes)
DEFAULT_BULLY_SIZES = [0, 32, 64, 128, 256, 512]

# System-level parameters
SYSTEM_PARAMS = {
    "frontier": {
        "node_count": 1000,
        "routing": "minimal",
        "delta_t": 15,
        "duration_minutes": 10,
    },
    "lassen": {
        "node_count": 1000,
        "routing": "minimal",
        "delta_t": 20,
        "duration_minutes": 10,
    },
    "bluewaters": {
        "node_count": 1000,
        "routing": None,   # torus3d uses dor_xyz implicitly
        "delta_t": 20,
        "duration_minutes": 10,
    },
}


@dataclass
class InterferenceResult:
    """Results from one bully-victim simulation run."""
    system: str
    routing: str
    bully_nodes: int
    victim_count: int
    victim_nodes: int
    duration: int
    total_nodes: int
    load_fraction: float          # bully_nodes / total_nodes
    # Victim metrics (averaged across all victim jobs)
    avg_victim_slowdown: float
    max_victim_slowdown: float
    avg_victim_stall_ratio: float
    # Bully metrics
    bully_slowdown: float
    # System-level
    avg_congestion: float
    max_congestion: float
    avg_net_util: float
    # Timing
    wall_time: float
    simulated_seconds: float


def run_one_experiment(
    system: str,
    bully_nodes: int,
    victim_count: int = 4,
    victim_nodes: int = 64,
    duration: int = 300,
    routing: Optional[str] = "minimal",
    node_count: int = 1000,
    delta_t: int = 15,
    duration_minutes: int = 10,
    tx_fraction: float = 1.0,
    allocation: str = "random",
) -> Optional[InterferenceResult]:
    """Run a single bully-victim DES experiment.

    Parameters
    ----------
    allocation : str
        Node allocation strategy.  Default is 'random' so that bully and
        victim jobs are scattered across the network, ensuring inter-group/
        inter-rack link sharing (the prerequisite for interference).
        Using 'contiguous' on fat-tree can cause zero interference because
        co-located jobs share only intra-ToR links.
    """
    total_nodes_needed = bully_nodes + victim_count * victim_nodes
    if total_nodes_needed > node_count:
        print(f"  [SKIP] bully={bully_nodes} + {victim_count}×{victim_nodes} = "
              f"{total_nodes_needed} > {node_count} nodes")
        return None

    # Get legacy config for workload generation
    base_sys_cfg = _override_system_config_uc(system, node_count)
    legacy_cfg = base_sys_cfg.get_legacy()
    trace_quanta = legacy_cfg.get("TRACE_QUANTA", delta_t)

    # Build jobs
    if bully_nodes == 0:
        # Baseline: only victim jobs, no bully
        from raps.network import max_throughput_per_tick
        per_tick_bw = max_throughput_per_tick(legacy_cfg, trace_quanta)
        victim_bw = 0.35 * per_tick_bw
        trace_len = max(1, duration // trace_quanta)
        jobs = []
        for k in range(victim_count):
            victim = Job(job_dict(
                id=k + 1,
                name=f"victim_stencil_{k}",
                account="interference_study",
                nodes_required=victim_nodes,
                scheduled_nodes=[],
                cpu_trace=[0.7] * trace_len,
                gpu_trace=[0.6] * trace_len,
                ntx_trace=[victim_bw] * trace_len,
                nrx_trace=[victim_bw] * trace_len,
                comm_pattern=CommunicationPattern.STENCIL_3D,
                message_size=65536,
                trace_quanta=trace_quanta,
                submit_time=0,
                expected_run_time=duration,
                time_limit=duration * 2,
                end_state="COMPLETED",
            ))
            jobs.append(victim)
    else:
        jobs = generate_bully_victim(
            legacy_cfg=legacy_cfg,
            trace_quanta=trace_quanta,
            bully_nodes=bully_nodes,
            victim_count=victim_count,
            victim_nodes=victim_nodes,
            duration=duration,
            tx_fraction=tx_fraction,
        )

    # duration_minutes must cover job duration + possible dilation slack
    effective_dur_min = max(duration_minutes, int(math.ceil(duration * 1.5 / 60)))

    config_dict = {
        'system': system,
        'time': timedelta(minutes=effective_dur_min),
        'time_delta': timedelta(seconds=delta_t),
        'simulate_network': True,
        'cooling': False,
        'uncertainties': False,
        'weather': False,
        'output': 'none',
        'noui': True,
        'verbose': False,
        'debug': False,
        'workload': 'synthetic',
        'policy': 'fcfs',
        'allocation': allocation,
        'allocation_seed': 42,
        'numjobs': len(jobs),
        'backfill': None,
        'jobsize_distribution': ['uniform'],
        'walltime_distribution': ['uniform'],
    }

    t_start = time.perf_counter()
    try:
        sim_config = UCSimConfig(**config_dict)
        sim_config._system_configs = [base_sys_cfg]

        topo = _get_topology(system)
        if routing and topo != 'torus3d':
            override_system_routing(sim_config, routing)

        engine = Engine(sim_config)
        engine.jobs = clone_jobs(jobs)

        for _ in engine.run_simulation(autoshutdown=True):
            pass

        t_end = time.perf_counter()
        wall_time = t_end - t_start
        simulated_seconds = engine.current_timestep

        net_stats = get_network_stats(engine)

        # Separate bully and victim jobs
        all_jobs = engine.jobs
        victim_slowdowns = []
        victim_stall_ratios = []
        bully_slowdown = 1.0

        for job in all_jobs:
            sf = getattr(job, 'slowdown_factor', 1.0)
            sr = getattr(job, 'stall_ratio', 0.0)
            if bully_nodes > 0 and job.id == 1:
                bully_slowdown = sf
            else:
                victim_slowdowns.append(sf)
                victim_stall_ratios.append(sr)

        avg_victim_sf = float(np.mean(victim_slowdowns)) if victim_slowdowns else 1.0
        max_victim_sf = float(max(victim_slowdowns)) if victim_slowdowns else 1.0
        avg_victim_sr = float(np.mean(victim_stall_ratios)) if victim_stall_ratios else 0.0

        # Congestion history from engine.net_congestion_history [(timestep, value), ...]
        cong_vals = []
        if hasattr(engine, 'net_congestion_history') and engine.net_congestion_history:
            cong_vals = [v for _, v in engine.net_congestion_history if v is not None]

        avg_cong = float(np.mean(cong_vals)) if cong_vals else 0.0
        max_cong = float(max(cong_vals)) if cong_vals else 0.0

        total_nodes_sim = node_count
        load_frac = bully_nodes / total_nodes_sim if total_nodes_sim > 0 else 0.0

        print(f"  bully={bully_nodes:4d}n  load={load_frac:.3f}  "
              f"avg_victim_sf={avg_victim_sf:.4f}  max_victim_sf={max_victim_sf:.4f}  "
              f"avg_cong={avg_cong:.4f}  wall={wall_time:.1f}s")

        return InterferenceResult(
            system=system,
            routing=routing or 'default',
            bully_nodes=bully_nodes,
            victim_count=victim_count,
            victim_nodes=victim_nodes,
            duration=duration,
            total_nodes=total_nodes_sim,
            load_fraction=load_frac,
            avg_victim_slowdown=avg_victim_sf,
            max_victim_slowdown=max_victim_sf,
            avg_victim_stall_ratio=avg_victim_sr,
            bully_slowdown=bully_slowdown,
            avg_congestion=avg_cong,
            max_congestion=max_cong,
            avg_net_util=net_stats.get('avg_network_util', 0.0),
            wall_time=wall_time,
            simulated_seconds=simulated_seconds,
        )

    except Exception as e:
        print(f"  [ERROR] bully={bully_nodes}: {e}")
        import traceback
        traceback.print_exc()
        return None


def sweep_bully_size(
    system: str,
    bully_sizes: List[int],
    victim_count: int = 4,
    victim_nodes: int = 64,
    duration: int = 300,
    routings: Optional[List[str]] = None,
    node_count: int = 1000,
    delta_t: int = 15,
    duration_minutes: int = 10,
    tx_fraction: float = 1.0,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
) -> pd.DataFrame:
    """Sweep bully sizes and collect interference metrics."""
    topo = _get_topology(system)
    if routings is None:
        if topo == 'dragonfly':
            routings = ["minimal", "valiant"]
        elif topo == 'fat-tree':
            routings = ["minimal", "ecmp"]
        else:  # torus3d
            routings = [None]

    out_dir = output_dir / system
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"bully_sweep_v{victim_count}x{victim_nodes}n.csv"

    if csv_path.exists() and not force:
        print(f"[{system}] Loading existing results: {csv_path}")
        return pd.read_csv(csv_path)

    all_results = []
    for routing in routings:
        routing_label = routing or 'dor_xyz'
        print(f"\n[{system}] routing={routing_label}, victim={victim_count}×{victim_nodes}n")
        for bully_nodes in bully_sizes:
            result = run_one_experiment(
                system=system,
                bully_nodes=bully_nodes,
                victim_count=victim_count,
                victim_nodes=victim_nodes,
                duration=duration,
                routing=routing,
                node_count=node_count,
                delta_t=delta_t,
                duration_minutes=duration_minutes,
                tx_fraction=tx_fraction,
            )
            if result is not None:
                row = asdict(result)
                row['routing_label'] = routing_label
                all_results.append(row)

    df = pd.DataFrame(all_results)
    if not df.empty:
        df.to_csv(csv_path, index=False)
        print(f"\n[{system}] Saved {len(df)} rows → {csv_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="NRAPS Bully-Victim Interference Study")
    parser.add_argument('--system', nargs='+', default=['frontier', 'lassen', 'bluewaters'],
                        choices=['frontier', 'lassen', 'bluewaters'],
                        help="Systems to run (default: all three)")
    parser.add_argument('--bully-sizes', nargs='+', type=int,
                        default=DEFAULT_BULLY_SIZES,
                        help="Bully node counts to sweep")
    parser.add_argument('--victim-count', type=int, default=4,
                        help="Number of victim jobs (default: 4)")
    parser.add_argument('--victim-nodes', type=int, default=64,
                        help="Nodes per victim job (default: 64)")
    parser.add_argument('--duration', type=int, default=300,
                        help="Job duration in seconds (default: 300)")
    parser.add_argument('--tx-fraction', type=float, default=1.0,
                        help="Bully TX fraction of max per-node throughput (default: 1.0)")
    parser.add_argument('--delta-t', type=int, default=None,
                        help="Simulation time step (overrides per-system default)")
    parser.add_argument('--duration-minutes', type=int, default=None,
                        help="Simulation window in minutes (overrides per-system default)")
    parser.add_argument('--node-count', type=int, default=None,
                        help="System node count (overrides per-system default)")
    parser.add_argument('--routing', nargs='+', default=None,
                        help="Routing algorithms to test (overrides topology default)")
    parser.add_argument('--force', action='store_true',
                        help="Re-run even if output CSV exists")
    parser.add_argument('--baseline', action='store_true',
                        help="Include bully_nodes=0 baseline in sweep")
    args = parser.parse_args()

    bully_sizes = args.bully_sizes
    if args.baseline and 0 not in bully_sizes:
        bully_sizes = [0] + sorted(bully_sizes)

    print("=" * 60)
    print("NRAPS Bully-Victim Interference Study")
    print(f"  systems    : {args.system}")
    print(f"  bully_sizes: {bully_sizes}")
    print(f"  victims    : {args.victim_count} × {args.victim_nodes} nodes")
    print(f"  duration   : {args.duration}s")
    print(f"  tx_fraction: {args.tx_fraction}")
    print("=" * 60)

    all_dfs = []
    for system in args.system:
        params = SYSTEM_PARAMS.get(system, SYSTEM_PARAMS['frontier'])
        delta_t = args.delta_t or params['delta_t']
        duration_minutes = args.duration_minutes or params['duration_minutes']
        node_count = args.node_count or params['node_count']

        df = sweep_bully_size(
            system=system,
            bully_sizes=bully_sizes,
            victim_count=args.victim_count,
            victim_nodes=args.victim_nodes,
            duration=args.duration,
            routings=args.routing,
            node_count=node_count,
            delta_t=delta_t,
            duration_minutes=duration_minutes,
            tx_fraction=args.tx_fraction,
            force=args.force,
        )
        all_dfs.append(df)

    # Combine and summarize
    if all_dfs:
        combined = pd.concat([d for d in all_dfs if not d.empty], ignore_index=True)
        combined_path = OUTPUT_DIR / "all_systems_interference.csv"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined results → {combined_path}")

        # Verification summary
        print("\n--- Verification ---")
        for system in args.system:
            subset = combined[combined['system'] == system] if 'system' in combined.columns else combined
            if not subset.empty:
                print(f"\n{system}:")
                cols = ['bully_nodes', 'routing_label', 'load_fraction',
                        'avg_victim_slowdown', 'max_victim_slowdown', 'avg_congestion']
                cols = [c for c in cols if c in subset.columns]
                print(subset[cols].to_string(index=False))


if __name__ == "__main__":
    main()

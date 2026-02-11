#!/usr/bin/env python3
"""
RAPS Use Case Evaluation Script
================================

Implements the four operational use cases from Section "ExaDigiT/RAPS as a Versatile Tool":

  UC1: Adaptive Routing and Congestion Mitigation
       - Compares static minimal routing vs adaptive routing on Dragonfly
       - Measures bully effect, job slowdown CDF, global throughput
       - Congestion heatmap (Time x Link utilization)

  UC2: Scheduler Policy Optimization
       - Compares FCFS vs backfill under realistic network interference
       - Measures system utilization, wait time distribution, fragmentation
       - System utilization stacked chart (Running/Idle/Fragmented)
       - Prediction error from dilation (planned vs actual completion)

  UC3: Topology-Aware Node Placement
       - Compares random vs contiguous placement
       - Measures hop count distribution, global/local traffic ratio
       - Application speedup vs communication intensity

  UC4: Energy Cost of Congestion
       - Quantifies hidden energy costs from network dilation
       - Measures energy-to-solution, power profile
       - Static power tax (dynamic/static power decomposition)

Each use case runs a full DES with the RAPS Engine.

Usage:
    python src/run_use_cases.py                     # Run all use cases
    python src/run_use_cases.py --uc 1              # Run only UC1
    python src/run_use_cases.py --uc 1 2            # Run UC1 and UC2
    python src/run_use_cases.py --system frontier    # Use Frontier config
    python src/run_use_cases.py --duration 15        # 15-minute simulations
    python src/run_use_cases.py --quick              # Quick mode (5 min, fewer jobs)
"""

import sys
import time
import json
import argparse
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from raps.engine import Engine
from raps.sim_config import SingleSimConfig
from raps.system_config import SystemConfig
from raps.job import Job, job_dict, CommunicationPattern
from raps.stats import get_network_stats, get_engine_stats, get_job_stats
from raps.network.base import (
    worst_link_util, get_link_util_stats, link_loads_for_pattern,
    get_effective_traffic,
)
from raps.network.fat_tree import node_id_to_host_name
from raps.network.dragonfly import parse_dragonfly_host

# Optional: traffic templates
try:
    from traffic_integration import TrafficMatrixTemplate, load_all_templates
    TEMPLATES_AVAILABLE = True
except ImportError:
    TEMPLATES_AVAILABLE = False

# Optional: matplotlib for plotting
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


# ==========================================
# Configuration
# ==========================================

OUTPUT_DIR = Path("/app/output/use_cases")
FIGURES_DIR = OUTPUT_DIR / "figures"
MATRIX_DIR = Path("/app/data/matrices")

MINI_APP_PATTERNS = {
    'lulesh': CommunicationPattern.STENCIL_3D,
    'comd': CommunicationPattern.STENCIL_3D,
    'hpgmg': CommunicationPattern.STENCIL_3D,
    'cosp2': CommunicationPattern.ALL_TO_ALL,
}

ROUTING_COLORS = {
    'minimal': '#4ECDC4',
    'ecmp': '#45B7D1',
    'ugal': '#F77F00',
    'valiant': '#FCBF49',
    'adaptive': '#E17055',
}

ALLOCATION_COLORS = {
    'contiguous': '#FF6B6B',
    'random': '#4ECDC4',
}

POLICY_COLORS = {
    'fcfs': '#6C5CE7',
    'backfill': '#FDCB6E',
}


# ==========================================
# SimConfig with proper routing override
# ==========================================

class UCSimConfig(SingleSimConfig):
    """SimConfig for use case experiments."""
    pass


def override_system_routing(sim_config, routing, ugal_threshold=2.0, valiant_bias=0.05):
    """Override routing algorithm in a SimConfig's SystemConfig."""
    sys_cfg = sim_config.system_configs[0]
    data = sys_cfg.model_dump(mode='json')
    if 'network' in data and data['network'] is not None:
        data['network']['routing_algorithm'] = routing
        data['network']['ugal_threshold'] = ugal_threshold
        data['network']['valiant_bias'] = valiant_bias
    new_sys_cfg = SystemConfig.model_validate(data)
    sim_config._system_configs = [new_sys_cfg]
    return sim_config


# ==========================================
# Workload Generation
# ==========================================

def generate_workload(
    num_jobs: int = 30,
    min_nodes: int = 8,
    max_nodes: int = 256,
    min_duration: int = 300,
    max_duration: int = 1800,
    seed: int = 42,
) -> List[Job]:
    """
    Generate synthetic workload with log-normal job sizes and Poisson arrivals.
    Jobs have realistic network traces assigned based on communication pattern.
    """
    rng = np.random.default_rng(seed)
    jobs = []

    # Log-normal job sizes (heavy-tailed, biased toward smaller jobs)
    raw_sizes = rng.lognormal(mean=3, sigma=1.2, size=num_jobs)
    sizes = np.clip(raw_sizes, min_nodes, max_nodes).astype(int)
    # Round to power-of-2 (common in HPC)
    sizes = 2 ** np.round(np.log2(sizes)).astype(int)
    sizes = np.clip(sizes, min_nodes, max_nodes)

    # Poisson inter-arrival times
    mean_inter_arrival = 60  # 1 job per minute on average
    inter_arrivals = rng.exponential(mean_inter_arrival, num_jobs)
    arrival_times = np.cumsum(inter_arrivals).astype(int)

    # Log-normal durations
    raw_durations = rng.lognormal(mean=6.5, sigma=0.8, size=num_jobs)
    durations = np.clip(raw_durations, min_duration, max_duration).astype(int)

    # Mini-app assignment
    mini_app_names = list(MINI_APP_PATTERNS.keys())

    for i in range(num_jobs):
        job_size = int(sizes[i])
        arrival = int(arrival_times[i])
        duration = int(durations[i])

        # Assign mini-app and communication pattern
        mini_app = mini_app_names[i % len(mini_app_names)]
        comm_pattern = MINI_APP_PATTERNS[mini_app]

        # Generate network traces based on pattern
        trace_quanta = 15
        trace_len = max(1, duration // trace_quanta)

        if comm_pattern == CommunicationPattern.STENCIL_3D:
            base_bw = job_size * 6 * 150.0  # 6 neighbors, ~150 B each
        else:
            base_bw = job_size * (job_size - 1) * 50.0  # all-to-all

        ntx_trace = [base_bw] * trace_len
        nrx_trace = [base_bw] * trace_len

        # CPU/GPU traces
        cpu_val = float(rng.uniform(0.4, 0.9))
        gpu_val = float(rng.uniform(0.5, 0.95))
        cpu_trace = [cpu_val] * trace_len
        gpu_trace = [gpu_val] * trace_len

        job = Job(job_dict(
            id=i + 1,
            name=f"job_{i+1}_{mini_app}",
            account="benchmark",
            nodes_required=job_size,
            scheduled_nodes=[],
            cpu_trace=cpu_trace,
            gpu_trace=gpu_trace,
            ntx_trace=ntx_trace,
            nrx_trace=nrx_trace,
            comm_pattern=comm_pattern,
            message_size=1048576,  # 1 MB
            trace_quanta=trace_quanta,
            submit_time=arrival,
            expected_run_time=duration,
            time_limit=duration * 2,
            end_state="COMPLETED",
        ))
        job.mini_app = mini_app
        # Store original expected runtime for prediction error analysis
        job.original_expected_run_time = duration
        jobs.append(job)

    return jobs


def clone_jobs(jobs: List[Job]) -> List[Job]:
    """Deep-copy a job list so each experiment starts fresh."""
    cloned = []
    for j in jobs:
        d = job_dict(
            id=j.id, name=j.name, account=j.account,
            nodes_required=j.nodes_required,
            scheduled_nodes=[],
            cpu_trace=list(j.cpu_trace) if j.cpu_trace else None,
            gpu_trace=list(j.gpu_trace) if j.gpu_trace else None,
            ntx_trace=list(j.ntx_trace) if j.ntx_trace else None,
            nrx_trace=list(j.nrx_trace) if j.nrx_trace else None,
            comm_pattern=j.comm_pattern,
            message_size=getattr(j, 'message_size', 1048576),
            trace_quanta=getattr(j, 'trace_quanta', 15),
            submit_time=j.submit_time,
            expected_run_time=j.expected_run_time,
            time_limit=j.time_limit,
            end_state="COMPLETED",
        )
        new_job = Job(d)
        new_job.mini_app = getattr(j, 'mini_app', 'unknown')
        new_job.original_expected_run_time = getattr(j, 'original_expected_run_time',
                                                      j.expected_run_time)
        cloned.append(new_job)
    return cloned


# ==========================================
# Post-Simulation Analysis Functions
# ==========================================

def compute_hop_counts(engine):
    """
    Compute hop count distribution for all running jobs.
    Returns list of hop counts (one per communicating pair).
    Uses the network graph and scheduled node placements.
    """
    hop_counts = []
    if not engine.simulate_network or not engine.network_model:
        return hop_counts

    net_model = engine.network_model
    G = net_model.net_graph
    if G is None:
        return hop_counts

    for job in engine.jobs:
        if not job.scheduled_nodes or len(job.scheduled_nodes) < 2:
            continue

        # Map real node IDs to topology host names
        try:
            if net_model.topology == "fat-tree":
                k = net_model.fattree_k
                hosts = [node_id_to_host_name(n, k) for n in job.scheduled_nodes]
            elif net_model.topology == "dragonfly":
                hosts = [net_model.real_to_fat_idx[n] for n in job.scheduled_nodes]
            else:
                continue  # torus3d hop count analysis not yet supported

            # Sample pairs (limit for large jobs)
            max_pairs = min(500, len(hosts) * (len(hosts) - 1) // 2)
            pairs_checked = 0
            for i in range(len(hosts)):
                for j_idx in range(i + 1, len(hosts)):
                    if pairs_checked >= max_pairs:
                        break
                    try:
                        path = nx.shortest_path(G, hosts[i], hosts[j_idx])
                        # Hop count = number of edges = len(path) - 1
                        hop_counts.append(len(path) - 1)
                        pairs_checked += 1
                    except nx.NetworkXNoPath:
                        pass
                if pairs_checked >= max_pairs:
                    break
        except (KeyError, ValueError):
            continue

    return hop_counts


def compute_global_local_ratio(engine):
    """
    Compute the ratio of global (inter-group) to local (intra-group) traffic.
    Only meaningful for Dragonfly topology.

    Returns dict with global_traffic, local_traffic, ratio.
    """
    result = {'global_traffic': 0, 'local_traffic': 0, 'ratio': 0.0}

    if not engine.simulate_network or not engine.network_model:
        return result

    net_model = engine.network_model
    if net_model.topology != "dragonfly":
        return result

    G = net_model.net_graph
    if G is None:
        return result

    global_count = 0
    local_count = 0

    for job in engine.jobs:
        if not job.scheduled_nodes or len(job.scheduled_nodes) < 2:
            continue

        try:
            hosts = [net_model.real_to_fat_idx[n] for n in job.scheduled_nodes]
        except (KeyError, ValueError):
            continue

        # Classify each communicating pair as intra-group or inter-group
        for i in range(len(hosts)):
            for j_idx in range(i + 1, len(hosts)):
                try:
                    src_group = parse_dragonfly_host(hosts[i])[0]
                    dst_group = parse_dragonfly_host(hosts[j_idx])[0]
                    if src_group == dst_group:
                        local_count += 1
                    else:
                        global_count += 1
                except (ValueError, IndexError):
                    pass

    result['global_traffic'] = global_count
    result['local_traffic'] = local_count
    total = global_count + local_count
    result['ratio'] = global_count / total if total > 0 else 0.0
    return result


def compute_prediction_error(engine):
    """
    Compute prediction error: how much dilation caused actual job completion
    to deviate from the scheduler's expected completion time.

    Returns list of (job_id, original_expected, actual_runtime, error_pct) tuples.
    """
    errors = []
    for job in engine.jobs:
        original = getattr(job, 'original_expected_run_time', None)
        if original is None:
            original = job.expected_run_time

        st = getattr(job, 'start_time', None)
        et = getattr(job, 'end_time', None)
        if st is not None and et is not None:
            actual_runtime = et - st
            if original > 0:
                error_pct = (actual_runtime - original) / original * 100
            else:
                error_pct = 0.0
            errors.append({
                'job_id': job.id,
                'original_expected': original,
                'actual_runtime': actual_runtime,
                'dilated': getattr(job, 'dilated', False),
                'error_pct': error_pct,
                'nodes': job.nodes_required,
            })
    return errors


def compute_comm_intensity(job):
    """Compute communication intensity for a job (bytes per node per second)."""
    if not job.ntx_trace or job.nodes_required == 0:
        return 0.0
    avg_tx = float(np.mean(job.ntx_trace))
    return avg_tx / job.nodes_required


def collect_per_tick_utilization(engine):
    """
    Collect per-tick node utilization breakdown from engine histories.
    Returns lists of (timestep, running_nodes, idle_nodes, fragmented_nodes).
    """
    total_nodes = engine.config['AVAILABLE_NODES']
    down_nodes_count = len(engine.config.get('DOWN_NODES', []))
    available = total_nodes - down_nodes_count

    util_breakdown = []
    for i, (ts, util_pct) in enumerate(engine.sys_util_history):
        running_nodes = int(round(engine.num_active_nodes
                                   if i == len(engine.sys_util_history) - 1
                                   else util_pct / 100.0 * available))
        idle_nodes = available - running_nodes
        # Fragmented = nodes that are free but can't be allocated due to fragmentation
        # Approximate: free nodes minus largest contiguous free block
        free_count = idle_nodes
        queue_len = engine.scheduler_running_history[i] if i < len(engine.scheduler_running_history) else 0
        fragmented = min(free_count, queue_len * 2) if queue_len > 0 else 0
        truly_idle = max(0, free_count - fragmented)

        util_breakdown.append({
            'timestep': ts,
            'running': running_nodes,
            'idle': truly_idle,
            'fragmented': fragmented,
        })

    return util_breakdown


def collect_congestion_heatmap_data(engine, sample_interval=10):
    """
    Build congestion heatmap data from engine's net_congestion_history.
    Returns (timesteps, congestion_values) suitable for a time-series plot.

    For a full link-level heatmap, we re-examine the top congested links
    by sampling the inter-job congestion at intervals.
    """
    if not engine.net_congestion_history:
        return [], []

    timesteps = [t for t, c in engine.net_congestion_history]
    congestion = [c for t, c in engine.net_congestion_history]

    return timesteps, congestion


def compute_power_decomposition(engine):
    """
    Decompose total power into static (idle) and dynamic components.
    Static power = power when all nodes are idle.
    Dynamic power = total - static.

    Returns dict with static_power_kw, dynamic_power_kw, static_pct.
    """
    result = {'static_power_kw': 0, 'dynamic_power_kw': 0, 'static_pct': 0,
              'power_timeline': [], 'static_timeline': [], 'dynamic_timeline': []}

    if not engine.power_manager or not engine.power_manager.history:
        return result

    # Compute idle power for one node (cpu_util=0, gpu_util=0)
    from raps.power import compute_node_power
    idle_node_power, _ = compute_node_power(0, 0, 0, engine.config)
    total_nodes = engine.config['AVAILABLE_NODES']
    down_nodes_count = len(engine.config.get('DOWN_NODES', []))
    available_nodes = total_nodes - down_nodes_count

    # Static power = idle_node_power * available_nodes (in kW)
    static_power_kw = float(idle_node_power) * available_nodes / 1000.0

    power_timeline = []
    static_timeline = []
    dynamic_timeline = []

    for ts, total_kw in engine.power_manager.history:
        power_timeline.append(total_kw)
        static_timeline.append(static_power_kw)
        dynamic_timeline.append(max(0, total_kw - static_power_kw))

    avg_total = float(np.mean(power_timeline)) if power_timeline else 0
    avg_dynamic = float(np.mean(dynamic_timeline)) if dynamic_timeline else 0

    result['static_power_kw'] = static_power_kw
    result['dynamic_power_kw'] = avg_dynamic
    result['static_pct'] = static_power_kw / avg_total * 100 if avg_total > 0 else 0
    result['power_timeline'] = power_timeline
    result['static_timeline'] = static_timeline
    result['dynamic_timeline'] = dynamic_timeline

    return result


# ==========================================
# DES Execution
# ==========================================

@dataclass
class SimResult:
    """Results from a single simulation run."""
    label: str
    system: str
    routing: str
    allocation: str
    policy: str
    num_jobs: int
    simulated_seconds: float
    wall_time: float
    speedup: float
    ticks: int

    # Network metrics
    avg_network_util: float = 0.0
    avg_job_slowdown: float = 1.0
    max_job_slowdown: float = 1.0
    avg_congestion: float = 0.0
    max_congestion: float = 0.0
    global_throughput_bps: float = 0.0

    # Job metrics
    jobs_completed: int = 0
    jobs_dilated: int = 0
    dilated_pct: float = 0.0
    avg_wait_time: float = 0.0
    max_wait_time: float = 0.0
    potential_bullies: int = 0

    # Power/energy
    total_energy_joules: float = 0.0
    avg_power_watts: float = 0.0
    peak_power_watts: float = 0.0
    idle_energy_pct: float = 0.0
    static_power_kw: float = 0.0
    dynamic_power_kw: float = 0.0
    static_power_pct: float = 0.0

    # Hop count and traffic locality (UC3)
    avg_hop_count: float = 0.0
    global_local_ratio: float = 0.0
    global_traffic_pairs: int = 0
    local_traffic_pairs: int = 0

    # Per-job details (for CDF etc.)
    job_slowdowns: list = field(default_factory=list)
    job_wait_times: list = field(default_factory=list)
    job_sizes: list = field(default_factory=list)
    job_comm_intensities: list = field(default_factory=list)
    power_history: list = field(default_factory=list)
    hop_counts: list = field(default_factory=list)

    # Per-tick histories (for heatmaps and stacked charts)
    congestion_timesteps: list = field(default_factory=list)
    congestion_values: list = field(default_factory=list)
    utilization_breakdown: list = field(default_factory=list)
    prediction_errors: list = field(default_factory=list)

    # Power decomposition timelines
    power_static_timeline: list = field(default_factory=list)
    power_dynamic_timeline: list = field(default_factory=list)


def run_simulation(
    jobs: List[Job],
    system: str = "frontier",
    duration_minutes: int = 30,
    delta_t: int = 1,
    routing: str = None,
    allocation: str = "contiguous",
    policy: str = "fcfs",
    simulate_network: bool = True,
    label: str = "",
) -> Optional[SimResult]:
    """
    Run a single DES simulation with full metric collection.
    """
    config_dict = {
        'system': system,
        'time': timedelta(minutes=duration_minutes),
        'time_delta': timedelta(seconds=delta_t),
        'simulate_network': simulate_network,
        'cooling': False,
        'uncertainties': False,
        'weather': False,
        'output': 'none',
        'noui': True,
        'verbose': False,
        'debug': False,
        'workload': 'synthetic',
        'policy': policy,
        'allocation': allocation,
        'numjobs': len(jobs),
        # Required defaults for synthetic workload generator
        'jobsize_distribution': ['uniform'],
        'walltime_distribution': ['uniform'],
    }

    try:
        t_start = time.perf_counter()

        sim_config = UCSimConfig(**config_dict)

        # Override routing
        if routing and simulate_network:
            override_system_routing(sim_config, routing)

        engine = Engine(sim_config)

        # Replace jobs with our prepared workload
        engine.jobs = clone_jobs(jobs)

        t_engine = time.perf_counter()

        # Run simulation
        tick_count = 0
        for tick_data in engine.run_simulation():
            tick_count += 1

        t_end = time.perf_counter()
        sim_time = t_end - t_engine
        total_time = t_end - t_start
        simulated_seconds = duration_minutes * 60
        speedup = simulated_seconds / sim_time if sim_time > 0 else float('inf')

        # Collect network stats
        net_stats = get_network_stats(engine) if simulate_network else {}

        # Collect per-job metrics
        all_jobs = engine.jobs
        job_slowdowns = []
        job_wait_times = []
        job_sizes = []
        job_comm_intensities = []
        dilated_count = 0

        for job in all_jobs:
            job_sizes.append(job.nodes_required)
            sf = getattr(job, 'slowdown_factor', 1.0)
            job_slowdowns.append(sf)
            job_comm_intensities.append(compute_comm_intensity(job))
            if getattr(job, 'dilated', False):
                dilated_count += 1

            st = getattr(job, 'start_time', None)
            if st is not None:
                wait = st - job.submit_time
                job_wait_times.append(max(0, wait))

        # Bully detection
        potential_bullies = 0
        if dilated_count > 0:
            median_size = np.median(job_sizes) if job_sizes else 0
            for job in all_jobs:
                if not getattr(job, 'dilated', False) and job.nodes_required > median_size:
                    potential_bullies += 1

        # Congestion history (for heatmap)
        cong_timesteps, cong_values = collect_congestion_heatmap_data(engine)
        max_cong = max(cong_values) if cong_values else 0.0
        avg_cong = float(np.mean(cong_values)) if cong_values else 0.0

        # Global throughput (absolute bandwidth in bps)
        max_link_bw = engine.config.get('NETWORK_MAX_BW', 12.5e9)
        global_throughput_bps = net_stats.get('avg_network_util', 0) / 100.0 * max_link_bw

        # Power metrics (fix: use engine.power_manager.history, not sys_power_history)
        total_energy = 0.0
        avg_power = 0.0
        peak_power = 0.0
        power_history = []
        if engine.power_manager and engine.power_manager.history:
            powers_kw = [p for ts, p in engine.power_manager.history]
            power_history = powers_kw
            avg_power = float(np.mean(powers_kw)) * 1000  # Convert kW to W
            peak_power = float(max(powers_kw)) * 1000
            # Energy in joules: sum(power_kw) * delta_t * 1000 (kW to W)
            total_energy = sum(powers_kw) * delta_t * 1000

        # Power decomposition (static vs dynamic)
        power_decomp = compute_power_decomposition(engine)

        # Hop count analysis
        hop_counts = compute_hop_counts(engine)
        avg_hops = float(np.mean(hop_counts)) if hop_counts else 0.0

        # Global vs local traffic ratio
        gl_ratio = compute_global_local_ratio(engine)

        # Utilization breakdown (for stacked chart)
        util_breakdown = collect_per_tick_utilization(engine)

        # Prediction error (planned vs actual due to dilation)
        pred_errors = compute_prediction_error(engine)

        # Jobs completed
        jobs_completed = sum(1 for j in all_jobs
                            if getattr(j, 'end_time', None) is not None)

        return SimResult(
            label=label,
            system=system,
            routing=routing or 'default',
            allocation=allocation,
            policy=policy,
            num_jobs=len(all_jobs),
            simulated_seconds=simulated_seconds,
            wall_time=total_time,
            speedup=speedup,
            ticks=tick_count,
            avg_network_util=net_stats.get('avg_network_util', 0.0),
            avg_job_slowdown=net_stats.get('avg_per_job_slowdown', 1.0),
            max_job_slowdown=net_stats.get('max_per_job_slowdown', 1.0),
            avg_congestion=avg_cong,
            max_congestion=max_cong,
            global_throughput_bps=global_throughput_bps,
            jobs_completed=jobs_completed,
            jobs_dilated=dilated_count,
            dilated_pct=dilated_count / len(all_jobs) * 100 if all_jobs else 0,
            avg_wait_time=float(np.mean(job_wait_times)) if job_wait_times else 0,
            max_wait_time=float(max(job_wait_times)) if job_wait_times else 0,
            potential_bullies=potential_bullies,
            total_energy_joules=total_energy,
            avg_power_watts=avg_power,
            peak_power_watts=peak_power,
            static_power_kw=power_decomp['static_power_kw'],
            dynamic_power_kw=power_decomp['dynamic_power_kw'],
            static_power_pct=power_decomp['static_pct'],
            avg_hop_count=avg_hops,
            global_local_ratio=gl_ratio['ratio'],
            global_traffic_pairs=gl_ratio['global_traffic'],
            local_traffic_pairs=gl_ratio['local_traffic'],
            job_slowdowns=job_slowdowns,
            job_wait_times=job_wait_times,
            job_sizes=job_sizes,
            job_comm_intensities=job_comm_intensities,
            power_history=power_history,
            hop_counts=hop_counts,
            congestion_timesteps=cong_timesteps,
            congestion_values=cong_values,
            utilization_breakdown=util_breakdown,
            prediction_errors=pred_errors,
            power_static_timeline=power_decomp['static_timeline'],
            power_dynamic_timeline=power_decomp['dynamic_timeline'],
        )

    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_result(result: SimResult, indent: str = "  "):
    """Print a concise summary of simulation results."""
    print(f"{indent}Wall time: {result.wall_time:.1f}s, "
          f"Speedup: {result.speedup:.1f}x, "
          f"Jobs completed: {result.jobs_completed}/{result.num_jobs}")
    print(f"{indent}Network util: {result.avg_network_util:.2f}%, "
          f"Avg slowdown: {result.avg_job_slowdown:.3f}, "
          f"Max slowdown: {result.max_job_slowdown:.3f}")
    print(f"{indent}Dilated: {result.jobs_dilated} ({result.dilated_pct:.1f}%), "
          f"Bullies: {result.potential_bullies}, "
          f"Max congestion: {result.max_congestion:.4f}")
    if result.global_throughput_bps > 0:
        print(f"{indent}Global throughput: {result.global_throughput_bps/1e9:.2f} Gbps")
    if result.avg_hop_count > 0:
        print(f"{indent}Avg hop count: {result.avg_hop_count:.1f}, "
              f"Global/Local ratio: {result.global_local_ratio:.3f}")
    if result.static_power_kw > 0:
        print(f"{indent}Static power: {result.static_power_kw:.1f} kW ({result.static_power_pct:.1f}%), "
              f"Dynamic: {result.dynamic_power_kw:.1f} kW")


# ==========================================
# UC1: Adaptive Routing and Congestion Mitigation
# ==========================================

def run_uc1_routing(jobs, system, duration_minutes, **kwargs):
    """
    UC1: Compare static minimal routing vs adaptive routing.

    Evaluates the impact of routing policies on the bully effect
    in Dragonfly topology. Compares minimal (fixed paths) against
    adaptive routing (sensing local queue depth / link utilization).
    """
    print("\n" + "=" * 60)
    print("UC1: Adaptive Routing and Congestion Mitigation")
    print("=" * 60)

    routing_algos = ['minimal', 'adaptive']
    results = {}

    for routing in routing_algos:
        print(f"\n  Running with routing={routing}...")
        result = run_simulation(
            jobs=jobs,
            system=system,
            duration_minutes=duration_minutes,
            routing=routing,
            allocation='contiguous',
            policy='fcfs',
            simulate_network=True,
            label=f"UC1_{routing}",
        )
        if result:
            results[routing] = result
            print_result(result, indent="    ")

    # Comparison
    if len(results) >= 2:
        print(f"\n  --- UC1 Comparison ---")
        print(f"  {'Metric':<30} {'Minimal':>12} {'Adaptive':>12} {'Delta':>12}")
        print(f"  {'-'*66}")
        r_min = results.get('minimal')
        r_ada = results.get('adaptive')
        if r_min and r_ada:
            comparisons = [
                ('Avg Slowdown', r_min.avg_job_slowdown, r_ada.avg_job_slowdown),
                ('Max Slowdown', r_min.max_job_slowdown, r_ada.max_job_slowdown),
                ('Dilated Jobs (%)', r_min.dilated_pct, r_ada.dilated_pct),
                ('Max Congestion', r_min.max_congestion, r_ada.max_congestion),
                ('Avg Network Util (%)', r_min.avg_network_util, r_ada.avg_network_util),
                ('Global Throughput (Gbps)', r_min.global_throughput_bps/1e9,
                 r_ada.global_throughput_bps/1e9),
                ('Potential Bullies', float(r_min.potential_bullies),
                 float(r_ada.potential_bullies)),
            ]
            for name, v_min, v_ada in comparisons:
                delta = v_ada - v_min
                print(f"  {name:<30} {v_min:>12.3f} {v_ada:>12.3f} {delta:>+12.3f}")

    return results


# ==========================================
# UC2: Scheduler Policy Optimization
# ==========================================

def run_uc2_scheduling(jobs, system, duration_minutes, **kwargs):
    """
    UC2: Compare FCFS vs Backfill scheduling under network interference.

    The key insight is that backfill scheduling's slot predictions are
    disrupted by network congestion dilation: running jobs take longer
    than predicted, invalidating reservation windows.
    """
    print("\n" + "=" * 60)
    print("UC2: Scheduler Policy Optimization")
    print("=" * 60)

    policies = ['fcfs', 'backfill']
    results = {}

    for policy in policies:
        print(f"\n  Running with policy={policy}...")
        result = run_simulation(
            jobs=jobs,
            system=system,
            duration_minutes=duration_minutes,
            routing='adaptive',
            allocation='contiguous',
            policy=policy,
            simulate_network=True,
            label=f"UC2_{policy}",
        )
        if result:
            results[policy] = result
            print_result(result, indent="    ")
            print(f"    Avg wait time: {result.avg_wait_time:.1f}s, "
                  f"Max wait time: {result.max_wait_time:.1f}s")

            # Print wait time breakdown by job size
            if result.job_wait_times and result.job_sizes:
                median_size = np.median(result.job_sizes)
                small_waits = [w for w, s in zip(result.job_wait_times, result.job_sizes)
                               if s <= median_size]
                large_waits = [w for w, s in zip(result.job_wait_times, result.job_sizes)
                               if s > median_size]
                if small_waits:
                    print(f"    Small jobs (<=  {int(median_size)} nodes) avg wait: "
                          f"{np.mean(small_waits):.1f}s")
                if large_waits:
                    print(f"    Large jobs (> {int(median_size)} nodes) avg wait: "
                          f"{np.mean(large_waits):.1f}s")

            # Print prediction error summary
            if result.prediction_errors:
                dilated_errors = [e for e in result.prediction_errors if e['dilated']]
                if dilated_errors:
                    avg_err = np.mean([e['error_pct'] for e in dilated_errors])
                    max_err = max(e['error_pct'] for e in dilated_errors)
                    print(f"    Prediction error (dilated jobs): avg={avg_err:.1f}%, "
                          f"max={max_err:.1f}%")

    # Comparison
    if len(results) >= 2:
        print(f"\n  --- UC2 Comparison ---")
        print(f"  {'Metric':<30} {'FCFS':>12} {'Backfill':>12} {'Delta':>12}")
        print(f"  {'-'*66}")
        r_fcfs = results.get('fcfs')
        r_bf = results.get('backfill')
        if r_fcfs and r_bf:
            comparisons = [
                ('Jobs Completed', float(r_fcfs.jobs_completed), float(r_bf.jobs_completed)),
                ('Avg Wait Time (s)', r_fcfs.avg_wait_time, r_bf.avg_wait_time),
                ('Max Wait Time (s)', r_fcfs.max_wait_time, r_bf.max_wait_time),
                ('Avg Slowdown', r_fcfs.avg_job_slowdown, r_bf.avg_job_slowdown),
                ('Dilated Jobs (%)', r_fcfs.dilated_pct, r_bf.dilated_pct),
                ('Avg Network Util (%)', r_fcfs.avg_network_util, r_bf.avg_network_util),
            ]
            for name, v1, v2 in comparisons:
                delta = v2 - v1
                print(f"  {name:<30} {v1:>12.3f} {v2:>12.3f} {delta:>+12.3f}")

    return results


# ==========================================
# UC3: Topology-Aware Node Placement
# ==========================================

def run_uc3_placement(jobs, system, duration_minutes, **kwargs):
    """
    UC3: Compare random vs contiguous node placement.

    Contiguous placement keeps jobs within a single electrical group
    to minimize global link usage (Short Circuit). Random placement
    spreads jobs across the fabric, increasing global link pressure.
    """
    print("\n" + "=" * 60)
    print("UC3: Topology-Aware Node Placement")
    print("=" * 60)

    allocations = ['contiguous', 'random']
    results = {}

    for alloc in allocations:
        print(f"\n  Running with allocation={alloc}...")
        result = run_simulation(
            jobs=jobs,
            system=system,
            duration_minutes=duration_minutes,
            routing='adaptive',
            allocation=alloc,
            policy='fcfs',
            simulate_network=True,
            label=f"UC3_{alloc}",
        )
        if result:
            results[alloc] = result
            print_result(result, indent="    ")

    # Comparison
    if len(results) >= 2:
        print(f"\n  --- UC3 Comparison ---")
        print(f"  {'Metric':<30} {'Contiguous':>12} {'Random':>12} {'Delta':>12}")
        print(f"  {'-'*66}")
        r_cont = results.get('contiguous')
        r_rand = results.get('random')
        if r_cont and r_rand:
            comparisons = [
                ('Avg Slowdown', r_cont.avg_job_slowdown, r_rand.avg_job_slowdown),
                ('Max Slowdown', r_cont.max_job_slowdown, r_rand.max_job_slowdown),
                ('Dilated Jobs (%)', r_cont.dilated_pct, r_rand.dilated_pct),
                ('Max Congestion', r_cont.max_congestion, r_rand.max_congestion),
                ('Avg Network Util (%)', r_cont.avg_network_util, r_rand.avg_network_util),
                ('Avg Hop Count', r_cont.avg_hop_count, r_rand.avg_hop_count),
                ('Global/Local Ratio', r_cont.global_local_ratio, r_rand.global_local_ratio),
                ('Potential Bullies', float(r_cont.potential_bullies),
                 float(r_rand.potential_bullies)),
            ]
            for name, v1, v2 in comparisons:
                delta = v2 - v1
                print(f"  {name:<30} {v1:>12.3f} {v2:>12.3f} {delta:>+12.3f}")

            # Application speedup from placement
            if r_rand.avg_job_slowdown > 0:
                placement_benefit = r_rand.avg_job_slowdown / r_cont.avg_job_slowdown
                print(f"\n  Contiguous placement speedup vs random: {placement_benefit:.2f}x")

            # Speedup vs communication intensity
            print(f"\n  --- Speedup vs Communication Intensity ---")
            if r_cont.job_comm_intensities and r_rand.job_comm_intensities:
                # Group jobs by comm intensity quartile
                all_intensities = r_cont.job_comm_intensities + r_rand.job_comm_intensities
                q25, q50, q75 = np.percentile(all_intensities, [25, 50, 75])
                labels_qi = ['Low', 'Med-Low', 'Med-High', 'High']
                thresholds = [0, q25, q50, q75, float('inf')]

                print(f"  {'Intensity':<12} {'Cont. Slowdown':>15} {'Rand. Slowdown':>15} {'Benefit':>10}")
                print(f"  {'-'*52}")
                for qi in range(4):
                    lo, hi = thresholds[qi], thresholds[qi+1]
                    cont_sds = [sd for sd, ci in zip(r_cont.job_slowdowns, r_cont.job_comm_intensities)
                                if lo <= ci < hi]
                    rand_sds = [sd for sd, ci in zip(r_rand.job_slowdowns, r_rand.job_comm_intensities)
                                if lo <= ci < hi]
                    cont_avg = float(np.mean(cont_sds)) if cont_sds else 1.0
                    rand_avg = float(np.mean(rand_sds)) if rand_sds else 1.0
                    benefit = rand_avg / cont_avg if cont_avg > 0 else 1.0
                    print(f"  {labels_qi[qi]:<12} {cont_avg:>15.3f} {rand_avg:>15.3f} {benefit:>10.2f}x")

    return results


# ==========================================
# UC4: Energy Cost of Congestion
# ==========================================

def run_uc4_energy(jobs, system, duration_minutes, **kwargs):
    """
    UC4: Quantify the energy cost of network congestion.

    When congestion dilates job runtime, static power (leakage) continues
    to accumulate even though dynamic work is stalled. This creates an
    "energy tax" proportional to the dilation factor.
    """
    print("\n" + "=" * 60)
    print("UC4: Energy Cost of Congestion")
    print("=" * 60)

    # Run with and without network (baseline vs congested)
    configs = {
        'no_congestion': {'simulate_network': False, 'label': 'Ideal (no congestion)'},
        'with_congestion': {'simulate_network': True, 'label': 'With network congestion'},
    }
    results = {}

    for config_name, cfg in configs.items():
        print(f"\n  Running: {cfg['label']}...")
        result = run_simulation(
            jobs=jobs,
            system=system,
            duration_minutes=duration_minutes,
            routing='adaptive',
            allocation='contiguous',
            policy='fcfs',
            simulate_network=cfg['simulate_network'],
            label=f"UC4_{config_name}",
        )
        if result:
            results[config_name] = result
            print_result(result, indent="    ")
            print(f"    Total energy: {result.total_energy_joules:.0f} J, "
                  f"Avg power: {result.avg_power_watts:.1f} W, "
                  f"Peak power: {result.peak_power_watts:.1f} W")

    # Energy comparison
    if 'no_congestion' in results and 'with_congestion' in results:
        r_ideal = results['no_congestion']
        r_cong = results['with_congestion']

        print(f"\n  --- UC4 Energy Cost Analysis ---")
        print(f"  {'Metric':<30} {'Ideal':>12} {'Congested':>12} {'Overhead':>12}")
        print(f"  {'-'*66}")

        e_ideal = r_ideal.total_energy_joules
        e_cong = r_cong.total_energy_joules
        overhead = (e_cong - e_ideal) / e_ideal * 100 if e_ideal > 0 else 0

        comparisons = [
            ('Total Energy (J)', e_ideal, e_cong),
            ('Avg Power (W)', r_ideal.avg_power_watts, r_cong.avg_power_watts),
            ('Peak Power (W)', r_ideal.peak_power_watts, r_cong.peak_power_watts),
            ('Jobs Completed', float(r_ideal.jobs_completed), float(r_cong.jobs_completed)),
        ]
        for name, v1, v2 in comparisons:
            delta = v2 - v1
            print(f"  {name:<30} {v1:>12.1f} {v2:>12.1f} {delta:>+12.1f}")

        if overhead != 0:
            print(f"\n  Energy overhead from congestion: {overhead:+.1f}%")

        # Static vs dynamic power decomposition
        print(f"\n  --- Power Decomposition ---")
        print(f"  {'Component':<25} {'Ideal (kW)':>12} {'Congested (kW)':>15}")
        print(f"  {'-'*52}")
        print(f"  {'Static (idle leakage)':<25} {r_ideal.static_power_kw:>12.1f} "
              f"{r_cong.static_power_kw:>15.1f}")
        print(f"  {'Dynamic (compute)':<25} {r_ideal.dynamic_power_kw:>12.1f} "
              f"{r_cong.dynamic_power_kw:>15.1f}")
        print(f"  {'Static fraction':<25} {r_ideal.static_power_pct:>11.1f}% "
              f"{r_cong.static_power_pct:>14.1f}%")

        # The "static power tax": extra static energy consumed during dilation
        if r_cong.jobs_dilated > 0:
            extra_static_j = (r_cong.static_power_kw - r_ideal.static_power_kw) * \
                             r_cong.simulated_seconds
            print(f"\n  Static power tax from dilation: {extra_static_j:.0f} J")
            print(f"  ({r_cong.jobs_dilated} dilated jobs consumed extra static energy")
            print(f"   while stalled by network congestion)")

    return results


# ==========================================
# Plotting
# ==========================================

def plot_uc1_slowdown_cdf(results, save_path):
    """Plot CDF of job slowdown factors for UC1."""
    if not PLOTTING_AVAILABLE:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for routing, result in results.items():
        if result.job_slowdowns:
            sorted_sd = np.sort(result.job_slowdowns)
            cdf = np.arange(1, len(sorted_sd) + 1) / len(sorted_sd)
            color = ROUTING_COLORS.get(routing, '#888')
            ax.step(sorted_sd, cdf, where='post', label=routing.title(),
                    color=color, linewidth=2)

    ax.set_xlabel('Job Slowdown Factor (T_actual / T_ideal)')
    ax.set_ylabel('CDF')
    ax.set_title('UC1: Job Slowdown CDF - Routing Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_uc1_congestion_heatmap(results, save_path):
    """Plot congestion over time for UC1 (time-series heatmap proxy)."""
    if not PLOTTING_AVAILABLE:
        return

    fig, axes = plt.subplots(len(results), 1, figsize=(12, 3 * len(results)),
                             sharex=True, squeeze=False)

    for idx, (routing, result) in enumerate(results.items()):
        ax = axes[idx, 0]
        if result.congestion_timesteps and result.congestion_values:
            ts = np.array(result.congestion_timesteps) / 60.0  # Convert to minutes
            cong = np.array(result.congestion_values)

            # Create a pseudo-heatmap using color-coded scatter/bar
            colors = plt.cm.hot(np.clip(cong / (max(cong) + 1e-9), 0, 1))
            ax.bar(ts, cong, width=(ts[1]-ts[0]) if len(ts) > 1 else 1,
                   color=colors, edgecolor='none')
            ax.set_ylabel('Mean Link\nUtilization')
            ax.set_title(f'Routing: {routing.title()}')
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Congestion threshold')
            ax.legend(loc='upper right', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No congestion data', transform=ax.transAxes,
                    ha='center', va='center')

    axes[-1, 0].set_xlabel('Time (minutes)')
    fig.suptitle('UC1: Congestion Heatmap Over Time', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_uc2_wait_times(results, save_path):
    """Plot wait time distribution for UC2, grouped by job size."""
    if not PLOTTING_AVAILABLE:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Wait time boxplot by policy
    ax = axes[0]
    data = []
    labels = []
    for policy, result in results.items():
        if result.job_wait_times:
            data.append(result.job_wait_times)
            labels.append(policy.upper())
    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        colors = [POLICY_COLORS.get(p, '#888') for p in results.keys()]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    ax.set_ylabel('Wait Time (seconds)')
    ax.set_title('(a) Wait Time Distribution')
    ax.grid(True, alpha=0.3)

    # (b) Wait time by job size group (small vs large)
    ax = axes[1]
    for policy, result in results.items():
        if result.job_wait_times and result.job_sizes:
            median_size = np.median(result.job_sizes)
            small_waits = [w for w, s in zip(result.job_wait_times, result.job_sizes)
                           if s <= median_size]
            large_waits = [w for w, s in zip(result.job_wait_times, result.job_sizes)
                           if s > median_size]
            x = np.arange(2)
            width = 0.35
            offset = -width/2 if policy == 'fcfs' else width/2
            color = POLICY_COLORS.get(policy, '#888')
            vals = [np.mean(small_waits) if small_waits else 0,
                    np.mean(large_waits) if large_waits else 0]
            ax.bar(x + offset, vals, width, label=policy.upper(), color=color, alpha=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Small Jobs', 'Large Jobs'])
    ax.set_ylabel('Avg Wait Time (seconds)')
    ax.set_title('(b) Wait Time by Job Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) Jobs completed
    ax = axes[2]
    policies = list(results.keys())
    completed = [results[p].jobs_completed for p in policies]
    colors = [POLICY_COLORS.get(p, '#888') for p in policies]
    ax.bar([p.upper() for p in policies], completed, color=colors, alpha=0.8)
    ax.set_ylabel('Jobs Completed')
    ax.set_title('(c) System Throughput')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_uc2_utilization_stacked(results, save_path):
    """Plot system utilization stacked area chart (Running/Idle/Fragmented)."""
    if not PLOTTING_AVAILABLE:
        return

    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 5),
                             squeeze=False)

    for idx, (policy, result) in enumerate(results.items()):
        ax = axes[0, idx]
        if result.utilization_breakdown:
            ts = [u['timestep'] / 60.0 for u in result.utilization_breakdown]
            running = [u['running'] for u in result.utilization_breakdown]
            idle = [u['idle'] for u in result.utilization_breakdown]
            fragmented = [u['fragmented'] for u in result.utilization_breakdown]

            ax.stackplot(ts, running, fragmented, idle,
                         labels=['Running', 'Fragmented', 'Idle'],
                         colors=['#2ecc71', '#e74c3c', '#95a5a6'], alpha=0.8)
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel('Nodes')
            ax.set_title(f'{policy.upper()} Node Utilization')
            ax.legend(loc='upper right', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No utilization data', transform=ax.transAxes,
                    ha='center', va='center')

    fig.suptitle('UC2: System Utilization Breakdown', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_uc2_prediction_error(results, save_path):
    """Plot prediction error: planned vs actual job completion time."""
    if not PLOTTING_AVAILABLE:
        return

    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 5),
                             squeeze=False)

    for idx, (policy, result) in enumerate(results.items()):
        ax = axes[0, idx]
        if result.prediction_errors:
            errors = result.prediction_errors
            dilated = [e for e in errors if e['dilated']]
            non_dilated = [e for e in errors if not e['dilated']]

            if non_dilated:
                ax.scatter([e['original_expected'] for e in non_dilated],
                          [e['actual_runtime'] for e in non_dilated],
                          c='#2ecc71', alpha=0.6, s=40, label='Non-dilated')
            if dilated:
                ax.scatter([e['original_expected'] for e in dilated],
                          [e['actual_runtime'] for e in dilated],
                          c='#e74c3c', alpha=0.6, s=40, label='Dilated')

            # Perfect prediction line
            max_val = max(e['actual_runtime'] for e in errors) * 1.1
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Perfect prediction')

            ax.set_xlabel('Planned Runtime (s)')
            ax.set_ylabel('Actual Runtime (s)')
            ax.set_title(f'{policy.upper()} Prediction Error')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No prediction data', transform=ax.transAxes,
                    ha='center', va='center')

    fig.suptitle('UC2: Scheduler Prediction Error (Planned vs Actual)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_uc3_placement(results, save_path):
    """Plot placement comparison for UC3."""
    if not PLOTTING_AVAILABLE:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    allocs = list(results.keys())
    colors = [ALLOCATION_COLORS.get(a, '#888') for a in allocs]

    # (a) Slowdown comparison
    ax = axes[0]
    slowdowns = [results[a].avg_job_slowdown for a in allocs]
    ax.bar([a.title() for a in allocs], slowdowns, color=colors, alpha=0.8)
    ax.set_ylabel('Avg Job Slowdown Factor')
    ax.set_title('(a) Application Performance')
    ax.grid(True, alpha=0.3)

    # (b) Global/Local traffic ratio
    ax = axes[1]
    ratios = [results[a].global_local_ratio for a in allocs]
    ax.bar([a.title() for a in allocs], ratios, color=colors, alpha=0.8)
    ax.set_ylabel('Global / Local Traffic Ratio')
    ax.set_title('(b) Traffic Locality')
    ax.grid(True, alpha=0.3)

    # (c) Congestion
    ax = axes[2]
    congestion = [results[a].max_congestion for a in allocs]
    ax.bar([a.title() for a in allocs], congestion, color=colors, alpha=0.8)
    ax.set_ylabel('Max Congestion')
    ax.set_title('(c) Network Congestion')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_uc3_hop_count(results, save_path):
    """Plot hop count distribution for UC3."""
    if not PLOTTING_AVAILABLE:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for alloc, result in results.items():
        if result.hop_counts:
            color = ALLOCATION_COLORS.get(alloc, '#888')
            ax.hist(result.hop_counts, bins=range(1, max(result.hop_counts) + 2),
                    alpha=0.6, color=color, label=f'{alloc.title()} (avg={result.avg_hop_count:.1f})',
                    edgecolor='white', density=True)

    ax.set_xlabel('Hop Count')
    ax.set_ylabel('Density')
    ax.set_title('UC3: Hop Count Distribution by Placement Strategy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_uc3_speedup_vs_comm(results, save_path):
    """Plot application speedup vs communication intensity for UC3."""
    if not PLOTTING_AVAILABLE:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for alloc, result in results.items():
        if result.job_comm_intensities and result.job_slowdowns:
            color = ALLOCATION_COLORS.get(alloc, '#888')
            ax.scatter(result.job_comm_intensities, result.job_slowdowns,
                      alpha=0.6, color=color, s=30, label=alloc.title())

    ax.set_xlabel('Communication Intensity (bytes/node/s)')
    ax.set_ylabel('Job Slowdown Factor')
    ax.set_title('UC3: Slowdown vs Communication Intensity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if any(r.job_comm_intensities for r in results.values()):
        ax.set_xscale('log')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_uc4_energy(results, save_path):
    """Plot energy comparison for UC4."""
    if not PLOTTING_AVAILABLE:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    configs = list(results.keys())
    labels = ['Ideal\n(no congestion)', 'With\ncongestion']

    # (a) Total energy
    ax = axes[0]
    energies = [results[c].total_energy_joules / 1e6 for c in configs]  # MJ
    colors_list = ['#4ECDC4', '#E17055']
    ax.bar(labels[:len(energies)], energies, color=colors_list[:len(energies)], alpha=0.8)
    ax.set_ylabel('Total Energy (MJ)')
    ax.set_title('(a) Energy-to-Solution')
    ax.grid(True, alpha=0.3)

    # (b) Power profile (if available)
    ax = axes[1]
    for i, (config, label) in enumerate(zip(configs, labels)):
        if results[config].power_history:
            ax.plot(results[config].power_history, label=label.replace('\n', ' '),
                    color=colors_list[i], alpha=0.7, linewidth=1)
    ax.set_xlabel('Tick')
    ax.set_ylabel('Power (kW)')
    ax.set_title('(b) Power Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) Static vs Dynamic power decomposition (stacked bar)
    ax = axes[2]
    x = np.arange(len(configs))
    static_vals = [results[c].static_power_kw for c in configs]
    dynamic_vals = [results[c].dynamic_power_kw for c in configs]
    ax.bar(labels[:len(configs)], static_vals, color='#95a5a6', alpha=0.8, label='Static (idle)')
    ax.bar(labels[:len(configs)], dynamic_vals, bottom=static_vals,
           color='#e67e22', alpha=0.8, label='Dynamic (compute)')
    ax.set_ylabel('Power (kW)')
    ax.set_title('(c) Power Decomposition')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description='RAPS Use Case Evaluation (4 operational scenarios)')
    parser.add_argument('--system', default='lassen', choices=['lassen', 'frontier'],
                        help='System to simulate')
    parser.add_argument('--duration', type=int, default=30,
                        help='Simulation duration in minutes (default: 30)')
    parser.add_argument('--delta-t', type=int, default=1,
                        help='Time step in seconds (default: 1)')
    parser.add_argument('--num-jobs', type=int, default=30,
                        help='Number of jobs (default: 30)')
    parser.add_argument('--uc', type=int, nargs='+', default=None,
                        help='Specific use cases to run (1-4, default: all)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 5 min, 10 jobs')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.duration = 5
        args.num_jobs = 10

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Which use cases to run
    uc_to_run = set(args.uc) if args.uc else {1, 2, 3, 4}

    print("=" * 60)
    print("RAPS Use Case Evaluation")
    print("=" * 60)
    print(f"System:    {args.system}")
    print(f"Duration:  {args.duration} min")
    print(f"Delta-t:   {args.delta_t}s")
    print(f"Jobs:      {args.num_jobs}")
    print(f"Use cases: {sorted(uc_to_run)}")
    print(f"Output:    {output_dir}")

    # Generate workload
    print("\n[Setup] Generating workload...")
    jobs = generate_workload(
        num_jobs=args.num_jobs,
        min_nodes=8,
        max_nodes=256,
    )
    print(f"  Generated {len(jobs)} jobs")
    sizes = [j.nodes_required for j in jobs]
    print(f"  Node sizes: min={min(sizes)}, max={max(sizes)}, "
          f"median={int(np.median(sizes))}")
    print(f"  Arrivals: {jobs[0].submit_time}s to {jobs[-1].submit_time}s")

    all_results = {}

    # UC1: Adaptive Routing
    if 1 in uc_to_run:
        results = run_uc1_routing(
            jobs, args.system, args.duration)
        all_results['uc1'] = results

        if results and not args.no_plots:
            plot_uc1_slowdown_cdf(results, figures_dir / "uc1_slowdown_cdf.png")
            plot_uc1_congestion_heatmap(results, figures_dir / "uc1_congestion_heatmap.png")

        # Save CSV
        rows = []
        for routing, r in results.items():
            rows.append({k: v for k, v in asdict(r).items()
                        if not isinstance(v, list)})
        if rows:
            pd.DataFrame(rows).to_csv(
                output_dir / "uc1_routing_results.csv", index=False)

    # UC2: Scheduling
    if 2 in uc_to_run:
        results = run_uc2_scheduling(
            jobs, args.system, args.duration)
        all_results['uc2'] = results

        if results and not args.no_plots:
            plot_uc2_wait_times(results, figures_dir / "uc2_wait_times.png")
            plot_uc2_utilization_stacked(results, figures_dir / "uc2_utilization_stacked.png")
            plot_uc2_prediction_error(results, figures_dir / "uc2_prediction_error.png")

        rows = []
        for policy, r in results.items():
            rows.append({k: v for k, v in asdict(r).items()
                        if not isinstance(v, list)})
        if rows:
            pd.DataFrame(rows).to_csv(
                output_dir / "uc2_scheduling_results.csv", index=False)

    # UC3: Node Placement
    if 3 in uc_to_run:
        results = run_uc3_placement(
            jobs, args.system, args.duration)
        all_results['uc3'] = results

        if results and not args.no_plots:
            plot_uc3_placement(results, figures_dir / "uc3_placement.png")
            plot_uc3_hop_count(results, figures_dir / "uc3_hop_count.png")
            plot_uc3_speedup_vs_comm(results, figures_dir / "uc3_speedup_vs_comm.png")

        rows = []
        for alloc, r in results.items():
            rows.append({k: v for k, v in asdict(r).items()
                        if not isinstance(v, list)})
        if rows:
            pd.DataFrame(rows).to_csv(
                output_dir / "uc3_placement_results.csv", index=False)

    # UC4: Energy Cost
    if 4 in uc_to_run:
        results = run_uc4_energy(
            jobs, args.system, args.duration)
        all_results['uc4'] = results

        if results and not args.no_plots:
            plot_uc4_energy(results, figures_dir / "uc4_energy.png")

        rows = []
        for config, r in results.items():
            rows.append({k: v for k, v in asdict(r).items()
                        if not isinstance(v, list)})
        if rows:
            pd.DataFrame(rows).to_csv(
                output_dir / "uc4_energy_results.csv", index=False)

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for uc_name, results in all_results.items():
        print(f"\n  {uc_name.upper()}:")
        for label, r in results.items():
            print(f"    {label}: speedup={r.speedup:.1f}x, "
                  f"dilated={r.jobs_dilated}, "
                  f"slowdown={r.avg_job_slowdown:.3f}")

    print(f"\n  Results saved to: {output_dir}")
    if not args.no_plots:
        print(f"  Figures saved to: {figures_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()

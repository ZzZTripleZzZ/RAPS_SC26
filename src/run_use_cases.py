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
       - Compares FCFS vs SJF under realistic network interference
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
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import math
from raps.engine import Engine
from raps.sim_config import SingleSimConfig
from raps.system_config import SystemConfig, get_system_config
from raps.job import Job, job_dict, CommunicationPattern
from raps.stats import get_network_stats, get_engine_stats, get_job_stats
from raps.network.base import (
    worst_link_util, get_link_util_stats, link_loads_for_pattern,
    get_effective_traffic,
)
from raps.network.fat_tree import node_id_to_host_name
from raps.network.dragonfly import parse_dragonfly_host, dragonfly_route

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
    # SC paper style — applied once at import time
    plt.rcParams.update({
        'font.family':        'sans-serif',
        'font.size':          10,
        'axes.labelsize':     11,
        'xtick.labelsize':    10,
        'ytick.labelsize':    10,
        'legend.fontsize':    9,
        'legend.framealpha':  0.85,
        'legend.edgecolor':   '#cccccc',
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'axes.axisbelow':     True,
        'figure.dpi':         300,
        'savefig.dpi':        300,
    })
except ImportError:
    PLOTTING_AVAILABLE = False


# ==========================================
# Node count override (reused from run_frontier.py logic)
# ==========================================

# Fat-tree k values: k^3/4 hosts
FATTREE_K = {
    100:     8,      # 128 hosts
    1_000:   16,     # 1024 hosts
    10_000:  36,     # 36^3/4 = 11664 hosts >= 10008
}

# Dragonfly (d groups, a routers/group, p hosts/router) => d*(d+1)*p hosts
DRAGONFLY_PARAMS = {
    100:     {"d": 10,  "a": 10,  "p": 1},   # 110
    1_000:   {"d": 10,  "a": 10,  "p": 10},  # 1100
    10_000:  {"d": 24,  "a": 24,  "p": 16},  # 9600
}


def _fattree_k_for_nodes(n: int) -> int:
    """Find smallest even k such that k^3/4 >= n."""
    k = 2
    while (k ** 3) // 4 < n:
        k += 2
    return k


def _dragonfly_params_for_nodes(n: int) -> dict:
    """Find d,a,p such that d*(d+1)*p >= n with balanced sizing."""
    d = max(2, int(math.ceil(n ** (1/3))))
    p = max(1, int(math.ceil(n / (d * (d + 1)))))
    a = d  # balanced
    while d * (d + 1) * p < n:
        p += 1
    return {"d": d, "a": a, "p": p}


def _override_system_config_uc(system_name: str, node_count: int) -> SystemConfig:
    """Load a base system config and override node count / network params."""
    base = get_system_config(system_name)
    data = base.model_dump(mode="json")

    # Adjust node grid
    nodes_per_rack = data["system"]["nodes_per_rack"]
    racks_per_cdu = data["system"]["racks_per_cdu"]
    needed_racks = math.ceil(node_count / nodes_per_rack)
    needed_cdus = math.ceil(needed_racks / racks_per_cdu)
    data["system"]["num_cdus"] = needed_cdus
    data["system"]["missing_racks"] = []
    data["system"]["down_nodes"] = []

    # Override network topology params
    if data.get("network"):
        topo = data["network"]["topology"]
        if topo == "fat-tree":
            if node_count not in FATTREE_K:
                FATTREE_K[node_count] = _fattree_k_for_nodes(node_count)
            k = FATTREE_K[node_count]
            data["network"]["fattree_k"] = k
        elif topo == "dragonfly":
            if node_count not in DRAGONFLY_PARAMS:
                DRAGONFLY_PARAMS[node_count] = _dragonfly_params_for_nodes(node_count)
            params = DRAGONFLY_PARAMS[node_count]
            data["network"]["dragonfly_d"] = params["d"]
            data["network"]["dragonfly_a"] = params["a"]
            data["network"]["dragonfly_p"] = params["p"]

    return SystemConfig.model_validate(data)


# ==========================================
# Configuration
# ==========================================

OUTPUT_DIR = PROJECT_ROOT / "output" / "use_cases"
FIGURES_DIR = OUTPUT_DIR / "figures"
MATRIX_DIR = PROJECT_ROOT / "data" / "matrices"

MINI_APP_PATTERNS = {
    'lulesh': CommunicationPattern.STENCIL_3D,
    'comd': CommunicationPattern.STENCIL_3D,
    'hpgmg': CommunicationPattern.STENCIL_3D,
    'cosp2': CommunicationPattern.STENCIL_3D,  # Use STENCIL_3D for performance (O(N) vs O(N²))
}

# ColorBrewer Dark2 — colorblind-friendly
ROUTING_COLORS = {
    'minimal':  '#1B9E77',
    'ecmp':     '#1B9E77',
    'ugal':     '#7570B3',
    'valiant':  '#D95F02',
    'adaptive': '#E7298A',
}

POLICY_COLORS = {
    'fcfs':          '#D95F02',
    'fcfs+easy':     '#1B9E77',
    'fcfs+firstfit': '#1B9E77',
    'sjf':           '#7570B3',
}

ALLOCATION_COLORS = {
    'contiguous': '#1B9E77',
    'random':     '#D95F02',
    'hybrid':     '#7570B3',
}

# Common figure dimensions
_FIGW = 3.5   # single-column width (inches)
_FIGH = 2.0   # ~7:4 ratio
_DPI  = 300


# ==========================================
# SimConfig with proper routing override
# ==========================================

class UCSimConfig(SingleSimConfig):
    """SimConfig for use case experiments."""
    pass


def override_system_routing(sim_config, routing, ugal_threshold=2.0, valiant_bias=0.3):
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
    min_duration: int = 120,
    max_duration: int = 600,
    seed: int = 42,
    mean_inter_arrival: float = 60.0,
) -> List[Job]:
    """
    Generate synthetic workload with log-normal job sizes and Poisson arrivals.
    Jobs have realistic network traces assigned based on communication pattern.

    Parameters
    ----------
    mean_inter_arrival : float
        Mean seconds between job arrivals (Poisson process). Default 60s.
        Set lower (e.g. 10–20s) for high-load scenarios.
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
    inter_arrivals = rng.exponential(mean_inter_arrival, num_jobs)
    arrival_times = np.cumsum(inter_arrivals).astype(int)

    # Log-normal durations (mean=5.8 => exp(5.8) ≈ 330s, centered in [120, 600])
    raw_durations = rng.lognormal(mean=5.8, sigma=0.6, size=num_jobs)
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

        # Realistic HPC traffic: ~1.25 GB/s per node on 100 Gbps NICs
        # tx_volume_bytes = per-node total send volume per trace_quanta.
        # The coefficient functions (compute_stencil_3d_coefficients etc.) already
        # iterate over all N nodes' pairs, so we must NOT multiply by job_size here.
        NIC_BW_BYTES_PER_S = 1.25e9  # 10 Gbps effective per node (typical HPC)
        base_bw = NIC_BW_BYTES_PER_S * trace_quanta  # bytes per node per quanta

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

    For dragonfly, uses routing-aware path computation so that non-minimal
    algorithms (ugal, valiant) show longer average hop counts than minimal.
    For fat-tree, uses shortest-path (minimal = ecmp hop lengths are equal;
    adaptive may differ but path enumeration is too expensive for sampling).
    """
    hop_counts = []
    if not engine.simulate_network or not engine.network_model:
        return hop_counts

    net_model = engine.network_model
    G = net_model.net_graph
    if G is None:
        return hop_counts

    routing = net_model.routing_algorithm

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
                        if net_model.topology == "dragonfly":
                            # Use routing-aware path so ugal/valiant show higher
                            # hop counts than minimal routing.
                            D = getattr(net_model, 'dragonfly_d', 0)
                            A = getattr(net_model, 'dragonfly_a', 0)
                            path = dragonfly_route(
                                hosts[i], hosts[j_idx], routing, D, A,
                                ugal_threshold=getattr(net_model, 'ugal_threshold', 2.0),
                                valiant_bias=getattr(net_model, 'valiant_bias', 0.0),
                            )
                        else:
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

    Uses sys_util_history (timestep, util%) to derive running nodes, and
    scheduler_running_history (queue length at each tick) to estimate
    fragmentation: if there are jobs waiting in the queue while nodes sit
    idle, those idle nodes are likely fragmented (too scattered to satisfy
    any pending request).

    Returns list of dicts with keys: timestep, running, idle, fragmented.
    """
    total_nodes = engine.config['AVAILABLE_NODES']
    down_nodes_count = len(engine.config.get('DOWN_NODES', []))
    available = total_nodes - down_nodes_count

    util_breakdown = []
    for i, (ts, util_pct) in enumerate(engine.sys_util_history):
        # Derive running node count from utilization percentage
        running_nodes = int(round(util_pct / 100.0 * available))
        running_nodes = max(0, min(running_nodes, available))
        free_count = available - running_nodes

        # scheduler_running_history stores queue length (pending jobs) at each tick.
        # If queue > 0 and free nodes exist, those free nodes are fragmented.
        queue_len = 0
        if i < len(engine.scheduler_running_history):
            queue_len = engine.scheduler_running_history[i]

        if queue_len > 0 and free_count > 0:
            # All free nodes are considered fragmented when jobs are waiting
            fragmented = free_count
        else:
            fragmented = 0
        truly_idle = free_count - fragmented

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
# Incremental CSV Helpers
# ==========================================

def _load_done_labels(csv_path) -> set:
    """Return set of variant labels already saved in a result CSV."""
    if csv_path is None or not Path(csv_path).exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
        if 'label' in df.columns:
            return set(df['label'].tolist())
    except Exception:
        pass
    return set()


def _append_result_to_csv(csv_path, result) -> None:
    """Append a single SimResult row to CSV immediately (create or append)."""
    if csv_path is None or result is None:
        return
    row = {k: v for k, v in asdict(result).items() if not isinstance(v, list)}
    row_df = pd.DataFrame([row])
    csv_path = Path(csv_path)
    if csv_path.exists():
        row_df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        row_df.to_csv(csv_path, index=False)
    print(f"  [Saved] {csv_path.name} <- {result.label}")


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
    energy_per_completed_job: float = 0.0  # MJ per completed job
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

    # Stall/packet ratio (Cassini counter analog)
    # stall_packet_ratio = (hni_tx_paused_0 + hni_tx_paused_1) / parbs_tarb_pi_posted_pkts
    avg_stall_ratio: float = 0.0

    # Per-job details (for CDF etc.)
    job_slowdowns: list = field(default_factory=list)
    job_stall_ratios: list = field(default_factory=list)
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
    backfill: str = None,
    simulate_network: bool = True,
    label: str = "",
    node_count: int = None,
    autoshutdown: bool = False,
) -> Optional[SimResult]:
    """
    Run a single DES simulation with full metric collection.

    Parameters
    ----------
    node_count : int, optional
        If given, override the system config to simulate this many nodes
        (adjusts CDU layout, network topology params, etc.).
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
        # Backfill policy (None = no backfill)
        'backfill': backfill,
        # Required defaults for synthetic workload generator
        'jobsize_distribution': ['uniform'],
        'walltime_distribution': ['uniform'],
    }

    try:
        t_start = time.perf_counter()

        sim_config = UCSimConfig(**config_dict)

        # Override system config for different node counts
        if node_count is not None:
            sys_cfg = _override_system_config_uc(system, node_count)
            sim_config._system_configs = [sys_cfg]

        # Override routing
        if routing and simulate_network:
            override_system_routing(sim_config, routing)

        engine = Engine(sim_config)

        # Replace jobs with our prepared workload
        engine.jobs = clone_jobs(jobs)

        t_engine = time.perf_counter()

        # Run simulation
        tick_count = 0
        for tick_data in engine.run_simulation(autoshutdown=autoshutdown):
            tick_count += 1

        t_end = time.perf_counter()
        sim_time = t_end - t_engine
        total_time = t_end - t_start
        # tick_count = number of outer loop iterations in run_simulation (one per
        # simulated second, regardless of delta_t). delta_t only controls how often
        # tick() is called, not how fast the outer loop advances.
        simulated_seconds = tick_count
        speedup = simulated_seconds / sim_time if sim_time > 0 else float('inf')

        # Collect network stats
        net_stats = get_network_stats(engine) if simulate_network else {}

        # Collect per-job metrics
        all_jobs = engine.jobs
        job_slowdowns = []
        job_wait_times = []
        job_sizes = []
        job_comm_intensities = []
        job_stall_ratios = []
        dilated_count = 0

        for job in all_jobs:
            job_sizes.append(job.nodes_required)
            # Use slowdown_factor (congestion level at last active tick)
            sf = getattr(job, 'slowdown_factor', 1.0)
            job_slowdowns.append(sf)
            job_comm_intensities.append(compute_comm_intensity(job))
            if getattr(job, 'dilated', False):
                dilated_count += 1

            st = getattr(job, 'start_time', None)
            if st is not None:
                wait = st - job.submit_time
                job_wait_times.append(max(0, wait))

            # Stall/packet ratio per job (Cassini counter analog: slowdown_factor - 1)
            job_stall_ratios.append(getattr(job, 'stall_ratio', 0.0))

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

        # Compute slowdown from per-job congestion levels at last active tick
        # (job.slowdown_factor = net_cong at last tick; 1.0 for non-congested jobs)
        avg_job_slowdown_val = float(np.mean(job_slowdowns)) if job_slowdowns else 1.0
        max_job_slowdown_val = float(max(job_slowdowns)) if job_slowdowns else 1.0

        # System-level average stall/packet ratio (Cassini counter analog)
        avg_stall_ratio_val = float(np.mean(job_stall_ratios)) if job_stall_ratios else 0.0

        # Jobs completed
        jobs_completed = sum(1 for j in all_jobs
                            if getattr(j, 'end_time', None) is not None)

        # Energy per completed job (MJ/job) — captures the real cost of congestion:
        # fewer completions in the same window → higher energy per useful job.
        energy_per_completed_job_mj = (
            (total_energy / 1e6) / jobs_completed if jobs_completed > 0 else 0.0
        )

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
            avg_job_slowdown=avg_job_slowdown_val,
            max_job_slowdown=max_job_slowdown_val,
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
            energy_per_completed_job=energy_per_completed_job_mj,
            avg_power_watts=avg_power,
            peak_power_watts=peak_power,
            static_power_kw=power_decomp['static_power_kw'],
            dynamic_power_kw=power_decomp['dynamic_power_kw'],
            static_power_pct=power_decomp['static_pct'],
            avg_hop_count=avg_hops,
            global_local_ratio=gl_ratio['ratio'],
            global_traffic_pairs=gl_ratio['global_traffic'],
            local_traffic_pairs=gl_ratio['local_traffic'],
            avg_stall_ratio=avg_stall_ratio_val,
            job_slowdowns=job_slowdowns,
            job_wait_times=job_wait_times,
            job_sizes=job_sizes,
            job_comm_intensities=job_comm_intensities,
            job_stall_ratios=job_stall_ratios,
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
    if result.avg_stall_ratio > 0:
        print(f"{indent}Avg stall/pkt ratio: {result.avg_stall_ratio:.3e} "
              f"(Cassini: hni_tx_paused / tarb_posted_pkts)")
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

def _get_topology(system: str) -> str:
    """Return the network topology for a system."""
    base = get_system_config(system)
    data = base.model_dump(mode="json")
    return data.get("network", {}).get("topology", "fat-tree")


def _all_routing_algos_for_system(system: str) -> List[str]:
    """Return all available routing algorithms for a system's topology."""
    topo = _get_topology(system)
    if topo == "dragonfly":
        return ["minimal", "ugal", "valiant"]
    else:  # fat-tree
        return ["minimal", "ecmp", "adaptive"]


def _adaptive_routing_for_system(system: str) -> str:
    """Return the best adaptive routing algorithm for a system's topology."""
    topo = _get_topology(system)
    if topo == "dragonfly":
        return "ugal"
    else:
        return "adaptive"


def run_uc1_routing(jobs, system, duration_minutes, node_count=None, delta_t=1,
                    resume_csv=None, done_labels=None, **kwargs):
    """
    UC1: Compare ALL routing algorithms for the system's topology.

    Evaluates the impact of routing policies on the bully effect.
    - Dragonfly (Frontier): minimal, ugal, valiant
    - Fat-tree (Lassen): minimal, ecmp, adaptive
    """
    print("\n" + "=" * 60)
    print("UC1: Adaptive Routing and Congestion Mitigation")
    print("=" * 60)

    routing_algos = _all_routing_algos_for_system(system)
    topo = _get_topology(system)
    print(f"  Topology: {topo}")
    print(f"  Routing algorithms: {', '.join(routing_algos)}")
    results = {}
    done_labels = done_labels or set()

    for routing in routing_algos:
        label = f"UC1_{routing}"
        if label in done_labels:
            print(f"\n  [SKIP] routing={routing} (already saved)")
            continue
        print(f"\n  Running with routing={routing}...")
        result = run_simulation(
            jobs=jobs,
            system=system,
            duration_minutes=duration_minutes,
            delta_t=delta_t,
            routing=routing,
            allocation='contiguous',
            policy='fcfs',
            simulate_network=True,
            label=label,
            node_count=node_count,
        )
        if result:
            results[routing] = result
            print_result(result, indent="    ")
            _append_result_to_csv(resume_csv, result)

    # Comparison table
    if len(results) >= 2:
        print(f"\n  --- UC1 Comparison ({topo}) ---")
        r_keys = list(results.keys())
        header = f"  {'Metric':<30}" + "".join(f" {k:>12}" for k in r_keys)
        print(header)
        print(f"  {'-'*(30 + 13 * len(r_keys))}")

        metrics = [
            ('Avg Slowdown', 'avg_job_slowdown'),
            ('Max Slowdown', 'max_job_slowdown'),
            ('Dilated Jobs (%)', 'dilated_pct'),
            ('Max Congestion', 'max_congestion'),
            ('Avg Network Util (%)', 'avg_network_util'),
            ('Avg Stall/Pkt Ratio', 'avg_stall_ratio'),
            ('Throughput (Gbps)', None),  # special
            ('Potential Bullies', None),  # special
        ]
        for name, attr in metrics:
            vals = []
            for k in r_keys:
                r = results[k]
                if attr:
                    vals.append(getattr(r, attr))
                elif name == 'Throughput (Gbps)':
                    vals.append(r.global_throughput_bps / 1e9)
                elif name == 'Potential Bullies':
                    vals.append(float(r.potential_bullies))
            row = f"  {name:<30}" + "".join(f" {v:>12.3f}" for v in vals)
            print(row)

    return results


# ==========================================
# UC2: Scheduler Policy Optimization
# ==========================================

def run_uc2_scheduling(jobs, system, duration_minutes, node_count=None, delta_t=1,
                       resume_csv=None, done_labels=None, **kwargs):
    """
    UC2: Compare scheduling policies under network interference.

    Compares FCFS (no backfill), FCFS+Easy backfill, FCFS+Firstfit backfill,
    and SJF. Under network congestion, runtime dilation disrupts scheduling
    reservations since actual job completion deviates from estimates.
    """
    print("\n" + "=" * 60)
    print("UC2: Scheduler Policy Optimization")
    print("=" * 60)

    adaptive_algo = _adaptive_routing_for_system(system)

    sched_configs = [
        ('fcfs',           'fcfs', None),
        ('fcfs+firstfit',  'fcfs', 'firstfit'),
        ('sjf',            'sjf',  None),
    ]
    print(f"  Routing: {adaptive_algo}")
    print(f"  Policies: {', '.join(c[0] for c in sched_configs)}")
    results = {}
    done_labels = done_labels or set()

    for sched_label, policy, bf in sched_configs:
        label = f"UC2_{sched_label}"
        if label in done_labels:
            print(f"\n  [SKIP] policy={sched_label} (already saved)")
            continue
        bf_str = f" + backfill={bf}" if bf else ""
        print(f"\n  Running with policy={policy}{bf_str}...")
        result = run_simulation(
            jobs=jobs,
            system=system,
            duration_minutes=duration_minutes,
            delta_t=delta_t,
            routing=adaptive_algo,
            allocation='contiguous',
            policy=policy,
            backfill=bf,
            simulate_network=True,
            label=label,
            node_count=node_count,
        )
        if result:
            results[sched_label] = result
            print_result(result, indent="    ")
            _append_result_to_csv(resume_csv, result)
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
                    print(f"    Small jobs (<= {int(median_size)} nodes) avg wait: "
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

    # Comparison table
    if len(results) >= 2:
        print(f"\n  --- UC2 Comparison ---")
        r_keys = list(results.keys())
        header = f"  {'Metric':<30}" + "".join(f" {k:>14}" for k in r_keys)
        print(header)
        print(f"  {'-'*(30 + 15 * len(r_keys))}")

        metric_attrs = [
            ('Jobs Completed', 'jobs_completed', True),
            ('Avg Wait Time (s)', 'avg_wait_time', False),
            ('Max Wait Time (s)', 'max_wait_time', False),
            ('Avg Slowdown', 'avg_job_slowdown', False),
            ('Dilated Jobs (%)', 'dilated_pct', False),
            ('Avg Network Util (%)', 'avg_network_util', False),
        ]
        for name, attr, as_int in metric_attrs:
            vals = []
            for k in r_keys:
                v = getattr(results[k], attr)
                vals.append(float(v))
            if as_int:
                row = f"  {name:<30}" + "".join(f" {int(v):>14d}" for v in vals)
            else:
                row = f"  {name:<30}" + "".join(f" {v:>14.3f}" for v in vals)
            print(row)

    return results


# ==========================================
# UC3: Topology-Aware Node Placement
# ==========================================

def run_uc3_placement(jobs, system, duration_minutes, node_count=None, delta_t=1,
                      resume_csv=None, done_labels=None, **kwargs):
    """
    UC3: Compare all node placement strategies.

    - Contiguous: keeps jobs within a single electrical group (Short Circuit)
    - Random: spreads jobs across the fabric
    - Hybrid: comm-intensive jobs get contiguous, others get random
    """
    print("\n" + "=" * 60)
    print("UC3: Topology-Aware Node Placement")
    print("=" * 60)

    adaptive_algo = _adaptive_routing_for_system(system)
    allocations = ['contiguous', 'random', 'hybrid']
    print(f"  Allocation strategies: {', '.join(allocations)}")
    results = {}
    done_labels = done_labels or set()

    for alloc in allocations:
        label = f"UC3_{alloc}"
        if label in done_labels:
            print(f"\n  [SKIP] allocation={alloc} (already saved)")
            continue
        print(f"\n  Running with allocation={alloc}...")
        result = run_simulation(
            jobs=jobs,
            system=system,
            duration_minutes=duration_minutes,
            delta_t=delta_t,
            routing=adaptive_algo,
            allocation=alloc,
            policy='fcfs',
            simulate_network=True,
            label=label,
            node_count=node_count,
        )
        if result:
            results[alloc] = result
            print_result(result, indent="    ")
            _append_result_to_csv(resume_csv, result)

    # Comparison table
    if len(results) >= 2:
        print(f"\n  --- UC3 Comparison ---")
        r_keys = list(results.keys())
        header = f"  {'Metric':<30}" + "".join(f" {k:>12}" for k in r_keys)
        print(header)
        print(f"  {'-'*(30 + 13 * len(r_keys))}")

        metric_attrs = [
            ('Avg Slowdown', 'avg_job_slowdown'),
            ('Max Slowdown', 'max_job_slowdown'),
            ('Dilated Jobs (%)', 'dilated_pct'),
            ('Max Congestion', 'max_congestion'),
            ('Avg Network Util (%)', 'avg_network_util'),
            ('Avg Hop Count', 'avg_hop_count'),
            ('Global/Local Ratio', 'global_local_ratio'),
            ('Potential Bullies', None),
        ]
        for name, attr in metric_attrs:
            vals = []
            for k in r_keys:
                r = results[k]
                if attr:
                    vals.append(getattr(r, attr))
                else:
                    vals.append(float(r.potential_bullies))
            row = f"  {name:<30}" + "".join(f" {v:>12.3f}" for v in vals)
            print(row)

        # Application speedup: each strategy relative to random
        r_rand = results.get('random')
        if r_rand and r_rand.avg_job_slowdown > 0:
            print(f"\n  Speedup relative to random placement:")
            for k in r_keys:
                if k != 'random':
                    benefit = r_rand.avg_job_slowdown / results[k].avg_job_slowdown
                    print(f"    {k}: {benefit:.2f}x")

        # Speedup vs communication intensity (contiguous vs random)
        r_cont = results.get('contiguous')
        if r_cont and r_rand:
            print(f"\n  --- Speedup vs Communication Intensity (Contiguous vs Random) ---")
            if r_cont.job_comm_intensities and r_rand.job_comm_intensities:
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

def run_uc4_energy(jobs, system, duration_minutes, node_count=None, delta_t=1,
                   resume_csv=None, done_labels=None, until_complete=True, **kwargs):
    """
    UC4: Quantify the energy cost of network congestion.

    Compares energy-to-solution under:
    1. No network (ideal baseline)
    2. Each routing algorithm (to show how routing choice affects energy)

    When congestion dilates job runtime, static power (leakage) continues
    to accumulate even though dynamic work is stalled — the "Energy Tax."
    """
    print("\n" + "=" * 60)
    print("UC4: Energy Cost of Congestion")
    print("=" * 60)

    routing_algos = _all_routing_algos_for_system(system)

    # For fat-tree systems (e.g. lassen), 'adaptive' routing behaves like 'ecmp' under
    # block allocation (near-zero inter-switch traffic).  Excluding it avoids an
    # extra ~88-min SLURM job while keeping the scientific comparison meaningful.
    if _get_topology(system) != 'dragonfly':
        adaptive_algo = _adaptive_routing_for_system(system)
        routing_algos = [a for a in routing_algos if a != adaptive_algo]

    # Build experiment configs: baseline (no network) + each routing algo
    configs = [('no_congestion', None, False)]  # (label, routing, simulate_network)
    for algo in routing_algos:
        configs.append((algo, algo, True))

    print(f"  Configurations: no_congestion, {', '.join(routing_algos)}")
    results = {}
    done_labels = done_labels or set()
    # Generous time budget for until_complete mode (cap at 10 h to avoid infinite runs)
    budget = min(duration_minutes * 5, 600) if until_complete else duration_minutes

    for config_label, routing, sim_net in configs:
        label = f"UC4_{config_label}"
        if label in done_labels:
            print(f"\n  [SKIP] {config_label} (already saved)")
            continue
        desc = "Ideal (no congestion)" if not sim_net else f"routing={routing}"
        if until_complete:
            desc += f" [until-complete, budget={budget}min]"
        print(f"\n  Running: {desc}...")
        result = run_simulation(
            jobs=jobs,
            system=system,
            duration_minutes=budget,
            delta_t=delta_t,
            routing=routing,
            allocation='contiguous',
            policy='fcfs',
            simulate_network=sim_net,
            label=label,
            node_count=node_count,
            autoshutdown=until_complete,
        )
        if result:
            results[config_label] = result
            print_result(result, indent="    ")
            _append_result_to_csv(resume_csv, result)
            print(f"    Total energy: {result.total_energy_joules:.0f} J, "
                  f"Avg power: {result.avg_power_watts:.1f} W, "
                  f"Peak power: {result.peak_power_watts:.1f} W, "
                  f"Makespan: {result.simulated_seconds/60:.1f} min")

    # Energy comparison table
    if len(results) >= 2:
        print(f"\n  --- UC4 Energy Cost Analysis ---")
        r_keys = list(results.keys())
        header = f"  {'Metric':<30}" + "".join(f" {k:>14}" for k in r_keys)
        print(header)
        print(f"  {'-'*(30 + 15 * len(r_keys))}")

        metric_list = [
            ('Makespan (min)', None),          # derived from simulated_seconds
            ('Total Energy (J)', 'total_energy_joules'),
            ('Energy/Job (MJ)', 'energy_per_completed_job'),
            ('Avg Power (W)', 'avg_power_watts'),
            ('Peak Power (W)', 'peak_power_watts'),
            ('Jobs Completed', 'jobs_completed'),
            ('Dilated Jobs (%)', 'dilated_pct'),
            ('Avg Slowdown', 'avg_job_slowdown'),
        ]
        for name, attr in metric_list:
            if attr is None:
                vals = [results[k].simulated_seconds / 60 for k in r_keys]
            else:
                vals = [float(getattr(results[k], attr)) for k in r_keys]
            row = f"  {name:<30}" + "".join(f" {v:>14.1f}" for v in vals)
            print(row)

        # Energy overhead relative to ideal
        r_ideal = results.get('no_congestion')
        if r_ideal and r_ideal.total_energy_joules > 0:
            e_ideal = r_ideal.total_energy_joules
            print(f"\n  Energy overhead vs ideal (no congestion):")
            for k in r_keys:
                if k == 'no_congestion':
                    continue
                e = results[k].total_energy_joules
                overhead = (e - e_ideal) / e_ideal * 100
                print(f"    {k}: {overhead:+.1f}%")

        # Power decomposition for ideal vs worst congestion
        congested_keys = [k for k in r_keys if k != 'no_congestion']
        if r_ideal and congested_keys:
            # Pick the routing with highest energy for detailed decomposition
            worst_key = max(congested_keys,
                           key=lambda k: results[k].total_energy_joules)
            r_cong = results[worst_key]

            print(f"\n  --- Power Decomposition (ideal vs {worst_key}) ---")
            print(f"  {'Component':<25} {'Ideal (kW)':>12} {worst_key+' (kW)':>15}")
            print(f"  {'-'*52}")
            print(f"  {'Static (idle leakage)':<25} {r_ideal.static_power_kw:>12.1f} "
                  f"{r_cong.static_power_kw:>15.1f}")
            print(f"  {'Dynamic (compute)':<25} {r_ideal.dynamic_power_kw:>12.1f} "
                  f"{r_cong.dynamic_power_kw:>15.1f}")
            print(f"  {'Static fraction':<25} {r_ideal.static_power_pct:>11.1f}% "
                  f"{r_cong.static_power_pct:>14.1f}%")

            # Static power tax
            if r_cong.jobs_dilated > 0 and r_cong.prediction_errors:
                extra_runtime_s = sum(
                    max(0, e['actual_runtime'] - e['original_expected'])
                    for e in r_cong.prediction_errors if e['dilated']
                )
                e_ideal_j = r_ideal.total_energy_joules
                e_cong_j = r_cong.total_energy_joules
                energy_overhead_j = e_cong_j - e_ideal_j
                static_frac = r_cong.static_power_pct / 100.0 if r_cong.static_power_pct > 0 else 0.5
                static_tax_j = energy_overhead_j * static_frac
                print(f"\n  Static power tax from dilation ({worst_key}): ~{static_tax_j:.0f} J "
                      f"({static_frac*100:.0f}% of {energy_overhead_j:.0f} J overhead)")
                print(f"  ({r_cong.jobs_dilated} dilated jobs, "
                      f"total extra runtime: {extra_runtime_s:.0f}s)")

    return results


# ==========================================
# Plotting  (SC single-column style, no subfigures)
# figsize = (_FIGW, _FIGH) = (3.5 in × 2.0 in)  ~7:4
# ==========================================

def _ygrid(ax):
    ax.yaxis.grid(True, linestyle='--', alpha=0.25, linewidth=0.7, color='#888')


def _save_fig(fig, path):
    fig.tight_layout(pad=0.5)
    out = Path(path).with_suffix('.png')
    fig.savefig(out, dpi=_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out.name}")


def _bar_single(ax, labels, values, colors, ylabel, baseline=None, yscale='linear'):
    """Populate a single-panel bar chart on ax (no title)."""
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor='none', width=0.52, alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    if baseline is not None:
        ax.axhline(baseline, color='#444', ls='--', lw=1.3, alpha=0.8)
    v_max = max(values) if values else 1.0
    for b, v in zip(bars, values):
        ypos = v * 1.12 if yscale == 'log' else v + v_max * 0.025
        ax.text(b.get_x() + b.get_width() / 2, ypos,
                f'{v:.2g}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    _ygrid(ax)


# ── UC1 ──────────────────────────────────────────────────────────────────────

def plot_uc1_slowdown_cdf(results, save_path):
    """CDF of per-job slowdown factors — one line per routing algo."""
    if not PLOTTING_AVAILABLE:
        return
    fig, ax = plt.subplots(figsize=(_FIGW, _FIGH))
    all_sd = [sd for r in results.values() for sd in r.job_slowdowns]
    use_log = bool(all_sd) and max(all_sd) > 100
    for routing, result in results.items():
        if result.job_slowdowns:
            s = np.sort(result.job_slowdowns)
            cdf = np.arange(1, len(s) + 1) / len(s)
            ax.step(s, cdf, where='post',
                    label=routing.title(),
                    color=ROUTING_COLORS.get(routing, '#888'),
                    linewidth=2)
    if use_log:
        ax.set_xscale('log')
    ax.set_xlim(left=0.9)
    ax.set_xlabel('Job slowdown factor' + (' (log)' if use_log else ''))
    ax.set_ylabel('CDF')
    ax.legend(loc='lower right')
    _ygrid(ax)
    _save_fig(fig, save_path)


def plot_uc1_congestion_heatmap(results, save_dir):
    """
    Congestion over time — one overlaid line per routing algo
    (single figure, no subfigures).
    save_dir: directory where uc1_congestion.png will be written.
    """
    if not PLOTTING_AVAILABLE:
        return
    fig, ax = plt.subplots(figsize=(_FIGW, _FIGH))
    has_data = False
    for routing, result in results.items():
        if result.congestion_timesteps and result.congestion_values:
            ts   = np.array(result.congestion_timesteps) / 60.0
            cong = np.array(result.congestion_values)
            ax.plot(ts, cong, label=routing.title(),
                    color=ROUTING_COLORS.get(routing, '#888'),
                    linewidth=1.8, alpha=0.88)
            has_data = True
    if not has_data:
        ax.text(0.5, 0.5, 'No congestion data', transform=ax.transAxes,
                ha='center', va='center', color='gray')
    ax.axhline(1.0, color='#d62728', linestyle='--', linewidth=1.2,
               alpha=0.7, label='Saturation (1.0)')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Max link overload ratio')
    ax.legend(loc='upper right', fontsize=8)
    _ygrid(ax)
    _save_fig(fig, Path(save_dir) / "uc1_congestion.png")


# ── UC2 ──────────────────────────────────────────────────────────────────────

def plot_uc2_wait_times(results, save_dir):
    """
    Three separate figures:
      uc2_wait_boxplot.png   — wait-time distribution by policy
      uc2_wait_by_size.png   — avg wait time, small vs large jobs
      uc2_throughput.png     — jobs completed
    """
    if not PLOTTING_AVAILABLE:
        return
    policies = list(results.keys())
    colors   = [POLICY_COLORS.get(p, '#888') for p in policies]
    save_dir = Path(save_dir)

    # (1) Wait-time boxplot
    data   = [r.job_wait_times for r in results.values() if r.job_wait_times]
    labels = [p.upper() for p, r in results.items() if r.job_wait_times]
    if data:
        fig, ax = plt.subplots(figsize=(_FIGW, _FIGH))
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5)
        box_colors = [POLICY_COLORS.get(p, '#888')
                      for p, r in results.items() if r.job_wait_times]
        for patch, c in zip(bp['boxes'], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        ax.set_ylabel('Wait time (s)')
        ax.tick_params(axis='x', rotation=15)
        _ygrid(ax)
        _save_fig(fig, save_dir / "uc2_wait_boxplot.png")

    # (2) Wait time by job size
    n = len(policies)
    width = 0.7 / n
    fig, ax = plt.subplots(figsize=(_FIGW, _FIGH))
    for i, (policy, result) in enumerate(results.items()):
        if result.job_wait_times and result.job_sizes:
            med = np.median(result.job_sizes)
            sw = [w for w, s in zip(result.job_wait_times, result.job_sizes) if s <= med]
            lw = [w for w, s in zip(result.job_wait_times, result.job_sizes) if s > med]
            offset = (i - n / 2 + 0.5) * width
            vals = [np.mean(sw) if sw else 0, np.mean(lw) if lw else 0]
            ax.bar(np.arange(2) + offset, vals, width,
                   label=policy.upper(),
                   color=POLICY_COLORS.get(policy, '#888'), alpha=0.85)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Small jobs', 'Large jobs'])
    ax.set_ylabel('Avg wait time (s)')
    ax.legend(fontsize=8, loc='upper left')
    _ygrid(ax)
    _save_fig(fig, save_dir / "uc2_wait_by_size.png")

    # (3) Jobs completed
    fig, ax = plt.subplots(figsize=(_FIGW, _FIGH))
    completed = [results[p].jobs_completed for p in policies]
    _bar_single(ax, [p.upper() for p in policies], completed, colors,
                'Jobs completed')
    _save_fig(fig, save_dir / "uc2_throughput.png")


def plot_uc2_utilization_stacked(results, save_dir):
    """
    Two separate figures:
      uc2_running_nodes.png  — running-node time series per policy
      uc2_node_state.png     — avg node state breakdown (grouped bars)
    """
    if not PLOTTING_AVAILABLE:
        return
    save_dir = Path(save_dir)

    # (1) Running nodes over time
    fig, ax = plt.subplots(figsize=(_FIGW, _FIGH))
    has_data = False
    for policy, result in results.items():
        if result.utilization_breakdown:
            ts      = [u['timestep'] / 60.0 for u in result.utilization_breakdown]
            running = [u['running']          for u in result.utilization_breakdown]
            ax.plot(ts, running, label=policy.upper(),
                    color=POLICY_COLORS.get(policy, '#888'),
                    linewidth=1.8, alpha=0.88)
            has_data = True
    if not has_data:
        ax.text(0.5, 0.5, 'No utilization data', transform=ax.transAxes,
                ha='center', va='center', color='gray')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Running nodes')
    ax.legend(fontsize=8, loc='lower right')
    _ygrid(ax)
    _save_fig(fig, save_dir / "uc2_running_nodes.png")

    # (2) Average node-state breakdown
    policies = list(results.keys())
    x, w = np.arange(len(policies)), 0.24
    avg_run, avg_frag, avg_idle = [], [], []
    for policy in policies:
        r = results[policy]
        if r.utilization_breakdown:
            avg_run.append(np.mean([u['running']    for u in r.utilization_breakdown]))
            avg_frag.append(np.mean([u['fragmented'] for u in r.utilization_breakdown]))
            avg_idle.append(np.mean([u['idle']       for u in r.utilization_breakdown]))
        else:
            avg_run.append(0); avg_frag.append(0); avg_idle.append(0)
    fig, ax = plt.subplots(figsize=(_FIGW, _FIGH))
    ax.bar(x - w, avg_run,  w, label='Running',    color='#1B9E77', alpha=0.85)
    ax.bar(x,     avg_frag, w, label='Fragmented', color='#D95F02', alpha=0.85)
    ax.bar(x + w, avg_idle, w, label='Idle',       color='#7570B3', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in policies], rotation=15)
    ax.set_ylabel('Avg nodes')
    ax.legend(fontsize=8, loc='upper right')
    _ygrid(ax)
    _save_fig(fig, save_dir / "uc2_node_state.png")


def plot_uc2_prediction_error(results, save_dir):
    """
    One scatter figure per policy:
      uc2_pred_error_{policy}.png  — planned vs actual runtime
    """
    if not PLOTTING_AVAILABLE:
        return
    save_dir = Path(save_dir)
    for policy, result in results.items():
        fig, ax = plt.subplots(figsize=(_FIGW, _FIGH))
        if result.prediction_errors:
            errors     = result.prediction_errors
            dilated    = [e for e in errors if     e['dilated']]
            non_dilated= [e for e in errors if not e['dilated']]
            if non_dilated:
                ax.scatter([e['original_expected'] for e in non_dilated],
                           [e['actual_runtime']    for e in non_dilated],
                           c='#1B9E77', alpha=0.6, s=18, label='On-time')
            if dilated:
                ax.scatter([e['original_expected'] for e in dilated],
                           [e['actual_runtime']    for e in dilated],
                           c='#D95F02', alpha=0.6, s=18, label='Dilated')
            mv = max(e['actual_runtime'] for e in errors) * 1.08
            ax.plot([0, mv], [0, mv], 'k--', linewidth=1, alpha=0.3,
                    label='Perfect prediction')
            ax.set_xlabel('Planned runtime (s)')
            ax.set_ylabel('Actual runtime (s)')
            ax.legend(fontsize=8, loc='upper left')
            _ygrid(ax)
        else:
            ax.text(0.5, 0.5, 'No prediction data', transform=ax.transAxes,
                    ha='center', va='center', color='gray')
        _save_fig(fig, save_dir / f"uc2_pred_error_{policy}.png")


# ── UC3 ──────────────────────────────────────────────────────────────────────

def plot_uc3_placement(results, save_dir):
    """
    Three separate figures:
      uc3_dilated.png    — jobs slowed by congestion
      uc3_locality.png   — global/local traffic ratio or hop count
      uc3_congestion.png — max congestion
    """
    if not PLOTTING_AVAILABLE:
        return
    save_dir = Path(save_dir)
    allocs = list(results.keys())
    colors = [ALLOCATION_COLORS.get(a, '#888') for a in allocs]
    labels = [a.title() for a in allocs]

    # (1) Dilated jobs
    fig, ax = plt.subplots(figsize=(_FIGW, _FIGH))
    _bar_single(ax, labels, [results[a].dilated_pct for a in allocs], colors,
                'Jobs slowed by congestion (%)')
    _save_fig(fig, save_dir / "uc3_dilated.png")

    # (2) Traffic locality
    fig, ax = plt.subplots(figsize=(_FIGW, _FIGH))
    ratios = [results[a].global_local_ratio for a in allocs]
    if any(r > 0 for r in ratios):
        _bar_single(ax, labels, ratios, colors, 'Global / local traffic ratio')
    else:
        _bar_single(ax, labels, [results[a].avg_hop_count for a in allocs], colors,
                    'Avg hop count')
    _save_fig(fig, save_dir / "uc3_locality.png")

    # (3) Max congestion
    fig, ax = plt.subplots(figsize=(_FIGW, _FIGH))
    _bar_single(ax, labels, [results[a].max_congestion for a in allocs], colors,
                'Max congestion')
    _save_fig(fig, save_dir / "uc3_congestion.png")


def plot_uc3_hop_count(results, save_path):
    """Hop count distribution histogram — one curve per allocation strategy."""
    if not PLOTTING_AVAILABLE:
        return
    fig, ax = plt.subplots(figsize=(_FIGW, _FIGH))
    for alloc, result in results.items():
        if result.hop_counts:
            ax.hist(result.hop_counts,
                    bins=range(1, max(result.hop_counts) + 2),
                    alpha=0.62,
                    color=ALLOCATION_COLORS.get(alloc, '#888'),
                    label=f'{alloc.title()} (μ={result.avg_hop_count:.1f})',
                    edgecolor='none', density=True)
    ax.set_xlabel('Hop count')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    _ygrid(ax)
    _save_fig(fig, save_path)


def plot_uc3_speedup_vs_comm(results, save_path):
    """Job slowdown vs communication intensity scatter plot."""
    if not PLOTTING_AVAILABLE:
        return
    fig, ax = plt.subplots(figsize=(_FIGW, _FIGH))
    for alloc, result in results.items():
        if result.job_comm_intensities and result.job_slowdowns:
            ax.scatter(result.job_comm_intensities, result.job_slowdowns,
                       alpha=0.55,
                       color=ALLOCATION_COLORS.get(alloc, '#888'),
                       s=16, label=alloc.title())
    if any(r.job_comm_intensities for r in results.values()):
        ax.set_xscale('log')
    ax.set_xlabel('Communication intensity (B/node/s)')
    ax.set_ylabel('Job slowdown factor')
    ax.legend(fontsize=8, loc='upper left')
    _ygrid(ax)
    _save_fig(fig, save_path)


# ── UC4 ──────────────────────────────────────────────────────────────────────

def plot_uc4_energy(results, save_dir):
    """
    Three separate figures:
      uc4_epj.png    — energy per completed job
      uc4_dilated.png— jobs slowed by network
      uc4_power.png  — static vs dynamic power (stacked bar)
    """
    if not PLOTTING_AVAILABLE:
        return
    save_dir = Path(save_dir)
    configs = list(results.keys())
    labels  = [c.replace('no_congestion', 'Ideal') for c in configs]
    pal     = ['#4DAF4A'] + ['#1B9E77', '#D95F02', '#7570B3'][:len(configs) - 1]
    colors  = pal[:len(configs)]

    # (1) Energy per completed job
    fig, ax = plt.subplots(figsize=(_FIGW, _FIGH))
    epj = [results[c].energy_per_completed_job for c in configs]
    _bar_single(ax, labels, epj, colors, 'Energy / job (MJ/job)')
    _save_fig(fig, save_dir / "uc4_epj.png")

    # (2) Dilated jobs
    fig, ax = plt.subplots(figsize=(_FIGW, _FIGH))
    dilated = [results[c].dilated_pct for c in configs]
    _bar_single(ax, labels, dilated, colors, 'Jobs slowed (%)')
    _save_fig(fig, save_dir / "uc4_dilated.png")

    # (3) Power decomposition (stacked bar, single panel)
    fig, ax = plt.subplots(figsize=(_FIGW, _FIGH))
    static_v  = [results[c].static_power_kw  for c in configs]
    dynamic_v = [results[c].dynamic_power_kw for c in configs]
    x = np.arange(len(configs))
    ax.bar(x, static_v,  color='#7570B3', alpha=0.85, label='Static (idle)',
           edgecolor='none', width=0.52)
    ax.bar(x, dynamic_v, bottom=static_v, color='#D95F02', alpha=0.85,
           label='Dynamic (compute)', edgecolor='none', width=0.52)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('Avg power (kW)')
    ax.legend(fontsize=8, loc='upper right')
    _ygrid(ax)
    _save_fig(fig, save_dir / "uc4_power.png")


# ==========================================
# CSV-based plot regeneration
# ==========================================

def _load_uc_results_from_csv(csv_path, key_col):
    """Load UCResult objects from a saved CSV. Only aggregate fields are populated."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    results = {}
    for _, row in df.iterrows():
        key = row[key_col]
        r = SimResult(
            label=str(row.get('label', key)),
            system=str(row.get('system', '')),
            routing=str(row.get('routing', 'default')),
            allocation=str(row.get('allocation', 'contiguous')),
            policy=str(row.get('policy', 'fcfs')),
            num_jobs=int(row.get('num_jobs', 0)),
            simulated_seconds=float(row.get('simulated_seconds', 0)),
            wall_time=float(row.get('wall_time', 1)),
            speedup=float(row.get('speedup', 1)),
            ticks=int(row.get('ticks', 0)),
            avg_network_util=float(row.get('avg_network_util', 0)),
            avg_job_slowdown=float(row.get('avg_job_slowdown', 1)),
            max_job_slowdown=float(row.get('max_job_slowdown', 1)),
            avg_congestion=float(row.get('avg_congestion', 0)),
            max_congestion=float(row.get('max_congestion', 0)),
            global_throughput_bps=float(row.get('global_throughput_bps', 0)),
            jobs_completed=int(row.get('jobs_completed', 0)),
            jobs_dilated=int(row.get('jobs_dilated', 0)),
            dilated_pct=float(row.get('dilated_pct', 0)),
            avg_wait_time=float(row.get('avg_wait_time', 0)),
            max_wait_time=float(row.get('max_wait_time', 0)),
            potential_bullies=int(row.get('potential_bullies', 0)),
            total_energy_joules=float(row.get('total_energy_joules', 0)),
            energy_per_completed_job=float(row.get('energy_per_completed_job', 0)),
            avg_power_watts=float(row.get('avg_power_watts', 0)),
            peak_power_watts=float(row.get('peak_power_watts', 0)),
            idle_energy_pct=float(row.get('idle_energy_pct', 0)),
            static_power_kw=float(row.get('static_power_kw', 0)),
            dynamic_power_kw=float(row.get('dynamic_power_kw', 0)),
            static_power_pct=float(row.get('static_power_pct', 0)),
            avg_hop_count=float(row.get('avg_hop_count', 0)),
            global_local_ratio=float(row.get('global_local_ratio', 0)),
            global_traffic_pairs=int(row.get('global_traffic_pairs', 0)),
            local_traffic_pairs=int(row.get('local_traffic_pairs', 0)),
        )
        results[key] = r
    return results


def regenerate_plots_from_csv(output_dir, figures_dir, uc_to_run):
    """Regenerate UC plots that only need aggregate CSV data (no per-job lists)."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    if 3 in uc_to_run:
        csv3 = output_dir / "uc3_placement_results.csv"
        if csv3.exists():
            results = _load_uc_results_from_csv(csv3, 'allocation')
            plot_uc3_placement(results, figures_dir)
            print(f"  [CSV] Regenerated uc3_*.png")
        else:
            print(f"  [SKIP] {csv3} not found")
    if 4 in uc_to_run:
        csv4 = output_dir / "uc4_energy_results.csv"
        if csv4.exists():
            results_raw = _load_uc_results_from_csv(csv4, 'label')
            results = {k.replace('UC4_', '', 1): v for k, v in results_raw.items()}
            plot_uc4_energy(results, figures_dir)
            print(f"  [CSV] Regenerated uc4_*.png")
        else:
            print(f"  [SKIP] {csv4} not found")


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description='RAPS Use Case Evaluation (4 operational scenarios)')
    parser.add_argument('--system', default='lassen', choices=['lassen', 'frontier'],
                        help='System to simulate (default: lassen)')
    parser.add_argument('--duration', type=int, default=60,
                        help='Simulation duration in minutes (default: 60)')
    parser.add_argument('--delta-t', type=int, default=1,
                        help='Time step in seconds (default: 1)')
    parser.add_argument('--num-jobs', type=int, default=300,
                        help='Number of jobs (default: 300)')
    parser.add_argument('--nodes', type=int, default=None,
                        help='Override system node count (e.g. 10000)')
    parser.add_argument('--uc', type=int, nargs='+', default=None,
                        help='Specific use cases to run (1-4, default: all)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 5 min, 10 jobs')
    parser.add_argument('--arrival-rate', type=float, default=None,
                        help='Mean seconds between job arrivals (default: auto-scale to fill ~80%% of simulation)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--plot-from-csv', action='store_true',
                        help='Regenerate UC3/UC4 plots from existing CSVs (no simulation)')
    parser.add_argument('--force', action='store_true',
                        help='Force re-run even if result CSVs already exist')

    args = parser.parse_args()

    # Output directory - include system and node count in path
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        suffix = f"{args.system}"
        if args.nodes:
            suffix += f"_n{args.nodes}"
        output_dir = PROJECT_ROOT / "output" / "use_cases" / suffix
    figures_dir = output_dir / "figures"

    # --plot-from-csv: skip simulation, regenerate UC3/UC4 plots from existing CSVs
    if args.plot_from_csv:
        uc_to_run = set(args.uc) if args.uc else {3, 4}
        print(f"[plot-from-csv] Regenerating plots for UCs {sorted(uc_to_run)} in {output_dir}")
        regenerate_plots_from_csv(output_dir, figures_dir, uc_to_run)
        return

    # Quick mode overrides
    if args.quick:
        args.duration = 5
        args.num_jobs = 10

    sim_seconds = args.duration * 60

    if args.arrival_rate is not None:
        mean_inter_arrival = args.arrival_rate
    else:
        mean_inter_arrival = sim_seconds * 0.8 / args.num_jobs
        mean_inter_arrival = max(2.0, mean_inter_arrival)
    print(f"[Config] Arrival rate: {mean_inter_arrival:.1f}s "
          f"({args.num_jobs} jobs over {args.duration} min, "
          f"~{int(sim_seconds * 0.8 / mean_inter_arrival)} arrive within window)")

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Which use cases to run
    uc_to_run = set(args.uc) if args.uc else {1, 2, 3, 4}

    # Scale workload for larger systems
    node_count = args.nodes
    if node_count and node_count >= 1000:
        # Scale max job size proportionally (~2.5% of system)
        max_job_nodes = min(node_count // 4, 4096)
        min_job_nodes = max(8, node_count // 500)
    else:
        max_job_nodes = 256
        min_job_nodes = 8

    print("=" * 60)
    print("RAPS Use Case Evaluation")
    print("=" * 60)
    print(f"System:    {args.system}")
    if node_count:
        print(f"Nodes:     {node_count}")
    print(f"Duration:  {args.duration} min")
    print(f"Delta-t:   {args.delta_t}s")
    print(f"Jobs:      {args.num_jobs}")
    print(f"Job sizes: {min_job_nodes} - {max_job_nodes} nodes")
    print(f"Use cases: {sorted(uc_to_run)}")
    print(f"Output:    {output_dir}")

    # Generate workload
    print("\n[Setup] Generating workload...")
    jobs = generate_workload(
        num_jobs=args.num_jobs,
        min_nodes=min_job_nodes,
        max_nodes=max_job_nodes,
        mean_inter_arrival=mean_inter_arrival,
    )
    print(f"  Generated {len(jobs)} jobs")
    sizes = [j.nodes_required for j in jobs]
    print(f"  Node sizes: min={min(sizes)}, max={max(sizes)}, "
          f"median={int(np.median(sizes))}")
    print(f"  Arrivals: {jobs[0].submit_time}s to {jobs[-1].submit_time}s")

    # Common kwargs for all UC functions
    uc_kwargs = dict(
        node_count=node_count,
        delta_t=args.delta_t,
    )

    all_results = {}

    # Helper: check if a UC result CSV already exists with expected variants
    def _uc_complete(csv_name, expected_variants):
        """Return True if the CSV has rows for all expected UNIQUE variant labels."""
        if args.force:
            return False
        csv_path = output_dir / csv_name
        if not csv_path.exists():
            return False
        try:
            df = pd.read_csv(csv_path)
            if 'label' not in df.columns:
                return len(df) >= expected_variants
            return df['label'].nunique() >= expected_variants
        except Exception:
            return False

    # UC1: Routing (3 variants per topology: minimal/ugal/valiant or minimal/ecmp/adaptive)
    num_routing_algos = len(_all_routing_algos_for_system(args.system))
    uc1_csv = "uc1_routing_results.csv"
    if 1 in uc_to_run:
        if _uc_complete(uc1_csv, num_routing_algos):
            print(f"\n[SKIP] UC1 already complete ({uc1_csv} has {num_routing_algos} variants)")
        else:
            uc1_path = output_dir / uc1_csv
            resume_path = None if args.force else uc1_path
            done = _load_done_labels(resume_path)
            if done:
                print(f"\n[RESUME] UC1: {len(done)} variant(s) already done, resuming...")
            results = run_uc1_routing(
                jobs, args.system, args.duration,
                resume_csv=uc1_path, done_labels=done, **uc_kwargs)
            all_results['uc1'] = results

            if results and not args.no_plots:
                plot_uc1_slowdown_cdf(results, figures_dir / "uc1_slowdown_cdf.png")
                plot_uc1_congestion_heatmap(results, figures_dir)

    # UC2: Scheduling (3 variants: fcfs, fcfs+firstfit, sjf)
    uc2_csv = "uc2_scheduling_results.csv"
    if 2 in uc_to_run:
        if _uc_complete(uc2_csv, 3):
            print(f"\n[SKIP] UC2 already complete ({uc2_csv} has 3 variants)")
        else:
            uc2_path = output_dir / uc2_csv
            resume_path = None if args.force else uc2_path
            done = _load_done_labels(resume_path)
            if done:
                print(f"\n[RESUME] UC2: {len(done)} variant(s) already done, resuming...")
            results = run_uc2_scheduling(
                jobs, args.system, args.duration,
                resume_csv=uc2_path, done_labels=done, **uc_kwargs)
            all_results['uc2'] = results

            if results and not args.no_plots:
                plot_uc2_wait_times(results, figures_dir)
                plot_uc2_utilization_stacked(results, figures_dir)
                plot_uc2_prediction_error(results, figures_dir)

    # UC3: Node Placement (3 variants: contiguous, random, hybrid)
    uc3_csv = "uc3_placement_results.csv"
    if 3 in uc_to_run:
        if _uc_complete(uc3_csv, 3):
            print(f"\n[SKIP] UC3 already complete ({uc3_csv} has 3 variants)")
        else:
            uc3_path = output_dir / uc3_csv
            resume_path = None if args.force else uc3_path
            done = _load_done_labels(resume_path)
            if done:
                print(f"\n[RESUME] UC3: {len(done)} variant(s) already done, resuming...")
            results = run_uc3_placement(
                jobs, args.system, args.duration,
                resume_csv=uc3_path, done_labels=done, **uc_kwargs)
            all_results['uc3'] = results

            if results and not args.no_plots:
                plot_uc3_placement(results, figures_dir)
                plot_uc3_speedup_vs_comm(results, figures_dir / "uc3_speedup_vs_comm.png")

    # UC4: Energy Cost (no_congestion + each routing algo).
    # For fat-tree systems, adaptive routing ≈ ecmp so it is excluded from UC4
    # (see run_uc4_energy), giving 1 + num_routing_algos - 1 = num_routing_algos variants.
    if _get_topology(args.system) != 'dragonfly':
        num_uc4_variants = num_routing_algos  # no_congestion + (all algos minus adaptive)
    else:
        num_uc4_variants = 1 + num_routing_algos  # no_congestion + all algos
    uc4_csv = "uc4_energy_results.csv"
    if 4 in uc_to_run:
        if _uc_complete(uc4_csv, num_uc4_variants):
            print(f"\n[SKIP] UC4 already complete ({uc4_csv} has {num_uc4_variants} variants)")
        else:
            uc4_path = output_dir / uc4_csv
            resume_path = None if args.force else uc4_path
            done = _load_done_labels(resume_path)
            if done:
                print(f"\n[RESUME] UC4: {len(done)} variant(s) already done, resuming...")
            results = run_uc4_energy(
                jobs, args.system, args.duration,
                resume_csv=uc4_path, done_labels=done, **uc_kwargs)
            all_results['uc4'] = results

            if results and not args.no_plots:
                plot_uc4_energy(results, figures_dir)

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

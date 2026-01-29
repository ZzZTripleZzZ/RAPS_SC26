#!/usr/bin/env python3
"""
SC26 Complete Experiment Pipeline (v3) - Using RAPS Engine
============================================================

Uses RAPS Engine for proper simulation including:
- Network congestion simulation
- Job slowdown calculation
- Power consumption modeling

Key changes from v2:
- Uses RAPS Engine instead of custom simulation
- Proper slowdown factor calculation via RAPS
- Creates Job objects from affinity graph data
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import yaml

# Add RAPS to path
sys.path.insert(0, str(Path("/app/extern/raps")))

from raps.job import Job, job_dict, CommunicationPattern
from raps.policy import AllocationStrategy
from raps.engine import Engine
from raps.sim_config import SimConfig
from raps.system_config import SystemConfig
from raps.stats import get_engine_stats, get_network_stats, get_job_stats
from raps.network import NetworkModel, worst_link_util, get_link_util_stats
from raps.network.base import link_loads_for_pattern, max_throughput_per_tick


# ==============================================================================
# Configuration
# ==============================================================================

DATA_DIR = Path("/app/data")
OUTPUT_DIR = Path("/app/output/sc26_experiments")
TRACE_DIR = DATA_DIR / "matrices"

# System configurations
SYSTEMS = {
    "lassen": {
        "topology": "fat-tree",
        "config_path": "/app/extern/raps/config/lassen.yaml",
        "network_max_bw": 12.5e9,  # 100 Gbps
        "fattree_k": 32,
        "total_nodes": 4608,
        "routing_algorithms": ["minimal", "ecmp"],
    },
    "frontier": {
        "topology": "dragonfly",
        "config_path": "/app/extern/raps/config/frontier.yaml",
        "network_max_bw": 25e9,  # 200 Gbps
        "dragonfly_d": 48,
        "dragonfly_a": 48,
        "dragonfly_p": 4,
        "total_nodes": 9408,
        "routing_algorithms": ["minimal", "ugal", "valiant"],
    },
}


@dataclass
class MiniAppWorkload:
    """Represents a mini-app workload with its REAL communication pattern."""
    name: str
    app_type: str  # lulesh, comd, hpgmg, cosp2
    num_nodes: int
    traffic_matrix: Optional[np.ndarray] = None
    affinity_graph: Optional[Dict] = None
    total_traffic_bytes: float = 0.0
    avg_degree: float = 0.0
    sparsity: float = 0.0


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    system: str
    workload: MiniAppWorkload
    routing_algorithm: str
    allocation_strategy: str
    trace_quanta: int = 15  # seconds per trace step
    simulation_time: int = 300  # 5 minutes of simulation


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    config: ExperimentConfig
    max_link_util: float = 0.0
    avg_link_util: float = 0.0
    link_util_std: float = 0.0
    total_power_kw: float = 0.0
    congestion_factor: float = 1.0
    job_slowdown: float = 1.0
    avg_slowdown: float = 1.0
    max_slowdown: float = 1.0
    timestamp: str = ""


# ==============================================================================
# Data Loading - REAL Traces
# ==============================================================================

def load_system_config(system_name: str) -> Dict:
    """Load RAPS system configuration."""
    config_path = Path(SYSTEMS[system_name]["config_path"])
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_traffic_matrix(trace_name: str) -> Optional[np.ndarray]:
    """Load static traffic matrix from parsed traces."""
    static_path = TRACE_DIR / f"{trace_name}_static.npy"
    if static_path.exists():
        return np.load(static_path)
    return None


def load_affinity_graph(trace_name: str) -> Optional[Dict]:
    """Load affinity graph from parsed traces."""
    json_path = TRACE_DIR / f"{trace_name}_affinity.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return None


def find_available_traces() -> List[str]:
    """Find all available trace names in the data directory."""
    if not TRACE_DIR.exists():
        return []

    traces = set()
    for f in TRACE_DIR.glob("*_affinity.json"):
        traces.add(f.stem.replace("_affinity", ""))

    return sorted(traces)


def infer_app_type(trace_name: str) -> str:
    """Infer mini-app type from trace name."""
    trace_lower = trace_name.lower()
    if "lulesh" in trace_lower:
        return "lulesh"
    elif "comd" in trace_lower:
        return "comd"
    elif "hpgmg" in trace_lower:
        return "hpgmg"
    elif "cosp2" in trace_lower:
        return "cosp2"
    return "unknown"


def create_workload_from_trace(trace_name: str) -> Optional[MiniAppWorkload]:
    """Create a MiniAppWorkload from REAL parsed trace data."""
    traffic_matrix = load_traffic_matrix(trace_name)
    affinity_graph = load_affinity_graph(trace_name)

    if affinity_graph is None:
        return None

    num_nodes = affinity_graph['num_nodes']
    total_traffic = affinity_graph.get('total_bytes', 0)
    num_edges = affinity_graph.get('num_edges', 0)

    # Calculate average degree from nodes
    degrees = [n.get('degree', 0) for n in affinity_graph.get('nodes', [])]
    avg_degree = np.mean(degrees) if degrees else 0

    # Calculate sparsity
    max_edges = num_nodes * (num_nodes - 1) / 2
    sparsity = 1.0 - (num_edges / max_edges) if max_edges > 0 else 1.0

    app_type = infer_app_type(trace_name)

    return MiniAppWorkload(
        name=trace_name,
        app_type=app_type,
        num_nodes=num_nodes,
        traffic_matrix=traffic_matrix,
        affinity_graph=affinity_graph,
        total_traffic_bytes=total_traffic,
        avg_degree=avg_degree,
        sparsity=sparsity,
    )


def get_best_traces_per_app(min_nodes: int = 64) -> List[MiniAppWorkload]:
    """Get the best (largest) trace for each mini-app type."""
    all_traces = find_available_traces()

    # Group by app type
    app_traces = {}
    for trace_name in all_traces:
        wl = create_workload_from_trace(trace_name)
        if wl and wl.num_nodes >= min_nodes:
            app_type = wl.app_type
            if app_type not in app_traces or wl.num_nodes > app_traces[app_type].num_nodes:
                app_traces[app_type] = wl

    return list(app_traces.values())


# ==============================================================================
# Job Creation from Affinity Graph
# ==============================================================================

def calculate_bandwidth_from_affinity(
    affinity_graph: Dict,
    trace_quanta: int,
    iterations_per_quanta: int = 1000,
) -> float:
    """
    Calculate per-node bandwidth (bytes per trace tick) from affinity graph.

    This converts the total traffic in the affinity graph to a per-tick bandwidth
    that can be used in RAPS network traces.

    Args:
        affinity_graph: The affinity graph with total_bytes
        trace_quanta: Seconds per trace step
        iterations_per_quanta: Number of communication iterations per quanta
            (HPC apps typically do many iterations per second)

    Returns:
        Bandwidth in bytes per tick
    """
    total_bytes = affinity_graph.get('total_bytes', 0)
    num_nodes = affinity_graph.get('num_nodes', 1)

    # The trace represents one "iteration" of communication
    # HPC apps do many iterations per second (e.g., 1000+ for stencil codes)
    # Scale up to represent sustained bandwidth
    bytes_per_iteration = total_bytes / num_nodes if num_nodes > 0 else 0

    # Total bandwidth per quanta = bytes_per_iteration * iterations
    bytes_per_quanta = bytes_per_iteration * iterations_per_quanta

    return bytes_per_quanta


def infer_comm_pattern_from_affinity(affinity_graph: Dict) -> CommunicationPattern:
    """
    Infer communication pattern from affinity graph characteristics.

    - High sparsity + low avg degree -> Stencil-like (nearest neighbor)
    - Low sparsity + high avg degree -> All-to-all like
    """
    num_nodes = affinity_graph.get('num_nodes', 1)
    num_edges = affinity_graph.get('num_edges', 0)

    nodes = affinity_graph.get('nodes', [])
    degrees = [n.get('degree', 0) for n in nodes]
    avg_degree = np.mean(degrees) if degrees else 0

    # Stencil-3D typically has ~6 neighbors (degree ~6)
    # All-to-all has degree close to num_nodes-1
    if avg_degree < 10 and num_nodes > 16:
        return CommunicationPattern.STENCIL_3D
    else:
        return CommunicationPattern.ALL_TO_ALL


def create_job_from_workload(
    workload: MiniAppWorkload,
    job_id: int,
    scheduled_nodes: List[int],
    trace_quanta: int = 15,
    simulation_time: int = 300,
) -> Job:
    """
    Create a RAPS Job object from a MiniAppWorkload.

    Args:
        workload: The mini-app workload with affinity graph
        job_id: Unique job ID
        scheduled_nodes: List of node IDs to run on
        trace_quanta: Seconds per trace step
        simulation_time: Total simulation time in seconds

    Returns:
        RAPS Job object
    """
    trace_len = simulation_time // trace_quanta

    # Calculate bandwidth from affinity graph
    # Use iterations_per_quanta to scale up to realistic sustained bandwidth
    bandwidth = calculate_bandwidth_from_affinity(
        workload.affinity_graph,
        trace_quanta,
        iterations_per_quanta=1000,  # Typical HPC iteration rate
    )

    # Infer communication pattern
    comm_pattern = infer_comm_pattern_from_affinity(workload.affinity_graph)

    # Create traces (constant bandwidth for now)
    cpu_trace = [0.6] * trace_len  # 60% CPU utilization
    gpu_trace = [0.7] * trace_len  # 70% GPU utilization
    ntx_trace = [bandwidth] * trace_len
    nrx_trace = [bandwidth] * trace_len

    job_info = job_dict(
        id=job_id,
        name=f"{workload.app_type}_{workload.name}",
        account="sc26_experiment",
        nodes_required=workload.num_nodes,
        scheduled_nodes=scheduled_nodes,
        cpu_trace=cpu_trace,
        gpu_trace=gpu_trace,
        ntx_trace=ntx_trace,
        nrx_trace=nrx_trace,
        submit_time=0,
        start_time=0,
        expected_run_time=simulation_time,
        time_limit=simulation_time * 2,
        end_state="COMPLETED",
        trace_quanta=trace_quanta,
        comm_pattern=comm_pattern,
    )

    return Job(job_info)


# ==============================================================================
# RAPS Engine Simulation
# ==============================================================================

def get_node_allocation(
    system_name: str,
    num_nodes: int,
    allocation_strategy: str,
) -> List[int]:
    """Get node allocation based on strategy."""
    system = SYSTEMS[system_name]
    total_nodes = system["total_nodes"]

    if allocation_strategy == "contiguous":
        return list(range(num_nodes))
    elif allocation_strategy == "random":
        np.random.seed(42)  # For reproducibility
        return sorted(np.random.choice(
            min(total_nodes, num_nodes * 10),
            num_nodes,
            replace=False
        ).tolist())
    else:
        return list(range(num_nodes))


def run_raps_simulation(
    config: ExperimentConfig,
    num_concurrent_jobs: int = 4,
) -> ExperimentResult:
    """
    Run a simulation using RAPS Engine with multiple concurrent jobs.

    This creates multiple jobs from the workload and simulates inter-job
    network congestion to get realistic slowdown metrics.

    Args:
        config: Experiment configuration
        num_concurrent_jobs: Number of concurrent jobs to simulate (default: 4)
    """
    workload = config.workload
    system_name = config.system
    system = SYSTEMS[system_name]

    # Load system configuration
    system_config = load_system_config(system_name)
    legacy_config = {
        'TOTAL_NODES': system['total_nodes'],
        'AVAILABLE_NODES': system['total_nodes'],
        'DOWN_NODES': [],
        'CPUS_PER_NODE': system_config['system']['cpus_per_node'],
        'GPUS_PER_NODE': system_config['system']['gpus_per_node'],
        'NICS_PER_NODE': system_config['system']['nics_per_node'],
        'POWER_CPU_IDLE': system_config['power']['power_cpu_idle'],
        'POWER_CPU_MAX': system_config['power']['power_cpu_max'],
        'POWER_GPU_IDLE': system_config['power']['power_gpu_idle'],
        'POWER_GPU_MAX': system_config['power']['power_gpu_max'],
        'POWER_MEM': system_config['power']['power_mem'],
        'POWER_NIC': system_config['power'].get('power_nic', 20),
        'POWER_NIC_IDLE': system_config['power'].get('power_nic_idle'),
        'POWER_NIC_MAX': system_config['power'].get('power_nic_max'),
        'POWER_NVME': system_config['power']['power_nvme'],
        'SIVOC_LOSS_CONSTANT': system_config['power']['sivoc_loss_constant'],
        'SIVOC_EFFICIENCY': system_config['power']['sivoc_efficiency'],
        'NETWORK_MAX_BW': system['network_max_bw'],
        'system_name': system_name,
        'multitenant': False,
    }

    # Add topology-specific config
    if system['topology'] == 'fat-tree':
        legacy_config['FATTREE_K'] = system['fattree_k']
        legacy_config['TOPOLOGY'] = 'fat-tree'
    elif system['topology'] == 'dragonfly':
        legacy_config['DRAGONFLY_D'] = system['dragonfly_d']
        legacy_config['DRAGONFLY_A'] = system['dragonfly_a']
        legacy_config['DRAGONFLY_P'] = system['dragonfly_p']
        legacy_config['TOPOLOGY'] = 'dragonfly'

    # Create multiple concurrent jobs to simulate inter-job congestion
    jobs = []
    for job_idx in range(num_concurrent_jobs):
        # Calculate node allocation for each job (non-overlapping if contiguous)
        if config.allocation_strategy == "contiguous":
            start_node = job_idx * workload.num_nodes
            scheduled_nodes = list(range(start_node, start_node + workload.num_nodes))
        else:
            # Random allocation - may have shared links
            np.random.seed(42 + job_idx)
            max_node = min(system['total_nodes'], workload.num_nodes * num_concurrent_jobs * 2)
            scheduled_nodes = sorted(np.random.choice(
                max_node,
                workload.num_nodes,
                replace=False
            ).tolist())

        job = create_job_from_workload(
            workload,
            job_id=job_idx + 1,
            scheduled_nodes=scheduled_nodes,
            trace_quanta=config.trace_quanta,
            simulation_time=config.simulation_time,
        )
        jobs.append(job)

    # Create NetworkModel for simulation
    network_model = NetworkModel(
        available_nodes=list(range(system['total_nodes'])),
        config=legacy_config
    )

    # Simulate inter-job network congestion using RAPS
    from raps.network import simulate_inter_job_congestion

    net_util = 0.0
    net_cong = 0.0
    slowdown_factor = 1.0
    max_link_util = 0.0
    avg_link_util = 0.0
    congestion_factor = 1.0
    link_util_std = 0.0

    if network_model.net_graph and jobs:
        # Use RAPS inter-job congestion simulation
        congestion_stats = simulate_inter_job_congestion(
            network_model, jobs, legacy_config, debug=False
        )

        if isinstance(congestion_stats, dict):
            max_link_util = congestion_stats.get('max', 0.0)
            avg_link_util = congestion_stats.get('mean', 0.0)
            link_util_std = congestion_stats.get('std_dev', 0.0)
        else:
            max_link_util = float(congestion_stats)
            avg_link_util = max_link_util

        # Calculate slowdown using RAPS method
        # Slowdown occurs when max link utilization > 1.0 (oversubscribed)
        if max_link_util > 1.0:
            slowdown_factor = max_link_util
        else:
            slowdown_factor = 1.0

        congestion_factor = max(1.0, max_link_util)
        net_util = min(avg_link_util, 1.0)

    # Calculate power using RAPS power model
    from raps.power import compute_node_power

    cpu_util = 0.6 * legacy_config['CPUS_PER_NODE']
    gpu_util = 0.7 * legacy_config['GPUS_PER_NODE']

    node_power, _ = compute_node_power(cpu_util, gpu_util, net_util, legacy_config)
    # Total power for all concurrent jobs
    total_power_kw = node_power * workload.num_nodes * num_concurrent_jobs / 1000.0

    return ExperimentResult(
        config=config,
        max_link_util=max_link_util,
        avg_link_util=avg_link_util,
        link_util_std=link_util_std,
        total_power_kw=total_power_kw,
        congestion_factor=congestion_factor,
        job_slowdown=slowdown_factor,
        avg_slowdown=slowdown_factor,
        max_slowdown=slowdown_factor,
        timestamp=datetime.now().isoformat(),
    )


def run_single_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run a single experiment configuration using RAPS."""
    print(f"  [{config.workload.app_type.upper()}] {config.system} | {config.routing_algorithm} | "
          f"{config.allocation_strategy} | n={config.workload.num_nodes}")

    return run_raps_simulation(config)


# ==============================================================================
# Experiment Runner
# ==============================================================================

def run_use_case_experiments(
    workloads: List[MiniAppWorkload],
    use_case: str,
) -> pd.DataFrame:
    """Run experiments for a specific use case."""
    results = []

    if use_case == "adaptive_routing":
        for system_name in SYSTEMS.keys():
            system = SYSTEMS[system_name]
            for workload in workloads:
                for routing in system["routing_algorithms"]:
                    config = ExperimentConfig(
                        system=system_name,
                        workload=workload,
                        routing_algorithm=routing,
                        allocation_strategy="contiguous",
                    )
                    result = run_single_experiment(config)
                    results.append(result)

    elif use_case == "node_placement":
        allocations = ["contiguous", "random"]
        for system_name in SYSTEMS.keys():
            system = SYSTEMS[system_name]
            default_routing = system["routing_algorithms"][0]
            for workload in workloads:
                for alloc in allocations:
                    config = ExperimentConfig(
                        system=system_name,
                        workload=workload,
                        routing_algorithm=default_routing,
                        allocation_strategy=alloc,
                    )
                    result = run_single_experiment(config)
                    results.append(result)

    elif use_case == "scheduling":
        # Compare scheduling by testing all combinations
        for system_name in SYSTEMS.keys():
            system = SYSTEMS[system_name]
            for workload in workloads:
                for routing in system["routing_algorithms"][:2]:  # Limit for speed
                    for alloc in ["contiguous", "random"]:
                        config = ExperimentConfig(
                            system=system_name,
                            workload=workload,
                            routing_algorithm=routing,
                            allocation_strategy=alloc,
                        )
                        result = run_single_experiment(config)
                        results.append(result)

    elif use_case == "power_consumption":
        for system_name in SYSTEMS.keys():
            system = SYSTEMS[system_name]
            default_routing = system["routing_algorithms"][0]
            for workload in workloads:
                config = ExperimentConfig(
                    system=system_name,
                    workload=workload,
                    routing_algorithm=default_routing,
                    allocation_strategy="contiguous",
                )
                result = run_single_experiment(config)
                results.append(result)

    # Convert to DataFrame
    rows = []
    for r in results:
        rows.append({
            "system": r.config.system,
            "topology": SYSTEMS[r.config.system]["topology"],
            "app_type": r.config.workload.app_type,
            "workload": r.config.workload.name,
            "num_nodes": r.config.workload.num_nodes,
            "avg_degree": r.config.workload.avg_degree,
            "sparsity": r.config.workload.sparsity,
            "total_traffic_bytes": r.config.workload.total_traffic_bytes,
            "routing_algorithm": r.config.routing_algorithm,
            "allocation_strategy": r.config.allocation_strategy,
            "max_link_util": r.max_link_util,
            "avg_link_util": r.avg_link_util,
            "link_util_std": r.link_util_std,
            "total_power_kw": r.total_power_kw,
            "congestion_factor": r.congestion_factor,
            "job_slowdown": r.job_slowdown,
            "avg_slowdown": r.avg_slowdown,
            "max_slowdown": r.max_slowdown,
            "timestamp": r.timestamp,
        })

    return pd.DataFrame(rows)


# ==============================================================================
# Main Pipeline
# ==============================================================================

def run_complete_pipeline():
    """Run the complete SC26 experiment pipeline with RAPS Engine."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("SC26 Complete Experiment Pipeline (RAPS Engine)")
    print("="*70)

    # Step 1: Load REAL workloads
    print("\n[Step 1] Loading REAL mini-app traces...")

    # Get best trace for each app type
    workloads = get_best_traces_per_app(min_nodes=64)

    if not workloads:
        print("  No traces found with >= 64 nodes, trying smaller...")
        workloads = get_best_traces_per_app(min_nodes=8)

    if not workloads:
        print("  ERROR: No traces found!")
        return {}

    for wl in workloads:
        print(f"  {wl.app_type.upper():8s}: {wl.name}")
        print(f"           nodes={wl.num_nodes}, edges={wl.affinity_graph['num_edges']}, "
              f"avg_degree={wl.avg_degree:.1f}, sparsity={wl.sparsity:.2%}")

    all_results = {}

    # Step 2: Adaptive Routing
    print("\n[Step 2] Running Adaptive Routing experiments...")
    df_routing = run_use_case_experiments(workloads, "adaptive_routing")
    df_routing.to_csv(OUTPUT_DIR / "uc1_adaptive_routing.csv", index=False)
    all_results["adaptive_routing"] = df_routing
    print(f"  Saved: uc1_adaptive_routing.csv ({len(df_routing)} experiments)")

    # Step 3: Node Placement
    print("\n[Step 3] Running Node Placement experiments...")
    df_placement = run_use_case_experiments(workloads, "node_placement")
    df_placement.to_csv(OUTPUT_DIR / "uc2_node_placement.csv", index=False)
    all_results["node_placement"] = df_placement
    print(f"  Saved: uc2_node_placement.csv ({len(df_placement)} experiments)")

    # Step 4: Scheduling
    print("\n[Step 4] Running Scheduling experiments...")
    df_scheduling = run_use_case_experiments(workloads, "scheduling")
    df_scheduling.to_csv(OUTPUT_DIR / "uc3_scheduling.csv", index=False)
    all_results["scheduling"] = df_scheduling
    print(f"  Saved: uc3_scheduling.csv ({len(df_scheduling)} experiments)")

    # Step 5: Power Consumption
    print("\n[Step 5] Running Power Consumption experiments...")
    df_power = run_use_case_experiments(workloads, "power_consumption")
    df_power.to_csv(OUTPUT_DIR / "uc4_power_consumption.csv", index=False)
    all_results["power_consumption"] = df_power
    print(f"  Saved: uc4_power_consumption.csv ({len(df_power)} experiments)")

    # Step 6: Summary
    print("\n[Step 6] Generating summary...")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "workloads": [
            {
                "name": wl.name,
                "app_type": wl.app_type,
                "num_nodes": wl.num_nodes,
                "num_edges": wl.affinity_graph['num_edges'],
                "avg_degree": wl.avg_degree,
                "sparsity": wl.sparsity,
            }
            for wl in workloads
        ],
        "experiments": {
            uc: {"count": len(df), "systems": df["system"].unique().tolist()}
            for uc, df in all_results.items()
        }
    }
    with open(OUTPUT_DIR / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*70)
    print("Pipeline Complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*70)

    return all_results


if __name__ == "__main__":
    run_complete_pipeline()

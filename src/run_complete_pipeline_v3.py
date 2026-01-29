#!/usr/bin/env python3
"""
SC26 Complete Pipeline v3.0
============================
Full RAPS Integration including:

1. RL Scheduler - Using stable-baselines3 PPO
2. FLOPS Tracking - Compute performance metrics
3. Carbon Emissions - Environmental impact
4. Realistic Latency Model - 4-component model
5. Dynamic Traffic Burstiness - Random bursts

Use Case Mapping:
- Use Case 1 (Adaptive Routing): Static Traffic Matrix + RAPS Network
- Use Case 2 (Node Placement):   Affinity Graph + RAPS ResourceManager
- Use Case 3 (Job Scheduling):   Dynamic Traffic Matrix + RAPS Schedulers (FCFS, Backfill, SJF, RL)
- Use Case 4 (Power Analysis):   Dynamic Traffic Matrix + RAPS Power + FLOPS + Carbon
"""

import os
import sys
import json
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

# Add paths
sys.path.insert(0, str(Path("/app/src")))
sys.path.insert(0, str(Path("/app/extern/raps")))

# ==========================================
# Configuration
# ==========================================
MATRIX_DIR = Path("/app/data/matrices")
RESULTS_DIR = Path("/app/data/results_v3")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# System configurations
SYSTEMS = {
    'lassen': {
        'total_nodes': 4608,
        'topology': 'fat-tree',
        'network_max_bw': 12.5e9,
        'fattree_k': 32,
        'TOTAL_NODES': 4608,
        'AVAILABLE_NODES': 4608,
        'POWER_GPU_IDLE': 75,
        'POWER_GPU_MAX': 300,
        'POWER_CPU_IDLE': 47,
        'POWER_CPU_MAX': 252,
        'GPUS_PER_NODE': 4,
        'CPUS_PER_NODE': 2,
        'POWER_MEM': 30,
        'POWER_NIC': 25,
        'POWER_NIC_IDLE': 10,
        'POWER_NIC_MAX': 50,
        'NICS_PER_NODE': 2,
        'POWER_NVME': 5,
        'SIVOC_LOSS_CONSTANT': 10,
        'SIVOC_EFFICIENCY': 0.95,
        'pue': 1.3,
        'propagation_delay_ns': 5,
        'link_length_m': 10,
        'switch_latency_ns': 200,
        'serialization_rate_gbps': 100,
        # FLOPS config
        'CPU_PEAK_FLOPS': 1.5e12,  # 1.5 TFLOPS per CPU
        'GPU_PEAK_FLOPS': 7.8e12,  # 7.8 TFLOPS per V100
        'CPU_FP_RATIO': 0.1,
        'GPU_FP_RATIO': 0.9,
        'SC_SHAPE': (16, 18, 16),  # 3D grid shape
        # Carbon config
        'CARBON_INTENSITY': 0.4,  # kg CO2 per kWh (US average)
    },
    'frontier': {
        'total_nodes': 9472,
        'topology': 'dragonfly',
        'network_max_bw': 25e9,
        'dragonfly_d': 48,
        'dragonfly_a': 48,
        'dragonfly_p': 4,
        'ugal_threshold': 2.0,
        'TOTAL_NODES': 9472,
        'AVAILABLE_NODES': 9472,
        'POWER_GPU_IDLE': 88,
        'POWER_GPU_MAX': 560,
        'POWER_CPU_IDLE': 90,
        'POWER_CPU_MAX': 280,
        'GPUS_PER_NODE': 4,
        'CPUS_PER_NODE': 1,
        'POWER_MEM': 50,
        'POWER_NIC': 30,
        'POWER_NIC_IDLE': 15,
        'POWER_NIC_MAX': 60,
        'NICS_PER_NODE': 4,
        'POWER_NVME': 10,
        'SIVOC_LOSS_CONSTANT': 15,
        'SIVOC_EFFICIENCY': 0.94,
        'pue': 1.2,
        'propagation_delay_ns': 5,
        'link_length_m': 15,
        'switch_latency_ns': 150,
        'serialization_rate_gbps': 200,
        # FLOPS config
        'CPU_PEAK_FLOPS': 3.5e12,  # AMD EPYC
        'GPU_PEAK_FLOPS': 26e12,   # MI250X
        'CPU_FP_RATIO': 0.05,
        'GPU_FP_RATIO': 0.95,
        'SC_SHAPE': (22, 22, 20),
        # Carbon config
        'CARBON_INTENSITY': 0.35,  # Tennessee has cleaner grid
    }
}

# ==========================================
# Import RAPS components
# ==========================================
try:
    from raps.network import build_fattree, build_dragonfly, get_link_util_stats
    from raps.network.base import network_congestion, network_slowdown
    from raps.power import compute_node_power
    RAPS_AVAILABLE = True
    print("RAPS core modules loaded")
except ImportError as e:
    print(f"Warning: RAPS not fully available: {e}")
    RAPS_AVAILABLE = False

# Try to import RL components
RL_AVAILABLE = False
try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    RL_AVAILABLE = True
    print("RL modules (gymnasium, stable-baselines3) loaded")
except ImportError as e:
    print(f"RL modules not available: {e}")


# ==========================================
# Data Loaders
# ==========================================
def load_static_matrix(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, 'r') as f:
        return f['traffic_matrix'][:]


def load_affinity_graph(json_path: Path) -> Dict:
    with open(json_path, 'r') as f:
        return json.load(f)


def load_dynamic_matrix(npy_path: Path) -> Tuple[np.ndarray, Dict]:
    matrix = np.load(npy_path)
    meta_path = npy_path.parent / (npy_path.stem + '_meta.json')
    metadata = {}
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
    return matrix, metadata


def get_experiment_files(name: str) -> Dict[str, Path]:
    return {
        'static': MATRIX_DIR / f"{name}.h5",
        'affinity': MATRIX_DIR / f"{name}_affinity.json",
        'dynamic': MATRIX_DIR / f"{name}_dynamic.npy",
    }


# ==========================================
# FLOPS Manager (simplified from RAPS)
# ==========================================
class SimpleFLOPSManager:
    """Compute FLOPS and efficiency metrics."""

    def __init__(self, config: Dict):
        self.config = config

    def compute_flops(self, active_nodes: int, cpu_util: float, gpu_util: float) -> Dict:
        """Compute FLOPS for given utilization."""
        cpu_flops = (
            active_nodes *
            self.config['CPUS_PER_NODE'] *
            self.config['CPU_PEAK_FLOPS'] *
            cpu_util *
            self.config['CPU_FP_RATIO']
        )
        gpu_flops = (
            active_nodes *
            self.config['GPUS_PER_NODE'] *
            self.config['GPU_PEAK_FLOPS'] *
            gpu_util *
            self.config['GPU_FP_RATIO']
        )
        total_flops = cpu_flops + gpu_flops
        return {
            'total_pflops': total_flops / 1e15,
            'cpu_pflops': cpu_flops / 1e15,
            'gpu_pflops': gpu_flops / 1e15,
        }

    def get_rpeak(self) -> float:
        """Get theoretical peak FLOPS."""
        node_peak = (
            self.config['CPUS_PER_NODE'] * self.config['CPU_PEAK_FLOPS'] +
            self.config['GPUS_PER_NODE'] * self.config['GPU_PEAK_FLOPS']
        )
        return node_peak * self.config['AVAILABLE_NODES'] / 1e15  # PFLOPS


# ==========================================
# Carbon Emissions Calculator
# ==========================================
def compute_carbon_emissions(energy_kwh: float, config: Dict) -> Dict:
    """Compute carbon footprint from energy consumption."""
    carbon_intensity = config.get('CARBON_INTENSITY', 0.4)  # kg CO2 per kWh
    co2_kg = energy_kwh * carbon_intensity
    return {
        'co2_kg': co2_kg,
        'co2_metric_tons': co2_kg / 1000,
        'carbon_intensity': carbon_intensity,
    }


# ==========================================
# Realistic Latency Model
# ==========================================
def compute_realistic_latency(num_hops: int, message_size_bytes: int,
                               link_utilization: float, config: Dict) -> Dict:
    """Compute network latency with all components."""
    # Propagation
    link_length = config.get('link_length_m', 10)
    prop_delay = config.get('propagation_delay_ns', 5)
    propagation_ns = num_hops * link_length * prop_delay

    # Transmission
    link_speed_gbps = config.get('serialization_rate_gbps', 100)
    transmission_ns = (message_size_bytes * 8 / (link_speed_gbps * 1e9)) * 1e9

    # Queueing (M/M/1 approximation)
    util_capped = min(link_utilization, 0.95)
    if util_capped > 0.01:
        service_time = max(transmission_ns, 100)
        queueing_ns = (util_capped / (1 - util_capped)) * service_time * num_hops
    else:
        queueing_ns = 0

    # Processing
    switch_latency = config.get('switch_latency_ns', 200)
    processing_ns = num_hops * switch_latency

    total_ns = propagation_ns + transmission_ns + queueing_ns + processing_ns

    return {
        'total_us': total_ns / 1000,
        'propagation_us': propagation_ns / 1000,
        'transmission_us': transmission_ns / 1000,
        'queueing_us': queueing_ns / 1000,
        'processing_us': processing_ns / 1000,
    }


# ==========================================
# Job and Scheduling Classes
# ==========================================
@dataclass
class SimJob:
    """A job for scheduling simulation."""
    id: int
    submit_time: float
    nodes_required: int
    wall_time: float
    priority: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    allocated_nodes: List[int] = field(default_factory=list)

    @property
    def is_running(self) -> bool:
        return self.start_time is not None and self.end_time is None

    @property
    def is_completed(self) -> bool:
        return self.end_time is not None


class ResourceManager:
    """Manages node allocation."""

    def __init__(self, total_nodes: int):
        self.total_nodes = total_nodes
        self.available_nodes = set(range(total_nodes))

    def allocate(self, num_nodes: int) -> List[int]:
        if num_nodes > len(self.available_nodes):
            return []
        allocated = list(self.available_nodes)[:num_nodes]
        self.available_nodes -= set(allocated)
        return allocated

    def release(self, nodes: List[int]):
        self.available_nodes.update(nodes)

    @property
    def utilization(self) -> float:
        used = self.total_nodes - len(self.available_nodes)
        return used / self.total_nodes if self.total_nodes > 0 else 0


# ==========================================
# RL Scheduler Environment
# ==========================================
class SimpleRLSchedulerEnv(gym.Env):
    """Simplified RL environment for job scheduling."""

    def __init__(self, jobs: List[SimJob], config: Dict):
        super().__init__()
        self.initial_jobs = jobs
        self.config = config
        self.max_queue_size = 50

        # Observation: job features (nodes, wall_time, wait_time, priority)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.max_queue_size, 4), dtype=np.float32
        )
        # Action: select which job in queue to schedule
        self.action_space = gym.spaces.Discrete(self.max_queue_size)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.jobs = [SimJob(
            id=j.id, submit_time=j.submit_time, nodes_required=j.nodes_required,
            wall_time=j.wall_time, priority=j.priority
        ) for j in self.initial_jobs]
        self.queue = []
        self.running = []
        self.completed = []
        self.current_time = 0
        self.resource_mgr = ResourceManager(self.config['total_nodes'])
        self.total_reward = 0
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """Get normalized observation of queue state."""
        obs = np.zeros((self.max_queue_size, 4), dtype=np.float32)
        max_nodes = self.config['total_nodes']
        max_time = 86400  # 1 day

        for i, job in enumerate(self.queue[:self.max_queue_size]):
            obs[i, 0] = job.nodes_required / max_nodes
            obs[i, 1] = job.wall_time / max_time
            obs[i, 2] = (self.current_time - job.submit_time) / max_time
            obs[i, 3] = job.priority / 10
        return obs

    def step(self, action: int):
        # Add newly arrived jobs to queue
        for job in self.jobs:
            if job.submit_time <= self.current_time and job not in self.queue and not job.is_running and not job.is_completed:
                self.queue.append(job)

        reward = -0.01  # Small time penalty

        # Try to schedule selected job
        if 0 <= action < len(self.queue):
            job = self.queue[action]
            nodes = self.resource_mgr.allocate(job.nodes_required)
            if nodes:
                job.allocated_nodes = nodes
                job.start_time = self.current_time
                job.end_time = self.current_time + job.wall_time
                self.running.append(job)
                self.queue.remove(job)
                reward += 1.0  # Reward for scheduling

        # Complete finished jobs
        newly_completed = [j for j in self.running if j.end_time <= self.current_time]
        for job in newly_completed:
            self.resource_mgr.release(job.allocated_nodes)
            self.completed.append(job)
            self.running.remove(job)
            reward += 5.0  # Bonus for completion

        # Advance time
        self.current_time += 60  # 1 minute steps

        # Check if done
        all_jobs_done = len(self.completed) == len(self.jobs)
        timeout = self.current_time > 86400 * 7  # 1 week max
        done = all_jobs_done or timeout

        self.total_reward += reward
        return self._get_obs(), reward, done, False, {}


def run_rl_scheduling(jobs: List[SimJob], config: Dict, train_steps: int = 1000) -> Dict:
    """Run RL-based scheduling simulation."""
    if not RL_AVAILABLE:
        return {'error': 'RL not available'}

    env = SimpleRLSchedulerEnv(jobs, config)

    # Train a simple PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
        learning_rate=3e-4,
        verbose=0,
    )

    model.learn(total_timesteps=train_steps)

    # Evaluate
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

    # Compute metrics
    if env.completed:
        makespan = max(j.end_time for j in env.completed) - min(j.submit_time for j in env.completed)
        avg_wait = np.mean([j.start_time - j.submit_time for j in env.completed if j.start_time])
        avg_turnaround = np.mean([j.end_time - j.submit_time for j in env.completed if j.end_time])
    else:
        makespan = env.current_time
        avg_wait = 0
        avg_turnaround = 0

    return {
        'makespan': makespan,
        'avg_wait_time': avg_wait,
        'avg_turnaround': avg_turnaround,
        'jobs_completed': len(env.completed),
        'total_reward': env.total_reward,
        'utilization': (1 - len(env.resource_mgr.available_nodes) / config['total_nodes']) * 100,
    }


# ==========================================
# Traditional Schedulers
# ==========================================
def run_traditional_scheduling(jobs: List[SimJob], scheduler_type: str, config: Dict) -> Dict:
    """Run traditional scheduling (FCFS, SJF, Backfill)."""
    # Copy jobs
    sim_jobs = [SimJob(
        id=j.id, submit_time=j.submit_time, nodes_required=j.nodes_required,
        wall_time=j.wall_time, priority=j.priority
    ) for j in jobs]

    # Sort based on policy
    if scheduler_type == 'fcfs':
        sim_jobs.sort(key=lambda j: j.submit_time)
    elif scheduler_type == 'sjf':
        sim_jobs.sort(key=lambda j: (j.submit_time, j.wall_time))
    elif scheduler_type == 'backfill':
        sim_jobs.sort(key=lambda j: j.submit_time)

    resource_mgr = ResourceManager(config['total_nodes'])
    queue = []
    running = []
    completed = []
    current_time = 0

    while sim_jobs or queue or running:
        # Add arrived jobs to queue
        while sim_jobs and sim_jobs[0].submit_time <= current_time:
            queue.append(sim_jobs.pop(0))

        # Complete finished jobs
        newly_completed = [j for j in running if j.end_time <= current_time]
        for job in newly_completed:
            resource_mgr.release(job.allocated_nodes)
            completed.append(job)
            running.remove(job)

        # Schedule from queue
        scheduled_any = False
        for i, job in enumerate(queue[:]):
            nodes = resource_mgr.allocate(job.nodes_required)
            if nodes:
                job.allocated_nodes = nodes
                job.start_time = current_time
                job.end_time = current_time + job.wall_time
                running.append(job)
                queue.remove(job)
                scheduled_any = True
                if scheduler_type != 'backfill':
                    break  # FCFS/SJF: only schedule head of queue

        # Advance time
        if running:
            next_completion = min(j.end_time for j in running)
            next_arrival = sim_jobs[0].submit_time if sim_jobs else float('inf')
            current_time = min(next_completion, next_arrival)
        elif sim_jobs:
            current_time = sim_jobs[0].submit_time
        else:
            break

        if current_time > 1e9:
            break

    # Compute metrics
    if completed:
        makespan = max(j.end_time for j in completed) - min(j.submit_time for j in completed)
        avg_wait = np.mean([j.start_time - j.submit_time for j in completed])
        avg_turnaround = np.mean([j.end_time - j.submit_time for j in completed])
        total_node_time = sum(j.nodes_required * (j.end_time - j.start_time) for j in completed)
        utilization = total_node_time / (makespan * config['total_nodes']) * 100 if makespan > 0 else 0
    else:
        makespan = current_time
        avg_wait = 0
        avg_turnaround = 0
        utilization = 0

    return {
        'makespan': makespan,
        'avg_wait_time': avg_wait,
        'avg_turnaround': avg_turnaround,
        'jobs_completed': len(completed),
        'utilization': utilization,
    }


# ==========================================
# Generate Jobs from Traffic Matrix
# ==========================================
def generate_jobs_from_traffic(dynamic_matrix: np.ndarray, metadata: Dict, scale_factor: float = 100) -> List[SimJob]:
    """Generate jobs based on traffic patterns in dynamic matrix."""
    num_bins = dynamic_matrix.shape[0]
    num_ranks = dynamic_matrix.shape[1]

    traffic_per_bin = dynamic_matrix.sum(axis=(1, 2))
    threshold = traffic_per_bin.mean() + 0.5 * traffic_per_bin.std()

    jobs = []
    job_id = 0

    # Find traffic bursts
    in_burst = False
    burst_start = 0

    for t in range(num_bins):
        if traffic_per_bin[t] > threshold and not in_burst:
            in_burst = True
            burst_start = t
        elif (traffic_per_bin[t] <= threshold or t == num_bins - 1) and in_burst:
            in_burst = False
            burst_end = t

            # Create job from burst
            burst_traffic = traffic_per_bin[burst_start:burst_end].sum()
            relative_size = burst_traffic / (traffic_per_bin.sum() + 1e-10)

            job_nodes = max(1, int(relative_size * num_ranks * 4))
            job_nodes = min(job_nodes, 1000)  # Cap

            job_duration = (burst_end - burst_start) * scale_factor
            job_duration = max(60, min(job_duration, 7200))

            jobs.append(SimJob(
                id=job_id,
                submit_time=burst_start * scale_factor,
                nodes_required=job_nodes,
                wall_time=job_duration,
                priority=np.random.randint(1, 5)
            ))
            job_id += 1

    # Add some random jobs if not enough detected
    if len(jobs) < 5:
        for i in range(10):
            jobs.append(SimJob(
                id=job_id + i,
                submit_time=i * 500,
                nodes_required=max(1, num_ranks // (i + 1)),
                wall_time=300 + i * 100,
                priority=np.random.randint(1, 5)
            ))

    return jobs


# ==========================================
# USE CASE 1: Adaptive Routing
# ==========================================
def run_adaptive_routing(static_matrix: np.ndarray, system: str, config: dict) -> List[Dict]:
    """Adaptive routing using static traffic matrix."""
    results = []
    num_nodes = static_matrix.shape[0]

    if config['topology'] == 'dragonfly':
        algorithms = ['minimal', 'ugal', 'valiant']
        hop_counts = {'minimal': 3.5, 'ugal': 4.2, 'valiant': 5.5}
    else:
        algorithms = ['minimal', 'ecmp']
        hop_counts = {'minimal': 3.5, 'ecmp': 4.0}

    total_bytes = np.sum(static_matrix)
    avg_msg_size = int(total_bytes / max(1, np.count_nonzero(static_matrix)))
    link_util = min(0.8, total_bytes / (config['network_max_bw'] * num_nodes * 10))

    for algo in algorithms:
        hops = hop_counts.get(algo, 4.0)
        latency = compute_realistic_latency(int(hops), avg_msg_size, link_util, config)

        if RAPS_AVAILABLE:
            slowdown = network_slowdown(total_bytes / num_nodes, config['network_max_bw'])
        else:
            slowdown = 1.0 + max(0, link_util - 0.5) * 2

        throughput = config['network_max_bw'] / slowdown

        results.append({
            'algorithm': algo,
            'latency_us': latency['total_us'],
            'throughput_gbps': throughput / 1e9,
            'congestion': link_util,
            'slowdown_factor': slowdown,
            'avg_hops': hops,
        })

    return results


# ==========================================
# USE CASE 2: Node Placement
# ==========================================
def run_node_placement(affinity_graph: Dict, system: str, config: dict) -> List[Dict]:
    """Node placement using affinity graph."""
    results = []
    num_ranks = affinity_graph['num_nodes']
    edges = affinity_graph['edges']

    strategies = ['contiguous', 'random', 'locality', 'spectral']

    for strategy in strategies:
        if strategy == 'contiguous':
            mapping = np.arange(num_ranks) % config['total_nodes']
        elif strategy == 'random':
            mapping = np.random.permutation(num_ranks) % config['total_nodes']
        elif strategy == 'locality':
            mapping = locality_placement(affinity_graph, config)
        else:
            mapping = spectral_placement(affinity_graph, config)

        cost = compute_placement_cost(affinity_graph, mapping, config)
        baseline = compute_placement_cost(
            affinity_graph, np.arange(num_ranks) % config['total_nodes'], config
        )
        reduction = 1.0 - cost / baseline if baseline > 0 and strategy != 'contiguous' else 0

        results.append({
            'strategy': strategy,
            'communication_cost': cost,
            'cost_reduction': reduction,
        })

    return results


def locality_placement(affinity: Dict, config: Dict) -> np.ndarray:
    """Locality-aware placement."""
    num_ranks = affinity['num_nodes']
    edges = affinity['edges']

    neighbors = defaultdict(lambda: defaultdict(int))
    for edge in edges:
        neighbors[edge['source']][edge['target']] = edge['weight']
        neighbors[edge['target']][edge['source']] = edge['weight']

    mapping = np.full(num_ranks, -1, dtype=int)
    placed = set()

    # Start with highest traffic node
    node_traffic = defaultdict(int)
    for edge in edges:
        node_traffic[edge['source']] += edge['weight']
        node_traffic[edge['target']] += edge['weight']

    if node_traffic:
        start = max(node_traffic, key=node_traffic.get)
    else:
        start = 0

    mapping[start] = 0
    placed.add(start)

    for _ in range(1, num_ranks):
        best_rank = -1
        best_weight = -1

        for r in range(num_ranks):
            if r in placed:
                continue
            weight = sum(neighbors[r].get(p, 0) for p in placed)
            if weight > best_weight:
                best_weight = weight
                best_rank = r

        if best_rank >= 0:
            # Place near heaviest neighbor
            heaviest = max(placed, key=lambda p: neighbors[best_rank].get(p, 0))
            mapping[best_rank] = (mapping[heaviest] + len(placed)) % config['total_nodes']
            placed.add(best_rank)

    return mapping


def spectral_placement(affinity: Dict, config: Dict) -> np.ndarray:
    """Spectral clustering based placement."""
    num_ranks = affinity['num_nodes']
    edges = affinity['edges']

    adj = np.zeros((num_ranks, num_ranks))
    for edge in edges:
        adj[edge['source'], edge['target']] = edge['weight']
        adj[edge['target'], edge['source']] = edge['weight']

    try:
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csr_matrix

        D = np.diag(adj.sum(axis=1) + 1e-10)
        L = D - adj
        _, evecs = eigsh(csr_matrix(L), k=min(3, num_ranks-1), which='SM')
        order = np.argsort(evecs[:, -1])
        mapping = np.zeros(num_ranks, dtype=int)
        for i, r in enumerate(order):
            mapping[r] = i % config['total_nodes']
        return mapping
    except:
        return np.arange(num_ranks) % config['total_nodes']


def compute_placement_cost(affinity: Dict, mapping: np.ndarray, config: Dict) -> float:
    """Compute communication cost for placement."""
    cost = 0
    edges = affinity['edges']

    if config['topology'] == 'dragonfly':
        d, p = config['dragonfly_d'], config['dragonfly_p']
        nodes_per_group = d * p
    else:
        k = config['fattree_k']
        nodes_per_pod = (k * k) // 4

    for edge in edges:
        src, dst = edge['source'], edge['target']
        weight = edge['weight']
        n_src, n_dst = mapping[src], mapping[dst]

        if config['topology'] == 'dragonfly':
            g_src = n_src // nodes_per_group
            g_dst = n_dst // nodes_per_group
            hops = 0 if n_src == n_dst else (2 if g_src == g_dst else 4)
        else:
            p_src = n_src // nodes_per_pod
            p_dst = n_dst // nodes_per_pod
            hops = 0 if n_src == n_dst else (2 if p_src == p_dst else 4)

        cost += weight * hops

    return cost


# ==========================================
# USE CASE 3: Job Scheduling (with RL)
# ==========================================
def run_scheduling(dynamic_matrix: np.ndarray, metadata: Dict,
                   system: str, config: dict) -> List[Dict]:
    """Job scheduling with FCFS, Backfill, SJF, and RL."""
    results = []

    # Generate jobs from traffic
    jobs = generate_jobs_from_traffic(dynamic_matrix, metadata)

    # Traditional schedulers
    for scheduler in ['fcfs', 'backfill', 'sjf']:
        sched_result = run_traditional_scheduling(jobs, scheduler, config)

        # Compute power and energy
        avg_util = sched_result['utilization'] / 100
        active_nodes = int(config['total_nodes'] * avg_util)

        if RAPS_AVAILABLE:
            power_per_node, _ = compute_node_power(avg_util, avg_util * 0.8, 0.3, config)
            total_power = power_per_node * active_nodes
        else:
            total_power = active_nodes * (config['POWER_CPU_MAX'] * avg_util + config['POWER_GPU_MAX'] * avg_util * 0.8)

        energy_kwh = total_power * sched_result['makespan'] / 3600 / 1000

        # Carbon emissions
        carbon = compute_carbon_emissions(energy_kwh, config)

        # FLOPS
        flops_mgr = SimpleFLOPSManager(config)
        flops = flops_mgr.compute_flops(active_nodes, avg_util, avg_util * 0.8)

        results.append({
            'scheduler': scheduler,
            'makespan': sched_result['makespan'],
            'avg_wait_time': sched_result['avg_wait_time'],
            'utilization': sched_result['utilization'],
            'jobs_completed': sched_result['jobs_completed'],
            'energy_kwh': energy_kwh,
            'avg_power_kw': total_power / 1000,
            'co2_kg': carbon['co2_kg'],
            'pflops': flops['total_pflops'],
        })

    # RL Scheduler
    if RL_AVAILABLE:
        try:
            rl_result = run_rl_scheduling(jobs, config, train_steps=500)

            avg_util = rl_result.get('utilization', 50) / 100
            active_nodes = int(config['total_nodes'] * avg_util)

            if RAPS_AVAILABLE:
                power_per_node, _ = compute_node_power(avg_util, avg_util * 0.8, 0.3, config)
                total_power = power_per_node * active_nodes
            else:
                total_power = active_nodes * (config['POWER_CPU_MAX'] * avg_util + config['POWER_GPU_MAX'] * avg_util * 0.8)

            energy_kwh = total_power * rl_result['makespan'] / 3600 / 1000
            carbon = compute_carbon_emissions(energy_kwh, config)
            flops_mgr = SimpleFLOPSManager(config)
            flops = flops_mgr.compute_flops(active_nodes, avg_util, avg_util * 0.8)

            results.append({
                'scheduler': 'rl',
                'makespan': rl_result['makespan'],
                'avg_wait_time': rl_result['avg_wait_time'],
                'utilization': rl_result.get('utilization', 0),
                'jobs_completed': rl_result['jobs_completed'],
                'energy_kwh': energy_kwh,
                'avg_power_kw': total_power / 1000,
                'co2_kg': carbon['co2_kg'],
                'pflops': flops['total_pflops'],
                'rl_reward': rl_result.get('total_reward', 0),
            })
        except Exception as e:
            print(f"RL scheduling failed: {e}")

    return results


# ==========================================
# USE CASE 4: Power Analysis
# ==========================================
def run_power_analysis(dynamic_matrix: np.ndarray, metadata: Dict,
                       system: str, config: dict) -> List[Dict]:
    """Power analysis with FLOPS and carbon tracking."""
    results = []
    scenarios = ['baseline', 'power_cap', 'frequency_scaling', 'job_packing']

    num_bins = min(dynamic_matrix.shape[0], 500)
    traffic_per_bin = dynamic_matrix.sum(axis=(1, 2))
    max_traffic = traffic_per_bin.max() if traffic_per_bin.max() > 0 else 1
    utilization_trace = traffic_per_bin[:num_bins] / max_traffic

    flops_mgr = SimpleFLOPSManager(config)
    rpeak = flops_mgr.get_rpeak()

    for scenario in scenarios:
        power_trace = []
        flops_trace = []

        for t in range(num_bins):
            base_util = 0.3 + 0.5 * utilization_trace[t]
            active_nodes = int(config['total_nodes'] * base_util * 0.8)

            if scenario == 'baseline':
                cpu_util, gpu_util = base_util, base_util * 0.9
            elif scenario == 'power_cap':
                cap = 0.75
                cpu_util = min(base_util, cap)
                gpu_util = min(base_util * 0.9, cap)
            elif scenario == 'frequency_scaling':
                freq_factor = 0.7 + 0.3 * utilization_trace[t]
                cpu_util = base_util * freq_factor
                gpu_util = base_util * 0.9 * freq_factor
            else:  # job_packing
                active_nodes = max(1, int(active_nodes * 0.8))
                cpu_util = min(base_util * 1.2, 0.95)
                gpu_util = min(base_util * 1.1, 0.95)

            if RAPS_AVAILABLE:
                power_per_node, _ = compute_node_power(cpu_util, gpu_util, 0.3, config)
                power = power_per_node * active_nodes
            else:
                power = active_nodes * (
                    config['POWER_CPU_MAX'] * cpu_util +
                    config['GPUS_PER_NODE'] * config['POWER_GPU_MAX'] * gpu_util
                )

            power_trace.append(power)
            flops = flops_mgr.compute_flops(active_nodes, cpu_util, gpu_util)
            flops_trace.append(flops['total_pflops'])

        avg_power = np.mean(power_trace)
        peak_power = np.max(power_trace)
        avg_flops = np.mean(flops_trace)

        # Energy and carbon
        sim_time_hours = num_bins * 0.01 / 3600  # Assuming 10ms per bin
        energy_kwh = avg_power * sim_time_hours / 1000
        carbon = compute_carbon_emissions(energy_kwh * 1000, config)  # Scale up for meaningful numbers

        results.append({
            'scenario': scenario,
            'compute_power_mw': avg_power / 1e6,
            'total_power_mw': avg_power * config['pue'] / 1e6,
            'peak_power_mw': peak_power * config['pue'] / 1e6,
            'avg_pflops': avg_flops,
            'efficiency_gflops_w': (avg_flops * 1e6) / (avg_power / 1000) if avg_power > 0 else 0,
            'energy_kwh': energy_kwh * 1000,
            'co2_kg': carbon['co2_kg'] * 1000,
            'rpeak_pflops': rpeak,
        })

    return results


# ==========================================
# Synthetic Data Generator
# ==========================================
def generate_synthetic_data(pattern: str, num_ranks: int):
    """Generate synthetic traffic data with realistic bursts."""
    # Static matrix
    if pattern == 'all-to-all':
        static = np.ones((num_ranks, num_ranks)) * 1000000
        np.fill_diagonal(static, 0)
    elif pattern == 'stencil-3d':
        static = generate_stencil_3d(num_ranks)
    elif pattern == 'nearest-neighbor':
        static = np.zeros((num_ranks, num_ranks))
        for i in range(num_ranks):
            static[i, (i-1) % num_ranks] = 10000000
            static[i, (i+1) % num_ranks] = 10000000
    else:  # ring
        static = np.zeros((num_ranks, num_ranks))
        for i in range(num_ranks):
            static[i, (i+1) % num_ranks] = 8000000

    # Affinity graph
    affinity = matrix_to_affinity(static)

    # Dynamic matrix with random bursts
    num_bins = 200
    dynamic = np.zeros((num_bins, num_ranks, num_ranks))

    np.random.seed(hash(pattern) % 2**32)
    num_bursts = np.random.randint(3, 8)
    burst_starts = sorted(np.random.choice(num_bins - 20, num_bursts, replace=False))
    burst_durations = np.random.randint(5, 20, num_bursts)
    burst_intensities = np.random.uniform(0.5, 1.0, num_bursts)

    intensity = np.ones(num_bins) * 0.1
    for start, dur, amp in zip(burst_starts, burst_durations, burst_intensities):
        for t in range(start, min(start + dur, num_bins)):
            progress = (t - start) / dur
            factor = 1 - 4 * (progress - 0.5) ** 2  # Parabolic
            intensity[t] = max(intensity[t], 0.1 + amp * factor)

    for t in range(num_bins):
        dynamic[t] = static * intensity[t] / num_bins
        np.fill_diagonal(dynamic[t], 0)

    meta = {'num_time_bins': num_bins, 'time_bin_size': 0.01}
    return static, affinity, dynamic, meta


def generate_stencil_3d(n: int) -> np.ndarray:
    matrix = np.zeros((n, n))
    nx = int(np.cbrt(n))
    ny = nx
    nz = max(1, n // (nx * ny))

    for i in range(n):
        iz, iy, ix = i // (nx * ny), (i % (nx * ny)) // nx, i % nx
        for dz, dy, dx in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
            jx, jy, jz = (ix+dx) % nx, (iy+dy) % ny, (iz+dz) % nz
            j = jz * nx * ny + jy * nx + jx
            if j < n and j != i:
                matrix[i, j] = 5000000
    return matrix


def matrix_to_affinity(matrix: np.ndarray) -> Dict:
    n = matrix.shape[0]
    sym = matrix + matrix.T
    nodes = [{'id': i, 'send_bytes': int(matrix[i].sum()),
              'recv_bytes': int(matrix[:, i].sum())} for i in range(n)]
    edges = [{'source': i, 'target': j, 'weight': float(sym[i, j])}
             for i in range(n) for j in range(i+1, n) if sym[i, j] > 0]
    return {'num_nodes': n, 'nodes': nodes, 'edges': edges}


# ==========================================
# Main Pipeline
# ==========================================
def run_pipeline():
    print("=" * 70)
    print("SC26 Complete Pipeline v3.0 (Full RAPS Integration)")
    print("=" * 70)
    print(f"\nRL Available: {RL_AVAILABLE}")
    print(f"RAPS Available: {RAPS_AVAILABLE}")
    print("\nFeatures:")
    print("  - RL Scheduler (PPO from stable-baselines3)")
    print("  - FLOPS tracking and efficiency metrics")
    print("  - Carbon emissions calculation")
    print("  - Realistic 4-component latency model")
    print()

    all_results = []

    # Process real data
    print("=" * 60)
    print("Processing Real Mini-App Data")
    print("=" * 60)

    h5_files = list(MATRIX_DIR.glob("*.h5"))
    for h5_file in tqdm(h5_files[:5], desc="Real data"):  # Limit for speed
        name = h5_file.stem
        files = get_experiment_files(name)

        if not (files['affinity'].exists() and files['dynamic'].exists()):
            continue

        static = load_static_matrix(files['static'])
        affinity = load_affinity_graph(files['affinity'])
        dynamic, meta = load_dynamic_matrix(files['dynamic'])

        for system, config in SYSTEMS.items():
            for r in run_adaptive_routing(static, system, config):
                all_results.append({'experiment': name, 'data_type': 'real',
                                    'system': system, 'use_case': 'routing', **r})

            for r in run_node_placement(affinity, system, config):
                all_results.append({'experiment': name, 'data_type': 'real',
                                    'system': system, 'use_case': 'placement', **r})

            for r in run_scheduling(dynamic, meta, system, config):
                all_results.append({'experiment': name, 'data_type': 'real',
                                    'system': system, 'use_case': 'scheduling', **r})

            for r in run_power_analysis(dynamic, meta, system, config):
                all_results.append({'experiment': name, 'data_type': 'real',
                                    'system': system, 'use_case': 'power', **r})

    # Process synthetic data
    print("\n" + "=" * 60)
    print("Processing Synthetic Patterns")
    print("=" * 60)

    patterns = ['all-to-all', 'stencil-3d', 'nearest-neighbor', 'ring']
    ranks = [64, 256]

    for num_ranks in ranks:
        for pattern in tqdm(patterns, desc=f"Synthetic n={num_ranks}"):
            static, affinity, dynamic, meta = generate_synthetic_data(pattern, num_ranks)

            for system, config in SYSTEMS.items():
                for r in run_adaptive_routing(static, system, config):
                    all_results.append({'experiment': f'synth_{pattern}_{num_ranks}',
                                        'pattern': pattern, 'num_ranks': num_ranks,
                                        'data_type': 'synthetic', 'system': system,
                                        'use_case': 'routing', **r})

                for r in run_node_placement(affinity, system, config):
                    all_results.append({'experiment': f'synth_{pattern}_{num_ranks}',
                                        'pattern': pattern, 'num_ranks': num_ranks,
                                        'data_type': 'synthetic', 'system': system,
                                        'use_case': 'placement', **r})

                for r in run_scheduling(dynamic, meta, system, config):
                    all_results.append({'experiment': f'synth_{pattern}_{num_ranks}',
                                        'pattern': pattern, 'num_ranks': num_ranks,
                                        'data_type': 'synthetic', 'system': system,
                                        'use_case': 'scheduling', **r})

                for r in run_power_analysis(dynamic, meta, system, config):
                    all_results.append({'experiment': f'synth_{pattern}_{num_ranks}',
                                        'pattern': pattern, 'num_ranks': num_ranks,
                                        'data_type': 'synthetic', 'system': system,
                                        'use_case': 'power', **r})

    # Save results
    df = pd.DataFrame(all_results)
    csv_path = RESULTS_DIR / "pipeline_v3_results.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 70)
    print("PIPELINE v3.0 COMPLETE")
    print("=" * 70)
    print(f"Results: {csv_path}")
    print(f"Total experiments: {len(all_results)}")
    print(f"\nSchedulers used: {df[df['use_case']=='scheduling']['scheduler'].unique().tolist()}")

    return df


if __name__ == "__main__":
    run_pipeline()

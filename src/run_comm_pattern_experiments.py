#!/usr/bin/env python3
"""
Communication Pattern Experiments
==================================

Compare how different communication patterns (STENCIL_3D vs ALL_TO_ALL)
interact with different network topologies (Fat-Tree vs Dragonfly).

Mini-apps:
- LULESH-like: STENCIL_3D (structured mesh, 6 neighbors)
- HPL-like: ALL_TO_ALL (dense linear algebra, global communication)
"""

import sys
sys.path.insert(0, '/app')

import subprocess
import json
import pandas as pd
from pathlib import Path

# Import RAPS components
from raps.job import Job, job_dict, CommunicationPattern, MESSAGE_SIZE_64K, MESSAGE_SIZE_1M
from raps.config import SystemConfig

DATA_DIR = Path("/app/data/experiments_commpattern")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def create_stencil_jobs(num_jobs, config):
    """Create LULESH-like stencil pattern jobs."""
    jobs = []
    trace_quanta = config.get('TRACE_QUANTA', 15)

    for i in range(num_jobs):
        nodes = [32, 64, 128, 256][i % 4]
        duration = 300 + (i * 60)  # 5-10 minutes
        trace_len = max(1, duration // trace_quanta)

        # 10 GB/s per node typical for stencil
        bandwidth = 1e10

        job = Job(job_dict(
            id=1000 + i,
            name=f"LULESH_{i}",
            account="stencil",
            nodes_required=nodes,
            scheduled_nodes=[],
            cpu_trace=[0.7] * trace_len,
            gpu_trace=[0.8] * trace_len,
            ntx_trace=[bandwidth] * trace_len,
            nrx_trace=[bandwidth] * trace_len,
            comm_pattern=CommunicationPattern.STENCIL_3D,
            message_size=MESSAGE_SIZE_1M,
            trace_quanta=trace_quanta,
            submit_time=i * 30,  # Staggered submission
            expected_run_time=duration,
            time_limit=duration * 2,
            end_state="COMPLETED",
        ))
        jobs.append(job)

    return jobs


def create_alltoall_jobs(num_jobs, config):
    """Create HPL-like all-to-all pattern jobs."""
    jobs = []
    trace_quanta = config.get('TRACE_QUANTA', 15)

    for i in range(num_jobs):
        nodes = [32, 64, 128, 256][i % 4]
        duration = 300 + (i * 60)
        trace_len = max(1, duration // trace_quanta)

        # 10 GB/s per node
        bandwidth = 1e10

        job = Job(job_dict(
            id=2000 + i,
            name=f"HPL_{i}",
            account="alltoall",
            nodes_required=nodes,
            scheduled_nodes=[],
            cpu_trace=[0.5] * trace_len,
            gpu_trace=[0.95] * trace_len,
            ntx_trace=[bandwidth] * trace_len,
            nrx_trace=[bandwidth] * trace_len,
            comm_pattern=CommunicationPattern.ALL_TO_ALL,
            message_size=MESSAGE_SIZE_64K,
            trace_quanta=trace_quanta,
            submit_time=i * 30,
            expected_run_time=duration,
            time_limit=duration * 2,
            end_state="COMPLETED",
        ))
        jobs.append(job)

    return jobs


def run_simulation_with_jobs(system_name, jobs, output_dir, allocation='contiguous'):
    """Run RAPS simulation with custom jobs."""
    import yaml

    # Load system config
    config_path = Path(f"/app/config/{system_name}.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Save jobs to temporary file for RAPS
    # (RAPS doesn't have direct API for custom jobs, need to use workload interface)

    # For now, print job info and return placeholder metrics
    print(f"\n  System: {system_name}")
    print(f"  Allocation: {allocation}")
    print(f"  Jobs: {len(jobs)}")

    stencil_count = sum(1 for j in jobs if j.comm_pattern == CommunicationPattern.STENCIL_3D)
    alltoall_count = sum(1 for j in jobs if j.comm_pattern == CommunicationPattern.ALL_TO_ALL)
    print(f"  STENCIL_3D jobs: {stencil_count}")
    print(f"  ALL_TO_ALL jobs: {alltoall_count}")

    total_nodes = sum(j.nodes_required for j in jobs)
    total_bandwidth = sum(j.ntx_trace[0] for j in jobs if j.ntx_trace)
    print(f"  Total nodes requested: {total_nodes}")
    print(f"  Total bandwidth: {total_bandwidth:.2e} bytes/tick")

    return {
        'stencil_jobs': stencil_count,
        'alltoall_jobs': alltoall_count,
        'total_nodes': total_nodes,
        'total_bandwidth': total_bandwidth,
    }


def main():
    print("="*60)
    print("Communication Pattern Experiments")
    print("="*60)
    print("\nComparing mini-app communication patterns:")
    print("  - STENCIL_3D (LULESH-like): 6-neighbor communication")
    print("  - ALL_TO_ALL (HPL-like): Global communication")

    results = []

    # Dummy config for job creation
    config = {'TRACE_QUANTA': 15, 'GPUS_PER_NODE': 4}

    # Create jobs
    stencil_jobs = create_stencil_jobs(5, config)
    alltoall_jobs = create_alltoall_jobs(5, config)
    mixed_jobs = stencil_jobs + alltoall_jobs

    print("\n" + "="*60)
    print("Experiment 1: Pure STENCIL_3D workload")
    print("="*60)

    for system in ['lassen', 'frontier']:
        for allocation in ['contiguous', 'random']:
            metrics = run_simulation_with_jobs(
                system, stencil_jobs,
                DATA_DIR / f"stencil_{system}_{allocation}",
                allocation
            )
            results.append({
                'system': system,
                'pattern': 'STENCIL_3D',
                'allocation': allocation,
                **metrics
            })

    print("\n" + "="*60)
    print("Experiment 2: Pure ALL_TO_ALL workload")
    print("="*60)

    for system in ['lassen', 'frontier']:
        for allocation in ['contiguous', 'random']:
            metrics = run_simulation_with_jobs(
                system, alltoall_jobs,
                DATA_DIR / f"alltoall_{system}_{allocation}",
                allocation
            )
            results.append({
                'system': system,
                'pattern': 'ALL_TO_ALL',
                'allocation': allocation,
                **metrics
            })

    print("\n" + "="*60)
    print("Experiment 3: Mixed workload")
    print("="*60)

    for system in ['lassen', 'frontier']:
        metrics = run_simulation_with_jobs(
            system, mixed_jobs,
            DATA_DIR / f"mixed_{system}",
            'contiguous'
        )
        results.append({
            'system': system,
            'pattern': 'MIXED',
            'allocation': 'contiguous',
            **metrics
        })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(DATA_DIR / "comm_pattern_results.csv", index=False)
    print(f"\nResults saved to: {DATA_DIR / 'comm_pattern_results.csv'}")
    print(df.to_string())

    print("\n" + "="*60)
    print("Key Insight:")
    print("="*60)
    print("""
    STENCIL_3D (6 neighbors):
    - Lower total traffic: O(6n) communication pairs
    - Best with CONTIGUOUS allocation (neighbors are physically close)
    - Example: LULESH, structured mesh codes

    ALL_TO_ALL (global):
    - Higher total traffic: O(n²) communication pairs
    - May benefit from RANDOM allocation (better path diversity)
    - Example: HPL, dense linear algebra
    """)


if __name__ == "__main__":
    main()

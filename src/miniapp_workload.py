#!/usr/bin/env python3
"""
Mini-App Communication Pattern Workloads
=========================================

Creates jobs with realistic communication patterns based on real HPC mini-apps:

1. LULESH-like: 3D Stencil pattern (STENCIL_3D)
   - Each node communicates with 6 neighbors
   - Typical for structured mesh codes

2. HPL-like: All-to-All pattern (ALL_TO_ALL)
   - Every node broadcasts to all others
   - Dense linear algebra

3. CoMD-like: Neighbor exchange (STENCIL_3D variant)
   - Molecular dynamics, halo exchange

This allows comparing how different communication patterns interact with
network topology and allocation strategies.
"""

import sys
sys.path.insert(0, '/app/extern/raps')

from raps.job import Job, job_dict, CommunicationPattern, MESSAGE_SIZE_64K, MESSAGE_SIZE_1M


def create_stencil_job(job_id, nodes, duration_sec=600, bandwidth_per_node=1e10):
    """
    Create a LULESH-like stencil job.

    STENCIL_3D: Each node sends to 6 neighbors (±x, ±y, ±z)
    Typical message size: 64KB - 1MB per exchange
    Communication pattern: Regular, structured
    """
    trace_quanta = 15  # seconds
    trace_len = max(1, duration_sec // trace_quanta)

    # Stencil communication: bandwidth split among 6 neighbors
    net_traffic = bandwidth_per_node

    return Job(job_dict(
        id=job_id,
        name=f"LULESH_stencil_{job_id}",
        account="miniapp",
        nodes_required=nodes,
        scheduled_nodes=[],  # Let allocation strategy decide

        # Typical stencil compute pattern
        cpu_trace=[0.7] * trace_len,  # 70% CPU
        gpu_trace=[0.8] * trace_len,  # 80% GPU

        # Network traces
        ntx_trace=[net_traffic] * trace_len,
        nrx_trace=[net_traffic] * trace_len,

        # Communication pattern - THIS IS THE KEY!
        comm_pattern=CommunicationPattern.STENCIL_3D,
        message_size=MESSAGE_SIZE_1M,  # 1MB messages typical for stencil

        # Timing
        trace_quanta=trace_quanta,
        submit_time=0,
        expected_run_time=duration_sec,
        time_limit=duration_sec * 2,
        end_state="COMPLETED",
    ))


def create_alltoall_job(job_id, nodes, duration_sec=600, bandwidth_per_node=1e10):
    """
    Create an HPL-like all-to-all job.

    ALL_TO_ALL: Every node sends to every other node
    Typical message size: 64KB for small messages, larger for broadcasts
    Communication pattern: Dense, irregular
    """
    trace_quanta = 15
    trace_len = max(1, duration_sec // trace_quanta)

    # All-to-all: total bandwidth = n*(n-1) pairs
    net_traffic = bandwidth_per_node

    return Job(job_dict(
        id=job_id,
        name=f"HPL_alltoall_{job_id}",
        account="miniapp",
        nodes_required=nodes,
        scheduled_nodes=[],

        # HPL-like compute (very high GPU utilization)
        cpu_trace=[0.5] * trace_len,
        gpu_trace=[0.95] * trace_len,  # 95% GPU for HPL

        # Network traces
        ntx_trace=[net_traffic] * trace_len,
        nrx_trace=[net_traffic] * trace_len,

        # Communication pattern
        comm_pattern=CommunicationPattern.ALL_TO_ALL,
        message_size=MESSAGE_SIZE_64K,

        trace_quanta=trace_quanta,
        submit_time=0,
        expected_run_time=duration_sec,
        time_limit=duration_sec * 2,
        end_state="COMPLETED",
    ))


def create_mixed_workload(num_stencil=5, num_alltoall=5, base_nodes=32):
    """
    Create a mixed workload with both communication patterns.
    This allows comparing how different patterns perform.
    """
    jobs = []
    job_id = 1

    # Stencil jobs (LULESH-like)
    for i in range(num_stencil):
        nodes = base_nodes * (2 ** (i % 3))  # 32, 64, 128, 32, 64...
        jobs.append(create_stencil_job(job_id, nodes))
        job_id += 1

    # All-to-all jobs (HPL-like)
    for i in range(num_alltoall):
        nodes = base_nodes * (2 ** (i % 3))
        jobs.append(create_alltoall_job(job_id, nodes))
        job_id += 1

    return jobs


# Test
if __name__ == "__main__":
    print("Creating mini-app workload with real communication patterns...")

    stencil_job = create_stencil_job(1, 64)
    alltoall_job = create_alltoall_job(2, 64)

    print(f"\nStencil Job (LULESH-like):")
    print(f"  - comm_pattern: {stencil_job.comm_pattern}")
    print(f"  - message_size: {stencil_job.message_size} bytes")
    print(f"  - nodes: {stencil_job.nodes_required}")
    print(f"  - ntx_trace[0]: {stencil_job.ntx_trace[0]:.2e} bytes/tick")

    print(f"\nAll-to-All Job (HPL-like):")
    print(f"  - comm_pattern: {alltoall_job.comm_pattern}")
    print(f"  - message_size: {alltoall_job.message_size} bytes")
    print(f"  - nodes: {alltoall_job.nodes_required}")
    print(f"  - ntx_trace[0]: {alltoall_job.ntx_trace[0]:.2e} bytes/tick")

    print("\nMixed workload:")
    mixed = create_mixed_workload(3, 3)
    for job in mixed:
        print(f"  - {job.name}: {job.comm_pattern.value}, {job.nodes_required} nodes")

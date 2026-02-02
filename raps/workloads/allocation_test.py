"""
Allocation Test Workload

Generates synthetic jobs with varying communication intensities to study the
impact of different node allocation strategies (contiguous, random, hybrid).

Based on the "Bully" phenomenon described in:
"Watch Out for the Bully! Job Interference Study on Dragonfly Network"
(Yang et al., SC16)

Key design:
- Jobs do NOT have pre-assigned scheduled_nodes, so allocation strategy is applied
- Jobs have network traces (ntx_trace, nrx_trace) to enable congestion measurement
- Mix of high-comm "bully" jobs and low-comm "victim" jobs
"""

import random
from typing import List

from raps.job import Job, job_dict
from raps.network import max_throughput_per_tick


class AllocationTestWorkload:
    """Workload generator for allocation strategy testing."""

    def allocation_test(self, args) -> List[Job]:
        """
        Generate jobs with varying communication intensities.

        Job mix (configurable):
        - High-comm jobs ("bullies"): 30% - high network traffic
        - Medium-comm jobs: 40% - moderate network traffic
        - Low-comm jobs ("victims"): 30% - low network traffic

        All jobs have nodes_required set but NOT scheduled_nodes,
        allowing the allocation strategy to determine placement.
        """
        legacy_cfg = self.config_map[self.partitions[0]]
        trace_quanta = legacy_cfg.get("TRACE_QUANTA", 15)

        return generate_allocation_test_jobs(
            legacy_cfg=legacy_cfg,
            num_jobs=args.numjobs,
            trace_quanta=trace_quanta,
            seed=args.seed,
            # Configurable parameters (could add CLI args later)
            high_comm_fraction=0.3,
            med_comm_fraction=0.4,
            low_comm_fraction=0.3,
        )


def generate_allocation_test_jobs(
    legacy_cfg: dict,
    num_jobs: int = 60,
    trace_quanta: int = 15,
    seed: int = None,
    high_comm_fraction: float = 0.3,
    med_comm_fraction: float = 0.4,
    low_comm_fraction: float = 0.3,
) -> List[Job]:
    """
    Generate synthetic jobs with varying communication intensities.

    Parameters:
    - legacy_cfg: System configuration dictionary
    - num_jobs: Total number of jobs to generate
    - trace_quanta: Time quantum for traces (seconds)
    - seed: Random seed for reproducibility
    - high_comm_fraction: Fraction of high-communication "bully" jobs
    - med_comm_fraction: Fraction of medium-communication jobs
    - low_comm_fraction: Fraction of low-communication "victim" jobs
    """
    if seed is not None:
        random.seed(seed)

    total_nodes = int(legacy_cfg["TOTAL_NODES"])
    max_nodes_per_job = min(
        int(legacy_cfg.get("MAX_NODES_PER_JOB", 128)),
        total_nodes // 4  # Don't let single job take > 25% of system
    )
    min_nodes_per_job = max(1, int(legacy_cfg.get("MIN_NODES_PER_JOB", 2)))

    # Compute bandwidth reference for network traces
    per_tick_bw = max_throughput_per_tick(legacy_cfg, trace_quanta)

    # Communication intensity levels (fraction of max bandwidth)
    HIGH_COMM_FACTOR = 0.7   # Bullies use 70% of available bandwidth
    MED_COMM_FACTOR = 0.3    # Medium jobs use 30%
    LOW_COMM_FACTOR = 0.05   # Victims use 5%

    # Job counts by category
    n_high = int(num_jobs * high_comm_fraction)
    n_med = int(num_jobs * med_comm_fraction)
    n_low = num_jobs - n_high - n_med

    print(f"[allocation_test] Generating {num_jobs} jobs:")
    print(f"  - High-comm (bullies): {n_high} jobs @ {HIGH_COMM_FACTOR*100:.0f}% bandwidth")
    print(f"  - Medium-comm: {n_med} jobs @ {MED_COMM_FACTOR*100:.0f}% bandwidth")
    print(f"  - Low-comm (victims): {n_low} jobs @ {LOW_COMM_FACTOR*100:.0f}% bandwidth")
    print(f"  - Max throughput/tick: {per_tick_bw:.2e} bytes")

    jobs: List[Job] = []
    jid = 1
    submit_time = 0

    # Generate high-comm jobs (bullies) - typically larger jobs
    for _ in range(n_high):
        nodes = random.randint(max(min_nodes_per_job, max_nodes_per_job // 4), max_nodes_per_job)
        net_traffic = HIGH_COMM_FACTOR * per_tick_bw
        job = make_allocation_test_job(
            jid=jid,
            nodes_required=nodes,
            net_traffic=net_traffic,
            trace_quanta=trace_quanta,
            submit_time=submit_time,
            job_category="high_comm",
            legacy_cfg=legacy_cfg,
        )
        jobs.append(job)
        jid += 1
        submit_time += random.randint(1, 30)  # Stagger arrivals

    # Generate medium-comm jobs
    for _ in range(n_med):
        nodes = random.randint(min_nodes_per_job, max_nodes_per_job // 2)
        net_traffic = MED_COMM_FACTOR * per_tick_bw
        job = make_allocation_test_job(
            jid=jid,
            nodes_required=nodes,
            net_traffic=net_traffic,
            trace_quanta=trace_quanta,
            submit_time=submit_time,
            job_category="med_comm",
            legacy_cfg=legacy_cfg,
        )
        jobs.append(job)
        jid += 1
        submit_time += random.randint(1, 30)

    # Generate low-comm jobs (victims) - typically smaller jobs
    for _ in range(n_low):
        nodes = random.randint(min_nodes_per_job, max(min_nodes_per_job + 1, max_nodes_per_job // 4))
        net_traffic = LOW_COMM_FACTOR * per_tick_bw
        job = make_allocation_test_job(
            jid=jid,
            nodes_required=nodes,
            net_traffic=net_traffic,
            trace_quanta=trace_quanta,
            submit_time=submit_time,
            job_category="low_comm",
            legacy_cfg=legacy_cfg,
        )
        jobs.append(job)
        jid += 1
        submit_time += random.randint(1, 30)

    # Shuffle to interleave job types (more realistic)
    random.shuffle(jobs)

    # Re-assign submit times after shuffle
    submit_time = 0
    for job in jobs:
        job.submit_time = submit_time
        submit_time += random.randint(5, 60)

    return jobs


def make_allocation_test_job(
    jid: int,
    nodes_required: int,
    net_traffic: float,
    trace_quanta: int,
    submit_time: int,
    job_category: str,
    legacy_cfg: dict,
) -> Job:
    """
    Create a job for allocation testing.

    Key: scheduled_nodes is NOT set, allowing allocation strategy to work.
    """
    # Job duration: 5-15 minutes
    job_duration = random.randint(300, 900)
    trace_len = max(1, job_duration // trace_quanta)

    # CPU/GPU utilization (moderate, focus is on network)
    gpus_per_node = legacy_cfg.get("GPUS_PER_NODE", 4)
    cpu_util = random.uniform(0.3, 0.7)
    gpu_util = random.uniform(0.4, 0.8) * gpus_per_node

    # Network traces - constant rate for simplicity
    # (could make time-varying for more realism)
    ntx_trace = [net_traffic] * trace_len
    nrx_trace = [net_traffic] * trace_len

    # CPU/GPU traces
    cpu_trace = [cpu_util] * trace_len
    gpu_trace = [gpu_util] * trace_len

    return Job(job_dict(
        id=jid,
        name=f"{job_category}_{jid}",
        account="allocation_test",
        nodes_required=nodes_required,
        # NOTE: scheduled_nodes is NOT set - this is the key difference!
        # The allocation strategy will assign nodes when the job is scheduled.
        scheduled_nodes=[],
        cpu_trace=cpu_trace,
        gpu_trace=gpu_trace,
        ntx_trace=ntx_trace,
        nrx_trace=nrx_trace,
        trace_quanta=trace_quanta,
        submit_time=submit_time,
        expected_run_time=job_duration,
        time_limit=job_duration * 2,
        end_state="COMPLETED",
    ))

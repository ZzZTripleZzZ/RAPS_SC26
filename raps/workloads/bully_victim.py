"""Bully-victim interference workload for NRAPS fidelity validation.

Generates a controlled workload where one "bully" job (ALL_TO_ALL, high bandwidth)
co-schedules with several "victim" jobs (STENCIL_3D) to measure inter-job interference.
"""

from typing import List

from raps.job import Job, job_dict, CommunicationPattern
from raps.network import max_throughput_per_tick


class BullyVictimWorkload:
    """Workload mixin for bully-victim interference experiments."""

    def bully_victim(self, args) -> List[Job]:
        legacy_cfg = self.config_map[self.partitions[0]]
        trace_quanta = legacy_cfg.get("TRACE_QUANTA", 15)
        bully_nodes = getattr(args, 'bully_nodes', 128)
        victim_count = getattr(args, 'victim_count', 4)
        victim_nodes = getattr(args, 'victim_nodes', 64)
        duration = getattr(args, 'bully_duration', 300)
        tx_fraction = getattr(args, 'txfrac', 1.0)
        return generate_bully_victim(
            legacy_cfg=legacy_cfg,
            trace_quanta=trace_quanta,
            bully_nodes=bully_nodes,
            victim_count=victim_count,
            victim_nodes=victim_nodes,
            duration=duration,
            tx_fraction=tx_fraction,
        )


def generate_bully_victim(
    legacy_cfg: dict,
    trace_quanta: int = 15,
    bully_nodes: int = 128,
    victim_count: int = 4,
    victim_nodes: int = 64,
    duration: int = 300,
    tx_fraction: float = 1.0,
) -> List[Job]:
    """Generate one bully + K victim jobs, all starting at t=0.

    Parameters
    ----------
    legacy_cfg:
        System config dict (from SystemConfig.get_legacy()).
    trace_quanta:
        Seconds per trace sample (should match system config).
    bully_nodes:
        Number of nodes for the bully (ALL_TO_ALL) job.
    victim_count:
        Number of victim (STENCIL_3D) jobs to generate.
    victim_nodes:
        Number of nodes per victim job.
    duration:
        Wall time (seconds) for all jobs.
    tx_fraction:
        Fraction of max per-node throughput used by the bully (0..1].
        Victims use half this fraction (lower intensity stencil traffic).
    """
    per_tick_bw = max_throughput_per_tick(legacy_cfg, trace_quanta)
    bully_bw = tx_fraction * per_tick_bw
    victim_bw = 0.35 * per_tick_bw  # stencil workloads are less network intensive

    trace_len = max(1, duration // trace_quanta)
    jobs: List[Job] = []

    # Bully job: ALL_TO_ALL, high network bandwidth
    bully = Job(job_dict(
        id=1,
        name="bully_all2all",
        account="interference_study",
        nodes_required=bully_nodes,
        scheduled_nodes=[],
        cpu_trace=[0.5] * trace_len,
        gpu_trace=[0.8] * trace_len,
        ntx_trace=[bully_bw] * trace_len,
        nrx_trace=[bully_bw] * trace_len,
        comm_pattern=CommunicationPattern.ALL_TO_ALL,
        message_size=1048576,  # 1 MB
        trace_quanta=trace_quanta,
        submit_time=0,
        expected_run_time=duration,
        time_limit=duration * 2,
        end_state="COMPLETED",
    ))
    jobs.append(bully)

    # Victim jobs: STENCIL_3D, moderate bandwidth
    for k in range(victim_count):
        victim = Job(job_dict(
            id=k + 2,
            name=f"victim_stencil_{k}",
            account="interference_study",
            nodes_required=victim_nodes,
            scheduled_nodes=[],
            cpu_trace=[0.7] * trace_len,
            gpu_trace=[0.6] * trace_len,
            ntx_trace=[victim_bw] * trace_len,
            nrx_trace=[victim_bw] * trace_len,
            comm_pattern=CommunicationPattern.STENCIL_3D,
            message_size=65536,  # 64 KB (stencil halo exchange)
            trace_quanta=trace_quanta,
            submit_time=0,
            expected_run_time=duration,
            time_limit=duration * 2,
            end_state="COMPLETED",
        ))
        jobs.append(victim)

    total_nodes = 1 + victim_count
    print(f"[BullyVictim] bully={bully_nodes}n (ALL_TO_ALL, bw={bully_bw:.2e} B/tick), "
          f"victims={victim_count}×{victim_nodes}n (STENCIL_3D, bw={victim_bw:.2e} B/tick), "
          f"duration={duration}s, total_jobs={total_nodes}")
    return jobs

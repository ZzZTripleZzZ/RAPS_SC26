"""Mix A/B/C workloads (paper §6 sensitivity analysis).

Generates jobs that use the ``MATRIX_TEMPLATE`` communication pattern driven
by tiled proxy-app traffic matrices from ``data/matrices/``.  Each mix is
defined by the fraction of stencil-like vs. all-to-all-like jobs:

    Mix A: 80% stencil   + 20% all-to-all  ("stencil-dominant")
    Mix B: 20% stencil   + 80% all-to-all  ("all-to-all-dominant")
    Mix C: 100% stencil  (baseline used elsewhere in the paper)

Stencil templates are drawn from LULESH / HPGMG / CoMD; all-to-all templates
from CoSP2 / QuickSilver.  Templates are row-normalized and tiled to each
job's rank count via ``traffic_templates.tile_template``.
"""
from __future__ import annotations

import math
import random

from raps.job import CommunicationPattern, Job, job_dict
from raps.workloads.traffic_templates import get_template_for_job

from .constants import JOB_NAMES, ACCT_NAMES, MAX_PRIORITY


# Proxy apps grouped by dominant spatial pattern.
STENCIL_APPS = ("lulesh", "hpgmg", "comd")
A2A_APPS = ("cosp2", "quicksilver")


def _mix_apps(stencil_frac: float, num_jobs: int, rng: random.Random) -> list[str]:
    """Return a list of proxy-app names sized ``num_jobs`` matching the ratio."""
    n_stencil = int(round(num_jobs * stencil_frac))
    apps = []
    for _ in range(n_stencil):
        apps.append(rng.choice(STENCIL_APPS))
    for _ in range(num_jobs - n_stencil):
        apps.append(rng.choice(A2A_APPS))
    rng.shuffle(apps)
    return apps


def _build_template_job(
    *,
    job_id: int,
    proxy_app: str,
    nodes_required: int,
    submit_time: int,
    expected_run_time: int,
    trace_quanta: int,
    time_limit: int,
    cpu_util: float,
    gpu_util: float,
    tx_rate: float,
    config: dict | None = None,
) -> Job:
    """Construct a single Job using a tiled proxy-app template."""
    template = get_template_for_job(proxy_app, nodes_required)

    trace_len = max(1, expected_run_time // trace_quanta)
    cpu_trace = [cpu_util] * trace_len
    gpu_trace = [gpu_util] * trace_len
    ntx_trace = [tx_rate] * trace_len
    nrx_trace = [tx_rate] * trace_len

    info = job_dict(
        id=job_id,
        name=f"{proxy_app}_mix_{job_id}",
        account=random.choice(ACCT_NAMES),
        nodes_required=nodes_required,
        cpu_trace=cpu_trace,
        gpu_trace=gpu_trace,
        ntx_trace=ntx_trace,
        nrx_trace=nrx_trace,
        submit_time=submit_time,
        start_time=0,
        expected_run_time=expected_run_time,
        time_limit=time_limit,
        trace_quanta=trace_quanta,
        end_state="COMPLETED",
        comm_pattern=CommunicationPattern.MATRIX_TEMPLATE,
        traffic_template=template,
    )
    return Job(info)


class TemplateMixWorkload:
    """Three mix-ratio workloads for the workload-sensitivity experiment."""

    _DEFAULT_NUM_JOBS = 200
    _DEFAULT_RUN_SECS = 600       # 10 min nominal per job
    _DEFAULT_TIME_LIMIT = 3600    # 1 hr wall limit
    _DEFAULT_TRACE_QUANTA = 15    # Frontier default; overridden for other sys
    _DEFAULT_TX_RATE = 1.25e9     # 1.25 GB/s per rank per trace quanta (matches Frontier cfg)

    # Node-count distribution: rough power-of-two job-size mix used elsewhere.
    _JOB_SIZES = [16, 32, 64, 128, 192, 256, 512]
    _JOB_SIZE_WEIGHTS = [10, 15, 25, 20, 15, 10, 5]

    def _mix(self, stencil_frac: float, **kwargs) -> list[Job]:
        args = kwargs.get("args", None)
        seed = getattr(args, "seed", 42) if args is not None else 42
        num_jobs = getattr(args, "numjobs", self._DEFAULT_NUM_JOBS) or self._DEFAULT_NUM_JOBS

        rng = random.Random(seed)
        apps = _mix_apps(stencil_frac, num_jobs, rng)

        config = self.config_map.get(self.args.system) if hasattr(self, "config_map") else None
        trace_quanta = (
            int(config.get("TRACE_QUANTA", self._DEFAULT_TRACE_QUANTA))
            if config else self._DEFAULT_TRACE_QUANTA
        )

        # Arrival rate: stagger submissions so the queue actually matters.
        arrival_interval = 30  # seconds between submissions

        jobs: list[Job] = []
        submit_time = 0
        for i, app in enumerate(apps):
            nodes_required = rng.choices(self._JOB_SIZES, weights=self._JOB_SIZE_WEIGHTS, k=1)[0]
            # Give stencil jobs slightly shorter runs than a2a (mirrors proxy-app defaults).
            run_secs = int(rng.uniform(0.6, 1.4) * self._DEFAULT_RUN_SECS)
            cpu_util = rng.uniform(0.5, 0.9)
            gpu_util = rng.uniform(0.4, 0.8)
            tx_rate = self._DEFAULT_TX_RATE * rng.uniform(0.8, 1.2)
            jobs.append(_build_template_job(
                job_id=i + 1,
                proxy_app=app,
                nodes_required=nodes_required,
                submit_time=submit_time,
                expected_run_time=run_secs,
                trace_quanta=trace_quanta,
                time_limit=self._DEFAULT_TIME_LIMIT,
                cpu_util=cpu_util,
                gpu_util=gpu_util,
                tx_rate=tx_rate,
                config=config,
            ))
            submit_time += arrival_interval

        return jobs

    def template_mix_a(self, **kwargs) -> list[Job]:
        """Mix A: stencil-dominant (80% stencil, 20% all-to-all)."""
        return self._mix(stencil_frac=0.80, **kwargs)

    def template_mix_b(self, **kwargs) -> list[Job]:
        """Mix B: all-to-all-dominant (20% stencil, 80% all-to-all)."""
        return self._mix(stencil_frac=0.20, **kwargs)

    def template_mix_c(self, **kwargs) -> list[Job]:
        """Mix C: pure stencil baseline (100% stencil)."""
        return self._mix(stencil_frac=1.00, **kwargs)

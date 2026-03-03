#!/usr/bin/env python3
"""
Generate a dragonfly congestion chord diagram from a single RAPS simulation tick.

Runs a minimal Frontier simulation with 3 large concurrent all-to-all jobs,
computes one full tick of inter-job congestion, dumps the link loads, then
calls plot_dragonfly_congestion.py to produce the figure.

Usage:
    python scripts/run_frontier_congestion_snapshot.py
    python scripts/run_frontier_congestion_snapshot.py --output output/figures/congestion.png
    python scripts/run_frontier_congestion_snapshot.py --routing minimal --output congestion_minimal.png
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Allow running from any directory
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="Generate Frontier dragonfly congestion figure")
    p.add_argument("--routing", default="ugal",
                   choices=["minimal", "ugal", "valiant"],
                   help="Routing algorithm (default: ugal)")
    p.add_argument("--valiant-bias", type=float, default=0.3,
                   help="Fraction of traffic routed non-minimally for Valiant (default: 0.3)")
    p.add_argument("--output", default="output/figures/dragonfly_congestion.png")
    p.add_argument("--links-dir", default="output/links_snapshot",
                   help="Directory for link-load CSV files")
    p.add_argument("--title", default=None)
    return p.parse_args()



def run_snapshot(routing, links_dir, valiant_bias=0.3):
    """Run one tick of network simulation and dump link loads."""
    from raps.network import NetworkModel, simulate_inter_job_congestion, worst_link_util

    # Load frontier config with specified routing
    from raps.system_config import get_system_config
    system_cfg = get_system_config('frontier')
    config = system_cfg.get_legacy()
    config['ROUTING_ALGORITHM'] = routing
    config['VALIANT_BIAS'] = valiant_bias

    trace_quanta = config.get('TRACE_QUANTA', 15)
    max_bw = config.get('NETWORK_MAX_BW', 25e9)
    total_nodes = config.get('TOTAL_NODES', 9408)

    print(f"Frontier config: {total_nodes} nodes, routing={routing}")
    print(f"Dragonfly: D={config['DRAGONFLY_D']}, A={config['DRAGONFLY_A']}, P={config['DRAGONFLY_P']}")
    print(f"Trace quanta: {trace_quanta}s, BW: {max_bw/1e9:.0f} GB/s")

    # Build network model
    available_nodes = list(range(total_nodes))
    nm = NetworkModel(available_nodes=available_nodes, config=config)

    # Build 4 concurrent large jobs, each concentrated in a block of groups.
    # This produces visible per-group-pair congestion in the chord diagram.
    # ntx per node: matches UC heavy experiment (1.25 GB/s × trace_quanta)
    ntx_per_node = 1.25e9 * trace_quanta  # bytes per trace interval

    D = config['DRAGONFLY_D']   # routers per group
    P = config['DRAGONFLY_P']   # nodes per router
    nodes_per_group = D * P     # 192 for Frontier

    # Job design: 4 jobs all spanning the SAME 4 groups but using different router
    # subsets within each group.  Since all jobs send all-to-all traffic across the
    # same 4 group pairs, the 6 inter-group link pairs get loaded by multiple jobs.
    # UGAL sees these loaded direct links and reroutes through uncongested intermediate
    # groups → clearly different chord diagram from minimal routing.
    # Each job has 4 groups × 12 routers × P=4 nodes = 192 nodes → fast UGAL computation.
    SHARED_GROUPS = [0, 10, 20, 30]   # 4 well-separated groups
    ROUTER_SUBSETS = [                 # exclusive router ranges per group
        (0,  11),  # job 0: routers  0-11
        (12, 23),  # job 1: routers 12-23
        (24, 35),  # job 2: routers 24-35
        (36, 47),  # job 3: routers 36-47 (D-1=47)
    ]

    from raps.job import Job, job_dict, CommunicationPattern
    trace_len = 20
    jobs = []
    for jid, (r_start, r_end) in enumerate(ROUTER_SUBSETS):
        # Collect node IDs for routers r_start..r_end across all shared groups
        node_list = []
        for g in SHARED_GROUPS:
            for r in range(r_start, r_end + 1):
                for h in range(P):
                    node_id = g * nodes_per_group + r * P + h
                    if node_id < total_nodes:
                        node_list.append(node_id)
        g_start = SHARED_GROUPS[0]
        g_end   = SHARED_GROUPS[-1]
        info = job_dict(
            id=jid + 1,
            name=f"job_r{r_start}-{r_end}",
            account="gen053",
            nodes_required=len(node_list),
            scheduled_nodes=node_list,
            cpu_trace=[1.0] * trace_len,
            gpu_trace=[0.8] * trace_len,
            ntx_trace=[ntx_per_node] * trace_len,
            nrx_trace=[ntx_per_node] * trace_len,
            submit_time=0,
            start_time=0,
            expected_run_time=3600,
            time_limit=7200,
            end_state="COMPLETED",
            trace_quanta=trace_quanta,
        )
        job = Job(info)
        job.comm_pattern = CommunicationPattern.ALL_TO_ALL
        jobs.append(job)

    print(f"\nJobs: {len(jobs)} jobs, nodes: {[j.nodes_required for j in jobs]}")

    # Reset link loads and compute per-job utilisation
    nm.reset_link_loads()
    for job in jobs:
        nm.simulate_network_utilization(job=job)

    # Run inter-job congestion
    dragonfly_params = {
        'd': config['DRAGONFLY_D'],
        'a': config['DRAGONFLY_A'],
        'ugal_threshold': config.get('UGAL_THRESHOLD', 2.0),
        'valiant_bias': valiant_bias,
    }
    stats = simulate_inter_job_congestion(
        nm, jobs, config, debug=False,
        apsp=nm._apsp,
        job_coeffs_cache=nm._job_load_coeffs,
        routing_algorithm=routing,
        dragonfly_params=dragonfly_params,
    )
    if isinstance(stats, dict):
        peak_util = stats.get('max', 0)
        mean_util = stats.get('mean', 0)
    else:
        peak_util = mean_util = stats
    print(f"\nInter-job congestion:  peak={peak_util:.3f},  mean={mean_util:.3f}")

    # Dump link loads
    os.makedirs(links_dir, exist_ok=True)
    out_csv = os.path.join(links_dir, f"snapshot_{routing}.csv")
    nm.dump_link_loads(out_csv, dt=trace_quanta)
    print(f"Link loads saved → {out_csv}")
    return out_csv


def main():
    args = parse_args()

    # Run network snapshot
    csv_path = run_snapshot(args.routing, args.links_dir, args.valiant_bias)

    # Generate chord diagram
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    title = args.title or f"Frontier Dragonfly — {args.routing.upper()} routing"

    script = ROOT / "scripts" / "plot_dragonfly_congestion.py"
    cmd = [
        sys.executable, str(script),
        csv_path,
        "-s", str(ROOT / "config" / "frontier.yaml"),
        "--title", title,
        "-o", args.output,
    ]
    print(f"\nGenerating chord diagram …")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

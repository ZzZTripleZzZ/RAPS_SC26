#!/usr/bin/env python3
"""
Baseline/raps/run_engine_sweep.py
Validates RAPS via the actual simulate_inter_job_congestion code path.

Unlike run_sweep.py (pure analytical), this script builds a NetworkModel and
calls the same network simulation code that engine.py calls each tick.
This validates the full RAPS network simulation pipeline — not just the M/D/1
formula — at varying injection rates for dragonfly and fat-tree topologies.

All-to-all synthetic job, trace_quanta = dt = 10s, minimal routing.

Usage:
    .venv/bin/python3 Baseline/raps/run_engine_sweep.py [--nodes 1000] [--dt 10]

Outputs:
    Baseline/raps/output/{topo}_engine/rho_{X}.csv
    Baseline/raps/output/{topo}_engine/summary_{X}.json
"""

import argparse
import csv
import json
import os
import sys
import time
import types

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from raps.network.dragonfly import build_dragonfly
from raps.network.fat_tree import build_fattree
from raps.network import NetworkModel
from raps.network.base import compute_all_to_all_coefficients
from raps.job import CommunicationPattern

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
RHO_VALUES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

# Topology configs — (d=routers/group, a=global_links/router, p=hosts/router)
TOPO_CONFIGS = {
    72:    dict(d=4,  a=8,  p=2,  bw=25e9,   topo="dragonfly", label="9g×4r×2h"),
    1000:  dict(d=10, a=9,  p=10, bw=25e9,   topo="dragonfly", label="10g×10r×10h"),
    10000: dict(d=10, a=9,  p=100,bw=25e9,   topo="dragonfly", label="10g×10r×100h"),
    16:    dict(k=4,        bw=12.5e9, topo="fat-tree",  label="k=4"),
}

ZERO_LOAD_LATENCY_NS = {"dragonfly": 350.0, "fat-tree": 450.0}


# ---------------------------------------------------------------------------
# Mock Job (lightweight — only the fields simulate_inter_job_congestion needs)
# ---------------------------------------------------------------------------
class SynthJob:
    """Minimal synthetic job for network simulation validation.

    Attributes must match what NetworkModel.simulate_network_utilization() reads:
      - nodes_required: must be > 1 (or single-node jobs are skipped)
      - ntx_trace / nrx_trace: constant float (bytes per trace_quanta per node)
      - trace_quanta: time step in seconds
      - scheduled_nodes: list of integer node indices (mapped via real_to_fat_idx)
      - comm_pattern: CommunicationPattern enum
      - current_run_time / trace_start_time: used by get_current_utilization
    """
    _id_counter = 0

    def __init__(self, node_indices: list, ntx_bytes: float, trace_quanta: int):
        SynthJob._id_counter += 1
        self.id = f"synth_{SynthJob._id_counter}"
        self.scheduled_nodes = node_indices     # list of ints 0..N-1
        self.nodes_required = len(node_indices)
        self.ntx_trace = ntx_bytes          # constant single float value
        self.nrx_trace = ntx_bytes
        self.trace_quanta = trace_quanta
        self.comm_pattern = CommunicationPattern.ALL_TO_ALL
        self.current_run_time = 0
        self.trace_start_time = 0
        self.message_size = None
        self.message_overhead_bytes = None
        self.dilated = False
        self.slowdown_factor = 1.0
        self.stall_ratio = 0.0


# ---------------------------------------------------------------------------
# Build topology + NetworkModel
# ---------------------------------------------------------------------------

def make_dragonfly_config(d: int, a: int, p: int, bw: float) -> dict:
    num_groups = a + 1
    total_nodes = num_groups * d * p
    return {
        "TOPOLOGY": "dragonfly",
        "DRAGONFLY_D": d,
        "DRAGONFLY_A": a,
        "DRAGONFLY_P": p,
        "TOTAL_NODES": total_nodes,
        "NETWORK_MAX_BW": bw,
        "ROUTING_ALGORITHM": "minimal",
    }


def make_fattree_config(k: int, bw: float) -> dict:
    return {
        "TOPOLOGY": "fat-tree",
        "FATTREE_K": k,
        "TOTAL_NODES": k**3 // 4,
        "DOWN_NODES": [],
        "NETWORK_MAX_BW": bw,
        "ROUTING_ALGORITHM": "minimal",
    }


def build_net_model(cfg: dict) -> tuple:
    """Returns (NetworkModel, host_list, max_coeff)."""
    topo = cfg.get("topo") or cfg.get("TOPOLOGY")

    if topo == "dragonfly":
        if "d" in cfg:  # suite-style config
            legacy = make_dragonfly_config(cfg["d"], cfg["a"], cfg["p"], cfg["bw"])
        else:
            legacy = cfg

        n_total = legacy["TOTAL_NODES"]
        model = NetworkModel(available_nodes=n_total, config=legacy)
        # Get host names in topology order via real_to_fat_idx
        hosts = [model.real_to_fat_idx[i] for i in range(n_total)]
    else:
        if "k" in cfg:
            legacy = make_fattree_config(cfg["k"], cfg["bw"])
        else:
            legacy = cfg
        n_total = legacy["TOTAL_NODES"]
        model = NetworkModel(available_nodes=n_total, config=legacy)
        hosts = [n for n in model.net_graph.nodes()
                 if str(n).startswith("h_")]
        hosts = sorted(hosts)[:n_total]

    # Pre-compute max all-to-all coefficient (for injection rate scaling)
    if len(hosts) <= 2000:
        print(f"  Pre-computing A2A coefficients for {len(hosts)} hosts...", end="", flush=True)
        t0 = time.time()
        coeffs = compute_all_to_all_coefficients(model.net_graph, hosts)
        max_coeff = max(coeffs.values()) if coeffs else 1.0
        print(f" done ({time.time()-t0:.1f}s), max_coeff={max_coeff:.4f}")
    else:
        # For very large N, estimate max_coeff from a small sample
        sample = hosts[:200]
        coeffs_s = compute_all_to_all_coefficients(model.net_graph, sample)
        max_coeff = max(coeffs_s.values()) if coeffs_s else 1.0
        print(f"  max_coeff estimate (200-host sample)={max_coeff:.4f}")

    return model, hosts, max_coeff, legacy


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_engine_sweep(topo_label: str, cfg: dict, dt: int, out_dir: str):
    """Run the RAPS engine network sweep at all injection rates."""
    os.makedirs(out_dir, exist_ok=True)

    topo_type = cfg.get("topo", "dragonfly")
    bw = cfg.get("bw", 25e9)
    if topo_type == "dragonfly":
        n_total = (cfg["a"] + 1) * cfg["d"] * cfg["p"]
    else:
        n_total = cfg["k"]**3 // 4

    print(f"\n[RAPS-engine] {topo_label}: N={n_total}, dt={dt}s, BW={bw/1e9:.1f}GB/s")

    model, hosts, max_coeff, legacy_cfg = build_net_model(cfg)
    # scheduled_nodes must be integer indices 0..N-1 (NetworkModel maps them via real_to_fat_idx)
    node_indices = list(range(n_total))
    max_throughput = bw * dt   # bytes per tick (= max_throughput_per_tick with trace_quanta=dt)

    zero_load_ns = ZERO_LOAD_LATENCY_NS.get(topo_type, 400.0)
    job_coeffs_cache = {}
    results = []

    for rho in RHO_VALUES:
        # Set ntx_trace so hottest link reaches utilization=rho
        ntx = rho * max_throughput / max_coeff

        job = SynthJob(node_indices=node_indices, ntx_bytes=ntx, trace_quanta=dt)

        # Call the same method the RAPS engine calls each tick
        net_util, net_cong, net_tx, net_rx, max_tp = model.simulate_network_utilization(job=job)

        # Congestion = worst-link utilization fraction
        congestion = net_cong     # max link load / max_throughput
        mean_util  = net_util     # mean of (tx_util + rx_util) / 2 (global)

        # Reconstruct max link utilization from congestion
        max_util = min(congestion, 1.0)   # congestion can exceed 1.0 when overloaded

        # M/D/1 slowdown applied by RAPS when 0.05 < congestion < 1.0
        if 0.05 < congestion < 1.0:
            slowdown = 1.0 + congestion**2 / (2.0 * (1.0 - congestion))
        elif congestion >= 1.0:
            slowdown = congestion           # RAPS uses net_cong directly
        else:
            slowdown = 1.0
        stall_ratio = max(0.0, slowdown - 1.0)
        avg_latency_ns = zero_load_ns * slowdown

        summary = {
            "simulator": "raps_engine",
            "topology": topo_label,
            "rho_target": rho,
            "N_hosts": n_total,
            "dt_s": dt,
            "trace_quanta_s": dt,
            "link_bandwidth_gbps": bw / 1e9,
            "ntx_bytes_per_tick": ntx,
            "max_coeff": max_coeff,
            "mean_utilization": mean_util,
            "mean_utilization_pct": mean_util * 100.0,
            "max_utilization": max_util,
            "max_utilization_pct": max_util * 100.0,
            "congestion": congestion,
            "slowdown": slowdown,
            "stall_ratio": stall_ratio,
            "avg_latency_ns": avg_latency_ns,
            "zero_load_latency_ns": zero_load_ns,
            "model": "raps_engine+M/D/1",
        }
        results.append(summary)

        rho_str = f"{rho:.2f}".replace(".", "p")
        json_path = os.path.join(out_dir, f"summary_{rho_str}.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"  ρ={rho:.2f}: mean_util={mean_util*100:.1f}% max_util={max_util*100:.1f}%  "
              f"congestion={congestion:.4f}  slowdown={slowdown:.3f}  stall={stall_ratio:.3f}")

    # Write summary CSV
    csv_path = os.path.join(out_dir, "sweep_results.csv")
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
    print(f"  CSV: {csv_path}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAPS engine network validation sweep")
    parser.add_argument("--nodes", type=int, choices=[72, 1000, 10000, 16], default=1000,
                        help="Number of hosts (default: 1000)")
    parser.add_argument("--dt", type=int, default=10,
                        help="Simulation time step in seconds (default: 10)")
    parser.add_argument("--topology", choices=["dragonfly", "fattree", "all"], default="dragonfly")
    parser.add_argument("--large", action="store_true",
                        help="Include N=10000 (slower)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    nodes_to_run = [args.nodes] if args.nodes else list(TOPO_CONFIGS.keys())

    for n, cfg in TOPO_CONFIGS.items():
        if args.nodes and n != args.nodes:
            continue
        if n >= 10000 and not args.large:
            continue
        if args.topology == "dragonfly" and cfg["topo"] != "dragonfly":
            continue
        if args.topology == "fattree" and cfg["topo"] != "fat-tree":
            continue

        label = f"{cfg['topo']}_{n}n"
        out_dir = os.path.join(base_dir, "output", f"{label}_engine")
        run_engine_sweep(f"{label}_{cfg['label']}", cfg, args.dt, out_dir)

    print("\n[RAPS-engine] Done.")


if __name__ == "__main__":
    main()

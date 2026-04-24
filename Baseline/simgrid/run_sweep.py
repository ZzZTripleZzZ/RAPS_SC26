#!/usr/bin/env python3
"""
SimGrid Baseline Sweep: link utilization vs. injection rate (ρ).

Implements SimGrid's flow-level max-min fairness bandwidth-sharing model
analytically using RAPS topology coefficients.

SimGrid is a FLOW-LEVEL simulator (not packet-level). Its congestion model:
  - Compute per-link bandwidth demand from all active flows
  - If any link is overloaded: throttle flows proportionally (max-min fairness)
  - No queuing delay — completion time comes purely from bandwidth allocation
  - Slowdown = requested_rate / allocated_rate (≥ 1 when congested)

For all-to-all uniform traffic at injection rate ρ × link_bw per host:
  - Load on link l = ρ × coeff(l)  where coeff = load factor from RAPS routing
  - Bottleneck link load = ρ × max_coeff
  - When ρ × max_coeff ≤ 1: all flows get requested rate, slowdown = 1.0
  - When ρ × max_coeff > 1: bottleneck saturates, flows throttled
    → slowdown = ρ × max_coeff  (bandwidth throttling, linear with ρ)

This is the SAME mathematical model as SimGrid's flow-level network simulation
(SimGrid's SURF network model with max-min fairness / HTB).
Results differ from M/D/1 (RAPS) because SimGrid has no queuing delay:
  - SimGrid: slowdown = 1 for ρ < ρ_sat, linear after
  - M/D/1:   slowdown = 1 + ρ²/(2(1-ρ)), starts from ρ=0

Topologies (using RAPS topology builders for exact match):
  - Dragonfly 72n:   9g × 4r × 2h = 72 hosts, 25 GB/s, minimal routing
  - Dragonfly 1000n: 10g × 10r × 10h = 1000 hosts, 25 GB/s, minimal routing
  - Fat-tree k=4:    16 hosts, 12.5 GB/s, ECMP routing

Usage:
    cd /lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26
    .venv/bin/python3 Baseline/simgrid/run_sweep.py [--topo dragonfly|dragonfly_1000n|fattree|all]

Outputs:
    Baseline/simgrid/output/{topo}/summary_{rho}.json
    Baseline/simgrid/output/{topo}/sweep_results.csv
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_SCRIPT_DIR)
_RAPS_ROOT = os.path.dirname(_BASE_DIR)
sys.path.insert(0, _RAPS_ROOT)

RHO_VALUES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

ZERO_LOAD_LATENCY_NS = {
    "dragonfly":       350.0,   # 3 hops × 100ns + 50ns injection
    "dragonfly_1000n": 350.0,
    "fattree":         450.0,   # 4 hops × 100ns + 50ns injection
    "torus3d":        1250.0,   # DOR avg ~12 hops × 100ns + 50ns injection
}


# ---------------------------------------------------------------------------
# Core: compute RAPS all-to-all coefficients for a topology
# ---------------------------------------------------------------------------

def get_topology_coeffs(topo_name: str, params: dict):
    """
    Build topology graph via RAPS and return (max_coeff, mean_coeff, N_hosts).
    Coefficients are cached after first call.
    """
    from raps.network.base import compute_all_to_all_coefficients

    if params["kind"] == "dragonfly":
        from raps.network.dragonfly import build_dragonfly
        G_net = build_dragonfly(
            d=params["routers_per_group"],
            a=params["groups"] - 1,
            p=params["hosts_per_router"],
        )
        hosts = [n for n in G_net.nodes() if G_net.nodes[n].get("layer") == "host"]
        if not hosts:
            hosts = [n for n in G_net.nodes() if str(n).startswith("h")]
        N = params["groups"] * params["routers_per_group"] * params["hosts_per_router"]
    elif params["kind"] == "torus3d":
        from raps.network.torus3d import build_torus3d, link_loads_for_job_torus
        dims = params["dims"]
        hpr = params["hpr"]
        G_net, meta = build_torus3d(dims, hosts_per_router=hpr)
        hosts = [n for n in G_net.nodes() if G_net.nodes[n].get("type") == "host"]
        if not hosts:
            hosts = [n for n in G_net.nodes() if str(n).startswith("h_")]
        N = dims[0] * dims[1] * dims[2] * hpr
        # For torus, use link_loads_for_job_torus with unit traffic to get coefficients
        unit_loads = link_loads_for_job_torus(G_net, meta, hosts, 1.0)
        vals = list(unit_loads.values()) if unit_loads else [1.0]
        max_coeff = float(max(vals))
        mean_coeff = float(np.mean(vals))
        return max_coeff, mean_coeff, N
    else:  # fat-tree
        from raps.network.fat_tree import build_fattree
        k = params["k"]
        N = k ** 3 // 4
        G_net = build_fattree(k=k, total_nodes=N)
        hosts = [n for n in G_net.nodes() if G_net.nodes[n].get("type") == "host"]
        if not hosts:
            hosts = [n for n in G_net.nodes() if str(n).startswith("h")]

    coeffs = compute_all_to_all_coefficients(G_net, hosts)
    if not coeffs:
        raise RuntimeError(f"No coefficients computed for {topo_name}")

    vals = list(coeffs.values())
    max_coeff = float(max(vals))
    mean_coeff = float(np.mean(vals))
    return max_coeff, mean_coeff, N


# ---------------------------------------------------------------------------
# SimGrid max-min fairness model
# ---------------------------------------------------------------------------

def simgrid_maxmin(topo_name: str, params: dict, rho: float,
                   max_coeff: float, mean_coeff: float, N: int) -> dict:
    """
    Compute SimGrid flow-level max-min fairness result for a given ρ.

    Key difference from M/D/1:
    - SimGrid: bandwidth throttling only, NO queuing delay
    - Slowdown = 1.0 when network is feasible (ρ × max_coeff < 1)
    - Slowdown = ρ × max_coeff when bottleneck saturates (bandwidth throttling)

    This produces a PIECEWISE LINEAR curve vs ρ, unlike M/D/1's quadratic curve.
    """
    link_bw = params["link_bw"]

    # Link utilization
    bottleneck_util = rho * max_coeff   # utilization of the most loaded link
    mean_util = rho * mean_coeff        # mean across all links

    # Bandwidth throttling under max-min:
    # When bottleneck saturates, all flows through it are throttled equally.
    # Allocated rate fraction = 1 / (rho * max_coeff) when overloaded.
    if bottleneck_util <= 1.0:
        # All flows feasible: no bandwidth throttling
        # SimGrid flow-level: slowdown = 1.0 (pure bandwidth, no queue)
        slowdown = 1.0
        stall_ratio = 0.0
    else:
        # Bottleneck saturated: flows throttled by factor 1/(rho * max_coeff)
        slowdown = bottleneck_util   # = rho * max_coeff
        stall_ratio = slowdown - 1.0

    zero_load_ns = ZERO_LOAD_LATENCY_NS.get(topo_name, 350.0)
    avg_latency_ns = zero_load_ns * slowdown

    return {
        "mean_utilization": min(mean_util, 1.0),
        "max_utilization": min(bottleneck_util, 1.0),
        "bottleneck_util_raw": bottleneck_util,   # may exceed 1 when overloaded
        "slowdown": slowdown,
        "stall_ratio": stall_ratio,
        "avg_latency_ns": avg_latency_ns,
        "max_coeff": max_coeff,
        "mean_coeff": mean_coeff,
        "rho_saturation": 1.0 / max_coeff,   # ρ at which bottleneck saturates
        "model": "simgrid_maxmin_bandwidth",
    }


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

TOPO_CONFIGS = {
    "dragonfly": dict(
        groups=9, routers_per_group=4, hosts_per_router=2,
        link_bw=25e9, name="dragonfly", kind="dragonfly",
    ),
    "dragonfly_1000n": dict(
        groups=33, routers_per_group=8, hosts_per_router=4,
        link_bw=25e9, name="dragonfly_1000n", kind="dragonfly",
    ),
    "fattree": dict(
        k=4, link_bw=12.5e9, name="fattree", kind="fattree",
    ),
    "torus3d": dict(
        dims=(8, 8, 8), hpr=2, link_bw=9.6e9, name="torus3d", kind="torus3d",
    ),
}


def run_sweep(topo_name: str, out_dir: str):
    params = TOPO_CONFIGS[topo_name]
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[SimGrid max-min] {topo_name} sweep...")

    t_coeff = time.perf_counter()
    print(f"  Computing topology coefficients...", end="", flush=True)
    max_coeff, mean_coeff, N = get_topology_coeffs(topo_name, params)
    print(f" done ({time.perf_counter()-t_coeff:.1f}s)  "
          f"N={N}, max_coeff={max_coeff:.4f}, ρ_sat={1/max_coeff:.3f}")

    rows = []
    for rho in RHO_VALUES:
        rho_str = f"{rho:.2f}".replace(".", "p")
        result = simgrid_maxmin(topo_name, params, rho, max_coeff, mean_coeff, N)

        summary = {
            "simulator": "simgrid",
            "topology": topo_name,
            "rho_target": rho,
            "N_hosts": N,
            "link_bandwidth_gbps": params["link_bw"] / 1e9,
            **result,
        }
        json_path = os.path.join(out_dir, f"summary_{rho_str}.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        rows.append(summary)
        sat_marker = " ← SAT" if result["bottleneck_util_raw"] >= 1.0 else ""
        print(f"  ρ={rho:.2f}: util={result['max_utilization']*100:.1f}%  "
              f"slowdown={result['slowdown']:.4f}  stall={result['stall_ratio']:.4f}{sat_marker}")

    # Write sweep CSV
    csv_path = os.path.join(out_dir, "sweep_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {csv_path}")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="SimGrid max-min fairness sweep (flow-level bandwidth model)"
    )
    parser.add_argument("--topo", default="all",
                        choices=list(TOPO_CONFIGS) + ["all"],
                        help=f"Topology to sweep (default: all). Choices: {list(TOPO_CONFIGS)}")
    args = parser.parse_args()

    topos = list(TOPO_CONFIGS) if args.topo == "all" else [args.topo]
    for topo in topos:
        out_dir = os.path.join(_SCRIPT_DIR, "output", topo)
        run_sweep(topo, out_dir)

    print("\n[SimGrid max-min] All sweeps complete.")
    print("Note: SimGrid flow-level model = bandwidth throttling only (no queuing delay).")
    print("      Slowdown = 1.0 for ρ < ρ_sat, then linear increase after saturation.")


if __name__ == "__main__":
    main()

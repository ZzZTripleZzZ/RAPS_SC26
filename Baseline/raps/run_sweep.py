#!/usr/bin/env python3
"""
RAPS Baseline Sweep: link utilization vs. injection rate (ρ).

Computes per-link utilization and M/D/1 slowdown analytically for the
reference dragonfly (9g×4r×2h, 25 GB/s) and fat-tree (k=4, 12.5 GB/s)
topologies under all-to-all uniform traffic at injection rates ρ ∈ {0.05..0.8}.

No SLURM job required — runs locally in under a minute.

Usage:
    cd /lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26
    .venv/bin/python3 Baseline/raps/run_sweep.py

Outputs:
    Baseline/raps/output/dragonfly/rho_{X}.csv    per-link utilization
    Baseline/raps/output/dragonfly/summary_{X}.json  aggregate metrics
    Baseline/raps/output/fattree/rho_{X}.csv
    Baseline/raps/output/fattree/summary_{X}.json
"""

import csv
import json
import os
import sys
import time

import networkx as nx
import numpy as np

# Ensure RAPS root is on path
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from raps.network.dragonfly import build_dragonfly
from raps.network.fat_tree import build_fattree
from raps.network.torus3d import build_torus3d, link_loads_for_job_torus
from raps.network.base import compute_all_to_all_coefficients

# ---------------------------------------------------------------------------
# Reference topology parameters (must match Baseline/configs/dragonfly_ref.yaml
# and fattree_ref.yaml)
# ---------------------------------------------------------------------------

# Topology suite — run all of these
DRAGONFLY_SUITE = [
    dict(d=4,  a=8,  p=2,   bw=25e9, name="dragonfly",       label="72-node  (9g×4r×2h)"),
    dict(d=8,  a=32, p=4,   bw=25e9, name="dragonfly_1000n",  label="1056-node (33g×8r×4h, matches BookSim2 k=4)"),
    dict(d=10, a=9,  p=100, bw=25e9, name="dragonfly_10000n", label="10000-node (10g×10r×100h)"),
]

FATTREE_SUITE = [
    dict(k=4,  bw=12.5e9, name="fattree",       label="k=4 (16 hosts)"),
]

# 3D torus (BlueWaters reference: 8×8×8×2 = 1024 hosts, 9.6 GB/s)
TORUS3D_SUITE = [
    dict(dims=(8, 8, 8), hpr=2, bw=9.6e9, name="torus3d", label="8×8×8×2 (1024 hosts)"),
]

# Legacy single-topology aliases (used by run_engine_sweep.py)
DRAGONFLY_PARAMS = DRAGONFLY_SUITE[0]
FATTREE_PARAMS   = FATTREE_SUITE[0]

# Injection rate sweep (fraction of link bandwidth)
RHO_VALUES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

# Simulation time window (seconds) — used to convert bytes to utilization
DT = 1.0  # 1-second tick

# Zero-load latency estimates (ns) for stall_ratio calculation
# Dragonfly minimal: avg ~3 hops × 100ns/hop + 50ns injection = 350 ns
# Fat-tree ECMP:     avg ~4 hops × 100ns/hop + 50ns injection = 450 ns
ZERO_LOAD_LATENCY_NS = {
    "dragonfly":        350.0,
    "dragonfly_1000n":  350.0,
    "dragonfly_10000n": 350.0,
    "fattree":          450.0,
    # Torus3d DOR: avg path ~(8/2)×3 = 12 hops, 100ns/hop + 50ns inject = 1250ns
    "torus3d":         1250.0,
}


# ---------------------------------------------------------------------------
# Topology builders
# ---------------------------------------------------------------------------

def build_dragonfly_topology(params):
    """Build reference dragonfly graph and return (G, host_list)."""
    G = build_dragonfly(d=params["d"], a=params["a"], p=params["p"])
    hosts = [n for n, attr in G.nodes(data=True) if attr.get("layer") == "host"]
    # Fall back to heuristic if 'layer' attribute not set
    if not hosts:
        hosts = [n for n in G.nodes() if str(n).startswith("h_")]
    hosts = sorted(hosts)
    return G, hosts


def build_fattree_topology(params):
    """Build reference fat-tree graph and return (G, host_list)."""
    k = params["k"]
    num_hosts = k**3 // 4
    G = build_fattree(k=k, total_nodes=num_hosts)
    hosts = [n for n in G.nodes() if str(n).startswith("h_")]
    hosts = sorted(hosts)
    return G, hosts


def build_torus3d_topology(params):
    """Build reference torus3d graph and return (G, meta, host_list, dims, hpr)."""
    dims = params["dims"]
    hpr = params["hpr"]
    G, meta = build_torus3d(dims, hosts_per_router=hpr)
    hosts = [n for n in G.nodes() if G.nodes[n].get("type") == "host"]
    if not hosts:
        hosts = [n for n in G.nodes() if str(n).startswith("h_")]
    hosts = sorted(hosts)
    return G, meta, hosts, dims, hpr


# ---------------------------------------------------------------------------
# M/D/1 slowdown model (same formula used in RAPS engine.py)
# ---------------------------------------------------------------------------

def md1_slowdown(rho: float) -> float:
    """
    M/D/1 queue slowdown factor.
    Valid for rho in (0, 1). At rho >= 1 returns infinity (saturated).
    """
    if rho <= 0.0:
        return 1.0
    if rho >= 1.0:
        return float("inf")
    return 1.0 + rho**2 / (2.0 * (1.0 - rho))


def md1_stall_ratio(rho: float) -> float:
    """Stall ratio = slowdown - 1 = rho^2 / (2*(1-rho))."""
    if rho <= 0.0:
        return 0.0
    if rho >= 1.0:
        return float("inf")
    return rho**2 / (2.0 * (1.0 - rho))


# ---------------------------------------------------------------------------
# Core sweep function
# ---------------------------------------------------------------------------

def run_topology_sweep(topo_name: str, G: nx.Graph, hosts: list, bw: float, out_dir: str):
    """
    For each ρ in RHO_VALUES:
      1. Set tx_volume so that the most-loaded link achieves utilization = ρ.
      2. Compute per-link loads via all-to-all coefficient formula.
      3. Save per-link CSV and summary JSON.

    Args:
        topo_name: "dragonfly" or "fattree"
        G:         NetworkX topology graph
        hosts:     sorted list of host node names
        bw:        link bandwidth in bytes/second
        out_dir:   output directory (Baseline/raps/output/{topo_name}/)
    """
    os.makedirs(out_dir, exist_ok=True)
    N = len(hosts)
    print(f"\n[RAPS] {topo_name}: {N} hosts, {G.number_of_edges()} links, BW={bw/1e9:.1f} GB/s")

    t0 = time.time()
    print(f"  Computing all-to-all coefficients for {N} hosts … ", end="", flush=True)
    coeffs = compute_all_to_all_coefficients(G, hosts)
    t1 = time.time()
    print(f"done ({t1-t0:.1f}s)")

    if not coeffs:
        print("  ERROR: No coefficients computed — check topology/host names")
        return

    max_coeff = max(coeffs.values())
    mean_coeff = np.mean(list(coeffs.values()))
    print(f"  Links with non-zero load: {len(coeffs)} / {G.number_of_edges()}")
    print(f"  Max coefficient: {max_coeff:.4f}, Mean: {mean_coeff:.4f}")

    zero_load_ns = ZERO_LOAD_LATENCY_NS[topo_name]

    for rho in RHO_VALUES:
        # Scale tx_volume so the worst link reaches exactly rho utilization
        max_throughput = bw * DT  # bytes in one DT window
        tx_volume = rho * max_throughput / max_coeff  # bytes per host

        # Per-link byte loads and utilizations
        link_rows = []
        utils = []
        for edge, coeff in coeffs.items():
            load_bytes = coeff * tx_volume
            util = load_bytes / max_throughput
            utils.append(util)
            src, dst = edge
            link_rows.append({
                "src": str(src),
                "dst": str(dst),
                "load_bytes": load_bytes,
                "utilization_pct": util * 100.0,
            })

        utils_arr = np.array(utils)
        mean_util = float(np.mean(utils_arr))
        max_util  = float(np.max(utils_arr))

        # M/D/1 slowdown based on the worst-case link utilization (= rho by construction)
        slowdown = md1_slowdown(rho)
        stall    = md1_stall_ratio(rho)

        # Estimated latency: zero_load × slowdown
        avg_latency_ns = zero_load_ns * slowdown

        # Write per-link CSV
        rho_str = f"{rho:.2f}".replace(".", "p")
        csv_path = os.path.join(out_dir, f"rho_{rho_str}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["src", "dst", "load_bytes", "utilization_pct"])
            writer.writeheader()
            writer.writerows(link_rows)

        # Write summary JSON
        summary = {
            "simulator": "raps",
            "topology": topo_name,
            "rho_target": rho,
            "N_hosts": N,
            "N_links": G.number_of_edges(),
            "N_links_loaded": len(coeffs),
            "link_bandwidth_gbps": bw / 1e9,
            "tx_volume_bytes_per_host": tx_volume,
            "mean_utilization": mean_util,
            "mean_utilization_pct": mean_util * 100.0,
            "max_utilization": max_util,
            "max_utilization_pct": max_util * 100.0,
            "slowdown": slowdown,
            "stall_ratio": stall,
            "avg_latency_ns": avg_latency_ns,
            "zero_load_latency_ns": zero_load_ns,
            "model": "M/D/1 queue (analytical)",
        }
        json_path = os.path.join(out_dir, f"summary_{rho_str}.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"  ρ={rho:.2f}: max_util={max_util*100:.1f}%  slowdown={slowdown:.3f}  "
              f"stall={stall:.3f}  latency={avg_latency_ns:.0f}ns → {csv_path}")

    print(f"  All ρ values done in {time.time()-t0:.1f}s total")


def run_torus3d_sweep(topo_name: str, G: nx.Graph, meta: dict, hosts: list, dims: tuple,
                      hpr: int, bw: float, out_dir: str):
    """
    Torus3d sweep using link_loads_for_job_torus (per-host byte injection).
    Computes max_coeff from link_loads under per-host traffic = 1 byte.
    """
    os.makedirs(out_dir, exist_ok=True)
    N = len(hosts)
    print(f"\n[RAPS] {topo_name}: {N} hosts, dims={dims}, hpr={hpr}, BW={bw/1e9:.1f} GB/s")

    t0 = time.time()
    print(f"  Computing torus3d all-to-all coefficients... ", end="", flush=True)

    # Use link_loads_for_job_torus with 1 byte per host → gives coefficient per link
    # (same as compute_all_to_all_coefficients but for torus DOR routing)
    unit_loads = link_loads_for_job_torus(G, meta, hosts, 1.0)
    t1 = time.time()
    print(f"done ({t1-t0:.1f}s)")

    if not unit_loads:
        print("  ERROR: No link loads computed — check torus3d topology/host names")
        return

    vals = list(unit_loads.values())
    max_coeff = max(vals)
    mean_coeff = np.mean(vals)
    print(f"  Links with non-zero load: {len(unit_loads)} / {G.number_of_edges()}")
    print(f"  Max coefficient: {max_coeff:.4f}, Mean: {mean_coeff:.4f}")
    print(f"  ρ_sat = 1/max_coeff = {1.0/max_coeff:.4f}")

    zero_load_ns = ZERO_LOAD_LATENCY_NS[topo_name]

    for rho in RHO_VALUES:
        max_throughput = bw * DT  # bytes in one DT window
        # Scale so max link = rho: tx_volume × max_coeff = rho × max_throughput
        tx_volume = rho * max_throughput / max_coeff  # bytes per host

        # Per-link utilizations
        utils = [coeff * tx_volume / max_throughput for coeff in vals]
        utils_arr = np.array(utils)
        mean_util = float(np.mean(utils_arr))
        max_util = float(np.max(utils_arr))

        slowdown = md1_slowdown(rho)
        stall = md1_stall_ratio(rho)
        avg_latency_ns = zero_load_ns * slowdown

        rho_str = f"{rho:.2f}".replace(".", "p")
        summary = {
            "simulator": "raps",
            "topology": topo_name,
            "rho_target": rho,
            "N_hosts": N,
            "N_links": G.number_of_edges(),
            "N_links_loaded": len(unit_loads),
            "link_bandwidth_gbps": bw / 1e9,
            "tx_volume_bytes_per_host": tx_volume,
            "mean_utilization": mean_util,
            "mean_utilization_pct": mean_util * 100.0,
            "max_utilization": max_util,
            "max_utilization_pct": max_util * 100.0,
            "slowdown": slowdown,
            "stall_ratio": stall,
            "avg_latency_ns": avg_latency_ns,
            "zero_load_latency_ns": zero_load_ns,
            "max_coeff": max_coeff,
            "model": "M/D/1 queue (analytical)",
        }
        json_path = os.path.join(out_dir, f"summary_{rho_str}.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"  ρ={rho:.2f}: max_util={max_util*100:.1f}%  slowdown={slowdown:.3f}  "
              f"stall={stall:.3f}  latency={avg_latency_ns:.0f}ns")

    print(f"  All ρ values done in {time.time()-t0:.1f}s total")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAPS analytical baseline sweep")
    parser.add_argument("--topology", choices=["dragonfly", "fattree", "torus3d", "all"], default="all")
    parser.add_argument("--nodes", type=int, default=0,
                        help="Filter to specific node count (0=run all)")
    parser.add_argument("--large", action="store_true",
                        help="Include large (10000-node) topologies (slower)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    run_df  = args.topology in ("dragonfly", "all")
    run_ft  = args.topology in ("fattree", "all")
    run_t3d = args.topology in ("torus3d", "all")

    if run_df:
        for params in DRAGONFLY_SUITE:
            n = params["d"] * (params["a"] + 1) * params["p"]
            if args.nodes and n != args.nodes:
                continue
            if n >= 10000 and not args.large:
                print(f"  Skipping {params['label']} (use --large to include)")
                continue
            G, hosts = build_dragonfly_topology(params)
            out = os.path.join(base_dir, "output", params["name"])
            run_topology_sweep(params["name"], G, hosts, params["bw"], out)

    if run_ft:
        for params in FATTREE_SUITE:
            G, hosts = build_fattree_topology(params)
            out = os.path.join(base_dir, "output", params["name"])
            run_topology_sweep(params["name"], G, hosts, params["bw"], out)

    if run_t3d:
        for params in TORUS3D_SUITE:
            G, meta, hosts, dims, hpr = build_torus3d_topology(params)
            out = os.path.join(base_dir, "output", params["name"])
            run_torus3d_sweep(params["name"], G, meta, hosts, dims, hpr, params["bw"], out)

    print("\n[RAPS] Analytical sweep complete.")


if __name__ == "__main__":
    main()

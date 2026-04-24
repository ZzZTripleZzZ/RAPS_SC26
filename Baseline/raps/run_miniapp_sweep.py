#!/usr/bin/env python3
"""
RAPS Mini-App Baseline Sweep: halo3d-equivalent stencil on torus3d.

Matches SST-Macro halo3d-26 scenario exactly:
  - Topology: 3D torus 8×8×8 = 512 nodes (no concentration)
  - Traffic: STENCIL_3D (6 face neighbors, periodic BC)
  - Vary injection rate via per-node tx_volume
  - Metrics: slowdown, avg_latency_ns, max link utilization

The 6-neighbor STENCIL_3D captures 96% of halo3d-26 traffic
(face exchanges dominate edge/vertex).

Usage:
    cd /lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26
    .venv/bin/python3 Baseline/raps/run_miniapp_sweep.py
"""

import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from raps.network.torus3d import build_torus3d, link_loads_for_job_torus, halo3d_26_pairs
from raps.job import CommunicationPattern

# Must match SST-Macro halo3d config exactly
TORUS_DIMS = (8, 8, 8)
HOSTS_PER_ROUTER = 1  # BookSim2 torus has concentration=1; match that
LINK_BW = 9.6e9       # bytes/s (BlueWaters Gemini)
ZERO_LOAD_NS = 1250.0 # avg ~12 hops × 100ns + 50ns injection

# halo3d-26: nx=ny=nz=50, vars=1, MPI_DOUBLE=8 bytes
# Per-iteration per-rank: 6 faces(120000) + 12 edges(4800) + 8 vertices(64) = 124864 bytes
HALO3D_BYTES_PER_ITER = 124864

# Sweep: vary the number of iterations (= vary total injection volume)
# This controls effective ρ
RHO_VALUES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

DT = 1.0  # seconds


def md1_slowdown(rho):
    if rho <= 0: return 1.0
    if rho >= 1.0: return float('inf')
    return 1.0 + rho**2 / (2.0 * (1.0 - rho))


def run_stencil_sweep():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "output", "torus3d_stencil")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Building torus3d {TORUS_DIMS}, hpr={HOSTS_PER_ROUTER}...")
    G, meta = build_torus3d(TORUS_DIMS, hosts_per_router=HOSTS_PER_ROUTER)
    hosts = sorted([n for n in G.nodes() if G.nodes[n].get("type") == "host"])
    if not hosts:
        hosts = sorted([n for n in G.nodes() if str(n).startswith("h_")])
    N = len(hosts)
    print(f"  {N} hosts, {G.number_of_edges()} links")

    # Compute halo3d-26 coefficient: run with 1 byte per host
    print("  Computing HALO3D_26 link loads (unit traffic)...", end=" ", flush=True)
    t0 = time.time()
    unit_loads = link_loads_for_job_torus(G, meta, hosts, 1.0,
                                          comm_pattern=CommunicationPattern.HALO3D_26)
    print(f"done ({time.time()-t0:.1f}s)")

    vals = list(unit_loads.values())
    max_coeff = max(vals)
    mean_coeff = np.mean(vals)
    print(f"  halo3d-26 max_coeff: {max_coeff:.6f}, mean: {mean_coeff:.6f}")
    print(f"  Links loaded: {len(unit_loads)} / {G.number_of_edges()}")

    max_throughput = LINK_BW * DT

    for rho in RHO_VALUES:
        # tx_volume per host so max link utilization = rho
        tx_volume = rho * max_throughput / max_coeff

        # Number of halo3d iterations this corresponds to
        n_iters = tx_volume / HALO3D_BYTES_PER_ITER

        # Per-link utilizations
        utils = np.array([c * tx_volume / max_throughput for c in vals])
        mean_util = float(np.mean(utils))
        max_util = float(np.max(utils))

        slowdown = md1_slowdown(rho)
        stall = slowdown - 1.0
        avg_latency_ns = ZERO_LOAD_NS * slowdown

        rho_str = f"{rho:.2f}".replace(".", "p")
        summary = {
            "simulator": "raps",
            "topology": "torus3d_stencil",
            "traffic_pattern": "halo3d_26 (26-neighbor: 6 face + 12 edge + 8 vertex)",
            "rho_target": rho,
            "N_hosts": N,
            "dims": list(TORUS_DIMS),
            "hosts_per_router": HOSTS_PER_ROUTER,
            "tx_volume_bytes_per_host": tx_volume,
            "halo3d_equivalent_iterations": n_iters,
            "halo3d_face_bytes_per_iter": HALO3D_BYTES_PER_ITER,
            "mean_utilization": mean_util,
            "max_utilization": max_util,
            "max_coeff": max_coeff,
            "slowdown": slowdown,
            "stall_ratio": stall,
            "avg_latency_ns": avg_latency_ns,
            "zero_load_latency_ns": ZERO_LOAD_NS,
            "link_bandwidth_gbps": LINK_BW / 1e9,
            "model": "M/D/1 queue (analytical, HALO3D_26)",
            "status": "ok",
        }
        json_path = os.path.join(out_dir, f"summary_{rho_str}.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"  ρ={rho:.2f}: max_util={max_util*100:.1f}%  slowdown={slowdown:.3f}  "
              f"~{n_iters:.0f} halo3d iters  latency={avg_latency_ns:.0f}ns")

    print(f"\n  Results in {out_dir}/")


if __name__ == "__main__":
    run_stencil_sweep()

#!/usr/bin/env python3
"""
RAPS Inter-Job Interference Sweep — traffic intensity model.

Matches SST-Macro experiment exactly:
  - Both apps use halo3d-26 communication pattern
  - Interleaved allocation: victim on even-indexed hosts, bully on odd-indexed
  - Both apps share ALL router-to-router links
  - Victim: fixed nx=50, bully: vary nx ∈ {0, 50, 100, 150, 200, 300, 400}
  - Three topologies: dragonfly (72 hosts), torus (1024 hosts), fat-tree (16 hosts)

Model:
  1. Compute per-link load coefficients for victim and bully (router-to-router only)
  2. Compute traffic intensity ratio (TIR) = combined_max_load / victim_only_max_load
  3. Apply temporal overlap: when bully >> victim, victim finishes before bully
     fully loads the network → effective TIR is lower
  4. Slowdown prediction: uses the RAPS M/D/1 model with ρ scaled by the
     effective contention level

Output: JSON files with raw data + comparison metrics for plotting.

Usage:
    cd /lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26
    .venv/bin/python3 Baseline/raps/run_interference_sweep.py
"""

import json
import os
import sys
import time

import networkx as nx_lib
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from raps.network.dragonfly import build_dragonfly, build_dragonfly_circulant
from raps.network.torus3d import (
    build_torus3d, link_loads_for_job_torus, halo3d_26_pairs,
    factorize_3d,
)
from raps.network.fat_tree import build_fattree
from raps.network.base import link_loads_for_job_stencil_3d
from raps.job import CommunicationPattern

# ── Topology configs ──────────────────────────────────────────────
# "small" = original small-scale, "large" = paper-quality scale
TOPOLOGIES_SMALL = {
    "dragonfly": {
        "builder": lambda: build_dragonfly(d=4, a=8, p=2),
        "link_bw": 25e9, "topo_type": "dragonfly",
    },
    "torus": {
        "builder": lambda: build_torus3d((8, 8, 8), hosts_per_router=2),
        "link_bw": 9.6e9, "topo_type": "torus",
    },
    "fattree": {
        "builder": lambda: build_fattree(k=4, total_nodes=16),
        "link_bw": 12.5e9, "topo_type": "fattree",
    },
}

TOPOLOGIES_LARGE = {
    "dragonfly": {
        # Circulant: 32g × 16r × 2h = 1024 hosts, h=14 inter-group links/router
        # Matches SST-Macro bully_victim_dragonfly.ini and RAPS circulant n=1000
        "builder": lambda: build_dragonfly_circulant(G=32, R=16, P=2, H=14)[0],
        "link_bw": 25e9, "topo_type": "dragonfly",
    },
    "torus": {
        "builder": lambda: build_torus3d((8, 8, 8), hosts_per_router=2),  # 1024 hosts
        "link_bw": 9.6e9, "topo_type": "torus",
    },
    "fattree": {
        "builder": lambda: build_fattree(k=16, total_nodes=1024),  # 1024 hosts
        "link_bw": 12.5e9, "topo_type": "fattree",
    },
}

# Circulant dragonfly variant — same scale as TOPOLOGIES_LARGE["dragonfly"] (1056 hosts)
# G=33, R=8, P=4, H=7 → 33*8*4=1056 hosts; port budget: 4+(8-1)+7=18 ≤ 64 ✓
# Used for Experiment 4: canonical all-to-all vs physical circulant comparison.
TOPOLOGIES_LARGE_CIRCULANT = {
    "dragonfly_circulant": {
        "builder": lambda: build_dragonfly_circulant(G=33, R=8, P=4, H=7)[0],
        "link_bw": 25e9, "topo_type": "dragonfly",  # same node-naming conventions
    },
}

# Small-scale defaults (quick, login-node friendly)
SMALL_VICTIM_NX = 50
SMALL_BULLY_NX_VALUES = [0, 50, 100, 150, 200, 300, 400]
SMALL_N_ITERS = 10

# Large-scale defaults (match SST-Macro large sweep)
LARGE_VICTIM_NX = 100
LARGE_BULLY_NX_VALUES = [0, 50, 100, 150, 200, 300, 400]
LARGE_N_ITERS = 1000


# ── Halo3d-26 traffic volume ─────────────────────────────────────
def halo3d_tx_bytes(nx, ny=None, nz=None, n_vars=1, n_iters=10):
    """Total bytes per host for halo3d-26 exchange over n_iters iterations."""
    if ny is None:
        ny = nx
    if nz is None:
        nz = nx
    dbl = 8
    face = 2 * (ny * nz + nx * nz + nx * ny) * dbl * n_vars
    edge = 4 * (nx + ny + nz) * dbl * n_vars
    vertex = 8 * dbl * n_vars
    return (face + edge + vertex) * n_iters


# ── Link-load computation ────────────────────────────────────────
def is_router_link(edge, topo_type):
    """Check if an edge is a router-to-router link (not host-to-router)."""
    a, b = edge
    a_str, b_str = str(a), str(b)
    if topo_type in ("dragonfly", "torus"):
        return not (a_str.startswith("h_") or b_str.startswith("h_"))
    elif topo_type == "fattree":
        return not ("host" in a_str.lower() or "host" in b_str.lower())
    return True


def filter_rr_loads(loads, topo_type):
    """Filter loads to router-to-router links only."""
    return {e: v for e, v in loads.items() if is_router_link(e, topo_type)}


def compute_halo3d_loads_generic(G, host_list, total_tx_bytes):
    """Compute halo3d-26 link loads on any topology via shortest paths."""
    if total_tx_bytes <= 0 or len(host_list) < 2:
        return {}
    triples = halo3d_26_pairs(host_list)
    host_total = {}
    for src, dst, msg in triples:
        host_total[src] = host_total.get(src, 0.0) + msg
    loads = {}
    for src, dst, msg_bytes in triples:
        ht = host_total.get(src, 1.0)
        scaled = msg_bytes * total_tx_bytes / ht if ht > 0 else 0.0
        try:
            path = nx_lib.shortest_path(G, src, dst)
        except nx_lib.NetworkXNoPath:
            continue
        for u, v in zip(path, path[1:]):
            e = tuple(sorted((u, v)))
            loads[e] = loads.get(e, 0.0) + scaled
    return loads


def compute_halo3d_loads_torus(G, meta, host_list, total_tx_bytes):
    """Compute halo3d-26 link loads on torus using dimension-order routing."""
    if total_tx_bytes <= 0 or len(host_list) < 2:
        return {}
    return link_loads_for_job_torus(
        G, meta, host_list, total_tx_bytes,
        comm_pattern=CommunicationPattern.HALO3D_26,
    )


def md1_slowdown(rho):
    """M/D/1 queueing delay factor."""
    if rho <= 0:
        return 1.0
    if rho >= 1.0:
        return float('inf')
    return 1.0 + rho**2 / (2.0 * (1.0 - rho))


# ── Main sweep ────────────────────────────────────────────────────
def run_sweep(scale="small"):
    TOPOLOGIES = TOPOLOGIES_LARGE if scale == "large" else TOPOLOGIES_SMALL
    suffix = "_large" if scale == "large" else ""
    out_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "output", f"interference{suffix}")
    os.makedirs(out_root, exist_ok=True)

    global VICTIM_NX, BULLY_NX_VALUES, N_ITERS
    if scale == "large":
        VICTIM_NX = LARGE_VICTIM_NX
        BULLY_NX_VALUES = LARGE_BULLY_NX_VALUES
        N_ITERS = LARGE_N_ITERS
    else:
        VICTIM_NX = SMALL_VICTIM_NX
        BULLY_NX_VALUES = SMALL_BULLY_NX_VALUES
        N_ITERS = SMALL_N_ITERS

    print(f"Scale: {scale} | victim_nx={VICTIM_NX}, n_iters={N_ITERS} | Output: {out_root}")

    for topo_name, cfg in TOPOLOGIES.items():
        topo_dir = os.path.join(out_root, topo_name)
        os.makedirs(topo_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Topology: {topo_name}")
        print(f"{'='*60}")

        # Build topology
        t0 = time.time()
        if topo_name == "torus":
            G, meta = cfg["builder"]()
        else:
            G = cfg["builder"]()
            meta = None

        # Get host list
        hosts = sorted([n for n in G.nodes()
                        if G.nodes[n].get("type") == "host"
                        or G.nodes[n].get("layer") == "host"
                        or str(n).startswith("h_")])
        if not hosts:
            hosts = sorted([n for n in G.nodes()
                            if "host" in str(n).lower()
                            or G.degree(n) == 1])
        N = len(hosts)

        # Split: even → victim, odd → bully
        victim_hosts = [hosts[i] for i in range(0, N, 2)]
        bully_hosts = [hosts[i] for i in range(1, N, 2)]
        print(f"  {N} hosts → {len(victim_hosts)} victim + {len(bully_hosts)} bully")

        link_bw = cfg["link_bw"]
        topo_type = cfg["topo_type"]
        victim_tx = halo3d_tx_bytes(VICTIM_NX, n_iters=N_ITERS)
        print(f"  Victim tx: {victim_tx/1e6:.3f} MB/host")

        # Compute victim loads (absolute bytes)
        t0 = time.time()
        if topo_name == "torus":
            victim_loads = compute_halo3d_loads_torus(G, meta, victim_hosts, victim_tx)
        else:
            victim_loads = compute_halo3d_loads_generic(G, victim_hosts, victim_tx)
        victim_rr = filter_rr_loads(victim_loads, topo_type)
        if not victim_rr:
            victim_rr = victim_loads
        max_victim_rr = max(victim_rr.values()) if victim_rr else 0
        print(f"  Victim RR loads: {len(victim_rr)} links, "
              f"max={max_victim_rr/1e6:.3f} MB, "
              f"computed in {time.time()-t0:.1f}s")

        # Compute average RR load across top-10 links (more robust than single max)
        sorted_victim = sorted(victim_rr.values(), reverse=True)
        top10_victim = np.mean(sorted_victim[:min(10, len(sorted_victim))])

        for bully_nx in BULLY_NX_VALUES:
            tag = f"{topo_name}_bully_nx{bully_nx}"
            json_path = os.path.join(topo_dir, f"{tag}.json")

            if bully_nx == 0:
                raps_victim_rho_base = max_victim_rr / victim_tx if victim_tx > 0 else 0.0
                result = {
                    "simulator": "raps",
                    "topology": topo_name,
                    "bully_nx": 0,
                    "victim_nx": VICTIM_NX,
                    "victim_tx_bytes": victim_tx,
                    "bully_tx_bytes": 0,
                    "traffic_intensity_ratio": 1.0,
                    "effective_tir": 1.0,
                    "temporal_overlap": 1.0,
                    "max_victim_rr_bytes": max_victim_rr,
                    "max_combined_rr_bytes": max_victim_rr,
                    "raps_victim_rho": raps_victim_rho_base,
                    "raps_combined_rho": raps_victim_rho_base,
                    "raps_raw_combined_rho": raps_victim_rho_base,
                    "raps_md1_slowdown": 1.0,
                    "relative_slowdown": 1.0,
                    "model": "traffic-intensity-ratio",
                    "n_victim_hosts": len(victim_hosts),
                    "n_bully_hosts": len(bully_hosts),
                    "link_bw_gbps": link_bw / 1e9,
                    "status": "ok",
                }
            else:
                bully_tx = halo3d_tx_bytes(bully_nx, n_iters=N_ITERS)
                t1 = time.time()
                if topo_name == "torus":
                    bully_loads = compute_halo3d_loads_torus(
                        G, meta, bully_hosts, bully_tx)
                else:
                    bully_loads = compute_halo3d_loads_generic(
                        G, bully_hosts, bully_tx)
                dt_compute = time.time() - t1

                bully_rr = filter_rr_loads(bully_loads, topo_type)
                if not bully_rr:
                    bully_rr = bully_loads

                # Traffic Intensity Ratio (TIR):
                # For each RR link, compute combined_load / victim_load
                # Take the max across all links where victim has traffic
                max_combined = 0
                for e in victim_rr:
                    vl = victim_rr.get(e, 0)
                    bl = bully_rr.get(e, 0)
                    combined = vl + bl
                    if combined > max_combined:
                        max_combined = combined

                # Also check bully-only links (victim might not use them)
                for e in bully_rr:
                    if e not in victim_rr:
                        bl = bully_rr[e]
                        if bl > max_combined:
                            max_combined = bl

                # Raw TIR: how much more loaded is the max link with bully
                tir = max_combined / max_victim_rr if max_victim_rr > 0 else 1.0

                # Temporal overlap factor:
                # Both apps start at t=0. Victim injects for T_v, bully for T_b.
                # T_v ∝ victim_tx, T_b ∝ bully_tx (both inject at NIC rate).
                # During victim's lifetime [0, T_v], bully injects
                #   min(T_v, T_b) / T_b of its total traffic.
                # = min(victim_tx, bully_tx) / bully_tx
                overlap = min(victim_tx, bully_tx) / bully_tx if bully_tx > 0 else 1.0

                # Effective TIR with temporal overlap:
                # Bully's effective contribution during victim's lifetime
                effective_bully_on_max = 0
                for e in victim_rr:
                    bl = bully_rr.get(e, 0) * overlap
                    combined = victim_rr[e] + bl
                    if combined > max_victim_rr * effective_bully_on_max / max_victim_rr \
                            if max_victim_rr > 0 else 0:
                        pass  # just tracking
                # Simpler: effective max combined = victim_max + bully_on_victim_max * overlap
                # Find bully load on the link where victim is max
                max_victim_link = max(victim_rr, key=victim_rr.get)
                bully_on_max_victim = bully_rr.get(max_victim_link, 0)
                eff_combined = max_victim_rr + bully_on_max_victim * overlap
                eff_tir = eff_combined / max_victim_rr if max_victim_rr > 0 else 1.0

                # Also compute per-link TIR distribution
                tir_values = []
                for e in victim_rr:
                    vl = victim_rr[e]
                    bl = bully_rr.get(e, 0) * overlap
                    if vl > 0:
                        tir_values.append((vl + bl) / vl)
                avg_tir = np.mean(tir_values) if tir_values else 1.0
                p90_tir = np.percentile(tir_values, 90) if tir_values else 1.0

                # ρ-based metrics (normalized by per-host victim traffic volume)
                # rho_victim: routing load concentration — high for fattree (many hosts
                #   share core links), low for torus (DOR keeps traffic local)
                # rho_combined: victim + effective bully load on worst link
                raps_victim_rho = max_victim_rr / victim_tx if victim_tx > 0 else 0.0
                eff_bully = bully_on_max_victim * overlap
                raps_combined_rho = (max_victim_rr + eff_bully) / victim_tx if victim_tx > 0 else raps_victim_rho
                # Raw combined rho (no temporal overlap correction) — increases with bully_nx
                raps_raw_combined_rho = max_combined / victim_tx if victim_tx > 0 else raps_victim_rho

                # M/D/1 slowdown ratio: valid when rho < 1 (torus); else falls back to eff_tir
                if raps_victim_rho < 1.0:
                    rv_cap = min(raps_victim_rho, 0.99)
                    rc_cap = min(raps_combined_rho, 0.99)
                    raps_md1_slowdown = md1_slowdown(rc_cap) / max(md1_slowdown(rv_cap), 1.0)
                else:
                    raps_md1_slowdown = eff_tir  # linear approximation for saturated links

                result = {
                    "simulator": "raps",
                    "topology": topo_name,
                    "bully_nx": bully_nx,
                    "victim_nx": VICTIM_NX,
                    "victim_tx_bytes": victim_tx,
                    "bully_tx_bytes": bully_tx,
                    "traffic_ratio": bully_tx / victim_tx,
                    "traffic_intensity_ratio": tir,
                    "effective_tir": eff_tir,
                    "temporal_overlap": overlap,
                    "avg_tir": avg_tir,
                    "p90_tir": p90_tir,
                    "max_victim_rr_bytes": max_victim_rr,
                    "max_combined_rr_bytes": max_combined,
                    "bully_on_victim_max_link": bully_on_max_victim,
                    "raps_victim_rho": raps_victim_rho,
                    "raps_combined_rho": raps_combined_rho,
                    "raps_raw_combined_rho": raps_raw_combined_rho,
                    "raps_md1_slowdown": raps_md1_slowdown,
                    "model": "traffic-intensity-ratio",
                    "n_victim_hosts": len(victim_hosts),
                    "n_bully_hosts": len(bully_hosts),
                    "n_loaded_rr_links": len(bully_rr),
                    "link_bw_gbps": link_bw / 1e9,
                    "compute_time_s": dt_compute,
                    "status": "ok",
                }

            with open(json_path, "w") as f:
                json.dump(result, f, indent=2)

            tir_val = result.get("traffic_intensity_ratio", 1.0)
            eff_val = result.get("effective_tir", 1.0)
            ovlp = result.get("temporal_overlap", 1.0)
            print(f"  bully_nx={bully_nx:>3d}: "
                  f"TIR={tir_val:.3f}, eff_TIR={eff_val:.3f}, "
                  f"overlap={ovlp:.3f}")

    # Summary table
    print(f"\n{'='*60}")
    print("  Summary: Traffic Intensity Ratio (TIR)")
    print(f"{'='*60}")
    TOPOLOGIES_iter = TOPOLOGIES_LARGE if scale == "large" else TOPOLOGIES_SMALL
    for topo_name in TOPOLOGIES_iter:
        topo_dir = os.path.join(out_root, topo_name)
        rows = []
        for bully_nx in BULLY_NX_VALUES:
            jp = os.path.join(topo_dir, f"{topo_name}_bully_nx{bully_nx}.json")
            if os.path.exists(jp):
                d = json.load(open(jp))
                rows.append((bully_nx,
                              d.get("traffic_intensity_ratio", 1),
                              d.get("effective_tir", 1),
                              d.get("temporal_overlap", 1)))
        if rows:
            print(f"\n{topo_name}:")
            print(f"  {'bnx':>4s}  {'TIR':>6s}  {'eff_TIR':>7s}  {'ovlp':>5s}")
            print(f"  {'-'*30}")
            for bnx, tir, etir, ovlp in rows:
                print(f"  {bnx:>4d}  {tir:>6.3f}  {etir:>7.3f}  {ovlp:>5.3f}")

    print(f"\nResults in {out_root}/")


def run_circulant_sweep():
    """
    Experiment 4A: Run bully-victim sweep for circulant dragonfly (large scale only).
    Saves results to output/interference_large_circulant/dragonfly_circulant/
    for direct comparison with output/interference_large/dragonfly/ (all-to-all).
    """
    global VICTIM_NX, BULLY_NX_VALUES, N_ITERS
    VICTIM_NX = LARGE_VICTIM_NX
    BULLY_NX_VALUES = LARGE_BULLY_NX_VALUES
    N_ITERS = LARGE_N_ITERS

    out_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "output", "interference_large_circulant")
    os.makedirs(out_root, exist_ok=True)
    print(f"Circulant sweep | victim_nx={VICTIM_NX}, n_iters={N_ITERS} | Output: {out_root}")

    for topo_name, cfg in TOPOLOGIES_LARGE_CIRCULANT.items():
        topo_dir = os.path.join(out_root, topo_name)
        os.makedirs(topo_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Topology: {topo_name} (Experiment 4 circulant variant)")
        print(f"{'='*60}")

        t0 = time.time()
        G = cfg["builder"]()
        meta = None

        hosts = sorted([n for n in G.nodes()
                        if G.nodes[n].get("type") == "host"
                        or G.nodes[n].get("layer") == "host"
                        or str(n).startswith("h_")])
        if not hosts:
            hosts = sorted([n for n in G.nodes()
                            if "host" in str(n).lower()
                            or G.degree(n) == 1])
        N = len(hosts)

        victim_hosts = [hosts[i] for i in range(0, N, 2)]
        bully_hosts = [hosts[i] for i in range(1, N, 2)]
        print(f"  {N} hosts → {len(victim_hosts)} victim + {len(bully_hosts)} bully")

        link_bw = cfg["link_bw"]
        topo_type = cfg["topo_type"]
        victim_tx = halo3d_tx_bytes(VICTIM_NX, n_iters=N_ITERS)
        print(f"  Victim tx: {victim_tx/1e6:.3f} MB/host")

        victim_loads = compute_halo3d_loads_generic(G, victim_hosts, victim_tx)
        victim_rr = filter_rr_loads(victim_loads, topo_type)
        if not victim_rr:
            victim_rr = victim_loads
        max_victim_rr = max(victim_rr.values()) if victim_rr else 0
        print(f"  Victim RR loads: {len(victim_rr)} links, "
              f"max={max_victim_rr/1e6:.3f} MB, "
              f"computed in {time.time()-t0:.1f}s")

        sorted_victim = sorted(victim_rr.values(), reverse=True)
        top10_victim = np.mean(sorted_victim[:min(10, len(sorted_victim))])

        for bully_nx in BULLY_NX_VALUES:
            tag = f"{topo_name}_bully_nx{bully_nx}"
            json_path = os.path.join(topo_dir, f"{tag}.json")

            if bully_nx == 0:
                raps_victim_rho_base = max_victim_rr / victim_tx if victim_tx > 0 else 0.0
                result = {
                    "simulator": "raps",
                    "topology": topo_name,
                    "bully_nx": 0,
                    "victim_nx": VICTIM_NX,
                    "victim_tx_bytes": victim_tx,
                    "bully_tx_bytes": 0,
                    "traffic_intensity_ratio": 1.0,
                    "effective_tir": 1.0,
                    "temporal_overlap": 1.0,
                    "max_victim_rr_bytes": max_victim_rr,
                    "max_combined_rr_bytes": max_victim_rr,
                    "bully_on_victim_max_link": 0,
                    "raps_victim_rho": raps_victim_rho_base,
                    "raps_combined_rho": raps_victim_rho_base,
                    "raps_raw_combined_rho": raps_victim_rho_base,
                    "raps_md1_slowdown": 1.0,
                    "model": "traffic-intensity-ratio",
                    "n_victim_hosts": len(victim_hosts),
                    "n_bully_hosts": len(bully_hosts),
                    "n_loaded_rr_links": len(victim_rr),
                    "link_bw_gbps": link_bw / 1e9,
                    "compute_time_s": 0.0,
                    "status": "ok",
                }
            else:
                t0 = time.time()
                n_bully = min(bully_nx ** 3, len(bully_hosts))
                active_bully = bully_hosts[:n_bully]
                bully_tx = halo3d_tx_bytes(bully_nx, n_iters=N_ITERS)

                bully_loads = compute_halo3d_loads_generic(G, active_bully, bully_tx)
                bully_rr = filter_rr_loads(bully_loads, topo_type)
                if not bully_rr:
                    bully_rr = bully_loads

                combined_rr = {}
                for e in set(victim_rr) | set(bully_rr):
                    combined_rr[e] = victim_rr.get(e, 0) + bully_rr.get(e, 0)

                max_combined = max(combined_rr.values()) if combined_rr else 0
                bully_on_max_victim = bully_rr.get(
                    max(victim_rr, key=victim_rr.get) if victim_rr else None, 0
                ) if victim_rr else 0

                dt_compute = time.time() - t0
                victim_bully_ratio = bully_tx / victim_tx if victim_tx > 0 else 0
                tir = max_combined / max_victim_rr if max_victim_rr > 0 else 1.0

                total_bully_load = sum(bully_rr.values()) if bully_rr else 0
                total_victim_load = sum(victim_rr.values()) if victim_rr else 0
                avg_tir = (total_victim_load + total_bully_load) / total_victim_load if total_victim_load > 0 else 1.0

                victim_sorted = sorted(victim_rr.values(), reverse=True)
                combined_sorted = sorted(combined_rr.values(), reverse=True)
                p90_tir = combined_sorted[int(0.1 * len(combined_sorted))] / victim_sorted[int(0.1 * len(victim_sorted))] if victim_sorted else 1.0

                n_v = len(victim_hosts)
                n_b = n_bully
                duration_victim = victim_tx / (link_bw + 1e-9)
                duration_bully = bully_tx / (link_bw + 1e-9)
                overlap = min(1.0, duration_victim / (duration_bully + 1e-9)) if duration_bully > 0 else 1.0
                eff_tir = 1.0 + (tir - 1.0) * overlap

                raps_victim_rho = max_victim_rr / victim_tx if victim_tx > 0 else 0.0
                raps_raw_combined_rho = max_combined / victim_tx if victim_tx > 0 else raps_victim_rho
                eff_bully = bully_on_max_victim * overlap
                raps_combined_rho = (max_victim_rr + eff_bully) / victim_tx if victim_tx > 0 else raps_victim_rho

                if raps_victim_rho < 1.0:
                    rv_cap = min(raps_victim_rho, 0.99)
                    rc_cap = min(raps_combined_rho, 0.99)
                    raps_md1_slowdown = md1_slowdown(rc_cap) / max(md1_slowdown(rv_cap), 1.0)
                else:
                    raps_md1_slowdown = eff_tir

                result = {
                    "simulator": "raps",
                    "topology": topo_name,
                    "bully_nx": bully_nx,
                    "victim_nx": VICTIM_NX,
                    "victim_tx_bytes": victim_tx,
                    "bully_tx_bytes": bully_tx,
                    "traffic_ratio": bully_tx / victim_tx,
                    "traffic_intensity_ratio": tir,
                    "effective_tir": eff_tir,
                    "temporal_overlap": overlap,
                    "avg_tir": avg_tir,
                    "p90_tir": p90_tir,
                    "max_victim_rr_bytes": max_victim_rr,
                    "max_combined_rr_bytes": max_combined,
                    "bully_on_victim_max_link": bully_on_max_victim,
                    "raps_victim_rho": raps_victim_rho,
                    "raps_combined_rho": raps_combined_rho,
                    "raps_raw_combined_rho": raps_raw_combined_rho,
                    "raps_md1_slowdown": raps_md1_slowdown,
                    "model": "traffic-intensity-ratio",
                    "n_victim_hosts": len(victim_hosts),
                    "n_bully_hosts": n_bully,
                    "n_loaded_rr_links": len(bully_rr),
                    "link_bw_gbps": link_bw / 1e9,
                    "compute_time_s": dt_compute,
                    "status": "ok",
                }

            with open(json_path, "w") as f:
                json.dump(result, f, indent=2)

            tir_val = result.get("traffic_intensity_ratio", 1.0)
            eff_val = result.get("effective_tir", 1.0)
            print(f"  bully_nx={bully_nx:>3d}: TIR={tir_val:.3f}, eff_TIR={eff_val:.3f}, "
                  f"raps_md1={result.get('raps_md1_slowdown', 1.0):.4f}")

    print(f"\nCirculant results in {out_root}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAPS interference sweep")
    parser.add_argument("--scale", choices=["small", "large", "both"],
                        default="both", help="Topology scale (default: both)")
    parser.add_argument("--topo-variant", choices=["all-to-all", "circulant", "both"],
                        default="all-to-all",
                        help="Dragonfly topology variant for Experiment 4 (default: all-to-all)")
    args = parser.parse_args()
    if args.topo_variant in ("circulant", "both"):
        run_circulant_sweep()
    if args.topo_variant in ("all-to-all", "both"):
        if args.scale == "both":
            run_sweep("small")
            run_sweep("large")
        else:
            run_sweep(args.scale)

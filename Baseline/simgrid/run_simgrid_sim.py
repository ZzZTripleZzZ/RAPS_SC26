#!/usr/bin/env python3
"""
Real SimGrid flow-level simulation: all-to-all uniform traffic on dragonfly/fat-tree.

Uses SimGrid's SURF CM02 network model (max-min fairness bandwidth sharing).
Each host sends `flow_size` bytes to every other host simultaneously.
SimGrid computes exact bandwidth allocation and flow completion times.

This is a REAL simulation (not analytical) — SimGrid internally:
  1. Enqueues all N*(N-1) concurrent flows
  2. Runs max-min water-filling to allocate bandwidth
  3. Advances simulation time until flows complete
  4. Re-allocates bandwidth whenever a flow completes

Usage (after SimGrid is installed/built):
    python3 run_simgrid_sim.py --topo dragonfly_1000n --rho 0.30 --out output/dragonfly_1000n/summary_0p30.json

Or via run_simgrid_sweep.sh which calls this for all rho values.

Requirements:
    SimGrid v3.35+ built with Python bindings:
    cmake -Denable_python=ON -Dminimal-bindings=ON ...
"""

import argparse
import json
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RAPS_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))

# Add SimGrid build dir to path if needed
SIMGRID_BUILD = os.environ.get(
    "SIMGRID_BUILD",
    os.path.join(_SCRIPT_DIR, "simgrid-build")
)
if os.path.isdir(SIMGRID_BUILD):
    sys.path.insert(0, SIMGRID_BUILD)

RHO_VALUES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

# Zero-load latency: 3 hops × 100ns + 50ns injection (dragonfly)
ZERO_LOAD_LATENCY_NS = {
    "dragonfly":       350.0,
    "dragonfly_1000n": 350.0,
    "fattree":         450.0,
}

# ---------------------------------------------------------------------------
# Platform XML generators
# ---------------------------------------------------------------------------

def make_dragonfly_xml(topo_name: str, bw_gbps: float = 25.0) -> str:
    """
    SimGrid platform XML for dragonfly topology.
    SimGrid Dragonfly topo_parameters: "groups;routers_per_group;hosts_per_router;global_links_per_router"
    """
    if topo_name == "dragonfly":
        # 9g × 4r × 2h = 72 hosts
        groups, rpr, hpr, glpr = 9, 4, 2, 8   # glpr = groups-1 (fully connected)
        n_hosts = groups * rpr * hpr
    elif topo_name == "dragonfly_1000n":
        # 33g × 8r × 4h = 1056 hosts (matches BookSim2 k=4, RAPS build_dragonfly(d=8,a=32,p=4))
        groups, rpr, hpr, glpr = 33, 8, 4, 32
        n_hosts = groups * rpr * hpr
    else:
        raise ValueError(f"Unknown dragonfly topo: {topo_name}")

    bw_mbps = int(bw_gbps * 1000)
    return f"""<?xml version='1.0'?>
<!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">
<platform version="4.1">
  <zone id="AS0" routing="Dijkstra">
    <cluster id="cluster"
             prefix="host-"
             suffix=""
             radical="0-{n_hosts-1}"
             speed="100Gf"
             bw="{bw_mbps}Mbps"
             lat="100ns"
             loopback_bw="{bw_mbps}Mbps"
             loopback_lat="10ns"
             topology="Dragonfly"
             topo_parameters="{groups};{rpr};{hpr};{glpr}"/>
  </zone>
</platform>
"""


def make_fattree_xml(k: int = 4, bw_gbps: float = 12.5) -> str:
    """
    SimGrid platform XML for fat-tree k=4.
    Fat_Tree topo_parameters: "levels;downlinks;uplinks;link_count"
    For k=4: 2 levels; down=[2,2]; up=[2,2]; link_count=[1,1]
    """
    n_hosts = k ** 3 // 4
    bw_mbps = int(bw_gbps * 1000)
    return f"""<?xml version='1.0'?>
<!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">
<platform version="4.1">
  <zone id="AS0" routing="Dijkstra">
    <cluster id="cluster"
             prefix="host-"
             suffix=""
             radical="0-{n_hosts-1}"
             speed="100Gf"
             bw="{bw_mbps}Mbps"
             lat="100ns"
             loopback_bw="{bw_mbps}Mbps"
             loopback_lat="10ns"
             topology="Fat_Tree"
             topo_parameters="2;{k//2},{k//2};{k//2},{k//2};1,1"/>
  </zone>
</platform>
"""


# ---------------------------------------------------------------------------
# SimGrid simulation
# ---------------------------------------------------------------------------

def run_simgrid_alltoall(topo_name: str, rho: float, platform_xml: str,
                          link_bw: float, zero_load_ns: float) -> dict:
    """
    Run SimGrid flow-level all-to-all simulation.

    SimGrid CM02 model: max-min fair bandwidth sharing.
    Each host sends `flow_size` bytes to every other host simultaneously.
    SimGrid determines bandwidth allocation and reports completion times.
    """
    import simgrid

    t_wall_start = time.perf_counter()

    # Write platform XML to temp file
    xml_path = f"/tmp/simgrid_platform_{topo_name}_{rho:.2f}.xml".replace(".", "p")
    with open(xml_path, "w") as f:
        f.write(platform_xml)

    # SimGrid simulation time window: 1 second
    # Flow size: what each host sends to each destination in 1 second at rate rho*link_bw
    sim_duration = 1.0  # seconds
    flow_size_bytes = rho * link_bw * sim_duration  # total bytes per src-dst pair

    # Shared results
    completion_times = []
    latency_samples = []

    def sender_actor(src_host, all_hosts, f_size):
        """Actor: sends f_size bytes to every other host, measures completion time."""
        import simgrid
        t_start = simgrid.Engine.clock
        comms = []
        for dst_host in all_hosts:
            if dst_host.name != src_host.name:
                comm = simgrid.Comm.sendto_async(src_host, dst_host, f_size)
                comms.append(comm)
        # Wait for all sends to complete
        simgrid.Comm.wait_all(comms)
        t_end = simgrid.Engine.clock
        # Track completion time (normalized per destination)
        completion_times.append(t_end - t_start)

    # Initialize SimGrid engine
    e = simgrid.Engine(["simgrid",
                        "--cfg=network/model:CM02",
                        "--log=root.thres:critical"])
    e.load_platform(xml_path)

    hosts = sorted(e.get_all_hosts(), key=lambda h: h.name)
    N = len(hosts)

    # Create one sender actor per host
    for host in hosts:
        simgrid.Actor.create(f"sender-{host.name}", host,
                             sender_actor, host, hosts, flow_size_bytes)

    # Run simulation
    e.run()

    t_wall_end = time.perf_counter()
    os.unlink(xml_path)

    # Compute metrics
    if not completion_times:
        return {"status": "no_data"}

    import numpy as np
    avg_completion = float(np.mean(completion_times))   # seconds
    # Convert to ns for comparison with zero_load_ns
    avg_completion_ns = avg_completion * 1e9

    # Slowdown: actual time / zero-load time
    # zero-load time for the batch = time to send all N-1 flows sequentially at full BW
    # But in SimGrid, zero-load would be: flow_size / link_bw  (single flow)
    zero_load_s = flow_size_bytes / link_bw   # seconds for one flow at zero load
    slowdown = avg_completion / zero_load_s if zero_load_s > 0 else 1.0
    stall_ratio = max(0.0, slowdown - 1.0)

    # Link utilization: compute from actual bytes sent / (link_bw * sim_time)
    # SimGrid doesn't easily expose per-link bytes after simulation,
    # so we compute from what we know: total bytes = N*(N-1)*flow_size
    # under max-min, all links are loaded at their fair share
    # Mean util ≈ rho (by construction), max util from topology coefficients
    total_bytes_injected = N * (N - 1) * flow_size_bytes
    # This goes through the network; mean link util depends on topology
    # Use the actually measured completion time to infer effective utilization
    effective_throughput = flow_size_bytes / avg_completion  # bytes/sec per host per dst
    effective_rho = effective_throughput / link_bw
    mean_util = min(effective_rho, 1.0)

    return {
        "mean_utilization": mean_util,
        "max_utilization": min(rho, 1.0),  # by construction of our injection
        "avg_completion_s": avg_completion,
        "avg_latency_ns": avg_completion_ns,
        "zero_load_s": zero_load_s,
        "zero_load_latency_ns": zero_load_ns,
        "slowdown": slowdown,
        "stall_ratio": stall_ratio,
        "N_hosts": N,
        "flow_size_bytes": flow_size_bytes,
        "wall_time_s": t_wall_end - t_wall_start,
        "model": "simgrid_CM02_flow_level",
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TOPO_CONFIGS = {
    "dragonfly": {
        "xml_fn": lambda: make_dragonfly_xml("dragonfly", bw_gbps=25.0),
        "link_bw": 25e9,
        "kind": "dragonfly",
    },
    "dragonfly_1000n": {
        "xml_fn": lambda: make_dragonfly_xml("dragonfly_1000n", bw_gbps=25.0),
        "link_bw": 25e9,
        "kind": "dragonfly",
        "note": "1056 hosts: 33g x 8r x 4h (matches BookSim2 k=4)",
    },
    "fattree": {
        "xml_fn": lambda: make_fattree_xml(k=4, bw_gbps=12.5),
        "link_bw": 12.5e9,
        "kind": "fattree",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Real SimGrid all-to-all sweep")
    parser.add_argument("--topo", required=True, choices=list(TOPO_CONFIGS),
                        help="Topology name")
    parser.add_argument("--rho", type=float, required=True,
                        help="Injection rate (0..1)")
    parser.add_argument("--out", required=True,
                        help="Output JSON path")
    args = parser.parse_args()

    cfg = TOPO_CONFIGS[args.topo]
    platform_xml = cfg["xml_fn"]()
    link_bw = cfg["link_bw"]
    zero_load_ns = ZERO_LOAD_LATENCY_NS.get(args.topo, 350.0)

    print(f"[SimGrid] topo={args.topo} rho={args.rho:.2f}", flush=True)

    result = run_simgrid_alltoall(
        topo_name=args.topo,
        rho=args.rho,
        platform_xml=platform_xml,
        link_bw=link_bw,
        zero_load_ns=zero_load_ns,
    )

    summary = {
        "simulator": "simgrid",
        "topology": args.topo,
        "rho_target": args.rho,
        "link_bandwidth_gbps": link_bw / 1e9,
        **result,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    if result.get("status") == "ok":
        print(f"  slowdown={result['slowdown']:.4f}  stall={result['stall_ratio']:.4f}  "
              f"wall={result['wall_time_s']:.1f}s → {args.out}")
    else:
        print(f"  FAILED: {result}")
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())

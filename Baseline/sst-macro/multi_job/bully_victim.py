#!/usr/bin/env python3
"""
Generate SST-Macro ini files for bully-victim interference sweep.

For each (topology, bully_size) combination, produces an ini file by:
1. Computing max link-load coefficients from RAPS topology
2. Setting constant_delay so that bully/victim apps achieve target ρ
3. Generating random-permutation destinations for offered_load app

The ini uses SST-Macro's offered_load app (same as the E1 baseline sweep).
app1 = bully (ranks 0..BULLY_N-1), app2 = victims (ranks BULLY_N..TOTAL_N-1).
Both run simultaneously — packets compete on shared links at the PISCES level.

Usage:
    python Baseline/sst-macro/multi_job/bully_victim.py
    python Baseline/sst-macro/multi_job/bully_victim.py --topology dragonfly
    python Baseline/sst-macro/multi_job/bully_victim.py --topology torus3d
    python Baseline/sst-macro/multi_job/bully_victim.py --dry-run
"""

import argparse
import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RAPS_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(RAPS_ROOT))

# ── Topology parameters matching RAPS ─────────────────────────────────────
TOPOLOGY_PARAMS = {
    "dragonfly": {
        "total_nodes": 1024,          # 32g × 16r × 2h
        "link_bw_bytes_s": 25e9,      # 25 GB/s
        "zero_load_ns": 350,          # ~3 hops × 100ns + 50ns injection
        "message_size": 65536,        # 64 KB
        "template": SCRIPT_DIR / "bully_victim_dragonfly.ini",
        "out_subdir": "dragonfly",
    },
    "fattree": {
        "total_nodes": 1024,          # k=16, k^3/4
        "link_bw_bytes_s": 12.5e9,    # 12.5 GB/s
        "zero_load_ns": 200,          # 2 hops × 100ns (up-down)
        "message_size": 65536,
        "template": SCRIPT_DIR / "bully_victim_fattree.ini",
        "out_subdir": "fattree",
    },
    "torus3d": {
        "total_nodes": 1024,          # 8×8×8 × 2 hosts/router
        "link_bw_bytes_s": 9.6e9,     # 9.6 GB/s (Gemini)
        "zero_load_ns": 500,          # avg ~5 hops × 100ns
        "message_size": 65536,
        "template": SCRIPT_DIR / "bully_victim_torus3d.ini",
        "out_subdir": "torus3d",
    },
}

BULLY_SIZES = [0, 32, 64, 128, 256, 512]
VICTIM_COUNT = 4
VICTIM_NODES = 64
BULLY_RHO    = 0.80    # target injection rate for bully (high load)
VICTIM_RHO   = 0.25    # target injection rate for victims (moderate stencil-like)

SEED = 42


def make_permutation(n: int, seed: int = SEED) -> list:
    """Random derangement (no fixed points) of {0, ..., n-1}."""
    rng = random.Random(seed)
    perm = list(range(n))
    while True:
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            return perm


def compute_max_coeff_permutation(topology: str, n_ranks: int, perm: list) -> int:
    """Compute max flows on any single link for a given permutation.

    This mirrors the RAPS network model: routes each (src, dst) pair through
    the topology graph and counts max link sharing.  The result determines
    the offered_load constant_delay needed to hit a target ρ.
    """
    import networkx as nx

    if topology == "dragonfly":
        from raps.network.dragonfly import build_dragonfly
        # d=16 routers/group, a=31 global links/router → num_groups=32, p=2 hosts/router
        G = build_dragonfly(d=16, a=31, p=2)
    elif topology == "fattree":
        from raps.network.fat_tree import build_fattree
        # k=16 → k^3/4 = 1024 hosts
        G = build_fattree(k=16, total_nodes=1024)
    elif topology == "torus3d":
        from raps.network.torus3d import build_torus3d
        # 8×8×8 routers, 2 hosts each → 1024 hosts
        G, _meta = build_torus3d(dims=(8, 8, 8), hosts_per_router=2)
    else:
        return 1

    hosts = sorted([n for n, attr in G.nodes(data=True)
                    if attr.get('layer') == 'host' or attr.get('type') == 'host'])
    if not hosts:
        hosts = sorted([n for n in G.nodes() if str(n).startswith('h_')])
    if len(hosts) < n_ranks:
        print(f"  [WARN] topology has {len(hosts)} hosts but need {n_ranks} ranks")
        n_ranks = len(hosts)

    link_count = {}
    for i in range(n_ranks):
        j = perm[i]
        if i == j:
            continue
        src, dst = hosts[i], hosts[j]
        try:
            path = nx.shortest_path(G, src, dst)
        except nx.NetworkXNoPath:
            continue
        for a, b in zip(path, path[1:]):
            key = tuple(sorted([a, b]))
            link_count[key] = link_count.get(key, 0) + 1

    return max(link_count.values()) if link_count else 1


def compute_delay_ns(message_size: int, max_coeff: int,
                     rho: float, link_bw: float) -> float:
    """constant_delay_ns = message_size × max_coeff / (ρ × link_bw) × 1e9"""
    if rho <= 0:
        return 1e12  # essentially infinite delay = zero injection
    delay_s = message_size * max_coeff / (rho * link_bw)
    return delay_s * 1e9


def generate_configs(topology: str, dry_run: bool = False) -> list:
    """Generate all ini files for a topology's bully-victim sweep."""
    params = TOPOLOGY_PARAMS[topology]
    total_nodes = params["total_nodes"]
    link_bw = params["link_bw_bytes_s"]
    msg_size = params["message_size"]
    template = params["template"]
    out_dir = SCRIPT_DIR / "output" / params["out_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Precompute destinations for each bully_size
    # app1 gets ranks 0..bully_n-1, app2 gets bully_n..bully_n+victim_n-1
    configs = []
    for bully_nodes in BULLY_SIZES:
        victim_total = VICTIM_COUNT * VICTIM_NODES  # 256
        if bully_nodes == 0:
            bully_n = 0
            total_n = victim_total
        else:
            total_needed = bully_nodes + victim_total
            if total_needed > total_nodes:
                print(f"  [SKIP] bully={bully_nodes}: needs {total_needed} > {total_nodes}")
                continue
            bully_n = bully_nodes
            total_n = bully_n + victim_total

        load_fraction = bully_n / total_nodes

        # Generate permutations for bully and victim rank-spaces separately
        if bully_n > 0:
            bully_perm = make_permutation(bully_n, seed=SEED)
            # Compute max_coeff for bully's own traffic pattern
            print(f"  Computing bully max_coeff ({bully_n} ranks)...", end=" ", flush=True)
            if not dry_run:
                bully_max_coeff = compute_max_coeff_permutation(topology, bully_n, bully_perm)
            else:
                bully_max_coeff = 1
            print(f"max_coeff={bully_max_coeff}")
            bully_delay = compute_delay_ns(msg_size, bully_max_coeff, BULLY_RHO, link_bw)
            bully_dests_str = ", ".join(str(x) for x in bully_perm)
        else:
            # baseline: app1 still needs at least 1 rank with near-zero injection
            bully_n = 1  # SST-Macro needs >=1 rank per app
            bully_perm = [0]
            bully_max_coeff = 1
            bully_delay = 1e12  # effectively zero injection
            bully_dests_str = "0"
            total_n = 1 + victim_total

        victim_perm = make_permutation(victim_total, seed=SEED + 1)
        print(f"  Computing victim max_coeff ({victim_total} ranks)...", end=" ", flush=True)
        if not dry_run:
            victim_max_coeff = compute_max_coeff_permutation(topology, victim_total, victim_perm)
        else:
            victim_max_coeff = 1
        print(f"max_coeff={victim_max_coeff}")
        victim_delay = compute_delay_ns(msg_size, victim_max_coeff, VICTIM_RHO, link_bw)
        victim_dests_str = ", ".join(str(x) for x in victim_perm)

        tag = f"bully_{bully_nodes if bully_nodes > 1 else 0:04d}n"
        ini_path = out_dir / f"{tag}.ini"
        json_path = out_dir / f"{tag}.json"

        subs = {
            "BULLY_DELAY_NS": f"{bully_delay:.3f}",
            "VICTIM_DELAY_NS": f"{victim_delay:.3f}",
            "BULLY_N": str(bully_n),
            "VICTIM_N": str(victim_total),
            "BULLY_DESTS": bully_dests_str,
            "VICTIM_DESTS": victim_dests_str,
            "TOTAL_N": str(total_n),
        }

        meta = {
            "topology": topology,
            "bully_nodes": bully_nodes if bully_n > 1 else 0,
            "victim_count": VICTIM_COUNT,
            "victim_nodes": VICTIM_NODES,
            "total_nodes": total_nodes,
            "load_fraction": load_fraction,
            "bully_rho": BULLY_RHO if bully_nodes > 0 else 0.0,
            "victim_rho": VICTIM_RHO,
            "bully_max_coeff": bully_max_coeff,
            "victim_max_coeff": victim_max_coeff,
            "bully_delay_ns": bully_delay,
            "victim_delay_ns": victim_delay,
            "message_size_bytes": msg_size,
            "zero_load_ns": params["zero_load_ns"],
            "ini_file": str(ini_path),
            "json_file": str(json_path),
        }

        if not dry_run:
            text = template.read_text()
            for key, val in subs.items():
                text = text.replace(key, str(val))
            ini_path.write_text(text)
            with open(json_path, "w") as f:
                json.dump({"status": "pending", **meta}, f, indent=2)

        actual_bully = bully_nodes if bully_n > 1 else 0
        print(f"  {'[DRY]' if dry_run else '[GEN]'} {ini_path.name}  "
              f"bully={actual_bully}n  load={load_fraction:.3f}  "
              f"bully_delay={bully_delay:.1f}ns  victim_delay={victim_delay:.1f}ns")
        configs.append(meta)

    print(f"  Generated {len(configs)} configs in {out_dir}")
    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Generate SST-Macro bully-victim ini files")
    parser.add_argument('--topology', nargs='+',
                        default=['dragonfly', 'fattree', 'torus3d'],
                        choices=['dragonfly', 'fattree', 'torus3d'],
                        help="Topologies to generate configs for")
    parser.add_argument('--dry-run', action='store_true',
                        help="Print what would be generated without writing files")
    args = parser.parse_args()

    for topo in args.topology:
        print(f"\n{'='*50}")
        print(f"  {topo} ({TOPOLOGY_PARAMS[topo]['total_nodes']} nodes)")
        print(f"{'='*50}")
        generate_configs(topo, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

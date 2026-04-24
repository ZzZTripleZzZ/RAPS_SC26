#!/usr/bin/env python3
"""Generate node_id allocation files for interleaved interference experiments.

Each topology gets two files: {topo}_victim_nodes.txt and {topo}_bully_nodes.txt.
Victim gets even-numbered host IDs (port 0), bully gets odd (port 1).
This ensures both apps share ALL router-to-router links.

Usage:
    python3 Baseline/sst-macro/multi_job/gen_node_files.py
"""

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "node_files")
os.makedirs(OUT_DIR, exist_ok=True)

TOPOLOGIES = {
    # name: (n_routers, concentration, n_hosts)
    "dragonfly": (36, 2, 72),     # 9 groups × 4 routers × 2 hosts
    "torus":     (512, 2, 1024),   # 8×8×8 routers × 2 hosts
    "fattree":   (4, 4, 16),       # 4 leaf switches × 4 hosts (k=4)
}

# Process grid factorizations for halo3d (pex × pey × pez = n_ranks)
FACTORIZATIONS = {
    36:  (3, 3, 4),
    512: (8, 8, 8),
    8:   (2, 2, 2),
}


def write_node_file(path, node_ids):
    with open(path, "w") as f:
        f.write(f"{len(node_ids)}\n")
        f.write(" ".join(str(x) for x in node_ids) + "\n")


def main():
    for topo, (n_routers, conc, n_hosts) in TOPOLOGIES.items():
        # host_id = router_id * concentration + port
        # Even host IDs = port 0 (victim), odd = port 1 (bully)
        victim_ids = list(range(0, n_hosts, 2))
        bully_ids = list(range(1, n_hosts, 2))

        n_victim = len(victim_ids)
        n_bully = len(bully_ids)
        pex, pey, pez = FACTORIZATIONS[n_victim]
        assert pex * pey * pez == n_victim, \
            f"{topo}: {pex}×{pey}×{pez} != {n_victim}"

        victim_path = os.path.join(OUT_DIR, f"{topo}_victim_nodes.txt")
        bully_path = os.path.join(OUT_DIR, f"{topo}_bully_nodes.txt")
        write_node_file(victim_path, victim_ids)
        write_node_file(bully_path, bully_ids)

        print(f"{topo:>12s}: {n_victim} victim + {n_bully} bully = {n_hosts} hosts "
              f"(grid {pex}×{pey}×{pez})")

    print(f"\nFiles written to {OUT_DIR}/")


if __name__ == "__main__":
    main()

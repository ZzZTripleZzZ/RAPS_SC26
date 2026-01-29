#!/usr/bin/env python3
"""
SC26 Network Topologies (v2.0)
==============================
Thin wrapper around RAPS network module for backwards compatibility.

For new code, use RAPS network module directly:
    from raps.network import build_fattree, build_dragonfly, build_torus3d
"""

import sys
from pathlib import Path

# Add RAPS to path
sys.path.insert(0, str(Path("/app/extern/raps")))

import networkx as nx
import numpy as np

# Try to import from RAPS
try:
    from raps.network import build_fattree as raps_build_fattree
    from raps.network import build_dragonfly as raps_build_dragonfly
    from raps.network import build_torus3d as raps_build_torus3d
    from raps.network.dragonfly import build_dragonfly_idx_map
    from raps.network.fat_tree import node_id_to_host_name
    RAPS_NETWORK_AVAILABLE = True
except ImportError:
    RAPS_NETWORK_AVAILABLE = False
    print("Warning: RAPS network module not available, using fallback implementations")


def build_fattree(k: int = 8, num_nodes: int = 64):
    """
    Build Fat-Tree topology (Lassen-style).

    Args:
        k: Switch radix (port count)
        num_nodes: Number of compute nodes

    Returns:
        NetworkX graph
    """
    if RAPS_NETWORK_AVAILABLE:
        return raps_build_fattree(k, num_nodes)

    # Fallback implementation
    G = nx.Graph()

    # Simple 2-level fat-tree approximation
    servers = range(num_nodes)
    num_edge_switches = max(1, num_nodes // 8)
    num_core_switches = max(1, num_edge_switches // 2)

    edges = range(num_nodes, num_nodes + num_edge_switches)
    cores = range(num_nodes + num_edge_switches,
                  num_nodes + num_edge_switches + num_core_switches)

    G.add_nodes_from(servers, type='host')
    G.add_nodes_from(edges, type='switch')
    G.add_nodes_from(cores, type='switch')

    # Connect servers to edge switches
    for i in servers:
        edge_id = num_nodes + (i * num_edge_switches // num_nodes)
        G.add_edge(i, edge_id, bandwidth=100e9)

    # Connect edge to core (all-to-all)
    for e in edges:
        for c in cores:
            G.add_edge(e, c, bandwidth=400e9)

    return G


def build_dragonfly(d: int = 16, a: int = 8, p: int = 4):
    """
    Build Dragonfly topology (Frontier-style).

    Args:
        d: Routers per group
        a: Global links per router (determines num_groups = a + 1)
        p: Compute nodes per router

    Returns:
        NetworkX graph
    """
    if RAPS_NETWORK_AVAILABLE:
        return raps_build_dragonfly(d, a, p)

    # Fallback implementation
    G = nx.Graph()
    num_groups = a + 1
    total_routers = num_groups * d
    total_nodes = total_routers * p
    router_start_id = total_nodes

    # Intra-group connections
    for g in range(num_groups):
        group_routers = []
        for r in range(d):
            r_id = router_start_id + g * d + r
            group_routers.append(r_id)
            G.add_node(r_id, type='router', group=g)

            # Connect nodes to router
            for n in range(p):
                node_id = (g * d + r) * p + n
                G.add_node(node_id, type='host', group=g)
                G.add_edge(node_id, r_id, type='local')

        # Intra-group all-to-all
        for i in range(len(group_routers)):
            for j in range(i + 1, len(group_routers)):
                G.add_edge(group_routers[i], group_routers[j], type='intra')

    # Inter-group connections
    for g1 in range(num_groups):
        for g2 in range(g1 + 1, num_groups):
            r1 = router_start_id + g1 * d + (g2 % d)
            r2 = router_start_id + g2 * d + (g1 % d)
            G.add_edge(r1, r2, type='global')

    return G


def build_torus3d(dims: tuple = (4, 4, 4), wrap: bool = True):
    """
    Build 3D Torus topology.

    Args:
        dims: Tuple (x, y, z) dimensions
        wrap: Whether edges wrap around (periodic)

    Returns:
        NetworkX graph
    """
    if RAPS_NETWORK_AVAILABLE:
        G, meta = raps_build_torus3d(dims, wrap)
        return G

    # Fallback: use NetworkX grid_graph
    G = nx.grid_graph(dim=list(dims), periodic=wrap)

    # Relabel nodes to integers
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    return G


def get_topology(name: str, num_nodes: int):
    """
    Get topology by name.

    Args:
        name: 'fattree', 'dragonfly', or 'torus3d'
        num_nodes: Number of compute nodes

    Returns:
        NetworkX graph
    """
    if name == "fattree":
        # Determine k based on node count
        k = max(4, int(np.ceil(np.sqrt(num_nodes / 2)) * 2))
        return build_fattree(k, num_nodes)

    elif name == "dragonfly":
        # Scale parameters for node count
        # Total nodes = (a+1) * d * p
        p = 4  # nodes per router
        d = max(4, int(np.ceil(np.sqrt(num_nodes / p))))
        a = max(2, d // 2)
        return build_dragonfly(d, a, p)

    elif name == "torus3d" or name == "torus":
        # Determine dimensions from node count
        k = int(np.ceil(num_nodes ** (1/3)))
        return build_torus3d((k, k, k), wrap=True)

    else:
        raise ValueError(f"Unknown topology: {name}. Use 'fattree', 'dragonfly', or 'torus3d'")


# Convenience functions for RAPS integration
def get_dragonfly_host_mapping(d: int, a: int, p: int, num_nodes: int) -> dict:
    """Get mapping from node index to Dragonfly host name."""
    if RAPS_NETWORK_AVAILABLE:
        return build_dragonfly_idx_map(d, a, p, num_nodes)

    # Fallback
    num_groups = a + 1
    total_hosts = num_groups * d * p
    mapping = {}
    for i in range(num_nodes):
        fat_idx = i % total_hosts
        group = fat_idx // (d * p)
        router = (fat_idx // p) % d
        host = fat_idx % p
        mapping[i] = f"h_{group}_{router}_{host}"
    return mapping


def get_fattree_host_name(node_id: int, k: int) -> str:
    """Get Fat-Tree host name for a node index."""
    if RAPS_NETWORK_AVAILABLE:
        return node_id_to_host_name(node_id, k)

    # Fallback
    pod = node_id // (k * k // 4)
    edge = (node_id % (k * k // 4)) // (k // 2)
    host = node_id % (k // 2)
    return f"h_{pod}_{edge}_{host}"

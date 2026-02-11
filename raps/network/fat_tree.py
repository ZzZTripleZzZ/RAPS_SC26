import random
from typing import Tuple, List
import networkx as nx


def node_id_to_host_name(node_id: int, k: int) -> str:
    """
    Convert an integer node id to the host name string in the fat-tree.
    Node IDs are assumed to be contiguous, mapping to h_{pod}_{edge}_{i}.
    """
    # need to match the scheme from build_fattree
    pod = node_id // (k * k // 4)
    edge = (node_id % (k * k // 4)) // (k // 2)
    host = node_id % (k // 2)
    return f"h_{pod}_{edge}_{host}"


def build_fattree(k, total_nodes):
    """
    Build a k-ary fat-tree:
      - k pods
      - each pod has k/2 edge switches, k/2 agg switches
      - core layer has (k/2)^2 core switches
      - each edge switch connects to k/2 hosts
    Returns a NetworkX Graph where:
      - hosts are named "h_{pod}_{edge}_{i}"
      - edge switches "e_{pod}_{edge}"
      - agg   switches "a_{pod}_{agg}"
      - core  switches "c_{i}_{j}"

    Examples
    --------
    >>> from raps.plotting import plot_network_graph
    >>> G = build_fattree(k=4, total_nodes=16)
    >>> plot_network_graph(G, 'fat_tree.png')
    """
    num_hosts = (k**3) // 4
    if num_hosts < total_nodes:
        raise ValueError(
           f"Fat-tree network with k={k} has {num_hosts} hosts, but the system has {total_nodes} nodes. "
           f"Please increase the value of 'fattree_k' in the system configuration file."
        )
    G = nx.Graph()
    # core
    # num_core = (k//2)**2  # Unused!
    for i in range(k // 2):
        for j in range(k // 2):
            core = f"c_{i}_{j}"
            G.add_node(core, type="core")
    # pods
    for pod in range(k):
        # agg switches
        for agg in range(k // 2):
            a = f"a_{pod}_{agg}"
            G.add_node(a, type="agg")
            # connect to all core switches in column agg
            for i in range(k // 2):
                core = f"c_{agg}_{i}"
                G.add_edge(a, core)
        # edge switches + hosts
        for edge in range(k // 2):
            e = f"e_{pod}_{edge}"
            G.add_node(e, type="edge")
            # connect edge→each agg in this pod
            for agg in range(k // 2):
                a = f"a_{pod}_{agg}"
                G.add_edge(e, a)
            # connect hosts
            for h in range(k // 2):
                host = f"h_{pod}_{edge}_{h}"
                G.add_node(host, type="host")
                G.add_edge(e, host)
    return G


def subsample_hosts(G, num_hosts):
    """Reduce the number of host nodes in the FatTree graph to match system size."""
    hosts = [n for n in G if n.startswith("h")]
    if num_hosts < len(hosts):
        keep = set(random.sample(hosts, num_hosts))
        remove = [n for n in hosts if n not in keep]
        G.remove_nodes_from(remove)
    return G


# =============================================================================
# Adaptive Routing Functions for Fat-Tree
# =============================================================================

def parse_fattree_host(name: str) -> Tuple[int, int, int]:
    """Parse a fat-tree host name into (pod, edge, host_idx).

    Args:
        name: Host name in format "h_{pod}_{edge}_{host}"

    Returns:
        Tuple of (pod, edge_switch_idx, host_idx)
    """
    parts = name.split("_")
    return int(parts[1]), int(parts[2]), int(parts[3])


def get_host_edge_switch(host: str) -> str:
    """Get the edge switch connected to a host.

    Args:
        host: Host name in format "h_{pod}_{edge}_{host}"

    Returns:
        Edge switch name "e_{pod}_{edge}"
    """
    pod, edge, _ = parse_fattree_host(host)
    return f"e_{pod}_{edge}"


def fattree_all_shortest_paths(G: nx.Graph, src: str, dst: str) -> List[List[str]]:
    """Find all shortest paths between source and destination in a fat-tree.

    In a fat-tree, there can be multiple equal-cost paths between hosts,
    especially when they are in different pods (going through different
    aggregation and core switches).

    Args:
        G: Fat-tree graph
        src: Source host name
        dst: Destination host name

    Returns:
        List of all shortest paths, each path is a list of node names
    """
    if src == dst:
        return [[src]]

    try:
        return list(nx.all_shortest_paths(G, src, dst))
    except nx.NetworkXNoPath:
        return []


def fattree_ecmp_select_path(G: nx.Graph, src: str, dst: str) -> List[str]:
    """Select a path using ECMP (Equal-Cost Multi-Path) routing.

    Randomly selects one of the shortest paths between source and destination.

    Args:
        G: Fat-tree graph
        src: Source host name
        dst: Destination host name

    Returns:
        Selected path as a list of node names
    """
    paths = fattree_all_shortest_paths(G, src, dst)
    if not paths:
        return []
    return random.choice(paths)


def estimate_path_load(path: List[str], link_loads: dict) -> float:
    """Estimate the total load on a path based on link loads.

    Args:
        path: List of node names forming the path
        link_loads: Dictionary mapping (node1, node2) tuples to load values

    Returns:
        Sum of loads on all links in the path
    """
    total_load = 0.0
    for i in range(len(path) - 1):
        edge = tuple(sorted([path[i], path[i + 1]]))
        total_load += link_loads.get(edge, 0.0)
    return total_load


def fattree_adaptive_select_path(
    G: nx.Graph,
    src: str,
    dst: str,
    link_loads: dict,
) -> List[str]:
    """Select the least congested path using Adaptive ECMP routing.

    This implements InfiniBand-style Adaptive Routing (AR), which selects
    the path with the lowest congestion among all equal-cost paths.

    Args:
        G: Fat-tree graph
        src: Source host name
        dst: Destination host name
        link_loads: Dictionary mapping edge tuples to current load values

    Returns:
        Selected path as a list of node names (least congested)
    """
    paths = fattree_all_shortest_paths(G, src, dst)
    if not paths:
        return []
    if len(paths) == 1:
        return paths[0]

    # Find path with minimum total load
    best_path = paths[0]
    best_load = estimate_path_load(paths[0], link_loads)

    for path in paths[1:]:
        load = estimate_path_load(path, link_loads)
        if load < best_load:
            best_load = load
            best_path = path

    return best_path


def fattree_route(
    G: nx.Graph,
    src: str,
    dst: str,
    algorithm: str = 'minimal',
    link_loads: dict = None,
    apsp: dict = None,
) -> List[str]:
    """Main routing dispatcher for fat-tree topology.

    Args:
        G: Fat-tree graph
        src: Source host name
        dst: Destination host name
        algorithm: Routing algorithm ('minimal', 'ecmp', or 'adaptive')
        link_loads: Current link loads (required for 'adaptive')
        apsp: Pre-computed all-pairs shortest path dict (optional)

    Returns:
        Selected path as a list of node names
    """
    if src == dst:
        return [src]

    if algorithm == 'minimal':
        if apsp is not None:
            try:
                return apsp[src][dst]
            except (KeyError, nx.NetworkXNoPath):
                return []
        try:
            return nx.shortest_path(G, src, dst)
        except nx.NetworkXNoPath:
            return []

    elif algorithm == 'ecmp':
        return fattree_ecmp_select_path(G, src, dst)

    elif algorithm == 'adaptive':
        if link_loads is None:
            link_loads = {}
        return fattree_adaptive_select_path(G, src, dst, link_loads)

    else:
        raise ValueError(f"Unknown fat-tree routing algorithm: {algorithm}")


def link_loads_for_job_fattree_adaptive(
    G: nx.Graph,
    job_hosts: List[str],
    tx_volume_bytes: float,
    algorithm: str = 'minimal',
    link_loads: dict = None,
    apsp: dict = None,
) -> dict:
    """Compute link loads for a job using adaptive routing on fat-tree.

    Args:
        G: Fat-tree network graph
        job_hosts: List of host names assigned to the job
        tx_volume_bytes: Traffic volume per host pair
        algorithm: Routing algorithm ('minimal', 'ecmp', or 'adaptive')
        link_loads: Current global link loads for adaptive decisions
        apsp: Pre-computed all-pairs shortest path dict (optional)

    Returns:
        Dictionary mapping edge tuples to accumulated load values
    """
    if link_loads is None:
        link_loads = {}

    loads = {}
    n = len(job_hosts)
    if n <= 1:
        return loads

    # All-to-all traffic pattern
    for i, src in enumerate(job_hosts):
        for j, dst in enumerate(job_hosts):
            if i == j:
                continue

            path = fattree_route(G, src, dst, algorithm, link_loads, apsp=apsp)

            # Accumulate loads on path edges
            for k in range(len(path) - 1):
                edge = tuple(sorted([path[k], path[k + 1]]))
                loads[edge] = loads.get(edge, 0.0) + tx_volume_bytes

    return loads

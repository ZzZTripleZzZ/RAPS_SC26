import networkx as nx
import numpy as np
from raps.utils import get_current_utilization
from raps.network.fat_tree import node_id_to_host_name
from raps.network.torus3d import link_loads_for_job_torus, torus_host_from_real_index
from raps.job import CommunicationPattern, normalize_comm_pattern


# Message overhead constants (typical HPC network headers)
MESSAGE_HEADER_OVERHEAD = 64  # bytes per message (conservative estimate)


def debug_print_trace(job, label: str = ""):
    """Print either the length (if iterable) or the value of job.gpu_trace."""
    if hasattr(job.gpu_trace, "__len__"):
        print(f"length of {len(job.gpu_trace)} {label}")
    else:
        print(f"gpu_trace value {job.gpu_trace} {label}")


def apply_job_slowdown(*, job, max_throughput, net_util, net_cong, net_tx, net_rx, debug: bool = False):
    # Get the maximum allowed bandwidth from the configuration.
    if net_cong > 1:
        if debug:
            print(f"congested net_cong: {net_cong}, max_throughput: {max_throughput}")
            debug_print_trace(job, "before dilation")

        throughput = net_tx + net_rx
        slowdown_factor = network_slowdown(throughput, max_throughput)

        if debug:
            print("***", hasattr(job, "dilated"), throughput, max_throughput, slowdown_factor)

        # Only apply slowdown once per job to avoid compounding the effect.
        if not job.dilated:
            if debug:
                print(f"Applying slowdown factor {slowdown_factor:.2f} to job {job.id} due to network congestion")
            job.apply_dilation(slowdown_factor)
            job.dilated = True
            if debug:
                debug_print_trace(job, "after dilation")
    else:
        slowdown_factor = 1
    job.slowdown_factor = slowdown_factor

    return slowdown_factor


def compute_system_network_stats(net_utils, net_tx_list, net_rx_list, slowdown_factors):

    # Compute network averages
    n = len(net_utils) or 1
    avg_tx = sum(net_tx_list) / n
    avg_rx = sum(net_rx_list) / n
    avg_net = sum(net_utils) / n
    # avg_slowdown_per_job = sum(slowdown_factors) / n
    # self.avg_slowdown_history.append(avg_slowdown_per_job)
    # max_slowdown_per_job = max(slowdown_factors)
    # self.max_slowdown_history.append(max_slowdown_per_job)

    return avg_tx, avg_rx, avg_net


def network_congestion(tx, rx, max_throughput):
    """
    Overload factor ≥0: average of send/recv NOT clamped.
    >1.0 means you’re pushing above capacity.
    """
    tx_util = float(tx) / max_throughput
    rx_util = float(rx) / max_throughput
    return (tx_util + rx_util) / 2.0


def network_utilization(tx, rx, max_throughput):
    """
    True utilization in [0,1]: average of send/recv clamped to 100%.
    """
    tx_u = min(float(tx) / max_throughput, 1.0)
    rx_u = min(float(rx) / max_throughput, 1.0)
    return (tx_u + rx_u) / 2.0


def network_slowdown(current_throughput, max_throughput):
    """
    Calculate a slowdown factor based on current network bandwidth usage.

    If current_bw is within limits, the factor is 1.0 (no slowdown).
    If current_bw exceeds max_bw, the factor is current_bw/max_bw.
    """
    if current_throughput <= max_throughput:
        return 1.0
    else:
        return current_throughput / max_throughput


def all_to_all_paths(G, hosts):
    """
    Given a list of host names, return shortest‐paths for every unordered pair.
    """
    paths = []
    for i in range(len(hosts)):
        for j in range(i + 1, len(hosts)):
            src, dst = hosts[i], hosts[j]
            p = nx.shortest_path(G, src, dst)
            paths.append((src, dst, p))
    return paths


def link_loads_for_job(G, job_hosts, tx_volume_bytes):
    """
    Distribute tx_volume_bytes from each host equally to all its peers;
    accumulate per-link loads and return a dict {(u,v):bytes, …}.
    This is the ALL-TO-ALL communication pattern.
    """
    paths = all_to_all_paths(G, job_hosts)
    loads = {edge: 0.0 for edge in G.edges()}
    # each host sends tx_volume_bytes to each of the (N-1) peers
    for src in job_hosts:
        if len(job_hosts) >= 2:
            per_peer = tx_volume_bytes / (len(job_hosts) - 1)
        else:
            per_peer = 0
        # find paths where src is the sender
        for s, d, p in paths:
            if s != src:
                continue
            # add per_peer to every link on p
            for u, v in zip(p, p[1:]):
                # ensure ordering matches loads keys
                edge = (u, v) if (u, v) in loads else (v, u)
                loads[edge] += per_peer
    return loads


def factorize_3d(n):
    """
    Factorize n into three dimensions (x, y, z) for a virtual 3D grid.
    Tries to make dimensions as equal as possible.
    Returns (x, y, z) where x * y * z >= n.
    """
    if n <= 0:
        return (1, 1, 1)

    # Find cube root as starting point
    cube_root = int(np.ceil(n ** (1/3)))

    # Search for best factorization
    best = (n, 1, 1)
    best_diff = n  # difference from cube

    for x in range(1, cube_root + 2):
        for y in range(1, cube_root + 2):
            z = int(np.ceil(n / (x * y)))
            if x * y * z >= n:
                # Prefer more cubic shapes
                diff = max(x, y, z) - min(x, y, z)
                if diff < best_diff or (diff == best_diff and x * y * z < best[0] * best[1] * best[2]):
                    best = (x, y, z)
                    best_diff = diff

    return best


def get_stencil_3d_neighbors(node_idx, dims, num_nodes):
    """
    Get the 6 neighbors for a node in a 3D stencil pattern.
    Uses periodic boundary conditions (wraps around).

    Args:
        node_idx: Linear index of the node (0 to num_nodes-1)
        dims: Tuple (x, y, z) dimensions of the virtual grid
        num_nodes: Total number of nodes in the job

    Returns:
        List of neighbor indices (may have fewer than 6 if at boundaries
        and neighbors would be outside node range)
    """
    x_dim, y_dim, z_dim = dims

    # Convert linear index to 3D coordinates
    z = node_idx // (x_dim * y_dim)
    remainder = node_idx % (x_dim * y_dim)
    y = remainder // x_dim
    x = remainder % x_dim

    neighbors = []

    # 6 neighbors: +x, -x, +y, -y, +z, -z
    neighbor_offsets = [
        (1, 0, 0), (-1, 0, 0),   # x-axis
        (0, 1, 0), (0, -1, 0),   # y-axis
        (0, 0, 1), (0, 0, -1),   # z-axis
    ]

    for dx, dy, dz in neighbor_offsets:
        # Apply periodic boundary conditions
        nx = (x + dx) % x_dim
        ny = (y + dy) % y_dim
        nz = (z + dz) % z_dim

        # Convert back to linear index
        neighbor_idx = nz * (x_dim * y_dim) + ny * x_dim + nx

        # Only include if within actual node count
        if neighbor_idx < num_nodes and neighbor_idx != node_idx:
            neighbors.append(neighbor_idx)

    return neighbors


def stencil_3d_pairs(job_hosts):
    """
    Generate communication pairs for 3D stencil pattern.
    Each node communicates with up to 6 neighbors.

    Args:
        job_hosts: List of host names

    Returns:
        List of (src_host, dst_host) pairs
    """
    num_nodes = len(job_hosts)
    dims = factorize_3d(num_nodes)

    pairs = []
    for idx, host in enumerate(job_hosts):
        neighbors = get_stencil_3d_neighbors(idx, dims, num_nodes)
        for neighbor_idx in neighbors:
            pairs.append((host, job_hosts[neighbor_idx]))

    return pairs


def link_loads_for_job_stencil_3d(G, job_hosts, tx_volume_bytes):
    """
    Distribute tx_volume_bytes using 3D stencil pattern.
    Each host sends to its 6 neighbors (±x, ±y, ±z).

    Args:
        G: NetworkX graph
        job_hosts: List of host names
        tx_volume_bytes: Total transmit volume per host

    Returns:
        dict {(u,v): bytes, ...} of link loads
    """
    loads = {edge: 0.0 for edge in G.edges()}

    if len(job_hosts) < 2:
        return loads

    # Get communication pairs
    pairs = stencil_3d_pairs(job_hosts)

    # Count neighbors per host to distribute traffic correctly
    neighbor_count = {}
    for src, dst in pairs:
        neighbor_count[src] = neighbor_count.get(src, 0) + 1

    # Compute paths and accumulate loads
    for src, dst in pairs:
        # Each host divides its tx_volume among its neighbors
        num_neighbors = neighbor_count.get(src, 1)
        per_neighbor = tx_volume_bytes / num_neighbors

        try:
            path = nx.shortest_path(G, src, dst)
            for u, v in zip(path, path[1:]):
                edge = (u, v) if (u, v) in loads else (v, u)
                loads[edge] += per_neighbor
        except nx.NetworkXNoPath:
            # No path between hosts (shouldn't happen in connected graph)
            pass

    return loads


def apply_message_size_overhead(tx_volume_bytes, message_size, num_peers, *, overhead_bytes=None):
    """
    Apply message size overhead to the traffic volume.

    Smaller messages incur more overhead per byte due to:
    - Fixed header costs per message
    - Protocol overhead
    - Latency costs (more round trips)

    Args:
        tx_volume_bytes: Raw transmit volume in bytes
        message_size: Size of each message in bytes (None = no overhead)
        num_peers: Number of peers this traffic is distributed across

    Returns:
        Effective transmit volume with overhead applied
    """
    if message_size is None or message_size <= 0:
        return tx_volume_bytes

    if tx_volume_bytes <= 0:
        return 0

    # Calculate number of messages needed
    # Each peer gets tx_volume_bytes / num_peers, sent in message_size chunks
    if num_peers <= 0:
        num_peers = 1

    bytes_per_peer = tx_volume_bytes / num_peers
    messages_per_peer = np.ceil(bytes_per_peer / message_size)
    total_messages = messages_per_peer * num_peers

    # Add header overhead for each message
    per_msg_overhead = MESSAGE_HEADER_OVERHEAD if overhead_bytes is None else overhead_bytes
    overhead_bytes = total_messages * per_msg_overhead

    return tx_volume_bytes + overhead_bytes


def get_effective_traffic(tx_volume_bytes, job, num_hosts):
    """
    Get effective traffic volume considering message size and pattern.

    Args:
        tx_volume_bytes: Raw transmit volume
        job: Job object with comm_pattern and message_size
        num_hosts: Number of hosts in the job

    Returns:
        Effective transmit volume with overhead applied
    """
    message_size = getattr(job, 'message_size', None)
    comm_pattern = normalize_comm_pattern(getattr(job, 'comm_pattern', CommunicationPattern.ALL_TO_ALL))
    overhead_bytes = getattr(job, 'message_overhead_bytes', None)

    # Calculate number of peers based on pattern
    if comm_pattern == CommunicationPattern.STENCIL_3D:
        # Stencil has up to 6 neighbors
        num_peers = min(6, num_hosts - 1)
    else:
        # All-to-all: everyone talks to everyone
        num_peers = max(1, num_hosts - 1)

    return apply_message_size_overhead(
        tx_volume_bytes,
        message_size,
        num_peers,
        overhead_bytes=overhead_bytes,
    )


def link_loads_for_pattern(
    G,
    job_hosts,
    tx_volume_bytes,
    comm_pattern,
    *,
    routing_algorithm: str | None = None,
    dragonfly_params: dict | None = None,
    fattree_params: dict | None = None,
    link_loads: dict | None = None,
):
    """
    Dispatch to appropriate link load calculation based on communication pattern
    and routing algorithm.

    Args:
        G: NetworkX graph
        job_hosts: List of host names
        tx_volume_bytes: Total transmit volume per host
        comm_pattern: CommunicationPattern enum value
        routing_algorithm: Routing algorithm
            - Dragonfly: 'minimal', 'ugal', 'valiant'
            - Fat-tree: 'minimal', 'ecmp', 'adaptive'
        dragonfly_params: Dict with 'd', 'a', 'ugal_threshold', 'valiant_bias' for Dragonfly
        fattree_params: Dict with 'k' for fat-tree (optional, for future use)
        link_loads: Current global link loads (for adaptive routing decisions)

    Returns:
        dict {(u,v): bytes, ...} of link loads
    """
    from raps.network.dragonfly import link_loads_for_job_dragonfly_adaptive
    from raps.network.fat_tree import link_loads_for_job_fattree_adaptive

    comm_pattern = normalize_comm_pattern(comm_pattern)

    # Handle adaptive routing for Dragonfly
    if routing_algorithm and dragonfly_params and routing_algorithm in ('ugal', 'valiant'):
        return link_loads_for_job_dragonfly_adaptive(
            G,
            job_hosts,
            tx_volume_bytes,
            algorithm=routing_algorithm,
            d=dragonfly_params['d'],
            a=dragonfly_params['a'],
            link_loads=link_loads,
            ugal_threshold=dragonfly_params.get('ugal_threshold', 2.0),
            valiant_bias=dragonfly_params.get('valiant_bias', 0.0),
        )

    # Handle adaptive routing for Fat-tree
    if routing_algorithm and routing_algorithm in ('ecmp', 'adaptive'):
        return link_loads_for_job_fattree_adaptive(
            G,
            job_hosts,
            tx_volume_bytes,
            algorithm=routing_algorithm,
            link_loads=link_loads,
        )

    # Standard routing (shortest path)
    if comm_pattern == CommunicationPattern.STENCIL_3D:
        return link_loads_for_job_stencil_3d(G, job_hosts, tx_volume_bytes)
    else:
        # Default to all-to-all
        return link_loads_for_job(G, job_hosts, tx_volume_bytes)


def worst_link_util(loads, throughput):
    """
    Given loads in **bytes** and capacity in **bits/sec**, convert:
      util = (bytes * 8) / throughput
    Return the maximum util over all links.
    """
    max_util = 0.0
    for edge, byte_load in loads.items():
        util = (byte_load * 8) / throughput
        if util > max_util:
            max_util = util
    return max_util


def get_link_util_stats(loads, throughput, top_n=10):
    """
    Calculates a distribution of link utilization stats.
    Returns a dictionary with min, mean, max, std_dev, and top N congested links.
    """
    if not loads:
        return {'max': 0, 'mean': 0, 'min': 0, 'std_dev': 0, 'top_links': []}

    # Calculate utilization for every link
    utilizations = {(edge): (byte_load * 8) / throughput for edge, byte_load in loads.items()}

    util_values = list(utilizations.values())

    stats = {
        'max': np.max(util_values),
        'mean': np.mean(util_values),
        'min': np.min(util_values),
        'std_dev': np.std(util_values)
    }

    # Get top N congested links
    sorted_links = sorted(utilizations.items(), key=lambda item: item[1], reverse=True)
    stats['top_links'] = sorted_links[:top_n]

    return stats


def max_throughput_per_tick(legacy_cfg: dict, trace_quanta: int) -> float:
    """Return bytes-per-tick throughput of a single link."""
    bw = legacy_cfg.get("NETWORK_MAX_BW") or 12.5e9
    return float(bw) * trace_quanta


def simulate_inter_job_congestion(network_model, jobs, legacy_cfg, debug=False):
    """
    Simulates network congestion from a list of concurrently running jobs.
    Supports different communication patterns and message sizes per job.
    """
    if not network_model.net_graph:
        print("[WARN] Network graph is not defined. Skipping congestion simulation.")
        return 0.0

    total_loads = {tuple(sorted(edge)): 0.0 for edge in network_model.net_graph.edges()}
    trace_quanta = jobs[0].trace_quanta if jobs else 0

    for job in jobs:
        # Assuming job.current_run_time is 0 for this static simulation
        job.current_run_time = 0
        job.trace_start_time = 0
        net_tx = get_current_utilization(job.ntx_trace, job)

        # Get communication pattern and apply message size overhead
        comm_pattern = getattr(job, 'comm_pattern', CommunicationPattern.ALL_TO_ALL)
        num_hosts = len(job.scheduled_nodes)
        effective_tx = get_effective_traffic(net_tx, job, num_hosts)

        if debug:
            print(f"  Job {job.id}: pattern={comm_pattern}, raw_tx={net_tx}, effective_tx={effective_tx}")

        job_loads = {}
        if network_model.topology in ("fat-tree", "dragonfly"):
            if network_model.topology == "fat-tree":
                k = int(legacy_cfg.get("FATTREE_K", 32))
                host_list = [node_id_to_host_name(n, k) for n in job.scheduled_nodes]
            else:  # dragonfly
                host_list = [network_model.real_to_fat_idx[real_n] for real_n in job.scheduled_nodes]

            job_loads = link_loads_for_pattern(network_model.net_graph, host_list, effective_tx, comm_pattern)

        elif network_model.topology == "torus3d":
            X = int(legacy_cfg.get("TORUS_X", 12))
            Y = int(legacy_cfg.get("TORUS_Y", 12))
            Z = int(legacy_cfg.get("TORUS_Z", 12))
            hosts_per_router = int(legacy_cfg.get("HOSTS_PER_ROUTER", 1))
            host_list = [
                torus_host_from_real_index(n, X, Y, Z, hosts_per_router)
                for n in job.scheduled_nodes
            ]
            # Use pattern-aware loading for stencil, torus-specific for all-to-all
            if comm_pattern == CommunicationPattern.STENCIL_3D:
                job_loads = link_loads_for_pattern(network_model.net_graph, host_list, effective_tx, comm_pattern)
            else:
                job_loads = link_loads_for_job_torus(network_model.net_graph, network_model.meta, host_list, effective_tx)

        for edge, load in job_loads.items():
            edge_key = tuple(sorted(edge))
            if edge_key in total_loads:
                total_loads[edge_key] += load

    max_throughput = max_throughput_per_tick(legacy_cfg, trace_quanta)
    net_stats = get_link_util_stats(total_loads, max_throughput)

    return net_stats

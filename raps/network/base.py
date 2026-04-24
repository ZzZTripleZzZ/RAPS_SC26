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


def compute_stall_ratio(slowdown_factor):
    """
    Compute the stall/packet ratio from a slowdown factor.

    This is the RAPS analog of the Cassini counter ratio:
        (hni_tx_paused_0 + hni_tx_paused_1) / parbs_tarb_pi_posted_pkts

    Derivation: tx_paused = (s - 1) * posted_pkts, so stall_ratio = s - 1.
    - No congestion (s=1): stall_ratio = 0
    - Frontier-level load (s≈6.7): stall_ratio ≈ 5.7

    Args:
        slowdown_factor: Network slowdown factor (≥1.0)

    Returns:
        Dimensionless stall/packet ratio (≥0.0)
    """
    return max(0.0, float(slowdown_factor) - 1.0)


def compute_link_stall_packet_stats(loads, link_bw_bps, mean_pkt_size_bytes, dt, slowdown_factor):
    """
    Compute per-link stall/packet stats from RAPS link loads.

    Maps to Cassini counters:
      posted_pkts  ~ parbs_tarb_pi_posted_pkts
      tx_paused    ~ hni_tx_paused_0 + hni_tx_paused_1
      stall_ratio  = tx_paused / posted_pkts  = slowdown_factor - 1

    Args:
        loads: dict {edge: byte_load} per-link byte load for the tick
        link_bw_bps: link bandwidth in bits/s (e.g. 25e9 * 8 for 25 GB/s)
        mean_pkt_size_bytes: mean packet size in bytes (116 for Frontier Slingshot)
        dt: timestep duration in seconds
        slowdown_factor: network slowdown factor (≥1.0); may be per-job average

    Returns:
        dict {edge: {'posted_pkts', 'tx_paused', 'stall_ratio', 'utilization'}}
    """
    max_pkt_rate = link_bw_bps / (mean_pkt_size_bytes * 8)  # pkts/s at 100% utilization
    stall = max(0.0, float(slowdown_factor) - 1.0)
    stats = {}
    for edge, byte_load in loads.items():
        rho = (byte_load * 8) / (link_bw_bps * dt) if dt > 0 else 0.0
        posted_pkts = rho * max_pkt_rate * dt
        stats[edge] = {
            'posted_pkts': posted_pkts,
            'tx_paused': stall * posted_pkts,
            'stall_ratio': stall,
            'utilization': min(rho, 1.0),
        }
    return stats


def aggregate_link_stall_stats(link_stats):
    """
    Aggregate per-link stall stats to system-level totals.

    Returns:
        dict with 'total_posted_pkts', 'total_tx_paused', 'system_stall_ratio'
    """
    total_posted = sum(s['posted_pkts'] for s in link_stats.values())
    total_paused = sum(s['tx_paused'] for s in link_stats.values())
    system_stall_ratio = total_paused / total_posted if total_posted > 0 else 0.0
    return {
        'total_posted_pkts': total_posted,
        'total_tx_paused': total_paused,
        'system_stall_ratio': system_stall_ratio,
    }


def apply_job_slowdown(*, job, max_throughput, net_util, net_cong, net_tx, net_rx, debug: bool = False):
    # Get the maximum allowed bandwidth from the configuration.
    if net_cong > 1:
        if debug:
            print(f"congested net_cong: {net_cong}, max_throughput: {max_throughput}")
            debug_print_trace(job, "before dilation")

        # Use net_cong directly as the slowdown factor: it is the worst-link
        # overload ratio (>1 means overloaded).  The previous approach computed
        # network_slowdown(net_tx+net_rx, max_throughput) which always returned
        # 1.0 because per-node TX << link capacity.
        slowdown_factor = net_cong

        if debug:
            print("***", hasattr(job, "dilated"), net_cong, max_throughput, slowdown_factor)

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
    # Track peak slowdown across all ticks (not just the last one).
    job.slowdown_factor = max(getattr(job, 'slowdown_factor', 1), slowdown_factor)
    job.stall_ratio = compute_stall_ratio(job.slowdown_factor)

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


def all_to_all_paths(G, hosts, apsp=None):
    """
    Given a list of host names, return shortest‐paths for every unordered pair.
    If apsp (all-pairs shortest path dict) is provided, use it instead of nx.shortest_path.
    """
    paths = []
    for i in range(len(hosts)):
        for j in range(i + 1, len(hosts)):
            src, dst = hosts[i], hosts[j]
            if apsp is not None:
                p = apsp[src][dst]
            else:
                p = nx.shortest_path(G, src, dst)
            paths.append((src, dst, p))
    return paths


def compute_random_ring_coefficients(G, job_hosts, apsp=None, seed=None):
    """
    Compute link-load coefficients for RANDOM_RING pattern.

    Models GPCNeT's "RR Two-sided BW" test: a random permutation of all N
    nodes forms a ring; each node communicates bidirectionally with its left
    and right neighbors in the permutation.

    Each node sends tx_volume_bytes total, split equally (1/2 each) to its
    two ring neighbors — matching the per_peer_coeff convention used in
    compute_all_to_all_coefficients (where coeff = 1/(N-1) per peer).
    """
    N = len(job_hosts)
    if N < 2:
        return {}

    rng = np.random.default_rng(seed if seed is not None
                                else abs(hash(tuple(job_hosts))) % (2**31))
    perm = rng.permutation(N)
    ordered = [job_hosts[i] for i in perm]

    edge_set = set(G.edges())
    coeffs = {}
    per_pair = 0.5  # each node sends 1/2 of tx_volume to each of 2 ring neighbors

    for i in range(N):
        src = ordered[i]
        dst = ordered[(i + 1) % N]
        if src == dst:
            continue
        if apsp is not None:
            path = apsp[src][dst]
        else:
            try:
                path = nx.shortest_path(G, src, dst)
            except Exception:
                continue
        for u, v in zip(path, path[1:]):
            edge = (u, v) if (u, v) in edge_set else (v, u)
            coeffs[edge] = coeffs.get(edge, 0.0) + per_pair

    return coeffs


def link_loads_for_job_ring(G, job_hosts, tx_volume_bytes, apsp=None, seed=None):
    """Link loads for RANDOM_RING communication pattern."""
    coeffs = compute_random_ring_coefficients(G, job_hosts, apsp=apsp, seed=seed)
    return {edge: coeff * tx_volume_bytes for edge, coeff in coeffs.items()}


def link_loads_for_job(G, job_hosts, tx_volume_bytes, apsp=None):
    """
    Distribute tx_volume_bytes from each host equally to all its peers;
    accumulate per-link loads and return a dict {(u,v):bytes, …}.
    This is the ALL-TO-ALL communication pattern.
    """
    coeffs = compute_all_to_all_coefficients(G, job_hosts, apsp=apsp)
    return {edge: coeff * tx_volume_bytes for edge, coeff in coeffs.items()}


def compute_all_to_all_coefficients(G, job_hosts, apsp=None):
    """
    Compute normalized link-load coefficients for ALL_TO_ALL pattern.

    Returns a sparse dict {(u,v): coefficient} such that the actual load
    for any tx_volume_bytes is simply ``coefficient * tx_volume_bytes``.

    Complexity: O(N^2 * path_length) — computed once per job lifetime.
    """
    edge_set = set(G.edges())
    coeffs = {}
    N = len(job_hosts)
    if N < 2:
        return coeffs

    per_peer_coeff = 1.0 / (N - 1)

    # Build paths grouped by source — avoids the O(N^3) scan in the
    # original code that iterated *all* N^2 paths for *each* of N sources.
    for i in range(N):
        src = job_hosts[i]
        for j in range(i + 1, N):
            dst = job_hosts[j]
            if apsp is not None:
                p = apsp[src][dst]
            else:
                p = nx.shortest_path(G, src, dst)
            for u, v in zip(p, p[1:]):
                edge = (u, v) if (u, v) in edge_set else (v, u)
                coeffs[edge] = coeffs.get(edge, 0.0) + per_peer_coeff

    return coeffs


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


def link_loads_for_job_stencil_3d(G, job_hosts, tx_volume_bytes, apsp=None):
    """
    Distribute tx_volume_bytes using 3D stencil pattern.
    Each host sends to its 6 neighbors (±x, ±y, ±z).

    Args:
        G: NetworkX graph
        job_hosts: List of host names
        tx_volume_bytes: Total transmit volume per host
        apsp: Pre-computed all-pairs shortest path dict (optional)

    Returns:
        dict {(u,v): bytes, ...} of link loads
    """
    coeffs = compute_stencil_3d_coefficients(G, job_hosts, apsp=apsp)
    return {edge: coeff * tx_volume_bytes for edge, coeff in coeffs.items()}


def compute_stencil_3d_coefficients(G, job_hosts, apsp=None):
    """
    Compute normalized link-load coefficients for STENCIL_3D pattern.

    Returns a sparse dict {(u,v): coefficient} such that the actual load
    for any tx_volume_bytes is simply ``coefficient * tx_volume_bytes``.

    Complexity: O(N * path_length) — computed once per job lifetime.
    """
    edge_set = set(G.edges())
    coeffs = {}

    if len(job_hosts) < 2:
        return coeffs

    # Get communication pairs
    pairs = stencil_3d_pairs(job_hosts)

    # Count neighbors per host to distribute traffic correctly
    neighbor_count = {}
    for src, dst in pairs:
        neighbor_count[src] = neighbor_count.get(src, 0) + 1

    # Compute paths and accumulate coefficients
    for src, dst in pairs:
        num_neighbors = neighbor_count.get(src, 1)
        per_neighbor_coeff = 1.0 / num_neighbors

        try:
            if apsp is not None:
                path = apsp[src][dst]
            else:
                path = nx.shortest_path(G, src, dst)
            for u, v in zip(path, path[1:]):
                edge = (u, v) if (u, v) in edge_set else (v, u)
                coeffs[edge] = coeffs.get(edge, 0.0) + per_neighbor_coeff
        except (nx.NetworkXNoPath, KeyError):
            pass

    return coeffs


def compute_matrix_template_coefficients(G, job_hosts, traffic_matrix, apsp=None):
    """Return {edge: coefficient} for a MATRIX_TEMPLATE communication pattern.

    ``traffic_matrix`` must be an M×M normalized weight matrix (rows summing
    to ≤1) where entry (i, j) is the fraction of rank i's send volume that
    is addressed to rank j.  ``job_hosts[i]`` maps rank i → physical host.

    The resulting coefficients can be multiplied by ``tx_volume_bytes`` to
    recover the per-link byte load contribution of this job — compatible with
    the caching layer used for ALL_TO_ALL and STENCIL_3D patterns.
    """
    if traffic_matrix is None:
        return {}

    m = traffic_matrix.shape[0]
    if m != len(job_hosts):
        raise ValueError(
            f"traffic_matrix shape {traffic_matrix.shape} does not match "
            f"number of job hosts {len(job_hosts)}"
        )

    edge_set = set(G.edges())
    coeffs: dict = {}
    # Iterate only over nonzero entries — tiled stencils stay extremely sparse.
    src_ranks, dst_ranks = np.nonzero(traffic_matrix)
    for i, j in zip(src_ranks.tolist(), dst_ranks.tolist()):
        if i == j:
            continue
        w = float(traffic_matrix[i, j])
        if w == 0.0:
            continue
        src = job_hosts[i]
        dst = job_hosts[j]
        if src == dst:
            continue
        try:
            if apsp is not None:
                path = apsp[src][dst]
            else:
                path = nx.shortest_path(G, src, dst)
        except (nx.NetworkXNoPath, KeyError):
            continue
        for u, v in zip(path, path[1:]):
            edge = (u, v) if (u, v) in edge_set else (v, u)
            coeffs[edge] = coeffs.get(edge, 0.0) + w
    return coeffs


def link_loads_for_job_matrix(G, job_hosts, traffic_matrix, tx_volume_bytes, apsp=None):
    """Link loads for a MATRIX_TEMPLATE pattern (thin wrapper over coefficients)."""
    coeffs = compute_matrix_template_coefficients(G, job_hosts, traffic_matrix, apsp=apsp)
    return {edge: coeff * tx_volume_bytes for edge, coeff in coeffs.items()}


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
        num_peers = min(6, num_hosts - 1)
    elif comm_pattern == CommunicationPattern.RANDOM_RING:
        num_peers = min(2, num_hosts - 1)
    else:
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
    apsp: dict | None = None,
    traffic_matrix=None,
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

    # Matrix-template always routes per-pair along shortest paths regardless
    # of the configured routing algorithm.  Adaptive routing with explicit
    # matrix weights is not yet supported, so we deliberately bypass the
    # adaptive dispatch below.
    if comm_pattern == CommunicationPattern.MATRIX_TEMPLATE:
        return link_loads_for_job_matrix(G, job_hosts, traffic_matrix, tx_volume_bytes, apsp=apsp)

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
            inter_group_adj=dragonfly_params.get('inter_group_adj'),
            comm_pattern=comm_pattern,
        )

    # Handle adaptive routing for Fat-tree
    if routing_algorithm and routing_algorithm in ('ecmp', 'adaptive'):
        return link_loads_for_job_fattree_adaptive(
            G,
            job_hosts,
            tx_volume_bytes,
            algorithm=routing_algorithm,
            link_loads=link_loads,
            paths_cache=fattree_params.get('paths_cache') if fattree_params else None,
            comm_pattern=comm_pattern,
        )

    # Standard routing (shortest path)
    if comm_pattern == CommunicationPattern.STENCIL_3D:
        return link_loads_for_job_stencil_3d(G, job_hosts, tx_volume_bytes, apsp=apsp)
    elif comm_pattern == CommunicationPattern.RANDOM_RING:
        return link_loads_for_job_ring(G, job_hosts, tx_volume_bytes, apsp=apsp)
    elif comm_pattern == CommunicationPattern.MATRIX_TEMPLATE:
        return link_loads_for_job_matrix(G, job_hosts, traffic_matrix, tx_volume_bytes, apsp=apsp)
    else:
        # Default to all-to-all
        return link_loads_for_job(G, job_hosts, tx_volume_bytes, apsp=apsp)


def worst_link_util(loads, throughput):
    """
    Given loads in **bytes** and capacity in **bytes/sec**, compute:
      util = byte_load / throughput
    Return the maximum util over all links.
    """
    max_util = 0.0
    for edge, byte_load in loads.items():
        util = byte_load / throughput
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
    utilizations = {(edge): byte_load / throughput for edge, byte_load in loads.items()}

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


def simulate_inter_job_congestion(network_model, jobs, legacy_cfg, debug=False, apsp=None,
                                   job_coeffs_cache=None, routing_algorithm=None,
                                   dragonfly_params=None, fattree_params=None):
    """
    Simulates network congestion from a list of concurrently running jobs.
    Supports different communication patterns and message sizes per job.

    Parameters
    ----------
    job_coeffs_cache : dict, optional
        Pre-computed link-load coefficients per job (job.id -> {edge: coefficient}).
        When provided, cached coefficients are scaled by the current traffic volume
        instead of recomputing paths from scratch, reducing cost from O(N²) to O(edges).
        Only used for deterministic (minimal) routing; adaptive routing always uses slow path.
    routing_algorithm : str, optional
        Routing algorithm to use in slow path ('minimal', 'ugal', 'valiant', 'ecmp', 'adaptive').
        If None, defaults to minimal (shortest-path) routing.
    dragonfly_params : dict, optional
        Dict with 'd', 'a', 'ugal_threshold', 'valiant_bias' for Dragonfly adaptive routing.
    fattree_params : dict, optional
        Dict with 'paths_cache' for fat-tree ECMP/adaptive routing.
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

        # Fast path: reuse cached coefficients if available.
        # Only valid for deterministic (minimal) routing; adaptive routing is not cached.
        is_adaptive = routing_algorithm in ('ugal', 'valiant', 'ecmp', 'adaptive')
        if not is_adaptive and job_coeffs_cache is not None and job.id in job_coeffs_cache:
            coeffs = job_coeffs_cache[job.id]
            if isinstance(coeffs, tuple):
                # NumPy array format (idx_arr, coeff_arr) from integer-indexed fast path.
                idx_arr, coeff_arr = coeffs
                idx_to_edge = network_model._idx_to_edge
                for idx, coeff in zip(idx_arr, coeff_arr):
                    edge_key = idx_to_edge[idx]
                    if edge_key in total_loads:
                        total_loads[edge_key] += coeff * effective_tx
            else:
                for edge, coeff in coeffs.items():
                    edge_key = tuple(sorted(edge))
                    if edge_key in total_loads:
                        total_loads[edge_key] += coeff * effective_tx
            continue

        # Slow path: compute link loads from scratch using the actual routing algorithm.
        job_loads = {}
        host_list = network_model.get_job_hosts(job)

        if network_model.topology in ("fat-tree", "dragonfly"):
            # Pass accumulated total_loads as link_loads so adaptive routing
            # (UGAL, adaptive fat-tree) can make informed path decisions.
            job_loads = link_loads_for_pattern(
                network_model.net_graph,
                host_list,
                effective_tx,
                comm_pattern,
                routing_algorithm=routing_algorithm,
                dragonfly_params=dragonfly_params,
                fattree_params=fattree_params,
                link_loads=total_loads,
                apsp=apsp,
                traffic_matrix=getattr(job, 'traffic_template', None),
            )

        elif network_model.topology == "torus3d":
            # Use pattern-aware loading for stencil, torus-specific for all-to-all
            if comm_pattern == CommunicationPattern.STENCIL_3D:
                job_loads = link_loads_for_pattern(network_model.net_graph, host_list, effective_tx, comm_pattern, apsp=apsp)
            elif comm_pattern == CommunicationPattern.MATRIX_TEMPLATE:
                job_loads = link_loads_for_job_matrix(
                    network_model.net_graph, host_list,
                    getattr(job, 'traffic_template', None),
                    effective_tx, apsp=apsp,
                )
            else:
                job_loads = link_loads_for_job_torus(network_model.net_graph, network_model.meta, host_list, effective_tx)

        for edge, load in job_loads.items():
            edge_key = tuple(sorted(edge))
            if edge_key in total_loads:
                total_loads[edge_key] += load

    max_throughput = max_throughput_per_tick(legacy_cfg, trace_quanta)
    net_stats = get_link_util_stats(total_loads, max_throughput)

    return net_stats

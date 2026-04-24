import random
import warnings
import networkx as nx
from itertools import combinations

def build_dragonfly(d, a, p):
    """
    Build a Dragonfly network graph.
    d = routers per group
    a = global connections per router
    p = compute nodes per router
    """
    G = nx.Graph()
    num_groups = a + 1  # standard Dragonfly rule

    # --- Routers and hosts ---
    for g in range(num_groups):
        for r in range(d):
            router = f"r_{g}_{r}"
            G.add_node(router, layer="router", group=g)

            # attach p hosts to each router
            for h in range(p):
                host = f"h_{g}_{r}_{h}"
                G.add_node(host, layer="host", group=g)
                G.add_edge(router, host)

    # --- Intra-group full mesh ---
    for g in range(num_groups):
        routers = [f"r_{g}_{r}" for r in range(d)]
        for i in range(d):
            for j in range(i + 1, d):
                G.add_edge(routers[i], routers[j])

    # --- Inter-group (global) links ---
    for g in range(num_groups):
        for r in range(d):
            src = f"r_{g}_{r}"
            for offset in range(1, a + 1):
                dst_group = (g + offset) % num_groups
                dst = f"r_{dst_group}_{r % d}"
                G.add_edge(src, dst)

    return G


def build_dragonfly2(D: int, A: int, P: int) -> nx.Graph:
    """
    Build a “simple” k-ary Dragonfly with:
       D = # of groups
       A = # of routers per group
       P = # of hosts (endpoints) per router

    Naming convention:
      - Router nodes: "r_{g}_{r}"   with g ∈ [0..D−1], r ∈ [0..A−1]
      - Host  nodes: "h_{g}_{r}_{p}"  with p ∈ [0..P−1]

    Topology:
      1. All routers within a group form a full clique.
      2. Each router r in group g has exactly one “global link” to router r in each other group.
      3. Each router r in group g attaches to P hosts ("h_{g}_{r}_{0..P−1}").

    Examples
    --------
    >>> from raps.plotting import plot_network_graph
    >>> G = build_dragonfly(D=2, A=2, P=2)
    >>> plot_network_graph(G, 'dragonfly.png')
    """
    G = nx.Graph()

    # 1) Create all router nodes
    for g in range(D):
        for r in range(A):
            router = f"r_{g}_{r}"
            G.add_node(router, type="router", group=g, index=r)

    # 2) Intra‐group full mesh of routers
    for g in range(D):
        routers_in_group = [f"r_{g}_{r}" for r in range(A)]
        for u, v in combinations(routers_in_group, 2):
            G.add_edge(u, v)

    # 3) Inter‐group “one‐to‐one” global links
    #    (router index r in group g  →  router index r in group g2)
    for g1 in range(D):
        for g2 in range(g1 + 1, D):
            for r in range(A):
                u = f"r_{g1}_{r}"
                v = f"r_{g2}_{r}"
                G.add_edge(u, v)

    # 4) Attach hosts to each router
    for g in range(D):
        for r in range(A):
            router = f"r_{g}_{r}"
            for p in range(P):
                host = f"h_{g}_{r}_{p}"
                G.add_node(host, type="host", group=g, router=r, index=p)
                G.add_edge(router, host)

    return G


def build_dragonfly_circulant(
    G: int, R: int, P: int, H: int, offsets: list | None = None
) -> tuple[nx.Graph, dict]:
    """
    Build a Dragonfly network with circulant inter-group connectivity.

    Models the real Frontier topology:
      G = num groups (74)
      R = routers per group (32)
      P = hosts per router (2)
      H = inter-group links per router (30)

    Port budget: P + (R-1) + H must fit within switch port count.
    For Frontier: 2 + 31 + 30 = 63 ≤ 64-port Slingshot ✓

    Inter-group (circulant): router r in group g connects to router r in
    group (g + offset) % G for each offset.  Default offsets are symmetric
    ±1..±(H//2) around the ring.

    Returns
    -------
    (graph, inter_group_adj)
        graph           : NetworkX graph with all router and host nodes
        inter_group_adj : dict {(g, r): frozenset of directly reachable group indices}
    """
    if offsets is None:
        half = H // 2
        pos_offsets = list(range(1, half + 1))
        neg_offsets = list(range(G - half, G))
        offsets = pos_offsets + neg_offsets
        if H % 2 == 1:
            # odd H: add one more positive offset
            offsets.append(half + 1)

    if len(offsets) != H:
        raise ValueError(f"Expected {H} offsets, got {len(offsets)}")

    port_budget = P + (R - 1) + H
    if port_budget > 64:
        warnings.warn(
            f"Port budget {port_budget} exceeds 64-port switch limit",
            RuntimeWarning,
        )

    net_g = nx.Graph()

    # Routers and hosts
    for g in range(G):
        for r in range(R):
            router = f"r_{g}_{r}"
            net_g.add_node(router, type="router", group=g, index=r)
            for p in range(P):
                host = f"h_{g}_{r}_{p}"
                net_g.add_node(host, type="host", group=g, router=r, index=p)
                net_g.add_edge(router, host)

    # Intra-group full mesh
    for g in range(G):
        routers = [f"r_{g}_{r}" for r in range(R)]
        for u, v in combinations(routers, 2):
            net_g.add_edge(u, v)

    # Inter-group circulant links:
    # Router r in group g → router r in group (g + offset) % G
    for g in range(G):
        for r in range(R):
            src = f"r_{g}_{r}"
            for offset in offsets:
                dst_group = (g + offset) % G
                dst = f"r_{dst_group}_{r}"
                if not net_g.has_edge(src, dst):
                    net_g.add_edge(src, dst)

    # Build inter_group_adj: same reachable set for all routers in a group
    inter_group_adj = {}
    for g in range(G):
        reachable = frozenset((g + offset) % G for offset in offsets)
        for r in range(R):
            inter_group_adj[(g, r)] = reachable

    return net_g, inter_group_adj


def dragonfly_node_id_to_host_name(fat_idx: int, D: int, A: int, P: int) -> str:
    """
    Convert a contiguous Dragonfly host index to its hierarchical name.

    For a Dragonfly with:
      D routers per group,
      A global links per router  ⇒ num_groups = A + 1,
      P compute nodes per router.

    Hosts are laid out in contiguous order:
      group g = floor(fat_idx / (D * P))
      router r = (fat_idx // P) % D
      host   h = fat_idx % P
    """
    num_groups = A + 1
    total_hosts = num_groups * D * P
    assert 0 <= fat_idx < total_hosts, f"fat_idx {fat_idx} out of range (max {total_hosts-1})"

    group = fat_idx // (D * P)
    router = (fat_idx // P) % D
    host = fat_idx % P
    return f"h_{group}_{router}_{host}"


def build_dragonfly_idx_map(d: int, a: int, p: int, total_real_nodes: int) -> dict[int, str]:
    """
    Build a mapping {real_node_index: host_name} for Dragonfly.
    Wrap around if total_real_nodes > total_hosts.
    """
    num_groups = a + 1
    total_hosts = num_groups * d * p

    mapping = {}
    for i in range(total_real_nodes):
        fat_idx = i % total_hosts  # <- wrap safely
        group = fat_idx // (d * p)
        router = (fat_idx // p) % d
        host = fat_idx % p
        mapping[i] = f"h_{group}_{router}_{host}"
    return mapping


def build_dragonfly_idx_map_circulant(
    G: int, R: int, P: int, total_real_nodes: int
) -> dict[int, str]:
    """
    Build a mapping {real_node_index: host_name} for circulant Dragonfly.

    Uses explicit G (num groups) instead of the a+1 formula.
    Wraps around if total_real_nodes > G*R*P.
    """
    total_hosts = G * R * P
    mapping = {}
    for i in range(total_real_nodes):
        fat_idx = i % total_hosts
        group = fat_idx // (R * P)
        router = (fat_idx // P) % R
        host = fat_idx % P
        mapping[i] = f"h_{group}_{router}_{host}"
    return mapping


# Adaptive Routing Functions for Dragonfly

def parse_dragonfly_host(host_name: str) -> tuple[int, int, int]:
    """
    Parse a Dragonfly host name into its components.

    Args:
        host_name: Host name in format 'h_{group}_{router}_{port}'

    Returns:
        Tuple of (group, router, port)
    """
    parts = host_name.split("_")
    return int(parts[1]), int(parts[2]), int(parts[3])


def parse_dragonfly_router(router_name: str) -> tuple[int, int]:
    """
    Parse a Dragonfly router name into its components.

    Args:
        router_name: Router name in format 'r_{group}_{router}'

    Returns:
        Tuple of (group, router_index)
    """
    parts = router_name.split("_")
    return int(parts[1]), int(parts[2])


def get_host_router(host_name: str) -> str:
    """
    Get the router name for a given host.

    Args:
        host_name: Host name in format 'h_{group}_{router}_{port}'

    Returns:
        Router name in format 'r_{group}_{router}'
    """
    group, router, _ = parse_dragonfly_host(host_name)
    return f"r_{group}_{router}"


def dragonfly_minimal_path(
    src_host: str,
    dst_host: str,
    d: int,
    a: int,
    inter_group_adj: dict | None = None,
) -> list[str]:
    """
    Compute the minimal path between two hosts in a Dragonfly network.

    Minimal paths in Dragonfly use at most one global link:
    - Intra-group: host → router → [local hop] → router → host (max 2 router hops)
    - Inter-group: host → router → [local] → global → [local] → router → host (3 router hops)

    When inter_group_adj is provided (circulant topology):
    - If dst_group is directly reachable: 1 global hop
    - Otherwise: 2 global hops via a reachable intermediate group

    Args:
        src_host: Source host name (h_g_r_p format)
        dst_host: Destination host name (h_g_r_p format)
        d: Number of routers per group
        a: Number of global links per router (num_groups = a + 1, used when inter_group_adj is None)
        inter_group_adj: Optional {(g, r): frozenset of reachable groups} for circulant topology

    Returns:
        List of node names forming the minimal path
    """
    src_group, src_router, _ = parse_dragonfly_host(src_host)
    dst_group, dst_router, _ = parse_dragonfly_host(dst_host)

    src_r = f"r_{src_group}_{src_router}"
    dst_r = f"r_{dst_group}_{dst_router}"

    # Same host
    if src_host == dst_host:
        return [src_host]

    # Same router
    if src_group == dst_group and src_router == dst_router:
        return [src_host, src_r, dst_host]

    # Intra-group: full mesh, so direct local link
    if src_group == dst_group:
        return [src_host, src_r, dst_r, dst_host]

    # Inter-group
    path = [src_host, src_r]

    if inter_group_adj is None:
        # All-to-all (original): router r connects to router (r % d) in every other group
        global_landing_router = src_router % d
        path.append(f"r_{dst_group}_{global_landing_router}")
        if global_landing_router != dst_router:
            path.append(dst_r)
        path.append(dst_host)
    else:
        # Circulant: router r connects to router r in offset groups
        reachable = inter_group_adj.get((src_group, src_router), frozenset())

        if dst_group in reachable:
            # Direct 1 global hop: lands at same router index in dst_group
            landing = src_router
            path.append(f"r_{dst_group}_{landing}")
            if landing != dst_router:
                path.append(dst_r)
            path.append(dst_host)
        else:
            # 2 global hops: src_group → mid_group → dst_group
            # Find intermediate whose reachable set (from router src_router) includes dst_group
            mid_group = None
            for g in reachable:
                mid_reachable = inter_group_adj.get((g, src_router), frozenset())
                if dst_group in mid_reachable:
                    mid_group = g
                    break

            if mid_group is None:
                # Fallback: use first reachable group
                mid_group = next(iter(reachable)) if reachable else (src_group + 1)

            # First global hop: src_router → mid_group (lands at src_router)
            mid_landing = src_router
            path.append(f"r_{mid_group}_{mid_landing}")

            # Second global hop: mid_landing → dst_group (lands at mid_landing)
            dst_landing = mid_landing
            path.append(f"r_{dst_group}_{dst_landing}")
            if dst_landing != dst_router:
                path.append(dst_r)
            path.append(dst_host)

    return path


def dragonfly_nonminimal_path(
    src_host: str,
    dst_host: str,
    intermediate_group: int,
    d: int,
    a: int,
    inter_group_adj: dict | None = None,
) -> list[str]:
    """
    Compute a non-minimal path via an intermediate group (Valiant routing).

    Non-minimal paths use two global links:
    src_group → intermediate_group → dst_group

    Args:
        src_host: Source host name
        dst_host: Destination host name
        intermediate_group: Group index to route through
        d: Number of routers per group
        a: Number of global links per router
        inter_group_adj: Optional {(g, r): frozenset} for circulant topology

    Returns:
        List of node names forming the non-minimal path
    """
    src_group, src_router, _ = parse_dragonfly_host(src_host)
    dst_group, dst_router, _ = parse_dragonfly_host(dst_host)

    src_r = f"r_{src_group}_{src_router}"
    dst_r = f"r_{dst_group}_{dst_router}"

    path = [src_host, src_r]

    # Phase 1: Source group → Intermediate group
    if src_group != intermediate_group:
        if inter_group_adj is None:
            inter_landing = src_router % d
        else:
            # Circulant: same router index
            inter_landing = src_router
        path.append(f"r_{intermediate_group}_{inter_landing}")
        current_router = inter_landing
    else:
        current_router = src_router

    # Phase 2: Intermediate group → Destination group
    if intermediate_group != dst_group:
        if inter_group_adj is None:
            dst_landing = current_router % d
        else:
            dst_landing = current_router
        path.append(f"r_{dst_group}_{dst_landing}")

        if dst_landing != dst_router:
            path.append(dst_r)
    else:
        if current_router != dst_router:
            path.append(dst_r)

    path.append(dst_host)
    return path


def estimate_path_latency(
    path: list[str],
    link_loads: dict,
    hop_latency: float = 1.0,
    congestion_weight: float = 1.0
) -> float:
    """
    Estimate the latency of a path based on hop count and link congestion.

    Used by UGAL to compare minimal vs non-minimal paths.

    Args:
        path: List of node names in the path
        link_loads: Dict {(u, v): load_bytes} of current link loads
        hop_latency: Base latency per hop (default 1.0)
        congestion_weight: Weight for congestion term (default 1.0)

    Returns:
        Estimated latency (higher = more congested/longer path)
    """
    if len(path) <= 1:
        return 0.0

    latency = 0.0

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]

        # Normalize edge key (try both orderings)
        edge = (u, v) if (u, v) in link_loads else (v, u)

        # Base hop latency
        latency += hop_latency

        # Add congestion component
        load = link_loads.get(edge, 0.0)
        latency += congestion_weight * load

    return latency


def ugal_select_path(
    src_host: str,
    dst_host: str,
    link_loads: dict,
    d: int,
    a: int,
    threshold: float = 2.0,
    inter_group_adj: dict | None = None,
) -> list[str]:
    """
    UGAL (Universal Globally-Adaptive Load-balanced) path selection.

    Compares minimal path latency to best non-minimal path latency.
    Decision rule: if minimal_latency < threshold * best_nonminimal_latency,
    use minimal path; otherwise use non-minimal.

    Args:
        src_host: Source host name
        dst_host: Destination host name
        link_loads: Current link load state {(u, v): bytes}
        d: Number of routers per group
        a: Number of global links per router (used when inter_group_adj is None)
        threshold: Decision threshold (default 2.0, standard UGAL)
        inter_group_adj: Optional {(g, r): frozenset} for circulant topology

    Returns:
        Selected path as list of node names
    """
    src_group, src_router, _ = parse_dragonfly_host(src_host)
    dst_group, _, _ = parse_dragonfly_host(dst_host)

    # Compute minimal path and its latency
    minimal_path = dragonfly_minimal_path(src_host, dst_host, d, a, inter_group_adj)
    minimal_latency = estimate_path_latency(minimal_path, link_loads)

    # For intra-group traffic, always use minimal (no benefit from non-minimal)
    if src_group == dst_group:
        return minimal_path

    # Determine candidate intermediate groups
    if inter_group_adj is None:
        num_groups = a + 1
        candidates = range(num_groups)
    else:
        candidates = inter_group_adj.get((src_group, src_router), frozenset())

    # Evaluate non-minimal paths through each candidate intermediate group
    best_nonminimal_path = None
    best_nonminimal_latency = float('inf')

    for inter_group in candidates:
        if inter_group == src_group or inter_group == dst_group:
            continue

        nonminimal_path = dragonfly_nonminimal_path(
            src_host, dst_host, inter_group, d, a, inter_group_adj
        )
        latency = estimate_path_latency(nonminimal_path, link_loads)

        if latency < best_nonminimal_latency:
            best_nonminimal_latency = latency
            best_nonminimal_path = nonminimal_path

    # UGAL decision
    if best_nonminimal_path is None:
        return minimal_path

    if minimal_latency < threshold * best_nonminimal_latency:
        return minimal_path
    else:
        return best_nonminimal_path


def valiant_select_path(
    src_host: str,
    dst_host: str,
    d: int,
    a: int,
    bias: float = 0.0,
    inter_group_adj: dict | None = None,
) -> list[str]:
    """
    Valiant routing with configurable bias toward non-minimal paths.

    Args:
        src_host: Source host name
        dst_host: Destination host name
        d: Number of routers per group
        a: Number of global links per router (used when inter_group_adj is None)
        bias: Fraction of traffic to route non-minimally (0.0-1.0)
              0.0 = always minimal, 1.0 = always non-minimal
              0.05 = 5% non-minimal, 95% minimal
        inter_group_adj: Optional {(g, r): frozenset} for circulant topology

    Returns:
        Selected path as list of node names
    """
    src_group, src_router, _ = parse_dragonfly_host(src_host)
    dst_group, _, _ = parse_dragonfly_host(dst_host)

    # Intra-group: always minimal (non-minimal makes no sense)
    if src_group == dst_group:
        return dragonfly_minimal_path(src_host, dst_host, d, a, inter_group_adj)

    # Probabilistic selection based on bias
    if random.random() >= bias:
        return dragonfly_minimal_path(src_host, dst_host, d, a, inter_group_adj)
    else:
        # Use non-minimal path via random intermediate group
        if inter_group_adj is None:
            num_groups = a + 1
            valid_intermediates = [
                g for g in range(num_groups)
                if g != src_group and g != dst_group
            ]
        else:
            reachable = inter_group_adj.get((src_group, src_router), frozenset())
            valid_intermediates = [
                g for g in reachable
                if g != src_group and g != dst_group
            ]

        if not valid_intermediates:
            return dragonfly_minimal_path(src_host, dst_host, d, a, inter_group_adj)

        inter_group = random.choice(valid_intermediates)
        return dragonfly_nonminimal_path(src_host, dst_host, inter_group, d, a, inter_group_adj)


def dragonfly_route(
    src_host: str,
    dst_host: str,
    algorithm: str,
    d: int,
    a: int,
    link_loads: dict | None = None,
    ugal_threshold: float = 2.0,
    valiant_bias: float = 0.0,
    inter_group_adj: dict | None = None,
) -> list[str]:
    """
    Main routing dispatcher for Dragonfly networks.

    Args:
        src_host: Source host name
        dst_host: Destination host name
        algorithm: Routing algorithm ('minimal', 'ugal', 'valiant')
        d: Number of routers per group
        a: Number of global links per router (used when inter_group_adj is None)
        link_loads: Current link loads (required for UGAL)
        ugal_threshold: UGAL decision threshold
        valiant_bias: Valiant non-minimal bias (0.0-1.0)
        inter_group_adj: Optional {(g, r): frozenset} for circulant topology

    Returns:
        Path as list of node names
    """
    if algorithm == 'minimal':
        return dragonfly_minimal_path(src_host, dst_host, d, a, inter_group_adj)

    elif algorithm == 'ugal':
        if link_loads is None:
            link_loads = {}
        return ugal_select_path(
            src_host, dst_host, link_loads, d, a, ugal_threshold, inter_group_adj
        )

    elif algorithm == 'valiant':
        return valiant_select_path(src_host, dst_host, d, a, valiant_bias, inter_group_adj)

    else:
        # Default to minimal
        return dragonfly_minimal_path(src_host, dst_host, d, a, inter_group_adj)


def link_loads_for_job_dragonfly_adaptive(
    G: nx.Graph,
    job_hosts: list[str],
    tx_volume_bytes: float,
    algorithm: str,
    d: int,
    a: int,
    link_loads: dict | None = None,
    ugal_threshold: float = 2.0,
    valiant_bias: float = 0.0,
    inter_group_adj: dict | None = None,
    comm_pattern=None,
) -> dict:
    """
    Compute link loads for a job using adaptive routing on Dragonfly.

    Args:
        G: NetworkX graph (for edge initialization)
        job_hosts: List of host names for this job
        tx_volume_bytes: Traffic volume per host
        algorithm: Routing algorithm ('minimal', 'ugal', 'valiant')
        d: Number of routers per group
        a: Number of global links per router (used when inter_group_adj is None)
        link_loads: Global link loads (for UGAL decisions)
        ugal_threshold: UGAL decision threshold
        valiant_bias: Valiant non-minimal bias
        inter_group_adj: Optional {(g, r): frozenset} for circulant topology
        comm_pattern: CommunicationPattern (ALL_TO_ALL, STENCIL_3D, RANDOM_RING)

    Returns:
        Dict {(u, v): bytes} of link loads from this job
    """
    from raps.job import CommunicationPattern, normalize_comm_pattern
    from raps.network.base import stencil_3d_pairs

    job_loads = {}

    if len(job_hosts) < 2:
        return job_loads

    comm = normalize_comm_pattern(comm_pattern)
    n = len(job_hosts)

    # Generate communication pairs based on pattern
    if comm == CommunicationPattern.STENCIL_3D:
        pairs = stencil_3d_pairs(job_hosts)
        # Count neighbors per host for correct per-peer volume
        from collections import Counter
        neighbor_count = Counter(src for src, _ in pairs)
        # pairs are directed (src→dst), route each once
        for src, dst in pairs:
            nc = neighbor_count[src]
            per_peer = tx_volume_bytes / nc if nc > 0 else 0.0
            path = dragonfly_route(
                src, dst, algorithm, d, a,
                link_loads=link_loads,
                ugal_threshold=ugal_threshold,
                valiant_bias=valiant_bias,
                inter_group_adj=inter_group_adj,
            )
            for u, v in zip(path, path[1:]):
                edge = tuple(sorted((u, v)))
                # Directed pairs: half weight per edge for undirected accounting
                job_loads[edge] = job_loads.get(edge, 0.0) + per_peer / 2
    else:
        # All-to-all: each host sends to every other host.
        # Iterate unordered pairs (i < j) to avoid double-counting.
        per_peer = tx_volume_bytes / (n - 1)
        for i in range(n):
            for j in range(i + 1, n):
                src = job_hosts[i]
                dst = job_hosts[j]
                path = dragonfly_route(
                    src, dst, algorithm, d, a,
                    link_loads=link_loads,
                    ugal_threshold=ugal_threshold,
                    valiant_bias=valiant_bias,
                    inter_group_adj=inter_group_adj,
                )
                for u, v in zip(path, path[1:]):
                    edge = tuple(sorted((u, v)))
                    job_loads[edge] = job_loads.get(edge, 0.0) + per_peer

    return job_loads

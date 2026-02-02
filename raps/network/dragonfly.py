import random
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


# =============================================================================
# Adaptive Routing Functions for Dragonfly
# =============================================================================

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


def dragonfly_minimal_path(src_host: str, dst_host: str, d: int, a: int) -> list[str]:
    """
    Compute the minimal path between two hosts in a Dragonfly network.

    Minimal paths in Dragonfly use at most one global link:
    - Intra-group: host → router → [local hop] → router → host (max 2 router hops)
    - Inter-group: host → router → [local] → global → [local] → router → host (3 router hops)

    Args:
        src_host: Source host name (h_g_r_p format)
        dst_host: Destination host name (h_g_r_p format)
        d: Number of routers per group
        a: Number of global links per router (num_groups = a + 1)

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

    # Inter-group: need to use global link
    # In build_dragonfly(), router r in group g connects to router (r % d) in other groups
    # So from src_router in src_group, the global link lands at router (src_router % d) in dst_group

    path = [src_host, src_r]

    # Global link destination router in dst_group
    global_landing_router = src_router % d

    # Take the global link
    path.append(f"r_{dst_group}_{global_landing_router}")

    # If we didn't land at the destination router, add local hop
    if global_landing_router != dst_router:
        path.append(dst_r)

    path.append(dst_host)
    return path


def dragonfly_nonminimal_path(
    src_host: str,
    dst_host: str,
    intermediate_group: int,
    d: int,
    a: int
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
        # Global link from src_router lands at router (src_router % d) in intermediate
        inter_landing = src_router % d
        path.append(f"r_{intermediate_group}_{inter_landing}")
        current_router = inter_landing
    else:
        # Already in intermediate group (shouldn't happen in valid Valiant)
        current_router = src_router

    # Phase 2: Intermediate group → Destination group
    if intermediate_group != dst_group:
        # Global link from current_router lands at router (current_router % d) in dst_group
        dst_landing = current_router % d
        path.append(f"r_{dst_group}_{dst_landing}")

        # Local hop to destination router if needed
        if dst_landing != dst_router:
            path.append(dst_r)
    else:
        # Intermediate is destination (shouldn't happen in valid Valiant)
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
    threshold: float = 2.0
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
        a: Number of global links per router
        threshold: Decision threshold (default 2.0, standard UGAL)

    Returns:
        Selected path as list of node names
    """
    num_groups = a + 1
    src_group, _, _ = parse_dragonfly_host(src_host)
    dst_group, _, _ = parse_dragonfly_host(dst_host)

    # Compute minimal path and its latency
    minimal_path = dragonfly_minimal_path(src_host, dst_host, d, a)
    minimal_latency = estimate_path_latency(minimal_path, link_loads)

    # For intra-group traffic, always use minimal (no benefit from non-minimal)
    if src_group == dst_group:
        return minimal_path

    # Evaluate non-minimal paths through each intermediate group
    best_nonminimal_path = None
    best_nonminimal_latency = float('inf')

    for inter_group in range(num_groups):
        # Skip source and destination groups (not valid intermediate)
        if inter_group == src_group or inter_group == dst_group:
            continue

        nonminimal_path = dragonfly_nonminimal_path(
            src_host, dst_host, inter_group, d, a
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
    bias: float = 0.0
) -> list[str]:
    """
    Valiant routing with configurable bias toward non-minimal paths.

    Args:
        src_host: Source host name
        dst_host: Destination host name
        d: Number of routers per group
        a: Number of global links per router
        bias: Fraction of traffic to route non-minimally (0.0-1.0)
              0.0 = always minimal, 1.0 = always non-minimal
              0.05 = 5% non-minimal, 95% minimal

    Returns:
        Selected path as list of node names
    """
    num_groups = a + 1
    src_group, _, _ = parse_dragonfly_host(src_host)
    dst_group, _, _ = parse_dragonfly_host(dst_host)

    # Intra-group: always minimal (non-minimal makes no sense)
    if src_group == dst_group:
        return dragonfly_minimal_path(src_host, dst_host, d, a)

    # Probabilistic selection based on bias
    if random.random() >= bias:
        # Use minimal path (1 - bias probability)
        return dragonfly_minimal_path(src_host, dst_host, d, a)
    else:
        # Use non-minimal path via random intermediate group
        valid_intermediates = [
            g for g in range(num_groups)
            if g != src_group and g != dst_group
        ]
        if not valid_intermediates:
            return dragonfly_minimal_path(src_host, dst_host, d, a)

        inter_group = random.choice(valid_intermediates)
        return dragonfly_nonminimal_path(src_host, dst_host, inter_group, d, a)


def dragonfly_route(
    src_host: str,
    dst_host: str,
    algorithm: str,
    d: int,
    a: int,
    link_loads: dict | None = None,
    ugal_threshold: float = 2.0,
    valiant_bias: float = 0.0
) -> list[str]:
    """
    Main routing dispatcher for Dragonfly networks.

    Args:
        src_host: Source host name
        dst_host: Destination host name
        algorithm: Routing algorithm ('minimal', 'ugal', 'valiant')
        d: Number of routers per group
        a: Number of global links per router
        link_loads: Current link loads (required for UGAL)
        ugal_threshold: UGAL decision threshold
        valiant_bias: Valiant non-minimal bias (0.0-1.0)

    Returns:
        Path as list of node names
    """
    if algorithm == 'minimal':
        return dragonfly_minimal_path(src_host, dst_host, d, a)

    elif algorithm == 'ugal':
        if link_loads is None:
            link_loads = {}
        return ugal_select_path(
            src_host, dst_host, link_loads, d, a, ugal_threshold
        )

    elif algorithm == 'valiant':
        return valiant_select_path(src_host, dst_host, d, a, valiant_bias)

    else:
        # Default to minimal
        return dragonfly_minimal_path(src_host, dst_host, d, a)


def link_loads_for_job_dragonfly_adaptive(
    G: nx.Graph,
    job_hosts: list[str],
    tx_volume_bytes: float,
    algorithm: str,
    d: int,
    a: int,
    link_loads: dict | None = None,
    ugal_threshold: float = 2.0,
    valiant_bias: float = 0.0
) -> dict:
    """
    Compute link loads for a job using adaptive routing on Dragonfly.

    Args:
        G: NetworkX graph (for edge initialization)
        job_hosts: List of host names for this job
        tx_volume_bytes: Traffic volume per host
        algorithm: Routing algorithm ('minimal', 'ugal', 'valiant')
        d: Number of routers per group
        a: Number of global links per router
        link_loads: Global link loads (for UGAL decisions)
        ugal_threshold: UGAL decision threshold
        valiant_bias: Valiant non-minimal bias

    Returns:
        Dict {(u, v): bytes} of link loads from this job
    """
    job_loads = {tuple(sorted(edge)): 0.0 for edge in G.edges()}

    if len(job_hosts) < 2:
        return job_loads

    # All-to-all traffic: each host sends to every other host
    per_peer = tx_volume_bytes / (len(job_hosts) - 1)

    for src in job_hosts:
        for dst in job_hosts:
            if src == dst:
                continue

            path = dragonfly_route(
                src, dst, algorithm, d, a,
                link_loads=link_loads,
                ugal_threshold=ugal_threshold,
                valiant_bias=valiant_bias
            )

            for u, v in zip(path, path[1:]):
                edge = tuple(sorted((u, v)))
                if edge in job_loads:
                    job_loads[edge] += per_peer

    return job_loads

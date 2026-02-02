import csv
import math
import networkx as nx
from pathlib import Path
from raps.job import CommunicationPattern, normalize_comm_pattern


def build_torus3d(
    dims,
    wrap=True,
    hosts_per_router: int = 1,
    torus_link_bw: float = None,
    latency_per_hop: float = None,
    network_max_bw: float = None,
):
    """
    Build a 3D torus network (routers + hosts).
    Each router r_x_y_z connects to 6 neighbors (±X, ±Y, ±Z)
    and attaches hosts h_x_y_z_p for p ∈ [0..hosts_per_router-1].

    Returns:
        (G, meta) where:
          - G: networkx.Graph
          - meta: dict with topology info for plotting/simulation
    """
    X, Y, Z = dims
    G = nx.Graph()

    # --- Add routers with normalized coordinates ---
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                name = f"r_{x}_{y}_{z}"
                G.add_node(
                    name,
                    type="router",
                    x=x / (X - 1 if X > 1 else 1),
                    y=y / (Y - 1 if Y > 1 else 1),
                    z=z / (Z - 1 if Z > 1 else 1),
                )

    # --- Add wrap-around router-to-router edges ---
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                src = f"r_{x}_{y}_{z}"

                nx_ = (x + 1) % X if wrap else x + 1
                if nx_ < X:
                    G.add_edge(
                        src, f"r_{nx_}_{y}_{z}",
                        bandwidth=torus_link_bw,
                        latency=latency_per_hop,
                        type="router_link"
                    )

                ny_ = (y + 1) % Y if wrap else y + 1
                if ny_ < Y:
                    G.add_edge(
                        src, f"r_{x}_{ny_}_{z}",
                        bandwidth=torus_link_bw,
                        latency=latency_per_hop,
                        type="router_link"
                    )

                nz_ = (z + 1) % Z if wrap else z + 1
                if nz_ < Z:
                    G.add_edge(
                        src, f"r_{x}_{y}_{nz_}",
                        bandwidth=torus_link_bw,
                        latency=latency_per_hop,
                        type="router_link"
                    )

    # --- Add hosts and host-router edges ---
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                router = f"r_{x}_{y}_{z}"
                for p in range(hosts_per_router):
                    host = f"h_{x}_{y}_{z}_{p}"
                    G.add_node(
                        host,
                        type="host",
                        x=(x + 0.1) / (X - 1 if X > 1 else 1),
                        y=(y + 0.1) / (Y - 1 if Y > 1 else 1),
                        z=(z + 0.1 * (p + 1)) / (Z - 1 if Z > 1 else 1),
                    )
                    G.add_edge(
                        host, router,
                        bandwidth=network_max_bw,
                        latency=latency_per_hop,
                        type="host_link"
                    )

    # --- Build host <-> router mappings for simulator use ---
    host_to_router = {}
    router_to_hosts = {}

    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                router = f"r_{x}_{y}_{z}"
                router_to_hosts[router] = []
                for p in range(hosts_per_router):
                    host = f"h_{x}_{y}_{z}_{p}"
                    host_to_router[host] = router
                    router_to_hosts[router].append(host)

    meta = {
        "topology": "torus3d",
        "dims": (X, Y, Z),
        "hosts_per_router": hosts_per_router,
        "wrap": wrap,
        "num_routers": X * Y * Z,
        "num_hosts": X * Y * Z * hosts_per_router,
        "host_to_router": host_to_router,
        "router_to_hosts": router_to_hosts,
    }

    print(f"Built 3D torus with {meta['num_routers']} routers and {meta['num_hosts']} hosts.")
    return G, meta


def _axis_steps(a, b, n, wrap=True):
    """Return minimal step sequence along one axis from a to b with wrap-around."""
    if a == b:
        return []
    fwd = (b - a) % n
    back = (a - b) % n
    if not wrap:
        step = 1 if b > a else -1
        return [step] * abs(b - a)
    if fwd <= back:
        return [1] * fwd
    else:
        return [-1] * back


def torus_route_xyz(src_r, dst_r, dims, wrap=True):
    """Router-level path (list of router names) using XYZ dimension-order routing."""
    X, Y, Z = dims

    def parse(r):
        _, x, y, z = r.split("_")
        return int(x), int(y), int(z)

    x1, y1, z1 = parse(src_r)
    x2, y2, z2 = parse(dst_r)

    path = [src_r]
    x, y, z = x1, y1, z1
    for step in _axis_steps(x, x2, X, wrap):
        x = (x + step) % X
        path.append(f"r_{x}_{y}_{z}")
    for step in _axis_steps(y, y2, Y, wrap):
        y = (y + step) % Y
        path.append(f"r_{x}_{y}_{z}")
    for step in _axis_steps(z, z2, Z, wrap):
        z = (z + step) % Z
        path.append(f"r_{x}_{y}_{z}")
    return path


def torus_host_path(G, meta, h_src, h_dst):
    r_src = meta["host_to_router"][h_src]
    r_dst = meta["host_to_router"][h_dst]
    routers = torus_route_xyz(r_src, r_dst, meta["dims"], meta["wrap"])
    # host->src_router + (router path) + dst_router->host
    path = [h_src, r_src] + routers[1:] + [h_dst]
    return path


def factorize_3d(n):
    """
    Factorize n into three dimensions (x, y, z) for a virtual 3D grid.
    Tries to make dimensions as equal as possible.
    Returns (x, y, z) where x * y * z >= n.
    """
    if n <= 0:
        return (1, 1, 1)

    cube_root = int(math.ceil(n ** (1/3)))
    best = (n, 1, 1)
    best_diff = n

    for x in range(1, cube_root + 2):
        for y in range(1, cube_root + 2):
            z = int(math.ceil(n / (x * y)))
            if x * y * z >= n:
                diff = max(x, y, z) - min(x, y, z)
                if diff < best_diff or (diff == best_diff and x * y * z < best[0] * best[1] * best[2]):
                    best = (x, y, z)
                    best_diff = diff

    return best


def get_stencil_3d_neighbors(node_idx, dims, num_nodes):
    """
    Get the 6 neighbors for a node in a 3D stencil pattern.
    Uses periodic boundary conditions (wraps around).
    """
    x_dim, y_dim, z_dim = dims

    z = node_idx // (x_dim * y_dim)
    remainder = node_idx % (x_dim * y_dim)
    y = remainder // x_dim
    x = remainder % x_dim

    neighbors = []
    neighbor_offsets = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ]

    for dx, dy, dz in neighbor_offsets:
        nx = (x + dx) % x_dim
        ny = (y + dy) % y_dim
        nz = (z + dz) % z_dim
        neighbor_idx = nz * (x_dim * y_dim) + ny * x_dim + nx
        if neighbor_idx < num_nodes and neighbor_idx != node_idx:
            neighbors.append(neighbor_idx)

    return neighbors


def stencil_3d_pairs(host_list):
    """Generate (src, dst) pairs for a 3D stencil pattern."""
    num_nodes = len(host_list)
    dims = factorize_3d(num_nodes)
    pairs = []
    for idx, host in enumerate(host_list):
        neighbors = get_stencil_3d_neighbors(idx, dims, num_nodes)
        for neighbor_idx in neighbors:
            pairs.append((host, host_list[neighbor_idx]))
    return pairs


def link_loads_for_job_torus(G, meta, host_list, traffic_bytes, *, comm_pattern=None):
    """
    Distribute traffic_bytes using torus routing, respecting comm_pattern.
    """
    loads = {edge: 0.0 for edge in G.edges()}
    comm_pattern = normalize_comm_pattern(comm_pattern or CommunicationPattern.ALL_TO_ALL)
    n = len(host_list)

    if n <= 1:
        return loads

    if comm_pattern == CommunicationPattern.ALL_TO_ALL:
        for src in host_list:
            per_peer = traffic_bytes / (n - 1)
            for dst in host_list:
                if dst == src:
                    continue
                p = torus_host_path(G, meta, src, dst)
                for u, v in zip(p, p[1:]):
                    e = tuple(sorted((u, v)))
                    loads[e] += per_peer
        return loads

    if comm_pattern == CommunicationPattern.STENCIL_3D:
        pairs = stencil_3d_pairs(host_list)
        neighbor_count = {}
        for src, _ in pairs:
            neighbor_count[src] = neighbor_count.get(src, 0) + 1
        for src, dst in pairs:
            per_neighbor = traffic_bytes / neighbor_count.get(src, 1)
            p = torus_host_path(G, meta, src, dst)
            for u, v in zip(p, p[1:]):
                e = tuple(sorted((u, v)))
                loads[e] += per_neighbor
        return loads

    raise ValueError(f"Unsupported comm_pattern: {comm_pattern}")


def torus_host_from_real_index(real_n, X, Y, Z, hosts_per_router):
    total_hosts = X * Y * Z * hosts_per_router
    idx = real_n % total_hosts
    r = idx // hosts_per_router
    h = idx % hosts_per_router
    z = r % Z
    y = (r // Z) % Y
    x = (r // (Y * Z)) % X
    return f"h_{x}_{y}_{z}_{h}"

import os
import warnings

import networkx as nx
from raps.job import CommunicationPattern

# Module-level cache: maps (fattree_k, total_nodes) -> {(src, dst): [paths]}
# Shared across all NetworkModel instances in the same process so that
# sequential UC simulations (different policies/allocations but same topology)
# do not re-compute nx.all_shortest_paths from scratch each time.
_FATTREE_TOPOLOGY_CACHES: dict = {}


class LazyAPSP:
    """Lazy all-pairs shortest path cache.

    Computes and caches shortest paths on demand instead of pre-computing
    all pairs upfront. This avoids the O(V^2) startup cost while still
    giving O(1) lookups on cache hits.
    """

    def __init__(self, graph):
        self._graph = graph
        self._cache = {}  # {src: {dst: path}}

    def __getitem__(self, src):
        if src not in self._cache:
            self._cache[src] = {}
        return _LazySrcPaths(self._graph, src, self._cache[src])

    def get(self, src, default=None):
        try:
            return self[src]
        except KeyError:
            return default


class _LazySrcPaths:
    """Lazy per-source path lookup. Computes paths on first access."""

    def __init__(self, graph, src, cache_dict):
        self._graph = graph
        self._src = src
        self._cache = cache_dict

    def __getitem__(self, dst):
        if dst not in self._cache:
            self._cache[dst] = nx.shortest_path(self._graph, self._src, dst)
        return self._cache[dst]

    def get(self, dst, default=None):
        try:
            return self[dst]
        except (KeyError, nx.NetworkXNoPath):
            return default

from .base import (
    all_to_all_paths,
    apply_job_slowdown,
    compute_link_stall_packet_stats,
    aggregate_link_stall_stats,
    compute_stall_ratio,
    compute_all_to_all_coefficients,
    compute_stencil_3d_coefficients,
    compute_system_network_stats,
    link_loads_for_job,
    link_loads_for_job_stencil_3d,
    link_loads_for_pattern,
    get_effective_traffic,
    apply_message_size_overhead,
    factorize_3d,
    stencil_3d_pairs,
    network_congestion,
    network_slowdown,
    network_utilization,
    worst_link_util,
    get_link_util_stats,
    simulate_inter_job_congestion,
    max_throughput_per_tick,
)

from .fat_tree import build_fattree, node_id_to_host_name, subsample_hosts
from .torus3d import build_torus3d, link_loads_for_job_torus, torus_host_from_real_index
from .dragonfly import build_dragonfly, dragonfly_node_id_to_host_name, build_dragonfly_idx_map
from raps.plotting import plot_fattree_hierarchy, plot_dragonfly, plot_torus2d, plot_torus3d

from raps.utils import get_current_utilization

__all__ = [
    "NetworkModel",
    "apply_job_slowdown",
    "compute_link_stall_packet_stats",
    "aggregate_link_stall_stats",
    "compute_stall_ratio",
    "compute_system_network_stats",
    "network_congestion",
    "network_utilization",
    "network_slowdown",
    "all_to_all_paths",
    "link_loads_for_job",
    "link_loads_for_job_stencil_3d",
    "link_loads_for_pattern",
    "get_effective_traffic",
    "apply_message_size_overhead",
    "factorize_3d",
    "stencil_3d_pairs",
    "worst_link_util",
    "build_fattree",
    "build_torus3d",
    "build_dragonfly",
    "dragonfly_node_id_to_host_name",
    "simulate_inter_job_congestion",
    "max_throughput_per_tick",
    "get_link_util_stats",
]


class NetworkModel:
    def __init__(self, *, available_nodes, config, **kwargs):
        self.config = config
        self.topology = config.get("TOPOLOGY")
        self.max_link_bw = config.get("NETWORK_MAX_BW", 1e9)  # default safeguard
        self.real_to_fat_idx = kwargs.get("real_to_fat_idx", {})

        # Routing algorithm configuration
        self.routing_algorithm = config.get("ROUTING_ALGORITHM", "minimal")
        self.ugal_threshold = config.get("UGAL_THRESHOLD", 2.0)
        self.valiant_bias = config.get("VALIANT_BIAS", 0.0)

        # Global link loads for adaptive routing (reset each tick)
        self.global_link_loads = {}

        if self.topology == "fat-tree":
            total_nodes = config['TOTAL_NODES'] - len(config['DOWN_NODES'])
            self.fattree_k = config.get("FATTREE_K")
            self.net_graph = build_fattree(self.fattree_k, total_nodes)
            # TODO: future testing of subsampling feature
            #self.net_graph = subsample_hosts(self.net_graph, num_hosts=4626)

            # Initialize global link loads for adaptive routing
            self.global_link_loads = {tuple(sorted(edge)): 0.0 for edge in self.net_graph.edges()}

            # Cache for nx.all_shortest_paths results (keyed by (src, dst)).
            # Use the module-level topology cache so that multiple sequential
            # simulations with the same fat-tree (same k, same node count) share
            # already-computed paths and skip expensive re-warmup.
            cache_key = (self.fattree_k, total_nodes)
            self._fattree_paths_cache = _FATTREE_TOPOLOGY_CACHES.setdefault(cache_key, {})

            routing_info = f"routing={self.routing_algorithm}"
            print(f"[DEBUG] Fat-tree k={self.fattree_k}: {total_nodes} nodes, {routing_info}")

        elif self.topology == "torus3d":
            dims = (
                int(config["TORUS_X"]),
                int(config["TORUS_Y"]),
                int(config["TORUS_Z"])
            )
            wrap = bool(config.get("TORUS_WRAP", True))
            hosts_per_router = int(config.get("HOSTS_PER_ROUTER", config.get("hosts_per_router", 1)))

            # Build the graph and metadata
            self.net_graph, self.meta = build_torus3d(dims, wrap, hosts_per_router=hosts_per_router)

            # Deterministic numeric → host mapping
            X, Y, Z = self.meta["dims"]
            self.id_to_host = {}
            nid = 0
            for x in range(X):
                for y in range(Y):
                    for z in range(Z):
                        for i in range(hosts_per_router):
                            h = f"h_{x}_{y}_{z}_{i}"
                            self.id_to_host[nid] = h
                            nid += 1

        elif self.topology == "dragonfly":
            D = self.config["DRAGONFLY_D"]
            A = self.config["DRAGONFLY_A"]
            P = self.config["DRAGONFLY_P"]
            self.net_graph = build_dragonfly(D, A, P)

            # Store dragonfly params for routing
            self.dragonfly_d = D
            self.dragonfly_a = A
            self.dragonfly_p = P

            # Use TOTAL_NODES (full node ID range) so all schedulable node IDs
            # are covered. available_nodes has gaps from missing racks / down_nodes
            # (e.g. Frontier: 9472 available out of IDs 0..9599), so using
            # len(available_nodes) would miss high-ID nodes and cause KeyError.
            total_real_nodes = config.get('TOTAL_NODES')
            if total_real_nodes is None:
                if available_nodes is None:
                    total_real_nodes = 4626  # fallback
                elif isinstance(available_nodes, int):
                    total_real_nodes = available_nodes
                else:
                    total_real_nodes = len(available_nodes)

            self.real_to_fat_idx = build_dragonfly_idx_map(D, A, P, total_real_nodes)

            # Initialize global link loads for adaptive routing
            self.global_link_loads = {tuple(sorted(edge)): 0.0 for edge in self.net_graph.edges()}

            routing_info = f"routing={self.routing_algorithm}"
            if self.routing_algorithm == 'ugal':
                routing_info += f", threshold={self.ugal_threshold}"
            elif self.routing_algorithm == 'valiant':
                routing_info += f", bias={self.valiant_bias}"
            print(f"[DEBUG] Dragonfly: {len(self.real_to_fat_idx)} nodes, {routing_info}")

        elif self.topology == "capacity":
            # Capacity-only model: no explicit graph
            self.net_graph = None

        else:
            raise ValueError(f"Unsupported topology: {self.topology}")

        # Lazy shortest-path cache: paths are computed on first access
        if self.net_graph is not None:
            self._apsp = LazyAPSP(self.net_graph)
        else:
            self._apsp = None

        # Per-job caches: host list mapping and computed paths
        self._job_host_cache = {}  # frozenset(scheduled_nodes) -> host_list
        # Cached normalized link-load coefficients per job (for deterministic routing)
        self._job_load_coeffs = {}  # job.id -> {edge: coefficient}

    def get_job_hosts(self, job):
        """Get cached host list for a job's scheduled nodes."""
        key = frozenset(job.scheduled_nodes)
        if key not in self._job_host_cache:
            if self.topology == "fat-tree":
                self._job_host_cache[key] = [node_id_to_host_name(n, self.fattree_k) for n in job.scheduled_nodes]
            elif self.topology == "dragonfly":
                self._job_host_cache[key] = [self.real_to_fat_idx[n] for n in job.scheduled_nodes]
            elif self.topology == "torus3d":
                X = self.config["TORUS_X"]
                Y = self.config["TORUS_Y"]
                Z = self.config["TORUS_Z"]
                hosts_per_router = self.config["HOSTS_PER_ROUTER"]
                self._job_host_cache[key] = [
                    torus_host_from_real_index(n, X, Y, Z, hosts_per_router)
                    for n in job.scheduled_nodes
                ]
            else:
                self._job_host_cache[key] = list(job.scheduled_nodes)
        return self._job_host_cache[key]

    def clear_job_cache(self, job):
        """Remove cached data for a completed job."""
        key = frozenset(job.scheduled_nodes)
        self._job_host_cache.pop(key, None)
        self._job_load_coeffs.pop(job.id, None)

    def _uses_adaptive_routing(self):
        """Check whether current routing algorithm requires per-tick path recomputation."""
        if self.topology == "fat-tree":
            return self.routing_algorithm in ('ecmp', 'adaptive')
        elif self.topology == "dragonfly":
            return self.routing_algorithm in ('ugal', 'valiant')
        return False

    def _compute_and_cache_coefficients(self, job, host_list, comm_pattern):
        """Compute link-load coefficients for a job and cache them.

        For deterministic routing (minimal, shortest-path), paths are fixed
        for a given set of scheduled_nodes, so coefficients only need to be
        computed once per job lifetime.
        """
        from raps.job import normalize_comm_pattern
        comm = normalize_comm_pattern(comm_pattern)

        if comm == CommunicationPattern.STENCIL_3D:
            coeffs = compute_stencil_3d_coefficients(
                self.net_graph, host_list, apsp=self._apsp)
        else:
            coeffs = compute_all_to_all_coefficients(
                self.net_graph, host_list, apsp=self._apsp)

        self._job_load_coeffs[job.id] = coeffs
        return coeffs

    def simulate_network_utilization(self, *, job, debug=False):
        net_util = net_cong = net_tx = net_rx = 0
        max_throughput = self.max_link_bw * job.trace_quanta

        if job.nodes_required <= 1:
            # Single node job, skip network impact
            return net_util, net_cong, net_tx, net_rx, max_throughput

        net_tx = get_current_utilization(job.ntx_trace, job)
        net_rx = get_current_utilization(job.nrx_trace, job)

        # Get communication pattern and message size from job
        comm_pattern = getattr(job, 'comm_pattern', CommunicationPattern.ALL_TO_ALL)
        message_size = getattr(job, 'message_size', None)

        # Apply message size overhead if specified
        num_hosts = len(job.scheduled_nodes)
        effective_tx = get_effective_traffic(net_tx, job, num_hosts)
        effective_rx = get_effective_traffic(net_rx, job, num_hosts)

        net_util = network_utilization(effective_tx, effective_rx, max_throughput)

        if debug:
            print(f"  comm_pattern: {comm_pattern}, message_size: {message_size}")
            print(f"  raw tx/rx: {net_tx}/{net_rx}, effective tx/rx: {effective_tx}/{effective_rx}")

        # -----------------------------------------------------------
        # Fast path: use cached coefficients for deterministic routing
        # -----------------------------------------------------------
        can_cache = (
            self.net_graph is not None
            and self.topology in ("fat-tree", "dragonfly")
            and not self._uses_adaptive_routing()
        )

        if can_cache:
            coeffs = self._job_load_coeffs.get(job.id)
            if coeffs is None:
                host_list = self.get_job_hosts(job)
                coeffs = self._compute_and_cache_coefficients(
                    job, host_list, comm_pattern)

            # Scale cached coefficients by current traffic volume
            if coeffs:
                max_load = 0.0
                for coeff in coeffs.values():
                    load = coeff * effective_tx
                    byte_util = load / max_throughput
                    if byte_util > max_load:
                        max_load = byte_util
                net_cong = max_load

                # Update global link loads (for inter-job congestion)
                for edge, coeff in coeffs.items():
                    edge_key = tuple(sorted(edge))
                    if edge_key in self.global_link_loads:
                        self.global_link_loads[edge_key] += coeff * effective_tx

            return net_util, net_cong, net_tx, net_rx, max_throughput

        # -----------------------------------------------------------
        # Slow path: adaptive routing or unsupported topology
        # -----------------------------------------------------------
        if self.topology == "fat-tree":
            host_list = self.get_job_hosts(job)
            if debug:
                print("  fat-tree hosts:", host_list)
                print(f"  routing: {self.routing_algorithm}")

            loads = link_loads_for_pattern(
                self.net_graph,
                host_list,
                effective_tx,
                comm_pattern,
                routing_algorithm=self.routing_algorithm,
                link_loads=self.global_link_loads,
                apsp=self._apsp,
                fattree_params={'paths_cache': getattr(self, '_fattree_paths_cache', None)},
            )
            net_cong = worst_link_util(loads, max_throughput)

            # Update global link loads for adaptive routing decisions
            if self.routing_algorithm in ('ecmp', 'adaptive'):
                for edge, load in loads.items():
                    edge_key = tuple(sorted(edge))
                    if edge_key in self.global_link_loads:
                        self.global_link_loads[edge_key] += load

        elif self.topology == "dragonfly":
            D = self.config["DRAGONFLY_D"]
            A = self.config["DRAGONFLY_A"]
            P = self.config["DRAGONFLY_P"]
            # Directly use mapped host names (cached)
            host_list = self.get_job_hosts(job)
            if debug:
                print("  dragonfly hosts:", host_list)
                print(f"  routing: {self.routing_algorithm}")
                print("Example nodes in graph:", list(self.net_graph.nodes)[:10])

            # Build dragonfly params for adaptive routing
            dragonfly_params = {
                'd': D,
                'a': A,
                'ugal_threshold': self.ugal_threshold,
                'valiant_bias': self.valiant_bias,
            }

            loads = link_loads_for_pattern(
                self.net_graph,
                host_list,
                effective_tx,
                comm_pattern,
                routing_algorithm=self.routing_algorithm,
                dragonfly_params=dragonfly_params,
                link_loads=self.global_link_loads,
                apsp=self._apsp,
            )
            net_cong = worst_link_util(loads, max_throughput)

            # Update global link loads for UGAL decisions
            if self.routing_algorithm in ('ugal', 'valiant'):
                for edge, load in loads.items():
                    edge_key = tuple(sorted(edge))
                    if edge_key in self.global_link_loads:
                        self.global_link_loads[edge_key] += load

        elif self.topology == "torus3d":
            host_list = self.get_job_hosts(job)
            loads = link_loads_for_job_torus(
                self.net_graph,
                self.meta,
                host_list,
                effective_tx,
                comm_pattern=comm_pattern,
            )
            net_cong = worst_link_util(loads, max_throughput)
            if debug:
                print("  torus3d hosts:", host_list)

        elif self.topology == "capacity":
            net_cong = network_congestion(effective_tx, effective_rx, max_throughput)

        else:
            raise ValueError(f"Unsupported topology: {self.topology}")

        return net_util, net_cong, net_tx, net_rx, max_throughput

    def compute_tick_stall_stats(self, *, mean_pkt_size_bytes, dt, avg_slowdown=1.0):
        """
        Compute system-level stall/packet stats for the current tick using
        accumulated global_link_loads.

        Args:
            mean_pkt_size_bytes: Mean packet size in bytes (from config)
            dt: Tick duration in seconds
            avg_slowdown: System-average slowdown factor for the tick

        Returns:
            dict with 'total_posted_pkts', 'total_tx_paused', 'system_stall_ratio'
        """
        if not self.global_link_loads:
            return {'total_posted_pkts': 0.0, 'total_tx_paused': 0.0, 'system_stall_ratio': 0.0}
        link_stats = compute_link_stall_packet_stats(
            self.global_link_loads,
            self.max_link_bw * 8,      # convert bytes/s → bits/s
            mean_pkt_size_bytes,
            dt,
            avg_slowdown,
        )
        return aggregate_link_stall_stats(link_stats)

    def dump_link_loads(self, path: str, *, dt: float | None = None) -> None:
        """Write current global_link_loads to a CSV file (src, dst, bytes).

        The resulting CSV can be fed directly to
        ``scripts/plot_dragonfly_congestion.py``.

        Parameters
        ----------
        path : str | Path
            Destination CSV file path.
        dt   : float, optional
            Simulation timestep (seconds).  If supplied it is written to a
            header comment so the plot script can auto-detect it.
        """
        import csv as _csv
        from pathlib import Path as _Path
        _Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', newline='') as f:
            if dt is not None:
                f.write(f"# dt={dt}\n")
            writer = _csv.writer(f)
            writer.writerow(['src', 'dst', 'bytes'])
            for (u, v), b in self.global_link_loads.items():
                if b > 0:
                    writer.writerow([u, v, f'{b:.0f}'])

    def reset_link_loads(self):
        """Reset global link loads at the start of each simulation tick."""
        if self.net_graph is not None:
            self.global_link_loads = {
                tuple(sorted(edge)): 0.0 for edge in self.net_graph.edges()
            }

    def plot_topology(self, output_dir):
        """Plot network topology - save as png file in output_dir."""
        if output_dir:
            if self.topology == "fat-tree":
                save_path = output_dir / "net-fat-tree.png"
                plot_fattree_hierarchy(self.net_graph, k=self.fattree_k, save_path=save_path)
            elif self.topology == "dragonfly":
                save_path = output_dir / "net-dragonfly.png"
                plot_dragonfly(self.net_graph, save_path=save_path)
            elif self.topology == "torus3d":
                save_path = output_dir / "net-torus2d.png"
                plot_torus2d(self.net_graph, save_path=save_path)
                save_path = output_dir / "net-torus3d.png"
                plot_torus3d(self.net_graph, save_path=save_path)
            else:
                warnings.warn(
                    f"plotting not supported for {self.topology} topology",
                    UserWarning
                )

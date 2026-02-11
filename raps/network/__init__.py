import os
import warnings

import networkx as nx
from raps.job import CommunicationPattern


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

            # total nodes seen by scheduler or job trace
            total_real_nodes = getattr(self, "available_nodes", None)
            if total_real_nodes is None:
                total_real_nodes = 4626  # fallback for Lassen

            # if available_nodes is a list, take its length
            if not isinstance(total_real_nodes, int):
                total_real_nodes = len(total_real_nodes)

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

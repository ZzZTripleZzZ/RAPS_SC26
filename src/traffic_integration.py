#!/usr/bin/env python3
"""
SC26 Traffic Integration Module
================================
Bridges mini-app traffic matrices with RAPS simulation.

This module:
1. Infers communication patterns from traffic matrices
2. Converts traffic matrices to RAPS-compatible formats
3. Creates custom Jobs with realistic communication patterns
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

# Add RAPS to path
sys.path.insert(0, str(Path("/app/extern/raps")))

try:
    from raps.job import CommunicationPattern, job_dict, MESSAGE_SIZE_64K
    from raps.network.base import (
        link_loads_for_pattern,
        link_loads_for_job,
        get_stencil_3d_neighbors,
        factorize_3d,
    )
    RAPS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RAPS import failed: {e}")
    RAPS_AVAILABLE = False

    class CommunicationPattern(Enum):
        ALL_TO_ALL = "all-to-all"
        STENCIL_3D = "stencil-3d"


class InferredPattern(Enum):
    """Extended communication patterns inferred from traffic matrices."""
    ALL_TO_ALL = "all-to-all"       # Dense, every node communicates with all others
    STENCIL_3D = "stencil-3d"       # Each node has ~6 neighbors
    NEAREST_NEIGHBOR = "nearest-neighbor"  # Sparse, only adjacent nodes
    HIERARCHICAL = "hierarchical"   # Tree-like pattern (e.g., multigrid)
    RING = "ring"                   # Ring topology
    SPARSE_RANDOM = "sparse-random" # Sparse but irregular


def analyze_traffic_pattern(traffic_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Analyze a traffic matrix to infer communication pattern.

    Args:
        traffic_matrix: 2D numpy array (src x dst) of bytes transferred

    Returns:
        Dict with pattern type, statistics, and recommendations
    """
    n = traffic_matrix.shape[0]
    total_traffic = np.sum(traffic_matrix)

    if total_traffic == 0:
        return {
            "pattern": InferredPattern.SPARSE_RANDOM,
            "sparsity": 1.0,
            "avg_degree": 0,
            "is_symmetric": True,
            "raps_pattern": CommunicationPattern.ALL_TO_ALL
        }

    # Remove self-communication
    np.fill_diagonal(traffic_matrix, 0)

    # Calculate statistics
    nonzero = traffic_matrix > 0
    num_edges = np.sum(nonzero)
    max_edges = n * (n - 1)
    sparsity = 1.0 - (num_edges / max_edges) if max_edges > 0 else 1.0

    # Calculate degree distribution
    out_degree = np.sum(nonzero, axis=1)  # Number of destinations per source
    in_degree = np.sum(nonzero, axis=0)   # Number of sources per destination
    avg_out_degree = np.mean(out_degree)
    avg_in_degree = np.mean(in_degree)

    # Check symmetry
    symmetric_traffic = traffic_matrix + traffic_matrix.T
    is_symmetric = np.allclose(traffic_matrix, traffic_matrix.T, rtol=0.1)

    # Determine pattern
    if sparsity < 0.3:
        # Dense communication - likely ALL_TO_ALL
        pattern = InferredPattern.ALL_TO_ALL
        raps_pattern = CommunicationPattern.ALL_TO_ALL
    elif 5 <= avg_out_degree <= 8 and is_symmetric:
        # ~6 neighbors, symmetric - likely STENCIL_3D
        pattern = InferredPattern.STENCIL_3D
        raps_pattern = CommunicationPattern.STENCIL_3D
    elif avg_out_degree <= 2 and is_symmetric:
        # Ring-like
        pattern = InferredPattern.RING
        raps_pattern = CommunicationPattern.STENCIL_3D  # Approximate with stencil
    elif _check_hierarchical(traffic_matrix, n):
        pattern = InferredPattern.HIERARCHICAL
        raps_pattern = CommunicationPattern.ALL_TO_ALL  # Use all-to-all for now
    else:
        pattern = InferredPattern.SPARSE_RANDOM
        raps_pattern = CommunicationPattern.ALL_TO_ALL

    # Calculate message statistics
    nonzero_traffic = traffic_matrix[nonzero]
    avg_message_size = np.mean(nonzero_traffic) if len(nonzero_traffic) > 0 else 0
    total_messages = len(nonzero_traffic)

    return {
        "pattern": pattern,
        "raps_pattern": raps_pattern,
        "sparsity": sparsity,
        "avg_out_degree": avg_out_degree,
        "avg_in_degree": avg_in_degree,
        "is_symmetric": is_symmetric,
        "total_traffic_bytes": total_traffic,
        "total_messages": total_messages,
        "avg_message_size": avg_message_size,
        "num_ranks": n
    }


def _check_hierarchical(traffic_matrix: np.ndarray, n: int) -> bool:
    """Check if traffic pattern looks hierarchical (tree-like)."""
    # Simple heuristic: check if there are "hub" nodes with high degree
    out_degree = np.sum(traffic_matrix > 0, axis=1)
    in_degree = np.sum(traffic_matrix > 0, axis=0)
    total_degree = out_degree + in_degree

    if n < 4:
        return False

    # Check for high variance in degree (some nodes are hubs)
    degree_std = np.std(total_degree)
    degree_mean = np.mean(total_degree)

    return degree_std > degree_mean * 0.5  # High variance suggests hierarchy


def traffic_matrix_to_link_loads(
    traffic_matrix: np.ndarray,
    graph,
    host_mapping: Dict[int, str],
    routing_algorithm: str = "minimal",
    dragonfly_params: Optional[Dict] = None,
    fattree_params: Optional[Dict] = None,
    link_loads: Optional[Dict] = None,
) -> Dict[Tuple[str, str], float]:
    """
    Convert a traffic matrix directly to link loads.

    This bypasses the RAPS communication pattern abstraction and uses
    the actual traffic matrix for more accurate simulation.

    Args:
        traffic_matrix: 2D array of bytes transferred
        link_loads: Optional existing link loads (for adaptive routing with interference)
        graph: NetworkX graph of the topology
        host_mapping: Dict mapping rank index to host name
        routing_algorithm: 'minimal', 'ugal', 'valiant', 'ecmp'
        dragonfly_params: Dragonfly routing parameters
        fattree_params: Fat-tree routing parameters

    Returns:
        Dict mapping edges to total bytes
    """
    import networkx as nx

    n = traffic_matrix.shape[0]

    # Use provided link_loads or create new one
    if link_loads is None:
        link_loads = {tuple(sorted(e)): 0.0 for e in graph.edges()}
    else:
        # Make a copy to avoid modifying the input
        link_loads = link_loads.copy()

    # Get routing function based on algorithm
    if routing_algorithm in ("ugal", "valiant") and dragonfly_params:
        from raps.network.dragonfly import ugal_select_path, valiant_select_path, dragonfly_minimal_path

        d = dragonfly_params.get('d', 16)
        a = dragonfly_params.get('a', 8)
        threshold = dragonfly_params.get('ugal_threshold', 2.0)
        bias = dragonfly_params.get('valiant_bias', 0.05)

        def get_path(src, dst, loads):
            if routing_algorithm == "ugal":
                return ugal_select_path(src, dst, loads, d, a, threshold=threshold)
            elif routing_algorithm == "valiant":
                return valiant_select_path(src, dst, d, a, bias=bias)
            else:
                return dragonfly_minimal_path(src, dst, d, a)
    else:
        def get_path(src, dst, loads):
            try:
                return nx.shortest_path(graph, src, dst)
            except nx.NetworkXNoPath:
                return None

    # Distribute traffic based on matrix
    sources, dests = traffic_matrix.nonzero()
    for s, d in zip(sources, dests):
        if s == d or s not in host_mapping or d not in host_mapping:
            continue

        volume = traffic_matrix[s, d]
        src_host = host_mapping[s]
        dst_host = host_mapping[d]

        path = get_path(src_host, dst_host, link_loads)
        if path is None:
            continue

        # Add traffic to path links
        for u, v in zip(path[:-1], path[1:]):
            edge = tuple(sorted((u, v)))
            if edge in link_loads:
                link_loads[edge] += volume

    return link_loads


def create_job_from_traffic_matrix(
    traffic_matrix: np.ndarray,
    job_id: int,
    job_name: str = "trace_job",
    duration_seconds: int = 3600,
) -> Dict:
    """
    Create a RAPS-compatible job dict from a traffic matrix.

    Args:
        traffic_matrix: 2D array of communication pattern
        job_id: Unique job ID
        job_name: Job name
        duration_seconds: Expected job duration

    Returns:
        Job dictionary compatible with RAPS
    """
    analysis = analyze_traffic_pattern(traffic_matrix)

    n = traffic_matrix.shape[0]
    total_traffic = analysis['total_traffic_bytes']

    # Calculate traces (simplified - constant rate)
    trace_len = min(duration_seconds, 1000)
    bytes_per_sec = total_traffic / duration_seconds if duration_seconds > 0 else 0

    # Create network traces (TX/RX per node, averaged)
    avg_tx_per_node = bytes_per_sec / n if n > 0 else 0
    ntx_trace = [avg_tx_per_node] * trace_len
    nrx_trace = [avg_tx_per_node] * trace_len  # Symmetric assumption

    # Create placeholder CPU/GPU traces
    cpu_trace = [0.5] * trace_len  # 50% CPU utilization
    gpu_trace = [0.7] * trace_len  # 70% GPU utilization

    return job_dict(
        nodes_required=n,
        name=job_name,
        account="sc26",
        id=job_id,
        cpu_trace=cpu_trace,
        gpu_trace=gpu_trace,
        ntx_trace=ntx_trace,
        nrx_trace=nrx_trace,
        time_limit=duration_seconds,
        expected_run_time=duration_seconds,
        comm_pattern=analysis['raps_pattern'],
        message_size=int(analysis['avg_message_size']) if analysis['avg_message_size'] > 0 else MESSAGE_SIZE_64K,
    )


def load_affinity_graph(json_path: Path) -> Optional[Dict]:
    """Load affinity graph from JSON file."""
    if not json_path.exists():
        return None

    with open(json_path, 'r') as f:
        return json.load(f)


def affinity_to_traffic_matrix(affinity: Dict) -> np.ndarray:
    """
    Convert affinity graph JSON to a traffic matrix.

    Args:
        affinity: Affinity graph dict with 'nodes' and 'edges'

    Returns:
        2D numpy array (src x dst) of bytes
    """
    n = affinity['num_nodes']
    matrix = np.zeros((n, n), dtype=np.float64)

    for edge in affinity['edges']:
        src = edge['source']
        dst = edge['target']
        weight = edge['weight']

        # Affinity graph is undirected, split traffic both ways
        matrix[src, dst] = weight / 2
        matrix[dst, src] = weight / 2

    return matrix


def load_dynamic_traffic(npy_path: Path, meta_path: Optional[Path] = None) -> Tuple[np.ndarray, Dict]:
    """
    Load dynamic traffic matrix from numpy file.

    Args:
        npy_path: Path to .npy file (3D: time x src x dst)
        meta_path: Optional path to metadata JSON

    Returns:
        Tuple of (traffic_matrix_3d, metadata)
    """
    matrix = np.load(npy_path)

    metadata = {}
    if meta_path and meta_path.exists():
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

    return matrix, metadata


def get_traffic_at_time(dynamic_matrix: np.ndarray, time_idx: int) -> np.ndarray:
    """Get traffic matrix snapshot at a specific time index."""
    if time_idx < 0 or time_idx >= dynamic_matrix.shape[0]:
        return np.zeros((dynamic_matrix.shape[1], dynamic_matrix.shape[2]))
    return dynamic_matrix[time_idx]


def aggregate_traffic(dynamic_matrix: np.ndarray, start_idx: int = 0, end_idx: int = -1) -> np.ndarray:
    """Aggregate traffic over a time range."""
    if end_idx < 0:
        end_idx = dynamic_matrix.shape[0]
    return np.sum(dynamic_matrix[start_idx:end_idx], axis=0)


# ==========================================
# DUMPI Trace Processing
# ==========================================
def parse_dumpi_to_events(dumpi_dir: Path) -> List[Tuple[float, int, int, int]]:
    """
    Parse SST-DUMPI traces into a list of communication events.

    Args:
        dumpi_dir: Directory containing DUMPI .bin files

    Returns:
        List of (timestamp, src_rank, dst_rank, bytes) tuples
    """
    import subprocess

    events = []
    bin_files = list(dumpi_dir.glob("*.bin"))

    for bin_file in bin_files:
        # Extract rank from filename (e.g., trace-0000.bin -> rank 0)
        rank_str = bin_file.stem.split('-')[-1]
        try:
            src_rank = int(rank_str)
        except ValueError:
            src_rank = 0

        try:
            result = subprocess.run(
                ["dumpi2ascii", str(bin_file)],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode != 0:
                continue

            for line in result.stdout.split('\n'):
                event = _parse_dumpi_line(line, src_rank)
                if event:
                    events.append(event)

        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    # Sort by timestamp
    events.sort(key=lambda x: x[0])
    return events


def _parse_dumpi_line(line: str, default_src: int) -> Optional[Tuple[float, int, int, int]]:
    """Parse a single line from dumpi2ascii output."""
    import re

    # MPI_Send format: timestamp MPI_Send(dest=X, count=Y, ...)
    if "MPI_Send" in line or "MPI_Isend" in line:
        ts_match = re.search(r'^([\d.]+)', line)
        dest_match = re.search(r'dest=(\d+)', line)
        count_match = re.search(r'count=(\d+)', line)
        type_match = re.search(r'datatype=(\w+)', line)

        if dest_match and count_match:
            ts = float(ts_match.group(1)) if ts_match else 0.0
            dst = int(dest_match.group(1))
            count = int(count_match.group(1))

            # Estimate bytes based on datatype
            datatype = type_match.group(1) if type_match else "MPI_BYTE"
            bytes_per_elem = _mpi_datatype_size(datatype)
            total_bytes = count * bytes_per_elem

            return (ts, default_src, dst, total_bytes)

    return None


def _mpi_datatype_size(datatype: str) -> int:
    """Return size in bytes for MPI datatype."""
    sizes = {
        "MPI_BYTE": 1,
        "MPI_CHAR": 1,
        "MPI_SHORT": 2,
        "MPI_INT": 4,
        "MPI_LONG": 8,
        "MPI_FLOAT": 4,
        "MPI_DOUBLE": 8,
        "MPI_LONG_DOUBLE": 16,
    }
    return sizes.get(datatype, 8)


def events_to_traffic_matrix(events: List[Tuple[float, int, int, int]], num_ranks: int) -> np.ndarray:
    """Convert communication events to a traffic matrix."""
    matrix = np.zeros((num_ranks, num_ranks), dtype=np.float64)

    for ts, src, dst, nbytes in events:
        if 0 <= src < num_ranks and 0 <= dst < num_ranks:
            matrix[src, dst] += nbytes

    return matrix


def events_to_dynamic_matrix(
    events: List[Tuple[float, int, int, int]],
    num_ranks: int,
    time_bin_size: float = 0.1
) -> Tuple[np.ndarray, Dict]:
    """Convert communication events to a dynamic traffic matrix."""
    if not events:
        return np.zeros((1, num_ranks, num_ranks)), {"time_bins": 1}

    timestamps = [e[0] for e in events]
    t_min, t_max = min(timestamps), max(timestamps)

    if t_max <= t_min:
        t_max = t_min + 1.0

    num_bins = max(1, int(np.ceil((t_max - t_min) / time_bin_size)))
    matrix = np.zeros((num_bins, num_ranks, num_ranks), dtype=np.float64)

    for ts, src, dst, nbytes in events:
        if 0 <= src < num_ranks and 0 <= dst < num_ranks:
            bin_idx = min(int((ts - t_min) / time_bin_size), num_bins - 1)
            matrix[bin_idx, src, dst] += nbytes

    metadata = {
        "time_bins": num_bins,
        "time_bin_size": time_bin_size,
        "time_min": t_min,
        "time_max": t_max
    }

    return matrix, metadata


# ==========================================
# Template-based Traffic Matrix Scaling
# ==========================================

class TrafficMatrixTemplate:
    """
    Encapsulates a mini-app traffic matrix as a reusable template.

    The template can be "tiled" onto jobs of any size, preserving the
    communication structure while scaling to match real workload traffic volumes.

    This approach does NOT classify jobs - it simply asks:
    "What if this job's communication looked like <mini-app>?"
    """

    def __init__(self, matrix: np.ndarray, name: str, source_path: Optional[Path] = None):
        """
        Args:
            matrix: The mini-app's traffic matrix (e.g., 64x64 from LULESH)
            name: Identifier for this template (e.g., "lulesh", "hpgmg")
            source_path: Optional path to the original matrix file
        """
        self.matrix = matrix.astype(np.float64)
        self.name = name
        self.source_path = source_path
        self.n_template = matrix.shape[0]

        # Precompute normalized version (sum = 1)
        total = np.sum(self.matrix)
        if total > 0:
            self.normalized = self.matrix / total
        else:
            self.normalized = self.matrix.copy()

        # Extract statistics
        self._compute_statistics()

    def _compute_statistics(self):
        """Compute statistics about this template."""
        nonzero = self.matrix[self.matrix > 0]

        self.stats = {
            "name": self.name,
            "template_size": self.n_template,
            "total_bytes": float(np.sum(self.matrix)),
            "num_edges": int(np.count_nonzero(self.matrix)),
            "sparsity": float(1 - np.count_nonzero(self.matrix) / self.matrix.size),
            "avg_edge_weight": float(np.mean(nonzero)) if len(nonzero) > 0 else 0,
            "std_edge_weight": float(np.std(nonzero)) if len(nonzero) > 0 else 0,
        }

    def tile_to_size(self, target_nodes: int, total_traffic_bytes: float) -> np.ndarray:
        """
        Tile this template to create a traffic matrix for a larger job.

        The template is repeated like tiles on a floor:
        - Nodes 0 to n_template-1 use row/col 0 to n_template-1
        - Nodes n_template to 2*n_template-1 wrap around and reuse row/col 0 to n_template-1
        - And so on...

        Args:
            target_nodes: Number of nodes in the target job
            total_traffic_bytes: Total traffic volume (from real workload's ib_tx)

        Returns:
            A target_nodes x target_nodes traffic matrix
        """
        # Calculate how many times we need to tile
        repeats = (target_nodes + self.n_template - 1) // self.n_template

        # Tile the normalized matrix
        tiled = np.tile(self.normalized, (repeats, repeats))

        # Crop to target size
        result = tiled[:target_nodes, :target_nodes].copy()

        # Clear diagonal (no self-communication)
        np.fill_diagonal(result, 0)

        # Re-normalize after cropping (cropping may have changed the sum)
        current_sum = np.sum(result)
        if current_sum > 0:
            result = result / current_sum * total_traffic_bytes

        return result

    def __repr__(self):
        return f"TrafficMatrixTemplate(name='{self.name}', size={self.n_template}x{self.n_template})"


def load_template(matrix_path: Path, name: Optional[str] = None) -> TrafficMatrixTemplate:
    """
    Load a traffic matrix template from file.

    Args:
        matrix_path: Path to .npy or .h5 file
        name: Optional name (defaults to filename stem)

    Returns:
        TrafficMatrixTemplate instance
    """
    if name is None:
        name = matrix_path.stem

    if matrix_path.suffix == ".npy":
        matrix = np.load(matrix_path)
    elif matrix_path.suffix == ".h5":
        import h5py
        with h5py.File(matrix_path, 'r') as f:
            matrix = f["traffic_matrix"][:]
    else:
        raise ValueError(f"Unsupported format: {matrix_path.suffix}")

    # Handle 3D matrices (dynamic) by summing over time
    if matrix.ndim == 3:
        matrix = np.sum(matrix, axis=0)

    return TrafficMatrixTemplate(matrix, name, matrix_path)


def load_all_templates(matrix_dir: Path) -> Dict[str, TrafficMatrixTemplate]:
    """
    Load all traffic matrix templates from a directory.

    Args:
        matrix_dir: Directory containing .npy or .h5 files

    Returns:
        Dict mapping template name to TrafficMatrixTemplate
    """
    templates = {}

    for ext in ["*.npy", "*.h5"]:
        for path in matrix_dir.glob(ext):
            # Skip dynamic matrices (they have _dynamic in name)
            if "_dynamic" in path.stem:
                continue
            try:
                template = load_template(path)
                templates[template.name] = template
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")

    return templates


def get_job_total_traffic(job) -> Tuple[float, float]:
    """
    Extract total TX and RX traffic from a RAPS Job object.

    Args:
        job: RAPS Job object with ntx_trace and nrx_trace

    Returns:
        Tuple of (total_tx_bytes, total_rx_bytes)
    """
    trace_quanta = getattr(job, 'trace_quanta', 15)  # Default 15 seconds

    ntx_trace = getattr(job, 'ntx_trace', None) or []
    nrx_trace = getattr(job, 'nrx_trace', None) or []

    # ntx_trace values are bytes per trace_quanta interval
    total_tx = sum(ntx_trace) if ntx_trace else 0
    total_rx = sum(nrx_trace) if nrx_trace else 0

    return float(total_tx), float(total_rx)


def apply_template_to_job(template: TrafficMatrixTemplate, job) -> Dict[str, Any]:
    """
    Apply a traffic matrix template to a real workload job.

    This creates a traffic matrix that:
    - Has the communication STRUCTURE of the mini-app template
    - Has the communication VOLUME of the real job
    - Has the SIZE matching the real job's node count

    Args:
        template: The mini-app template to apply
        job: RAPS Job object from real workload

    Returns:
        Dict containing:
        - traffic_matrix: The generated matrix
        - job_id: Original job ID
        - template_name: Which template was used
        - metadata: Additional info
    """
    nodes = job.nodes_required
    total_tx, total_rx = get_job_total_traffic(job)

    # Use average of TX and RX (they should be similar for symmetric communication)
    total_traffic = (total_tx + total_rx) / 2

    # If no traffic data, use a default based on job size and duration
    if total_traffic == 0:
        # Estimate: 1 GB per node per hour
        duration = getattr(job, 'expected_run_time', 3600)
        total_traffic = nodes * 1e9 * (duration / 3600)

    # Generate the traffic matrix
    traffic_matrix = template.tile_to_size(nodes, total_traffic)

    return {
        "traffic_matrix": traffic_matrix,
        "job_id": getattr(job, 'id', None),
        "job_name": getattr(job, 'name', 'unknown'),
        "template_name": template.name,
        "num_nodes": nodes,
        "total_traffic_bytes": total_traffic,
        "original_tx": total_tx,
        "original_rx": total_rx,
    }


def apply_all_templates_to_job(templates: Dict[str, TrafficMatrixTemplate],
                                job) -> Dict[str, Dict[str, Any]]:
    """
    Apply all available templates to a single job.

    This enables what-if analysis: "How would network behave if this job
    communicated like LULESH? Like HPGMG? Like CoSP2?"

    Args:
        templates: Dict of template_name -> TrafficMatrixTemplate
        job: RAPS Job object

    Returns:
        Dict mapping template_name -> result from apply_template_to_job
    """
    results = {}
    for name, template in templates.items():
        results[name] = apply_template_to_job(template, job)
    return results


def batch_apply_template(template: TrafficMatrixTemplate,
                         jobs: List,
                         progress: bool = True) -> List[Dict[str, Any]]:
    """
    Apply a single template to multiple jobs.

    Args:
        template: The template to apply
        jobs: List of RAPS Job objects
        progress: Whether to show progress bar

    Returns:
        List of results from apply_template_to_job
    """
    results = []

    iterator = jobs
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(jobs, desc=f"Applying {template.name}")
        except ImportError:
            pass

    for job in iterator:
        result = apply_template_to_job(template, job)
        results.append(result)

    return results


# ==========================================
# Validation and Testing
# ==========================================
def validate_traffic_matrix(matrix: np.ndarray) -> Dict[str, Any]:
    """Validate a traffic matrix and return diagnostics."""
    issues = []

    # Check shape
    if matrix.ndim != 2:
        issues.append(f"Expected 2D matrix, got {matrix.ndim}D")
    elif matrix.shape[0] != matrix.shape[1]:
        issues.append(f"Matrix not square: {matrix.shape}")

    # Check for negative values
    if np.any(matrix < 0):
        issues.append("Contains negative values")

    # Check for NaN/Inf
    if np.any(np.isnan(matrix)):
        issues.append("Contains NaN values")
    if np.any(np.isinf(matrix)):
        issues.append("Contains Inf values")

    # Check diagonal (should be zero for P2P communication)
    diag_sum = np.sum(np.diag(matrix))
    if diag_sum > 0:
        issues.append(f"Non-zero diagonal (self-communication): {diag_sum}")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "shape": matrix.shape,
        "total_bytes": float(np.sum(matrix)),
        "sparsity": float(1 - np.count_nonzero(matrix) / matrix.size) if matrix.size > 0 else 1.0
    }


if __name__ == "__main__":
    # Test with a sample traffic matrix
    print("Testing traffic integration module...")
    print("=" * 60)

    # Create test matrices
    n = 64

    # Test ALL_TO_ALL pattern
    all_to_all = np.ones((n, n)) * 1000
    np.fill_diagonal(all_to_all, 0)
    print("\nALL_TO_ALL pattern:")
    print(analyze_traffic_pattern(all_to_all))

    # Test STENCIL_3D pattern (6 neighbors)
    stencil = np.zeros((n, n))
    dims = factorize_3d(n)
    for i in range(n):
        neighbors = get_stencil_3d_neighbors(i, dims, n) if RAPS_AVAILABLE else []
        for j in neighbors:
            if j < n:
                stencil[i, j] = 1000
    print("\nSTENCIL_3D pattern:")
    print(analyze_traffic_pattern(stencil))

    # Test sparse random
    sparse = np.zeros((n, n))
    for _ in range(n * 2):
        i, j = np.random.randint(0, n, 2)
        if i != j:
            sparse[i, j] = np.random.randint(100, 10000)
    print("\nSPARSE_RANDOM pattern:")
    print(analyze_traffic_pattern(sparse))

    print("\nValidation test:")
    print(validate_traffic_matrix(all_to_all))

    # ==========================================
    # Test Template-based Scaling (NEW)
    # ==========================================
    print("\n" + "=" * 60)
    print("Testing Template-based Traffic Matrix Scaling")
    print("=" * 60)

    # Create a template from the stencil pattern (simulating a mini-app)
    print("\n1. Creating template from 64-node stencil pattern...")
    template = TrafficMatrixTemplate(stencil, name="test_stencil")
    print(f"   Template: {template}")
    print(f"   Stats: {template.stats}")

    # Tile to different sizes
    print("\n2. Tiling template to different job sizes...")

    test_cases = [
        (64, 1e9),    # Same size, 1 GB traffic
        (128, 2e9),   # 2x size, 2 GB traffic
        (256, 10e9),  # 4x size, 10 GB traffic
        (100, 5e9),   # Non-power-of-2, 5 GB traffic
    ]

    for target_nodes, total_traffic in test_cases:
        matrix = template.tile_to_size(target_nodes, total_traffic)
        print(f"\n   Target: {target_nodes} nodes, {total_traffic/1e9:.1f} GB")
        print(f"   Result shape: {matrix.shape}")
        print(f"   Result sum: {np.sum(matrix)/1e9:.2f} GB")
        print(f"   Non-zero edges: {np.count_nonzero(matrix)}")
        print(f"   Sparsity: {1 - np.count_nonzero(matrix)/matrix.size:.2%}")

    # Test with mock job
    print("\n3. Testing apply_template_to_job with mock job...")

    class MockJob:
        def __init__(self, job_id, nodes, ntx_trace, nrx_trace):
            self.id = job_id
            self.name = f"mock_job_{job_id}"
            self.nodes_required = nodes
            self.ntx_trace = ntx_trace
            self.nrx_trace = nrx_trace
            self.trace_quanta = 15
            self.expected_run_time = 3600

    mock_job = MockJob(
        job_id=12345,
        nodes=256,
        ntx_trace=[1e8] * 100,   # 100 intervals * 100 MB = 10 GB TX
        nrx_trace=[1e8] * 100,   # 100 intervals * 100 MB = 10 GB RX
    )

    result = apply_template_to_job(template, mock_job)
    print(f"   Job ID: {result['job_id']}")
    print(f"   Template: {result['template_name']}")
    print(f"   Nodes: {result['num_nodes']}")
    print(f"   Total traffic: {result['total_traffic_bytes']/1e9:.2f} GB")
    print(f"   Matrix shape: {result['traffic_matrix'].shape}")
    print(f"   Matrix sum: {np.sum(result['traffic_matrix'])/1e9:.2f} GB")

    # Test loading templates from directory
    print("\n4. Testing load_all_templates...")
    matrix_dir = Path("/app/data/matrices")
    if matrix_dir.exists():
        templates = load_all_templates(matrix_dir)
        print(f"   Found {len(templates)} templates: {list(templates.keys())}")

        if templates:
            # Apply all templates to mock job
            print("\n5. Applying all templates to mock job (what-if analysis)...")
            all_results = apply_all_templates_to_job(templates, mock_job)
            for name, res in all_results.items():
                mat = res['traffic_matrix']
                sparsity = 1 - np.count_nonzero(mat) / mat.size
                print(f"   {name}: {mat.shape}, sparsity={sparsity:.2%}")
    else:
        print(f"   Directory {matrix_dir} not found, skipping...")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

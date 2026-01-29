#!/usr/bin/env python3
"""
SC26 Trace Parser (v2.0)
========================
Parses MPI traces and generates:
1. Static Affinity Graph (JSON) - Communication topology
2. Dynamic Traffic Matrices (numpy .npy) - Time-series traffic data
3. Aggregated Traffic Matrix (HDF5) - For backwards compatibility

Supports:
- SST-DUMPI traces (.bin files)
- Custom libtracer logs (stderr.log with [TRACE] format)
"""

import os
import re
import json
import numpy as np
import h5py
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# Configuration
# ==========================================
INPUT_ROOT = Path("/app/data/raw_traces")
OUTPUT_DIR = Path("/app/data/matrices")

# Time bin size for dynamic matrices (seconds)
TIME_BIN_SIZE = 0.1


def parse_tracer_log(folder_path):
    """
    Parse custom libtracer stderr.log format.
    Returns list of (timestamp, src, dst, bytes) tuples.
    """
    log_file = folder_path / "stderr.log"
    if not log_file.exists():
        return None

    # New format: timestamp,src,dst,bytes (from logger.c v4.0)
    csv_pattern = re.compile(r"^([\d.]+),(\d+),(\d+),(\d+)$")
    # Old format: [TRACE] R2 -> R6 | Isend | 968 bytes
    trace_pattern = re.compile(r"\[TRACE\] R(\d+) -> R(\d+) \| .* \| (\d+) bytes")

    events = []
    max_rank = 0

    # First try to find CSV files from logger.c
    csv_files = list(folder_path.glob("traffic_rank_*.csv"))
    if csv_files:
        for csv_file in csv_files:
            with open(csv_file, "r") as f:
                for line in f:
                    match = csv_pattern.match(line.strip())
                    if match:
                        ts = float(match.group(1))
                        src = int(match.group(2))
                        dst = int(match.group(3))
                        nbytes = int(match.group(4))
                        events.append((ts, src, dst, nbytes))
                        max_rank = max(max_rank, src, dst)
        if events:
            return events, max_rank + 1

    # Fallback to stderr.log parsing
    with open(log_file, "r") as f:
        current_time = 0.0
        for line in f:
            # Try CSV format first
            csv_match = csv_pattern.match(line.strip())
            if csv_match:
                ts = float(csv_match.group(1))
                src = int(csv_match.group(2))
                dst = int(csv_match.group(3))
                nbytes = int(csv_match.group(4))
                events.append((ts, src, dst, nbytes))
                max_rank = max(max_rank, src, dst)
                continue

            # Try [TRACE] format
            if "[TRACE]" in line:
                trace_match = trace_pattern.search(line)
                if trace_match:
                    src = int(trace_match.group(1))
                    dst = int(trace_match.group(2))
                    nbytes = int(trace_match.group(3))
                    # Assign incrementing timestamp for old format
                    events.append((current_time, src, dst, nbytes))
                    current_time += 0.001
                    max_rank = max(max_rank, src, dst)

    if not events:
        return None

    return events, max_rank + 1


def parse_dumpi_traces(folder_path):
    """
    Parse SST-DUMPI binary traces.
    Returns list of (timestamp, src, dst, bytes) tuples.
    """
    dumpi_dir = folder_path / "dumpi"
    if not dumpi_dir.exists():
        return None

    # Look for .bin files
    bin_files = list(dumpi_dir.glob("*.bin"))
    if not bin_files:
        return None

    events = []
    max_rank = 0

    # Try using dumpi2ascii if available
    import subprocess
    try:
        for bin_file in bin_files:
            result = subprocess.run(
                ["dumpi2ascii", str(bin_file)],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                # Parse dumpi2ascii output
                # Format varies, but typically includes MPI calls with timestamps
                for line in result.stdout.split("\n"):
                    # Look for MPI_Send/MPI_Isend calls
                    if "MPI_Send" in line or "MPI_Isend" in line:
                        # Extract timestamp, destination, and count
                        # This is a simplified parser - adjust based on actual output
                        parts = line.split()
                        try:
                            ts = float(parts[0]) if parts[0].replace(".", "").isdigit() else 0
                            # Parse dest and count from the line
                            dest_match = re.search(r"dest=(\d+)", line)
                            count_match = re.search(r"count=(\d+)", line)
                            rank_match = re.search(r"rank (\d+)", line)

                            if dest_match and count_match:
                                src = int(rank_match.group(1)) if rank_match else 0
                                dst = int(dest_match.group(1))
                                nbytes = int(count_match.group(1)) * 8  # Assume 8 bytes per element
                                events.append((ts, src, dst, nbytes))
                                max_rank = max(max_rank, src, dst)
                        except (ValueError, IndexError):
                            continue
    except FileNotFoundError:
        print("    dumpi2ascii not found, trying binary parse...")
        # Fallback: read binary format directly (simplified)
        return None
    except subprocess.TimeoutExpired:
        print("    dumpi2ascii timeout")
        return None

    if not events:
        return None

    return events, max_rank + 1


def build_affinity_graph(events, num_ranks):
    """
    Build static affinity graph from communication events.
    Returns a JSON-serializable dictionary.
    """
    # Count communications between each pair
    edge_weights = defaultdict(int)
    edge_counts = defaultdict(int)

    for ts, src, dst, nbytes in events:
        if src != dst:
            edge_key = (min(src, dst), max(src, dst))  # Undirected
            edge_weights[edge_key] += nbytes
            edge_counts[edge_key] += 1

    # Build graph structure
    nodes = [{"id": i, "rank": i} for i in range(num_ranks)]

    edges = []
    for (src, dst), weight in edge_weights.items():
        edges.append({
            "source": src,
            "target": dst,
            "weight": weight,
            "count": edge_counts[(src, dst)]
        })

    # Compute node-level statistics
    node_send_bytes = defaultdict(int)
    node_recv_bytes = defaultdict(int)
    node_degree = defaultdict(set)

    for ts, src, dst, nbytes in events:
        if src != dst:
            node_send_bytes[src] += nbytes
            node_recv_bytes[dst] += nbytes
            node_degree[src].add(dst)
            node_degree[dst].add(src)

    for node in nodes:
        nid = node["id"]
        node["send_bytes"] = node_send_bytes[nid]
        node["recv_bytes"] = node_recv_bytes[nid]
        node["degree"] = len(node_degree[nid])

    affinity_graph = {
        "num_nodes": num_ranks,
        "num_edges": len(edges),
        "total_bytes": sum(edge_weights.values()),
        "nodes": nodes,
        "edges": edges
    }

    return affinity_graph


def build_dynamic_traffic_matrix(events, num_ranks, time_bin_size=TIME_BIN_SIZE):
    """
    Build dynamic traffic matrix as a 3D numpy array.
    Shape: (num_time_bins, num_ranks, num_ranks)
    """
    if not events:
        return None

    # Find time range
    timestamps = [e[0] for e in events]
    t_min = min(timestamps)
    t_max = max(timestamps)

    if t_max <= t_min:
        t_max = t_min + 1.0

    # Calculate number of time bins
    num_bins = max(1, int(np.ceil((t_max - t_min) / time_bin_size)))

    # Limit to reasonable size
    if num_bins > 10000:
        time_bin_size = (t_max - t_min) / 10000
        num_bins = 10000
        print(f"    Adjusted time_bin_size to {time_bin_size:.4f}s ({num_bins} bins)")

    # Initialize 3D matrix
    traffic_matrix = np.zeros((num_bins, num_ranks, num_ranks), dtype=np.float64)

    # Fill matrix
    for ts, src, dst, nbytes in events:
        if src < num_ranks and dst < num_ranks:
            bin_idx = min(int((ts - t_min) / time_bin_size), num_bins - 1)
            traffic_matrix[bin_idx, src, dst] += nbytes

    return traffic_matrix, t_min, t_max, time_bin_size


def build_static_traffic_matrix(events, num_ranks):
    """
    Build static (aggregated) traffic matrix.
    Shape: (num_ranks, num_ranks)
    """
    matrix = np.zeros((num_ranks, num_ranks), dtype=np.float64)

    for ts, src, dst, nbytes in events:
        if src < num_ranks and dst < num_ranks:
            matrix[src, dst] += nbytes

    return matrix


def save_heatmap(matrix, name, output_dir):
    """Save a heatmap visualization of the traffic matrix."""
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap="viridis", interpolation="nearest", aspect='auto')
    plt.colorbar(label="Bytes Transferred")
    plt.title(f"Traffic Matrix: {name}")
    plt.xlabel("Destination Rank")
    plt.ylabel("Source Rank")
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}.png", dpi=150)
    plt.close()


def process_trace_folder(folder_path, output_dir):
    """Process a single trace folder and generate all outputs."""
    experiment_name = folder_path.name

    # Try to load metadata
    metadata_file = folder_path / "metadata.json"
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

    # Parse traces
    result = None

    # Try DUMPI first
    if (folder_path / "dumpi").exists():
        result = parse_dumpi_traces(folder_path)
        if result:
            print(f"  Parsed DUMPI traces")

    # Fallback to custom tracer
    if result is None:
        result = parse_tracer_log(folder_path)
        if result:
            print(f"  Parsed tracer logs")

    if result is None:
        print(f"  No traces found in {experiment_name}")
        return None

    events, num_ranks = result
    print(f"  Found {len(events)} events, {num_ranks} ranks")

    # 1. Build and save Affinity Graph (JSON)
    affinity_graph = build_affinity_graph(events, num_ranks)
    affinity_graph["metadata"] = metadata

    json_path = output_dir / f"{experiment_name}_affinity.json"
    with open(json_path, "w") as f:
        json.dump(affinity_graph, f, indent=2)
    print(f"  Saved affinity graph: {json_path}")

    # 2. Build and save Dynamic Traffic Matrix (numpy)
    dynamic_result = build_dynamic_traffic_matrix(events, num_ranks)
    if dynamic_result:
        dynamic_matrix, t_min, t_max, actual_bin_size = dynamic_result

        npy_path = output_dir / f"{experiment_name}_dynamic.npy"
        np.save(npy_path, dynamic_matrix)

        # Save metadata for dynamic matrix
        dynamic_meta = {
            "shape": list(dynamic_matrix.shape),
            "num_time_bins": dynamic_matrix.shape[0],
            "num_ranks": num_ranks,
            "time_min": t_min,
            "time_max": t_max,
            "time_bin_size": actual_bin_size,
            "dtype": str(dynamic_matrix.dtype)
        }
        meta_path = output_dir / f"{experiment_name}_dynamic_meta.json"
        with open(meta_path, "w") as f:
            json.dump(dynamic_meta, f, indent=2)

        print(f"  Saved dynamic matrix: {npy_path} (shape: {dynamic_matrix.shape})")

    # 3. Build and save Static Traffic Matrix (HDF5 + numpy)
    static_matrix = build_static_traffic_matrix(events, num_ranks)

    # Save as numpy
    static_npy_path = output_dir / f"{experiment_name}_static.npy"
    np.save(static_npy_path, static_matrix)

    # Save as HDF5 for backwards compatibility
    h5_path = output_dir / f"{experiment_name}.h5"
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset("traffic_matrix", data=static_matrix)
        f.attrs["app_name"] = experiment_name
        f.attrs["num_ranks"] = num_ranks
        f.attrs["total_events"] = len(events)
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float)):
                    f.attrs[key] = value

    print(f"  Saved static matrix: {h5_path} (shape: {static_matrix.shape})")

    # 4. Save heatmap visualization
    save_heatmap(static_matrix, experiment_name, output_dir)

    return {
        "name": experiment_name,
        "num_ranks": num_ranks,
        "num_events": len(events),
        "affinity_json": str(json_path),
        "dynamic_npy": str(npy_path) if dynamic_result else None,
        "static_h5": str(h5_path)
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all trace folders
    run_folders = [f for f in INPUT_ROOT.iterdir() if f.is_dir()]

    if not run_folders:
        print(f"No trace folders found in {INPUT_ROOT}")
        return

    print(f"Found {len(run_folders)} trace folders. Processing...")

    results = []
    for folder in tqdm(run_folders, desc="Parsing traces"):
        print(f"\nProcessing {folder.name}...")
        result = process_trace_folder(folder, OUTPUT_DIR)
        if result:
            results.append(result)

    # Save summary
    summary_path = OUTPUT_DIR / "parse_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "num_processed": len(results),
            "traces": results
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Parsing complete!")
    print(f"  Processed: {len(results)}/{len(run_folders)} traces")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Summary: {summary_path}")
    print(f"\nGenerated files per trace:")
    print(f"  - *_affinity.json  : Static affinity graph")
    print(f"  - *_dynamic.npy    : Dynamic traffic matrix (time-series)")
    print(f"  - *_static.npy     : Aggregated traffic matrix")
    print(f"  - *.h5             : HDF5 format (backwards compatible)")
    print(f"  - *.png            : Heatmap visualization")


if __name__ == "__main__":
    main()

# Communication Patterns and Message Size Implementation Report

## Overview

This report documents the implementation of configurable communication patterns and message sizes in the RAPS network model, enabling more realistic simulation of HPC application network behavior.

## What Was Implemented

### 1. Communication Patterns

Two communication patterns are now supported:

| Pattern | Description | Traffic Distribution |
|---------|-------------|---------------------|
| `ALL_TO_ALL` | Every node sends to every other node | `tx / (N-1)` per peer |
| `STENCIL_3D` | Each node sends to 6 neighbors (±x, ±y, ±z) | `tx / 6` per neighbor |

**Implementation Details:**

- **`CommunicationPattern` enum** (`raps/job.py`): Defines pattern types
- **`factorize_3d(n)`**: Maps N nodes to a virtual 3D grid (e.g., 8→2×2×2, 64→4×4×4)
- **`get_stencil_3d_neighbors()`**: Computes 6 neighbors with periodic boundary conditions
- **`link_loads_for_job_stencil_3d()`**: Routes traffic only to stencil neighbors
- **`link_loads_for_pattern()`**: Dispatcher that selects the correct routing function

### 2. Message Size

Jobs can now specify a message size, which affects network overhead:

| Constant | Value | Use Case |
|----------|-------|----------|
| `MESSAGE_SIZE_64K` | 64 KiB | Small messages, higher overhead |
| `MESSAGE_SIZE_1M` | 1 MiB | Large messages, lower overhead |
| `None` | N/A | Raw bandwidth model (no overhead) |

**Overhead Model:**
```
effective_traffic = raw_traffic + (num_messages × header_overhead)
num_messages = ceil(bytes_per_peer / message_size) × num_peers
header_overhead = 64 bytes per message
```

### 3. Files Modified

- `raps/job.py`: Added `CommunicationPattern` enum, `message_size` and `comm_pattern` to Job
- `raps/network/base.py`: Added pattern routing functions and message overhead calculation
- `raps/network/__init__.py`: Updated `NetworkModel.simulate_network_utilization()` to use patterns

## Why These Choices

### Communication Patterns

**All-to-all** represents collective operations like `MPI_Alltoall`, common in FFT, matrix transpose, and some machine learning workloads. It creates high network load as traffic scales O(N²).

**3D Stencil** represents nearest-neighbor communication patterns used in:
- Computational fluid dynamics (CFD)
- Weather/climate modeling
- Finite difference methods
- Many physics simulations

Stencil patterns are O(N) in traffic and exhibit strong locality, making them ideal for torus topologies.

### Message Size

Message size affects real network performance through:
1. **Protocol overhead**: Each message incurs fixed header costs
2. **Latency hiding**: Smaller messages have worse bandwidth utilization
3. **Congestion dynamics**: More messages = more contention for resources

The 64 KiB and 1 MiB sizes represent common HPC message sizes—64K is typical for latency-sensitive applications, while 1M is common for bulk data transfers.

## Test Results

### Fat-Tree Topology (k=8, 128 nodes)

| Nodes | Pattern | Congestion | Change vs All-to-All |
|-------|---------|------------|---------------------|
| 8 | all-to-all | 14.63 | — |
| 8 | stencil-3d | 17.07 | +17% (worse) |
| 27 | all-to-all | 43.32 | — |
| 27 | stencil-3d | 51.20 | +18% (worse) |
| 64 | all-to-all | 78.02 | — |
| 64 | stencil-3d | 68.27 | **-12% (better)** |

**Analysis**: On fat-tree, stencil shows *worse* congestion for small jobs because it concentrates traffic on fewer links (creating hotspots). At 64 nodes, stencil becomes beneficial as the traffic spreads more evenly.

### 3D Torus Topology (4×4×4)

| Nodes | Pattern | Congestion | Change vs All-to-All |
|-------|---------|------------|---------------------|
| 8 | all-to-all | 44.80 | — |
| 8 | stencil-3d | 12.80 | **-71% (better)** |
| 27 | all-to-all | 166.40 | — |
| 27 | stencil-3d | 26.67 | **-84% (better)** |
| 64 | all-to-all | 403.20 | — |
| 64 | stencil-3d | 12.80 | **-97% (better)** |

**Analysis**: On torus topology, stencil shows *dramatic* congestion reduction because:
1. Torus is optimized for nearest-neighbor communication
2. Stencil traffic stays local (1-hop to neighbors)
3. All-to-all must traverse many hops, creating bottlenecks

### Message Size Impact

| Message Size | Overhead |
|--------------|----------|
| None (raw) | 0% |
| 64 KiB | ~0.1% |
| 1 MiB | ~0.006% |

The current header overhead model (64 bytes/message) produces minimal impact. This is realistic for large transfers but may underestimate overhead for latency-bound small-message workloads.

## Usage Example

```python
from raps.job import job_dict, CommunicationPattern, MESSAGE_SIZE_64K, MESSAGE_SIZE_1M

# CFD simulation with stencil pattern and 1 MiB messages
cfd_job = job_dict(
    nodes_required=64,
    name='cfd_simulation',
    account='physics',
    id=1,
    scheduled_nodes=list(range(64)),
    cpu_trace=[0.8],
    gpu_trace=[0.9],
    ntx_trace=[5e9],  # 5 GB/s per node
    nrx_trace=[5e9],
    comm_pattern=CommunicationPattern.STENCIL_3D,
    message_size=MESSAGE_SIZE_1M,
)

# ML training with all-to-all pattern and 64 KiB messages
ml_job = job_dict(
    nodes_required=32,
    name='ml_training',
    account='ai',
    id=2,
    scheduled_nodes=list(range(32, 64)),
    cpu_trace=[0.5],
    gpu_trace=[0.95],
    ntx_trace=[10e9],  # 10 GB/s per node
    nrx_trace=[10e9],
    comm_pattern=CommunicationPattern.ALL_TO_ALL,
    message_size=MESSAGE_SIZE_64K,
)
```

## Key Findings

1. **Topology-pattern matching matters**: Stencil patterns on torus show 70-97% congestion reduction vs all-to-all, while fat-tree shows mixed results.

2. **Job placement affects pattern efficiency**: Stencil benefits require nodes to be allocated in a way that preserves locality.

3. **Message size overhead is minimal** with the current model (~0.1% for 64K messages). Consider increasing header overhead or adding latency-based penalties for more impact.

4. **Pattern choice significantly affects congestion**: Can be more impactful than message size for determining network performance.

## Future Enhancements

1. **Additional patterns**: Ring, tree, butterfly, 2D stencil, custom patterns
2. **Latency modeling**: Small messages should incur latency penalties beyond just overhead
3. **Topology-aware stencil**: Use actual torus coordinates when available instead of virtual grid
4. **Adaptive message sizing**: Allow message size to vary with communication phase

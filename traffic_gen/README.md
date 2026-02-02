# Frontier-Sim: HPC Network Traffic Profiling Framework

**Frontier-Sim** is a profiling and simulation toolchain for analyzing MPI communication patterns in HPC applications. It captures low-level MPI traces using SST-DUMPI and custom interceptors, then generates static affinity graphs and dynamic traffic tensors for network simulation.

## Key Features

* **Multi-Layer Profiling**: SST-DUMPI traces + custom loggers for time-series traffic data
* **Transparent Analysis**: Stateful DUMPI parser with MPI coverage reporting
* **Dual-Mode Output**:
    * Static Profile (JSON): Weighted affinity graph + pattern classification
    * Dynamic Tensor (NPY): Time-series traffic matrix $(T \times N \times N)$
* **Traffic Replay Engine**: High-fidelity replay of captured patterns
* **Virtual Time Dilation**: Simulate GPU-accelerated compute while preserving communication
* **Supported MPI Ops**: P2P (`Send`, `Isend`, `Irecv`), Collectives (`Allreduce`, `Barrier`)
* **Multi-Benchmark Support**: LULESH, HPGMG, CoMD, CoHMM, CoSP2

## Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y build-essential openmpi-bin libopenmpi-dev \
    cmake automake autoconf libtool git python3 python3-pip

# Python dependencies
pip3 install numpy pandas matplotlib seaborn
```

## Installation

### 1. Build and Install SST-DUMPI

```bash
cd /tmp
git clone https://github.com/sstsimulator/sst-dumpi.git
cd sst-dumpi
./bootstrap.sh
./configure --prefix=/usr/local/sst-dumpi --enable-libdumpi CC=mpicc CXX=mpicxx
make -j$(nproc)
sudo make install

# Add to ~/.bashrc for persistence
export LD_LIBRARY_PATH=/usr/local/sst-dumpi/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/sst-dumpi/bin:$PATH
```

### 2. Build Frontier-Sim

```bash
git clone <repository-url> frontier-sim
cd frontier-sim/src
./build_all.sh
```

Builds all components: profiling libraries, replay tool, and HPC benchmarks (10-20 min).

## Quick Start

### Run Single Experiment

```bash
cd frontier-sim/src
./run_sim.sh  # Runs LULESH with 64 ranks, saves to ../data/lulesh/
```

### Run Batch Experiments

```bash
python3 run_batch.py  # Executes baseline, GPU sim, and replay experiments
```

## Usage

### Capture Traffic from Your Application

```bash
export LD_LIBRARY_PATH=/usr/local/sst-dumpi/lib:$LD_LIBRARY_PATH
export DUMPI_LIB=/usr/local/sst-dumpi/lib/libdumpi.so
export LOGGER_LIB=/path/to/frontier-sim/src/liblogger.so
export MPI_TRACE_LIB=/path/to/frontier-sim/src/libmpitrace.so
export LD_PRELOAD=$LOGGER_LIB:$MPI_TRACE_LIB:$DUMPI_LIB

# Standard run
mpirun -n 8 --oversubscribe --allow-run-as-root ./your_app [args]

# With 10x time dilation (simulate faster compute)
export LOGGER_TIME_SCALE=10.0
mpirun -n 8 --oversubscribe --allow-run-as-root ./your_app [args]

# Docker: avoid shared memory issues
mpirun --mca btl self,tcp -n 8 --oversubscribe --allow-run-as-root ./your_app
```

### Analyze Captured Data

```bash
# Generate JSON profile + NPY tensor from DUMPI traces
python3 analyze_research.py --data_dir ../data/lulesh_baseline \
                             --out_dir raps_output \
                             --ranks 8 \
                             --bin_size 0.01

# Visualize patterns
python3 analyze.py              # Basic heatmaps → ../plots/
python3 analyze_advanced.py     # Advanced metrics → ../plots_advanced/
python3 compare_apps.py         # Multi-app comparison
```

### Replay Traffic

```bash
mpirun -n 8 --oversubscribe --allow-run-as-root \
    ./replay_tool ../data/lulesh_baseline/traffic_timeseries.csv
```

## Output Files

```
data/<experiment_name>/
├── dumpi-*.bin                    # Binary DUMPI traces
├── dumpi-*.meta                   # Metadata
├── traffic_timeseries.csv         # Time-series traffic log
└── runtime.log                    # Application output

src/raps_output/
├── static_profile.json            # Affinity graph + pattern type
└── dynamic_traffic.npy            # 3D tensor (time × ranks × ranks)
```

### static_profile.json Format

```json
{
  "job_profile": {
    "pattern_type": "STENCIL_NEAREST_NEIGHBOR",
    "sparsity": 0.875,
    "diagonal_dominance": 0.923
  },
  "affinity_graph": [
    {"u": 0, "v": 1, "weight": 0.9543},
    ...
  ]
}
```

## Advanced Features

### Virtual Time Dilation

Compress compute time while preserving communication patterns:

```bash
export LOGGER_TIME_SCALE=10.0  # 10x faster compute
mpirun -n 8 ./your_app
```

### MPI Coverage Reporting

Analysis explicitly shows which MPI calls are supported vs. ignored:

```
========================================
MPI COVERAGE REPORT
========================================
Supported: ['MPI_Send', 'MPI_Isend', 'MPI_Allreduce']
Ignored:
  MPI_Bcast  : 145 occurrences
  MPI_Reduce : 89 occurrences
========================================
```

### Pattern Classification

Auto-detects traffic patterns:
- **STENCIL_NEAREST_NEIGHBOR**: High sparsity + diagonal dominance
- **ALL_TO_ALL_DENSE**: Low sparsity
- **SPARSE_RANDOM**: High sparsity, no structure

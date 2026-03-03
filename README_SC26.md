# RAPS SC26 — Reproducibility Guide

This repository contains all experiment code, SLURM submission scripts, and plotting scripts
for the SC26 paper on **ExaDigiT/RAPS** — a discrete-event simulator for HPC system scheduling,
network congestion, and power estimation.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Repository Structure](#2-repository-structure)
3. [Running Use Case Experiments (UC1–UC4)](#3-running-use-case-experiments-uc1uc4)
4. [Running Scaling Benchmarks](#4-running-scaling-benchmarks)
5. [Dragonfly Congestion Visualization](#5-dragonfly-congestion-visualization)
6. [Generating All Figures](#6-generating-all-figures)
7. [Output Layout](#7-output-layout)

---

## 1. Environment Setup

All experiments run on **OLCF Frontier** (or any system with Python 3.11+ and the required
packages). Each SLURM job requests 1 node, 2-hour wall time (`-p batch`, account `GEN053`).

### 1.1 Clone and enter the repository

```bash
git clone https://github.com/ZzZTripleZzZ/RAPS_SC26.git
cd RAPS_SC26
```

### 1.2 Load modules (Frontier)

```bash
module load PrgEnv-gnu cray-python
```

### 1.3 Create the virtual environment and install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

> **Note:** The virtual environment must be at `.venv/` relative to the repo root because all
> SLURM scripts activate it with `source .venv/bin/activate`.

### 1.4 Verify installation

```bash
python -c "import raps; print('RAPS OK')"
raps run -h
```

---

## 2. Repository Structure

```
RAPS_SC26/
├── config/                     # System YAML configs (frontier, lassen, …)
├── raps/                       # Core simulator library
│   ├── engine.py               #   Main DES loop + M/D/1 slowdown model
│   ├── job.py                  #   Job class, dilation, communication patterns
│   ├── network/
│   │   ├── __init__.py         #   NetworkModel, dump_link_loads
│   │   ├── base.py             #   Link loads, simulate_inter_job_congestion
│   │   ├── dragonfly.py        #   UGAL/Valiant routing for dragonfly
│   │   └── fat_tree.py         #   ECMP/adaptive routing for fat-tree
│   ├── policy.py               #   FCFS, SJF, backfill, allocation strategies
│   └── system_config.py        #   Parses YAML into typed dataclasses
├── src/
│   ├── run_use_cases.py        # UC1–UC4 experiment runner
│   ├── run_frontier.py         # Scaling benchmark runner
│   ├── plot_energy_impact.py   # UC figure generator
│   ├── plot_benchmark_comparison.py  # Benchmark figure generator
│   ├── plot_energy_overhead.py       # Energy overhead summary figure
│   ├── plot_dt_tradeoff.py     # Δt tradeoff figures
│   └── plot_motivation.py      # Motivation figures
├── scripts/
│   ├── plot_dragonfly_congestion.py  # Chord diagram renderer (from ndt branch)
│   └── run_frontier_congestion_snapshot.py  # Snapshot runner → chord diagram
├── benchmarks/
│   ├── gpcnet/                 # GPCNeT source (network_test, network_load_test)
│   └── sst-macro/              # SST-Macro dragonfly config
├── tests/
│   └── unit/test_stall_ratio.py
│
│   # SLURM submission scripts (submit once, auto-resubmit on timeout)
├── submit_usecases_parallel.slurm      # Standard UCs (200 jobs, dt=30s)
├── submit_energy_impact.slurm          # Heavy UCs  (300 jobs, dt=1s)
├── submit_benchmark_parallel.slurm     # Scaling benchmarks
├── submit_gpcnet.slurm                 # GPCNeT real-system measurement
├── submit_gpcnet_dumpi.slurm           # Generate DUMPI traces for SST-Macro
├── submit_sstmacro.slurm               # SST-Macro simulation
└── output/                     # All results (git-ignored)
    ├── use_cases/              #   UC CSV results
    ├── frontier_scaling/       #   Benchmark CSV + per-experiment dirs
    └── figures/main/           #   Final paper figures (PNG)
```

---

## 3. Running Use Case Experiments (UC1–UC4)

Four use cases study different aspects of HPC system management under network congestion:

| UC | Topic | Variants |
|----|-------|---------|
| UC1 | Adaptive routing & congestion mitigation | minimal / ugal / valiant (dragonfly); minimal / ecmp / adaptive (fat-tree) |
| UC2 | Scheduler policy optimization | FCFS / FCFS+FirstFit / SJF |
| UC3 | Topology-aware node placement | random / contiguous / hybrid |
| UC4 | Energy cost of congestion | 4 routing configs × energy breakdown |

### 3.1 Standard run (200 jobs, Δt = 30 s, 1000 nodes)

```bash
sbatch submit_usecases_parallel.slurm
```

- Runs all 4 UCs for both `frontier` (dragonfly) and `lassen` (fat-tree) **in parallel**.
- Phase 1: Frontier (faster); Phase 2: Lassen.
- Results: `output/use_cases/{system}_n1000/uc{N}_*.csv`
- Auto-resubmits up to 5× if the 2-hour wall time is exceeded (incomplete UCs are skipped on resume).

### 3.2 Heavy-load run (300 jobs, Δt = 1 s, 1000 nodes)

```bash
sbatch submit_energy_impact.slurm
```

- Same structure but with 50% more jobs to create real queueing pressure.
- Results: `output/use_cases/{system}_n1000_heavy/uc{N}_*.csv`
- Auto-resubmits up to 10×.

> **Note on timing:** Lassen with adaptive routing and Δt = 1 s is prohibitively slow (~880 min/config).
> The heavy script uses Δt = 10 s for Lassen and Δt = 1–5 s for Frontier.
> UC4 for Lassen excludes adaptive routing (would always time out).

### 3.3 Running a single use case manually

```bash
source .venv/bin/activate

# Run UC1 on Frontier only
python src/run_use_cases.py \
    --system frontier \
    --uc 1 \
    --nodes 1000 \
    --duration 60 \
    --delta-t 30 \
    --num-jobs 200 \
    --output-dir output/use_cases/frontier_n1000

# Quick smoke test (5 min, 20 jobs)
python src/run_use_cases.py --quick --uc 1 --system frontier
```

Key CLI flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--system` | `frontier` | `frontier` (dragonfly) or `lassen` (fat-tree) |
| `--uc` | all | Which use cases to run (1 2 3 4) |
| `--nodes` | 1000 | Number of compute nodes |
| `--duration` | 60 | Simulated wall time in minutes |
| `--delta-t` | 30 | Simulation time step in seconds |
| `--num-jobs` | 200 | Jobs injected into the workload |
| `--output-dir` | auto | Directory for CSV output |
| `--force` | off | Overwrite existing CSVs |
| `--quick` | off | Short 5-minute smoke test |

---

## 4. Running Scaling Benchmarks

The benchmark sweeps node count × time quantum × system to measure simulation speed (× real-time speedup).

### 4.1 Submit the benchmark job

```bash
sbatch submit_benchmark_parallel.slurm
```

- Sweeps: `systems = [frontier, lassen]`, `nodes = [100, 1000, 10000]`, `dt = [0.1, 1, 10, 60]`, 3 repeats each.
- Simulates 12 hours of system operation per configuration.
- Results written incrementally to `output/frontier_scaling/results.csv`.
- Auto-resubmits up to 10× (the slowest config — frontier, 10000 nodes, Δt = 0.1 s — takes many hours total).

### 4.2 Run locally

```bash
source .venv/bin/activate

python src/run_frontier.py \
    --systems frontier lassen \
    --nodes 100 1000 \
    --dt 1 10 60 \
    --repeats 3 \
    --duration 12 \
    --workers 1 \
    --output output/frontier_scaling
```

Key CLI flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--systems` | `frontier lassen` | Systems to benchmark |
| `--nodes` | `100 1000 10000` | Node counts |
| `--dt` | `0.1 1 10 60` | Time quanta (seconds) |
| `--repeats` | `3` | Repeats per config |
| `--duration` | `12` | Simulated hours |
| `--workers` | `1` | Parallel workers (use 1 on Frontier to avoid spawn crashes) |
| `--output` | required | Output directory |

Results are written to `{output}/results.csv` with columns:
`system, node_count, delta_t, repeat, status, speedup, per_tick_ms, avg_congestion, …`

---

## 5. Dragonfly Congestion Visualization

Generates chord diagrams showing inter-group link utilization on the Frontier dragonfly network.

### 5.1 Minimal routing (concentrated congestion)

```bash
source .venv/bin/activate

.venv/bin/python3 scripts/run_frontier_congestion_snapshot.py \
    --routing minimal \
    --output output/figures/main/dragonfly_congestion.png
```

Produces a diagram with 4 congested groups, 6 overloaded inter-group link pairs (peak ~241% utilization).

### 5.2 Valiant routing (distributed load, bias = 0.3)

```bash
.venv/bin/python3 scripts/run_frontier_congestion_snapshot.py \
    --routing valiant \
    --valiant-bias 0.3 \
    --output output/figures/main/dragonfly_congestion_ugal.png
```

Produces a diagram with all 49 groups active, 186 inter-group pairs (peak ~176% utilization).

> **Why Valiant instead of UGAL?** In the RAPS dragonfly model, UGAL degenerates to minimal
> routing in all tested multi-job scenarios because each router has unique global links — different
> jobs never contend for the same physical link, so UGAL always perceives zero load on direct paths
> and chooses minimal. The UC simulations use Valiant with `valiant_bias = 0.3` (30% non-minimal
> traffic) and show measurably different energy outcomes, so the visualization uses the same setting.

### 5.3 Standalone chord diagram from a CSV

```bash
.venv/bin/python3 scripts/plot_dragonfly_congestion.py \
    output/links_snapshot/snapshot_minimal.csv \
    -s config/frontier.yaml \
    --title "Frontier Dragonfly — Minimal Routing" \
    -o output/figures/dragonfly_congestion.png
```

CSV format: `src,dst,bytes` (one row per directed link, zero-load links omitted).

---

## 6. Generating All Figures

All scripts auto-detect available results and fall back to the standard (non-heavy) dataset if
the heavy dataset is incomplete.

```bash
source .venv/bin/activate

# UC1–UC4 impact figures (requires completed use case CSVs)
python src/plot_energy_impact.py

# Scaling benchmark figures (speedup, accuracy, speedup-vs-dt)
python src/plot_benchmark_comparison.py

# Energy overhead summary (bar chart across all UC variants)
python src/plot_energy_overhead.py

# Δt tradeoff figures
python src/plot_dt_tradeoff.py

# Motivation figures (sim speed, congestion patterns, sim comparison)
python src/plot_motivation.py

# Dragonfly chord diagrams
.venv/bin/python3 scripts/run_frontier_congestion_snapshot.py --routing minimal \
    --output output/figures/main/dragonfly_congestion.png
.venv/bin/python3 scripts/run_frontier_congestion_snapshot.py --routing valiant \
    --valiant-bias 0.3 \
    --output output/figures/main/dragonfly_congestion_ugal.png
```

All figures are saved to `output/figures/main/`.

### Figure → Paper section mapping

| File | Content |
|------|---------|
| `dragonfly_congestion.png` | Minimal routing chord diagram |
| `dragonfly_congestion_ugal.png` | Valiant (UGAL) routing chord diagram |
| `uc1_routing_stall*.png` | UC1 congestion heatmap |
| `uc1_routing_slowdown*.png` | UC1 job slowdown CDF |
| `uc2_scheduling_*.png` | UC2 scheduler comparison |
| `uc3_placement_*.png` | UC3 placement locality & stall |
| `uc4_energy_*.png` | UC4 energy breakdown |
| `benchmark_speedup.png` | Simulation speedup vs node count |
| `benchmark_accuracy.png` | RAPS vs GPCNeT accuracy |
| `benchmark_speedup_vs_dt.png` | Speedup vs time quantum |
| `energy_overhead.png` | Cross-system energy overhead summary |
| `motiv_speed.png` | Motivation: simulation speed |
| `motiv_congestion.png` | Motivation: congestion patterns |
| `motiv_sim_cmp.png` | Motivation: simulator comparison |

---

## 7. Output Layout

```
output/
├── use_cases/
│   ├── frontier_n1000/              # Standard Frontier results
│   │   ├── uc1_routing_results.csv
│   │   ├── uc2_scheduling_results.csv
│   │   ├── uc3_placement_results.csv
│   │   └── uc4_energy_results.csv
│   ├── lassen_n1000/                # Standard Lassen results
│   ├── frontier_n1000_heavy/        # Heavy-load Frontier results
│   └── lassen_n1000_heavy/          # Heavy-load Lassen results
├── frontier_scaling/
│   ├── results.csv                  # All benchmark runs (incremental)
│   └── {system}_{nodes}n_dt{dt}_r{repeat}/  # Per-run RAPS output
├── links_snapshot/                  # Link-load CSVs for chord diagrams
└── figures/
    └── main/                        # Final paper figures (PNG)
```

### UC result CSV columns

Each `uc{N}_*.csv` contains one row per algorithm variant:

| Column | Description |
|--------|-------------|
| `label` | Algorithm variant (e.g., `minimal`, `ugal`, `fcfs`, `contiguous`) |
| `jobs_completed` | Jobs that finished within wall time |
| `avg_job_slowdown` | Mean slowdown factor due to congestion |
| `total_energy_mj` | Total system energy in MJ |
| `makespan_min` | Total simulated time in minutes |
| `avg_congestion` | Mean network link utilization |

### Benchmark result CSV columns

| Column | Description |
|--------|-------------|
| `system` | `frontier` or `lassen` |
| `node_count` | 100 / 1000 / 10000 |
| `delta_t` | Time quantum (seconds) |
| `repeat` | Repeat index (0–2) |
| `speedup` | Simulation speed / real time |
| `per_tick_ms` | Wall time per simulation tick |
| `avg_congestion` | Mean link utilization across ticks |

---

## Troubleshooting

**`ImportError: cannot import name 'cached_property'`**
System Python is too old (< 3.8). Use the virtualenv: `source .venv/bin/activate`.

**SLURM job always times out before finishing**
The account `GEN053` has a hard 2-hour limit. The auto-resubmit mechanism handles this:
scripts checkpoint progress and restart from where they left off. Set `MAX_RESUBMIT` higher
in the script if more restarts are needed.

**UGAL routing results identical to minimal**
See Section 5.2. This is a known limitation of the RAPS dragonfly model's link-assignment scheme.

**Lassen UC simulations crash midway**
Ensure you are using the code from this repo (not an older version). Three critical fixes were
applied: (1) `apply_dilation` now updates `expected_run_time`; (2) the walltime-kill condition
uses elapsed time only; (3) `prepare_timestep` passes `replay=False` to actually kill timed-out jobs.

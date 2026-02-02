# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ExaDigiT/RAPS (Resource Allocator and Power Simulator) is a discrete-event simulator for HPC systems that schedules workloads and estimates dynamic system power. It supports synthetic workloads, telemetry replay from real supercomputers, network simulation, and cooling model integration via FMUs.

## Build and Development Commands

```bash
# Install in editable mode (requires Python 3.12+)
pip install -e .

# Run tests (parallel execution with xdist)
RAPS_DATA_DIR=/opt/data pytest -n auto -x

# Run specific test markers
pytest -m network           # Network model tests only
pytest -m unit              # Unit tests only
pytest -k "multi_part_sim"  # Filter by test name

# Run a single test file
RAPS_DATA_DIR=/opt/data pytest tests/systems/test_engine.py

# Fetch cooling FMU models
make fetch-example-fmus

# CLI help
raps run -h
```

## Architecture Overview

### Core Simulation Loop
- `raps/engine.py`: `Engine` class - main simulation loop that orchestrates scheduling, power computation, network simulation, and cooling. The `tick()` method advances simulation by `time_delta` seconds.
- `raps/multi_part_engine.py`: `MultiPartEngine` - runs multiple partitions (e.g., CPU + GPU) concurrently.
- `main.py`: CLI entry point, installed as the `raps` command. Uses subparsers for `run`, `run-parts`, `telemetry`, etc.

### Configuration System
- `config/`: YAML files defining system specs (nodes, power, network topology, cooling). Referenced by `--system` flag.
- `raps/system_config.py`: `SystemConfig` - parses YAML configs into typed dataclasses.
- `raps/sim_config.py`: `SimConfig` - simulation parameters (time, workload, scheduling policy).

### Scheduling and Jobs
- `raps/job.py`: `Job` class with states (PENDING, RUNNING, COMPLETED, KILLED) and communication patterns (ALL_TO_ALL, STENCIL_3D).
- `raps/policy.py`: `PolicyType` (FCFS, BACKFILL, REPLAY) and `AllocationStrategy` (contiguous, random, hybrid).
- `raps/schedulers/`: Scheduler implementations including RL-based (`rl.py`) and third-party (`scheduleflow.py`).
- `raps/resmgr/`: Resource managers that handle node allocation.

### Network Simulation
- `raps/network/`: Network topology models and congestion simulation.
  - `fat_tree.py`: Fat-tree topology builder.
  - `torus3d.py`: 3D torus topology with routing.
  - `dragonfly.py`: Dragonfly topology.
  - `base.py`: Common functions for link loads, congestion, slowdown.
- Communication patterns: `CommunicationPattern.ALL_TO_ALL` and `CommunicationPattern.STENCIL_3D` with message size overhead.

### Data Loaders (Telemetry Replay)
- `raps/dataloaders/`: System-specific parsers for telemetry data (Frontier, Lassen, Marconi100, MIT Supercloud, etc.).
- Each loader transforms raw telemetry (Parquet, CSV) into `Job` objects with traces.

### Workload Generation
- `raps/workloads/`: Synthetic workload generators.
  - `basic.py`: Standard synthetic workloads.
  - `network.py`, `inter_job_congestion.py`: Network-focused test workloads.
  - `allocation_test.py`: For testing node allocation strategies.

### Power and Cooling
- `raps/power.py`: `PowerManager` - computes node power from CPU/GPU utilization traces.
- `raps/cooling.py`: `ThermoFluidsModel` - interfaces with FMU cooling models.
- `models/POWER9CSM/fmus/`: Pre-built FMU files for thermal simulation.

## Key Patterns

### Running Simulations
```bash
# Default synthetic workload
raps run

# Telemetry replay with specific system
raps run --system lassen -f /opt/data/lassen/dataset --policy fcfs -t 12h

# Network simulation enabled
raps run --system lassen -w network_test --net -t 15m

# Multi-partition simulation
raps run-parts -x setonix  # Runs setonix/part-cpu and setonix/part-gpu
```

### Extending the Simulator
- New system: Add YAML config to `config/` following existing patterns.
- New dataloader: Add module to `raps/dataloaders/` implementing job parsing.
- New workload: Add to `raps/workloads/` and register in `__init__.py`.
- New network topology: Add to `raps/network/` with builder function and integrate in `NetworkModel`.

## Test Markers (pytest.ini)
- `unit`, `system`: Test scope.
- `network`: Network model tests.
- `withdata`, `nodata`: Data dependency.
- System-specific: `frontier`, `lassen`, `marconi100`, `mit_supercloud`, etc.

## Environment Variables
- `RAPS_DATA_DIR`: Path to telemetry datasets (required for data-backed tests).

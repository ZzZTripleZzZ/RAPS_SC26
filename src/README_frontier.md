# Running RAPS Scaling Experiments on Frontier

This guide covers environment setup, Claude Code configuration, tmux usage,
and how to launch the full experiment suite on OLCF Frontier.

## 1. Environment Setup on Frontier

### 1.1 Login

```bash
ssh <username>@frontier.olcf.ornl.gov
```

### 1.2 Clone the Repository

```bash
cd $HOME/projects/<project_id>
git clone <repo_url> raps
cd raps
```

### 1.3 Create a Python Environment

Frontier uses modules. Load Python and create a virtual environment:

```bash
module load cpe/23.12
module load cray-python/3.11.5

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Verify the installation:

```bash
python -c "from raps.engine import Engine; print('RAPS OK')"
```

### 1.4 Set Data Directory (if using telemetry data)

If you have telemetry datasets on Frontier's filesystem:

```bash
export RAPS_DATA_DIR=/path/to/your/data
```

For synthetic-only experiments (which `run_frontier.py` uses by default), this is
not required.

## 2. Claude Code Setup

### 2.1 Install Claude Code

```bash
# Install Node.js (if not available via modules)
module load nodejs

# Install Claude Code globally
npm install -g @anthropic-ai/claude-code

# Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 2.2 Launch Claude Code

```bash
cd $HOME/projects/<project_id>/raps
claude
```

Claude Code will automatically detect the project via `CLAUDE.md` and understand
the codebase structure.

### 2.3 Using Claude Code with the Experiments

Inside Claude Code, you can:

```
# Check the experiment grid
/bash python src/run_frontier.py --dry-run

# Run a quick test
/bash python src/run_frontier.py --systems lassen --nodes 100 --dt 60 --repeats 1 --duration 0.5

# Launch the full experiment suite
/bash python src/run_frontier.py -j 8
```

## 3. tmux Setup

tmux is essential for long-running experiments on Frontier. Experiments can run
for many hours, and a dropped SSH connection would kill the process without tmux.

### 3.1 Start a New tmux Session

```bash
tmux new-session -s raps-experiments
```

### 3.2 tmux Pane Layout (Recommended)

Create a 3-pane layout for monitoring:

```
# Split horizontally
Ctrl-b "

# Split the bottom pane vertically
Ctrl-b %

# Navigate between panes
Ctrl-b <arrow-key>
```

Layout:

```
+---------------------------------------------+
|  Pane 0: Main experiment runner              |
|  (python src/run_frontier.py ...)            |
+----------------------+----------------------+
|  Pane 1: htop        |  Pane 2: tail logs   |
|  (htop)              |  (tail -f output/..  |
+----------------------+----------------------+
```

### 3.3 Running Experiments in tmux

In **Pane 0** (top):

```bash
cd $HOME/projects/<project_id>/raps
source .venv/bin/activate
python src/run_frontier.py -j 8 2>&1 | tee experiment_log.txt
```

In **Pane 1** (bottom-left):

```bash
htop
```

In **Pane 2** (bottom-right):

```bash
watch -n 5 'wc -l output/frontier_scaling/results.csv'
```

### 3.4 Detach and Reattach

```bash
# Detach (keep running in background)
Ctrl-b d

# List sessions
tmux ls

# Reattach
tmux attach -t raps-experiments
```

### 3.5 Running Claude Code Inside tmux

You can also run Claude Code inside a tmux pane. This lets you interact with
Claude while experiments run in another pane:

```
+---------------------------------------------+
|  Pane 0: claude (Claude Code)                |
+---------------------------------------------+
|  Pane 1: python src/run_frontier.py ...      |
+---------------------------------------------+
```

## 4. Running the Experiments

### 4.1 Full Experiment Suite

The default configuration runs all combinations:

| Parameter     | Values                          | Count |
|---------------|---------------------------------|-------|
| Systems       | lassen (fat-tree), frontier (dragonfly) | 2 |
| Node counts   | 100, 1000, 10000, 100000        | 4     |
| Time quanta   | 0.1s, 1s, 10s, 60s             | 4     |
| Repeats       | 3                               | 3     |
| **Total**     |                                 | **96** |

Each experiment simulates 12 hours. Estimated wall time depends on node count
and time quantum:

| Nodes   | dt=60s   | dt=10s   | dt=1s      | dt=0.1s     |
|---------|----------|----------|------------|-------------|
| 100     | ~20 min  | ~20 min  | ~20 min    | ~3 hours    |
| 1,000   | ~30 min  | ~30 min  | ~30 min    | ~5 hours    |
| 10,000  | ~1 hour  | ~1 hour  | ~1 hour    | ~10 hours   |
| 100,000 | ~3 hours | ~3 hours | ~3 hours   | ~30+ hours  |

**Note**: These are rough estimates. Actual times depend on how many jobs are
concurrently running and the complexity of the network topology.

### 4.2 Launch Commands

```bash
# Full suite (96 experiments, 8 workers in parallel)
python src/run_frontier.py -j 8

# Dry run first to verify the grid
python src/run_frontier.py --dry-run

# Quick smoke test (3-minute simulations)
python src/run_frontier.py --duration 0.05 --repeats 1 -j 4

# Only Frontier system, all node counts
python src/run_frontier.py --systems frontier -j 4

# Only small node counts (fast experiments)
python src/run_frontier.py --nodes 100 1000 -j 4

# Only coarse time quanta (fast experiments)
python src/run_frontier.py --dt 10 60 -j 4

# Single specific experiment for debugging
python src/run_frontier.py --systems lassen --nodes 100 --dt 60 --repeats 1 -j 1
```

### 4.3 Choosing the Number of Workers (`-j`)

On a Frontier login node, use `-j 4` or `-j 8` (be considerate of shared
resources). On a compute node allocation, you can use more:

```bash
# Request an interactive node
salloc -A <project_id> -t 24:00:00 -N 1

# Use all cores
python src/run_frontier.py -j $(nproc)
```

For large node counts (10k, 100k), each worker uses significant memory for the
network graph. Monitor with `htop` and reduce `-j` if memory is tight.

### 4.4 Incremental Runs

If you need to re-run only specific configurations (e.g., after a failure):

```bash
# Only the failed node count
python src/run_frontier.py --nodes 100000 --dt 0.1 --repeats 1 -j 1
```

The output CSV will be written to a new file. You can merge results manually
or re-run the full suite.

## 5. Output

### 5.1 Results CSV

All results are written to `output/frontier_scaling/results.csv`:

```
system,node_count,delta_t,repeat,label,status,ticks,engine_init_s,sim_wall_s,total_wall_s,per_tick_ms,speedup,jobs_total,jobs_completed,avg_net_util_pct,avg_slowdown,max_slowdown,avg_congestion,max_congestion
```

Key columns:

| Column           | Description                                          |
|------------------|------------------------------------------------------|
| `per_tick_ms`    | Wall-clock milliseconds per simulation tick           |
| `speedup`        | Ratio of simulated time to wall time (higher=faster) |
| `engine_init_s`  | Time to build the network graph and engine            |
| `sim_wall_s`     | Wall-clock time for the simulation loop only          |
| `avg_congestion` | Average inter-job network congestion                  |

### 5.2 Per-Experiment Output

Each experiment writes its RAPS output (power CSVs, etc.) to a subdirectory:

```
output/frontier_scaling/
  results.csv
  lassen_n100_dt1_r0/
  lassen_n100_dt1_r1/
  lassen_n100_dt1_r2/
  frontier_n1000_dt10_r0/
  ...
```

### 5.3 Quick Analysis

```python
import pandas as pd

df = pd.read_csv("output/frontier_scaling/results.csv")

# Average over repeats
summary = df.groupby(["system", "node_count", "delta_t"]).agg({
    "per_tick_ms": "mean",
    "speedup": "mean",
    "engine_init_s": "mean",
    "sim_wall_s": "mean",
}).round(2)

print(summary)
```

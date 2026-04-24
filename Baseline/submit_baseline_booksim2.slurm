#!/bin/bash
#SBATCH --job-name=baseline_booksim2
#SBATCH --account=gen053
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --signal=B:USR1@180

# BookSim2 Baseline Sweep — SLURM submission with checkpoint/resubmit
# Dragonfly 1056-node: k=4 → 33g×8r×4h = 1056 hosts
# Dragonfly 72-node:   k=2 → 9g×4r×2h  = 72 hosts
# Completed rho values are skipped on resubmit (skip logic in run_sweep.sh).
#
# Submit: sbatch Baseline/submit_baseline_booksim2.slurm

MAX_RESUBMIT=5
RESUBMIT_COUNT=${SLURM_RESUBMIT_COUNT:-0}

_CHILD_PID=""
_handle_timeout() {
    echo "Timeout signal received at $(date), resubmitting ($RESUBMIT_COUNT/$MAX_RESUBMIT)..."
    [[ -n "$_CHILD_PID" ]] && kill -TERM "$_CHILD_PID" 2>/dev/null
    wait
    if [[ $RESUBMIT_COUNT -lt $MAX_RESUBMIT ]]; then
        SLURM_RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) sbatch --export=ALL "$0"
    fi
    exit 99
}
trap '_handle_timeout' SIGUSR1

set -euo pipefail

cd /lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26

echo "=== BookSim2 Baseline Sweep ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job:  $SLURM_JOB_ID"
echo "Resubmit: $RESUBMIT_COUNT/$MAX_RESUBMIT"
echo ""

# Load GCC environment (BookSim2 uses C++11)
module load PrgEnv-gnu 2>/dev/null || true
which g++ && g++ --version | head -1

# Skip build on resubmit (binary already exists)
SKIP_BUILD_FLAG=""
[[ $RESUBMIT_COUNT -gt 0 ]] && SKIP_BUILD_FLAG="--skip-build"

# 1000-node dragonfly (primary comparison target, may need resubmit for rho=0.5-0.8)
echo "--- 1000-node dragonfly sweep ---"
bash Baseline/booksim2/run_sweep.sh $SKIP_BUILD_FLAG --nodes 1000 --topo dragonfly 2>&1 &
_CHILD_PID=$!
wait $_CHILD_PID
_CHILD_PID=""

# 72-node reference dragonfly
echo "--- 72-node reference dragonfly sweep ---"
bash Baseline/booksim2/run_sweep.sh --skip-build --nodes 72 --topo dragonfly 2>&1 &
_CHILD_PID=$!
wait $_CHILD_PID
_CHILD_PID=""

# Fat-tree
echo "--- fat-tree sweep ---"
bash Baseline/booksim2/run_sweep.sh --skip-build --nodes 16 --topo fattree 2>&1 &
_CHILD_PID=$!
wait $_CHILD_PID
_CHILD_PID=""

echo ""
echo "=== Sweep complete at $(date) ==="
echo "Results in Baseline/booksim2/output/"

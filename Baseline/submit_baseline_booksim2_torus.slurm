#!/bin/bash
#SBATCH --job-name=baseline_booksim2_torus
#SBATCH --account=gen053
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --signal=B:USR1@180

# BookSim2 Baseline Sweep — Torus3d + Fattree with checkpoint/resubmit
# Completed rho values are skipped on resubmit (skip logic in run_sweep.sh).
#
# Submit: sbatch Baseline/submit_baseline_booksim2_torus.slurm
# Output: Baseline/booksim2/output/torus3d/ and fattree/

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

echo "=== BookSim2 Torus3d + Fattree Baseline Sweep ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job:  $SLURM_JOB_ID"
echo "Resubmit: $RESUBMIT_COUNT/$MAX_RESUBMIT"
echo ""

module load PrgEnv-gnu 2>/dev/null || true
which g++ && g++ --version | head -1

BOOKSIM="Baseline/booksim2/build/src/booksim"
if [[ ! -x "$BOOKSIM" ]]; then
    echo "ERROR: BookSim2 binary not found at $BOOKSIM"
    echo "  Run submit_baseline_booksim2.slurm first to build."
    exit 1
fi

# Fat-tree
echo "--- fat-tree k=4 sweep ---"
bash Baseline/booksim2/run_sweep.sh --skip-build --nodes 16 --topo fattree 2>&1 &
_CHILD_PID=$!
wait $_CHILD_PID
_CHILD_PID=""

# Torus3d (8x8x8x2 = 1024 hosts, pass --nodes 1000 to match default filter)
echo "--- torus3d 8x8x8x2 sweep ---"
bash Baseline/booksim2/run_sweep.sh --skip-build --nodes 1000 --topo torus3d 2>&1 &
_CHILD_PID=$!
wait $_CHILD_PID
_CHILD_PID=""

echo ""
echo "=== Sweep complete at $(date) ==="
echo "Results in Baseline/booksim2/output/fattree/ and torus3d/"

#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J raps_bench
#SBATCH -o output/frontier_scaling/benchmark-%j.out
#SBATCH -e output/frontier_scaling/benchmark-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhangzifan@ornl.gov
#SBATCH --signal=B:USR1@180

# Parallel benchmark runner: 32 workers (fork context, no mp.spawn crashes).
# First run uses --force to clear stale results.  Resubmissions use resume logic
# to skip already-completed experiments.
# On timeout: SIGUSR1 → bash trap → SIGTERM to Python → checkpoint → resubmit.
MAX_RESUBMIT=10
RESUBMIT_COUNT=${SLURM_RESUBMIT_COUNT:-0}

_handle_timeout() {
    echo "Timeout signal at $(date)"
    kill -TERM "$child_pid" 2>/dev/null
    sleep 60                              # give Python time to save checkpoint
    kill -KILL "$child_pid" 2>/dev/null   # force exit if still running
    wait "$child_pid" 2>/dev/null
    if [ $RESUBMIT_COUNT -lt $MAX_RESUBMIT ]; then
        echo "Resubmitting (attempt $((RESUBMIT_COUNT + 1))/$MAX_RESUBMIT)..."
        SLURM_RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) sbatch --export=ALL "$0"
    fi
    exit 99
}
trap '_handle_timeout' SIGUSR1

echo "=========================================="
echo "RAPS Benchmark Experiments (Parallel)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Nodelist: $SLURM_NODELIST"
echo "Resubmit: $RESUBMIT_COUNT/$MAX_RESUBMIT"
echo "Start time: $(date)"
echo ""

# Load environment
module load PrgEnv-gnu cray-python
source .venv/bin/activate

mkdir -p output/frontier_scaling

echo "Configuration:"
echo "  - Systems: lassen, frontier, bluewaters"
echo "  - Node counts: 100, 1000, 10000"
echo "  - Time quanta: 0.1s, 1s, 10s, 60s"
echo "  - Duration: 12 hours simulated"
echo "  - Workers: 32 (fork-based parallel)"
echo ""

# First run: --force clears stale results CSV and reruns everything from scratch.
# Resubmits: no --force, resume logic skips already-completed experiments.
FORCE_FLAG=""
[ "$RESUBMIT_COUNT" -eq 0 ] && FORCE_FLAG="--force"

python -u src/run_frontier.py \
    --systems lassen frontier bluewaters \
    --nodes 100 1000 10000 \
    --dt 0.1 1 10 60 \
    --repeats 3 \
    --duration 12 \
    --workers 32 \
    --output output/frontier_scaling \
    $FORCE_FLAG &
child_pid=$!

wait "$child_pid"
rc=$?

echo ""
echo "=========================================="
echo "Benchmark job finished"
echo "Exit code: $rc"
echo "End time: $(date)"
echo "=========================================="

if [ $rc -eq 0 ]; then
    echo "All experiments complete."
else
    echo "Job exited with rc=$rc (interrupted or error)"
fi

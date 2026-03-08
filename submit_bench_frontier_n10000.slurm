#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J raps_bench_frontier_10000
#SBATCH -o output/frontier_scaling/bench_frontier_10000-%j.out
#SBATCH -e output/frontier_scaling/bench_frontier_10000-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --signal=B:USR1@180
#SBATCH --requeue

# frontier n=10000: 4 dt × 3 repeats = 12 experiments run with 12 parallel workers.
# --requeue: SLURM auto-requeues on node failure (SIGKILL cannot be caught in bash).
# --force is skipped if checkpoints already exist, so requeue is safe even when
# SLURM resets SLURM_RESUBMIT_COUNT to 0 on automatic requeue.
MAX_RESUBMIT=10
RESUBMIT_COUNT=${SLURM_RESUBMIT_COUNT:-0}

_handle_timeout() {
    echo "Timeout at $(date)"
    kill -TERM "$child_pid" 2>/dev/null
    sleep 60
    kill -KILL "$child_pid" 2>/dev/null
    wait "$child_pid" 2>/dev/null
    if [ $RESUBMIT_COUNT -lt $MAX_RESUBMIT ]; then
        SLURM_RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) sbatch --export=ALL "$0"
    fi
    exit 99
}
trap '_handle_timeout' SIGUSR1

echo "frontier n=10000 | Job $SLURM_JOB_ID | Resub $RESUBMIT_COUNT/$MAX_RESUBMIT | $(date)"
module load PrgEnv-gnu cray-python
source .venv/bin/activate
mkdir -p output/frontier_scaling

# Only force-clear if this is the very first run AND no checkpoints exist yet.
# This protects against --requeue resetting RESUBMIT_COUNT to 0 on node failure.
FORCE_FLAG=""
if [ "$RESUBMIT_COUNT" -eq 0 ]; then
    has_ckpt=$(ls output/frontier_scaling/frontier_n10000_*/checkpoints/engine.ckpt 2>/dev/null | wc -l)
    [ "$has_ckpt" -eq 0 ] && FORCE_FLAG="--force"
fi

python -u src/run_frontier.py \
    --systems frontier \
    --nodes 10000 \
    --dt 0.1 1 10 60 \
    --repeats 3 --duration 12 \
    --workers 12 \
    --output output/frontier_scaling \
    $FORCE_FLAG &
child_pid=$!
wait "$child_pid"
rc=$?
echo "Done | exit $rc | $(date)"
exit $rc

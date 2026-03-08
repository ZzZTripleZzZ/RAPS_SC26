#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J raps_bench_bluewaters_1000
#SBATCH -o output/frontier_scaling/bench_bluewaters_1000-%j.out
#SBATCH -e output/frontier_scaling/bench_bluewaters_1000-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --signal=B:USR1@180

# bluewaters n=1000: 4 dt × 3 repeats = 12 experiments run with 12 parallel workers.
# First run: --force clears stale results.  Resubmit: resume skips completed.
MAX_RESUBMIT=5
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

echo "bluewaters n=1000 | Job $SLURM_JOB_ID | Resub $RESUBMIT_COUNT/$MAX_RESUBMIT | $(date)"
module load PrgEnv-gnu cray-python
source .venv/bin/activate
mkdir -p output/frontier_scaling

FORCE_FLAG=""
[ "$RESUBMIT_COUNT" -eq 0 ] && FORCE_FLAG="--force"

python -u src/run_frontier.py \
    --systems bluewaters \
    --nodes 1000 \
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

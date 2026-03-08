#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J raps_bw_bench
#SBATCH -o output/frontier_scaling/bw_bench-%j.out
#SBATCH -e output/frontier_scaling/bw_bench-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --mail-type=END,FAIL
#SBATCH --signal=B:USR1@180

# Blue Waters (torus3d) scaling benchmark.
# Shares the results.csv with frontier/lassen benchmarks.
# Resume-safe: skips already-completed experiments.

MAX_RESUBMIT=10
RESUBMIT_COUNT=${SLURM_RESUBMIT_COUNT:-0}

_handle_timeout() {
    echo "Timeout signal at $(date)"
    kill -TERM "$child_pid" 2>/dev/null
    sleep 60
    kill -KILL "$child_pid" 2>/dev/null
    wait "$child_pid" 2>/dev/null
    if [ $RESUBMIT_COUNT -lt $MAX_RESUBMIT ]; then
        echo "Resubmitting (attempt $((RESUBMIT_COUNT + 1))/$MAX_RESUBMIT)..."
        SLURM_RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) sbatch --export=ALL "$0"
    fi
    exit 99
}
trap '_handle_timeout' SIGUSR1

echo "=========================================="
echo "Blue Waters Scaling Benchmark"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Resubmit: $RESUBMIT_COUNT/$MAX_RESUBMIT"
echo "Start time: $(date)"
echo ""

module load PrgEnv-gnu cray-python
source .venv/bin/activate

mkdir -p output/frontier_scaling

python -u src/run_frontier.py \
    --systems bluewaters \
    --nodes 100 1000 10000 \
    --dt 0.1 1 10 60 \
    --repeats 3 \
    --duration 12 \
    --workers 1 \
    --output output/frontier_scaling &
child_pid=$!

wait "$child_pid"
rc=$?

echo ""
echo "=========================================="
echo "Benchmark job finished"
echo "Exit code: $rc"
echo "End time: $(date)"
echo "=========================================="

exit $rc

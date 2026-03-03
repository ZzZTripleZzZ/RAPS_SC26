#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J dt01_f10k
#SBATCH -o output/frontier_scaling/dt01_f10k-%j.out
#SBATCH -e output/frontier_scaling/dt01_f10k-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --mail-type=END,FAIL
#SBATCH --signal=B:USR1@180

MAX_RESUBMIT=20
RESUBMIT_COUNT=${SLURM_RESUBMIT_COUNT:-0}

_handle_timeout() {
    echo "Timeout signal received at $(date)"
    kill -USR1 $PID 2>/dev/null
    wait $PID
    if [ $RESUBMIT_COUNT -lt $MAX_RESUBMIT ]; then
        echo "Timeout detected, resubmitting (attempt $((RESUBMIT_COUNT + 1))/$MAX_RESUBMIT)..."
        SLURM_RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) sbatch --export=ALL "$0"
    fi
    exit 99
}
trap '_handle_timeout' SIGUSR1

echo "=========================================="
echo "Frontier n=10000 dt=0.1s (3 repeats)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Resubmit attempt: $RESUBMIT_COUNT/$MAX_RESUBMIT"
echo "Start time: $(date)"
echo ""

module load PrgEnv-gnu cray-python
source .venv/bin/activate

mkdir -p output/frontier_scaling

export RAPS_SINGLE_WORKER=1

python -u src/run_frontier.py \
    --systems frontier \
    --nodes 10000 \
    --dt 0.1 \
    --repeats 3 \
    --duration 12 \
    --workers 1 \
    --output output/frontier_scaling &
PID=$!
wait $PID
exit_code=$?

echo ""
echo "=========================================="
echo "Exit code: $exit_code"
echo "End time: $(date)"
echo "Resubmit count: $RESUBMIT_COUNT"
echo "=========================================="

if [ $exit_code -eq 99 ] && [ $RESUBMIT_COUNT -lt $MAX_RESUBMIT ]; then
    echo "Timeout detected, resubmitting (attempt $((RESUBMIT_COUNT + 1))/$MAX_RESUBMIT)..."
    SLURM_RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) sbatch --export=ALL "$0"
fi

exit $exit_code

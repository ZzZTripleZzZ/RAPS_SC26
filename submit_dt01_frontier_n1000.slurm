#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J dt01_f1k
#SBATCH -o output/frontier_scaling/dt01_f1k-%j.out
#SBATCH -e output/frontier_scaling/dt01_f1k-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhangzifan@ornl.gov
#SBATCH --signal=B:USR1@180

# Optimized: frontier n=1000, dt=0.1s (3 experiments, sequential)
MAX_RESUBMIT=5
RESUBMIT_COUNT=${SLURM_RESUBMIT_COUNT:-0}

trap 'echo "Timeout signal received at $(date)"; exit 99' SIGUSR1

echo "=========================================="
echo "Frontier n=1000 dt=0.1s (Optimized)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Resubmit attempt: $RESUBMIT_COUNT/$MAX_RESUBMIT"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Start time: $(date)"
echo "Strategy: Sequential (1 at a time, lower memory)"
echo ""

# Load environment
module load PrgEnv-gnu cray-python
source .venv/bin/activate

mkdir -p output/frontier_scaling

echo "Running 3 frontier_n1000_dt0.1 experiments sequentially..."
echo "  - r0, r1, r2"
echo "  - Checkpoint interval: 10 minutes (reduced frequency)"
echo ""

# Run sequentially (not parallel) to reduce memory pressure
python -u src/run_frontier.py \
    --systems frontier \
    --nodes 1000 \
    --dt 0.1 \
    --repeats 3 \
    --duration 12 \
    --workers 1 \
    --output output/frontier_scaling

exit_code=$?

echo ""
echo "=========================================="
echo "Frontier n=1000 dt=0.1s finished"
echo "Exit code: $exit_code"
echo "End time: $(date)"
echo "Resubmit count: $RESUBMIT_COUNT"
echo "=========================================="

# Auto-resubmit on timeout
if [ $exit_code -eq 99 ] && [ $RESUBMIT_COUNT -lt $MAX_RESUBMIT ]; then
    echo "Timeout detected, resubmitting (attempt $((RESUBMIT_COUNT + 1))/$MAX_RESUBMIT)..."
    sbatch --export=ALL,SLURM_RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) "$0"
fi

exit $exit_code

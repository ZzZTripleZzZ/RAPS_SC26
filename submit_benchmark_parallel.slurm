#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J raps_bench
#SBATCH -o output/frontier_scaling/benchmark-%j.out
#SBATCH -e output/frontier_scaling/benchmark-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 24
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhangzifan@ornl.gov
#SBATCH --signal=B:USR1@180

# Benchmark experiments with 24-node parallelization
MAX_RESUBMIT=10
RESUBMIT_COUNT=${SLURM_RESUBMIT_COUNT:-0}

trap 'echo "Timeout signal received at $(date)"; exit 99' SIGUSR1

echo "=========================================="
echo "RAPS Benchmark Experiments (Parallel)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Nodelist: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Workers: 24 (parallel experiments)"
echo ""

# Load environment
module load PrgEnv-gnu cray-python
source .venv/bin/activate

# Ensure output directory exists
mkdir -p output/frontier_scaling

echo "Running benchmark with optimized code..."
echo "Configuration:"
echo "  - Systems: lassen, frontier"
echo "  - Node counts: 100, 1000, 10000"
echo "  - Time quanta: 0.1s, 1s, 10s, 60s"
echo "  - Duration: 12 hours simulated"
echo "  - Repeats: 3"
echo "  - Parallel workers: 24"
echo ""

# Run with 24 parallel workers
python -u src/run_frontier.py \
    --systems lassen frontier \
    --nodes 100 1000 10000 \
    --dt 0.1 1 10 60 \
    --repeats 3 \
    --duration 12 \
    --workers 24 \
    --output output/frontier_scaling

exit_code=$?

echo ""
echo "=========================================="
echo "Benchmark job finished"
echo "Exit code: $exit_code"
echo "End time: $(date)"
echo "=========================================="

# Auto-resubmit on timeout
if [ $exit_code -eq 99 ] && [ $RESUBMIT_COUNT -lt $MAX_RESUBMIT ]; then
    echo "Timeout detected, resubmitting (attempt $((RESUBMIT_COUNT + 1))/$MAX_RESUBMIT)..."
    sbatch --export=ALL,SLURM_RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) $0
fi

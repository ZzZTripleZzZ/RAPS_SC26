#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J raps_dt01
#SBATCH -o output/frontier_scaling/missing_dt01-%j.out
#SBATCH -e output/frontier_scaling/missing_dt01-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhangzifan@ornl.gov
#SBATCH --signal=B:USR1@180

# Complete missing dt=0.1s experiments with checkpoint/auto-resubmit
MAX_RESUBMIT=20
RESUBMIT_COUNT=${SLURM_RESUBMIT_COUNT:-0}

trap 'echo "Timeout signal received at $(date)"; exit 99' SIGUSR1

echo "=========================================="
echo "RAPS Missing dt=0.1s Experiments"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Resubmit attempt: $RESUBMIT_COUNT/$MAX_RESUBMIT"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Nodelist: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Workers: 8 (parallel experiments)"
echo ""

# Load environment
module load PrgEnv-gnu cray-python
source .venv/bin/activate

# Ensure output directory exists
mkdir -p output/frontier_scaling

echo "Running 8 missing dt=0.1s experiments:"
echo "  - frontier n=10000 dt=0.1 r=0,1,2 (3 experiments)"
echo "  - frontier n=1000 dt=0.1 r=0,1,2 (3 experiments)"
echo "  - lassen n=10000 dt=0.1 r=0,1 (2 experiments)"
echo ""
echo "Configuration:"
echo "  - Duration: 12 hours simulated"
echo "  - Parallel workers: 8"
echo "  - Checkpoint enabled (every 5 minutes)"
echo "  - Auto-resubmit on timeout (max $MAX_RESUBMIT attempts)"
echo ""

# Run only the missing dt=0.1s experiments with 8 parallel workers
# The script automatically resumes from checkpoint if available
python -u src/run_frontier.py \
    --systems frontier lassen \
    --nodes 1000 10000 \
    --dt 0.1 \
    --repeats 3 \
    --duration 12 \
    --workers 8 \
    --output output/frontier_scaling

exit_code=$?

echo ""
echo "=========================================="
echo "Missing dt=0.1s experiments finished"
echo "Exit code: $exit_code"
echo "End time: $(date)"
echo "Resubmit count: $RESUBMIT_COUNT"
echo "=========================================="

# Auto-resubmit on timeout
if [ $exit_code -eq 99 ] && [ $RESUBMIT_COUNT -lt $MAX_RESUBMIT ]; then
    echo "Timeout detected, resubmitting (attempt $((RESUBMIT_COUNT + 1))/$MAX_RESUBMIT)..."
    echo "Experiments will resume from checkpoint."
    sbatch --export=ALL,SLURM_RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) "$0"
elif [ $RESUBMIT_COUNT -ge $MAX_RESUBMIT ]; then
    echo "WARNING: Maximum resubmit count reached ($MAX_RESUBMIT)"
    echo "Some experiments may be incomplete."
fi

exit $exit_code

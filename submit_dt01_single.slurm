#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J dt01_single
#SBATCH -o output/frontier_scaling/dt01_%x-%j.out
#SBATCH -e output/frontier_scaling/dt01_%x-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --mail-type=FAIL
#SBATCH --signal=B:USR1@180

# Single dt=0.1s experiment (for large n=10000 cases)
# Usage: sbatch --export=SYSTEM=frontier,NODES=10000,REPEAT=0 submit_dt01_single.slurm

MAX_RESUBMIT=10
RESUBMIT_COUNT=${SLURM_RESUBMIT_COUNT:-0}

# On timeout: Python already handles SIGUSR1 (sets flag, checkpoints, exits 99).
# We run Python in background, wait for it, so the resubmit code below can execute.
_PYTHON_PID=""
_handle_timeout() {
    echo "Timeout signal received at $(date)"
    # Python already received SIGUSR1 and will checkpoint+exit with code 99.
    # Just wait for it — do NOT exit bash here so resubmit logic runs.
}
trap '_handle_timeout' SIGUSR1

SYSTEM=${SYSTEM:-frontier}
NODES=${NODES:-10000}
REPEAT=${REPEAT:-0}
LABEL="${SYSTEM}_n${NODES}_dt0.1_r${REPEAT}"

echo "=========================================="
echo "Single dt=0.1s Experiment (Optimized)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Label: $LABEL"
echo "System: $SYSTEM"
echo "Nodes: $NODES"
echo "Repeat: $REPEAT"
echo "Resubmit attempt: $RESUBMIT_COUNT/$MAX_RESUBMIT"
echo "Start time: $(date)"
echo ""

# Load environment
module load PrgEnv-gnu cray-python
source .venv/bin/activate

mkdir -p output/frontier_scaling

echo "Running single experiment with checkpoint..."
echo "  - Checkpoint interval: 10 minutes"
echo "  - Max resubmits: $MAX_RESUBMIT"
echo ""

# Run Python in background so the SIGUSR1 trap doesn't block resubmit logic
python -u src/run_single_experiment.py \
    --system "$SYSTEM" \
    --nodes "$NODES" \
    --dt 0.1 \
    --repeat "$REPEAT" \
    --duration 12 \
    --output output/frontier_scaling &
_PYTHON_PID=$!
wait $_PYTHON_PID
exit_code=$?

echo ""
echo "=========================================="
echo "$LABEL finished"
echo "Exit code: $exit_code"
echo "End time: $(date)"
echo "Resubmit count: $RESUBMIT_COUNT"
echo "=========================================="

# Auto-resubmit on timeout
if [ $exit_code -eq 99 ] && [ $RESUBMIT_COUNT -lt $MAX_RESUBMIT ]; then
    echo "Timeout detected, resubmitting (attempt $((RESUBMIT_COUNT + 1))/$MAX_RESUBMIT)..."
    sbatch --export=ALL,SYSTEM="$SYSTEM",NODES="$NODES",REPEAT="$REPEAT",SLURM_RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) "$0"
fi

exit $exit_code

#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J raps_10k
#SBATCH -o output/raps_10k-%j.out
#SBATCH -e output/raps_10k-%j.err
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH --mail-user=zhangzifan@gatech.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --signal=B:USR1@300

# =============================================================================
# RAPS 10K-Node Experiments (checkpoint/resume chain)
# =============================================================================
# - Single worker (10k experiments are memory-heavy)
# - Graceful checkpoint on pre-timeout signal (5 min before kill)
# - Auto-resubmits until all 10k experiments complete
# - Usage:  sbatch submit_10k.slurm
# =============================================================================

PROJECT_DIR="/lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26"
RESULTS_CSV="$PROJECT_DIR/output/frontier_scaling/results.csv"
# 10k nodes × 4 dt × 3 repeats = 12 experiments per system
# lassen + frontier = 24 total 10k experiments
TOTAL_10K_NEEDED=24
MAX_RESUBMIT=100
RESUBMIT_COUNT=${RAPS_RESUBMIT_COUNT:-0}

# --- Auto-resubmit on exit ---
resubmit_if_needed() {
    # Count completed 10k experiments only
    COMPLETED=$(grep ",10000," "$RESULTS_CSV" 2>/dev/null | grep -c ",OK," || echo 0)
    echo ""
    echo "Exit handler: $COMPLETED / $TOTAL_10K_NEEDED 10k experiments completed"

    if [ "$COMPLETED" -ge "$TOTAL_10K_NEEDED" ]; then
        echo "ALL $TOTAL_10K_NEEDED 10K EXPERIMENTS COMPLETE!"
        return 0
    fi

    if [ "$RESUBMIT_COUNT" -ge "$MAX_RESUBMIT" ]; then
        echo "WARNING: Reached max resubmit limit ($MAX_RESUBMIT)."
        return 0
    fi

    NEXT_COUNT=$((RESUBMIT_COUNT + 1))
    echo "Auto-resubmitting (round $((NEXT_COUNT+1))/$MAX_RESUBMIT)..."
    NEXT_JOB=$(sbatch --export=ALL,RAPS_RESUBMIT_COUNT=$NEXT_COUNT "$PROJECT_DIR/submit_10k.slurm" 2>&1)
    echo "Resubmitted: $NEXT_JOB"
}
trap resubmit_if_needed EXIT

# --- Forward SLURM pre-timeout signal to Python process ---
PYTHON_PID=""
handle_signal() {
    echo ""
    echo ">>> Pre-timeout signal at $(date). Forwarding USR1 to Python (PID=$PYTHON_PID)..."
    if [ -n "$PYTHON_PID" ]; then
        kill -USR1 "$PYTHON_PID" 2>/dev/null || true
        # Give Python time to checkpoint (up to 4 min)
        for i in $(seq 1 240); do
            if ! kill -0 "$PYTHON_PID" 2>/dev/null; then
                echo "Python exited after checkpoint."
                break
            fi
            sleep 1
        done
    fi
}
trap handle_signal USR1

cd "$PROJECT_DIR"

echo "=============================================="
echo "RAPS 10K-Node Experiments (Round $((RESUBMIT_COUNT+1))/$MAX_RESUBMIT)"
echo "Started on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "=============================================="

# Setup environment
module load PrgEnv-gnu 2>/dev/null || true
source "$PROJECT_DIR/.venv/bin/activate"

export RAPS_DATA_DIR="$PROJECT_DIR/data/lassen"
export RAPS_SINGLE_WORKER=1

echo "Python: $(which python)"
echo ""

# Verify data
if [ ! -d "$RAPS_DATA_DIR/Lassen-Supercomputer-Job-Dataset" ]; then
    echo "ERROR: Lassen data not found"
    exit 1
fi

# Run ONLY 10k experiments, 1 worker, sequential execution
# This ensures checkpoint/resume works correctly across SLURM jobs
python src/run_frontier.py \
    --workers 1 \
    --output output/frontier_scaling \
    --nodes 10000 \
    --dt 60 10 1 0.1 \
    --repeats 3 \
    --duration 12 &

PYTHON_PID=$!
echo "Python PID: $PYTHON_PID"

# Wait for python process (allows signal handling)
wait $PYTHON_PID
PYTHON_EXIT=$?

echo ""
echo "Python exited with code $PYTHON_EXIT at $(date)"
# EXIT trap will handle resubmission

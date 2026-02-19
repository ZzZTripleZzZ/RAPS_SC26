#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J raps_scaling
#SBATCH -o output/raps_scaling-%j.out
#SBATCH -e output/raps_scaling-%j.err
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH --mail-user=zhangzifan@gatech.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --signal=B:USR1@300

# =============================================================================
# RAPS Frontier Scaling Experiments (100/1000 nodes only)
# =============================================================================
# - 8 parallel workers on 1 node
# - Results saved incrementally after each experiment
# - Auto-resubmits on timeout
# - For 10k node experiments, use submit_10k.slurm instead
# =============================================================================

PROJECT_DIR="/lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26"
RESULTS_CSV="$PROJECT_DIR/output/frontier_scaling/results.csv"
# 2 systems × 2 node counts (100, 1000) × 4 dt × 3 repeats = 48
TOTAL_NEEDED=48
MAX_RESUBMIT=50
RESUBMIT_COUNT=${RAPS_RESUBMIT_COUNT:-0}

# --- Auto-resubmit on exit ---
resubmit_if_needed() {
    # Count completed non-10k experiments
    COMPLETED_100=$(grep ",100," "$RESULTS_CSV" 2>/dev/null | grep -c ",OK," || echo 0)
    COMPLETED_1000=$(grep ",1000," "$RESULTS_CSV" 2>/dev/null | grep -c ",OK," || echo 0)
    COMPLETED=$((COMPLETED_100 + COMPLETED_1000))
    echo ""
    echo "Exit handler: $COMPLETED / $TOTAL_NEEDED experiments completed (100+1000 nodes)"

    if [ "$COMPLETED" -ge "$TOTAL_NEEDED" ]; then
        echo "ALL $TOTAL_NEEDED EXPERIMENTS COMPLETE!"
        return 0
    fi

    if [ "$RESUBMIT_COUNT" -ge "$MAX_RESUBMIT" ]; then
        echo "WARNING: Reached max resubmit limit ($MAX_RESUBMIT)."
        return 0
    fi

    NEXT_COUNT=$((RESUBMIT_COUNT + 1))
    echo "Auto-resubmitting (round $((NEXT_COUNT+1))/$MAX_RESUBMIT)..."
    NEXT_JOB=$(sbatch --export=ALL,RAPS_RESUBMIT_COUNT=$NEXT_COUNT "$PROJECT_DIR/submit.slurm" 2>&1)
    echo "Resubmitted: $NEXT_JOB"
}
trap resubmit_if_needed EXIT

# --- Forward pre-timeout signal to Python ---
PYTHON_PID=""
handle_signal() {
    echo ""
    echo ">>> Pre-timeout signal at $(date). Forwarding USR1 to Python (PID=$PYTHON_PID)..."
    if [ -n "$PYTHON_PID" ]; then
        kill -USR1 "$PYTHON_PID" 2>/dev/null || true
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
echo "RAPS Scaling Experiments (Round $((RESUBMIT_COUNT+1))/$MAX_RESUBMIT)"
echo "Started on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "=============================================="

# Setup environment
module load PrgEnv-gnu 2>/dev/null || true
source "$PROJECT_DIR/.venv/bin/activate"

export RAPS_DATA_DIR="$PROJECT_DIR/data/lassen"

echo "Python: $(which python)"
echo ""

# Verify data
if [ ! -d "$RAPS_DATA_DIR/Lassen-Supercomputer-Job-Dataset" ]; then
    echo "ERROR: Lassen data not found"
    exit 1
fi

# Run 100/1000 node experiments only (10k handled by submit_10k.slurm)
python src/run_frontier.py \
    --workers 8 \
    --output output/frontier_scaling \
    --nodes 100 1000 \
    --dt 60 10 1 0.1 \
    --repeats 3 \
    --duration 12 &

PYTHON_PID=$!
echo "Python PID: $PYTHON_PID"

wait $PYTHON_PID
PYTHON_EXIT=$?

echo ""
echo "Python exited with code $PYTHON_EXIT at $(date)"
# EXIT trap will handle resubmission

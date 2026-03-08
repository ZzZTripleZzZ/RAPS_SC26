#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J raps_frontier_circ_uc
#SBATCH -o output/use_cases/frontier_circ_uc-%j.out
#SBATCH -e output/use_cases/frontier_circ_uc-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --mail-type=END,FAIL
#SBATCH --signal=B:USR1@180

# ============================================================
# Frontier UCs with circulant dragonfly topology (standard load)
# ============================================================

MAX_RESUBMIT=5
RESUBMIT_COUNT=${SLURM_RESUBMIT_COUNT:-0}

_CHILD_PIDS=()
_handle_timeout() {
    echo "Timeout signal at $(date)"
    for pid in "${_CHILD_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null
    done
    wait
    echo "Children terminated, resubmitting..."
    if [ $RESUBMIT_COUNT -lt $MAX_RESUBMIT ]; then
        SLURM_RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) sbatch --export=ALL "$0"
    fi
    exit 99
}
trap '_handle_timeout' SIGUSR1

echo "=========================================="
echo "Frontier UCs — circulant dragonfly"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Resubmit: $RESUBMIT_COUNT/$MAX_RESUBMIT"
echo "Start: $(date)"
echo ""

module load PrgEnv-gnu cray-python
source .venv/bin/activate

mkdir -p output/use_cases

NUM_JOBS=200
DURATION=60
DELTA_T=30
SYSTEM=frontier
OUT_DIR="output/use_cases/${SYSTEM}_n1000"

mkdir -p "$OUT_DIR"

echo "Config: ${NUM_JOBS} jobs, ${DURATION}min, dt=${DELTA_T}s, 1000 nodes"
echo ""

# Clear old results on first submission
if [ "$RESUBMIT_COUNT" -eq 0 ]; then
    echo "Clearing old Frontier UC results..."
    rm -f "${OUT_DIR}"/uc*.csv 2>/dev/null || true
fi

FORCE_FLAG=""
[ "$RESUBMIT_COUNT" -eq 0 ] && FORCE_FLAG="--force"

for UC in 1 2 3 4; do
    LOG="output/use_cases/${SYSTEM}_n1000_uc${UC}.log"
    echo "[UC${UC}] Starting..."
    python -u src/run_use_cases.py \
        --system "$SYSTEM" \
        --nodes 1000 \
        --duration "$DURATION" \
        --delta-t "$DELTA_T" \
        --num-jobs "$NUM_JOBS" \
        --uc "$UC" \
        $FORCE_FLAG \
        --output-dir "$OUT_DIR" \
        > "$LOG" 2>&1 &
    _CHILD_PIDS+=($!)
done

echo "Waiting for all UCs..."
failed=0
for pid in "${_CHILD_PIDS[@]}"; do
    wait $pid
    [ $? -ne 0 ] && failed=$((failed + 1))
done

echo ""
echo "Results:"
for UC in 1 2 3 4; do
    CSV=$(ls "${OUT_DIR}/uc${UC}_"*.csv 2>/dev/null | head -1)
    if [ -n "$CSV" ] && [ -f "$CSV" ]; then
        LINES=$(($(wc -l < "$CSV") - 1))
        echo "  UC${UC}: $LINES variants in $CSV"
    else
        echo "  UC${UC}: NOT FOUND"
    fi
done

echo ""
echo "Failed: $failed"
echo "End: $(date)"
exit $failed

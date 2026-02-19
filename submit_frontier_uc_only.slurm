#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J raps_uc_frontier
#SBATCH -o output/use_cases/uc_frontier-%j.out
#SBATCH -e output/use_cases/uc_frontier-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --mail-type=END,FAIL
#SBATCH --signal=B:USR1@180

MAX_RESUBMIT=3
RESUBMIT_COUNT=${SLURM_RESUBMIT_COUNT:-0}

_CHILD_PIDS=()
_handle_timeout() {
    echo "Timeout signal received at $(date)"
    for pid in "${_CHILD_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null
    done
    wait
    echo "Children terminated, resubmitting..."
    if [ $RESUBMIT_COUNT -lt $MAX_RESUBMIT ]; then
        sbatch --export=ALL,SLURM_RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) "$0"
    fi
    exit 99
}
trap '_handle_timeout' SIGUSR1

echo "=========================================="
echo "RAPS Frontier-Only UC Rerun"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Resubmit attempt: $RESUBMIT_COUNT/$MAX_RESUBMIT"
echo "Start time: $(date)"
echo ""

module load PrgEnv-gnu cray-python
source .venv/bin/activate

mkdir -p output/use_cases/frontier_n1000

NUM_JOBS=200
DURATION=60
DELTA_T=30

echo "Configuration: frontier, 1000 nodes, ${DELTA_T}s dt, ${DURATION}min, ${NUM_JOBS} jobs"
echo ""

# Only clear/force Frontier CSVs — preserve Lassen CSVs (already correct)
if [ "$RESUBMIT_COUNT" -eq 0 ]; then
    echo "Fresh start: clearing only Frontier UC CSVs (preserving Lassen)..."
    rm -f output/use_cases/frontier_n1000/uc*.csv 2>/dev/null || true
    echo "Done clearing."
else
    echo "Resubmit #$RESUBMIT_COUNT: preserving completed CSVs."
fi
echo ""

run_uc_frontier() {
    local UC=$1
    local OUTPUT_DIR="output/use_cases/frontier_n1000"
    local LOG_FILE="output/use_cases/frontier_n1000_uc${UC}.log"

    echo "[UC${UC}] Starting frontier..."

    # On first run use --force; on resubmit, skip completed variants
    FORCE_FLAG=""
    [ "$RESUBMIT_COUNT" -eq 0 ] && FORCE_FLAG="--force"

    python -u src/run_use_cases.py \
        --system frontier \
        --nodes 1000 \
        --duration "$DURATION" \
        --delta-t "$DELTA_T" \
        --num-jobs "$NUM_JOBS" \
        --uc "$UC" \
        $FORCE_FLAG \
        --output-dir "$OUTPUT_DIR" \
        > "$LOG_FILE" 2>&1

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[UC${UC}] frontier completed successfully"
    else
        echo "[UC${UC}] frontier FAILED with exit code $exit_code"
        echo "  Last 10 lines of log:"
        tail -10 "$LOG_FILE" >&2
    fi
    return $exit_code
}

echo "=== Running all 4 Frontier UCs in parallel ==="
for UC in 1 2 3 4; do
    run_uc_frontier "$UC" &
    _CHILD_PIDS+=($!)
done

failed_count=0
for pid in "${_CHILD_PIDS[@]}"; do
    wait $pid
    [ $? -ne 0 ] && failed_count=$((failed_count + 1))
done

echo ""
echo "=========================================="
echo "Frontier UC experiments finished"
echo "Failed: $failed_count"
echo "End time: $(date)"
echo "=========================================="

echo ""
echo "Results summary:"
for UC in 1 2 3 4; do
    DIR="output/use_cases/frontier_n1000"
    CSV=$(ls "${DIR}/uc${UC}_"*.csv 2>/dev/null | head -1)
    if [ -n "$CSV" ] && [ -f "$CSV" ]; then
        LINES=$(wc -l < "$CSV")
        echo "  OK  UC${UC}: $CSV ($LINES lines)"
    else
        echo "  --  UC${UC}: Not found"
    fi
done

exit $failed_count

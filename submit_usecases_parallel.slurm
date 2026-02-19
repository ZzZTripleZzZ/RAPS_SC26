#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J raps_uc_all
#SBATCH -o output/use_cases/usecases-%j.out
#SBATCH -e output/use_cases/usecases-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --mail-type=END,FAIL
#SBATCH --signal=B:USR1@180

MAX_RESUBMIT=5
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
echo "RAPS Use Case Experiments (v3 - fixed)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Resubmit attempt: $RESUBMIT_COUNT/$MAX_RESUBMIT"
echo "Start time: $(date)"
echo ""

module load PrgEnv-gnu cray-python
source .venv/bin/activate

mkdir -p output/use_cases

NUM_JOBS=200
DURATION=60
DELTA_T=30

echo "Configuration:"
echo "  - Systems: lassen, frontier"
echo "  - Node count: 1000"
echo "  - Time quantum: ${DELTA_T}s"
echo "  - Duration: ${DURATION} minutes simulated"
echo "  - Jobs: ${NUM_JOBS}"
echo "  - Use cases: UC1 (routing), UC2 (scheduling), UC3 (placement), UC4 (energy)"
echo ""

if [ "$RESUBMIT_COUNT" -eq 0 ]; then
    echo "Fresh start: clearing old UC result CSVs..."
    for SYSTEM in lassen frontier; do
        rm -f output/use_cases/${SYSTEM}_n1000/uc*.csv 2>/dev/null || true
    done
    echo "Done clearing."
else
    echo "Resubmit #$RESUBMIT_COUNT: preserving completed CSVs (skip logic will handle incomplete ones)."
fi
echo ""

run_uc() {
    local SYSTEM=$1
    local UC=$2
    local OUTPUT_DIR="output/use_cases/${SYSTEM}_n1000"
    local LOG_FILE="output/use_cases/${SYSTEM}_n1000_uc${UC}.log"

    echo "[UC${UC}] Starting ${SYSTEM}..."

    mkdir -p "$OUTPUT_DIR"

    # On resubmit, omit --force so run_use_cases.py skips already-complete UCs.
    FORCE_FLAG=""
    [ "$RESUBMIT_COUNT" -eq 0 ] && FORCE_FLAG="--force"

    python -u src/run_use_cases.py \
        --system "$SYSTEM" \
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
        echo "[UC${UC}] ${SYSTEM} completed successfully"
    else
        echo "[UC${UC}] ${SYSTEM} FAILED with exit code $exit_code"
        echo "  Last 10 lines of log:"
        tail -10 "$LOG_FILE" >&2
    fi

    return $exit_code
}

# Run Frontier first (faster: dragonfly APSP is quick), then Lassen
echo "=== Phase 1: Frontier UCs (parallel) ==="
for UC in 1 2 3 4; do
    run_uc frontier "$UC" &
    _CHILD_PIDS+=($!)
done
wait "${_CHILD_PIDS[@]}"
_CHILD_PIDS=()
echo ""

echo "=== Phase 2: Lassen UCs (parallel) ==="
for UC in 1 2 3 4; do
    run_uc lassen "$UC" &
    _CHILD_PIDS+=($!)
done

echo "Waiting for all Lassen UCs..."
failed_count=0
for pid in "${_CHILD_PIDS[@]}"; do
    wait $pid
    [ $? -ne 0 ] && failed_count=$((failed_count + 1))
done

echo ""
echo "=========================================="
echo "Use case experiments finished"
echo "Failed: $failed_count"
echo "End time: $(date)"
echo "=========================================="

echo ""
echo "Results summary:"
for SYSTEM in lassen frontier; do
    echo ""
    echo "$SYSTEM use cases:"
    for UC in 1 2 3 4; do
        DIR="output/use_cases/${SYSTEM}_n1000"
        CSV=$(ls "${DIR}/uc${UC}_"*.csv 2>/dev/null | head -1)
        if [ -n "$CSV" ] && [ -f "$CSV" ]; then
            LINES=$(wc -l < "$CSV")
            echo "  OK  UC${UC}: $CSV ($LINES lines)"
        else
            echo "  --  UC${UC}: Not found"
        fi
    done
done

exit $failed_count

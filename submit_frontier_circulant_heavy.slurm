#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J raps_frontier_circ_heavy
#SBATCH -o output/use_cases/frontier_circ_heavy-%j.out
#SBATCH -e output/use_cases/frontier_circ_heavy-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --mail-type=END,FAIL
#SBATCH --signal=B:USR1@180

# ============================================================
# Frontier UCs with circulant dragonfly — HEAVY LOAD
# UC1/UC2/UC4: 300 jobs, dt=1s
# UC3: 300 jobs, dt=5s
# ============================================================

MAX_RESUBMIT=10
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
echo "Frontier Heavy UCs — circulant dragonfly"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Resubmit: $RESUBMIT_COUNT/$MAX_RESUBMIT"
echo "Start: $(date)"
echo ""

module load PrgEnv-gnu cray-python
source .venv/bin/activate

SYSTEM=frontier
NUM_JOBS=300
DURATION=60
NODES=1000
OUT_DIR="output/use_cases/${SYSTEM}_n${NODES}_heavy"

mkdir -p "$OUT_DIR"

echo "Config: ${NUM_JOBS} jobs, ${DURATION}min, ${NODES} nodes (heavy load)"
echo ""

# Clear old results on first submission
if [ "$RESUBMIT_COUNT" -eq 0 ]; then
    echo "Clearing old heavy Frontier UC results..."
    rm -f "${OUT_DIR}"/uc*.csv 2>/dev/null || true
fi

FORCE_FLAG=""
[ "$RESUBMIT_COUNT" -eq 0 ] && FORCE_FLAG="--force"

# UC1, UC2, UC4: dt=1
for UC in 1 2 4; do
    LOG="${OUT_DIR}/uc${UC}_heavy.log"
    echo "[UC${UC}] Starting (dt=1s)..."
    python -u src/run_use_cases.py \
        --system "$SYSTEM" \
        --nodes "$NODES" \
        --duration "$DURATION" \
        --delta-t 1 \
        --num-jobs "$NUM_JOBS" \
        --uc "$UC" \
        --output-dir "$OUT_DIR" \
        $FORCE_FLAG \
        > "$LOG" 2>&1 &
    _CHILD_PIDS+=($!)
done

# UC3: dt=5 (placement experiment needs coarser dt for reasonable runtime)
UC=3
LOG="${OUT_DIR}/uc${UC}_heavy.log"
echo "[UC${UC}] Starting (dt=5s)..."
python -u src/run_use_cases.py \
    --system "$SYSTEM" \
    --nodes "$NODES" \
    --duration "$DURATION" \
    --delta-t 5 \
    --num-jobs "$NUM_JOBS" \
    --uc "$UC" \
    --output-dir "$OUT_DIR" \
    $FORCE_FLAG \
    > "$LOG" 2>&1 &
_CHILD_PIDS+=($!)

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

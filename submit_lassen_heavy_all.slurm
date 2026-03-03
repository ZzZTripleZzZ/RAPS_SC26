#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J raps_l_heavy
#SBATCH -o output/use_cases/lassen_heavy_all-%j.out
#SBATCH -e output/use_cases/lassen_heavy_all-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --mail-type=END,FAIL
#SBATCH --signal=B:USR1@180

# ============================================================
# Dedicated job: Lassen heavy UC1+UC2+UC3+UC4 (all in parallel)
# Account walltime limit: 120 min. dt=1s is too slow for adaptive routing.
# Solution: dt=10s → adaptive ~88 min/config (fits in 2h per config).
# All existing lassen_n1000_heavy CSVs deleted for consistency.
# UC1: 1 adaptive config (~88 min) → done in run 1.
# UC4: 1 adaptive config (~88 min) → done in run 1.
# UC2: 3 sequential adaptive configs → 1 per run → done after run 3.
# UC3: 3 sequential adaptive configs → 1 per run → done after run 3.
# UC2 and UC3 run in parallel → bottleneck is 3 runs.
# MAX_RESUBMIT=2 → 3 total runs → all configs complete.
# ============================================================

MAX_RESUBMIT=2
RESUBMIT_COUNT=${SLURM_RESUBMIT_COUNT:-0}

_CHILD_PIDS=()
_handle_timeout() {
    echo "Timeout signal at $(date)"
    for pid in "${_CHILD_PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null; done
    wait
    if [ $RESUBMIT_COUNT -lt $MAX_RESUBMIT ]; then
        SLURM_RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) sbatch --export=ALL "$0"
    fi
    exit 99
}
trap '_handle_timeout' SIGUSR1

echo "========================================================"
echo "Lassen Heavy — All UCs (UC1+UC2+UC3+UC4 in parallel)"
echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Resubmit: $RESUBMIT_COUNT/$MAX_RESUBMIT"
echo "Start: $(date)"
echo ""

module load PrgEnv-gnu cray-python
source .venv/bin/activate

OUT_DIR="output/use_cases/lassen_n1000_heavy"
mkdir -p "$OUT_DIR"

NUM_JOBS=300
DURATION=60
DELTA_T=10
NODES=1000

echo "Config: ${NUM_JOBS} jobs, ${DURATION}min, dt=${DELTA_T}s, ${NODES} nodes"
echo "UCs: 1+2+3+4 in parallel (resume logic skips completed variants)"
echo ""

run_uc() {
    local UC=$1
    local LOG="${OUT_DIR}/uc${UC}_dedicated.log"
    echo "[UC${UC}] lassen starting..."
    python -u src/run_use_cases.py \
        --system    lassen       \
        --nodes     "$NODES"     \
        --duration  "$DURATION"  \
        --delta-t   "$DELTA_T"   \
        --num-jobs  "$NUM_JOBS"  \
        --uc        "$UC"        \
        --output-dir "$OUT_DIR"  \
        > "$LOG" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[UC${UC}] lassen OK"
    else
        echo "[UC${UC}] lassen FAILED (rc=$rc)"
        tail -5 "$LOG" >&2
    fi
}

for UC in 1 2 3 4; do
    run_uc "$UC" &
    _CHILD_PIDS+=($!)
done
wait "${_CHILD_PIDS[@]}"
_CHILD_PIDS=()

echo ""
echo "=== UC completion status ==="
for UC in 1 2 3 4; do
    CSV=$(ls "${OUT_DIR}/uc${UC}_"*.csv 2>/dev/null | head -1)
    if [ -n "$CSV" ] && [ -f "$CSV" ]; then
        LINES=$(( $(wc -l < "$CSV") - 1 ))
        echo "  UC${UC}: $LINES variants done"
    else
        echo "  UC${UC}: MISSING"
    fi
done

echo ""
echo "Regenerating figures..."
python -u src/plot_energy_impact.py 2>&1
python -u src/plot_energy_overhead.py 2>&1

echo ""
echo "End: $(date)"

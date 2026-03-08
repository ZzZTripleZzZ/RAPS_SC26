#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J raps_energy_impact
#SBATCH -o output/use_cases/energy_impact-%j.out
#SBATCH -e output/use_cases/energy_impact-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --mail-type=END,FAIL
#SBATCH --signal=B:USR1@180

# ============================================================
# Heavy-load energy impact experiment
# 300 jobs (vs 200 in standard UC) → system overloaded →
# different routing/placement/scheduling leads to different
# jobs_completed and thus different energy per useful job.
# ============================================================

MAX_RESUBMIT=10
RESUBMIT_COUNT=${SLURM_RESUBMIT_COUNT:-0}

_CHILD_PIDS=()
_handle_timeout() {
    echo "Timeout signal at $(date)"
    for pid in "${_CHILD_PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null; done
    wait
    if [ $RESUBMIT_COUNT -lt $MAX_RESUBMIT ]; then
        # Set new count in env BEFORE sbatch so --export=ALL picks up the incremented value.
        # The syntax --export=ALL,VAR=N fails on Frontier when VAR already exists in env
        # (ALL overrides the explicit assignment). Prefix assignment fixes this.
        SLURM_RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) sbatch --export=ALL "$0"
    fi
    exit 99
}
trap '_handle_timeout' SIGUSR1

echo "========================================================"
echo "RAPS Energy Impact Experiment (heavy load)"
echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Resubmit: $RESUBMIT_COUNT/$MAX_RESUBMIT"
echo "Start: $(date)"
echo ""

module load PrgEnv-gnu cray-python
source .venv/bin/activate

mkdir -p output/use_cases

# ---- configuration ----
NUM_JOBS=300        # 50% more than standard 200 → real queueing
DURATION=60         # simulated minutes
DELTA_T=1           # time quantum
NODES=1000

echo "Config: ${NUM_JOBS} jobs, ${DURATION}min, dt=${DELTA_T}s, ${NODES} nodes"
echo "UCs: 1 (routing) + 2 (scheduling) + 3 (placement) + 4 (energy)"
echo ""

FORCE_FLAG=""
[ "$RESUBMIT_COUNT" -eq 0 ] && FORCE_FLAG="--force"

run_uc() {
    local SYSTEM=$1
    local UC=$2
    local OUT_DIR="output/use_cases/${SYSTEM}_n${NODES}_heavy"
    local LOG="${OUT_DIR}/uc${UC}.log"

    mkdir -p "$OUT_DIR"
    echo "[UC${UC}] ${SYSTEM} starting..."

    python -u src/run_use_cases.py \
        --system    "$SYSTEM"   \
        --nodes     "$NODES"    \
        --duration  "$DURATION" \
        --delta-t   "$DELTA_T"  \
        --num-jobs  "$NUM_JOBS" \
        --uc        "$UC"       \
        --output-dir "$OUT_DIR" \
        $FORCE_FLAG             \
        > "$LOG" 2>&1

    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[UC${UC}] ${SYSTEM} OK"
    else
        echo "[UC${UC}] ${SYSTEM} FAILED (rc=$rc)"
        tail -10 "$LOG" >&2
    fi
    return $rc
}

# Run all 3 systems × 4 UCs = 12 processes simultaneously.
# This avoids sequential phases so the 2-hour limit is set by the SLOWEST
# individual UC (lassen adaptive ~118 min) rather than the sum of phases.
echo "=== All systems UCs (12 parallel) ==="
for SYSTEM in frontier lassen bluewaters; do
    for UC in 1 2 3 4; do
        run_uc "$SYSTEM" "$UC" &
        _CHILD_PIDS+=($!)
    done
done

echo "Waiting for all 12 UC processes..."
failed_count=0
for pid in "${_CHILD_PIDS[@]}"; do
    wait $pid
    [ $? -ne 0 ] && failed_count=$((failed_count + 1))
done
_CHILD_PIDS=()
echo ""

# ---- Regenerate plots ----
echo "=== Generating energy impact figures ==="
python -u src/plot_energy_impact.py 2>&1

echo ""
echo "========================================================"
echo "Summary"
echo "========================================================"
for SYSTEM in frontier lassen bluewaters; do
    DIR="output/use_cases/${SYSTEM}_n${NODES}_heavy"
    echo "$SYSTEM:"
    for UC in 1 2 3 4; do
        CSV=$(ls "${DIR}/uc${UC}_"*.csv 2>/dev/null | head -1)
        if [ -n "$CSV" ] && [ -f "$CSV" ]; then
            LINES=$(( $(wc -l < "$CSV") - 1 ))
            echo "  UC${UC}: $LINES variants  ($CSV)"
        else
            echo "  UC${UC}: MISSING"
        fi
    done
done
echo ""
echo "End: $(date)"

#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J raps_lassen_uc4
#SBATCH -o output/use_cases/lassen_uc4_adaptive-%j.out
#SBATCH -e output/use_cases/lassen_uc4_adaptive-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --mail-type=END,FAIL
#SBATCH --signal=B:USR1@180

# ============================================================
# Lassen UC4 with adaptive routing (heavy load)
# Uses simulation-level checkpointing to span multiple SLURM
# submissions (adaptive routing on fat-tree is slow due to
# per-tick O(n^2) path recomputation).
# ============================================================

MAX_RESUBMIT=10
RESUBMIT_COUNT=${SLURM_RESUBMIT_COUNT:-0}

_CHILD_PID=""
_handle_timeout() {
    echo "Timeout signal at $(date)"
    # Send SIGTERM to the python process so it saves checkpoint
    if [ -n "$_CHILD_PID" ]; then
        kill -TERM "$_CHILD_PID" 2>/dev/null
        wait "$_CHILD_PID" 2>/dev/null
    fi
    echo "Child terminated, resubmitting..."
    if [ $RESUBMIT_COUNT -lt $MAX_RESUBMIT ]; then
        SLURM_RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) sbatch --export=ALL "$0"
    fi
    exit 99
}
trap '_handle_timeout' SIGUSR1

echo "========================================================"
echo "Lassen UC4 (heavy load, all routing incl. adaptive)"
echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Resubmit: $RESUBMIT_COUNT/$MAX_RESUBMIT"
echo "Start: $(date)"
echo ""

module load PrgEnv-gnu cray-python
source .venv/bin/activate

mkdir -p output/use_cases

NUM_JOBS=300
DURATION=60
DELTA_T=10
NODES=1000
SYSTEM=lassen
OUT_DIR="output/use_cases/${SYSTEM}_n${NODES}_heavy"

mkdir -p "$OUT_DIR"

echo "Config: ${NUM_JOBS} jobs, ${DURATION}min, dt=${DELTA_T}s, ${NODES} nodes"
echo ""

# No --force: preserve existing no_cong/minimal/ecmp variants, add only adaptive
python -u src/run_use_cases.py \
    --system    "$SYSTEM"   \
    --nodes     "$NODES"    \
    --duration  "$DURATION" \
    --delta-t   "$DELTA_T"  \
    --num-jobs  "$NUM_JOBS" \
    --uc        4           \
    --output-dir "$OUT_DIR" \
    > "${OUT_DIR}/uc4_adaptive.log" 2>&1 &

_CHILD_PID=$!
wait "$_CHILD_PID"
rc=$?

echo ""
echo "Exit code: $rc"
echo "End: $(date)"

# Show results
CSV="${OUT_DIR}/uc4_energy_results.csv"
if [ -f "$CSV" ]; then
    LINES=$(( $(wc -l < "$CSV") - 1 ))
    echo "UC4 results: $LINES variants in $CSV"
    cat "$CSV"
fi

# Check for remaining checkpoints (incomplete variants)
CKPTS=$(ls "${OUT_DIR}/checkpoints/"*.ckpt 2>/dev/null | wc -l)
if [ "$CKPTS" -gt 0 ]; then
    echo ""
    echo "WARNING: $CKPTS checkpoint(s) remain — variant not complete."
    echo "Resubmitting if allowed..."
    if [ $RESUBMIT_COUNT -lt $MAX_RESUBMIT ]; then
        SLURM_RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) sbatch --export=ALL "$0"
    else
        echo "Max resubmits reached."
    fi
    exit 99
fi

exit $rc

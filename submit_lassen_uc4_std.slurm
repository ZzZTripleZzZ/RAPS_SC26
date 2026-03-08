#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J lassen_uc4_std
#SBATCH -o output/use_cases/lassen_uc4_std-%j.out
#SBATCH -e output/use_cases/lassen_uc4_std-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --signal=B:USR1@180

# Lassen UC4 standard: add adaptive routing variant to existing results.
# No --force: preserves existing no_cong/minimal/ecmp, adds only adaptive.

MAX_RESUBMIT=10
RESUBMIT_COUNT=${SLURM_RESUBMIT_COUNT:-0}

_CHILD_PID=""
_handle_timeout() {
    echo "Timeout signal at $(date)"
    if [ -n "$_CHILD_PID" ]; then
        kill -TERM "$_CHILD_PID" 2>/dev/null
        wait "$_CHILD_PID" 2>/dev/null
    fi
    if [ $RESUBMIT_COUNT -lt $MAX_RESUBMIT ]; then
        SLURM_RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) sbatch --export=ALL "$0"
    fi
    exit 99
}
trap '_handle_timeout' SIGUSR1

echo "Lassen UC4 std (add adaptive) | Job $SLURM_JOB_ID | Resub $RESUBMIT_COUNT/$MAX_RESUBMIT | $(date)"
module load PrgEnv-gnu cray-python
source .venv/bin/activate

OUT_DIR="output/use_cases/lassen_n1000"
mkdir -p "$OUT_DIR"

python -u src/run_use_cases.py \
    --system    lassen  \
    --nodes     1000    \
    --duration  60      \
    --delta-t   30      \
    --num-jobs  200     \
    --uc        4       \
    --output-dir "$OUT_DIR" \
    > "${OUT_DIR}/uc4_adaptive.log" 2>&1 &

_CHILD_PID=$!
wait "$_CHILD_PID"
rc=$?

echo "Exit $rc | $(date)"

CSV="${OUT_DIR}/uc4_energy_results.csv"
if [ -f "$CSV" ]; then
    LINES=$(( $(wc -l < "$CSV") - 1 ))
    echo "UC4 results: $LINES variants in $CSV"
fi

CKPTS=$(ls "${OUT_DIR}/checkpoints/"*.ckpt 2>/dev/null | wc -l)
if [ "$CKPTS" -gt 0 ]; then
    echo "WARNING: $CKPTS checkpoint(s) remain — resubmitting..."
    if [ $RESUBMIT_COUNT -lt $MAX_RESUBMIT ]; then
        SLURM_RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) sbatch --export=ALL "$0"
    fi
    exit 99
fi

exit $rc

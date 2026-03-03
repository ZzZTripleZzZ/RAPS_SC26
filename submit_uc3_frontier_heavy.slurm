#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J raps_uc3_heavy
#SBATCH -o output/use_cases/uc3_heavy-%j.out
#SBATCH -e output/use_cases/uc3_heavy-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhangzifan@ornl.gov
#SBATCH --signal=B:USR1@180

# ============================================================
# Resume UC3 (placement) for frontier_n1000_heavy.
# contiguous is already done; this runs random + hybrid.
# Each variant ~5800s (~1.6h), so one per 2h job with resubmit.
# ============================================================

MAX_RESUBMIT=5
RESUBMIT_COUNT=${SLURM_RESUBMIT_COUNT:-0}

_handle_timeout() {
    echo "Timeout signal at $(date)"
    kill -TERM "$child_pid" 2>/dev/null
    wait "$child_pid" 2>/dev/null
    if [ $RESUBMIT_COUNT -lt $MAX_RESUBMIT ]; then
        SLURM_RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) sbatch --export=ALL "$0"
    fi
    exit 99
}
trap '_handle_timeout' SIGUSR1

echo "========================================================"
echo "RAPS UC3 Frontier Heavy (resume)"
echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Resubmit: $RESUBMIT_COUNT/$MAX_RESUBMIT"
echo "Start: $(date)"
echo ""

module load PrgEnv-gnu cray-python
source .venv/bin/activate

OUT_DIR="output/use_cases/frontier_n1000_heavy"
LOG="${OUT_DIR}/uc3.log"
mkdir -p "$OUT_DIR"

echo "Running UC3 for frontier (resume, no --force)..."

python -u src/run_use_cases.py \
    --system    frontier \
    --nodes     1000     \
    --duration  60       \
    --delta-t   1        \
    --num-jobs  300      \
    --uc        3        \
    --output-dir "$OUT_DIR" \
    >> "$LOG" 2>&1 &
child_pid=$!

wait "$child_pid"
rc=$?

if [ $rc -eq 0 ]; then
    echo "UC3 frontier OK"
    # Show final CSV
    csv="${OUT_DIR}/uc3_placement_results.csv"
    if [ -f "$csv" ]; then
        echo "Result CSV:"
        cat "$csv"
    fi
else
    echo "UC3 frontier FAILED or interrupted (rc=$rc)"
    tail -5 "$LOG" >&2
fi

echo ""
echo "End: $(date)"

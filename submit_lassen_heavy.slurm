#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J raps_lassen_heavy
#SBATCH -o output/use_cases/lassen_heavy-%j.out
#SBATCH -e output/use_cases/lassen_heavy-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhangzifan@ornl.gov
#SBATCH --signal=B:USR1@180

# ============================================================
# Lassen n=1000 heavy: all 4 UCs (300 jobs, dt=1, 60 min).
# Adaptive routing makes each tick expensive; MAX_RESUBMIT=10
# to handle multi-slot completion. Resume mode: skips labels
# already written to CSV.
# ============================================================

MAX_RESUBMIT=10
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
echo "RAPS Lassen Heavy (all UCs, resume)"
echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Resubmit: $RESUBMIT_COUNT/$MAX_RESUBMIT"
echo "Start: $(date)"
echo ""

module load PrgEnv-gnu cray-python
source .venv/bin/activate

OUT_DIR="output/use_cases/lassen_n1000_heavy"
LOG="${OUT_DIR}/lassen_heavy.log"
mkdir -p "$OUT_DIR"

echo "Running all UCs for lassen (resume, no --force)..."

python -u src/run_use_cases.py \
    --system    lassen \
    --nodes     1000   \
    --duration  60     \
    --delta-t   1      \
    --num-jobs  300    \
    --uc        1 2 3  \
    --output-dir "$OUT_DIR" \
    >> "$LOG" 2>&1 &
child_pid=$!

wait "$child_pid"
rc=$?

if [ $rc -eq 0 ]; then
    echo "Lassen heavy OK"
    echo "Results:"
    for csv in "${OUT_DIR}"/uc*.csv; do
        [ -f "$csv" ] || continue
        echo "  $csv:"
        python3 -c "import pandas as pd; df=pd.read_csv('$csv'); print(df[['label','jobs_completed','dilated_pct','avg_congestion']].to_string(index=False))" 2>/dev/null || cat "$csv"
    done
else
    echo "Lassen heavy FAILED or interrupted (rc=$rc)"
    tail -5 "$LOG" >&2
fi

echo ""
echo "End: $(date)"

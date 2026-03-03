#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J raps_uc4_hl
#SBATCH -o output/use_cases/uc4_hl-%j.out
#SBATCH -e output/use_cases/uc4_hl-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhangzifan@ornl.gov
#SBATCH --signal=B:USR1@180

# UC4 only for lassen_n1000_heavy (until_complete mode, new code)
# Runs after 4142610 (raps_lassen_heavy) finishes; cleans up its old-code UC4 CSV first.

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
echo "RAPS UC4 Lassen Heavy (until-complete mode)"
echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Resubmit: $RESUBMIT_COUNT/$MAX_RESUBMIT"
echo "Start: $(date)"
echo ""

module load PrgEnv-gnu cray-python
source .venv/bin/activate

OUT_DIR="output/use_cases/lassen_n1000_heavy"
LOG="${OUT_DIR}/uc4_hl.log"
mkdir -p "$OUT_DIR"

# Remove any stale UC4 CSV from the old fixed-time code run
if [ "$RESUBMIT_COUNT" -eq 0 ]; then
    echo "Removing stale UC4 CSV from old-code run (if any)..."
    rm -f "${OUT_DIR}/uc4_energy_results.csv"
fi

echo "Running UC4 for lassen_n1000_heavy (resume on resubmit)..."

python -u src/run_use_cases.py \
    --system    lassen  \
    --nodes     1000    \
    --duration  60      \
    --delta-t   1       \
    --num-jobs  300     \
    --uc        4       \
    --output-dir "$OUT_DIR" \
    >> "$LOG" 2>&1 &
child_pid=$!

wait "$child_pid"
rc=$?

if [ $rc -eq 0 ]; then
    echo "UC4 lassen heavy OK"
    csv="${OUT_DIR}/uc4_energy_results.csv"
    if [ -f "$csv" ]; then
        echo "Result CSV:"
        python3 -c "import pandas as pd; df=pd.read_csv('$csv'); print(df[['label','simulated_seconds','total_energy_joules','jobs_completed','dilated_pct']].to_string(index=False))" 2>/dev/null || cat "$csv"
    fi
else
    echo "UC4 lassen heavy FAILED or interrupted (rc=$rc)"
    tail -10 "$LOG" >&2
fi

echo ""
echo "End: $(date)"

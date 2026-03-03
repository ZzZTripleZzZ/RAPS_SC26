#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J raps_uc4_hf
#SBATCH -o output/use_cases/uc4_hf-%j.out
#SBATCH -e output/use_cases/uc4_hf-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhangzifan@ornl.gov
#SBATCH --signal=B:USR1@180

# UC4 only for frontier_n1000_heavy (until_complete mode, new code)
# Safe to run concurrently with 4142514 which only writes uc3_placement_results.csv

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
echo "RAPS UC4 Frontier Heavy (until-complete mode)"
echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Resubmit: $RESUBMIT_COUNT/$MAX_RESUBMIT"
echo "Start: $(date)"
echo ""

module load PrgEnv-gnu cray-python
source .venv/bin/activate

OUT_DIR="output/use_cases/frontier_n1000_heavy"
LOG="${OUT_DIR}/uc4_hf.log"
mkdir -p "$OUT_DIR"

echo "Running UC4 for frontier_n1000_heavy (resume, no --force)..."

python -u src/run_use_cases.py \
    --system    frontier \
    --nodes     1000     \
    --duration  60       \
    --delta-t   1        \
    --num-jobs  300      \
    --uc        4        \
    --output-dir "$OUT_DIR" \
    >> "$LOG" 2>&1 &
child_pid=$!

wait "$child_pid"
rc=$?

if [ $rc -eq 0 ]; then
    echo "UC4 frontier heavy OK"
    csv="${OUT_DIR}/uc4_energy_results.csv"
    if [ -f "$csv" ]; then
        echo "Result CSV:"
        python3 -c "import pandas as pd; df=pd.read_csv('$csv'); print(df[['label','simulated_seconds','total_energy_joules','jobs_completed','dilated_pct']].to_string(index=False))" 2>/dev/null || cat "$csv"
    fi
else
    echo "UC4 frontier heavy FAILED or interrupted (rc=$rc)"
    tail -10 "$LOG" >&2
fi

echo ""
echo "End: $(date)"

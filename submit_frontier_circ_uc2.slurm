#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J fr_uc2
#SBATCH -o output/use_cases/fr_uc2-%j.out
#SBATCH -e output/use_cases/fr_uc2-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --signal=B:USR1@180

MAX_RESUBMIT=5
RESUBMIT_COUNT=${SLURM_RESUBMIT_COUNT:-0}
_CHILD_PID=""
_handle_timeout() {
    echo "Timeout at $(date)"
    [ -n "$_CHILD_PID" ] && kill -TERM "$_CHILD_PID" 2>/dev/null && wait "$_CHILD_PID" 2>/dev/null
    if [ $RESUBMIT_COUNT -lt $MAX_RESUBMIT ]; then
        SLURM_RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) sbatch --export=ALL "$0"
    fi
    exit 99
}
trap '_handle_timeout' SIGUSR1

echo "Frontier UC2 std | Job $SLURM_JOB_ID | Resub $RESUBMIT_COUNT/$MAX_RESUBMIT | $(date)"
module load PrgEnv-gnu cray-python
source .venv/bin/activate
mkdir -p output/use_cases/frontier_n1000

FORCE_FLAG=""
[ "$RESUBMIT_COUNT" -eq 0 ] && FORCE_FLAG="--force"

python -u src/run_use_cases.py \
    --system frontier --nodes 1000 --duration 60 \
    --delta-t 30 --num-jobs 200 \
    --uc 2 --output-dir output/use_cases/frontier_n1000 \
    $FORCE_FLAG > output/use_cases/frontier_n1000/uc2.log 2>&1 &
_CHILD_PID=$!
wait "$_CHILD_PID"
rc=$?
echo "Exit $rc | $(date)"
CSV=$(ls output/use_cases/frontier_n1000/uc2_*.csv 2>/dev/null | head -1)
[ -f "$CSV" ] && echo "Result: $(($(wc -l < "$CSV") - 1)) variants"
exit $rc

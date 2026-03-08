#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J bw_b1k_dt0p1
#SBATCH -o output/frontier_scaling/bw_b1k_dt0p1-%j.out
#SBATCH -e output/frontier_scaling/bw_b1k_dt0p1-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --signal=B:USR1@180

MAX_RESUBMIT=10
RESUBMIT_COUNT=${SLURM_RESUBMIT_COUNT:-0}
_handle_timeout() {
    echo "Timeout at $(date)"
    kill -TERM "$child_pid" 2>/dev/null
    sleep 60
    kill -KILL "$child_pid" 2>/dev/null
    wait "$child_pid" 2>/dev/null
    if [ $RESUBMIT_COUNT -lt $MAX_RESUBMIT ]; then
        SLURM_RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) sbatch --export=ALL "$0"
    fi
    exit 99
}
trap '_handle_timeout' SIGUSR1

echo "Blue Waters bench n=1000 dt=0.1 | Job $SLURM_JOB_ID | Resub $RESUBMIT_COUNT/$MAX_RESUBMIT | $(date)"
module load PrgEnv-gnu cray-python
source .venv/bin/activate
mkdir -p output/frontier_scaling

NO_RESUME_FLAG=""
[ "$RESUBMIT_COUNT" -eq 0 ] && NO_RESUME_FLAG="--no-resume"

python -u src/run_frontier.py \
    --systems bluewaters \
    --nodes 1000 \
    --dt 0.1 \
    --repeats 3 --duration 12 --workers 1 \
    --output output/frontier_scaling \
    $NO_RESUME_FLAG &
child_pid=$!
wait "$child_pid"
rc=$?
echo "Exit $rc | $(date)"
exit $rc

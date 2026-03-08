#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J raps_frontier_circ_bench
#SBATCH -o output/frontier_scaling/frontier_circ_bench-%j.out
#SBATCH -e output/frontier_scaling/frontier_circ_bench-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --mail-type=END,FAIL
#SBATCH --signal=B:USR1@180

# ============================================================
# Frontier benchmark with circulant dragonfly topology
# ============================================================

MAX_RESUBMIT=10
RESUBMIT_COUNT=${SLURM_RESUBMIT_COUNT:-0}

_handle_timeout() {
    echo "Timeout signal at $(date)"
    kill -TERM "$child_pid" 2>/dev/null
    sleep 60
    kill -KILL "$child_pid" 2>/dev/null
    wait "$child_pid" 2>/dev/null
    if [ $RESUBMIT_COUNT -lt $MAX_RESUBMIT ]; then
        echo "Resubmitting ($((RESUBMIT_COUNT + 1))/$MAX_RESUBMIT)..."
        SLURM_RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) sbatch --export=ALL "$0"
    fi
    exit 99
}
trap '_handle_timeout' SIGUSR1

echo "=========================================="
echo "Frontier Benchmark — circulant dragonfly"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Resubmit: $RESUBMIT_COUNT/$MAX_RESUBMIT"
echo "Start: $(date)"
echo ""

module load PrgEnv-gnu cray-python
source .venv/bin/activate

mkdir -p output/frontier_scaling

echo "Config: frontier only, n=100/1000/10000, dt=0.1/1/10/60, 3 repeats"
echo ""

# Clear old Frontier results on first submission
if [ "$RESUBMIT_COUNT" -eq 0 ]; then
    echo "Clearing old Frontier benchmark results..."
    # Remove frontier entries from results.csv but keep lassen/bluewaters
    RFILE="output/frontier_scaling/results.csv"
    if [ -f "$RFILE" ]; then
        head -1 "$RFILE" > "${RFILE}.tmp"
        grep -v "^frontier," "$RFILE" >> "${RFILE}.tmp" 2>/dev/null || true
        mv "${RFILE}.tmp" "$RFILE"
        echo "Kept non-frontier entries in $RFILE"
    fi
fi

python -u src/run_frontier.py \
    --systems frontier \
    --nodes 100 1000 10000 \
    --dt 0.1 1 10 60 \
    --repeats 3 \
    --duration 12 \
    --workers 1 \
    --output output/frontier_scaling &
child_pid=$!

wait "$child_pid"
rc=$?

echo ""
echo "Exit code: $rc"
echo "End: $(date)"

# Count frontier results
RFILE="output/frontier_scaling/results.csv"
if [ -f "$RFILE" ]; then
    FRONTIER_LINES=$(grep -c "^frontier," "$RFILE" 2>/dev/null || echo 0)
    echo "Frontier results: $FRONTIER_LINES rows in $RFILE"
fi

exit $rc

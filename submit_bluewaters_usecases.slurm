#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J raps_bw_uc
#SBATCH -o output/use_cases/bw_uc-%j.out
#SBATCH -e output/use_cases/bw_uc-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --mail-type=END,FAIL
#SBATCH --signal=B:USR1@180

# =============================================================
# Blue Waters use case experiments (UC1-UC4, standard + heavy)
# torus3d + DOR_XYZ: cached routing, fast per-tick.
# Standard: 200 jobs, dt=30s, 60 min
# Heavy:    300 jobs, dt=1s,  60 min
# =============================================================

MAX_RESUBMIT=5
RESUBMIT_COUNT=${SLURM_RESUBMIT_COUNT:-0}

_CHILD_PIDS=()
_handle_timeout() {
    echo "Timeout signal at $(date)"
    for pid in "${_CHILD_PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null; done
    wait
    if [ $RESUBMIT_COUNT -lt $MAX_RESUBMIT ]; then
        SLURM_RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) sbatch --export=ALL "$0"
    fi
    exit 99
}
trap '_handle_timeout' SIGUSR1

echo "=================================================="
echo "Blue Waters Use Case Experiments"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Resubmit: $RESUBMIT_COUNT/$MAX_RESUBMIT"
echo "Start: $(date)"
echo ""

module load PrgEnv-gnu cray-python
source .venv/bin/activate

mkdir -p output/use_cases/bluewaters_n1000
mkdir -p output/use_cases/bluewaters_n1000_heavy

NODES=1000

run_uc_std() {
    local UC=$1
    local OUT_DIR="output/use_cases/bluewaters_n${NODES}"
    local LOG="${OUT_DIR}/uc${UC}.log"
    mkdir -p "$OUT_DIR"
    echo "[STD UC${UC}] bluewaters starting..."
    python -u src/run_use_cases.py \
        --system    bluewaters  \
        --nodes     "$NODES"    \
        --duration  60          \
        --delta-t   30          \
        --num-jobs  200         \
        --uc        "$UC"       \
        --output-dir "$OUT_DIR" \
        > "$LOG" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[STD UC${UC}] OK"
    else
        echo "[STD UC${UC}] FAILED (rc=$rc)"
        tail -10 "$LOG" >&2
    fi
    return $rc
}

run_uc_heavy() {
    local UC=$1
    local OUT_DIR="output/use_cases/bluewaters_n${NODES}_heavy"
    local LOG="${OUT_DIR}/uc${UC}.log"
    mkdir -p "$OUT_DIR"
    echo "[HVY UC${UC}] bluewaters starting..."
    python -u src/run_use_cases.py \
        --system    bluewaters  \
        --nodes     "$NODES"    \
        --duration  60          \
        --delta-t   1           \
        --num-jobs  300         \
        --uc        "$UC"       \
        --output-dir "$OUT_DIR" \
        > "$LOG" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[HVY UC${UC}] OK"
    else
        echo "[HVY UC${UC}] FAILED (rc=$rc)"
        tail -10 "$LOG" >&2
    fi
    return $rc
}

failed=0

echo "=== Standard UCs (parallel, dt=30s, 200 jobs) ==="
for UC in 1 2 3 4; do
    run_uc_std "$UC" &
    _CHILD_PIDS+=($!)
done
for pid in "${_CHILD_PIDS[@]}"; do
    wait $pid
    [ $? -ne 0 ] && failed=$((failed + 1))
done
_CHILD_PIDS=()
echo ""

echo "=== Heavy UCs (parallel, dt=1s, 300 jobs) ==="
for UC in 1 2 3 4; do
    run_uc_heavy "$UC" &
    _CHILD_PIDS+=($!)
done
for pid in "${_CHILD_PIDS[@]}"; do
    wait $pid
    [ $? -ne 0 ] && failed=$((failed + 1))
done
_CHILD_PIDS=()
echo ""

echo "=================================================="
echo "Summary"
echo "=================================================="
for TAG in "" "_heavy"; do
    DIR="output/use_cases/bluewaters_n${NODES}${TAG}"
    echo "bluewaters_n${NODES}${TAG}:"
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
echo "Failed: $failed"
echo "End: $(date)"

exit $failed

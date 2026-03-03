#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J raps_f_uc3h
#SBATCH -o output/use_cases/frontier_uc3_heavy-%j.out
#SBATCH -e output/use_cases/frontier_uc3_heavy-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --mail-type=END,FAIL
#SBATCH --signal=B:USR1@180

# ============================================================
# Dedicated job: Frontier UC3 heavy (all 3 placement configs)
# Account walltime limit: 120 min. dt=1s is too slow (~165 min/config).
# Solution: dt=5s → ugal ~21.5 min/contiguous + ~33 min/random + ~33 min/hybrid
# All 3 configs sequential ≈ 87.5 min → fits in a single 2h job.
# Existing uc3_placement_results.csv (dt=1s, contiguous only) is deleted
# and all 3 configs rerun at dt=5s for consistency.
# MAX_RESUBMIT=1 as safety net only.
# ============================================================

MAX_RESUBMIT=1
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

echo "========================================================"
echo "Frontier UC3 Heavy — Dedicated Job"
echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Resubmit: $RESUBMIT_COUNT/$MAX_RESUBMIT"
echo "Start: $(date)"
echo ""

module load PrgEnv-gnu cray-python
source .venv/bin/activate

OUT_DIR="output/use_cases/frontier_n1000_heavy"
mkdir -p "$OUT_DIR"

NUM_JOBS=300
DURATION=60
DELTA_T=5
NODES=1000

echo "Config: ${NUM_JOBS} jobs, ${DURATION}min, dt=${DELTA_T}s, ${NODES} nodes"
echo "Running UC3 (placement) — all 3 configs (contiguous+random+hybrid) with dt=${DELTA_T}s"
echo ""

python -u src/run_use_cases.py \
    --system    frontier     \
    --nodes     "$NODES"     \
    --duration  "$DURATION"  \
    --delta-t   "$DELTA_T"   \
    --num-jobs  "$NUM_JOBS"  \
    --uc        3            \
    --output-dir "$OUT_DIR"  \
    > "${OUT_DIR}/uc3_dedicated.log" 2>&1 &
_CHILD_PIDS+=($!)

wait "${_CHILD_PIDS[@]}"
rc=$?

echo ""
if [ $rc -eq 0 ]; then
    echo "UC3 frontier heavy COMPLETE"
    python -u src/plot_energy_impact.py 2>&1
else
    echo "UC3 frontier heavy not yet complete (rc=$rc) — will resubmit if budget allows"
fi

echo ""
echo "End: $(date)"

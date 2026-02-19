#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J raps_uc
#SBATCH -o output/use_cases/raps_uc-%j.out
#SBATCH -e output/use_cases/raps_uc-%j.err
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH --mail-user=zhangzifan@gatech.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --signal=B:USR1@180

# =============================================================================
# RAPS Use Case Experiments  (single system per job)
# =============================================================================
# Submit twice — once for lassen, once for frontier:
#   sbatch --export=ALL,RAPS_UC_SYSTEM=lassen   submit_use_cases.slurm
#   sbatch --export=ALL,RAPS_UC_SYSTEM=frontier  submit_use_cases.slurm
#
# Each job runs 4 use cases (UC1-UC4), 10k nodes, dt=1, 12h simulated.
# Auto-resubmits on timeout.
# =============================================================================

PROJECT_DIR="/lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26"
SYSTEM="${RAPS_UC_SYSTEM:-lassen}"
NODES=1000
DURATION=720          # 12 hours in minutes
DELTA_T=10
NUM_JOBS=50
OUT_DIR="$PROJECT_DIR/output/use_cases/${SYSTEM}_n${NODES}"

MAX_RESUBMIT=50
RESUBMIT_COUNT=${RAPS_UC_RESUBMIT_COUNT:-0}

# --- Auto-resubmit on exit ---
resubmit_if_needed() {
    # Check if ALL 4 use case CSVs exist with at least 2 rows (header + 2 variants)
    ALL_DONE=true
    for uc_csv in uc1_routing_results.csv uc2_scheduling_results.csv uc3_placement_results.csv uc4_energy_results.csv; do
        csv_path="$OUT_DIR/$uc_csv"
        if [ ! -f "$csv_path" ] || [ "$(wc -l < "$csv_path")" -lt 3 ]; then
            ALL_DONE=false
            break
        fi
    done

    if $ALL_DONE; then
        echo ""
        echo "ALL 4 USE CASES COMPLETE for $SYSTEM!"
        return 0
    fi

    if [ "$RESUBMIT_COUNT" -ge "$MAX_RESUBMIT" ]; then
        echo "WARNING: Reached max resubmit limit ($MAX_RESUBMIT)."
        return 0
    fi

    NEXT_COUNT=$((RESUBMIT_COUNT + 1))
    echo ""
    echo "Auto-resubmitting $SYSTEM (round $((NEXT_COUNT+1))/$MAX_RESUBMIT)..."
    NEXT_JOB=$(sbatch --export=ALL,RAPS_UC_SYSTEM=$SYSTEM,RAPS_UC_RESUBMIT_COUNT=$NEXT_COUNT \
        "$PROJECT_DIR/submit_use_cases.slurm" 2>&1)
    echo "Resubmitted: $NEXT_JOB"
}
trap resubmit_if_needed EXIT

# --- Catch SLURM pre-timeout signal ---
handle_signal() {
    echo ""
    echo ">>> Received pre-timeout signal at $(date). Wrapping up..."
    kill %1 2>/dev/null || true
}
trap handle_signal USR1

cd "$PROJECT_DIR"

echo "=============================================="
echo "RAPS Use Cases: $SYSTEM (Round $((RESUBMIT_COUNT+1))/$MAX_RESUBMIT)"
echo "Nodes: $NODES   dt: ${DELTA_T}s   Duration: ${DURATION}min"
echo "Started on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "=============================================="

# Setup environment
module load PrgEnv-gnu 2>/dev/null || true
source "$PROJECT_DIR/.venv/bin/activate"
export RAPS_DATA_DIR="$PROJECT_DIR/data/lassen"

mkdir -p "$OUT_DIR/figures"

python -u src/run_use_cases.py \
    --system "$SYSTEM" \
    --nodes "$NODES" \
    --duration "$DURATION" \
    --delta-t "$DELTA_T" \
    --num-jobs "$NUM_JOBS" \
    --output-dir "$OUT_DIR" &

wait $!
echo ""
echo "Python exited with code $? at $(date)"

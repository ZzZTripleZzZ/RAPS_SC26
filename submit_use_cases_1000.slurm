#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J raps_uc_1k
#SBATCH -o output/use_cases/raps_uc_1k-%j.out
#SBATCH -e output/use_cases/raps_uc_1k-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhangzifan@ornl.gov

# Auto-resubmit on timeout
#SBATCH --signal=B:USR1@180
MAX_RESUBMIT=50

# Trap timeout signal
trap 'echo "Job timed out, will resubmit..."; exit 99' SIGUSR1

echo "=========================================="
echo "RAPS Use Cases - node=1000"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "System: ${SYSTEM}"
echo "=========================================="
echo ""

# Load environment
module load PrgEnv-gnu
module load cray-python
source .venv/bin/activate

# Set data directory
export RAPS_DATA_DIR=/lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26/data

# Determine system from job name or environment
if [ -z "$SYSTEM" ]; then
    echo "ERROR: SYSTEM environment variable not set"
    exit 1
fi

echo "Running use cases for system: $SYSTEM with node=1000"
echo ""

# Run use cases with unbuffered output
python -u src/run_use_cases.py \
    --system $SYSTEM \
    --nodes 1000 \
    --duration 30 \
    --delta-t 1 \
    --num-jobs 30

exit_code=$?

echo ""
echo "=========================================="
echo "Job completed with exit code: $exit_code"
echo "End time: $(date)"
echo "=========================================="

# Check if all use cases completed
function resubmit_if_needed() {
    local OUT_DIR="output/use_cases/${SYSTEM}_n1000"
    
    # Check if all 4 use case CSVs exist and have results (header + 2 data rows)
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
        echo "ALL 4 USE CASES COMPLETE for $SYSTEM (node=1000)!"
        return 0
    fi
    
    # Check resubmit count
    if [ -z "$RESUBMIT_COUNT" ]; then
        RESUBMIT_COUNT=0
    fi
    
    if [ $RESUBMIT_COUNT -ge $MAX_RESUBMIT ]; then
        echo "Max resubmit limit ($MAX_RESUBMIT) reached, stopping."
        return 1
    fi
    
    echo "Use cases incomplete, resubmitting... (attempt $((RESUBMIT_COUNT + 1))/$MAX_RESUBMIT)"
    
    # Resubmit with incremented counter
    sbatch --export=ALL,SYSTEM=$SYSTEM,RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) $0
    return 0
}

# Handle exit codes
if [ $exit_code -eq 99 ]; then
    # Timeout signal
    resubmit_if_needed
elif [ $exit_code -eq 0 ]; then
    # Success - check if really done
    resubmit_if_needed
else
    echo "Job failed with error code $exit_code"
    exit $exit_code
fi

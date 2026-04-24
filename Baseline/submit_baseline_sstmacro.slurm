#!/bin/bash
#SBATCH --job-name=baseline_sstmacro
#SBATCH --account=gen053
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --signal=B:USR1@180

# SST-Macro Baseline Sweep — dragonfly 1056-node with offered_load
# Topology: 33g×8r×4h = 1056 hosts, h=32 (even ✓), matches BookSim2 k=4 and RAPS build_dragonfly(d=8,a=32,p=4)
# Sweeps injection rate ρ ∈ {0.05..0.80}
# Checkpoint/resubmit: completed ρ values are skipped on resubmit.
#
# Submit: sbatch Baseline/submit_baseline_sstmacro.slurm
# Output: Baseline/sst-macro/output/dragonfly_1000n/

MAX_RESUBMIT=5
RESUBMIT_COUNT=${SLURM_RESUBMIT_COUNT:-0}

_CHILD_PID=""
_handle_timeout() {
    echo "Timeout signal received at $(date), resubmitting ($RESUBMIT_COUNT/$MAX_RESUBMIT)..."
    [[ -n "$_CHILD_PID" ]] && kill -TERM "$_CHILD_PID" 2>/dev/null
    wait
    if [[ $RESUBMIT_COUNT -lt $MAX_RESUBMIT ]]; then
        SLURM_RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) sbatch --export=ALL "$0"
    fi
    exit 99
}
trap '_handle_timeout' SIGUSR1

set -euo pipefail
cd /lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26

echo "=== SST-Macro Baseline Sweep ==="
echo "Date: $(date) | Node: $(hostname) | Job: ${SLURM_JOB_ID:-local}"
echo "Resubmit: $RESUBMIT_COUNT/$MAX_RESUBMIT"

module load PrgEnv-gnu cray-python 2>/dev/null || true

# Set up SST-Macro library paths
NDT=/lustre/orion/proj-shared/gen053/ndt
export LD_LIBRARY_PATH="${NDT}/sst-macro/sstmac/main/.libs:${NDT}/sst-macro/sstmac/install/.libs:${NDT}/sst-macro/sst-dumpi/dumpi/libundumpi/.libs:${LD_LIBRARY_PATH:-}"
export SSTMAC_BIN="${NDT}/sst-run/sstmac"

if [[ ! -x "${SSTMAC_BIN}" ]]; then
    echo "ERROR: sstmac not found at ${SSTMAC_BIN}"
    exit 1
fi

echo "sstmac: ${SSTMAC_BIN}"

bash Baseline/sst-macro/run_sweep.sh 2>&1 &
_CHILD_PID=$!
wait $_CHILD_PID
_CHILD_PID=""

echo "=== Complete at $(date) ==="

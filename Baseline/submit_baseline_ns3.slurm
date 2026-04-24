#!/bin/bash
#SBATCH --job-name=baseline_ns3
#SBATCH --account=gen053
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --partition=batch
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --signal=B:USR1@180

# ns-3 Fat-tree Baseline Sweep — build from source + run 9 ρ values
# Topology: k=4 fat-tree, 16 servers, 12.5 GB/s links
# SIM_TIME_SEC=0.05 in fattree_sweep.cc keeps each rho under ~2 min.
# Checkpoint/resubmit: completed ρ values are skipped on resubmit.

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

RAPS_ROOT=/lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26
NS3_DIR="${RAPS_ROOT}/Baseline/ns3"
OUT_DIR="${NS3_DIR}/output/fattree"
SRC_DIR="${NS3_DIR}/ns3-src"
SCRATCH_SRC="${NS3_DIR}/scratch/fattree_sweep.cc"

mkdir -p "${OUT_DIR}"

echo "=== ns-3 Fat-tree Baseline Sweep ==="
echo "Date: $(date) | Node: $(hostname) | Job: ${SLURM_JOB_ID:-local}"
echo "Resubmit: $RESUBMIT_COUNT/$MAX_RESUBMIT"

# Load GCC 14 (required for C++17 which ns-3.42+ needs)
module load PrgEnv-gnu 2>/dev/null || true
GCC14=/opt/cray/pe/gcc-native/14/bin
export CC="${GCC14}/gcc"
export CXX="${GCC14}/g++"
export PATH="${GCC14}:${PATH}"
which g++ && g++ --version | head -1

# ---------------------------------------------------------------------------
# Step 1: Verify ns-3 source
# ---------------------------------------------------------------------------
if [[ ! -d "${SRC_DIR}" ]]; then
    echo "ERROR: ns-3 source not found at ${SRC_DIR}"
    echo "  Run on login node: git clone --depth=1 --branch ns-3.42 https://gitlab.com/nsnam/ns-3-dev.git ${SRC_DIR}"
    exit 1
fi

# Always copy scratch script to pick up any fixes
mkdir -p "${SRC_DIR}/scratch"
cp "${SCRATCH_SRC}" "${SRC_DIR}/scratch/fattree_sweep.cc"
echo "[ns-3] Copied fattree_sweep.cc to scratch/"

# ---------------------------------------------------------------------------
# Step 2: Configure and build (skip on resubmit if binary already present)
# ---------------------------------------------------------------------------
NS3_BIN="${SRC_DIR}/build/scratch/ns3.42-fattree_sweep-optimized"
cd "${SRC_DIR}"

if [[ -x "${NS3_BIN}" && $RESUBMIT_COUNT -gt 0 ]]; then
    echo "[ns-3] Binary present (resubmit), skipping rebuild."
    # Still rebuild to pick up fattree_sweep.cc changes
    python3 ns3 build fattree_sweep -- -j${SLURM_CPUS_PER_TASK:-8} 2>&1 | tail -5
else
    rm -f "${NS3_BIN}"
    rm -rf cmake-cache

    echo "[ns-3] Configuring (optimized build, minimal modules)..."
    python3 ns3 configure \
        --build-profile=optimized \
        --disable-examples \
        --disable-tests \
        --enable-modules="core,network,internet,point-to-point,applications,flow-monitor,traffic-control" \
        2>&1 | tail -5

    echo "[ns-3] Building fattree_sweep (${SLURM_CPUS_PER_TASK:-8} cores)..."
    python3 ns3 build fattree_sweep -- -j${SLURM_CPUS_PER_TASK:-8} 2>&1 | tail -10
    echo "[ns-3] Build complete."
fi

# ---------------------------------------------------------------------------
# Step 3: Calibrate zero-load latency at rho=0.001
# ---------------------------------------------------------------------------
RHO_VALUES=(0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80)
CAL_JSON="${OUT_DIR}/summary_calibration.json"
ZERO_LOAD_NS=0

if [[ -f "${CAL_JSON}" ]]; then
    ZERO_LOAD_NS=$(python3 -c "import json; d=json.load(open('${CAL_JSON}')); print(d.get('avg_latency_ns', 0))" 2>/dev/null || echo 0)
    echo "[ns-3] Using cached zero-load: ${ZERO_LOAD_NS} ns"
else
    echo -n "[ns-3] Calibrating zero-load latency (rho=0.001)... "
    timeout 300 python3 ns3 run \
        "fattree_sweep --rho=0.001 --out=${CAL_JSON}" \
        > "${OUT_DIR}/rho_calibration.log" 2>&1 &
    _CHILD_PID=$!
    wait $_CHILD_PID
    _CHILD_PID=""
    if [[ -f "${CAL_JSON}" ]]; then
        ZERO_LOAD_NS=$(python3 -c "import json; d=json.load(open('${CAL_JSON}')); print(d.get('avg_latency_ns', 0))" 2>/dev/null || echo 0)
        echo "done (${ZERO_LOAD_NS} ns)"
    else
        echo "WARN: calibration failed, using analytical zero-load"
    fi
fi

# ---------------------------------------------------------------------------
# Step 4: Run sweep (skip already-completed ρ values)
# ---------------------------------------------------------------------------
echo ""
echo "[ns-3] Running fat-tree sweep (k=4, 16 hosts, 12.5 GB/s, zero_load=${ZERO_LOAD_NS} ns)..."
for rho in "${RHO_VALUES[@]}"; do
    rho_str="${rho/./p}"
    out_json="${OUT_DIR}/summary_${rho_str}.json"

    # Skip if already completed (flow_count > 0 means the sim produced data)
    if [[ -f "${out_json}" ]] && python3 -c "
import json, sys
d = json.load(open('${out_json}'))
sys.exit(0 if d.get('flow_count', 0) > 0 else 1)
" 2>/dev/null; then
        echo "  ρ=${rho}: already done (skip)"
        continue
    fi

    echo -n "  ρ=${rho}: "
    timeout 600 python3 ns3 run \
        "fattree_sweep --rho=${rho} --zero-load-ns=${ZERO_LOAD_NS} --out=${out_json}" \
        2>&1 | grep -E "rho=|[Ee]rror" &
    _CHILD_PID=$!
    wait $_CHILD_PID
    _CHILD_PID=""

    if [[ -f "${out_json}" ]]; then
        echo "  → ${out_json}"
    else
        echo "  TIMEOUT/ERROR"
    fi
done

echo ""
echo "=== Complete at $(date) ==="
echo "Results in ${OUT_DIR}/"
ls -la "${OUT_DIR}/"

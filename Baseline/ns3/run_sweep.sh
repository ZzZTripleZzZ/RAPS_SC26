#!/bin/bash
# ns-3 Fat-tree Baseline Sweep
# Fat-tree k=4, 16 servers, 12.5 GB/s, ECMP routing
# NOTE: Dragonfly topology is not natively supported in ns-3; fat-tree only.
#
# Usage:
#   cd /lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26/Baseline/ns3
#   bash run_sweep.sh [--skip-build]
#
# Build time: ~15-30 min (ns-3 from source) or < 1 min (pip install ns3)
# Run time: ~5-20 min per ρ
#
# Outputs: output/fattree/summary_rho_X.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS3_DIR="${SCRIPT_DIR}/ns3-src"
OUT_DIR="${SCRIPT_DIR}/output/fattree"
SCRATCH_SRC="${SCRIPT_DIR}/scratch/fattree_sweep.cc"

RHO_VALUES=(0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80)

mkdir -p "${OUT_DIR}"

SKIP_BUILD=0
if [[ "${1:-}" == "--skip-build" ]]; then SKIP_BUILD=1; fi

# ---------------------------------------------------------------------------
# Step 1: Get ns-3
# ---------------------------------------------------------------------------
if [[ $SKIP_BUILD -eq 0 ]]; then
    echo "[ns-3] Setting up ns-3..."

    # Option A: pip install (fastest, if available for Python 3.11+)
    if python3 -c "import ns" 2>/dev/null; then
        echo "  ns-3 Python bindings already installed"
    else
        echo "  Attempting pip install ns3..."
        python3 -m pip install ns3 2>/dev/null && echo "  pip install succeeded" || {
            echo "  pip install failed. Trying source build..."

            # Option B: source build
            if [[ ! -d "${NS3_DIR}" ]]; then
                git clone https://gitlab.com/nsnam/ns-3-dev.git "${NS3_DIR}"
                cd "${NS3_DIR}"
                git checkout ns-3.42
            fi

            cd "${NS3_DIR}"
            # Copy sweep script to scratch
            cp "${SCRATCH_SRC}" "${NS3_DIR}/scratch/fattree_sweep.cc"
            python3 ns3 configure --build-profile=optimized
            python3 ns3 build fattree_sweep
            echo "  ns-3 build complete"
        }
    fi
fi

# ---------------------------------------------------------------------------
# Step 2: Determine run command
# ---------------------------------------------------------------------------
NS3_RUN=""
if [[ -d "${NS3_DIR}" && -f "${NS3_DIR}/ns3" ]]; then
    NS3_RUN="python3 ${NS3_DIR}/ns3 run"
elif command -v ns3 &>/dev/null; then
    NS3_RUN="ns3 run"
else
    echo "ERROR: ns3 not found. Install with 'pip install ns3' or build from source."
    echo "Generating placeholder outputs for all rho values..."
    for rho in "${RHO_VALUES[@]}"; do
        rho_str="${rho/./p}"
        python3 -c "
import json
data = {
    'simulator': 'ns3',
    'topology': 'fattree',
    'rho_target': ${rho},
    'status': 'not_installed',
    'note': 'ns-3 not available on this system. Build from source or pip install ns3.',
}
print(json.dumps(data, indent=2))
" > "${OUT_DIR}/summary_${rho_str}.json"
    done
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 3: Calibrate zero-load latency at rho=0.001
# ---------------------------------------------------------------------------
CAL_JSON="${OUT_DIR}/summary_calibration.json"
echo -n "[ns-3] Calibrating zero-load latency (rho=0.001)... "
timeout 300 ${NS3_RUN} "fattree_sweep --rho=0.001 --out=${CAL_JSON}" \
    > "${OUT_DIR}/rho_calibration.log" 2>&1 || {
    echo "WARN: calibration failed, using analytical zero-load"
    ZERO_LOAD_NS=0
}
if [[ -f "${CAL_JSON}" ]]; then
    ZERO_LOAD_NS=$(python3 -c "import json; d=json.load(open('${CAL_JSON}')); print(d.get('avg_latency_ns', 0))" 2>/dev/null || echo 0)
fi
echo "done (zero_load=${ZERO_LOAD_NS} ns)"

# ---------------------------------------------------------------------------
# Step 4: Run sweep
# ---------------------------------------------------------------------------
echo "[ns-3] Fat-tree sweep (k=4, 12.5 GB/s, ECMP, zero_load=${ZERO_LOAD_NS} ns)..."
for rho in "${RHO_VALUES[@]}"; do
    rho_str="${rho/./p}"
    out_json="${OUT_DIR}/summary_${rho_str}.json"

    echo -n "  ρ=${rho}: running... "
    timeout 600 ${NS3_RUN} "fattree_sweep \
        --rho=${rho} \
        --zero-load-ns=${ZERO_LOAD_NS} \
        --out=${out_json}" \
        2>&1 | tee "${OUT_DIR}/rho_${rho_str}.log" || {
        echo "TIMEOUT/ERROR"
        continue
    }
    echo "done → ${out_json}"
done

echo ""
echo "[ns-3] Sweep complete. Results in ${OUT_DIR}/"
echo "Note: Dragonfly topology not natively available in ns-3."
echo "  Only fat-tree (k=4) results are in this directory."

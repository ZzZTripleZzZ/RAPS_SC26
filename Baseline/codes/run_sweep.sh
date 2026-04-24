#!/bin/bash
# CODES/ROSS Dragonfly Baseline Sweep
# Builds ROSS + CODES from source and runs dragonfly validation workload.
#
# Topology: 9g×4r×2h dragonfly, 72 hosts, 25 GB/s, minimal routing
# Traffic: uniform random all-to-all at ρ ∈ {0.05..0.80}
#
# Prerequisites:
#   module load PrgEnv-gnu cmake/3.23 cray-mpich
#   Internet access (git clone) or pre-downloaded tarballs
#
# Usage:
#   cd /lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26/Baseline/codes
#   bash run_sweep.sh [--skip-build]
#
# Build time: ~2-4h (ROSS + CODES + Cray build fixes)
# Run time: ~5-10 min per ρ value × 9 = ~90 min total
#
# Outputs:
#   build/          ROSS + CODES binaries
#   output/dragonfly/rho_X.log   LP-IO statistics
#   output/dragonfly/summary_X.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
OUT_DIR="${SCRIPT_DIR}/output/dragonfly"

RHO_VALUES=(0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80)
LINK_BW_GBPS=25.0
ZERO_LOAD_NS=350

mkdir -p "${OUT_DIR}" "${BUILD_DIR}"

SKIP_BUILD=0
if [[ "${1:-}" == "--skip-build" ]]; then SKIP_BUILD=1; fi

# ---------------------------------------------------------------------------
# Step 1: Build ROSS
# ---------------------------------------------------------------------------
if [[ $SKIP_BUILD -eq 0 ]]; then
    echo "[CODES] Building ROSS..."
    ROSS_DIR="${BUILD_DIR}/ROSS"
    ROSS_INSTALL="${BUILD_DIR}/ross-install"

    if [[ ! -d "${ROSS_DIR}/.git" ]]; then
        git clone https://github.com/ROSS-org/ROSS.git "${ROSS_DIR}"
    fi

    mkdir -p "${ROSS_DIR}/build"
    cd "${ROSS_DIR}/build"

    # Cray-specific: use CC/CXX wrappers for MPI
    cmake .. \
        -DCMAKE_C_COMPILER=cc \
        -DCMAKE_CXX_COMPILER=CC \
        -DCMAKE_INSTALL_PREFIX="${ROSS_INSTALL}" \
        -DROSS_BUILD_MODELS=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        2>&1 | tee cmake_ross.log

    make -j8 install 2>&1 | tee make_ross.log
    echo "[CODES] ROSS installed to ${ROSS_INSTALL}"

    # ---------------------------------------------------------------------------
    # Step 2: Build CODES
    # ---------------------------------------------------------------------------
    echo "[CODES] Building CODES..."
    CODES_DIR="${BUILD_DIR}/codes"
    CODES_INSTALL="${BUILD_DIR}/codes-install"

    if [[ ! -d "${CODES_DIR}/.git" ]]; then
        # Use dragonfly-validation branch if available
        git clone https://github.com/codes-org/codes.git "${CODES_DIR}"
        cd "${CODES_DIR}"
        git checkout dragonfly-validation 2>/dev/null || true
    fi

    mkdir -p "${CODES_DIR}/build"
    cd "${CODES_DIR}/build"

    cmake .. \
        -DCMAKE_C_COMPILER=cc \
        -DCMAKE_CXX_COMPILER=CC \
        -DCMAKE_INSTALL_PREFIX="${CODES_INSTALL}" \
        -DROSS_DIR="${ROSS_INSTALL}" \
        -DCMAKE_BUILD_TYPE=Release \
        2>&1 | tee cmake_codes.log

    make -j8 install 2>&1 | tee make_codes.log
    echo "[CODES] CODES installed to ${CODES_INSTALL}"
    cd "${SCRIPT_DIR}"
fi

# ---------------------------------------------------------------------------
# Step 3: Generate dragonfly config for CODES
# ---------------------------------------------------------------------------
# CODES dragonfly model config format (lp-io workload):
# groups, group_links, num_router_rows, num_router_cols, num_groups_per_rep,
# num_global_channels, num_terminals, link_bandwidth
generate_codes_config() {
    local rho="$1"
    local cfg_path="$2"
    local inj_bw
    inj_bw=$(python3 -c "print(f'{${rho} * ${LINK_BW_GBPS}:.4f}')")

    cat > "$cfg_path" << EOF
LPGROUPS
{
  MODELNET_GRP
  {
    repetitions="72";  # total hosts
    server="1";
    dragonfly="1";
  }
}
PARAMS
{
  # Dragonfly parameters matching reference topology:
  # 9 groups × 4 routers/group × 2 hosts/router
  num_groups="9";
  num_routers="4";
  num_ports="2";
  global_bandwidth="25.0";  # GB/s
  local_bandwidth="25.0";
  cn_bandwidth="25.0";
  global_latency="100";     # ns
  local_latency="100";
  cn_latency="50";

  # Traffic: uniform random (all-to-all equivalent)
  traffic_pattern="uniform_random";
  injection_bandwidth="${inj_bw}";  # = ρ × link_bw

  # Routing: minimal
  routing="minimal";

  # Simulation
  num_messages="1000";
  message_size="4096";

  # Statistics output
  lp_io_dir="lp_io_${rho/./p}";
  lp_io_use_suffix="0";
}
EOF
}

# ---------------------------------------------------------------------------
# Step 4: Run sweep
# ---------------------------------------------------------------------------
CODES_SIM="${BUILD_DIR}/codes-install/bin/model-net-mpi-replay"
if [[ ! -x "${CODES_SIM:-}" ]]; then
    # Try alternate binary names
    CODES_SIM=$(find "${BUILD_DIR}" -name "dragonfly*" -executable 2>/dev/null | head -1 || true)
fi

if [[ -z "${CODES_SIM:-}" || ! -x "${CODES_SIM:-}" ]]; then
    echo "WARNING: CODES binary not found. Generating configs only."
    echo "  After build, update CODES_SIM path and re-run."
    for rho in "${RHO_VALUES[@]}"; do
        rho_str="${rho/./p}"
        generate_codes_config "$rho" "${SCRIPT_DIR}/configs/dragonfly_rho_${rho_str}.conf"
        echo "  Config written: configs/dragonfly_rho_${rho_str}.conf"
    done
    exit 0
fi

echo "[CODES] Dragonfly sweep (9g×4r×2h, minimal routing)..."
for rho in "${RHO_VALUES[@]}"; do
    rho_str="${rho/./p}"
    cfg="${SCRIPT_DIR}/configs/dragonfly_rho_${rho_str}.conf"
    log_file="${OUT_DIR}/rho_${rho_str}.log"
    json_file="${OUT_DIR}/summary_${rho_str}.json"

    mkdir -p "${SCRIPT_DIR}/configs"
    generate_codes_config "$rho" "$cfg"

    echo -n "  ρ=${rho}: running... "
    # CODES runs with MPI; use 4 MPI ranks for 72 hosts
    timeout 600 mpirun -n 4 "${CODES_SIM}" \
        --sync=3 \
        --conf="${cfg}" \
        > "$log_file" 2>&1 || {
        echo "TIMEOUT/ERROR (see ${log_file})"
        continue
    }
    echo "done"

    # Parse LP-IO statistics
    # CODES outputs per-link stats in lp_io directory
    lp_io_dir="lp_io_${rho_str}"
    link_util=$(find "${lp_io_dir}" -name "*.stat" 2>/dev/null \
        | xargs grep -h "link_util" 2>/dev/null \
        | awk '{sum += $2; n++} END {if (n>0) print sum/n; else print "null"}' || echo "null")
    stall_cycles=$(find "${lp_io_dir}" -name "*.stat" 2>/dev/null \
        | xargs grep -h "stall_cycles" 2>/dev/null \
        | awk '{sum += $2; n++} END {if (n>0) print sum/n; else print "null"}' || echo "null")
    total_pkts=$(find "${lp_io_dir}" -name "*.stat" 2>/dev/null \
        | xargs grep -h "total_packets" 2>/dev/null \
        | awk '{sum += $2} END {print sum+0}' || echo "null")

    python3 -c "
import json
data = {
    'simulator': 'codes',
    'topology': 'dragonfly',
    'rho_target': ${rho},
    'mean_link_utilization': ${link_util},
    'stall_cycles': ${stall_cycles},
    'total_packets': ${total_pkts},
    'zero_load_latency_ns': ${ZERO_LOAD_NS},
    'log_file': '${log_file}',
}
print(json.dumps(data, indent=2))
" > "$json_file"
    echo "  → $json_file"
done

echo ""
echo "[CODES] Sweep complete. Results in ${OUT_DIR}/"

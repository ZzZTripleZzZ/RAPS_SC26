#!/bin/bash
# BookSim2 Baseline Sweep (updated for multi-size topologies)
# Supports: dragonfly (72n, 1000n, 10000n) and fat-tree (k=4, 16n)
#
# Usage:
#   cd /lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26/Baseline/booksim2
#   bash run_sweep.sh [--skip-build] [--nodes 1000] [--topo dragonfly|fattree|all]
#
# Examples:
#   bash run_sweep.sh --nodes 1000               # 1000-node dragonfly only
#   bash run_sweep.sh --nodes all --topo all     # everything (slow)
#   bash run_sweep.sh --skip-build --nodes 1000  # skip build, run sweep only
#
# Outputs:
#   output/dragonfly/rho_X.{log,json}
#   output/dragonfly_1000n/rho_X.{log,json}
#   output/fattree/rho_X.{log,json}

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BOOKSIM="${BUILD_DIR}/src/booksim"
CONFIGS_DIR="${SCRIPT_DIR}/configs"

RHO_VALUES=(0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80)

# Default: run 1000-node dragonfly only (most useful for comparison)
SKIP_BUILD=0
NODES_FILTER="1000"
TOPO_FILTER="dragonfly"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build)   SKIP_BUILD=1 ;;
        --nodes)        NODES_FILTER="$2"; shift ;;
        --topo)         TOPO_FILTER="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# Topology suite definition
# ---------------------------------------------------------------------------
# Array of "topo_name:config_template:out_dir" (space-separated in each element)
declare -A TOPO_CONFIGS=(
    ["dragonfly"]="dragonfly.cfg.template:output/dragonfly"
    ["dragonfly_1000n"]="dragonfly_1000n.cfg.template:output/dragonfly_1000n"
    ["dragonfly_10000n"]="dragonfly_10000n.cfg.template:output/dragonfly_10000n"
    ["fattree"]="fattree.cfg.template:output/fattree"
    ["torus3d"]="torus3d.cfg.template:output/torus3d"
)

# ---------------------------------------------------------------------------
# Step 1: Build BookSim2
# ---------------------------------------------------------------------------
if [[ $SKIP_BUILD -eq 0 ]]; then
    echo "[BookSim2] Cloning and building..."
    if [[ ! -d "${BUILD_DIR}/.git" ]]; then
        git clone https://github.com/booksim/booksim2.git "${BUILD_DIR}"
    else
        echo "  Source already cloned in ${BUILD_DIR}"
    fi

    cd "${BUILD_DIR}/src"
    make -j8 2>&1 | tail -5
    echo "[BookSim2] Build complete: ${BOOKSIM}"
    cd "${SCRIPT_DIR}"
fi

if [[ ! -x "${BOOKSIM}" ]]; then
    echo "ERROR: BookSim2 binary not found at ${BOOKSIM}"
    echo "  Re-run without --skip-build to build."
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 2: Parse BookSim2 stdout → JSON
# ---------------------------------------------------------------------------
parse_booksim_log() {
    local log_file="$1"
    local topo="$2"
    local rho="$3"
    local out_json="$4"
    local zero_load_cycles="$5"   # BookSim2 zero-load latency in cycles
    local zero_load_ns_real="$6"  # Real zero-load latency in nanoseconds

    # Extract metrics
    # Extract metrics — BookSim2 final summary has format "X.XXX (N samples)" on the last match
    local avg_latency injected_rate accepted_rate throughput
    avg_latency=$(python3 -c "
import re, sys
lines = open('${log_file}').readlines()
lats = [float(re.search(r'= *([0-9.]+)', l).group(1)) for l in lines
        if 'Packet latency average' in l and re.search(r'= *([0-9.]+)', l)]
print(f'{lats[-1]:.4f}' if lats else 'null')
" 2>/dev/null || echo "null")
    injected_rate=$(python3 -c "
import re, sys
lines = open('${log_file}').readlines()
vals = [float(re.search(r'= *([0-9.e+-]+)', l).group(1)) for l in lines
        if 'Injected packet rate average' in l and re.search(r'= *([0-9.e+-]+)', l)]
print(f'{vals[-1]:.6f}' if vals else 'null')
" 2>/dev/null || echo "null")
    accepted_rate=$(python3 -c "
import re, sys
lines = open('${log_file}').readlines()
vals = [float(re.search(r'= *([0-9.e+-]+)', l).group(1)) for l in lines
        if 'Accepted packet rate average' in l and re.search(r'= *([0-9.e+-]+)', l)]
print(f'{vals[-1]:.6f}' if vals else 'null')
" 2>/dev/null || echo "null")
    throughput="null"

    # Channel utilization (print_channel_stats not supported in this BookSim2 build)
    local mean_util max_util n_chans
    mean_util="null"
    max_util="null"
    n_chans=0

    # Stall ratio and slowdown from cycle latency; convert avg_latency to ns via real zero-load
    local stall_ratio slowdown avg_latency_ns
    if [[ "$avg_latency" != "null" && "$avg_latency" != "" && "$avg_latency" != "0" ]]; then
        # Dimensionless slowdown = cycles / zero_load_cycles
        # avg_latency_ns = slowdown × zero_load_ns_real (correct unit conversion)
        stall_ratio=$(python3 -c "
lat=float('${avg_latency}'); z=${zero_load_cycles}
print(max(0.0, (lat - z) / z) if z > 0 else 0.0)
" 2>/dev/null || echo "null")
        slowdown=$(python3 -c "
lat=float('${avg_latency}'); z=${zero_load_cycles}
print(lat / z if z > 0 else 1.0)
" 2>/dev/null || echo "null")
        avg_latency_ns=$(python3 -c "
sd=float('${slowdown:-1}'); z_ns=${zero_load_ns_real}
print(sd * z_ns)
" 2>/dev/null || echo "null")
    else
        avg_latency_ns="null"
        stall_ratio="null"
        slowdown="null"
    fi

    python3 - << PYEOF > "$out_json"
import json
data = {
    "simulator": "booksim2",
    "topology": "${topo}",
    "rho_target": ${rho},
    "avg_latency_cycles": None if "${avg_latency}" == "null" else float("${avg_latency:-0}"),
    "avg_latency_ns": None if "${avg_latency_ns}" == "null" else float("${avg_latency_ns:-0}"),
    "injected_rate": None if "${injected_rate}" == "null" else float("${injected_rate:-0}"),
    "accepted_rate": None if "${accepted_rate}" == "null" else float("${accepted_rate:-0}"),
    "throughput": None if "${throughput}" == "null" else float("${throughput:-0}"),
    "mean_utilization": None if "${mean_util}" == "null" else None,
    "max_utilization": None if "${max_util}" == "null" else None,
    "n_channels": ${n_chans},
    "stall_ratio": None if "${stall_ratio}" == "null" else float("${stall_ratio:-0}"),
    "slowdown": None if "${slowdown}" == "null" else float("${slowdown:-0}"),
    "zero_load_latency_cycles": ${zero_load_cycles},
    "zero_load_latency_ns": ${zero_load_ns_real},
    "log_file": "${log_file}",
    "status": "ok",
}
print(json.dumps({k: v for k, v in data.items()}, indent=2))
PYEOF
}

# ---------------------------------------------------------------------------
# Step 3: Run sweeps
# ---------------------------------------------------------------------------
run_topology_sweep() {
    local topo_key="$1"       # e.g., "dragonfly_1000n"
    local cfg_tmpl="$2"       # e.g., "dragonfly_1000n.cfg.template"
    local out_dir="${SCRIPT_DIR}/$3"
    local zero_load_ns="$4"       # BookSim2 zero-load latency in cycles
    local zero_load_ns_real="$5"  # Real zero-load latency in nanoseconds

    # Filter by --nodes / --topo
    if [[ "$NODES_FILTER" != "all" ]]; then
        if [[ "$topo_key" == *"${NODES_FILTER}n"* ]]; then :
        elif [[ "$NODES_FILTER" == "72" && "$topo_key" == "dragonfly" ]]; then :
        elif [[ "$NODES_FILTER" == "16" && "$topo_key" == "fattree" ]]; then :
        elif [[ "$topo_key" == "torus3d" && ( "$NODES_FILTER" == "1000" || "$NODES_FILTER" == "1024" ) ]]; then :
        else return 0; fi
    fi
    if [[ "$TOPO_FILTER" != "all" ]]; then
        if [[ "$topo_key" != *"${TOPO_FILTER}"* ]]; then return 0; fi
    fi

    local cfg_path="${CONFIGS_DIR}/${cfg_tmpl}"
    if [[ ! -f "$cfg_path" ]]; then
        echo "  SKIP: config not found: ${cfg_path}"
        return 0
    fi

    mkdir -p "${out_dir}"
    echo ""
    echo "[BookSim2] ${topo_key} sweep..."

    # ---------------------------------------------------------------------------
    # Step 3a: Calibrate zero-load latency by running at ρ=0.001 first
    # This gives the actual BookSim2 cycle latency at near-zero load, which is
    # used to normalize all subsequent measurements into dimensionless slowdown.
    # ---------------------------------------------------------------------------
    local zero_load_cycles_measured
    local cal_cfg_tmp="/tmp/booksim_${topo_key}_cal.cfg"
    local cal_log="${out_dir}/rho_zeroload.log"
    sed "s/INJECTION_RATE/0.001/g" "${cfg_path}" > "$cal_cfg_tmp"
    echo -n "  calibrating zero-load (ρ=0.001)... "
    timeout 120 "${BOOKSIM}" "$cal_cfg_tmp" > "$cal_log" 2>&1 || true
    rm -f "$cal_cfg_tmp"
    if grep -q "Total run time" "$cal_log" 2>/dev/null; then
        zero_load_cycles_measured=$(python3 -c "
import re
lines = open('${cal_log}').readlines()
lats = []
for l in lines:
    if 'Packet latency average' in l:
        m = re.search(r'= *([0-9.]+)', l)
        if m: lats.append(float(m.group(1)))
print(f'{min(lats):.3f}' if lats else '${zero_load_ns}')
" 2>/dev/null || echo "${zero_load_ns}")
        echo "zero_load=${zero_load_cycles_measured} cycles"
    else
        # Fallback: use ρ=0.05 from existing data if available, else use passed estimate
        zero_load_cycles_measured="${zero_load_ns}"
        echo "calibration failed, using estimate=${zero_load_cycles_measured} cycles"
    fi

    for rho in "${RHO_VALUES[@]}"; do
        local rho_str="${rho/./p}"
        local cfg_tmp="/tmp/booksim_${topo_key}_${rho_str}.cfg"
        local log_file="${out_dir}/rho_${rho_str}.log"
        local json_file="${out_dir}/summary_${rho_str}.json"

        # Skip if already completed (checkpoint/resume support)
        if [[ -f "$json_file" ]] && python3 -c "
import json, sys
d = json.load(open('${json_file}'))
sys.exit(0 if d.get('status','') == 'ok' else 1)
" 2>/dev/null; then
            echo "  ρ=${rho}: already done (skip)"
            continue
        fi

        sed "s/INJECTION_RATE/${rho}/g" "${cfg_path}" > "$cfg_tmp"

        echo -n "  ρ=${rho}: running... "
        # Note: booksim2 exits with code 255 even on success; check log content instead
        timeout 900 "${BOOKSIM}" "$cfg_tmp" > "$log_file" 2>&1 || true
        if grep -q "Total run time" "$log_file" 2>/dev/null; then
            echo "done"
            parse_booksim_log "$log_file" "$topo_key" "$rho" "$json_file" \
                "$zero_load_cycles_measured" "$zero_load_ns_real"
        else
            echo "TIMEOUT/ERROR (see ${log_file})"
        fi

        rm -f "$cfg_tmp"
    done
}

# Call signature: run_topology_sweep <topo_key> <cfg_template> <out_dir> <zero_load_cycles> <zero_load_ns_real>
# zero_load_cycles: BookSim2 cycle-based zero-load latency (used for dimensionless slowdown)
# zero_load_ns_real: actual nanosecond zero-load latency (used for avg_latency_ns output)
#
# Dragonfly minimal routing: 3 hops × 100ns + 50ns injection = 350ns real
#   BookSim2 cycle estimate: injection(1) + local_hop(1) + global_hop(1) + local_hop(1) + eject(1) = 5 cycles
# Fat-tree ECMP: 4 hops × 100ns + 50ns injection = 450ns real; BookSim2 ~6 cycles
# Torus3d DOR: avg ~12 hops × 100ns + 50ns injection = 1250ns real; BookSim2 ~12 cycles

if [[ "$TOPO_FILTER" == "all" || "$TOPO_FILTER" == "dragonfly" ]]; then
    run_topology_sweep "dragonfly"        "dragonfly.cfg.template"        "output/dragonfly"        5   350
    run_topology_sweep "dragonfly_1000n"  "dragonfly_1000n.cfg.template"  "output/dragonfly_1000n"  5   350
    run_topology_sweep "dragonfly_10000n" "dragonfly_10000n.cfg.template" "output/dragonfly_10000n" 5   350
fi

if [[ "$TOPO_FILTER" == "all" || "$TOPO_FILTER" == "fattree" ]]; then
    run_topology_sweep "fattree"          "fattree.cfg.template"          "output/fattree"           6   450
fi

if [[ "$TOPO_FILTER" == "all" || "$TOPO_FILTER" == "torus3d" ]]; then
    run_topology_sweep "torus3d"          "torus3d.cfg.template"          "output/torus3d"           12  1250
fi

echo ""
echo "[BookSim2] Sweep complete."

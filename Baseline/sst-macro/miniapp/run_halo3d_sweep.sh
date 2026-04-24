#!/bin/bash
# SST-Macro halo3d-26 Baseline Sweep on 3D Torus
# ================================================
# Matches RAPS STENCIL_3D scenario:
#   - Topology: 8×8×8 torus, 512 nodes, 9.6 GB/s links
#   - Traffic: halo3d-26 (6 face + 12 edge + 8 vertex neighbors)
#   - Vary grid size to control injection rate (ρ)
#
# halo3d message sizes per iteration per rank (MPI_DOUBLE=8 bytes):
#   Face: 6 × ny*nz doubles = 6 × nx² × 8 bytes (for cubic grid nx=ny=nz)
#   Edge: 12 × nx doubles = 12 × nx × 8 bytes
#   Vertex: 8 × 1 double = 64 bytes
#   Total ≈ 6 × nx² × 8 bytes (face-dominated)
#
# To achieve target ρ on max link:
#   tx_volume = ρ × link_bw × DT / max_coeff
#   nx = sqrt(tx_volume / (6 × n_iters × 8))
#
# Usage:
#   bash Baseline/sst-macro/miniapp/run_halo3d_sweep.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAPS_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUT_DIR="${SCRIPT_DIR}/output/halo3d_torus"
mkdir -p "${OUT_DIR}"

SSTMAC="${SSTMAC_BIN:-/lustre/orion/proj-shared/gen053/ndt/sst-run/sstmac}"
NDT=/lustre/orion/proj-shared/gen053/ndt
SST_LIBPATH="${NDT}/sst-macro/sstmac/main/.libs:${NDT}/sst-macro/sstmac/install/.libs:${NDT}/sst-macro/sst-dumpi/dumpi/libundumpi/.libs"
export LD_LIBRARY_PATH="${SST_LIBPATH}:${LD_LIBRARY_PATH:-}"

PYTHON="${RAPS_ROOT}/.venv/bin/python3"
TEMPLATE="${SCRIPT_DIR}/halo3d_torus_sweep.ini.template"

ZERO_LOAD_NS=1250  # avg ~12 hops × 100ns + 50ns
N_ITERATIONS=20    # fixed iterations per run

echo "============================================"
echo "SST-Macro halo3d-26 Torus Sweep"
echo "  Topology: 8×8×8 torus (512 nodes)"
echo "  App: halo3d-26, ${N_ITERATIONS} iterations"
echo "  sstmac: ${SSTMAC}"
echo "============================================"

# Compute grid sizes for each target ρ
# We need to know max_coeff for stencil traffic on this torus.
# For a regular torus with DOR routing and 6-neighbor stencil, each link carries
# traffic from at most 1 flow direction → max_coeff ≈ 1/6 (normalized).
# We'll compute it from RAPS to be exact.
echo ""
echo "[Step 1] Computing stencil max_coeff via RAPS..."
MAX_COEFF=$("${PYTHON}" -c "
import sys; sys.path.insert(0, '${RAPS_ROOT}')
from raps.network.torus3d import build_torus3d, link_loads_for_job_torus
from raps.job import CommunicationPattern
G, meta = build_torus3d((8,8,8), hosts_per_router=1)
hosts = sorted([n for n in G.nodes() if G.nodes[n].get('type') == 'host'])
loads = link_loads_for_job_torus(G, meta, hosts, 1.0, comm_pattern=CommunicationPattern.STENCIL_3D)
print(f'{max(loads.values()):.8f}')
")
echo "  max_coeff = ${MAX_COEFF}"

LINK_BW=9600000000  # 9.6 GB/s

echo ""
echo "[Step 2] Computing grid sizes for each target ρ..."
# For each ρ, compute the grid size nx=ny=nz such that max link utilization = ρ
# tx_volume = ρ × link_bw × 1s / max_coeff  (bytes per host total)
# Per iteration: tx_per_iter = 6 × nx² × 8 (face) + 12 × nx × 8 (edge) + 64 (vertex)
# tx_volume = n_iters × tx_per_iter
# Solve for nx: 6 × nx² × 8 × n_iters ≈ tx_volume → nx ≈ sqrt(tx_volume / (48 × n_iters))

GRID_SIZES=$("${PYTHON}" -c "
import math
max_coeff = float('${MAX_COEFF}')
link_bw = ${LINK_BW}
n_iters = ${N_ITERATIONS}
rhos = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
for rho in rhos:
    tx_vol = rho * link_bw * 1.0 / max_coeff
    # Solve: n_iters × (6×nx²×8 + 12×nx×8 + 64) = tx_vol
    # Approximate: nx = sqrt(tx_vol / (48 × n_iters))
    nx = int(math.sqrt(tx_vol / (48.0 * n_iters)))
    nx = max(nx, 2)  # minimum grid size
    # Recompute actual tx_per_iter
    tx_per_iter = 6 * nx * nx * 8 + 12 * nx * 8 + 64
    actual_tx = n_iters * tx_per_iter
    actual_rho = actual_tx * max_coeff / (link_bw * 1.0)
    print(f'{rho}|{nx}|{actual_rho:.4f}|{tx_per_iter}')
")

echo "${GRID_SIZES}" | while IFS='|' read rho nx actual_rho tx_per_iter; do
    echo "  ρ=${rho}: nx=${nx}, actual_ρ=${actual_rho}, tx/iter=${tx_per_iter} bytes"
done

echo ""
echo "[Step 3] Running halo3d sweep..."

echo "${GRID_SIZES}" | while IFS='|' read rho nx actual_rho tx_per_iter; do
    rho_str="${rho/./p}"
    cfg_path="/tmp/halo3d_torus_${rho_str}.ini"
    log_file="${OUT_DIR}/rho_${rho_str}.log"
    json_file="${OUT_DIR}/summary_${rho_str}.json"

    # Skip if already done
    if [[ -f "${json_file}" ]] && "${PYTHON}" -c "
import json, sys
d = json.load(open('${json_file}'))
sys.exit(0 if d.get('status') == 'ok' else 1)
" 2>/dev/null; then
        echo "  ρ=${rho}: already done (skip)"
        continue
    fi

    # Generate config
    sed -e "s/GRID_NX/${nx}/g" \
        -e "s/GRID_NY/${nx}/g" \
        -e "s/GRID_NZ/${nx}/g" \
        -e "s/N_ITERATIONS/${N_ITERATIONS}/g" \
        "${TEMPLATE}" > "${cfg_path}"

    echo -n "  ρ=${rho} (nx=${nx}): running... "

    if timeout 600 env LD_LIBRARY_PATH="${SST_LIBPATH}" "${SSTMAC}" \
        -f "${cfg_path}" > "${log_file}" 2>&1; then
        echo "done"
    else
        echo "TIMEOUT/ERROR"
        "${PYTHON}" -c "
import json
json.dump({'status':'failed','rho_target':${rho},'grid_nx':${nx},'log_file':'${log_file}'},
          open('${json_file}','w'), indent=2)
"
        rm -f "${cfg_path}"
        continue
    fi

    # Parse halo3d output: per-iteration times
    "${PYTHON}" - << PYEOF > "${json_file}"
import json, re

log_file = "${log_file}"
rho = ${rho}
nx = ${nx}
n_iters = ${N_ITERATIONS}
zero_load_ns = ${ZERO_LOAD_NS}

log_text = open(log_file).read()

# Parse per-rank iteration times: "Rank R = [x,y,z] iteration I:  T s"
iter_times = {}  # rank -> [times]
for m in re.finditer(r'Rank\s+(\d+)\s+=.*iteration\s+(\d+):\s+([\d.]+)s', log_text):
    rank, it, t = int(m.group(1)), int(m.group(2)), float(m.group(3))
    iter_times.setdefault(rank, []).append(t)

# Parse total time
total_time = None
m = re.search(r'Total time\s*=\s*([\d.eE+-]+)', log_text)
if m:
    total_time = float(m.group(1))

# Parse estimated runtime
est_runtime = None
m = re.search(r'Estimated total runtime of\s+([\d.eE+-]+)\s+seconds', log_text)
if m:
    est_runtime = float(m.group(1))

# Parse wall time
wall_time = None
m = re.search(r'SST/macro ran for\s+([\d.]+)\s+seconds', log_text)
if m:
    wall_time = float(m.group(1))

# Compute average iteration time across all ranks
all_iters = []
for rank, times in iter_times.items():
    all_iters.extend(times)

avg_iter_time_s = sum(all_iters) / len(all_iters) if all_iters else None
avg_iter_time_ns = avg_iter_time_s * 1e9 if avg_iter_time_s else None

# Slowdown = observed / zero_load
slowdown = avg_iter_time_ns / zero_load_ns if avg_iter_time_ns else None
stall_ratio = slowdown - 1.0 if slowdown else None

# tx per iteration
tx_per_iter = 6 * nx * nx * 8 + 12 * nx * 8 + 64
actual_tx = n_iters * tx_per_iter

data = {
    "simulator": "sst-macro",
    "app": "halo3d-26",
    "topology": "torus3d_stencil",
    "rho_target": rho,
    "rho_actual": float("${actual_rho}"),
    "grid_nx": nx,
    "grid_ny": nx,
    "grid_nz": nx,
    "n_iterations": n_iters,
    "N_hosts": 512,
    "dims": [8, 8, 8],
    "tx_per_iter_bytes": tx_per_iter,
    "total_tx_bytes": actual_tx,
    "avg_iter_time_ns": avg_iter_time_ns,
    "total_time_s": total_time,
    "estimated_runtime_s": est_runtime,
    "wall_time_s": wall_time,
    "slowdown": slowdown,
    "stall_ratio": stall_ratio,
    "zero_load_latency_ns": zero_load_ns,
    "n_ranks_parsed": len(iter_times),
    "n_iter_samples": len(all_iters),
    "log_file": log_file,
    "status": "ok",
}
print(json.dumps(data, indent=2))
PYEOF

    rm -f "${cfg_path}"
done

echo ""
echo "[Done] Results in ${OUT_DIR}/"

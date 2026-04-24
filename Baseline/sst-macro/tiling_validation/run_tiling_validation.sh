#!/bin/bash
# SST-Macro Tiling Validation Sweep
# ===================================
#
# Runs halo3d-26 on a 1056-node dragonfly at multiple rank counts:
#   N = 64, 216, 512  (template scales — small, the "tiles")
#   M = 512, 1000     (target scales — what RAPS predicts via tiling)
#
# For each N, varies the local grid size (nx) across target injection rates
# ρ ∈ {0.05, 0.10, 0.20, 0.30, 0.40, 0.50}.
#
# Outputs: output/dragonfly/halo3d_N<N>_rho<rho>.json
#
# Usage:
#   export SSTMAC_BIN=/path/to/sstmac   # optional override
#   bash Baseline/sst-macro/tiling_validation/run_tiling_validation.sh
#   bash Baseline/sst-macro/tiling_validation/run_tiling_validation.sh --rhos 0.1 0.3 0.5
#   bash Baseline/sst-macro/tiling_validation/run_tiling_validation.sh --rank-counts 64 512

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAPS_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${RAPS_ROOT}/.venv/bin/python3"

SSTMAC="${SSTMAC_BIN:-/lustre/orion/proj-shared/gen053/ndt/sst-run/sstmac}"
NDT=/lustre/orion/proj-shared/gen053/ndt
SST_LIBPATH="${NDT}/sst-macro/sstmac/main/.libs:${NDT}/sst-macro/sstmac/install/.libs:${NDT}/sst-macro/sst-dumpi/dumpi/libundumpi/.libs"
export LD_LIBRARY_PATH="${SST_LIBPATH}:${LD_LIBRARY_PATH:-}"

TEMPLATE="${SCRIPT_DIR}/halo3d_dragonfly.ini.template"
OUT_DIR="${SCRIPT_DIR}/output/dragonfly"
mkdir -p "${OUT_DIR}"

# ── Parse args ────────────────────────────────────────────────────────────────
RANK_COUNTS=(64 216 512)   # perfect cubes → cubic 3D grids
RHOS=(0.05 0.10 0.20 0.30 0.40 0.50)
N_ITERATIONS=30
LINK_BW=25000000000        # 25 GB/s (bytes/s)
ZERO_LOAD_NS=350           # ~3 hops × 100ns + 50ns injection
TIMEOUT_S=900              # 15 min per config

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rank-counts) shift; RANK_COUNTS=("$@"); break ;;
        --rhos)        shift; RHOS=("$@"); break ;;
        --timeout)     TIMEOUT_S="$2"; shift 2 ;;
        *) echo "[WARN] Unknown arg: $1"; shift ;;
    esac
done

echo "=============================================="
echo "SST-Macro Tiling Validation — Dragonfly"
echo "  Rank counts : ${RANK_COUNTS[*]}"
echo "  Target ρ    : ${RHOS[*]}"
echo "  N_iterations: ${N_ITERATIONS}"
echo "  sstmac      : ${SSTMAC}"
echo "=============================================="

# ── Helper: factor N into (PEX, PEY, PEZ) with PEX × PEY × PEZ = N ─────────
factor_3d() {
    local N=$1
    "${PYTHON}" -c "
import math, sys
N = int('${N}')
# Find most cubic factorization
cbrt = int(round(N ** (1/3)))
best = (N, 1, 1)
best_diff = N
for x in range(max(1, cbrt-2), cbrt+3):
    if N % x != 0: continue
    rem = N // x
    sqrt_rem = int(round(rem ** 0.5))
    for y in range(max(1, sqrt_rem-2), sqrt_rem+3):
        if rem % y != 0: continue
        z = rem // y
        diff = max(x,y,z) - min(x,y,z)
        if diff < best_diff or (diff == best_diff and x*y*z < best[0]*best[1]*best[2]):
            best = (x, y, z)
            best_diff = diff
x, y, z = best
print(f'{x} {y} {z}')
"
}

# ── Helper: compute max_link_coeff for stencil3d at N ranks on this dragonfly ─
max_stencil_coeff() {
    local N=$1
    "${PYTHON}" -c "
import sys; sys.path.insert(0, '${RAPS_ROOT}')
from raps.network.dragonfly import build_dragonfly_circulant, build_dragonfly_idx_map_circulant
from raps.network.base import link_loads_for_pattern
from raps.job import CommunicationPattern
G, _ = build_dragonfly_circulant(33, 16, 2, 15, None)
real_to_fat = build_dragonfly_idx_map_circulant(33, 16, 2, 33*16*2)
hosts = [real_to_fat[i] for i in range(int('${N}'))]
loads = link_loads_for_pattern(G, hosts, 1.0, CommunicationPattern.STENCIL_3D)
mc = max(loads.values()) if loads else 1e-6
print(f'{mc:.8f}')
"
}

echo ""
echo "Pre-computing stencil max_link_coeff for each rank count..."
declare -A MAX_COEFFS
for N in "${RANK_COUNTS[@]}"; do
    mc=$(max_stencil_coeff "${N}")
    MAX_COEFFS[$N]="${mc}"
    echo "  N=${N}: max_coeff=${mc}"
done

echo ""
echo "Running sweeps..."

for N in "${RANK_COUNTS[@]}"; do
    read -r PEX PEY PEZ <<< "$(factor_3d "${N}")"
    mc="${MAX_COEFFS[$N]}"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  N=${N}  grid=${PEX}×${PEY}×${PEZ}  max_coeff=${mc}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for rho in "${RHOS[@]}"; do
        rho_str="${rho/./p}"
        json_file="${OUT_DIR}/halo3d_N${N}_rho${rho_str}.json"
        log_file="${OUT_DIR}/halo3d_N${N}_rho${rho_str}.log"
        cfg_file="/tmp/halo3d_df_N${N}_rho${rho_str}.ini"

        # Skip if done
        if [[ -f "${json_file}" ]] && "${PYTHON}" -c "
import json,sys; d=json.load(open('${json_file}')); sys.exit(0 if d.get('status')=='ok' else 1)
" 2>/dev/null; then
            echo "  ρ=${rho}: already done (skip)"
            continue
        fi

        # Compute grid size nx so that max link util ≈ ρ
        # tx_per_iter = 6 * nx² * 8 (face) + 12 * nx * 8 (edge) + 64 (vertex)
        # total_tx = N_iterations × tx_per_iter
        # target: total_tx × mc = ρ × link_bw × 1s
        nx=$("${PYTHON}" -c "
import math
rho = float('${rho}')
mc = float('${mc}')
link_bw = ${LINK_BW}
n_iters = ${N_ITERATIONS}
tx_vol = rho * link_bw / mc
# Solve: n_iters * (6*nx^2*8) ≈ tx_vol  (face-dominated)
nx = max(2, int(math.sqrt(tx_vol / (48.0 * n_iters))))
print(nx)
")

        # Compute actual ρ
        actual_rho=$("${PYTHON}" -c "
nx = int('${nx}')
mc = float('${mc}')
link_bw = ${LINK_BW}
n_iters = ${N_ITERATIONS}
tx_per_iter = 6*nx*nx*8 + 12*nx*8 + 64
total_tx = n_iters * tx_per_iter
rho_actual = total_tx * mc / link_bw
print(f'{rho_actual:.4f}')
")

        echo -n "  ρ_target=${rho} (nx=${nx}, ρ_actual=${actual_rho}): running... "

        # Generate config
        sed -e "s/N_RANKS/${N}/g" \
            -e "s/PEX/${PEX}/g" \
            -e "s/PEY/${PEY}/g" \
            -e "s/PEZ/${PEZ}/g" \
            -e "s/GRID_NX/${nx}/g" \
            -e "s/GRID_NY/${nx}/g" \
            -e "s/GRID_NZ/${nx}/g" \
            -e "s/N_ITERATIONS/${N_ITERATIONS}/g" \
            "${TEMPLATE}" > "${cfg_file}"

        if timeout "${TIMEOUT_S}" env LD_LIBRARY_PATH="${SST_LIBPATH}" \
                "${SSTMAC}" -f "${cfg_file}" > "${log_file}" 2>&1; then
            echo "done"
        else
            echo "TIMEOUT/ERROR"
            "${PYTHON}" -c "
import json
json.dump({'status':'failed','N_ranks':${N},'rho_target':${rho},'grid_nx':${nx},
           'log_file':'${log_file}'},
          open('${json_file}','w'), indent=2)
"
            rm -f "${cfg_file}"
            continue
        fi

        # Parse halo3d output
        "${PYTHON}" - << PYEOF > "${json_file}"
import json, re, math

log_file  = "${log_file}"
N_ranks   = ${N}
rho       = ${rho}
nx        = ${nx}
n_iters   = ${N_ITERATIONS}
zero_load = ${ZERO_LOAD_NS}
actual_rho = float("${actual_rho}")

log_text = open(log_file).read()

# Per-rank iteration times: "Rank R = [x,y,z] iteration I:  T s"
iter_times = {}
for m in re.finditer(r'Rank\s+(\d+)\s+=.*iteration\s+(\d+):\s+([\d.]+)s', log_text):
    rank, it, t = int(m.group(1)), int(m.group(2)), float(m.group(3))
    iter_times.setdefault(rank, []).append(t)

# Estimated runtime
est_runtime = None
m = re.search(r'Estimated total runtime of\s+([\d.eE+-]+)\s+seconds', log_text)
if m:
    est_runtime = float(m.group(1))

# Wall time
wall_time = None
m = re.search(r'SST/macro ran for\s+([\d.]+)\s+seconds', log_text)
if m:
    wall_time = float(m.group(1))

all_iters = [t for times in iter_times.values() for t in times]
avg_iter_time_s  = sum(all_iters) / len(all_iters) if all_iters else None
avg_iter_time_ns = avg_iter_time_s * 1e9 if avg_iter_time_s else None
slowdown         = avg_iter_time_ns / zero_load if avg_iter_time_ns else None
stall_ratio      = slowdown - 1.0 if slowdown else None

tx_per_iter = 6 * nx * nx * 8 + 12 * nx * 8 + 64
total_tx    = n_iters * tx_per_iter

data = {
    "simulator": "sst-macro", "app": "halo3d-26",
    "topology": "dragonfly", "routing": "dragonfly_minimal",
    "N_ranks": N_ranks, "rho_target": rho, "rho_actual": actual_rho,
    "pex": ${PEX}, "pey": ${PEY}, "pez": ${PEZ},
    "grid_nx": nx, "grid_ny": nx, "grid_nz": nx,
    "n_iterations": n_iters,
    "tx_per_iter_bytes": tx_per_iter, "total_tx_bytes": total_tx,
    "avg_iter_time_ns": avg_iter_time_ns,
    "estimated_runtime_s": est_runtime,
    "wall_time_s": wall_time,
    "slowdown": slowdown,
    "stall_ratio": stall_ratio,
    "zero_load_ns": zero_load,
    "n_ranks_parsed": len(iter_times),
    "log_file": log_file,
    "status": "ok",
}
print(json.dumps(data, indent=2))
PYEOF

        rm -f "${cfg_file}"

        # Print quick summary
        "${PYTHON}" -c "
import json
d = json.load(open('${json_file}'))
sd = d.get('slowdown')
sd_s = f'{sd:.3f}' if sd else '?'
print(f'    N={d[\"N_ranks\"]}  ρ={d[\"rho_target\"]}→{d[\"rho_actual\"]:.3f}  slowdown={sd_s}')
"
    done  # rhos
done  # rank counts

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Summary (all N, all ρ):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
"${PYTHON}" - << 'PYEOF'
import json, glob, sys
sys.path.insert(0, "$(realpath ${RAPS_ROOT})")
files = sorted(glob.glob("${OUT_DIR}/halo3d_N*.json"))
print(f"{'N':>6} {'rho_tgt':>8} {'rho_act':>8} {'slowdown':>10} {'stall':>8} {'status':>8}")
for f in files:
    d = json.load(open(f))
    N  = d.get('N_ranks', '?')
    rt = d.get('rho_target', '?')
    ra = d.get('rho_actual', '?')
    sd = d.get('slowdown')
    sr = d.get('stall_ratio')
    st = d.get('status', '?')
    sd_s = f"{sd:.3f}" if sd else "-"
    sr_s = f"{sr:.3f}" if sr else "-"
    ra_s = f"{ra:.3f}" if isinstance(ra, float) else "-"
    print(f"{N:>6} {rt:>8} {ra_s:>8} {sd_s:>10} {sr_s:>8} {st:>8}")
PYEOF

echo ""
echo "[Done] Results in ${OUT_DIR}/"
echo "  Next: python src/validate_tiling.py --part 2"
echo "  Then: python src/plot_tiling_validation.py"

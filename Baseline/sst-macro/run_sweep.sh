#!/bin/bash
# SST-Macro Dragonfly 1056-node Baseline Sweep
# Topology: 33 groups × 8 routers/group × 4 hosts/router = 1056 hosts
# Matches BookSim2 dragonflynew k=4 and RAPS build_dragonfly(d=8, a=32, p=4)
# h=32 (even) → no SST-Macro parity error
# Uses offered_load app with controlled injection rate.
#
# ρ → constant_delay conversion (per-send delay to each destination):
#   message_size = 65536 bytes (64KB)
#   max_coeff    = computed by RAPS for 1056-node dragonfly (build_dragonfly(d=8,a=32,p=4))
#   link_bw      = 25e9 bytes/sec
#   constant_delay = message_size × max_coeff / (ρ × link_bw)
#
# Usage:
#   export SSTMAC_BIN=/path/to/sstmac
#   export LD_LIBRARY_PATH=/path/to/sst-macro/libs
#   bash Baseline/sst-macro/run_sweep.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAPS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${SCRIPT_DIR}/output/dragonfly_1000n"
mkdir -p "${OUT_DIR}"

# SST-Macro binary — set SSTMAC_BIN env var or update here
SSTMAC="${SSTMAC_BIN:-/lustre/orion/proj-shared/gen053/ndt/sst-run/sstmac}"

# Library paths for SST-Macro
NDT=/lustre/orion/proj-shared/gen053/ndt
SST_LIBPATH="${NDT}/sst-macro/sstmac/main/.libs:${NDT}/sst-macro/sstmac/install/.libs:${NDT}/sst-macro/sst-dumpi/dumpi/libundumpi/.libs"
export LD_LIBRARY_PATH="${SST_LIBPATH}:${LD_LIBRARY_PATH:-}"

# Verify sstmac is runnable
echo "[SST-Macro] Using: ${SSTMAC}"

# Sweep parameters
RHO_VALUES=(0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80)
MESSAGE_SIZE=65536          # 64 KB
LINK_BW=25000000000         # 25 GB/s
ZERO_LOAD_NS=350            # 3 hops × 100ns + 50ns injection

# ---------------------------------------------------------------------------
# Compute destinations (random permutation, seed=42) and max_coeff for it.
# Topology: build_dragonfly(d=8, a=32, p=4) = 33g × 8r × 4h = 1056 hosts
# app1 uses N=1052 ranks (last 4 hosts reserved for dummy app2 workaround).
# For offered_load, destinations[i] = rank that rank i sends to.
# max_coeff = max number of flows on any single link for this permutation
# (used to set constant_delay so that max link utilization == rho_target).
# NOTE: offered_load crashes as single-app on dragonfly → use two-app workaround.
# ---------------------------------------------------------------------------
echo "[SST-Macro] Computing destinations+max_coeff for 1052-rank permutation via RAPS..."
PYTHON="${RAPS_ROOT}/.venv/bin/python3"

DEST_AND_COEFF=$(cd "${RAPS_ROOT}" && "${PYTHON}" - << 'PYEOF'
import sys, random
sys.path.insert(0, '.')
from raps.network.dragonfly import build_dragonfly
from raps.network.base import link_loads_for_job

G = build_dragonfly(d=8, a=32, p=4)
hosts = sorted([n for n, attr in G.nodes(data=True) if attr.get('layer') == 'host'])
if not hosts:
    hosts = sorted([n for n in G.nodes() if str(n).startswith('h_')])
N = min(len(hosts), 1052)  # app1 uses 1052 ranks; last 4 hosts reserved for dummy app2

# Random permutation, seed=42, no fixed points
rng = random.Random(42)
perm = list(range(N))
while True:
    rng.shuffle(perm)
    if all(perm[i] != i for i in range(N)):
        break

# Compute max flows on any link for this permutation using minimal routing
import networkx as nx
link_flow_count = {}
for i, j in enumerate(perm):
    src, dst = hosts[i], hosts[j]
    try:
        path = nx.shortest_path(G, src, dst)
    except nx.NetworkXNoPath:
        continue
    for a, b in zip(path, path[1:]):
        key = tuple(sorted([a, b]))
        link_flow_count[key] = link_flow_count.get(key, 0) + 1

max_coeff = max(link_flow_count.values()) if link_flow_count else 1
dest_str = ', '.join(str(x) for x in perm)
print(f"{max_coeff:.4f}|{dest_str}")
PYEOF
)

if [[ -z "$DEST_AND_COEFF" ]]; then
    echo "ERROR: Failed to compute destinations/max_coeff from RAPS."
    exit 1
fi

MAX_COEFF="${DEST_AND_COEFF%%|*}"
DESTINATIONS="${DEST_AND_COEFF#*|}"

echo "[SST-Macro] 1056-node dragonfly sweep (33g×8r×4h, h=32) — app1 uses 1052 ranks, dummy app2 uses 4"
echo "  message_size=${MESSAGE_SIZE} B, max_coeff=${MAX_COEFF} (for 1052-rank permutation), link_bw=${LINK_BW} B/s"

for rho in "${RHO_VALUES[@]}"; do
    rho_str="${rho/./p}"

    # Compute constant_delay in nanoseconds
    # constant_delay = message_size × max_coeff / (rho × link_bw) seconds
    delay_s=$(python3 -c "
msg=${MESSAGE_SIZE}; mc=${MAX_COEFF}; bw=${LINK_BW}; rho=${rho}
d = msg * mc / (rho * bw)
print(f'{d:.6e}')
")
    delay_ns=$(python3 -c "print(f'{float(\"${delay_s}\") * 1e9:.3f}')")

    echo ""
    echo -n "  ρ=${rho} (delay=${delay_ns}ns): "

    # Generate config with substituted delay and destinations (use Python to
    # avoid shell/sed issues with very long destination lists)
    cfg_path="/tmp/sstmacro_1056n_${rho_str}.ini"
    "${PYTHON}" - << PYEOF
import sys
template = open('${SCRIPT_DIR}/dragonfly_1000n_sweep.ini').read()
cfg = template.replace('CONSTANT_DELAY_NS', '${delay_ns}').replace('DESTINATIONS_LIST', '[${DESTINATIONS}]')
open('${cfg_path}', 'w').write(cfg)
PYEOF

    log_file="${OUT_DIR}/rho_${rho_str}.log"
    json_file="${OUT_DIR}/summary_${rho_str}.json"

    # Skip if already completed successfully
    if [[ -f "${json_file}" ]] && python3 -c "
import json, sys
d = json.load(open('${json_file}'))
sys.exit(0 if d.get('status','') not in ('','failed') else 1)
" 2>/dev/null; then
        echo "already done (skip)"
        rm -f "${cfg_path}"
        continue
    fi

    timeout 600 env LD_LIBRARY_PATH="${SST_LIBPATH}" "${SSTMAC}" \
        -f "${cfg_path}" > "${log_file}" 2>&1 || {
        echo "TIMEOUT/ERROR (${log_file})"
        rm -f "${cfg_path}"
        python3 -c "
import json
print(json.dumps({'simulator':'sst-macro','topology':'dragonfly_1000n',
    'rho_target':${rho},'status':'failed','log_file':'${log_file}'}, indent=2))
" > "${json_file}"
        continue
    }

    echo "done"

    # Parse SST-Macro offered_load output
    # Format: "Message SRC->DST on iteration I of size S took  T s"
    # We extract per-message latency and compute average
    # Also: "Estimated total runtime of  T seconds"
    avg_latency_s=$("${PYTHON}" -c "
import re, sys
lats = []
with open('${log_file}') as f:
    for line in f:
        m = re.match(r'Message\s+\d+->(\d+)\s+on iteration.*took\s+([\d.eE+-]+)s', line)
        if m:
            lats.append(float(m.group(2)))
if lats:
    print(f'{sum(lats)/len(lats):.10e}')
else:
    print('null')
" 2>/dev/null || echo "null")

    wall_time=$(grep -o 'SST/macro ran for\s*[0-9.]*' "${log_file}" 2>/dev/null \
        | awk '{print $NF}' | head -1 || echo "null")

    # Convert avg latency from seconds to nanoseconds
    if [[ "$avg_latency_s" != "null" && "$avg_latency_s" != "" ]]; then
        avg_latency=$("${PYTHON}" -c "print(f'{float(\"${avg_latency_s}\") * 1e9:.3f}')" 2>/dev/null || echo "null")
        stall=$("${PYTHON}" -c "
lat=${avg_latency}; z=${ZERO_LOAD_NS}
print(max(0.0, (lat-z)/z) if z>0 and lat>0 else 0.0)
" 2>/dev/null || echo "null")
        slowdown=$("${PYTHON}" -c "
lat=${avg_latency}; z=${ZERO_LOAD_NS}
print(lat/z if z>0 and lat>0 else 1.0)
" 2>/dev/null || echo "null")
    else
        avg_latency="null"
        stall="null"
        slowdown="null"
    fi

    "${PYTHON}" - << PYEOF > "${json_file}"
import json
data = {
    "simulator": "sst-macro",
    "topology": "dragonfly_1000n",
    "rho_target": ${rho},
    "constant_delay_ns": ${delay_ns},
    "message_size_bytes": ${MESSAGE_SIZE},
    "avg_latency_ns": ${avg_latency} if "${avg_latency}" != "null" else None,
    "wall_time_s": ${wall_time} if "${wall_time}" != "null" else None,
    "stall_ratio": ${stall} if "${stall}" != "null" else None,
    "slowdown": ${slowdown} if "${slowdown}" != "null" else None,
    "zero_load_latency_ns": ${ZERO_LOAD_NS},
    "max_coeff": ${MAX_COEFF},
    "N_hosts": 1056,
    "N_ranks_app1": 1052,
    "topology_detail": "33g x 8r x 4h (h=32, matches BookSim2 k=4); app1=1052 ranks, dummy app2=4 ranks",
    "log_file": "${log_file}",
    "status": "ok",
}
print(json.dumps({k:v for k,v in data.items()}, indent=2))
PYEOF
    echo "    → ${json_file}"

    rm -f "${cfg_path}"
done

echo ""
echo "[SST-Macro] Sweep complete. Results in ${OUT_DIR}/"

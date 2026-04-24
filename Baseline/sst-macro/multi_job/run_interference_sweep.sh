#!/bin/bash
# ================================================================
# SST-Macro Inter-Job Interference Sweep
# ================================================================
# Validates RAPS inter-job congestion model against SST-Macro SNAPPR.
#
# Design:
#   - Both apps use halo3d-26 (offered_load crashes on dragonfly)
#   - Interleaved allocation: victim on even hosts, bully on odd hosts
#   - Both apps share ALL router-to-router links
#   - Victim: fixed nx=50, bully: vary nx ∈ {0,50,100,150,200,300,400}
#   - Three topologies: dragonfly, torus, fat-tree
#
# Usage:
#   bash Baseline/sst-macro/multi_job/run_interference_sweep.sh [topo]
#   # topo = dragonfly|torus|fattree|all (default: all)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAPS_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
NODE_DIR="${SCRIPT_DIR}/node_files"
OUT_DIR="${SCRIPT_DIR}/output/interference"
mkdir -p "${OUT_DIR}"

SSTMAC="${SSTMAC_BIN:-/lustre/orion/proj-shared/gen053/ndt/sst-run/sstmac}"
NDT=/lustre/orion/proj-shared/gen053/ndt
SST_LIBPATH="${NDT}/sst-macro/sstmac/main/.libs:${NDT}/sst-macro/sstmac/install/.libs:${NDT}/sst-macro/sst-dumpi/dumpi/libundumpi/.libs"
export LD_LIBRARY_PATH="${SST_LIBPATH}:${LD_LIBRARY_PATH:-}"
PYTHON="${RAPS_ROOT}/.venv/bin/python3"

VICTIM_NX=50
VICTIM_ITERS=10
BULLY_ITERS=10
BULLY_NX_VALUES=(0 50 100 150 200 300 400)

TOPO_FILTER="${1:-all}"

# ================================================================
# Topology definitions
# ================================================================

gen_ini_dragonfly() {
    local ini="$1" has_bully="$2" bully_nx="${3:-0}"
    local victim_nodes="${NODE_DIR}/dragonfly_victim_nodes.txt"
    local bully_nodes="${NODE_DIR}/dragonfly_bully_nodes.txt"

    cat > "${ini}" << EOF
node {
  app1 {
    name = halo3d-26
    launch_cmd = aprun -n 36 -N 1
    argv = -pex 3 -pey 3 -pez 4 -nx ${VICTIM_NX} -ny ${VICTIM_NX} -nz ${VICTIM_NX} -iterations ${VICTIM_ITERS} -vars 1 -sleep 0 -print 1
    allocation = node_id
    node_id_file = ${victim_nodes}
    indexing = block
    start = 0ms
  }
EOF
    if [[ "${has_bully}" == "yes" ]]; then
        cat >> "${ini}" << EOF
  app2 {
    name = halo3d-26
    launch_cmd = aprun -n 36 -N 1
    argv = -pex 3 -pey 3 -pez 4 -nx ${bully_nx} -ny ${bully_nx} -nz ${bully_nx} -iterations ${BULLY_ITERS} -vars 1 -sleep 0 -print 1
    allocation = node_id
    node_id_file = ${bully_nodes}
    indexing = block
    start = 0ms
  }
EOF
    fi
    cat >> "${ini}" << 'EOF'
  nic {
    name = snappr
    mtu = 4KB
    credits = 64KB
    bandwidth = 25.0GB/s
    latency = 50ns
    injection {
      bandwidth = 25.0GB/s
      latency = 50ns
      credits = 64KB
    }
  }
  memory {
    name = snappr
    channel_bandwidth = 100GB/s
    num_channels = 1
    latency = 10ns
  }
  proc {
    ncores = 2
    frequency = 2.5GHz
  }
  name = simple
}
switch {
  name = snappr
  mtu = 4KB
  credits = 64KB
  link {
    bandwidth = 25.0GB/s
    latency = 100ns
    credits = 64KB
  }
  logp {
    bandwidth = 25GB/s
    hop_latency = 100ns
    out_in_latency = 100ns
  }
  router {
    name = dragonfly_minimal
    seed = 42
  }
}
topology {
  name = dragonfly
  geometry = [4, 9]
  concentration = 2
  h = 8
  inter_group = alltoall
}
EOF
}

gen_ini_torus() {
    local ini="$1" has_bully="$2" bully_nx="${3:-0}"
    local victim_nodes="${NODE_DIR}/torus_victim_nodes.txt"
    local bully_nodes="${NODE_DIR}/torus_bully_nodes.txt"

    cat > "${ini}" << EOF
node {
  app1 {
    name = halo3d-26
    launch_cmd = aprun -n 512 -N 1
    argv = -pex 8 -pey 8 -pez 8 -nx ${VICTIM_NX} -ny ${VICTIM_NX} -nz ${VICTIM_NX} -iterations ${VICTIM_ITERS} -vars 1 -sleep 0 -print 1
    allocation = node_id
    node_id_file = ${victim_nodes}
    indexing = block
    start = 0ms
  }
EOF
    if [[ "${has_bully}" == "yes" ]]; then
        cat >> "${ini}" << EOF
  app2 {
    name = halo3d-26
    launch_cmd = aprun -n 512 -N 1
    argv = -pex 8 -pey 8 -pez 8 -nx ${bully_nx} -ny ${bully_nx} -nz ${bully_nx} -iterations ${BULLY_ITERS} -vars 1 -sleep 0 -print 1
    allocation = node_id
    node_id_file = ${bully_nodes}
    indexing = block
    start = 0ms
  }
EOF
    fi
    cat >> "${ini}" << 'EOF'
  nic {
    name = snappr
    mtu = 4KB
    credits = 64KB
    bandwidth = 9.6GB/s
    latency = 50ns
    injection {
      bandwidth = 9.6GB/s
      latency = 50ns
      credits = 64KB
    }
  }
  memory {
    name = snappr
    channel_bandwidth = 100GB/s
    num_channels = 1
    latency = 10ns
  }
  proc {
    ncores = 1
    frequency = 2.5GHz
  }
  name = simple
}
switch {
  name = snappr
  mtu = 4KB
  credits = 64KB
  link {
    bandwidth = 9.6GB/s
    latency = 100ns
    credits = 64KB
  }
  logp {
    bandwidth = 9.6GB/s
    hop_latency = 100ns
    out_in_latency = 100ns
  }
  router {
    name = torus_minimal
    seed = 42
  }
}
topology {
  name = torus
  geometry = [8,8,8]
  concentration = 2
}
EOF
}

gen_ini_fattree() {
    local ini="$1" has_bully="$2" bully_nx="${3:-0}"
    local victim_nodes="${NODE_DIR}/fattree_victim_nodes.txt"
    local bully_nodes="${NODE_DIR}/fattree_bully_nodes.txt"

    cat > "${ini}" << EOF
node {
  app1 {
    name = halo3d-26
    launch_cmd = aprun -n 8 -N 1
    argv = -pex 2 -pey 2 -pez 2 -nx ${VICTIM_NX} -ny ${VICTIM_NX} -nz ${VICTIM_NX} -iterations ${VICTIM_ITERS} -vars 1 -sleep 0 -print 1
    allocation = node_id
    node_id_file = ${victim_nodes}
    indexing = block
    start = 0ms
  }
EOF
    if [[ "${has_bully}" == "yes" ]]; then
        cat >> "${ini}" << EOF
  app2 {
    name = halo3d-26
    launch_cmd = aprun -n 8 -N 1
    argv = -pex 2 -pey 2 -pez 2 -nx ${bully_nx} -ny ${bully_nx} -nz ${bully_nx} -iterations ${BULLY_ITERS} -vars 1 -sleep 0 -print 1
    allocation = node_id
    node_id_file = ${bully_nodes}
    indexing = block
    start = 0ms
  }
EOF
    fi
    cat >> "${ini}" << 'EOF'
  nic {
    name = snappr
    mtu = 4KB
    credits = 64KB
    bandwidth = 12.5GB/s
    latency = 50ns
    injection {
      bandwidth = 12.5GB/s
      latency = 50ns
      credits = 64KB
    }
  }
  memory {
    name = snappr
    channel_bandwidth = 100GB/s
    num_channels = 1
    latency = 10ns
  }
  proc {
    ncores = 1
    frequency = 2.5GHz
  }
  name = simple
}
switch {
  name = snappr
  mtu = 4KB
  credits = 64KB
  link {
    bandwidth = 12.5GB/s
    latency = 100ns
    credits = 64KB
  }
  logp {
    bandwidth = 12.5GB/s
    hop_latency = 100ns
    out_in_latency = 100ns
  }
  router {
    name = fat_tree
    seed = 42
  }
}
topology {
  name = fat_tree
  num_core_switches = 2
  num_agg_subtrees = 2
  leaf_switches_per_subtree = 2
  agg_switches_per_subtree = 2
  up_ports_per_leaf_switch = 2
  down_ports_per_agg_switch = 2
  up_ports_per_agg_switch = 1
  down_ports_per_core_switch = 2
  concentration = 4
}
EOF
}

# ================================================================
# Result parser
# ================================================================

parse_result() {
    local logfile="$1" topo="$2" bully_nx="$3" jsonfile="$4"
    "${PYTHON}" << PYEOF > "${jsonfile}"
import json, re, sys
import numpy as np
from collections import defaultdict

log = open("${logfile}").read()

# Parse all iteration times
iters = defaultdict(list)
for m in re.finditer(r'Rank\s+(\d+)\s+=.*iteration\s+(\d+):\s+([\d.]+)s', log):
    rank, it, t = int(m.group(1)), int(m.group(2)), float(m.group(3))
    iters[rank].append(t)

totals = [float(t) for t in re.findall(r'Total time\s*=\s+([\d.eE+-]+)', log)]
wall_m = re.search(r'SST/macro ran for\s+([\d.]+)\s+seconds', log)

bully_nx = ${bully_nx}

if not iters:
    print(json.dumps({"topology": "${topo}", "bully_nx": bully_nx, "status": "failed"}))
    sys.exit(0)

all_times = [t for times in iters.values() for t in times]

if bully_nx == 0:
    # No bully: all iterations are victim
    victim_times = all_times
    bully_times = []
    victim_total = totals[0] if totals else None
else:
    # Two apps: need to separate victim vs bully iterations
    # Both use same local rank numbering (0..N-1)
    # With halo3d, bully (nx=${bully_nx}) has larger messages → longer iterations
    # Victim (nx=50) has shorter iterations
    # Strategy: each rank has VICTIM_ITERS + BULLY_ITERS data points
    # First VICTIM_ITERS are app1 (victim), next BULLY_ITERS are app2 (bully)
    # BUT SST-Macro interleaves output by simulated time, not by app
    # So we separate by magnitude: compute expected ratio
    # Face bytes: nx^2 * 8 * 6 ≈ volume
    victim_vol = 50**2
    bully_vol = bully_nx**2
    # If bully_nx > 50, bully iterations take longer
    # Threshold: geometric mean of expected times
    if bully_nx > 70:
        threshold = np.median(all_times)
        victim_times = [t for t in all_times if t <= threshold]
        bully_times = [t for t in all_times if t > threshold]
    else:
        # Similar traffic levels: can't separate reliably by magnitude
        # Use per-rank: first N iters = victim, last N = bully
        victim_times = []
        bully_times = []
        for rank, times in iters.items():
            mid = len(times) // 2
            victim_times.extend(times[:mid])
            bully_times.extend(times[mid:])

    # Victim total is the smaller total (finishes first)
    victim_total = min(totals) if totals else None

# Per-iteration average for victim
n_iters = ${VICTIM_ITERS}
per_iter = {}
if victim_times:
    n_ranks = len(iters)
    for it_idx in range(n_iters):
        start = it_idx * n_ranks
        end = start + n_ranks
        if end <= len(victim_times):
            per_iter[str(it_idx)] = float(np.mean(victim_times[start:end])) * 1e6

data = {
    "topology": "${topo}",
    "bully_nx": bully_nx,
    "victim_nx": 50,
    "victim_iters": ${VICTIM_ITERS},
    "bully_iters": ${BULLY_ITERS},
    "victim_avg_iter_us": float(np.mean(victim_times)) * 1e6 if victim_times else None,
    "victim_median_iter_us": float(np.median(victim_times)) * 1e6 if victim_times else None,
    "victim_max_iter_us": float(np.max(victim_times)) * 1e6 if victim_times else None,
    "victim_total_us": victim_total * 1e6 if victim_total else None,
    "bully_avg_iter_us": float(np.mean(bully_times)) * 1e6 if bully_times else None,
    "per_iter_avg_us": per_iter,
    "n_ranks_parsed": len(iters),
    "n_victim_samples": len(victim_times),
    "n_bully_samples": len(bully_times),
    "wall_time_s": float(wall_m.group(1)) if wall_m else None,
    "status": "ok",
}
print(json.dumps(data, indent=2))
PYEOF
}

# ================================================================
# Main sweep loop
# ================================================================

run_topology() {
    local topo="$1"
    local gen_fn="gen_ini_${topo}"
    local topo_dir="${OUT_DIR}/${topo}"
    mkdir -p "${topo_dir}"

    echo ""
    echo "========================================"
    echo "  Topology: ${topo}"
    echo "========================================"

    for bully_nx in "${BULLY_NX_VALUES[@]}"; do
        local tag="${topo}_bully_nx${bully_nx}"
        local json_file="${topo_dir}/${tag}.json"
        local log_file="${topo_dir}/${tag}.log"
        local ini_file="/tmp/interf_${tag}.ini"

        # Skip if already done
        if [[ -f "${json_file}" ]] && "${PYTHON}" -c "
import json,sys;d=json.load(open('${json_file}'));sys.exit(0 if d.get('status')=='ok' else 1)
" 2>/dev/null; then
            local sd=$("${PYTHON}" -c "
import json
d=json.load(open('${json_file}'))
print(f'{d.get(\"victim_avg_iter_us\",0):.1f}μs')
")
            echo "  bully_nx=${bully_nx}: skip (done, avg=${sd})"
            continue
        fi

        # Generate INI
        if [[ "${bully_nx}" == "0" ]]; then
            ${gen_fn} "${ini_file}" "no"
        else
            ${gen_fn} "${ini_file}" "yes" "${bully_nx}"
        fi

        echo -n "  bully_nx=${bully_nx}: running... "
        if timeout 600 "${SSTMAC}" -f "${ini_file}" > "${log_file}" 2>&1; then
            echo -n "done, "
        else
            echo "TIMEOUT/ERROR"
            echo "{\"topology\":\"${topo}\",\"bully_nx\":${bully_nx},\"status\":\"failed\"}" > "${json_file}"
            continue
        fi

        parse_result "${log_file}" "${topo}" "${bully_nx}" "${json_file}"

        # Print summary
        "${PYTHON}" -c "
import json
d = json.load(open('${json_file}'))
avg = d.get('victim_avg_iter_us')
total = d.get('victim_total_us')
if avg: print(f'avg={avg:.1f}μs, total={total:.0f}μs')
else: print('PARSE FAILED')
"
        rm -f "${ini_file}"
    done
}

# ================================================================
# Run selected topologies
# ================================================================

echo "============================================"
echo "SST-Macro Inter-Job Interference Sweep"
echo "  Victim: halo3d-26 nx=${VICTIM_NX}, ${VICTIM_ITERS} iters"
echo "  Bully:  halo3d-26 nx=varying, ${BULLY_ITERS} iters"
echo "  Bully NX: ${BULLY_NX_VALUES[*]}"
echo "============================================"

if [[ "${TOPO_FILTER}" == "all" ]]; then
    for topo in dragonfly torus fattree; do
        run_topology "${topo}"
    done
else
    run_topology "${TOPO_FILTER}"
fi

echo ""
echo "========================================"
echo "  Summary"
echo "========================================"

"${PYTHON}" << 'PYEOF'
import json, os, glob

out_dir = os.environ.get("OUT_DIR_PY", "Baseline/sst-macro/multi_job/output/interference")
# Find all result files
results = {}
for topo in ["dragonfly", "torus", "fattree"]:
    topo_dir = os.path.join(out_dir, topo)
    if not os.path.isdir(topo_dir):
        continue
    baseline_avg = None
    for f in sorted(glob.glob(os.path.join(topo_dir, f"{topo}_bully_nx*.json"))):
        d = json.load(open(f))
        if d.get("status") != "ok":
            continue
        bully_nx = d.get("bully_nx", -1)
        avg = d.get("victim_avg_iter_us", 0)
        if bully_nx == 0:
            baseline_avg = avg
        results.setdefault(topo, []).append((bully_nx, avg))

    if baseline_avg and results.get(topo):
        print(f"\n{topo}:")
        print(f"  {'bully_nx':>8s}  {'avg_iter':>9s}  {'slowdown':>8s}")
        print(f"  {'-'*30}")
        for bully_nx, avg in sorted(results[topo]):
            sd = avg / baseline_avg if baseline_avg else 0
            print(f"  {bully_nx:>8d}  {avg:>8.1f}μs  {sd:>7.3f}x")
PYEOF

echo ""
echo "[Done] Results in ${OUT_DIR}/"

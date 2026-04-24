#!/bin/bash
# ================================================================
# SST-Macro Inter-Job Interference Sweep — LARGE SCALE
# ================================================================
# Paper-quality validation: all topologies at ~1000 hosts, 1000 iterations.
#
# Topologies (all ~1000 hosts):
#   - Dragonfly: 1024 hosts (32g × 16r × 2h, circulant h=14) — matches RAPS TOPOLOGIES_LARGE
#   - Torus: 1024 hosts (8×8×8 × 2h/r)
#   - Fat-tree: 1024 hosts (k=16)
#
# Design:
#   - Victim: halo3d-26, nx=100, 1000 iterations
#   - Bully: halo3d-26, nx varying, 1000 iterations
#   - Interleaved allocation: victim on even hosts, bully on odd hosts
#
# Estimated time per topology (7 configs):
#   - Baseline (no bully): ~6 min
#   - With bully nx=50-200: ~10-25 min each
#   - With bully nx=300-400: ~30-40 min each (heaviest)
#   - Total per topology: ~100-160 min
#   → Each topology must be run as a separate SLURM job (2h limit)
#
# Usage:
#   bash Baseline/sst-macro/multi_job/run_interference_sweep_large.sh [topo]
#   # topo = dragonfly|torus|fattree|all (default: all)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAPS_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUT_DIR="${SCRIPT_DIR}/output/interference_large"
mkdir -p "${OUT_DIR}"

SSTMAC="${SSTMAC_BIN:-/lustre/orion/proj-shared/gen053/ndt/sst-run/sstmac}"
NDT=/lustre/orion/proj-shared/gen053/ndt
SST_LIBPATH="${NDT}/sst-macro/sstmac/main/.libs:${NDT}/sst-macro/sstmac/install/.libs:${NDT}/sst-macro/sst-dumpi/dumpi/libundumpi/.libs"
export LD_LIBRARY_PATH="${SST_LIBPATH}:${LD_LIBRARY_PATH:-}"
PYTHON="${RAPS_ROOT}/.venv/bin/python3"

VICTIM_NX=100
VICTIM_ITERS=1000
BULLY_ITERS=1000
BULLY_NX_VALUES=(0 50 100 150 200 300 400)

TOPO_FILTER="${1:-all}"

# ================================================================
# Generate node files
# ================================================================
gen_node_files() {
    local n_hosts="$1" prefix="$2"
    local victim_file="${OUT_DIR}/${prefix}_victim_nodes.txt"
    local bully_file="${OUT_DIR}/${prefix}_bully_nodes.txt"

    if [[ -f "${victim_file}" && -f "${bully_file}" ]]; then
        return
    fi

    "${PYTHON}" -c "
n = ${n_hosts}
victim = list(range(0, n, 2))
bully = list(range(1, n, 2))
with open('${victim_file}', 'w') as f:
    f.write(f'{len(victim)}\n')
    f.write(' '.join(str(x) for x in victim) + '\n')
with open('${bully_file}', 'w') as f:
    f.write(f'{len(bully)}\n')
    f.write(' '.join(str(x) for x in bully) + '\n')
print(f'  Node files: {len(victim)} victim + {len(bully)} bully = {n} hosts')
"
}

# ================================================================
# Topology: Dragonfly 1024 hosts (32g × 16r × 2h, circulant h=14)
# Matches RAPS TOPOLOGIES_LARGE["dragonfly"] and CIRCULANT_PARAMS[1000]
# Port budget: 2 + (16-1) + 14 = 31 ≤ 64 ✓
# ================================================================
gen_ini_dragonfly() {
    local ini="$1" has_bully="$2" bully_nx="${3:-0}"
    local n_ranks=512  # 1024/2
    local victim_nodes="${OUT_DIR}/dragonfly_victim_nodes.txt"
    local bully_nodes="${OUT_DIR}/dragonfly_bully_nodes.txt"

    cat > "${ini}" << EOF
node {
  app1 {
    name = halo3d-26
    launch_cmd = aprun -n ${n_ranks} -N 1
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
    launch_cmd = aprun -n ${n_ranks} -N 1
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
  # Circulant: 16 routers/group, 32 groups, h=14 inter-group links/router
  # Matches RAPS circulant params for node_count=1000 (32*16*2=1024 hosts)
  geometry = [16, 32]
  concentration = 2
  h = 14
  inter_group = circulant
}
EOF
}

# ================================================================
# Topology: 3D Torus 1024 hosts (8×8×8 × 2h/r)
# ================================================================
gen_ini_torus() {
    local ini="$1" has_bully="$2" bully_nx="${3:-0}"
    local n_ranks=512  # 1024/2
    local victim_nodes="${OUT_DIR}/torus_victim_nodes.txt"
    local bully_nodes="${OUT_DIR}/torus_bully_nodes.txt"

    cat > "${ini}" << EOF
node {
  app1 {
    name = halo3d-26
    launch_cmd = aprun -n ${n_ranks} -N 1
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
    launch_cmd = aprun -n ${n_ranks} -N 1
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

# ================================================================
# Topology: Fat-tree k=16, 1024 hosts
# ================================================================
gen_ini_fattree() {
    local ini="$1" has_bully="$2" bully_nx="${3:-0}"
    local n_ranks=512  # 1024/2
    local victim_nodes="${OUT_DIR}/fattree_victim_nodes.txt"
    local bully_nodes="${OUT_DIR}/fattree_bully_nodes.txt"

    cat > "${ini}" << EOF
node {
  app1 {
    name = halo3d-26
    launch_cmd = aprun -n ${n_ranks} -N 1
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
    launch_cmd = aprun -n ${n_ranks} -N 1
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
  num_core_switches = 64
  num_agg_subtrees = 16
  leaf_switches_per_subtree = 8
  agg_switches_per_subtree = 8
  up_ports_per_leaf_switch = 8
  down_ports_per_agg_switch = 8
  up_ports_per_agg_switch = 8
  down_ports_per_core_switch = 16
  concentration = 8
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
    victim_times = all_times
    bully_times = []
    victim_total = totals[0] if totals else None
else:
    if bully_nx > 120:
        threshold = np.median(all_times)
        victim_times = [t for t in all_times if t <= threshold]
        bully_times = [t for t in all_times if t > threshold]
    else:
        victim_times = []
        bully_times = []
        for rank, times in iters.items():
            mid = len(times) // 2
            victim_times.extend(times[:mid])
            bully_times.extend(times[mid:])
    victim_total = min(totals) if totals else None

data = {
    "topology": "${topo}",
    "bully_nx": bully_nx,
    "victim_nx": ${VICTIM_NX},
    "victim_iters": ${VICTIM_ITERS},
    "bully_iters": ${BULLY_ITERS},
    "scale": "large",
    "n_hosts": {"dragonfly": 1024, "torus": 1024, "fattree": 1024}["${topo}"],
    "victim_avg_iter_us": float(np.mean(victim_times)) * 1e6 if victim_times else None,
    "victim_median_iter_us": float(np.median(victim_times)) * 1e6 if victim_times else None,
    "victim_max_iter_us": float(np.max(victim_times)) * 1e6 if victim_times else None,
    "victim_total_us": victim_total * 1e6 if victim_total else None,
    "bully_avg_iter_us": float(np.mean(bully_times)) * 1e6 if bully_times else None,
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

    # Determine n_hosts for node file generation
    local n_hosts
    case "${topo}" in
        dragonfly) n_hosts=1024 ;;
        torus) n_hosts=1024 ;;
        fattree) n_hosts=1024 ;;
    esac

    echo ""
    echo "========================================"
    echo "  Topology: ${topo} (${n_hosts} hosts)"
    echo "  Victim: halo3d-26 nx=${VICTIM_NX}, ${VICTIM_ITERS} iters"
    echo "  Bully: halo3d-26 nx=varying, ${BULLY_ITERS} iters"
    echo "========================================"

    gen_node_files "${n_hosts}" "${topo}"

    for bully_nx in "${BULLY_NX_VALUES[@]}"; do
        local tag="${topo}_bully_nx${bully_nx}"
        local json_file="${topo_dir}/${tag}.json"
        local log_file="${topo_dir}/${tag}.log"
        local ini_file="/tmp/interf_large_${tag}.ini"

        # Skip if already done
        if [[ -f "${json_file}" ]] && "${PYTHON}" -c "
import json,sys;d=json.load(open('${json_file}'));sys.exit(0 if d.get('status')=='ok' else 1)
" 2>/dev/null; then
            local sd=$("${PYTHON}" -c "
import json; d=json.load(open('${json_file}')); print(f'{d.get(\"victim_avg_iter_us\",0):.1f}μs, wall={d.get(\"wall_time_s\",0):.1f}s')
")
            echo "  bully_nx=${bully_nx}: skip (done, avg=${sd})"
            continue
        fi

        if [[ "${bully_nx}" == "0" ]]; then
            ${gen_fn} "${ini_file}" "no"
        else
            ${gen_fn} "${ini_file}" "yes" "${bully_nx}"
        fi

        echo -n "  bully_nx=${bully_nx}: running... "
        local t_start=$(date +%s)
        if timeout 7200 "${SSTMAC}" -f "${ini_file}" > "${log_file}" 2>&1; then
            local t_end=$(date +%s)
            local dt=$((t_end - t_start))
            echo -n "done (${dt}s), "
        else
            local t_end=$(date +%s)
            local dt=$((t_end - t_start))
            echo "TIMEOUT/ERROR after ${dt}s"
            echo "{\"topology\":\"${topo}\",\"bully_nx\":${bully_nx},\"status\":\"failed\"}" > "${json_file}"
            continue
        fi

        parse_result "${log_file}" "${topo}" "${bully_nx}" "${json_file}"

        "${PYTHON}" -c "
import json
d = json.load(open('${json_file}'))
avg = d.get('victim_avg_iter_us')
wt = d.get('wall_time_s')
if avg: print(f'avg={avg:.1f}μs, wall={wt:.1f}s')
else: print('PARSE FAILED')
"
        rm -f "${ini_file}"
    done
}

# ================================================================
# Run
# ================================================================
echo "============================================"
echo "SST-Macro Inter-Job Interference Sweep (LARGE)"
echo "  All topologies: ~1000 hosts"
echo "  Victim: halo3d-26 nx=${VICTIM_NX}, ${VICTIM_ITERS} iters"
echo "  Bully:  halo3d-26 nx=varying, ${BULLY_ITERS} iters"
echo "  Bully NX: ${BULLY_NX_VALUES[*]}"
echo "  Topologies: dragonfly(1056), torus(1024), fattree(1024)"
echo "============================================"

if [[ "${TOPO_FILTER}" == "all" ]]; then
    for topo in dragonfly fattree torus; do
        run_topology "${topo}"
    done
else
    run_topology "${TOPO_FILTER}"
fi

# Summary
echo ""
echo "========================================"
echo "  Summary"
echo "========================================"

export OUT_DIR_PY="${OUT_DIR}"
"${PYTHON}" << 'PYEOF'
import json, os, glob

out_dir = os.environ.get("OUT_DIR_PY", ".")
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
        wall = d.get("wall_time_s", 0)
        if bully_nx == 0:
            baseline_avg = avg
        results.setdefault(topo, []).append((bully_nx, avg, wall))

    if baseline_avg and results.get(topo):
        n_hosts = {"dragonfly": 1056, "torus": 1024, "fattree": 1024}.get(topo, "?")
        print(f"\n{topo} ({n_hosts} hosts):")
        print(f"  {'bully_nx':>8s}  {'avg_iter':>9s}  {'slowdown':>8s}  {'wall(s)':>7s}")
        print(f"  {'-'*40}")
        for bully_nx, avg, wall in sorted(results[topo]):
            sd = avg / baseline_avg if baseline_avg else 0
            print(f"  {bully_nx:>8d}  {avg:>8.1f}μs  {sd:>7.3f}x  {wall:>7.1f}")
PYEOF

echo ""
echo "[Done] Results in ${OUT_DIR}/"

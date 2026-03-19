#!/bin/bash
# SST-Macro Bully-Victim Interference Sweep
# ===========================================
#
# Runs bully-victim multi-job experiments for all three topologies:
#   dragonfly (Frontier), fat-tree (Lassen), torus3d (Blue Waters)
#
# Pre-requisite: bully_victim.py generates ini files with the correct
#   offered_load parameters (constant_delay, destinations).
#
# app1 = bully (high injection), app2 = victims (moderate injection).
# Both run simultaneously; PISCES packet-level model captures interference.
#
# Usage:
#   export SSTMAC_BIN=/path/to/sstmac
#   bash Baseline/sst-macro/multi_job/run_multi_job.sh
#   bash Baseline/sst-macro/multi_job/run_multi_job.sh dragonfly
#   bash Baseline/sst-macro/multi_job/run_multi_job.sh fattree torus3d

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAPS_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${RAPS_ROOT}/.venv/bin/python3"

# SST-Macro binary
SSTMAC="${SSTMAC_BIN:-/lustre/orion/proj-shared/gen053/ndt/sst-run/sstmac}"
NDT=/lustre/orion/proj-shared/gen053/ndt
SST_LIBPATH="${NDT}/sst-macro/sstmac/main/.libs:${NDT}/sst-macro/sstmac/install/.libs:${NDT}/sst-macro/sst-dumpi/dumpi/libundumpi/.libs"
export LD_LIBRARY_PATH="${SST_LIBPATH}:${LD_LIBRARY_PATH:-}"

# ── Parse topology args (default: all three) ──────────────────────────────
if [[ $# -gt 0 ]]; then
    TOPOLOGIES=("$@")
else
    TOPOLOGIES=("dragonfly" "fattree" "torus3d")
fi

echo "============================================"
echo "SST-Macro Bully-Victim Interference Sweep"
echo "  Topologies : ${TOPOLOGIES[*]}"
echo "  sstmac     : ${SSTMAC}"
echo "============================================"

# ── Step 1: Generate ini configs ──────────────────────────────────────────
echo ""
echo "[Step 1] Generating ini configs via bully_victim.py ..."
"${PYTHON}" "${SCRIPT_DIR}/bully_victim.py" --topology "${TOPOLOGIES[@]}"

# ── Step 2: Run each config ──────────────────────────────────────────────
for TOPO in "${TOPOLOGIES[@]}"; do
    OUT_DIR="${SCRIPT_DIR}/output/${TOPO}"
    [[ -d "${OUT_DIR}" ]] || continue

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ${TOPO}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for ini_file in "${OUT_DIR}"/bully_*.ini; do
        [[ -f "${ini_file}" ]] || continue
        tag=$(basename "${ini_file}" .ini)
        json_file="${OUT_DIR}/${tag}.json"
        log_file="${OUT_DIR}/${tag}.log"

        # Skip if already done
        if [[ -f "${json_file}" ]] && "${PYTHON}" -c "
import json, sys
d = json.load(open('${json_file}'))
sys.exit(0 if d.get('status') == 'ok' else 1)
" 2>/dev/null; then
            echo "  ${tag}: already done (skip)"
            continue
        fi

        echo -n "  ${tag}: running... "

        # Run SST-Macro (timeout 15 min per config)
        if timeout 900 env LD_LIBRARY_PATH="${SST_LIBPATH}" "${SSTMAC}" \
            -f "${ini_file}" > "${log_file}" 2>&1; then
            echo "done"
        else
            echo "TIMEOUT/ERROR"
            # Mark as failed
            "${PYTHON}" -c "
import json, os
jf = '${json_file}'
meta = json.load(open(jf)) if os.path.exists(jf) else {}
meta['status'] = 'failed'
meta['log_file'] = '${log_file}'
json.dump(meta, open(jf, 'w'), indent=2)
"
            continue
        fi

        # ── Parse SST-Macro output ────────────────────────────────────
        # Look for app completion times and latency stats
        # SST-Macro outputs vary by version; try several patterns
        "${PYTHON}" - << 'PYEOF'
import json, re, sys, os

log_file = "${log_file}"
json_file = "${json_file}"

meta = json.load(open(json_file)) if os.path.exists(json_file) else {}
log_text = open(log_file).read()

# --- Parse app2 (victim) finish time ---
# Pattern: "app 2 total running time" or "Rank N finished"
victim_time = None
for pattern in [
    r'app\s*2.*running\s*time.*?(\d+\.?\d*(?:e[+-]?\d+)?)\s*(us|ms|s)',
    r'Estimated\s+total\s+runtime.*?(\d+\.?\d*(?:e[+-]?\d+)?)\s*(us|ms|s)',
]:
    m = re.search(pattern, log_text, re.IGNORECASE)
    if m:
        val, unit = float(m.group(1)), m.group(2).lower()
        if unit == 'us': victim_time = val / 1e3  # → ms
        elif unit == 'ms': victim_time = val
        elif unit == 's': victim_time = val * 1e3
        break

# --- Parse app1 (bully) finish time ---
bully_time = None
for pattern in [
    r'app\s*1.*running\s*time.*?(\d+\.?\d*(?:e[+-]?\d+)?)\s*(us|ms|s)',
]:
    m = re.search(pattern, log_text, re.IGNORECASE)
    if m:
        val, unit = float(m.group(1)), m.group(2).lower()
        if unit == 'us': bully_time = val / 1e3
        elif unit == 'ms': bully_time = val
        elif unit == 's': bully_time = val * 1e3
        break

# --- Parse average latency from any available metric ---
avg_latency_ns = None
for pattern in [
    r'(?:avg|mean)\s+(?:message\s+)?latency\s*[:=]\s*(\d+\.?\d*(?:e[+-]?\d+)?)\s*(ns|us|ms)',
    r'Latency.*?(\d+\.?\d*(?:e[+-]?\d+)?)\s*(ns|us|ms)',
]:
    m = re.search(pattern, log_text, re.IGNORECASE)
    if m:
        val, unit = float(m.group(1)), m.group(2).lower()
        if unit == 'ns': avg_latency_ns = val
        elif unit == 'us': avg_latency_ns = val * 1e3
        elif unit == 'ms': avg_latency_ns = val * 1e6
        break

# --- Compute slowdown ---
zero_load = meta.get('zero_load_ns', 200)
if avg_latency_ns is not None and zero_load > 0 and avg_latency_ns > 0:
    avg_victim_slowdown = avg_latency_ns / zero_load
else:
    avg_victim_slowdown = None

meta['status'] = 'ok'
meta['log_file'] = log_file
meta['victim_time_ms'] = victim_time
meta['bully_time_ms'] = bully_time
meta['avg_latency_ns'] = avg_latency_ns
meta['avg_victim_slowdown'] = avg_victim_slowdown
meta['max_victim_slowdown'] = avg_victim_slowdown  # conservative estimate

json.dump(meta, open(json_file, 'w'), indent=2)

# Print summary line
bn = meta.get('bully_nodes', '?')
lf = meta.get('load_fraction', 0)
sd = f"{avg_victim_slowdown:.3f}" if avg_victim_slowdown else "?"
print(f"    bully={bn}n  load={lf:.3f}  slowdown={sd}")
PYEOF

    done  # ini files

    # ── Summary table ─────────────────────────────────────────────────
    echo ""
    echo "  Summary for ${TOPO}:"
    "${PYTHON}" - << 'PYEOF'
import json, glob
files = sorted(glob.glob("${OUT_DIR}/bully_*.json"))
print(f"  {'bully_n':>8}  {'load':>6}  {'slowdown':>10}  {'latency_ns':>12}  {'status':>8}")
for f in files:
    d = json.load(open(f))
    bn = d.get('bully_nodes', '?')
    lf = d.get('load_fraction', 0)
    sd = d.get('avg_victim_slowdown')
    lat = d.get('avg_latency_ns')
    st = d.get('status', '?')
    sd_s = f"{sd:.3f}" if sd else "-"
    lat_s = f"{lat:.1f}" if lat else "-"
    print(f"  {bn:>8}  {lf:>6.3f}  {sd_s:>10}  {lat_s:>12}  {st:>8}")
PYEOF

done  # topologies

echo ""
echo "[Done] Results in ${SCRIPT_DIR}/output/"
echo "  Next: python src/plot_interference_validation.py"

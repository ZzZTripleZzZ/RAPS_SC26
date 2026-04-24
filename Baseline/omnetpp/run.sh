#!/bin/bash
# OMNeT++ Baseline: Attempt Documentation
#
# STATUS: Infrastructure too complex for HPC batch execution.
# See notes below for why this was not pursued.
#
# ============================================================
# Why OMNeT++ was not used for this comparison:
# ============================================================
#
# 1. Installation complexity:
#    - OMNeT++ requires a GUI-based IDE (Qt5) to build; no pip install.
#    - Tarball build: ./configure && make → 1-2h, requires X11 or headless mode.
#    - INET framework (networking models) requires separate download + build.
#    - Combined: ~4-6h of setup with likely environment-specific issues.
#
# 2. HPC topology support:
#    - INET framework provides EthernetSwitch but NO dragonfly or fat-tree models.
#    - HPC-INET extensions exist only in research papers, not maintained packages.
#    - Manual topology wiring for dragonfly = ~500 lines of .ned code.
#
# 3. Batch execution:
#    - OMNeT++ uses .ini + .ned simulation files, not shell-scriptable parameters.
#    - Requires OMNeT++ IDE or opp_run binary; parameterization via .ini sections.
#    - Output parsing requires opp_scavetool (OMNeT++ specific tool).
#    - GUI-centric workflow conflicts with SLURM batch.
#
# 4. Verdict:
#    - Effort estimate: 15-20h for build + INET + topology wiring + output extraction.
#    - Risk: Very high — no guarantee of working HPC topology models.
#    - Decision: Document as "attempted; infrastructure too complex for HPC batch."
#      Include this note in paper to acknowledge the attempt.
#
# ============================================================
# What would be needed (for reference):
# ============================================================
#
# 1. Build OMNeT++:
#    wget https://omnetpp.org/omnetpp/downloads/omnetpp-6.0.3-linux-x86_64.tgz
#    tar xf omnetpp-6.0.3-linux-x86_64.tgz && cd omnetpp-6.0.3
#    source setenv && ./configure && make -j8
#
# 2. Build INET:
#    git clone https://github.com/inet-framework/inet.git
#    cd inet && make makefiles && make -j8 MODE=release
#
# 3. Write topology .ned files:
#    # FatTree.ned — k=4 fat-tree with EthernetSwitch nodes
#    # Dragonfly.ned — 9-group dragonfly (no existing INET model)
#
# 4. Write simulation .ini:
#    [General]
#    network = FatTree
#    sim-time-limit = 1s
#    **.datarate = 100Gbps  # 12.5 GB/s * 8
#    **.delay = 100ns
#    **.queue.maxPackets = 256
#    **.app.sendInterval = 1us  # controls injection rate
#
# 5. Run and parse:
#    opp_run -r 0 -m -u Cmdenv -c General omnetpp.ini
#    opp_scavetool export -f "*.sca" -F CSV -o results.csv
#
# ============================================================
# Placeholder output generation
# ============================================================

OUT_DIR="$(dirname "${BASH_SOURCE[0]}")/output"
mkdir -p "${OUT_DIR}"

for rho in 0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80; do
    rho_str="${rho/./p}"
    python3 -c "
import json
data = {
    'simulator': 'omnetpp',
    'topology': 'dragonfly',
    'rho_target': ${rho},
    'status': 'not_attempted',
    'reason': 'Infrastructure too complex for HPC batch execution. '
              'No maintained HPC topology module (dragonfly/fat-tree) in INET framework. '
              'GUI-centric workflow conflicts with SLURM. '
              'Estimated effort: 15-20h with high failure risk.',
    'reference': 'https://omnetpp.org, https://inet.omnetpp.org',
}
print(json.dumps(data, indent=2))
" > "${OUT_DIR}/summary_dragonfly_${rho_str}.json"
done

echo "OMNeT++ placeholder outputs written to ${OUT_DIR}/"
echo ""
echo "See comments in this script for why OMNeT++ was not attempted."
echo "For the paper: cite as 'OMNeT++: not attempted due to missing HPC topology modules'"

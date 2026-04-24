#!/bin/bash
#SBATCH --job-name=baseline_simgrid
#SBATCH --account=gen053
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# SimGrid v3.35 Baseline Sweep — build from source + real flow-level simulation
# Topology: dragonfly (72n, 1056n) + fat-tree k=4
# 1056-node dragonfly: 33g×8r×4h, matches BookSim2 k=4 and RAPS build_dragonfly(d=8,a=32,p=4)
# Traffic: all-to-all uniform at rho in {0.05..0.80}
# Model: SimGrid CM02 max-min fairness (flow-level, NOT packet-level)
#
# Prerequisites (done on login node before submitting):
#   git clone --depth=1 --branch v3.35 https://framagit.org/simgrid/simgrid.git \
#       Baseline/simgrid/simgrid-src
#
# Estimated times:
#   Build SimGrid: ~25 min with 16 cores
#   Each rho per topo: ~30s (dragonfly 72n) to ~5 min (dragonfly 1000n)
#   Total sim time: ~30 min

set -euo pipefail

RAPS_ROOT=/lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26
SG_SCRIPT_DIR="${RAPS_ROOT}/Baseline/simgrid"
SG_SRC="${SG_SCRIPT_DIR}/simgrid-src"
SG_BUILD="${SG_SCRIPT_DIR}/simgrid-build"
PYTHON="${RAPS_ROOT}/.venv/bin/python3"

echo "=== SimGrid Baseline Sweep ==="
echo "Date: $(date) | Node: $(hostname) | Job: ${SLURM_JOB_ID:-local}"

# GCC 14 + Cray Python 3.11 headers
module load PrgEnv-gnu 2>/dev/null || true
which g++ && g++ --version | head -1

PYTHON_INC=/opt/cray/pe/python/3.11.5/include/python3.11
PYTHON_LIB=/opt/cray/pe/python/3.11.5/lib/libpython3.11.so
PYBIND11_DIR=$("${PYTHON}" -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null || echo "")

echo "Python headers: ${PYTHON_INC}"
echo "Python lib:     ${PYTHON_LIB}"
echo "pybind11:       ${PYBIND11_DIR:-not found}"

# ---------------------------------------------------------------------------
# Step 1: Verify source exists (must be pre-cloned on login node)
# ---------------------------------------------------------------------------
if [[ ! -d "${SG_SRC}/CMakeLists.txt" && ! -f "${SG_SRC}/CMakeLists.txt" ]]; then
    echo "ERROR: SimGrid source not found at ${SG_SRC}"
    echo "  Run on login node: git clone --depth=1 --branch v3.35 https://framagit.org/simgrid/simgrid.git ${SG_SRC}"
    exit 1
fi
echo "[SimGrid] Source at ${SG_SRC}"

# ---------------------------------------------------------------------------
# Step 2: Configure and build
# ---------------------------------------------------------------------------
if [[ ! -f "${SG_BUILD}/simgrid.so" && ! -f "${SG_BUILD}/lib/simgrid.so" ]]; then
    mkdir -p "${SG_BUILD}"
    cd "${SG_BUILD}"

    echo "[SimGrid] Configuring cmake..."
    CMAKE_EXTRA=""
    if [[ -n "${PYBIND11_DIR}" ]]; then
        CMAKE_EXTRA="-Dpybind11_DIR=${PYBIND11_DIR}"
    fi

    cmake "${SG_SRC}" \
        -DCMAKE_BUILD_TYPE=Release \
        -Denable_python=ON \
        -Dminimal-bindings=ON \
        -Denable_smpi=OFF \
        -Denable_java=OFF \
        -Denable_documentation=OFF \
        -Denable_examples=OFF \
        -Denable_tests=OFF \
        -DPYTHON_EXECUTABLE="${PYTHON}" \
        -DPYTHON_INCLUDE_DIR="${PYTHON_INC}" \
        -DPYTHON_LIBRARY="${PYTHON_LIB}" \
        ${CMAKE_EXTRA} \
        2>&1 | tail -15

    echo "[SimGrid] Building with ${SLURM_CPUS_PER_TASK:-8} cores..."
    make -j${SLURM_CPUS_PER_TASK:-8} simgrid 2>&1 | tail -5
    # Build Python bindings
    make -j${SLURM_CPUS_PER_TASK:-8} python-bindings 2>/dev/null \
        || make -j${SLURM_CPUS_PER_TASK:-8} 2>&1 | tail -5

    echo "[SimGrid] Build complete."
    ls "${SG_BUILD}"/*.so 2>/dev/null || ls "${SG_BUILD}"/lib/*.so 2>/dev/null || true
    cd "${RAPS_ROOT}"
else
    echo "[SimGrid] Build already present, skipping."
fi

# Find the .so file
SG_SO=$(find "${SG_BUILD}" -name "simgrid*.so" 2>/dev/null | head -1)
if [[ -z "${SG_SO}" ]]; then
    echo "ERROR: simgrid.so not found after build"
    find "${SG_BUILD}" -name "*.so" 2>/dev/null | head -10
    exit 1
fi
SG_PYDIR=$(dirname "${SG_SO}")
echo "[SimGrid] Python module at ${SG_SO}"

# ---------------------------------------------------------------------------
# Step 3: Run sweep
# ---------------------------------------------------------------------------
RHO_VALUES=(0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80)
TOPOS=(dragonfly dragonfly_1000n fattree)

export SIMGRID_BUILD="${SG_BUILD}"
export PYTHONPATH="${SG_PYDIR}:${PYTHONPATH:-}"

echo ""
echo "[SimGrid] Starting sweep..."
for topo in "${TOPOS[@]}"; do
    OUT_DIR="${SG_SCRIPT_DIR}/output/${topo}"
    mkdir -p "${OUT_DIR}"
    echo "  Topology: ${topo}"

    for rho in "${RHO_VALUES[@]}"; do
        rho_str="${rho/./p}"
        out_json="${OUT_DIR}/summary_${rho_str}.json"

        echo -n "    ρ=${rho}: "
        timeout 600 "${PYTHON}" "${SG_SCRIPT_DIR}/run_simgrid_sim.py" \
            --topo "${topo}" --rho "${rho}" --out "${out_json}" \
            2>&1 || echo "TIMEOUT/ERROR"
    done
done

echo ""
echo "=== Complete at $(date) ==="
echo "Results:"
for topo in "${TOPOS[@]}"; do
    ls "${SG_SCRIPT_DIR}/output/${topo}"/summary_*.json 2>/dev/null | wc -l \
        | xargs -I{} echo "  ${topo}: {} JSON files"
done

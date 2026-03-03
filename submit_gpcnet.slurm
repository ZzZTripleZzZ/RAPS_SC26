#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J gpcnet
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -N 128
#SBATCH --ntasks-per-node=8
# 128 nodes × 8 ranks/node = 1024 ranks
# Matches RAPS "1000-node" scenario (closest GPCNeT-friendly power-of-2 alignment)

# ── Resubmit logic ──────────────────────────────────────────────────────────
MAX_RESUBMIT=20
count=${SLURM_RESUBMIT_COUNT:-0}

# ── Setup ────────────────────────────────────────────────────────────────────
module load PrgEnv-gnu
module list

BENCH_DIR=/lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26/benchmarks/gpcnet
RUN_DIR=$MEMBERWORK/gen053/gpcnet-${SLURM_JOB_ID}

mkdir -p "$RUN_DIR"
cp -r "$BENCH_DIR/"* "$RUN_DIR/"
cd "$RUN_DIR"

# Compile both binaries fresh in the run directory
echo "=== Compiling GPCNeT ==="
make all CC=cc
if [ ! -f ./network_test ] || [ ! -f ./network_load_test ]; then
    echo "ERROR: compilation failed"
    exit 1
fi
echo "=== Compilation done ==="

NNODES=$SLURM_NNODES
NRANKS=$SLURM_NTASKS
echo "Running on $NNODES nodes, $NRANKS total ranks ($SLURM_NTASKS_PER_NODE ranks/node)"
echo "START: $(date)"

# ── Run 1: Isolated network test (baseline) ──────────────────────────────────
echo ""
echo "=== network_test (isolated baseline) ==="
srun -N $NNODES -n $NRANKS ./network_test 2>&1 | tee network_test_${NNODES}n.out

# ── Run 2: Network load test (under congestion) ──────────────────────────────
echo ""
echo "=== network_load_test (under congestion) ==="
srun -N $NNODES -n $NRANKS ./network_load_test 2>&1 | tee network_load_test_${NNODES}n.out

echo ""
echo "END: $(date)"

# ── Copy results back to persistent storage ───────────────────────────────────
OUT_DIR=/lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26/output/gpcnet
mkdir -p "$OUT_DIR"
cp network_test_${NNODES}n.out network_load_test_${NNODES}n.out "$OUT_DIR/" 2>/dev/null
echo "Results saved to $OUT_DIR"
echo "DONE"

# ── Resubmit if needed (e.g., if cluster was too busy and job was preempted) ──
# GPCNeT runs quickly (<1h), so resubmit is only for infrastructure failures.
# Uncomment if needed:
# if [ $count -lt $MAX_RESUBMIT ]; then
#     SLURM_RESUBMIT_COUNT=$((count+1)) sbatch --export=ALL "$0"
# fi

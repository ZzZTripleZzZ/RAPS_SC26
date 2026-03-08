#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J gpcnet_1000n
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -N 1000
#SBATCH --ntasks-per-node=8
# 1000 nodes × 8 ranks/node = 8000 ranks

module load PrgEnv-gnu

BENCH_DIR=/lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26/benchmarks/gpcnet
RUN_DIR=$MEMBERWORK/gen053/gpcnet-1000n-${SLURM_JOB_ID}

mkdir -p "$RUN_DIR"
cp -r "$BENCH_DIR/"* "$RUN_DIR/"
cd "$RUN_DIR"

echo "=== Compiling GPCNeT ==="
make all CC=cc
if [ ! -f ./network_test ] || [ ! -f ./network_load_test ]; then
    echo "ERROR: compilation failed"; exit 1
fi

NNODES=$SLURM_NNODES
NRANKS=$SLURM_NTASKS
echo "Running on $NNODES nodes, $NRANKS total ranks"
echo "START: $(date)"

echo "=== network_test (isolated baseline) ==="
srun -N $NNODES -n $NRANKS ./network_test 2>&1 | tee network_test_${NNODES}n.out

echo "=== network_load_test (under congestion) ==="
srun -N $NNODES -n $NRANKS ./network_load_test 2>&1 | tee network_load_test_${NNODES}n.out

echo "END: $(date)"

OUT_DIR=/lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26/output/gpcnet
mkdir -p "$OUT_DIR"
cp network_test_${NNODES}n.out network_load_test_${NNODES}n.out "$OUT_DIR/"
echo "Results saved to $OUT_DIR"
echo "DONE"

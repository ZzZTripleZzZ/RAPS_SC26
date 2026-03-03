#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J gpcnet_dumpi_1000n
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -N 1000
#SBATCH --ntasks-per-node=8
# 1000 nodes × 8 ranks = 8000 ranks — matches RAPS "1000-node" scenario

module load PrgEnv-gnu

SST_MACRO_DIR=/lustre/orion/proj-shared/gen053/ndt/sst-macro
SST_DUMPI_LIB=$SST_MACRO_DIR/sst-dumpi/lib/libdumpi.so
BENCH_DIR=/lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26/benchmarks/gpcnet
RUN_DIR=$MEMBERWORK/gen053/gpcnet-dumpi-1000n-${SLURM_JOB_ID}
OUT_DIR=/lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26/benchmarks/sst-macro

mkdir -p "$RUN_DIR"
cp -r "$BENCH_DIR/"* "$RUN_DIR/"
cd "$RUN_DIR"

make all CC=cc
if [ ! -f ./network_test ]; then
    echo "ERROR: compilation failed"; exit 1
fi

echo "=== GPCNeT network_test with DUMPI tracing (1000 nodes / 8000 ranks) ==="

if [ ! -f "$SST_DUMPI_LIB" ]; then
    SST_DUMPI_LIB=$(find $SST_MACRO_DIR -name "libdumpi.so" 2>/dev/null | head -1)
    echo "Found libdumpi at: $SST_DUMPI_LIB"
fi

if [ -f "$SST_DUMPI_LIB" ]; then
    export LD_PRELOAD=$SST_DUMPI_LIB
    echo "LD_PRELOAD=$LD_PRELOAD"
else
    echo "WARNING: libdumpi.so not found, running without DUMPI tracing"; exit 1
fi

echo "Running on $SLURM_NNODES nodes, $SLURM_NTASKS ranks"
echo "START: $(date)"

srun -N $SLURM_NNODES -n $SLURM_NTASKS ./network_test 2>&1 | tee gpcnet_1000n.out

echo "END: $(date)"

ls -la dumpi-*.meta dumpi-*.bin 2>/dev/null | head -10

if ls dumpi-*.meta 1>/dev/null 2>&1; then
    META=$(ls dumpi-*.meta | head -1)
    echo "DUMPI meta file: $META"

    # Copy traces with a node-count prefix so they don't collide
    cp dumpi-*.meta "$OUT_DIR/gpcnet_dumpi_1000n.meta"
    cp dumpi-*.bin  "$OUT_DIR/" 2>/dev/null
    echo "Traces saved to $OUT_DIR/gpcnet_dumpi_1000n.meta"
else
    echo "ERROR: No DUMPI traces generated"; exit 1
fi

echo "DONE"

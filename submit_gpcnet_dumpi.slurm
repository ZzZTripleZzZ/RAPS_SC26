#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J gpcnet_dumpi
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -N 32
#SBATCH --ntasks-per-node=8
# 32 nodes × 8 ranks = 256 ranks
# Generates DUMPI traces from GPCNeT network_test for SST-Macro replay

module load PrgEnv-gnu
module list

SST_MACRO_DIR=/lustre/orion/proj-shared/gen053/ndt/sst-macro
SST_DUMPI_LIB=$SST_MACRO_DIR/sst-dumpi/lib/libdumpi.so
BENCH_DIR=/lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26/benchmarks/gpcnet
RUN_DIR=$MEMBERWORK/gen053/gpcnet-dumpi-${SLURM_JOB_ID}
OUT_DIR=/lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26/benchmarks/sst-macro

mkdir -p "$RUN_DIR"
cp -r "$BENCH_DIR/"* "$RUN_DIR/"
cd "$RUN_DIR"

make all CC=cc
if [ ! -f ./network_test ]; then
    echo "ERROR: compilation failed"; exit 1
fi

echo "=== GPCNeT network_test with DUMPI tracing ==="
echo "SST-DUMPI lib: $SST_DUMPI_LIB"

if [ ! -f "$SST_DUMPI_LIB" ]; then
    # Try to find libdumpi in any subdirectory
    SST_DUMPI_LIB=$(find $SST_MACRO_DIR -name "libdumpi.so" 2>/dev/null | head -1)
    echo "Found libdumpi at: $SST_DUMPI_LIB"
fi

if [ -f "$SST_DUMPI_LIB" ]; then
    export LD_PRELOAD=$SST_DUMPI_LIB
    echo "LD_PRELOAD=$LD_PRELOAD"
else
    echo "WARNING: libdumpi.so not found, running without DUMPI tracing"
fi

NNODES=$SLURM_NNODES
NRANKS=$SLURM_NTASKS

echo "Running on $NNODES nodes, $NRANKS ranks"
echo "START: $(date)"

srun -N $NNODES -n $NRANKS ./network_test 2>&1 | tee gpcnet_run.out

echo "END: $(date)"

# Collect DUMPI traces and metadata
ls -la dumpi-*.meta dumpi-*.bin 2>/dev/null | head -20

if ls dumpi-*.meta 1>/dev/null 2>&1; then
    META=$(ls dumpi-*.meta | head -1)
    echo "DUMPI meta file: $META"

    # Copy traces to our sst-macro benchmark dir
    cp dumpi-*.meta dumpi-*.bin "$OUT_DIR/" 2>/dev/null

    # Update dragonfly.ini with the actual meta filename and rank count
    sed -i "s/gpcnet_dumpi.meta/$META/" "$OUT_DIR/dragonfly.ini"
    sed -i "s/NRANKS/$NRANKS/g" "$OUT_DIR/dragonfly.ini"
    sed -i "s/RANKSPERNODE/$SLURM_NTASKS_PER_NODE/g" "$OUT_DIR/dragonfly.ini"
    echo "Updated dragonfly.ini with meta=$META nranks=$NRANKS"
else
    echo "WARNING: No DUMPI traces generated (libdumpi not available?)"
    echo "SST-Macro replay will not be possible without traces"
fi

echo "Results saved to $OUT_DIR"

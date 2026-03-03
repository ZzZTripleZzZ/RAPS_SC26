#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J sstmacro_100n
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 2:00:00
#SBATCH -p extended
#SBATCH -N 1
# SST-Macro serial simulation for 100-node (800-rank) GPCNeT trace

MAX_RESUBMIT=50
count=${SLURM_RESUBMIT_COUNT:-0}

if [ $count -ge $MAX_RESUBMIT ]; then
    echo "Reached MAX_RESUBMIT=$MAX_RESUBMIT, stopping."
    exit 0
fi

echo "=== SST-Macro 100n run attempt $((count+1)) / $MAX_RESUBMIT ==="

module load PrgEnv-gnu

SST_MACRO_BIN=/lustre/orion/proj-shared/gen053/ndt/sst-macro/bin/sstmac
BENCH_DIR=/lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26/benchmarks/sst-macro
INI_FILE=dragonfly_100n.ini
TRACE_META=gpcnet_dumpi_100n.meta
RUN_DIR=$MEMBERWORK/gen053/sstmacro-100n-${SLURM_JOB_ID}
OUT_DIR=/lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26/output/sstmacro

# Wait for DUMPI traces from gpcnet_dumpi_100n job
if [ ! -f "$BENCH_DIR/$TRACE_META" ]; then
    echo "WARNING: $TRACE_META not found yet (gpcnet_dumpi_100n job may still be running)."
    echo "Resubmitting to wait for traces... (attempt $((count+1)) / $MAX_RESUBMIT)"
    SLURM_RESUBMIT_COUNT=$((count+1)) sbatch --export=ALL "$0"
    exit 0
fi

mkdir -p "$RUN_DIR"
cp -r "$BENCH_DIR/"* "$RUN_DIR/"
cd "$RUN_DIR"

echo "Using sstmac binary: $SST_MACRO_BIN"
echo "Working dir: $RUN_DIR"
echo "Config: $INI_FILE"
echo "DUMPI meta: $TRACE_META"
echo "START: $(date)"

WALL_START=$(date +%s)

"$SST_MACRO_BIN" -f "$INI_FILE" 2>&1 | tee sstmacro_100n_run.out
RC=$?

WALL_END=$(date +%s)
WALL_SEC=$((WALL_END - WALL_START))
echo "Wall-clock time: ${WALL_SEC}s"

mkdir -p "$OUT_DIR"
cp sstmacro_100n_run.out "$OUT_DIR/sstmacro_100n_run_attempt${count}.out" 2>/dev/null
ls ftq*.dat 2>/dev/null && cp ftq*.dat "$OUT_DIR/" 2>/dev/null

echo "attempt=$count wall_sec=$WALL_SEC rc=$RC" >> "$OUT_DIR/timing_100n.txt"

echo "END: $(date) (RC=$RC)"

if [ $RC -ne 0 ]; then
    echo "SST-Macro failed (RC=$RC), resubmitting..."
    SLURM_RESUBMIT_COUNT=$((count+1)) sbatch --export=ALL "$0"
    exit 0
fi

echo "SST-Macro 100n completed successfully."
echo "Results in $OUT_DIR"

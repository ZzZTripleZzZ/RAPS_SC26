#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J sstmacro
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 2:00:00
#SBATCH -p extended
#SBATCH -N 1
# SST-Macro runs in serial simulation mode on 1 node.
# May need multiple submissions if the simulation is large.

# ── Resubmit logic ──────────────────────────────────────────────────────────
MAX_RESUBMIT=50
count=${SLURM_RESUBMIT_COUNT:-0}

if [ $count -ge $MAX_RESUBMIT ]; then
    echo "Reached MAX_RESUBMIT=$MAX_RESUBMIT, stopping."
    exit 0
fi

echo "=== SST-Macro run attempt $((count+1)) / $MAX_RESUBMIT ==="

# ── Setup ────────────────────────────────────────────────────────────────────
module load PrgEnv-gnu
module list

SST_MACRO_BIN=/lustre/orion/proj-shared/gen053/ndt/sst-macro/bin/sstmac
BENCH_DIR=/lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26/benchmarks/sst-macro
RUN_DIR=$MEMBERWORK/gen053/sstmacro-${SLURM_JOB_ID}

# Check that DUMPI traces exist — resubmit and wait if not yet generated
if ! ls "$BENCH_DIR"/dumpi-*.meta 1>/dev/null 2>&1; then
    echo "WARNING: No DUMPI traces found in $BENCH_DIR yet (gpcnet_dumpi job may still be running)."
    echo "Resubmitting to wait for traces... (attempt $((count+1)) / $MAX_RESUBMIT)"
    SLURM_RESUBMIT_COUNT=$((count+1)) sbatch --export=ALL "$0"
    exit 0
fi

mkdir -p "$RUN_DIR"
cp -r "$BENCH_DIR/"* "$RUN_DIR/"
cd "$RUN_DIR"

echo "Using sstmac binary: $SST_MACRO_BIN"
echo "Working dir: $RUN_DIR"
echo "Config: dragonfly.ini"
echo "DUMPI meta: $(ls dumpi-*.meta | head -1)"
echo "START: $(date)"

# Record wall-clock start time for speedup calculation
WALL_START=$(date +%s)

# Run SST-Macro
"$SST_MACRO_BIN" -f dragonfly.ini 2>&1 | tee sstmacro_run.out
RC=$?

WALL_END=$(date +%s)
WALL_SEC=$((WALL_END - WALL_START))
echo "Wall-clock time: ${WALL_SEC}s"

# ── Save results ─────────────────────────────────────────────────────────────
OUT_DIR=/lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26/output/sstmacro
mkdir -p "$OUT_DIR"
cp sstmacro_run.out "$OUT_DIR/sstmacro_run_attempt${count}.out" 2>/dev/null
ls ftq*.dat 2>/dev/null && cp ftq*.dat "$OUT_DIR/" 2>/dev/null

# Write wall-clock timing
echo "attempt=$count wall_sec=$WALL_SEC rc=$RC" >> "$OUT_DIR/timing.txt"

echo "END: $(date) (RC=$RC)"

if [ $RC -ne 0 ]; then
    echo "SST-Macro failed (RC=$RC), resubmitting..."
    SLURM_RESUBMIT_COUNT=$((count+1)) sbatch --export=ALL "$0"
    exit 0
fi

echo "SST-Macro completed successfully."
echo "Results in $OUT_DIR"

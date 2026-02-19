#!/bin/bash
# Resubmit the 5 failed single experiments

cd "$(dirname "$0")/.." || exit 1

echo "=========================================="
echo "Resubmitting 5 failed dt=0.1s experiments"
echo "=========================================="
echo ""

# frontier n=10000 (3 experiments)
echo "[1/5] Resubmitting frontier_n10000_r0..."
JOB1=$(sbatch --parsable --export=SYSTEM=frontier,NODES=10000,REPEAT=0 submit_dt01_single.slurm)
echo "  Job ID: $JOB1"
sleep 2

echo "[2/5] Resubmitting frontier_n10000_r1..."
JOB2=$(sbatch --parsable --export=SYSTEM=frontier,NODES=10000,REPEAT=1 submit_dt01_single.slurm)
echo "  Job ID: $JOB2"
sleep 2

echo "[3/5] Resubmitting frontier_n10000_r2..."
JOB3=$(sbatch --parsable --export=SYSTEM=frontier,NODES=10000,REPEAT=2 submit_dt01_single.slurm)
echo "  Job ID: $JOB3"
sleep 2

# lassen n=10000 (2 experiments)
echo "[4/5] Resubmitting lassen_n10000_r0..."
JOB4=$(sbatch --parsable --export=SYSTEM=lassen,NODES=10000,REPEAT=0 submit_dt01_single.slurm)
echo "  Job ID: $JOB4"
sleep 2

echo "[5/5] Resubmitting lassen_n10000_r1..."
JOB5=$(sbatch --parsable --export=SYSTEM=lassen,NODES=10000,REPEAT=1 submit_dt01_single.slurm)
echo "  Job ID: $JOB5"

echo ""
echo "=========================================="
echo "✅ All 5 jobs resubmitted (fixed script)"
echo "=========================================="
echo ""
echo "  $JOB1 - frontier_n10000_r0"
echo "  $JOB2 - frontier_n10000_r1"
echo "  $JOB3 - frontier_n10000_r2"
echo "  $JOB4 - lassen_n10000_r0"
echo "  $JOB5 - lassen_n10000_r1"
echo ""

#!/bin/bash
# Submit all 8 missing dt=0.1s experiments with optimized strategy

cd "$(dirname "$0")/.." || exit 1

echo "=========================================="
echo "Submitting 8 dt=0.1s experiments (Optimized)"
echo "=========================================="
echo ""

# Job 1: frontier n=1000 (3 experiments, sequential, fast)
echo "[1/6] Submitting frontier_n1000 (3 experiments)..."
JOB1=$(sbatch --parsable submit_dt01_frontier_n1000.slurm)
echo "  Job ID: $JOB1"
echo ""

sleep 2

# Job 2-4: frontier n=10000 (1 experiment each, independent)
echo "[2/6] Submitting frontier_n10000_r0..."
JOB2=$(sbatch --parsable --export=SYSTEM=frontier,NODES=10000,REPEAT=0 submit_dt01_single.slurm)
echo "  Job ID: $JOB2"
echo ""

sleep 2

echo "[3/6] Submitting frontier_n10000_r1..."
JOB3=$(sbatch --parsable --export=SYSTEM=frontier,NODES=10000,REPEAT=1 submit_dt01_single.slurm)
echo "  Job ID: $JOB3"
echo ""

sleep 2

echo "[4/6] Submitting frontier_n10000_r2..."
JOB4=$(sbatch --parsable --export=SYSTEM=frontier,NODES=10000,REPEAT=2 submit_dt01_single.slurm)
echo "  Job ID: $JOB4"
echo ""

sleep 2

# Job 5-6: lassen n=10000 (1 experiment each, independent)
echo "[5/6] Submitting lassen_n10000_r0..."
JOB5=$(sbatch --parsable --export=SYSTEM=lassen,NODES=10000,REPEAT=0 submit_dt01_single.slurm)
echo "  Job ID: $JOB5"
echo ""

sleep 2

echo "[6/6] Submitting lassen_n10000_r1..."
JOB6=$(sbatch --parsable --export=SYSTEM=lassen,NODES=10000,REPEAT=1 submit_dt01_single.slurm)
echo "  Job ID: $JOB6"
echo ""

echo "=========================================="
echo "✅ All 6 jobs submitted!"
echo "=========================================="
echo ""
echo "Submitted jobs:"
echo "  $JOB1 - frontier_n1000 (r0,r1,r2)"
echo "  $JOB2 - frontier_n10000_r0"
echo "  $JOB3 - frontier_n10000_r1"
echo "  $JOB4 - frontier_n10000_r2"
echo "  $JOB5 - lassen_n10000_r0"
echo "  $JOB6 - lassen_n10000_r1"
echo ""
echo "Strategy:"
echo "  • Reduced parallelism (1 experiment per node)"
echo "  • Increased checkpoint interval (10 min)"
echo "  • Independent jobs for large experiments"
echo "  • Auto-resubmit enabled (max 5-10 times)"
echo ""
echo "Monitoring:"
echo "  squeue -u \$USER"
echo "  tail -f output/frontier_scaling/dt01_*.out"
echo ""

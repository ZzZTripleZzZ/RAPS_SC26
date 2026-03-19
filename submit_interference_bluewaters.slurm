#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J e2_bw
#SBATCH -o e2_bw_%j.out
#SBATCH -e e2_bw_%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1

# E2 Bully-Victim Interference — Blue Waters (torus3d)
# Sweeps bully_sizes=[0, 32, 64, 128, 256, 512] × routings=[dor_xyz]
# = 6 experiments, each ~1-3 min → total ~15 min

module load PrgEnv-gnu cray-python
cd /lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26

.venv/bin/python3 src/run_interference_study.py \
    --system bluewaters \
    --bully-sizes 0 32 64 128 256 512 \
    --victim-count 4 \
    --victim-nodes 64 \
    --duration 300 \
    --force

echo "Blue Waters interference study complete"

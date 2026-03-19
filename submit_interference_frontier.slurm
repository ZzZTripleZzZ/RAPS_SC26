#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J e2_frontier
#SBATCH -o e2_frontier_%j.out
#SBATCH -e e2_frontier_%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1

# E2 Bully-Victim Interference — Frontier (dragonfly)
# Sweeps bully_sizes=[0, 32, 64, 128, 256, 512] × routings=[minimal, valiant]
# = 12 experiments, each ~1-3 min → total ~30 min

module load PrgEnv-gnu cray-python
cd /lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26

.venv/bin/python3 src/run_interference_study.py \
    --system frontier \
    --bully-sizes 0 32 64 128 256 512 \
    --victim-count 4 \
    --victim-nodes 64 \
    --duration 300 \
    --force

echo "Frontier interference study complete"

#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J e2_sst
#SBATCH -o e2_sst_%j.out
#SBATCH -e e2_sst_%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1

# E2 SST-Macro Bully-Victim Interference
# 3 topologies × 6 bully_sizes = 18 runs, each ~5-10 min → total ~90 min

module load PrgEnv-gnu cray-python
cd /lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26

bash Baseline/sst-macro/multi_job/run_multi_job.sh dragonfly fattree torus3d

echo "SST-Macro multi-job sweep complete"

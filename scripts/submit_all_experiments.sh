#!/bin/bash
# Submit all optimized experiments (benchmark + use cases)
# Usage: bash scripts/submit_all_experiments.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo "Submitting All RAPS Experiments"
echo "=========================================="
echo "Time: $(date)"
echo ""

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "⚠️  Warning: You have uncommitted changes"
    echo "   Consider committing your optimized code before running experiments"
    echo ""
fi

# 1. Submit benchmark experiments (24 nodes)
echo "1. Submitting benchmark experiments..."
echo "   - 24 SLURM nodes for parallelization"
echo "   - Systems: lassen, frontier"
echo "   - Node counts: 100, 1000, 10000"
echo "   - Time quanta: 0.1s, 1s, 10s, 60s"
echo "   - Duration: 12h simulated"
echo ""

BENCH_JOB=$(sbatch submit_benchmark_parallel.slurm | awk '{print $NF}')
echo "   ✓ Benchmark job submitted: $BENCH_JOB"
echo ""

# 2. Submit use case experiments (16 nodes)
echo "2. Submitting use case experiments..."
echo "   - 16 SLURM nodes for parallelization"
echo "   - Systems: lassen, frontier"
echo "   - Node count: 1000"
echo "   - Time quantum: 10s"
echo "   - Duration: 30min simulated"
echo "   - Use cases: UC1, UC2, UC3, UC4"
echo ""

UC_JOB=$(sbatch submit_usecases_parallel.slurm | awk '{print $NF}')
echo "   ✓ Use case job submitted: $UC_JOB"
echo ""

echo "=========================================="
echo "Submission Complete!"
echo "=========================================="
echo ""
echo "Job IDs:"
echo "  - Benchmark: $BENCH_JOB"
echo "  - Use cases: $UC_JOB"
echo ""
echo "Monitor jobs:"
echo "  squeue -u $USER"
echo "  watch -n 10 'squeue -u $USER'"
echo ""
echo "Check output:"
echo "  tail -f output/frontier_scaling/benchmark-${BENCH_JOB}.out"
echo "  tail -f output/use_cases/usecases-${UC_JOB}.out"
echo ""
echo "Check progress:"
echo "  # Benchmark"
echo "  wc -l output/frontier_scaling/results.csv"
echo "  tail output/frontier_scaling/results.csv"
echo ""
echo "  # Use cases"
echo "  ls -lh output/use_cases/*/uc*.csv"
echo ""

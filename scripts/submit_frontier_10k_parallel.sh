#!/bin/bash
# Parallel submission script for frontier_n10000 experiments
# Usage: bash scripts/submit_frontier_10k_parallel.sh [8|12|24]

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

NUM_PARALLEL=${1:-8}  # Default to 8 nodes

echo "=========================================="
echo "Frontier n=10000 并行实验提交"
echo "=========================================="
echo "并行节点数: $NUM_PARALLEL"
echo "时间: $(date)"
echo ""

# 实验配置
SYSTEM="frontier"
NODE_COUNT=10000
SIM_HOURS=12

# 根据并行度选择实验组合
if [ "$NUM_PARALLEL" -eq 8 ]; then
    echo "方案: 8节点 - 优先 dt=60s"
    EXPERIMENTS=(
        "60:0"  # dt=60s, repeat=0
        "60:1"
        "60:2"
        "10:0"  # dt=10s, repeat=0
        "10:1"
        "10:2"
        "1:0"   # dt=1s, repeat=0
        "1:1"
    )
elif [ "$NUM_PARALLEL" -eq 12 ]; then
    echo "方案: 12节点 - 全部 dt 的部分 repeats"
    EXPERIMENTS=(
        "60:0"
        "60:1"
        "60:2"
        "10:0"
        "10:1"
        "10:2"
        "1:0"
        "1:1"
        "1:2"
        "0.1:0"
        "0.1:1"
        "0.1:2"
    )
elif [ "$NUM_PARALLEL" -eq 24 ]; then
    echo "方案: 24节点 - 全部实验 + 额外配置"
    EXPERIMENTS=(
        # Frontier n=10000
        "60:0" "60:1" "60:2"
        "10:0" "10:1" "10:2"
        "1:0" "1:1" "1:2"
        "0.1:0" "0.1:1" "0.1:2"
        # 额外配置或重复提交
        "60:0" "60:1" "60:2"
        "10:0" "10:1" "10:2"
        "1:0" "1:1" "1:2"
        "0.1:0" "0.1:1" "0.1:2"
    )
else
    echo "错误: NUM_PARALLEL 必须是 8, 12, 或 24"
    exit 1
fi

echo "将提交 ${#EXPERIMENTS[@]} 个实验"
echo ""

# 创建单个实验的提交脚本
cat > submit_frontier_10k_single.slurm << 'EOFSLURM'
#!/bin/bash
#SBATCH -A GEN053
#SBATCH -J f10k
#SBATCH -o output/frontier_scaling/f10k-%j.out
#SBATCH -e output/frontier_scaling/f10k-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=zhangzifan@ornl.gov
#SBATCH --signal=B:USR1@180

# Max resubmit
MAX_RESUBMIT=50
RESUBMIT_COUNT=${SLURM_RESUBMIT_COUNT:-0}

trap 'echo "Timeout signal, checkpointing..."; exit 99' SIGUSR1

echo "=========================================="
echo "Frontier n=10000 Single Experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Delta-t: ${FRONTIER_DT}s, Repeat: ${FRONTIER_REPEAT}"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

# Load environment
module load PrgEnv-gnu cray-python
source .venv/bin/activate

# Run single experiment
python -u src/run_frontier.py \
    --systems frontier \
    --nodes $NODE_COUNT \
    --dt $FRONTIER_DT \
    --repeats 1 \
    --duration $SIM_HOURS \
    --workers 1 \
    --output output/frontier_scaling

exit_code=$?

echo "Job completed: exit=$exit_code at $(date)"

# Auto-resubmit on timeout/incomplete
if [ $exit_code -eq 99 ] || [ $exit_code -ne 0 ]; then
    if [ $RESUBMIT_COUNT -lt $MAX_RESUBMIT ]; then
        echo "Resubmitting (attempt $((RESUBMIT_COUNT + 1))/$MAX_RESUBMIT)..."
        sbatch --export=ALL,SLURM_RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) $0
    fi
fi
EOFSLURM

echo "✓ 创建了 submit_frontier_10k_single.slurm"
echo ""

# 提交所有实验
SUBMITTED=0
for exp in "${EXPERIMENTS[@]}"; do
    DT=$(echo $exp | cut -d: -f1)
    REPEAT=$(echo $exp | cut -d: -f2)
    
    echo "提交 frontier_n10000_dt${DT}_r${REPEAT}..."
    
    JOB_ID=$(sbatch \
        --export=ALL,FRONTIER_DT=$DT,FRONTIER_REPEAT=$REPEAT,NODE_COUNT=10000,SIM_HOURS=12 \
        submit_frontier_10k_single.slurm | awk '{print $NF}')
    
    echo "  ✓ Job $JOB_ID"
    SUBMITTED=$((SUBMITTED + 1))
    
    # 避免过快提交
    sleep 0.5
done

echo ""
echo "=========================================="
echo "提交完成!"
echo "=========================================="
echo "总共提交: $SUBMITTED 个作业"
echo ""
echo "监控命令:"
echo "  squeue -u $USER"
echo "  watch -n 30 squeue -u $USER"
echo ""
echo "检查进度:"
echo "  tail -f output/frontier_scaling/f10k-*.out"
echo "  python src/plot_scaling_results.py"
echo ""

#!/bin/bash
# Monitor experiment progress
# Usage: bash scripts/monitor_experiments.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo "RAPS Experiment Progress Monitor"
echo "=========================================="
echo "Time: $(date)"
echo ""

# Check job queue
echo "1. SLURM Job Status:"
echo "--------------------"
squeue -u $USER -o "%.10i %.9P %.12j %.8u %.2t %.10M %.6D %R" || echo "No jobs running"
echo ""

# Check benchmark progress
echo "2. Benchmark Progress:"
echo "----------------------"
if [ -f "output/frontier_scaling/results.csv" ]; then
    TOTAL_LINES=$(wc -l < output/frontier_scaling/results.csv)
    TOTAL_EXPERIMENTS=$((TOTAL_LINES - 1))  # minus header
    echo "Completed experiments: $TOTAL_EXPERIMENTS / 72"
    
    if [ $TOTAL_EXPERIMENTS -gt 0 ]; then
        echo ""
        echo "Last 5 completed experiments:"
        tail -5 output/frontier_scaling/results.csv | column -t -s','
        echo ""
        
        # Calculate completion percentage
        PERCENT=$((TOTAL_EXPERIMENTS * 100 / 72))
        echo "Progress: $PERCENT%"
        
        # Show breakdown by configuration
        echo ""
        echo "Breakdown by system:"
        awk -F',' 'NR>1 {print $1}' output/frontier_scaling/results.csv | sort | uniq -c
        
        echo ""
        echo "Breakdown by node count:"
        awk -F',' 'NR>1 {print $2}' output/frontier_scaling/results.csv | sort | uniq -c
        
        echo ""
        echo "Breakdown by time quantum:"
        awk -F',' 'NR>1 {print $3}' output/frontier_scaling/results.csv | sort | uniq -c
    fi
else
    echo "No results yet (results.csv not found)"
fi
echo ""

# Check use case progress
echo "3. Use Case Progress:"
echo "---------------------"
for SYSTEM in lassen frontier; do
    echo "$SYSTEM:"
    for UC in 1 2 3 4; do
        CSV_FILE="output/use_cases/${SYSTEM}_n1000/uc${UC}_results.csv"
        if [ -f "$CSV_FILE" ]; then
            LINES=$(wc -l < "$CSV_FILE")
            SIZE=$(du -h "$CSV_FILE" | cut -f1)
            echo "  ✓ UC${UC}: $LINES lines, $SIZE"
        else
            echo "  ⏳ UC${UC}: Not started yet"
        fi
    done
done
echo ""

# Check latest log files
echo "4. Latest Log Output:"
echo "---------------------"
echo "Benchmark logs:"
LATEST_BENCH_LOG=$(ls -t output/frontier_scaling/benchmark-*.out 2>/dev/null | head -1)
if [ -n "$LATEST_BENCH_LOG" ]; then
    echo "  File: $LATEST_BENCH_LOG"
    echo "  Last 10 lines:"
    tail -10 "$LATEST_BENCH_LOG" | sed 's/^/    /'
else
    echo "  No benchmark logs yet"
fi
echo ""

echo "Use case logs:"
LATEST_UC_LOG=$(ls -t output/use_cases/usecases-*.out 2>/dev/null | head -1)
if [ -n "$LATEST_UC_LOG" ]; then
    echo "  File: $LATEST_UC_LOG"
    echo "  Last 10 lines:"
    tail -10 "$LATEST_UC_LOG" | sed 's/^/    /'
else
    echo "  No use case logs yet"
fi
echo ""

# Estimate completion time
echo "5. Time Estimates:"
echo "------------------"
if [ -f "output/frontier_scaling/results.csv" ] && [ $TOTAL_EXPERIMENTS -gt 0 ]; then
    # Get job start time from log
    if [ -n "$LATEST_BENCH_LOG" ]; then
        START_TIME=$(grep "Start time:" "$LATEST_BENCH_LOG" | head -1 | cut -d: -f2- | xargs)
        if [ -n "$START_TIME" ]; then
            START_EPOCH=$(date -d "$START_TIME" +%s 2>/dev/null || echo "")
            if [ -n "$START_EPOCH" ]; then
                NOW_EPOCH=$(date +%s)
                ELAPSED_SEC=$((NOW_EPOCH - START_EPOCH))
                ELAPSED_MIN=$((ELAPSED_SEC / 60))
                
                echo "Elapsed time: ${ELAPSED_MIN} minutes"
                
                if [ $TOTAL_EXPERIMENTS -gt 0 ]; then
                    AVG_SEC_PER_EXP=$((ELAPSED_SEC / TOTAL_EXPERIMENTS))
                    REMAINING_EXP=$((72 - TOTAL_EXPERIMENTS))
                    ESTIMATED_REMAINING_SEC=$((AVG_SEC_PER_EXP * REMAINING_EXP / 24))  # divided by parallelism
                    ESTIMATED_REMAINING_MIN=$((ESTIMATED_REMAINING_SEC / 60))
                    
                    echo "Average time per experiment: $((AVG_SEC_PER_EXP / 60)) minutes"
                    echo "Estimated remaining time: ${ESTIMATED_REMAINING_MIN} minutes"
                    
                    # Calculate ETA
                    ETA_EPOCH=$((NOW_EPOCH + ESTIMATED_REMAINING_SEC))
                    ETA=$(date -d "@$ETA_EPOCH" "+%Y-%m-%d %H:%M:%S")
                    echo "Estimated completion: $ETA"
                fi
            fi
        fi
    fi
else
    echo "Not enough data yet for estimates"
fi
echo ""

echo "=========================================="
echo "Refresh this view with:"
echo "  bash scripts/monitor_experiments.sh"
echo ""
echo "Or watch continuously:"
echo "  watch -n 30 bash scripts/monitor_experiments.sh"
echo "=========================================="

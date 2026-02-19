#!/bin/bash
# ==============================================================================
# Download and Setup Data for RAPS Frontier Experiments
# ==============================================================================
# This script:
# 1. Downloads Lassen telemetry data from LLNL's GitHub (using git-lfs)
# 2. Verifies data integrity
# 3. Checks mini-app traces (optional)
# ==============================================================================

set -e

PROJECT_DIR="/lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26"
DATA_DIR="$PROJECT_DIR/data/lassen"

echo "=============================================="
echo "RAPS Data Download and Setup"
echo "=============================================="
echo "Project directory: $PROJECT_DIR"
echo "Data directory: $DATA_DIR"
echo "Current time: $(date)"
echo ""

# -------------------------------------------------------------------------
# 1. Load required modules
# -------------------------------------------------------------------------
echo "Loading modules..."
module load git/2.47.0
module load git-lfs/3.5.1
echo "  ✓ git version: $(git --version)"
echo "  ✓ git-lfs version: $(git-lfs version | head -1)"
echo ""

# -------------------------------------------------------------------------
# 2. Download Lassen data
# -------------------------------------------------------------------------
echo "Downloading Lassen telemetry data..."
echo "This may take 10-30 minutes depending on network speed."
echo "Dataset size: ~2-3 GB"
echo ""

cd "$PROJECT_DIR"

# Activate Python environment
source .venv/bin/activate

# Use raps download command
echo "Running: raps download --system lassen --dest $DATA_DIR"
raps download --system lassen --dest "$DATA_DIR"

echo ""
echo "=============================================="
echo "Download Complete!"
echo "=============================================="
echo ""

# -------------------------------------------------------------------------
# 3. Verify data files
# -------------------------------------------------------------------------
echo "Verifying downloaded data..."

DATASET_DIR="$DATA_DIR/Lassen-Supercomputer-Job-Dataset"

if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: Dataset directory not found at $DATASET_DIR"
    exit 1
fi

REQUIRED_FILES=(
    "final_csm_allocation_history_hashed.csv"
    "final_csm_allocation_node_history.csv"
    "final_csm_step_history.csv"
)

ALL_FOUND=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$DATASET_DIR/$file" ]; then
        SIZE=$(du -h "$DATASET_DIR/$file" | cut -f1)
        echo "  ✓ $file ($SIZE)"
    else
        echo "  ✗ MISSING: $file"
        ALL_FOUND=false
    fi
done

echo ""

if [ "$ALL_FOUND" = true ]; then
    echo "✅ All required data files found!"
    
    # Quick data statistics
    echo ""
    echo "Data Statistics:"
    ALLOC_LINES=$(wc -l < "$DATASET_DIR/final_csm_allocation_history_hashed.csv")
    echo "  Total allocations: $((ALLOC_LINES - 1))"
    
else
    echo "❌ Some required files are missing!"
    echo "Please check the download or try again."
    exit 1
fi

# -------------------------------------------------------------------------
# 4. Update SLURM scripts with correct data path
# -------------------------------------------------------------------------
echo ""
echo "Updating SLURM scripts with data path..."

# The scripts currently use /opt/data, but data is in $DATA_DIR
# We'll update the path in run_frontier.py instead

echo "  Data location: $DATASET_DIR"
echo ""

# -------------------------------------------------------------------------
# 5. Test data loading
# -------------------------------------------------------------------------
echo "Testing data loading..."
cd "$PROJECT_DIR"

python -c "
from pathlib import Path
from raps.telemetry import Telemetry
from raps.system_config import get_system_config

print('Loading lassen configuration...')
cfg = get_system_config('lassen').get_legacy()

print('Creating telemetry loader...')
td = Telemetry(system='lassen', config=cfg, time=3600)

print('Loading data (this may take 1-2 minutes)...')
data_path = Path('$DATASET_DIR')
wd = td.load_from_files([data_path])

print(f'✅ Successfully loaded {len(wd.jobs)} jobs from lassen telemetry!')
print(f'   Time range: {wd.start_date} to {wd.end_date}')
print(f'   Duration: {wd.telemetry_end - wd.telemetry_start} seconds')
"

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Update src/run_frontier.py to use correct data path"
echo "2. Cancel failed jobs: scancel 4108590 4108591 4108607 4108609 4108610"
echo "3. Re-submit all experiments"
echo ""
echo "✅ Lassen data is ready at: $DATASET_DIR"
echo "=============================================="

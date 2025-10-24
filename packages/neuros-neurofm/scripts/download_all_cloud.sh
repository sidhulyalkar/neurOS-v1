#!/bin/bash

################################################################################
# Cloud Data Download Orchestration Script for NeuroFMx
#
# This script orchestrates parallel downloads of all neural datasets on cloud
# servers (CoreWeave/Crusoe/Lambda Labs H100 instances).
#
# Usage:
#   bash download_all_cloud.sh [OPTIONS]
#
# Options:
#   --data-dir DIR        Base directory for data (default: ./data)
#   --parallel N          Number of parallel downloads (default: 4)
#   --modalities LIST     Comma-separated modalities to download
#                         Options: all,ibl,allen,eeg,fmri,ecog,emg,lfp
#                         (default: all)
#   --skip-existing       Skip datasets that already exist
#   --log-dir DIR         Directory for logs (default: ./logs/downloads)
#
# Example:
#   bash download_all_cloud.sh --data-dir /mnt/data --parallel 6 --modalities all
################################################################################

set -e  # Exit on error

# Default configuration
DATA_DIR="./data"
PARALLEL=4
MODALITIES="all"
SKIP_EXISTING=false
LOG_DIR="./logs/downloads"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --modalities)
            MODALITIES="$2"
            shift 2
            ;;
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create directories
mkdir -p "$DATA_DIR"
mkdir -p "$LOG_DIR"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}NeuroFMx Cloud Data Download${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Parallel downloads: $PARALLEL"
echo "  Modalities: $MODALITIES"
echo "  Skip existing: $SKIP_EXISTING"
echo "  Log directory: $LOG_DIR"
echo ""

# Check Python and required packages
echo -e "${YELLOW}Checking dependencies...${NC}"

if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

# Check pip packages
REQUIRED_PACKAGES=(
    "numpy"
    "scipy"
    "tqdm"
)

MISSING_PACKAGES=()

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if ! python -c "import $pkg" 2>/dev/null; then
        MISSING_PACKAGES+=("$pkg")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
    echo -e "${YELLOW}Installing missing packages: ${MISSING_PACKAGES[*]}${NC}"
    pip install "${MISSING_PACKAGES[@]}"
fi

echo -e "${GREEN}Dependencies OK${NC}"
echo ""

# Define download jobs
declare -A DOWNLOAD_JOBS

if [[ "$MODALITIES" == "all" ]] || [[ "$MODALITIES" == *"ibl"* ]]; then
    DOWNLOAD_JOBS["ibl"]="python scripts/data_acquisition/download_ibl.py --output_dir $DATA_DIR/spike/processed --cache_dir $DATA_DIR/ibl_cache"
fi

if [[ "$MODALITIES" == "all" ]] || [[ "$MODALITIES" == *"allen"* ]]; then
    DOWNLOAD_JOBS["allen"]="python scripts/data_acquisition/download_allen_2p.py --output_dir $DATA_DIR/calcium/processed --cache_dir $DATA_DIR/allen_cache"
fi

if [[ "$MODALITIES" == "all" ]] || [[ "$MODALITIES" == *"eeg"* ]]; then
    DOWNLOAD_JOBS["eeg"]="python scripts/data_acquisition/download_eeg.py --output_dir $DATA_DIR/eeg/processed --cache_dir $DATA_DIR/eeg_cache"
fi

if [[ "$MODALITIES" == "all" ]] || [[ "$MODALITIES" == *"fmri"* ]]; then
    DOWNLOAD_JOBS["fmri"]="python scripts/data_acquisition/download_fmri.py --output_dir $DATA_DIR/fmri/processed --cache_dir $DATA_DIR/fmri_cache"
fi

if [[ "$MODALITIES" == "all" ]] || [[ "$MODALITIES" == *"ecog"* ]]; then
    DOWNLOAD_JOBS["ecog"]="python scripts/data_acquisition/download_ecog.py --output_dir $DATA_DIR/ecog/processed --cache_dir $DATA_DIR/ecog_cache"
fi

if [[ "$MODALITIES" == "all" ]] || [[ "$MODALITIES" == *"emg"* ]]; then
    DOWNLOAD_JOBS["emg"]="python scripts/data_acquisition/download_emg.py --output_dir $DATA_DIR/emg/processed --cache_dir $DATA_DIR/emg_cache"
fi

if [[ "$MODALITIES" == "all" ]] || [[ "$MODALITIES" == *"lfp"* ]]; then
    DOWNLOAD_JOBS["lfp"]="python scripts/data_acquisition/download_lfp_ieeg.py --output_dir $DATA_DIR/lfp/processed --cache_dir $DATA_DIR/lfp_cache --source allen"
fi

echo -e "${BLUE}Download jobs configured: ${#DOWNLOAD_JOBS[@]}${NC}"
echo ""

# Function to run download job
run_download() {
    local name=$1
    local cmd=$2
    local log_file="$LOG_DIR/${name}_$(date +%Y%m%d_%H%M%S).log"

    echo -e "${YELLOW}Starting $name download...${NC}"
    echo "  Command: $cmd"
    echo "  Log: $log_file"

    if $cmd > "$log_file" 2>&1; then
        echo -e "${GREEN}✓ $name download completed${NC}"
        return 0
    else
        echo -e "${RED}✗ $name download failed (see $log_file)${NC}"
        return 1
    fi
}

export -f run_download
export DATA_DIR LOG_DIR YELLOW GREEN RED NC

# Check for GNU parallel
if command -v parallel &> /dev/null; then
    echo -e "${BLUE}Using GNU parallel for downloads${NC}"
    echo ""

    # Create job list
    JOB_LIST="$LOG_DIR/job_list.txt"
    > "$JOB_LIST"

    for name in "${!DOWNLOAD_JOBS[@]}"; do
        echo "$name|${DOWNLOAD_JOBS[$name]}" >> "$JOB_LIST"
    done

    # Run with GNU parallel
    cat "$JOB_LIST" | parallel --colsep '|' -j "$PARALLEL" run_download {1} {2}

else
    echo -e "${YELLOW}GNU parallel not found, running sequentially${NC}"
    echo -e "${YELLOW}Install with: sudo apt-get install parallel (Ubuntu) or brew install parallel (macOS)${NC}"
    echo ""

    # Sequential execution
    for name in "${!DOWNLOAD_JOBS[@]}"; do
        run_download "$name" "${DOWNLOAD_JOBS[$name]}"
    done
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Download Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check which datasets were successfully downloaded
SUCCESS_COUNT=0
FAIL_COUNT=0

for name in "${!DOWNLOAD_JOBS[@]}"; do
    # Check if processed data exists
    case $name in
        ibl)
            CHECK_DIR="$DATA_DIR/spike/processed"
            ;;
        allen)
            CHECK_DIR="$DATA_DIR/calcium/processed"
            ;;
        eeg)
            CHECK_DIR="$DATA_DIR/eeg/processed"
            ;;
        fmri)
            CHECK_DIR="$DATA_DIR/fmri/processed"
            ;;
        ecog)
            CHECK_DIR="$DATA_DIR/ecog/processed"
            ;;
        emg)
            CHECK_DIR="$DATA_DIR/emg/processed"
            ;;
        lfp)
            CHECK_DIR="$DATA_DIR/lfp/processed"
            ;;
    esac

    if [ -d "$CHECK_DIR" ] && [ "$(ls -A $CHECK_DIR 2>/dev/null)" ]; then
        TRAIN_COUNT=$(ls "$CHECK_DIR/train"/*.npz 2>/dev/null | wc -l)
        echo -e "${GREEN}✓ $name: $TRAIN_COUNT training sequences${NC}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo -e "${RED}✗ $name: No data found${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

echo ""
echo "Total: $SUCCESS_COUNT succeeded, $FAIL_COUNT failed"

# Disk usage summary
echo ""
echo -e "${BLUE}Disk Usage:${NC}"
du -sh "$DATA_DIR"/* 2>/dev/null | sort -h

echo ""
echo -e "${GREEN}Download orchestration complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Verify data quality: ls -lh $DATA_DIR/*/processed/train/ | head"
echo "  2. Check logs for errors: tail $LOG_DIR/*.log"
echo "  3. Start training: python training/train_multimodal.py --config configs/model_small.yaml --data_dir $DATA_DIR"
echo ""

# Generate data manifest
MANIFEST_FILE="$DATA_DIR/data_manifest.txt"
echo "Generating data manifest: $MANIFEST_FILE"

{
    echo "NeuroFMx Data Manifest"
    echo "Generated: $(date)"
    echo ""
    echo "Data Directory: $DATA_DIR"
    echo ""

    for name in "${!DOWNLOAD_JOBS[@]}"; do
        case $name in
            ibl)
                CHECK_DIR="$DATA_DIR/spike/processed"
                ;;
            allen)
                CHECK_DIR="$DATA_DIR/calcium/processed"
                ;;
            eeg)
                CHECK_DIR="$DATA_DIR/eeg/processed"
                ;;
            fmri)
                CHECK_DIR="$DATA_DIR/fmri/processed"
                ;;
            ecog)
                CHECK_DIR="$DATA_DIR/ecog/processed"
                ;;
            emg)
                CHECK_DIR="$DATA_DIR/emg/processed"
                ;;
            lfp)
                CHECK_DIR="$DATA_DIR/lfp/processed"
                ;;
        esac

        if [ -d "$CHECK_DIR" ]; then
            echo "[$name]"
            echo "  Train: $(ls $CHECK_DIR/train/*.npz 2>/dev/null | wc -l) files"
            echo "  Val: $(ls $CHECK_DIR/val/*.npz 2>/dev/null | wc -l) files"
            echo "  Test: $(ls $CHECK_DIR/test/*.npz 2>/dev/null | wc -l) files"
            echo "  Size: $(du -sh $CHECK_DIR 2>/dev/null | cut -f1)"
            echo ""
        fi
    done
} > "$MANIFEST_FILE"

echo -e "${GREEN}Manifest saved to: $MANIFEST_FILE${NC}"

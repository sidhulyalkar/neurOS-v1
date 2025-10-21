#!/bin/bash
# NeuroFM-X Setup and Training Script for Linux/Mac/Git Bash
# This script creates a conda environment, downloads data, and trains the model

echo "================================================================================"
echo "NeuroFM-X Complete Setup and Training"
echo "================================================================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: Conda not found! Please install Anaconda or Miniconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "[1/4] Creating conda environment..."
echo ""
conda env create -f environment.yml -y
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create environment. See error above."
    exit 1
fi

echo ""
echo "[2/4] Activating environment..."
echo ""
# Source conda for bash
eval "$(conda shell.bash hook)"
conda activate neurofm
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate environment."
    echo "Try running manually: conda activate neurofm"
    exit 1
fi

echo ""
echo "[3/4] Downloading Allen Brain Observatory data..."
echo "This will download ~5-10 GB of data and may take 15-30 minutes."
echo ""
python download_allen_data.py --num-sessions 5
if [ $? -ne 0 ]; then
    echo "WARNING: Data download failed. You can try running it manually later:"
    echo "  conda activate neurofm"
    echo "  python download_allen_data.py --num-sessions 5"
    echo ""
    echo "Continuing anyway in case data already exists..."
fi

echo ""
echo "[4/4] Starting model training..."
echo "This will take approximately 2-4 hours on RTX 3070 Ti."
echo ""
python train_allen_data.py
if [ $? -ne 0 ]; then
    echo "ERROR: Training failed. Check error above."
    exit 1
fi

echo ""
echo "================================================================================"
echo "ALL DONE!"
echo "================================================================================"
echo ""
echo "Trained model saved to: checkpoints_allen/best.pt"
echo ""

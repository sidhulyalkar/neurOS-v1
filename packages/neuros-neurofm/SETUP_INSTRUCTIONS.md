# Complete Setup Instructions for NeuroFM-X Training

Follow these steps to set up a complete conda environment and download the Allen dataset.

## Option 1: Conda Environment Setup (RECOMMENDED)

This is the cleanest approach that avoids Python 3.14 compatibility issues.

### Step 1: Create Conda Environment

```bash
cd packages/neuros-neurofm

# Create environment from YAML file
conda env create -f environment.yml

# Activate the environment
conda activate neurofm
```

### Step 2: Verify Installation

```bash
# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Expected output:
# PyTorch: 2.x.x
# CUDA available: True
# GPU: NVIDIA GeForce RTX 3070 Ti
```

### Step 3: Download Allen Data

```bash
# Download 5 sessions (~5-10 GB, takes 15-30 minutes)
python download_allen_data.py --num-sessions 5

# Or download more for better performance
python download_allen_data.py --num-sessions 10
```

### Step 4: Train the Model

```bash
# Start training
python train_allen_data.py
```

---

## Option 2: Manual Pip Installation (If Conda Not Available)

If you can't use conda, follow these manual steps:

### Step 1: Create Virtual Environment

```bash
# Create virtual environment with Python 3.10 (NOT 3.14!)
python3.10 -m venv venv_neurofm

# Activate it
# On Windows:
venv_neurofm\Scripts\activate
# On Linux/Mac:
source venv_neurofm/bin/activate
```

### Step 2: Install PyTorch with CUDA

```bash
# For CUDA 11.8 (RTX 3070 Ti)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Install Core Dependencies

```bash
pip install numpy scipy pandas h5py tqdm
```

### Step 4: Install Allen SDK and Dependencies

```bash
# Install Allen SDK with specific versions that work
pip install allensdk==2.14.1
pip install SimpleITK
pip install xarray nest-asyncio
pip install psycopg2-binary
pip install pynwb
```

### Step 5: Download Data and Train

```bash
# Download data
python download_allen_data.py --num-sessions 5

# Train model
python train_allen_data.py
```

---

## Option 3: Quick Start (Single Command)

Create this batch file (`setup_and_train.bat` for Windows or `setup_and_train.sh` for Linux):

```bash
#!/bin/bash
# setup_and_train.sh

# Create conda environment
conda env create -f environment.yml -y

# Activate environment
conda activate neurofm

# Download data (5 sessions)
python download_allen_data.py --num-sessions 5

# Train model
python train_allen_data.py
```

Then run:
```bash
chmod +x setup_and_train.sh
./setup_and_train.sh
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'SimpleITK'"

```bash
pip install SimpleITK
```

### Issue: "CUDA not available"

1. Check NVIDIA drivers are installed: `nvidia-smi`
2. Reinstall PyTorch with correct CUDA version
3. Visit https://pytorch.org/get-started/locally/ for correct install command

### Issue: Python 3.14 compatibility problems

Allen SDK doesn't fully support Python 3.14 yet. Use Python 3.10 instead:

```bash
conda create -n neurofm python=3.10
conda activate neurofm
# Then install packages...
```

### Issue: Download fails or is very slow

The Allen data is large (~2-3 GB per session). If download fails:
1. Check internet connection
2. Reduce number of sessions: `--num-sessions 3`
3. The script automatically skips failed sessions
4. Re-run the script - it will use cached data and continue

### Issue: Out of memory during training

Reduce batch size in `train_allen_data.py`:
```bash
python train_allen_data.py --batch-size 2
```

---

## Quick Reference Commands

```bash
# Create environment
conda env create -f environment.yml

# Activate
conda activate neurofm

# Download data (5 sessions, ~5-10 GB)
python download_allen_data.py --num-sessions 5

# Train (will take 2-4 hours)
python train_allen_data.py

# Check training checkpoints
ls checkpoints_allen/

# Load trained model in Python
python
>>> import torch
>>> checkpoint = torch.load('checkpoints_allen/best.pt')
>>> print(f"Best loss: {checkpoint['best_val_loss']}")
```

---

## What Gets Downloaded

When you run `download_allen_data.py`, you'll get:

- **Location:** `./data/allen_neuropixels/cache/`
- **Size:** ~2-3 GB per session
- **Data:**
  - Neuropixels spike times from visual cortex
  - Unit metadata (brain area, quality, depth)
  - Stimulus presentation times
  - Behavioral data (running, pupil)

## Training Output

Training will create:

- **Checkpoints:** `./checkpoints_allen/best.pt` and `latest.pt`
- **Logs:** `./logs_allen/`
- **Progress:** Real-time loss and metrics in terminal

Expected training time on RTX 3070 Ti:
- 5 sessions: ~2-3 hours for 50 epochs
- 10 sessions: ~4-6 hours for 50 epochs

---

## Need Help?

1. Make sure you're using **Python 3.10** (not 3.14!)
2. Make sure CUDA is working: `nvidia-smi`
3. Check the TRAINING_GUIDE_RTX3070TI.md for detailed info
4. All scripts have `--help` flags:
   ```bash
   python download_allen_data.py --help
   python train_allen_data.py --help
   ```

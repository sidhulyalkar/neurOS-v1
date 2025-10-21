# NeuroFM-X Quick Start Guide

Since you're using Git Bash on Windows with conda already installed, here's the fastest way to get started:

## ✅ One-Command Setup

```bash
cd packages/neuros-neurofm
./setup_and_train.sh
```

This will automatically:
1. Create conda environment with Python 3.10
2. Install all dependencies (PyTorch + CUDA, Allen SDK, etc.)
3. Download Allen Brain Observatory data (~5-10 GB)
4. Train NeuroFM-X model (~2-4 hours)

---

## 🔧 Step-by-Step Manual Setup (If Preferred)

If you want more control, follow these steps:

### Step 1: Create the conda environment

```bash
cd packages/neuros-neurofm
conda env create -f environment.yml
```

This creates a new environment called `neurofm` with Python 3.10 and all dependencies.

### Step 2: Activate the environment

```bash
conda activate neurofm
```

You should see `(neurofm)` appear in your terminal prompt.

### Step 3: Verify GPU access

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
CUDA: True
GPU: NVIDIA GeForce RTX 3070 Ti
```

### Step 4: Download Allen data

```bash
# Download 5 sessions (~5-10 GB, takes 15-30 minutes)
python download_allen_data.py --num-sessions 5

# Options:
# --num-sessions: How many sessions to download (default: 10)
# --stimulus-type: natural_images, drifting_gratings, or all
# --brain-area: VISp, VISl, VISal, or all
```

The script will show progress:
```
[1/5] Checking dependencies...
  ✓ All dependencies installed

[2/5] Initializing Allen SDK cache...
  Cache directory: ./data/allen_neuropixels/cache
  ✓ Cache initialized

[3/5] Fetching session metadata...
  ✓ Found XX total sessions in database

[4/5] Filtering sessions...
  ✓ Selected 5 sessions to download

[5/5] Downloading sessions...
  [1/5] Downloading session 12345...
    ✓ Success!
      - Units: 456
      - Duration: 2500.0 seconds
      - Brain areas: VISp, VISl
  ...
```

### Step 5: Train the model

```bash
python train_allen_data.py

# Options:
# --batch-size: Batch size (default: 4 for RTX 3070 Ti)
# --max-epochs: Number of epochs (default: 50)
# --learning-rate: Learning rate (default: 3e-4)
```

Training will show progress:
```
================================================================================
NeuroFM-X Training on Allen Brain Observatory Data
================================================================================

GPU: NVIDIA GeForce RTX 3070 Ti
VRAM: 8.00 GB

Loading Allen Neuropixels Dataset
================================================================================
Sessions to process: 5
Processing sessions...
Loading sessions: 100%|████████| 5/5 [00:15<00:00]

✓ Dataset created with 12,345 training sequences
================================================================================

================================================================================
NeuroFM-X Trainer
================================================================================
Parameters: 15,234,567
Device: cuda
Batch size: 4
Effective batch size: 32
Training batches: 625
Validation batches: 110
================================================================================

Epoch 1/50 [TRAIN]: 100%|████| 625/625 [03:24<00:00, loss=0.0234, lr=6.00e-05]
Epoch 1/50 [VAL]:   100%|████| 110/110 [00:28<00:00]

────────────────────────────────────────────────────────────────────────────────
Epoch 1/50:
  Train Loss: 0.023456
  Val Loss:   0.021234
  LR: 6.00e-05
  ✓ Saved best checkpoint (val_loss: 0.021234)
```

### Step 6: Use the trained model

After training completes:

```bash
python
```

```python
import torch
from train_allen_data import NeuroFMX, Config

# Load trained model
config = Config()
model = NeuroFMX(config)

checkpoint = torch.load('checkpoints_allen/best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.cuda()
model.eval()

print(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")
print(f"Trained for {checkpoint['epoch']} epochs")

# Use for inference
import numpy as np

# Create dummy neural data (replace with your real data)
# Shape: (batch=1, sequence_length=100, num_units=384)
dummy_spikes = torch.randn(1, 100, 384).cuda()

with torch.no_grad():
    latents, reconstructed = model(dummy_spikes)
    print(f"Latent shape: {latents.shape}")  # (1, 64, 256)
    print(f"Reconstructed shape: {reconstructed.shape}")  # (1, 100, 384)
```

---

## 📊 Expected Results

After training on 5 sessions for 50 epochs:

- **Validation Loss:** 0.015 - 0.025
- **Training Time:** 2-4 hours on RTX 3070 Ti
- **Model Size:** ~60 MB (FP32)
- **Dataset Size:** ~5-10 GB

---

## 🐛 Troubleshooting

### Issue: "conda: command not found" in Git Bash

Your conda is installed but not in Git Bash's PATH. Try:

```bash
# Option 1: Use Windows-style path
/c/Users/$USER/anaconda3/Scripts/conda env create -f environment.yml

# Option 2: Initialize conda for Git Bash
conda init bash
# Then restart Git Bash
```

### Issue: Environment already exists

```bash
# Remove old environment and recreate
conda env remove -n neurofm
conda env create -f environment.yml
```

### Issue: Download is slow or fails

- The Allen data is large (2-3 GB per session)
- Reduce sessions: `python download_allen_data.py --num-sessions 3`
- If it fails partway, just re-run - it will skip downloaded sessions

### Issue: Out of memory during training

```bash
# Reduce batch size
python train_allen_data.py --batch-size 2
```

### Issue: CUDA not available

1. Check drivers: Open Windows CMD and run `nvidia-smi`
2. The environment.yml includes CUDA-enabled PyTorch
3. If still not working, manually install PyTorch with CUDA:
   ```bash
   conda activate neurofm
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

---

## 📂 Files You'll Get

After setup:

```
packages/neuros-neurofm/
├── data/
│   └── allen_neuropixels/
│       ├── cache/              # Downloaded NWB files (~5-10 GB)
│       │   ├── manifest.json
│       │   └── session_XXXXX/
│       └── dataset_info.txt    # Summary of downloaded sessions
│
└── checkpoints_allen/
    ├── best.pt                 # Best model (lowest val loss)
    └── latest.pt              # Most recent checkpoint
```

---

## ⚡ Quick Reference

```bash
# Create environment
conda env create -f environment.yml

# Activate environment (do this every time you open a new terminal)
conda activate neurofm

# Download data (only need to do once)
python download_allen_data.py --num-sessions 5

# Train model
python train_allen_data.py

# Check checkpoints
ls -lh checkpoints_allen/

# Deactivate environment when done
conda deactivate
```

---

## 🎯 Next Steps

1. **Run `./setup_and_train.sh`** or follow the manual steps above
2. **Monitor training** - watch the validation loss decrease
3. **Experiment** - try different hyperparameters, more sessions, etc.
4. **Use the model** - load checkpoints for inference on new data

Have fun training your foundation model! 🚀

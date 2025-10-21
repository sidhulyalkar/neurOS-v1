# NeuroFM-X Training Guide for RTX 3070 Ti

Complete guide for training NeuroFM-X foundation model on your RTX 3070 Ti GPU using real Allen Brain Observatory data.

## 🎯 Overview

This guide will help you:
1. Install required dependencies
2. Download Allen Brain Observatory Neuropixels dataset (~20-50 GB)
3. Train NeuroFM-X foundation model on real neural data
4. Evaluate the trained model

**Hardware Requirements:**
- GPU: NVIDIA RTX 3070 Ti (8GB VRAM) or better
- RAM: 16GB+ recommended
- Storage: ~50GB free (dataset + checkpoints)
- OS: Windows/Linux with CUDA support

## 📦 Step 1: Install Dependencies

### Option A: Using pip (Recommended)

```bash
# Navigate to the package directory
cd packages/neuros-neurofm

# Install core dependencies
pip install numpy scipy pandas tqdm

# Install PyTorch with CUDA support (for RTX 3070 Ti with CUDA 11.8)
# Visit https://pytorch.org/get-started/locally/ for the correct command for your system
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Allen SDK for data download
pip install allensdk
```

### Option B: Using conda (Alternative)

```bash
# Create new environment
conda create -n neurofm python=3.10
conda activate neurofm

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install numpy scipy pandas tqdm allensdk
```

### Verify Installation

```bash
# Check if PyTorch can see your GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3070 Ti
```

## 📥 Step 2: Download Allen Brain Observatory Data

Run the data download script to fetch real Neuropixels recordings:

```bash
# Download 10 sessions (recommended for initial training, ~10-20 GB)
python download_allen_data.py --num-sessions 10

# Or download more sessions for better model performance (~30-50 GB)
python download_allen_data.py --num-sessions 20

# Or download with specific filters
python download_allen_data.py \
    --num-sessions 15 \
    --stimulus-type natural_images \
    --brain-area VISp
```

### Download Options:

- `--data-dir`: Directory to store data (default: `./data/allen_neuropixels`)
- `--num-sessions`: Number of sessions to download (default: 10, max: ~60)
- `--stimulus-type`: Filter by stimulus (`natural_images`, `drifting_gratings`, or `all`)
- `--brain-area`: Filter by brain area (e.g., `VISp`, `VISl`, or `all`)

### What Gets Downloaded:

The script downloads:
- Neuropixels spike data from visual cortex
- Stimulus presentation times
- Unit metadata (brain area, depth, quality)
- Behavioral data (running speed, pupil diameter)

**Download time:** 30 minutes to 2 hours depending on internet speed and number of sessions.

**Storage:** ~1-3 GB per session

### Verify Download:

After download completes, check the dataset info:

```bash
cat data/allen_neuropixels/dataset_info.txt
```

## 🚀 Step 3: Train NeuroFM-X Model

Now train the model on the downloaded data:

```bash
# Train with default settings (optimized for RTX 3070 Ti)
python train_allen_data.py

# Or customize training parameters
python train_allen_data.py \
    --batch-size 4 \
    --max-epochs 50 \
    --learning-rate 3e-4
```

### Training Options:

- `--data-dir`: Path to Allen data (default: `./data/allen_neuropixels`)
- `--batch-size`: Batch size (default: 4 for 8GB VRAM)
- `--max-epochs`: Number of training epochs (default: 50)
- `--learning-rate`: Learning rate (default: 3e-4)

### Training Configuration (Optimized for RTX 3070 Ti):

```python
Model Architecture:
- d_model: 256
- Transformer blocks: 8
- Latent dimensions: 64 x 256
- Total parameters: ~15M

Training Settings:
- Batch size: 4
- Gradient accumulation: 8 steps (effective batch size: 32)
- Mixed precision (FP16): Enabled
- Sequence length: 100 time bins (1 second at 10ms resolution)
- Max units: 384 neurons per sequence
```

### Training Time:

- **10 sessions:** ~2-4 hours for 50 epochs
- **20 sessions:** ~4-8 hours for 50 epochs

### Monitor Training:

The training script will display:
```
Epoch 1/50 [TRAIN]: 100%|████| 625/625 [03:24<00:00, loss=0.0234, lr=6.00e-05]
Epoch 1/50 [VAL]:   100%|████| 110/110 [00:28<00:00]

────────────────────────────────────────────────────────────────────────────────
Epoch 1/50:
  Train Loss: 0.023456
  Val Loss:   0.021234
  LR: 6.00e-05
  ✓ Saved best checkpoint (val_loss: 0.021234)
```

### Checkpoints:

Checkpoints are saved to `./checkpoints_allen/`:
- `best.pt`: Best model (lowest validation loss)
- `latest.pt`: Most recent model

## 📊 Step 4: Evaluate the Model

After training completes, you can load and use the model:

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

# Use model for inference
with torch.no_grad():
    # spikes: (batch, seq_len, n_units) - your neural data
    latents, reconstructed = model(spikes)

    # latents: (batch, 64, 256) - learned neural representations
    # reconstructed: (batch, seq_len, n_units) - reconstructed activity
```

## 🎛️ Advanced: Tuning for Your GPU

### If you have more VRAM (12GB+):

```python
# Increase batch size and model capacity
python train_allen_data.py \
    --batch-size 8 \
    --learning-rate 5e-4
```

### If you have less VRAM (6GB):

```python
# Reduce batch size
python train_allen_data.py \
    --batch-size 2 \
    --learning-rate 2e-4
```

## 📈 Expected Performance

After training on 10-20 sessions for 50 epochs:

| Metric | Expected Value |
|--------|---------------|
| Validation Loss | 0.015 - 0.025 |
| Reconstruction R² | 0.60 - 0.75 |
| Training Time | 2-8 hours |
| Model Size | ~60 MB (FP32) |

## 🐛 Troubleshooting

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size: `--batch-size 2`
2. Reduce max units in `train_allen_data.py`: `max_units = 256`
3. Enable gradient checkpointing (edit model code)

### Download Fails

```
Failed to download session XXXXX
```

**Solutions:**
1. Check internet connection
2. The script automatically skips failed sessions
3. Re-run the download script - it will use cached data

### Slow Training

**Solutions:**
1. Verify GPU is being used: Check "Device: cuda" in training output
2. Update NVIDIA drivers
3. Reduce number of workers if CPU is bottleneck

### Allen SDK Import Error

```
ModuleNotFoundError: No module named 'allensdk'
```

**Solution:**
```bash
pip install allensdk
```

## 📚 File Structure

After setup, your directory will look like:

```
packages/neuros-neurofm/
├── download_allen_data.py          # Data download script
├── train_allen_data.py             # Training script
├── TRAINING_GUIDE_RTX3070TI.md     # This guide
├── data/
│   └── allen_neuropixels/
│       ├── cache/                   # Allen SDK cache (~20-50 GB)
│       └── dataset_info.txt         # Downloaded sessions info
├── checkpoints_allen/
│   ├── best.pt                      # Best model checkpoint
│   └── latest.pt                    # Latest model checkpoint
└── logs_allen/                      # Training logs
```

## 🎓 Next Steps

After successful training:

1. **Fine-tune on your own data:** Use the trained model as a starting point
2. **Transfer learning:** Adapt to new brain areas or tasks
3. **Downstream tasks:** Use latent representations for decoding, prediction
4. **Scaling up:** Train on more sessions for better performance

## 📧 Support

If you encounter issues:

1. Check this guide's troubleshooting section
2. Verify all dependencies are installed correctly
3. Ensure CUDA and GPU drivers are up to date
4. Check GPU memory usage: `nvidia-smi`

## 🔗 Resources

- **Allen Brain Observatory:** https://allensdk.readthedocs.io/
- **PyTorch Documentation:** https://pytorch.org/docs/
- **CUDA Toolkit:** https://developer.nvidia.com/cuda-downloads

---

**Happy Training! 🚀**

Your NeuroFM-X model will learn rich representations of neural population dynamics from real visual cortex recordings. These learned representations can be used for downstream tasks like behavioral decoding, neural prediction, and transfer learning to new datasets.

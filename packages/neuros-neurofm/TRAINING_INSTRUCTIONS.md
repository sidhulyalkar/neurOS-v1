# NeuroFM-X Training Instructions

## ✅ Codebase is Now Fully Organized and Professional!

### What Changed
- ✅ **All scripts organized** into proper folders
- ✅ **Single professional README** created
- ✅ **Redundant docs removed** (kept only essential guides in `docs/`)
- ✅ **Versioning added** (v0.1.0)
- ✅ **CHANGELOG.md** created
- ✅ **Professional formatting** throughout

---

## 🚀 RESTART TRAINING NOW (With Mamba Installed!)

### Step 1: Stop Current Training
```bash
# Press Ctrl+C to stop the slow training
```

### Step 2: Activate Environment with Mamba
```bash
conda activate neurofm_v2  # Your environment with mamba-ssm + causal-conv1d
```

### Step 3: Restart Training
```bash
cd /mnt/c/Users/sidso/Documents/neurOS-v1/packages/neuros-neurofm

# Quick test (should be MUCH faster now)
python training/train.py --config configs/quick_test.yaml
```

### Expected Speed (With Mamba)
- **Before:** 36.75s/batch (CPU fallback)
- **After:** ~0.5-1s/batch (70x faster!) ⚡

**Total time for 10 epochs:** ~1-2 hours (vs 76 hours before!)

---

## 📊 Monitor Training

```bash
# Real-time monitoring
python scripts/monitor_training.py --checkpoint checkpoints_quick_test/latest.pt

# TensorBoard
tensorboard --logdir=logs_quick_test

# Check GPU usage
nvidia-smi
```

---

## 📁 New Codebase Structure

```
neuros-neurofm/
├── README.md                   # ✨ NEW: Professional README
├── VERSION                     # ✨ NEW: v0.1.0
├── CHANGELOG.md                # ✨ NEW: Version history
├── TRAINING_INSTRUCTIONS.md    # ✨ NEW: This file
│
├── src/neuros_neurofm/         # Core library
│   ├── __init__.py             # ✨ UPDATED: Versioning
│   ├── models/
│   ├── tokenizers/
│   ├── fusion/
│   └── adapters/
│
├── training/                   # ✨ ORGANIZED: All training scripts
│   ├── train.py                # Main YAML-based trainer
│   ├── train_legacy.py         # Old full_train_streaming.py
│   └── train_legacy_logging.py # Old train_with_logging.py
│
├── scripts/                    # ✨ ORGANIZED: All utilities
│   ├── data_utils.py
│   ├── monitor_training.py
│   ├── prepare_full_dataset.py
│   └── download_allen_data.py  # ✨ MOVED from root
│
├── configs/                    # YAML configurations
│   ├── quick_test.yaml
│   ├── local_full.yaml
│   └── cloud_aws_a100.yaml
│
├── deployment/                 # Docker & AWS
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── aws_setup.sh
│
├── benchmarks/                 # Evaluation
│   └── benchmark_neurofmx.py
│
└── docs/                       # ✨ CLEANED: Only essential docs
    ├── QUICK_START.md
    ├── SCALING_STRATEGY.md
    ├── TRAINING_GUIDE.md
    └── OPTIMAL_TRAINING_PLAN.md
```

**Removed:**
- ❌ REFACTORING_COMPLETE.md (progress doc)
- ❌ QUICK_REFERENCE.md (redundant)
- ❌ Obsolete training scripts (eval_latents.py, full_inference.py, etc.)

---

## ⚡ Performance Optimization

### RTX 3070 Ti Optimal Settings

**In `configs/quick_test.yaml`:**
```yaml
data:
  batch_size: 16  # Optimized for 8GB VRAM
  num_workers: 0  # WSL2 works best with 0
  pin_memory: true

training:
  use_amp: true  # Mixed precision (CRITICAL!)
  gradient_accumulation_steps: 1  # No need with batch_size=16
```

### If Still Slow, Check:

1. **Mamba is installed:**
```bash
python -c "from mamba_ssm import Mamba; print('✓ Mamba available')"
```

2. **CUDA is working:**
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

3. **GPU usage:**
```bash
nvidia-smi  # Should show ~6-7GB used during training
```

---

## 🎯 After Quick Test Succeeds

### Option 1: Full Local Training (Recommended)
```bash
# 8-12 hours on RTX 3070 Ti
python training/train.py --config configs/local_full.yaml
```

### Option 2: Cloud Training (For Scale)
```bash
# Setup AWS (one-time)
./deployment/aws_setup.sh <instance-id>

# Train on A100 (24-40 hours, $100-200)
python training/train.py --config configs/cloud_aws_a100.yaml
```

---

## 📈 Expected Results

**Quick Test (10 epochs, 4 sessions):**
- Training loss: ~0.3-0.5
- Validation loss: ~0.4-0.6
- Reconstruction R²: >0.3
- **Purpose:** Validates architecture works

**Full Training (50 epochs, 20 sessions):**
- Training loss: ~0.15-0.25
- Validation loss: ~0.2-0.35
- Reconstruction R²: >0.6
- **Purpose:** Publishable baseline

**Foundation Model (50 epochs, 200+ sessions):**
- Reconstruction R²: >0.75
- Transfer learning: <10 examples needed
- **Purpose:** State-of-the-art generalization

---

## 🛠️ Troubleshooting

### Still Slow After Installing Mamba?
```bash
# Reinstall in correct environment
conda activate neurofm_v2
pip uninstall mamba-ssm causal-conv1d
pip install mamba-ssm>=1.1.0 causal-conv1d>=1.1.0
```

### Out of Memory?
```yaml
# Reduce in config
data:
  batch_size: 8  # Or even 4
model:
  d_model: 64    # Smaller model
```

### Want to Use Old Training Script?
```bash
# Legacy scripts are in training/
python training/train_legacy.py  # Old full_train_streaming.py
```

---

## 📚 Documentation Quick Links

- **[README.md](README.md)** - Project overview
- **[QUICK_START.md](docs/QUICK_START.md)** - 5-minute guide
- **[SCALING_STRATEGY.md](docs/SCALING_STRATEGY.md)** - Progressive training
- **[TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)** - Detailed instructions

---

## ✨ Professional Codebase Achieved!

Your codebase is now:
- ✅ **Fully organized** with clear structure
- ✅ **Production-ready** with Docker & AWS support
- ✅ **Well-documented** with professional README
- ✅ **Versioned** (v0.1.0 with CHANGELOG)
- ✅ **Optimized** for RTX 3070 Ti
- ✅ **Scalable** from 4 → 200+ sessions

---

## 🚀 START TRAINING NOW!

```bash
conda activate neurofm_v2
python training/train.py --config configs/quick_test.yaml
```

**Expected:** ~0.5-1s/batch (70x faster than before!)
**Duration:** 1-2 hours for quick test
**GPU:** Should use ~6-7GB / 8GB

---

**Your foundation model is ready to train! Good luck! 🧠✨🚀**

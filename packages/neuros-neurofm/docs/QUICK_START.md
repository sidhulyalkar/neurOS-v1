# NeuroFM-X Quick Start Guide

## 🎉 Bug Fixed! Ready to Train

The tokenizer dimension mismatch (384 vs 128) has been fixed. Your model is ready to train!

---

## 🚀 Start Training NOW

```bash
cd /mnt/c/Users/sidso/Documents/neurOS-v1/packages/neuros-neurofm

# Option 1: Basic training
python3 full_train_streaming.py

# Option 2: Enhanced training with TensorBoard
python3 train_with_logging.py

# Option 3: With WandB logging
python3 train_with_logging.py --use-wandb
```

**Expected time:** ~15-20 minutes per epoch, ~12-17 hours for 50 epochs

---

## 📊 Monitor Training

### Real-time GPU monitoring:
```bash
watch -n 1 nvidia-smi
```

### Check training progress:
```bash
# Analyze latest checkpoint
python3 monitor_training.py --checkpoint latest.pt

# Compare checkpoints
python3 monitor_training.py --compare epoch_5.pt epoch_10.pt
```

### TensorBoard (if using `train_with_logging.py`):
```bash
# In another terminal
tensorboard --logdir=logs_neurofmx_full_run --bind_all

# Access at: http://localhost:6006
```

---

## ✅ What Was Fixed

1. **`Config.max_units`** changed from `128` to `384` to match preprocessed data
2. **BinnedTokenizer** now uses `encoder_output_dim` parameter instead of hardcoded value
3. All components aligned to use **384 units** consistently

---

## 📁 Output Files

Training creates:
- `checkpoints_neurofmx_full_run/` - Model checkpoints
  - `latest.pt` - Most recent checkpoint
  - `best.pt` - Best validation loss
  - `epoch_N.pt` - Epoch snapshots (every 5 epochs)
- `logs_neurofmx_full_run/` - Training logs and TensorBoard data

---

## 🔍 After Training: Evaluate

```bash
# Run comprehensive benchmark
python3 benchmark_neurofmx.py --checkpoint checkpoints_neurofmx_full_run/best.pt

# View results
cat benchmark_results/benchmark_report.md
```

Evaluates:
- Neural reconstruction quality (R², MSE, correlation)
- Behavioral decoding performance
- Population geometry and latent dynamics
- Computational efficiency
- Latent space quality

---

## 🛠️ Troubleshooting

### GPU Out of Memory
```python
# Edit Config in full_train_streaming.py
batch_size = 1  # Reduce from 2
```

### Training too slow
```python
# Reduce epochs for testing
max_epochs = 5  # Instead of 50
```

### Loss is NaN
```python
# Reduce learning rate
learning_rate = 1e-4  # Instead of 3e-4

# Or disable mixed precision
use_amp = False
```

---

## 📚 Documentation

- **Comprehensive Plan:** [NEUROFM_FOUNDATION_MODEL_PLAN.md](NEUROFM_FOUNDATION_MODEL_PLAN.md)
- **Full Training Guide:** [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- **Architecture Details:** [src/neuros_neurofm/models/](src/neuros_neurofm/models/)

---

## 🎯 Current Model Configuration

```python
# Architecture
d_model = 128              # Model dimension
n_mamba_blocks = 4         # Mamba layers
n_latents = 32             # PopT latents
latent_dim = 128           # Latent dimension

# Training
batch_size = 2
gradient_accumulation_steps = 8  # Effective batch = 16
learning_rate = 3e-4
max_epochs = 50

# Data
sequence_length = 100
max_units = 384  # ✅ FIXED!
```

**Total Parameters:** ~3M (efficient!)
**GPU Memory:** ~6-7 GB / 8 GB

---

## 🎓 Training Pipeline

```
Data (384 units, 100 timesteps)
    ↓
BinnedTokenizer (384 → 128-dim)
    ↓
MambaBackbone (4 blocks, multi-rate)
    ↓
PerceiverIO (32 latents @ 128-dim)
    ↓
PopT (2 layers)
    ↓
MultiTaskHeads (Encoder, Decoder, Contrastive)
```

---

## 🌟 Next Steps

### Immediate (Now)
1. ✅ **Start training** - Run the command above
2. Monitor GPU usage (should be >80%)
3. Check loss decreasing after first epoch

### Short-term (After 5-10 epochs)
1. Validate training is stable
2. Check reconstruction quality
3. Adjust hyperparameters if needed

### Long-term (After full training)
1. Run comprehensive benchmarks
2. Compare against baselines (CEBRA, LFADS)
3. Scale to full Allen dataset
4. Implement transfer learning

---

## ⚡ Performance Targets

After 50 epochs, expect:

**Reconstruction:**
- R² > 0.5 (target: 0.7+)
- MSE < 0.1
- Correlation > 0.7

**Computational:**
- Inference: <10 ms/sample
- GPU Memory: <500 MB
- Throughput: >100 samples/sec

---

## 💡 Pro Tips

1. **Run overnight** - Training takes 12-17 hours, perfect for overnight runs
2. **Check progress in morning** - Use `monitor_training.py` to check without stopping
3. **Save early checkpoints** - Epoch 5, 10, 15 snapshots let you compare progress
4. **Use TensorBoard** - Real-time visualization helps catch issues early
5. **Monitor GPU** - Should stay at 85-95% utilization

---

## 🆘 Need Help?

1. Check [NEUROFM_FOUNDATION_MODEL_PLAN.md](NEUROFM_FOUNDATION_MODEL_PLAN.md) for detailed guidance
2. Use `monitor_training.py` to diagnose issues
3. Review logs in `logs_neurofmx_full_run/`
4. Check GPU usage with `nvidia-smi`

---

**Ready? Start training now! 🚀**

```bash
python3 full_train_streaming.py
```

Your foundation model awaits! Good luck! 🎉

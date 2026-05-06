# neuros-astro Compute Requirements Guide

## 🖥️ TL;DR: You Can Run This Locally!

**Good News**: The neuros-astro pipeline is **very lightweight** and runs entirely on CPU.

**Your RTX 3070 Ti** is more than enough for validation experiments!

---

## Component-by-Component Breakdown

### ✅ **neuros-astro Pipeline** (Days 1-7)

**Hardware**: CPU only
**Memory**: 4-8 GB RAM
**Time**: Seconds to minutes
**Your hardware**: ✅ Perfect

| Component | Compute | Memory | Time (1000 cells, 10min recording) |
|-----------|---------|--------|-------------------------------------|
| Event detection | CPU | 1-2 GB | 5-30 seconds |
| Network construction | CPU | <1 GB | 1-5 seconds |
| Tokenization | CPU | <1 GB | <1 second |
| Visualization | CPU | <1 GB | 1-2 seconds per plot |

**Example timings on modest laptop**:
- Load Allen session: ~0.5s
- Detect events (100 cells): ~10s
- Build networks: ~2s
- Generate all visualizations: ~5s

**Total**: Under 20 seconds per session!

---

### ⚠️ **neuroFMx Training** (Week 2-3)

This is where GPU matters, but you have options:

#### **Option 1: Local Validation (RECOMMENDED)**
Run smaller-scale experiments on your RTX 3070 Ti (8GB VRAM)

**What you CAN run locally**:
- ✅ Synthetic data experiments (fully controlled)
- ✅ Single-session ablations
- ✅ Short sequence lengths (1-5 minutes)
- ✅ Smaller model sizes (reduce hidden dims)
- ✅ Proof-of-concept experiments

**Strategies for local training**:
```python
# Small config for RTX 3070 Ti
config = {
    "batch_size": 4,  # Reduce from 32
    "sequence_length": 1000,  # Reduce from 5000
    "hidden_dim": 256,  # Reduce from 512
    "n_layers": 4,  # Reduce from 8
    "use_mixed_precision": True,  # FP16 to save memory
}
```

**Expected training time**:
- Synthetic validation: 10-30 min
- Single Allen session: 30-60 min
- Good enough for proof-of-concept!

#### **Option 2: Cloud for Full Experiments**
For publication-quality multi-session training

**When you need cloud**:
- Multiple sessions (10+)
- Long sequences (>10 minutes)
- Large models (>1B parameters)
- Hyperparameter sweeps
- Final publication experiments

**Recommended cloud options**:
1. **Google Colab Pro** ($10/month)
   - A100 GPU (40GB VRAM)
   - Good for medium experiments
   - Easy to use

2. **Lambda Labs** (~$1/hour)
   - A100 GPU on demand
   - Pay as you go
   - Good for focused experiments

3. **Vast.ai** (~$0.30-1.00/hour)
   - Cheapest option
   - Rent community GPUs
   - Good for budget-conscious research

**Cost estimate for publication**:
- Synthetic validation: $0 (local)
- Real data validation: $0-10 (local or Colab free)
- Full ablation experiments: $20-50 (Colab Pro or Lambda)
- Hyperparameter search: $50-200 (Lambda or Vast.ai)

**Total for publication**: $100-300 max

---

## 📊 Recommended Workflow

### **Week 1: Pure neuros-astro (LOCAL)**
✅ All on your laptop/desktop
```
- Generate synthetic data
- Run event detection
- Build networks
- Create all visualizations
- Process Allen sessions
- Generate publication figures
```
**Hardware needed**: Any laptop with 8GB RAM
**Your setup**: More than enough! ✅

### **Week 2: neuroFMx Integration (LOCAL + CLOUD)**

**Days 8-10: Local development**
```python
# Test on your RTX 3070 Ti
- Implement astro modality loader
- Test with synthetic data
- Verify integration works
- Run tiny validation experiment
```
**Hardware**: RTX 3070 Ti (perfect for this!)

**Days 11-13: Small-scale experiments (LOCAL)**
```python
# Validate on your RTX 3070 Ti
- Synthetic ablation (neural vs neural+astro)
- Single Allen session test
- Verify astro tokens actually load
- Check if pipeline end-to-end works
```
**Hardware**: RTX 3070 Ti (will work, may be slow)

**Day 14: Cloud for real ablation (CLOUD)**
```python
# If local is too slow, move to cloud
- Colab Pro A100 or Lambda GPU
- Run full multi-session ablation
- Generate final experiment results
```
**Hardware**: Cloud GPU (only if needed!)

### **Week 3: Publication Materials (LOCAL)**
✅ All on your laptop
```
- Analyze results from Week 2
- Generate all figures
- Write manuscript
- Create supplementary materials
```
**Hardware needed**: Any laptop ✅

---

## 💰 Cost Breakdown

### **Minimum Cost Path** (Local Only)
- neuros-astro pipeline: $0 (your hardware)
- Synthetic experiments: $0 (your RTX 3070 Ti)
- Small Allen validation: $0 (your RTX 3070 Ti)
- Figures & writing: $0

**Total**: $0 if you're patient!

### **Recommended Cost Path** (Local + Minimal Cloud)
- Week 1 pipeline: $0 (local)
- Week 2 integration: $0 (local)
- Week 2 small experiments: $0 (local RTX 3070 Ti)
- Week 2 final ablation: $10-30 (Colab Pro for 1-2 days)
- Week 3 analysis: $0 (local)

**Total**: $10-30 for publication!

### **Optimal Path** (Local + Cloud When Needed)
- Use local for development & testing
- Use cloud for final experiments
- Total cost: $50-100

---

## 🚀 Practical Recommendations

### **Start Local, Move to Cloud Only if Needed**

1. **Run neuros-astro pipeline completely locally** ✅
   - No GPU needed
   - Fast enough
   - Zero cost

2. **Test neuroFMx integration locally** ✅
   - Your RTX 3070 Ti is perfect
   - Synthetic data runs fine
   - Validates the code works

3. **Decide based on results**:
   - If local experiments work: Use them for paper!
   - If local is too slow: Move to cloud for finals
   - If experiments fail: No point buying cloud until fixed

### **Memory-Efficient Training Tips**

If training on your RTX 3070 Ti (8GB):

```python
# memory_efficient_config.yaml
training:
  batch_size: 2  # Small batches
  gradient_accumulation_steps: 16  # Effective batch_size = 32
  mixed_precision: true  # FP16
  gradient_checkpointing: true  # Trade compute for memory

model:
  hidden_dim: 256  # Smaller than default
  n_layers: 4  # Fewer layers
  max_sequence_length: 1000  # Shorter sequences

data:
  num_workers: 4  # Use CPU for data loading
  pin_memory: true  # Speed up transfers
```

### **When to Use Cloud**

Move to cloud if:
- ❌ Local training takes >2 hours per experiment
- ❌ You run out of VRAM (OOM errors)
- ❌ You need to run 10+ experiments
- ❌ You're on a deadline

Stay local if:
- ✅ Training takes <1 hour
- ✅ Experiments fit in 8GB VRAM
- ✅ You have time to wait
- ✅ Results look promising locally

---

## 📈 Expected Timeline & Costs

| Phase | Duration | Hardware | Cost |
|-------|----------|----------|------|
| **Week 1: Pipeline** | 5-7 days | Laptop CPU | $0 |
| Implement visualization | 1 day | CPU | $0 |
| Process Allen data | 2 days | CPU | $0 |
| Generate figures | 1 day | CPU | $0 |
| Validation report | 1 day | CPU | $0 |
| **Week 2: Integration** | 5-7 days | Local GPU + Cloud | $10-50 |
| Code integration | 2 days | CPU | $0 |
| Synthetic validation | 1 day | RTX 3070 Ti | $0 |
| Real data experiment | 2 days | RTX 3070 Ti / Cloud | $0-30 |
| Ablation experiments | 1 day | Cloud A100 | $10-20 |
| **Week 3: Writing** | 5-7 days | Laptop CPU | $0 |
| Figure generation | 1 day | CPU | $0 |
| Results analysis | 2 days | CPU | $0 |
| Manuscript draft | 2 days | CPU | $0 |
| **Total** | **3 weeks** | **Mixed** | **$10-50** |

---

## ✅ **Bottom Line**

**You can absolutely do this on your RTX 3070 Ti!**

**Local-first strategy**:
1. Week 1: 100% local (neuros-astro pipeline)
2. Week 2: Start local, cloud if needed (experiments)
3. Week 3: 100% local (writing)

**Budget needed**: $0-50 (most likely $0-20)

**Your hardware is perfect for**:
- ✅ All data processing
- ✅ All visualization
- ✅ Development and testing
- ✅ Small-scale experiments
- ✅ Proof-of-concept validation

**Cloud only needed for**:
- ❓ Large-scale multi-session training (maybe)
- ❓ Final publication experiments (if local is too slow)
- ❓ Hyperparameter sweeps (optional)

---

## 🎯 **Action Plan**

### **This Week (Week 1)**
- Run everything locally
- Process Allen data (CPU only)
- Generate all figures (CPU only)
- Zero cost, zero GPU needed

### **Next Week (Week 2)**
- Test integration on RTX 3070 Ti
- Run synthetic experiment locally
- Try one Allen session locally
- If too slow → Colab Pro for $10

### **Week After (Week 3)**
- Analyze results locally
- Write manuscript locally
- Submit preprint!

---

## 💡 Pro Tips

1. **Start with synthetic data**: Validates everything works, runs in minutes
2. **Use small configs locally**: Proves concept before scaling up
3. **Profile before cloud**: Make sure code is efficient first
4. **Colab free tier first**: Try before paying
5. **Spot instances**: If using cloud, use spot/preemptible (3x cheaper)

---

**Questions to Ask Yourself**:

1. Does the pipeline work on synthetic data? → **Test locally (free)**
2. Does integration work with neuroFMx? → **Test locally (free)**
3. Do synthetic experiments show signal? → **Test locally (free)**
4. Does one real session work? → **Try local first (free)**
5. Do I need 10+ sessions? → **Then maybe cloud ($20-50)**

**Most likely**: You'll get publication-worthy results for $0-20!

Your RTX 3070 Ti is MORE than sufficient for this project. 🚀

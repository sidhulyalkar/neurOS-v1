# neuros-astro Installation Guide

## 🚀 Quick Install (Conda Environment)

### **Step 1: Activate Your Conda Environment**

```bash
# Activate your existing conda environment
conda activate base  # or whatever your env is named

# Verify Python version (need 3.10+)
python --version
```

### **Step 2: Navigate to neuros-astro**

```bash
cd /mnt/c/Users/sidso/Documents/neurOS-v1/packages/neuros-astro
```

### **Step 3: Install neuros-astro**

```bash
# Option A: Install with all features (RECOMMENDED)
pip install -e ".[all]"

# Option B: Install with just visualization (minimal)
pip install -e ".[viz]"

# Option C: Install base only (no visualization)
pip install -e .
```

**Recommended**: Use Option A for everything you need!

### **Step 4: Verify Installation**

```bash
# Check if neuros-astro CLI works
neuros-astro --help

# Run tests to verify everything works
pytest tests/ -v

# Quick check
python -c "import neuros_astro; print('✓ neuros-astro installed!')"
```

### **Step 5: Run Quick Demo**

```bash
# Generate synthetic data and see results
python examples/05_get_started_today.py

# Check outputs
ls -lh starter_output/
ls -lh starter_output/figures/
```

**Expected output**:
- `starter_output/` directory with NPZ files
- `starter_output/figures/` with PNG plots
- Console output showing event detection results

---

## 📦 What Gets Installed

### **Core Dependencies** (always installed):
- `numpy` - Array operations
- `scipy` - Scientific computing
- `pandas` - Data handling
- `pydantic` - Data validation
- `typer` - CLI framework
- `rich` - Pretty terminal output
- `networkx` - Graph analysis

### **Optional: Visualization** `[viz]`:
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization

### **Optional: NWB Support** `[nwb]`:
- `pynwb` - Neurodata Without Borders
- `hdmf` - HDF5 data format

### **Optional: DANDI** `[dandi]`:
- `dandi` - DANDI archive access

### **Optional: Imaging** `[imaging]`:
- `tifffile` - TIFF file handling
- `scikit-image` - Image processing

### **Development** `[dev]`:
- `pytest` - Testing
- `pytest-cov` - Coverage
- `ruff` - Linting
- `mypy` - Type checking

### **All Features** `[all]`:
All of the above!

---

## 🔧 Troubleshooting

### **Issue: `pip install -e` fails**

```bash
# Make sure pip is up to date
pip install --upgrade pip setuptools wheel

# Try again
pip install -e ".[all]"
```

### **Issue: pytest not found**

```bash
# Install dev dependencies
pip install pytest pytest-cov
```

### **Issue: matplotlib import error**

```bash
# Install visualization dependencies
pip install matplotlib seaborn
```

### **Issue: Allen SDK conflicts**

```bash
# Allen SDK might have specific version requirements
# Install neuros-astro first, then Allen SDK
pip install -e ".[all]"
pip install allensdk
```

### **Issue: Permission denied**

```bash
# Use --user flag
pip install --user -e ".[all]"
```

---

## ✅ Verification Checklist

After installation, verify:

```bash
# 1. CLI works
neuros-astro --help
# Should show command list

# 2. Python imports work
python -c "from neuros_astro import __version__; print(__version__)"
# Should print: 0.1.0

# 3. Visualization works
python -c "from neuros_astro.visualization import plot_event_raster; print('✓')"
# Should print: ✓

# 4. Tests pass
pytest tests/ -v
# Should show: 46 passed

# 5. Quick demo runs
python examples/05_get_started_today.py
# Should generate outputs
```

---

## 🎯 Quick Start Commands

### **After Installation**:

```bash
# 1. Generate synthetic data (5 seconds)
python examples/05_get_started_today.py

# 2. Process your Allen data (1-2 minutes)
python examples/06_process_allen_data.py

# 3. Interactive Jupyter (learning)
cd notebooks
jupyter notebook
# Open: 01_astro_pipeline_walkthrough.ipynb
```

---

## 📊 Conda Environment Setup (Fresh Start)

If you want a dedicated environment:

```bash
# Create new conda environment
conda create -n neuros-astro python=3.10 -y

# Activate it
conda activate neuros-astro

# Install neuros-astro
cd /mnt/c/Users/sidso/Documents/neurOS-v1/packages/neuros-astro
pip install -e ".[all]"

# Verify
neuros-astro --help
pytest tests/ -v
```

---

## 🔗 Integration with Other neurOS Packages

If you need integration with other packages:

```bash
# Install neuros-foundation (if needed)
cd ../neuros-foundation
pip install -e .

# Install neuros-neurofm (for Week 2)
cd ../neuros-neurofm
pip install -e .

# Install Allen SDK (for real data)
pip install allensdk

# Back to neuros-astro
cd ../neuros-astro
```

---

## 💻 System Requirements

**Minimum**:
- Python 3.10+
- 4GB RAM
- 1GB disk space

**Recommended**:
- Python 3.10+
- 8GB RAM
- 2GB disk space
- GPU (RTX 3070 Ti or better) for Week 2 experiments

**Your System**: ✅ Perfect!
- RTX 3070 Ti (8GB VRAM)
- More than enough RAM
- Windows/WSL2 environment

---

## 🐛 Common Issues

### **WSL2-Specific**

If running in WSL2 (which you are):

```bash
# Make sure you're in the WSL filesystem
pwd
# Should show: /mnt/c/Users/sidso/...

# If matplotlib display issues:
export MPLBACKEND=Agg
# Or add to ~/.bashrc
```

### **Jupyter in WSL2**

```bash
# Install jupyter in conda env
pip install jupyter

# Launch with browser
jupyter notebook --no-browser

# Copy the URL shown (http://localhost:8888/...)
# Paste in Windows browser
```

### **GPU Access in WSL2**

For Week 2 experiments:

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True (if you have PyTorch with CUDA)

# If needed, install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 📝 Post-Installation Next Steps

### **Immediate** (5 min):

```bash
# Verify everything works
python examples/05_get_started_today.py

# Check outputs
ls -lh starter_output/
```

### **Today** (30 min):

```bash
# Try interactive notebook
cd notebooks
jupyter notebook
# Open: 01_astro_pipeline_walkthrough.ipynb
```

### **This Week**:

```bash
# Process your Allen data
python examples/06_process_allen_data.py --all
```

---

## 🎓 Learning Path

1. **Install** (5 min) ← You are here!
2. **Quick demo** (5 min) → `python examples/05_get_started_today.py`
3. **Notebook 01** (30 min) → Learn the pipeline
4. **Notebook 02** (30 min) → Process Allen data
5. **Batch processing** (10 min) → Process all sessions
6. **Week 2** → neuroFMx integration

---

## 💡 Pro Tips

1. **Always use editable install** (`-e`):
   - Changes to code take effect immediately
   - Good for development

2. **Install with `[all]`**:
   - Gets everything you might need
   - Avoids dependency headaches later

3. **Run tests after install**:
   - Validates installation
   - Catches issues early

4. **Keep conda env activated**:
   - Add to your shell profile
   - Or use conda auto-activation

---

## 🚀 Ready to Go!

After running these commands, you'll have:
- ✅ neuros-astro installed
- ✅ All dependencies ready
- ✅ CLI working
- ✅ Tests passing
- ✅ Ready to run experiments

**Next**: Run `python examples/05_get_started_today.py` and see it in action!

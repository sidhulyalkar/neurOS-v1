# neuros-astro Jupyter Notebooks

Interactive tutorials for learning and using neuros-astro.

## 📚 Notebooks

### 01_astro_pipeline_walkthrough.ipynb
**Complete introduction to neuros-astro**

- Generate synthetic astrocyte data
- Detect calcium events
- Build functional networks
- Tokenize for foundation models
- Create publication figures

**Level**: Beginner
**Time**: 20-30 minutes
**Hardware**: CPU only

### 02_allen_data_processing.ipynb
**Process real Allen Brain Observatory data**

- Load your existing Allen 2P sessions
- Run neuros-astro pipeline on real data
- Validate biological plausibility
- Export tokens for neuroFMx

**Level**: Intermediate
**Time**: 30-45 minutes
**Hardware**: CPU only

## 🚀 Getting Started

### Installation

```bash
# Install neuros-astro with visualization
cd packages/neuros-astro
pip install -e ".[viz]"

# Or install all features
pip install -e ".[all]"
```

### Launch Jupyter

```bash
# From the notebooks directory
cd packages/neuros-astro/notebooks
jupyter notebook

# Or use JupyterLab
jupyter lab
```

### Run Notebooks

1. Open `01_astro_pipeline_walkthrough.ipynb` first
2. Run all cells to see the complete pipeline
3. Experiment with parameters
4. Move to `02_allen_data_processing.ipynb` for real data

## 💻 Compute Requirements

**All notebooks run on CPU!**

- No GPU required
- 8GB RAM recommended
- Typical runtime: 5-10 minutes per notebook
- Perfect for your laptop ✅

## 📊 Expected Outputs

Each notebook creates:
- Event detection results
- Network graphs
- Publication-quality figures
- Token files for neuroFMx
- Summary statistics

Outputs saved to:
- `notebook_outputs/` (notebook 01)
- `allen_outputs/` (notebook 02)

## 🎯 Learning Path

**Day 1**: Run notebook 01
- Understand the pipeline
- See synthetic validation
- Generate example figures

**Day 2**: Run notebook 02
- Process your Allen data
- Validate on real recordings
- Check biological plausibility

**Day 3**: Customize
- Adjust detection parameters
- Try different network metrics
- Explore your own data

## 📖 Additional Resources

**Documentation**:
- [Full Roadmap](../NEUROS_ASTRO_PUBLICATION_ROADMAP.md)
- [Next Steps Guide](../NEUROS_ASTRO_NEXT_STEPS.md)
- [Compute Requirements](../COMPUTE_REQUIREMENTS.md)
- [Scientific Whitepaper](../../../neuros_astro_whitepaper.md)

**Example Scripts**:
- `examples/05_get_started_today.py` - Quick demo
- `examples/06_process_allen_data.py` - Batch processing

**Python API**:
- See `examples/02_python_api_example.py`
- All functions documented with docstrings

## 🔧 Troubleshooting

### Kernel crashes
- Reduce data size (fewer cells/timepoints)
- Close other applications
- Restart kernel

### Imports fail
```bash
# Install missing dependencies
pip install -e ".[viz]"
```

### Visualizations don't show
```python
# Add to top of notebook
%matplotlib inline
import matplotlib.pyplot as plt
```

### Allen data not found
- Check path in notebook 02
- Verify sessions are in correct directory
- Update path if needed

## 💡 Tips

**Best practices**:
1. Run all cells in order first time
2. Restart kernel if changing parameters
3. Save interesting plots immediately
4. Keep notes on what parameters work best

**For publication**:
- Use notebook 01 for methods validation
- Use notebook 02 for real data figures
- Export high-res figures (300 DPI)
- Document all parameters used

**Performance**:
- Notebooks run in seconds/minutes
- No need to wait for GPU
- Can process multiple sessions quickly
- Batch processing via script is faster

## 🎓 Next Steps

After completing notebooks:

1. **Generate publication figures**
   - Run both notebooks
   - Save all visualizations
   - Document statistics

2. **Process all Allen sessions**
   - Use `examples/06_process_allen_data.py --all`
   - Aggregate results
   - Compare across sessions

3. **neuroFMx integration**
   - Use generated token files
   - Load astro modality
   - Run ablation experiments

4. **Write manuscript**
   - Methods from notebook code
   - Results from outputs
   - Figures from visualizations

## 📞 Help

**Questions?**
- Check docstrings: `help(function_name)`
- Read the whitepaper
- Review example scripts
- Check GitHub issues

**Found a bug?**
- Report in GitHub Issues
- Include notebook cell output
- Mention Python/package versions

## 🎉 You're Ready!

These notebooks give you everything you need to:
- ✅ Understand the pipeline
- ✅ Process real data
- ✅ Generate figures
- ✅ Export for models
- ✅ Move toward publication

**Time to start experimenting!** 🚀

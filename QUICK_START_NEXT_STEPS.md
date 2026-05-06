# Quick Start: Next Steps

**Created:** 2026-05-05
**Status:** Ready to proceed with neuroFMx integration

---

## ✅ What's Done

- ✅ 10 Allen sessions processed (9,153 events, 7,551 networks)
- ✅ Results validated (ALL CHECKS PASSED - publication-ready)
- ✅ Cross-session analysis complete (high consistency found)
- ✅ Ablation framework setup (ready for training)

---

## 🚀 What to Do Next

### Today (5 minutes)

**1. Review the summary:**
```bash
cat ALLEN_PROCESSING_COMPLETE_SUMMARY.md
```

**2. Check validation results:**
```bash
cat validation_report.txt
```

**3. View cross-session findings:**
```bash
cat cross_session_analysis/cross_session_summary.txt
```

---

### This Week (Integration)

**Step 1: Create multimodal dataset loader**

File: `packages/neuros-neurofm/src/neuros_neurofm/datasets/allen_multimodal_dataset.py`

Copy the template from `ALLEN_PROCESSING_COMPLETE_SUMMARY.md` (search for "AllenMultimodalDataset")

**Step 2: Test loading data**

```python
from neuros_neurofm.datasets.allen_multimodal_dataset import AllenMultimodalDataset

# Test baseline
dataset = AllenMultimodalDataset(
    session_ids=['545446482', '581150104', '613968705'],
    modalities=['neural']
)

# Test multimodal
dataset_multi = AllenMultimodalDataset(
    session_ids=['545446482', '581150104', '613968705'],
    modalities=['neural', 'astro']
)

print(f"Loaded {len(dataset)} sessions")
print(f"Sample batch: {dataset[0].keys()}")
```

**Step 3: Verify token alignment**

```bash
# Check token files are accessible
python -c "
import numpy as np
tokens = np.load('allen_nwb_results/session_545446482/astro_tokens.npz')
print(f'Tokens: {tokens[\"tokens\"].shape}')
print(f'Timestamps: {tokens[\"timestamps\"].shape}')
"
```

---

### Next Week (Training)

**Read ablation instructions:**
```bash
cat ablation_study/README.md
```

**Implement training:**
1. Edit `ablation_study/train_ablation_template.py`
2. Replace TODO sections with your neuroFMx training code
3. Run baseline: `python ablation_study/train_ablation_template.py --condition neural_only`
4. Run test: `python ablation_study/train_ablation_template.py --condition neural_astro`

---

## 📊 Quick Data Reference

**Available Sessions:**
- 545446482 (171 ROIs, 966 events)
- 581150104 (8 ROIs, 579 events)
- 613968705 (15 ROIs, 163 events)
- 627823695 (21 ROIs, 397 events)
- 644026238 (240 ROIs, 2,874 events)
- 645086975 (46 ROIs, 559 events)
- 652091264 (17 ROIs, 1,081 events)
- 652094901 (207 ROIs, 1,289 events)
- 657078119 (9 ROIs, 616 events)
- 662361096 (16 ROIs, 629 events)

**Token Files:**
```
allen_nwb_results/session_*/astro_tokens.npz
```

**Neural Data:**
```
packages/neuros-mechint/examples/allen_data_demo/data/2p_sessions/2p_session_*.npz
```

---

## 🎯 Key Metrics to Track

When training your models, track these metrics:

**Primary (Lower is Better):**
- `prediction_loss`: Cross-entropy or MSE

**Primary (Higher is Better):**
- `decoding_accuracy`: Stimulus decoding accuracy
- `cross_session_transfer`: Generalization to unseen sessions
- `r2_score`: Variance explained

**Compare:**
- Baseline (neural-only) vs Test (neural+astro)
- Expected improvement: 5-15% based on literature

---

## 📖 Documentation

**Main guides:**
- `ALLEN_PROCESSING_COMPLETE_SUMMARY.md` - Complete overview
- `ablation_study/README.md` - Training instructions
- `packages/neuros-astro/ADVANCED_FEATURES.md` - API reference
- `cross_session_analysis/cross_session_summary.txt` - Statistics

**Example code:**
- `packages/neuros-astro/examples/07_advanced_analysis_demo.py` - Full API demo
- `packages/neuros-astro/examples/process_allen_nwb.py` - NWB processing
- `ablation_study/train_ablation_template.py` - Training template

---

## 🐛 Troubleshooting

**Can't find token files?**
```bash
find . -name "astro_tokens.npz" -type f
```

**Need to re-run analysis?**
```bash
# Cross-session
python packages/neuros-astro/examples/analyze_allen_sessions.py

# Validation
python packages/neuros-astro/examples/validate_allen_results.py

# Ablation setup
python packages/neuros-astro/examples/setup_ablation_study.py
```

**Check data quality:**
```bash
cat validation_report.txt
```

---

## 💡 Tips

1. **Start small:** Test with 3 sessions first, then scale to all 10
2. **Verify alignment:** Make sure neural and astro timestamps align
3. **Track experiments:** Use the experiment tracker to record all runs
4. **Save checkpoints:** Keep model checkpoints for each condition
5. **Document changes:** Update ablation_study/README.md with your notes

---

## ✅ Checklist

**Before Training:**
- [ ] Create multimodal dataset loader
- [ ] Test loading neural data
- [ ] Test loading astro tokens
- [ ] Verify data alignment
- [ ] Read ablation_study/README.md

**During Training:**
- [ ] Train baseline (neural-only)
- [ ] Train test (neural+astro)
- [ ] Track metrics for both conditions
- [ ] Save model checkpoints

**After Training:**
- [ ] Compare baseline vs test
- [ ] Generate ablation figures
- [ ] Run statistical comparison
- [ ] Write results summary

---

## 📞 Need Help?

**Check these first:**
1. `ALLEN_PROCESSING_COMPLETE_SUMMARY.md` - Full documentation
2. `ablation_study/README.md` - Training guide
3. `packages/neuros-astro/ADVANCED_FEATURES.md` - API reference

**Key commands:**
```bash
# View summary
cat ALLEN_PROCESSING_COMPLETE_SUMMARY.md

# Check data
ls allen_nwb_results/

# Review ablation setup
ls ablation_study/

# See cross-session results
ls cross_session_analysis/figures/
```

---

**Ready to integrate and train! 🚀**

Next immediate action: Create `AllenMultimodalDataset` class and test loading data.

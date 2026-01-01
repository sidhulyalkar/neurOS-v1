# Circuit Extraction & Mechanistic Interpretability Workflow

**Status**: ✅ All scripts created | Ready to run complete pipeline

This guide shows you how to use your neurOS-v1 mechanistic interpretability toolkit to extract and analyze circuits from your trained SAE.

---

## 🎯 What You've Built

You now have a complete pipeline implementing Experiments 1.1, 1.2, 2.1, and 3.1 from your [ADVANCED_RESEARCH_ROADMAP.md](ADVANCED_RESEARCH_ROADMAP.md):

| Experiment | Script | What It Does |
|------------|--------|--------------|
| **1.1: Feature Attribution** | `experiments/circuit_extraction/attribution_analysis.py` | ✅ COMPLETED - Identifies which neurons contribute to each SAE feature using Integrated Gradients |
| **1.2: Circuit Ablation** | `experiments/circuit_extraction/ablation_study.py` | Validates circuits via causal perturbation (ablate neurons, measure impact) |
| **2.1: Cross-Modal Decoding** | `experiments/cross_modal/visual_behavior_decoding.py` | Tests if SAE features predict behavior better than raw neurons |
| **3.1: Feature Dynamics** | `experiments/dynamics/feature_dynamics.py` | Analyzes temporal evolution (fast vs slow, transient vs sustained features) |

---

## 🧬 Your Circuit Extraction Results (Session 754829445)

### Summary Statistics

From your **completed** Feature Attribution analysis:

```
✅ Circuits extracted: 20 (for top 20 SAE features)
✅ Total neurons used: 71/92 (77%)
✅ Reused neurons: 46 (neuron multifunctionality!)
✅ Max reuse: 9 features per neuron (computational hubs)
✅ Mean circuit sparsity: 49% (features use ~half neurons)
```

### Key Insights

1. **Neuron Reuse** - 46/71 neurons contribute to multiple features
   - This suggests neurons are NOT "grandmother cells"
   - Instead: neurons are **computational building blocks** reused across features

2. **Sparse Circuits** - Features use ~50% of available neurons
   - Efficient coding: features don't need all neurons
   - Matches theoretical predictions for sparse representations

3. **Computational Hubs** - Some neurons drive 9 different features
   - These are critical nodes in the circuit
   - Ablating these would disrupt many features

### Visualizations Generated

Check `results/circuits/`:
- `circuit_feature_48.png` - Circuit for Feature 48 (70% selective, 92° orientation)
- `circuit_feature_82.png` - Circuit for Feature 82 (63% selective, 180° orientation)
- ... (20 total circuit diagrams)
- `circuit_motif_analysis.png` - Overall neuron reuse patterns

---

## 🚀 Complete Workflow Commands

Activate your conda environment:
```bash
conda activate mechint_playground  # or neurofm
```

### Step 1: Circuit Extraction (✅ DONE)

```bash
# You already ran this successfully!
python experiments/circuit_extraction/attribution_analysis.py \
    --sae-model sae_models/sae_session_754829445.pt \
    --session-id 754829445 \
    --allen-cache allen_validation_cache \
    --output-dir results/circuits \
    --top-features 20 \
    --device cpu
```

**Outputs**: `results/circuits/attribution_results_session_754829445.json` + 20 circuit diagrams

---

### Step 2: Causal Validation via Ablation

Validate your circuits by ablating neurons and measuring impact:

```bash
python experiments/circuit_extraction/ablation_study.py \
    --sae-model sae_models/sae_session_754829445.pt \
    --attribution-results results/circuits/attribution_results_session_754829445.json \
    --session-id 754829445 \
    --allen-cache allen_validation_cache \
    --output-dir results/circuits/ablation \
    --top-features 10 \
    --device cpu
```

**Expected runtime**: ~15-20 minutes
**What it tests**: Which neurons are **causally necessary** vs just correlated

**Outputs**:
- `results/circuits/ablation/ablation_results_session_754829445.json`
- `results/circuits/ablation/ablation_feature_*.png` - Impact plots for each feature
- Minimal circuit analysis (which neurons can be removed without degrading features)

---

### Step 3: Cross-Modal Analysis (Visual→Behavior)

Test if SAE features capture behaviorally relevant information:

```bash
python experiments/cross_modal/visual_behavior_decoding.py \
    --sae-model sae_models/sae_session_754829445.pt \
    --session-id 754829445 \
    --allen-cache allen_validation_cache \
    --output-dir results/cross_modal \
    --device cpu
```

**Expected runtime**: ~10 minutes
**What it answers**:
- Do SAE features predict running speed better than raw neurons?
- Are orientation-selective features also behavior-selective?
- Which features encode task-relevant information?

**Outputs**:
- `results/cross_modal/cross_modal_results_session_754829445.json`
- `results/cross_modal/decoding_comparison.png` - SAE vs Raw performance
- `results/cross_modal/selectivity_*.png` - Feature overlap analysis

---

### Step 4: Temporal Dynamics

Analyze how SAE features evolve over time:

```bash
python experiments/dynamics/feature_dynamics.py \
    --sae-model sae_models/sae_session_754829445.pt \
    --session-id 754829445 \
    --allen-cache allen_validation_cache \
    --output-dir results/dynamics \
    --top-features 10 \
    --device cpu
```

**Expected runtime**: ~20 minutes (analyzing time-resolved data)
**What it measures**:
- Response latencies (how fast features activate)
- Decay constants (transient vs sustained responses)
- Feature timecourses during stimulus presentation

**Outputs**:
- `results/dynamics/dynamics_results_session_754829445.json`
- `results/dynamics/dynamics_feature_*.png` - Temporal response profiles
- Classification: Fast/Slow, Transient/Sustained

---

## 📊 Interpreting Your Results

### Circuit Attribution (Step 1 - ✅ Done)

**Look for**:
- Which neurons have highest attribution scores?
- Are highly selective neurons also highly attributed?
- Do circuits cluster by preferred orientation?

**Your results show**:
- Feature 48: Driven by neurons [79, 22, 14, 69, 37...] (top attribution: 0.035)
- Feature 82: Your best feature (0.707 correlation) - check its circuit!

### Ablation Study (Step 2)

**Look for**:
- Which neurons cause >20% disruption when ablated? (critical neurons)
- Can you find minimal circuits (subset sufficient for feature)?
- Compare attribution scores vs ablation impact (causal validation)

**Expected finding**: Not all high-attribution neurons are high-impact (correlation ≠ causation)

### Cross-Modal (Step 3)

**Look for**:
- Does SAE R² > Raw R² for behavior decoding?
- Which features are both orientation AND behavior selective?

**Interpretations**:
- If SAE > Raw: Features capture behaviorally relevant info (good!)
- If SAE = Raw: Features preserve all information (good!)
- If SAE < Raw: Features specialize on stimulus only (expected for vision)

### Dynamics (Step 4)

**Look for**:
- Mean latency: How quickly do features respond? (<100ms = fast V1)
- Response types:
  - **Transient** (fast decay): Simple cell-like
  - **Sustained** (slow decay): Complex cell-like
  - **Intermediate**: Mixed properties

---

## 🔬 Scientific Questions You Can Now Answer

### 1. Circuit Composition
✅ **Question**: Which neurons create orientation-selective SAE features?
✅ **Answer**: Use attribution analysis results
✅ **Validation**: Use ablation study

### 2. Neuron Reuse
✅ **Question**: Do neurons contribute to multiple features?
✅ **Answer**: Yes! 46/71 neurons are reused (motif analysis)
✅ **Implication**: Neurons are building blocks, not grandmother cells

### 3. Causal Structure
⏳ **Question**: Which neurons are **necessary** vs **sufficient**?
⏳ **Answer**: Run ablation study to find out
⏳ **Method**: Minimal circuit extraction

### 4. Behavior Relevance
⏳ **Question**: Do SAE features encode task-relevant information?
⏳ **Answer**: Run cross-modal decoding
⏳ **Test**: Predict running speed, pupil size from features

### 5. Temporal Properties
⏳ **Question**: Are there fast vs slow feature classes?
⏳ **Answer**: Run dynamics analysis
⏳ **Classification**: Transient vs Sustained responses

---

## 📝 Next Steps for Publication

### Paper Outline: "Sparse Autoencoders Reveal Interpretable Circuits in Neural Data"

**Figure 1**: SAE Architecture + Training Results
- Your 71% selective features (vs 37% raw neurons)
- Training curves, reconstruction quality

**Figure 2**: Circuit Extraction ✅
- Top circuit diagrams (from Step 1)
- Neuron reuse analysis
- Circuit sparsity statistics

**Figure 3**: Causal Validation
- Ablation impact plots (from Step 2)
- Critical neuron identification
- Minimal circuit analysis

**Figure 4**: Cross-Modal Integration
- Behavior decoding performance (from Step 3)
- Feature selectivity overlap
- SAE vs Raw comparison

**Figure 5**: Temporal Dynamics
- Feature timecourses (from Step 4)
- Latency distributions
- Fast vs Slow feature classes

**Supplementary**:
- Multi-session validation (repeat on 10 sessions)
- Synthetic data validation (ground truth circuits)
- Comparison to other methods (PCA, ICA, NMF)

---

## 🛠️ Troubleshooting

### Memory Issues
If you run out of RAM during dynamics analysis:
```bash
# Reduce number of features analyzed
--top-features 5
```

### GPU Not Available
All scripts default to CPU, but you can speed up with GPU:
```bash
--device cuda
```

### Missing Behavioral Data
Some sessions may not have running speed/pupil data. Check data availability in script output.

---

## 📚 Integration with neuros-mechint

Your experiments now use these `neuros-mechint` components:

| Component | Where Used | Purpose |
|-----------|------------|---------|
| `IntegratedGradients` | Attribution | Feature→neuron attribution |
| `NeuronAblation` | Ablation study | Causal testing |
| `CrossModalAnalyzer` | Cross-modal | Visual→behavior decoding |
| `DynamicalSystemsAnalyzer` | Dynamics | Temporal analysis |

**Full toolbox available** in `packages/neuros-mechint/src/neuros_mechint/`:
- `attribution.py` - Attribution methods
- `interventions/ablation.py` - Ablation tools
- `dynamics/` - Dynamical systems analysis
- `alignment/` - Cross-modal alignment
- `circuits/` - Circuit discovery

---

## 🎯 Quick Summary

**What you have**:
- ✅ Trained SAE with 71% selective features (92% better than raw!)
- ✅ Circuit extraction complete (20 circuits identified)
- ✅ 4 experiment scripts ready to run
- ✅ Integration with neuros-mechint toolkit

**What to do next**:
1. ✅ Analyze your circuit extraction results (already done!)
2. ⏳ Run ablation study (Step 2) - validate circuits causally
3. ⏳ Run cross-modal analysis (Step 3) - test behavior relevance
4. ⏳ Run dynamics analysis (Step 4) - characterize temporal properties
5. 📝 Write up results for publication

**Estimated total runtime**: ~45-60 minutes for all experiments

---

**You're ready to push the boundaries of mechanistic interpretability for neuroscience!** 🧠✨

Questions? Check:
- [ADVANCED_RESEARCH_ROADMAP.md](ADVANCED_RESEARCH_ROADMAP.md) - Full research plan
- [COMPREHENSIVE_SAE_ANALYSIS.md](COMPREHENSIVE_SAE_ANALYSIS.md) - SAE validation
- `packages/neuros-mechint/` - Full mechint toolkit documentation

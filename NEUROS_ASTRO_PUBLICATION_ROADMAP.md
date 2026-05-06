# neuros-astro: Publication-Ready Roadmap

**Goal**: Transform neuros-astro from a working prototype into publication-worthy research with reproducible experiments and compelling results.

**Timeline**: 2-3 weeks to first submission-ready manuscript

---

## Phase 1: Visualization & Analysis Tools (Week 1, Days 1-3)

### Priority: CRITICAL
**Why**: Need publication-quality figures and exploratory analysis capabilities

### Tasks:

#### Task 1.1: Event Visualization (Day 1)
**Files to create:**
- `neuros_astro/visualization/event_plots.py`
- `neuros_astro/visualization/network_plots.py`

**Functions needed:**
```python
def plot_event_raster(events, frame_rate_hz, ax=None)
def plot_event_feature_distributions(events, figsize=(12, 8))
def plot_spatial_event_map(events, image_shape, ax=None)
def plot_event_traces_with_detection(traces, events, region_id, frame_rate_hz)
```

**Deliverable**: Script `examples/03_visualize_results.py` that generates all key figures

#### Task 1.2: Network Visualization (Day 2)
**Functions needed:**
```python
def plot_astro_network(graph, spatial_positions=None, ax=None)
def plot_network_evolution(graphs, figsize=(15, 5))
def plot_degree_distribution(graph, ax=None)
def plot_connectivity_matrix(graph, ax=None)
```

**Deliverable**: Network analysis notebook showing temporal evolution

#### Task 1.3: Statistical Analysis (Day 3)
**Functions needed:**
```python
def compute_event_statistics(events) -> dict
def compare_event_distributions(events_a, events_b) -> dict
def network_stability_analysis(graphs) -> dict
```

**Deliverable**: Statistical summary functions for results section

---

## Phase 2: Real Dataset Integration (Week 1, Days 4-7)

### Priority: CRITICAL
**Why**: Need real data to validate biological relevance

### Task 2.1: NWB Metadata Loader (Day 4)

**File to create**: `neuros_astro/io/nwb_loader.py`

**Functions needed:**
```python
def summarize_nwb(path: str) -> dict
def list_ophys_series(path: str) -> list[dict]
def load_roi_response_series(path: str, series_name: str | None = None) -> tuple
```

**Test with**: Allen Visual Coding dataset (already in your environment!)

#### Task 2.2: Allen Dataset Validation (Days 5-6)

**Target dataset**: Use existing Allen Visual Coding data from your `allen_data_demo` work

**Script to create**: `examples/04_allen_dataset_pipeline.py`

**Workflow:**
1. Load Allen ROI traces (you already have this!)
2. Run dataset triage scoring
3. Detect events from traces
4. Build coactivation networks
5. Generate validation report with visualizations

**Success criteria:**
- Detects biologically plausible events (1-10s duration)
- Network shows non-random structure
- Statistical validation passes

#### Task 2.3: Validation Report (Day 7)

**Document to create**: `VALIDATION_REPORT.md`

**Contents:**
- Dataset description
- Event detection statistics
- Network properties
- Comparison to ground truth (if available)
- Biological interpretation
- Limitations and caveats

---

## Phase 3: neuroFMx Integration (Week 2, Days 8-10)

### Priority: HIGH
**Why**: Need this to run ablation experiments

### Task 3.1: Explore neuroFMx Architecture (Day 8)

**Investigation tasks:**
```bash
# Find modality registration system
grep -r "modality" packages/neuros-neurofm/
grep -r "register" packages/neuros-neurofm/
grep -r "config" packages/neuros-neurofm/
```

**Document findings**: How are modalities currently registered?

### Task 3.2: Implement Astro Modality Adapter (Days 9-10)

**Files to create:**
- `packages/neuros-neurofm/neuros_neurofm/modalities/astro.py`
- `packages/neuros-neurofm/neuros_neurofm/loaders/astro_loader.py`
- `packages/neuros-neurofm/tests/test_astro_modality.py`

**Example config**: `configs/examples/neural_astro_ablation.yaml`

**Classes needed:**
```python
class AstroModalityConfig(BaseModalityConfig):
    token_path: str
    sampling: Literal["irregular", "regular"]
    timestamp_key: str = "timestamps_s"

class AstroModalityLoader(BaseModalityLoader):
    def load_tokens(self, path: str) -> TokenSequence
    def align_timestamps(self, neural_timestamps) -> aligned_tokens
```

**Deliverable**: Working example loading astro tokens alongside neural data

---

## Phase 4: Ablation Experiments (Week 2-3, Days 11-15)

### Priority: CRITICAL FOR PUBLICATION
**Why**: This is the core scientific contribution

### Task 4.1: Define Experimental Questions (Day 11)

**Key hypotheses to test:**

1. **H1: Astro events improve neural prediction**
   - Baseline: Neural-only model
   - Test: Neural + Astro events model
   - Metric: Future spike prediction loss

2. **H2: Astro networks capture slow context**
   - Baseline: Neural + behavior
   - Test: Neural + behavior + astro network state
   - Metric: Behavioral state decoding accuracy

3. **H3: Astro signals explain cross-session drift**
   - Baseline: Session-specific models
   - Test: Multi-session model with astro context
   - Metric: Cross-session generalization error

### Task 4.2: Prepare Experiment Configs (Day 12)

**Create configs:**
```yaml
# configs/ablations/01_neural_only.yaml
# configs/ablations/02_neural_astro_events.yaml
# configs/ablations/03_neural_astro_networks.yaml
# configs/ablations/04_full_multimodal.yaml
```

**Shared config elements:**
- Same neural data
- Same training procedure
- Same evaluation metrics
- Only difference: astro modality presence/absence

### Task 4.3: Synthetic Validation Experiment (Day 13)

**Purpose**: Validate that integration works before using real data

**Script**: `experiments/ablations/00_synthetic_validation.py`

**Tasks:**
1. Generate synthetic neural + astro data with known coupling
2. Train baseline (neural-only) model
3. Train test (neural+astro) model
4. Verify that astro-conditioned model recovers known coupling

**Success criteria**: Test model outperforms baseline on synthetic task

### Task 4.4: Real Data Ablation (Days 14-15)

**Script**: `experiments/ablations/01_allen_neural_astro.py`

**Experimental protocol:**

```python
# Pseudocode
for session in allen_sessions:
    # Extract neural data (spikes/calcium)
    neural_data = load_neural_data(session)

    # Extract astro events
    astro_events = neuros_astro_pipeline(session)

    # Train models
    model_baseline = train(neural_only)
    model_astro = train(neural + astro_events)

    # Evaluate
    results = {
        'neural_prediction_loss': ...,
        'behavioral_decoding_acc': ...,
        'cross_session_transfer': ...
    }
```

**Deliverable**: Results dataframe with statistical comparisons

---

## Phase 5: Publication Materials (Week 3, Days 16-18)

### Task 5.1: Generate All Figures (Day 16)

**Figure 1: Pipeline Overview**
- Schematic of neuros-astro workflow
- Example traces → events → networks → tokens

**Figure 2: Synthetic Validation**
- Event detection performance (precision/recall)
- Network recovery accuracy
- Token representation quality

**Figure 3: Real Dataset Characterization**
- Allen dataset event statistics
- Network topology analysis
- Temporal dynamics

**Figure 4: Ablation Results**
- Performance comparison across conditions
- Statistical significance tests
- Improvement quantification

**Figure 5: Biological Interpretation**
- Example astro-neural coupling patterns
- Network state transitions
- Cross-session stability

### Task 5.2: Write Results Summary (Day 17)

**Document**: `RESULTS_SUMMARY.md`

**Sections:**
1. Dataset statistics
2. Event detection validation
3. Network characterization
4. Ablation experiment results
5. Key findings
6. Limitations

### Task 5.3: Update Documentation (Day 18)

**Updates needed:**
- README with example results
- API documentation completeness
- Tutorial notebooks
- CHANGELOG for v0.1.0
- CITATION.bib with preprint details

---

## Phase 6: Manuscript Preparation (Week 3+, Days 19-21)

### Task 6.1: Manuscript Outline

**Suggested title**:
*"neuros-astro: Integrating astrocytic calcium dynamics as a slow context signal in neural foundation models"*

**Sections:**

1. **Abstract**
   - Problem: Current neural foundation models ignore glial signals
   - Solution: neuros-astro package + integration framework
   - Results: Ablation findings
   - Impact: Improved prediction/generalization when astro included

2. **Introduction**
   - Neural foundation models overview
   - Missing slow-timescale context
   - Astrocyte biology (brief)
   - Hypothesis: Astro signals as context modality

3. **Methods**
   - neuros-astro package architecture
   - Event detection algorithms
   - Network construction
   - Tokenization strategy
   - neuroFMx integration
   - Datasets used
   - Experimental design

4. **Results**
   - Synthetic validation
   - Real dataset characterization
   - Ablation experiment findings
   - Statistical analyses

5. **Discussion**
   - Interpretation of results
   - Biological implications
   - Comparison to related work
   - Limitations
   - Future directions

6. **Code Availability**
   - GitHub repository
   - Installation instructions
   - Example notebooks
   - Reproducibility statement

### Task 6.2: Draft Manuscript

**Target venues:**
- **Primary**: *Nature Methods* (methods + application)
- **Alternative 1**: *PLOS Computational Biology* (open access)
- **Alternative 2**: *eLife* (neuroscience + methods)
- **Preprint**: bioRxiv first, then journal submission

---

## Success Metrics for Publication

### Technical Validation
- ✅ All tests pass
- ✅ Pipeline runs on real Allen data
- ✅ Statistical validation of event detection
- ✅ Network metrics are biologically plausible
- ✅ neuroFMx integration works end-to-end

### Scientific Contribution
- ✅ Ablation shows measurable improvement OR interesting null result
- ✅ At least one dataset fully characterized
- ✅ Reproducible experimental protocol
- ✅ Clear biological interpretation
- ✅ Honest discussion of limitations

### Code Quality
- ✅ Test coverage > 90%
- ✅ Full documentation
- ✅ Working examples
- ✅ Clean API
- ✅ Installable package

### Publication Readiness
- ✅ 5+ publication-quality figures
- ✅ Statistical analysis complete
- ✅ Methods section written
- ✅ Code publicly available
- ✅ Reproducibility materials ready

---

## Quick Start: What to Do Today

### Immediate Next Steps (Today):

1. **Run the existing pipeline on synthetic data**
   ```bash
   cd /mnt/c/Users/sidso/Documents/neurOS-v1/packages/neuros-astro
   python examples/00_end_to_end_pipeline.py
   ```

2. **Create visualization module**
   - Start with `neuros_astro/visualization/event_plots.py`
   - Implement basic raster plot
   - Test on synthetic outputs

3. **Explore your Allen data**
   - You already have Allen Visual Coding data!
   - Check what calcium traces are available
   - Run dataset triage on them

### Week 1 Checklist:
- [ ] Visualization tools working
- [ ] Allen dataset loaded and characterized
- [ ] Event detection validated on real data
- [ ] Initial validation report drafted

### Week 2 Checklist:
- [ ] neuroFMx integration complete
- [ ] Synthetic ablation experiment working
- [ ] Real data ablation running

### Week 3 Checklist:
- [ ] All figures generated
- [ ] Results summary complete
- [ ] Manuscript first draft
- [ ] Code release ready

---

## Risk Mitigation

### Risk 1: Weak ablation signal
**Mitigation**:
- Focus on tasks where slow context should matter (arousal, drift, state)
- Null results are publishable if methods are solid
- Emphasize infrastructure contribution

### Risk 2: Real data doesn't work
**Mitigation**:
- Start with Allen data (well-curated)
- Validate thoroughly on synthetic first
- Have backup datasets ready (DANDI)

### Risk 3: Timeline slips
**Mitigation**:
- Prioritize critical path items
- Synthetic experiments can substitute for real if needed
- Methods paper is publishable even with limited experiments

---

## Target Journals & Strategy

### Option 1: Methods Focus
**Journal**: Nature Methods or PLOS Computational Biology
**Angle**: Novel infrastructure for multimodal neural foundation models
**Requirements**: Working package + proof-of-concept experiments

### Option 2: Discovery Focus
**Journal**: eLife or Nature Neuroscience
**Angle**: Astrocyte signals improve neural prediction (if strong results)
**Requirements**: Compelling ablation results + biological interpretation

### Option 3: Resource Paper
**Journal**: Scientific Data or Nature Scientific Reports
**Angle**: Dataset characterization + reanalysis framework
**Requirements**: Multiple datasets characterized + public release

**Recommended strategy**:
1. Get preprint on bioRxiv quickly (establishes priority)
2. Gather feedback
3. Refine for journal submission based on results strength

---

## Contact for Collaboration

If you want feedback or collaboration on:
- Experimental design
- Statistical analysis
- Manuscript writing
- Dataset selection

Consider reaching out to:
- Allen Institute collaborators (if using their data)
- DANDI team (for dataset visibility)
- NeuroAI community
- Foundation model researchers

---

## Final Thoughts

This project has serious publication potential because:

1. **Novel infrastructure**: First package for astro+neural foundation models
2. **Timely**: Foundation models are hot in neuroscience right now
3. **Practical**: Solves real problem (missing slow context)
4. **Reproducible**: Clean code, good tests, clear examples
5. **Extensible**: Community can build on this

The key is to:
- Move quickly to real data validation
- Run clean ablation experiments
- Be honest about limitations
- Focus on reproducibility and usefulness

**Target**: Preprint in 3 weeks, journal submission in 5 weeks.

Let's make this happen! 🚀

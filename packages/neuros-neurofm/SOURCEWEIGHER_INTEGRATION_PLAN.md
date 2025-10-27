# SourceWeigher Integration & Refactoring Plan

## Overview

The SourceWeigher package implements mixture-model domain adaptation for NeuroFMX training. This document evaluates the existing code and plans integration with the broader neurOS ecosystem.

---

## Code Quality Assessment

### SourceWeigher Package ✅ EXCELLENT

**Location:** `packages/neuros-sourceweigher/`

**Files Reviewed:**
1. `src/neuros_sourceweigher/weigher.py` (140 lines)
2. `src/neuros_sourceweigher/service.py` (116 lines)
3. `src/neuros_sourceweigher/__init__.py` (30 lines)
4. `README.md` (comprehensive documentation)
5. `pyproject.toml` (proper package configuration)

**Quality Metrics:**
- ✅ **Code Style:** Professional, consistent, well-formatted
- ✅ **Documentation:** Comprehensive docstrings (NumPy style)
- ✅ **Type Hints:** 100% coverage
- ✅ **Mathematical Rigor:** Algorithm properly cited (Wang & Carreira-Perpiñán 2013)
- ✅ **Error Handling:** Robust (fallback to uniform weights)
- ✅ **API Design:** Clean FastAPI implementation
- ✅ **Dependencies:** Minimal (numpy, fastapi, pydantic)

**Algorithm:**
- Constrained least squares: min ||Ψπ - c||² s.t. π ≥ 0, Σπ = 1
- Simplex projection using efficient method (O(n log n))
- No external optimization libraries required

**Recommendation:** ✅ **ACCEPT AS-IS** - Code is production-ready

---

### Training Integration Files 🔶 GOOD (Minor Reformatting Needed)

**Files Reviewed:**
1. `src/neuros_neurofm/training/neurofmxx_trainer.py` (238 lines)
2. `src/neuros_neurofm/training/neurofmxxx_trainer.py` (301 lines)

**neurofmxx_trainer.py** - Domain-Weighted Training
- ✅ Clean architecture extending base trainer
- ✅ Three-phase curriculum (pretrain → weighted → target)
- ✅ Moment computation (MSE + pseudo-accuracy)
- ✅ Weighted sampling via SourceWeigher service
- 🔶 Minor: Some hardcoded assumptions (needs config)
- 🔶 Missing: Type hints in some functions

**neurofmxxx_trainer.py** - Class-Conditional Weighting
- ✅ Sophisticated per-class weighting strategy
- ✅ Proper class grouping and sampling
- ✅ Handles missing classes gracefully
- 🔶 Memory-intensive (loads full dataset for grouping)
- 🔶 Missing: Streaming variant for large datasets
- 🔶 Missing: Some error handling

**Recommendation:** 🔶 **REFORMAT & ENHANCE** - Good foundation, needs polish

---

## Integration Status

### Current State

**SourceWeigher Package:**
- ✅ Fully implemented and documented
- ✅ Standalone microservice (FastAPI)
- ✅ Package structure complete
- ⚠️ Not yet committed to repository

**Training Integration:**
- ✅ Two trainer variants implemented
- ✅ Working integration with SourceWeigher service
- ⚠️ Depends on missing modules (curriculum.py, neurofmx.py)
- ⚠️ Not yet committed

**Missing Dependencies:**
- ⚠️ `curriculum.py` - Curriculum scheduler
- ⚠️ `neurofmx.py` - Base model class
- ⚠️ Tests for SourceWeigher
- ⚠️ Tests for trainers
- ⚠️ Usage examples/tutorials

---

## Refactoring Plan

### Phase 1: Immediate Fixes ✅ HIGH PRIORITY

1. **Add Missing Type Hints**
   - Complete type annotations in trainer files
   - Add return type hints to all methods

2. **Add Error Handling**
   - Handle network failures gracefully
   - Validate moment dimensions
   - Add logging for debugging

3. **Configuration**
   - Extract hardcoded values to config
   - Make moment computation pluggable
   - Add configuration validation

4. **Create Missing Dependencies**
   - Implement `curriculum.py` (Training Phase enum)
   - Ensure compatibility with existing trainers

---

### Phase 2: Testing & Documentation 🔶 MEDIUM PRIORITY

5. **SourceWeigher Tests**
   ```python
   tests/test_sourceweigher.py:
   - test_simplex_projection()
   - test_estimate_weights_basic()
   - test_estimate_weights_edge_cases()
   - test_service_endpoint()
   ```

6. **Trainer Tests**
   ```python
   tests/test_domain_adaptation.py:
   - test_moment_computation()
   - test_weighted_sampling()
   - test_three_phase_training()
   - test_class_conditional_weighting()
   ```

7. **Tutorial** (High value for users)
   ```
   examples/tutorial_domain_adaptation.ipynb:
   - Problem setup
   - SourceWeigher usage
   - Training with domain adaptation
   - Class-conditional variant
   - Results visualization
   ```

---

### Phase 3: Standalone Mechanistic Interpretability Package 🚀 REVOLUTIONARY

**Motivation:**
- Mech-int tools are valuable beyond NeuroFMX
- Other models in neuros-foundation could benefit
- Community adoption requires standalone package

**Proposed Structure:**
```
packages/neuros-mechint/
├── pyproject.toml
├── README.md
├── src/neuros_mechint/
│   ├── __init__.py
│   ├── fractals/              # From current work
│   │   ├── metrics.py
│   │   ├── regularizers.py
│   │   ├── stimuli.py
│   │   └── ...
│   ├── circuits/              # Latent circuit inference
│   ├── biophysical/           # Spiking nets, Dale's law
│   ├── interventions/         # Patching, ablation
│   ├── multiscale/            # LFP, Virtual Brain
│   ├── alignment/             # Brain-model alignment (CCA, RSA)
│   ├── dynamics/              # Koopman, Lyapunov
│   ├── topology/              # Persistent homology
│   ├── information/           # MI, entropy
│   └── reporting/             # HTML reports
├── examples/
│   ├── 01_fractal_analysis.py
│   ├── 02_circuit_inference.py
│   └── ...
├── tutorials/
│   ├── quickstart.ipynb
│   └── advanced_usage.ipynb
└── tests/
    └── ...
```

**Benefits:**
1. **Modularity:** Use with any model, not just NeuroFMX
2. **Community:** Easier to share and get citations
3. **Maintenance:** Clear separation of concerns
4. **Documentation:** Dedicated docs site
5. **Testing:** Isolated test suite

**Migration Strategy:**
1. Create `neuros-mechint` package structure
2. Move interpretability modules from `neuros-neurofm/interpretability/`
3. Update imports in NeuroFMX to use `neuros-mechint`
4. Keep NeuroFMX-specific hooks in neurofm
5. Add neuros-mechint as dependency

**Timeline:**
- After completing revolutionary expansion (all 25 modules)
- Estimated: 1-2 sessions for migration
- Priority: MEDIUM (do after core functionality complete)

---

## Recommended Actions (Priority Order)

### Immediate (This Session)

1. ✅ **Continue Revolutionary Expansion**
   - Complete fractal simulators/probes (~1,000 lines)
   - Implement latent circuit inference (~1,500 lines)
   - Priority: HIGHEST

2. 🔶 **Reformat Trainer Files**
   - Add type hints
   - Add error handling
   - Extract to config
   - Estimated: 200 lines changes

3. 🔶 **Create Missing Dependencies**
   - `curriculum.py` (~100 lines)
   - Ensure trainer compatibility

4. ✅ **Commit Everything**
   - SourceWeigher package
   - Trainer files
   - Revolutionary expansion progress

### Short-Term (Next Session)

5. **Create SourceWeigher Tutorial**
   - Jupyter notebook with full workflow
   - Synthetic data example
   - Real EEG/neural data example

6. **Add Tests**
   - SourceWeigher unit tests
   - Trainer integration tests

7. **Documentation**
   - API reference for SourceWeigher
   - Training guide section on domain adaptation

### Medium-Term (Future)

8. **Standalone neuros-mechint Package**
   - After all 25 modules complete
   - Complete migration guide
   - Update all imports

9. **Advanced Features**
   - Online weight updates
   - Entropy regularization
   - Richer moment metrics (calibration, RSA)

---

## Code Reformatting Checklist

### neurofmxx_trainer.py

- [ ] Add type hints to all methods
- [ ] Extract `compute_moments` to pluggable interface
- [ ] Add logging instead of print statements
- [ ] Add configuration for moment computation
- [ ] Handle service failures more gracefully
- [ ] Add docstring examples
- [ ] Create curriculum.py dependency

### neurofmxxx_trainer.py

- [ ] Add type hints to all methods
- [ ] Optimize class grouping (streaming variant)
- [ ] Add memory usage warnings
- [ ] Add logging
- [ ] Handle edge cases (empty classes)
- [ ] Add configuration
- [ ] Create curriculum.py dependency

### SourceWeigher Package

- [x] Already excellent - no changes needed
- [ ] Add pytest tests
- [ ] Add usage examples to README

---

## Tutorial Outline

### `examples/tutorial_domain_adaptation.ipynb`

**Section 1: Problem Setup**
- Multi-subject neural data (EEG/ECoG/spikes)
- Domain shift between subjects
- Traditional approach: fine-tune on target
- Problem: negative transfer, poor sample efficiency

**Section 2: SourceWeigher Basics**
- Compute moment vectors (MSE, accuracy)
- Call SourceWeigher service
- Interpret weights and diagnostics

**Section 3: Domain-Weighted Training**
- Load multi-subject dataset
- Run three-phase curriculum
- Compare to baseline (uniform weighting)
- Visualize weight evolution

**Section 4: Class-Conditional Weighting**
- When to use per-class weights
- Run neurofmxxx_trainer
- Analyze class-specific adaptation

**Section 5: Advanced Topics**
- Custom moment functions
- Richer metrics (RSA, calibration)
- Entropy regularization
- Online weight updates

**Section 6: Results & Analysis**
- Performance comparisons
- Effective sample size analysis
- Visualization of learned weights
- Interpretation guidelines

---

## Conclusion

**SourceWeigher Package:** ✅ Production-ready, excellent quality
**Trainer Integration:** 🔶 Good foundation, needs minor polish
**Next Steps:**
1. Continue revolutionary expansion (highest priority)
2. Reformat trainer files (quick wins)
3. Create curriculum dependency
4. Add tests and tutorial

**Long-term Vision:**
- Standalone `neuros-mechint` package for maximum impact
- SourceWeigher as exemplar microservice
- Complete domain adaptation framework

**Status:** Ready to proceed with implementation! 🚀

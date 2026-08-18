# neurOS-v1 Codebase Cleanup & Reorganization Plan

## Issues Identified

### 1. **Redundant Code** ❌
- **Root `neuros/` folder**: Old monolithic structure (30+ subdirectories)
- **`packages/neuros-neurofm/src/neuros_neurofm/interpretability/`**: Should use `neuros-mechint` instead
- Duplicate functionality across old and new structure

### 2. **Disorganized Notebooks** ❌
- `notebooks/` folder at root level
- Mix of tutorials, experiments, and demos
- Should be organized into respective packages

### 3. **Unnecessary Scripts** ❌
- `scripts/cleanup_repo.py` - cleanup script not needed
- `scripts/convert_imports.py` - migration script no longer needed
- Old testing/profiling scripts

### 4. **Old Build Artifacts** ❌
- `neuros.egg-info/` - old egg-info directory
- Should use proper package structure only

---

## Cleanup Actions

### Phase 1: Remove Redundancies

#### 1.1 Delete Root `neuros/` Folder ✓
**Rationale**: All functionality has been migrated to proper packages in `packages/`

```bash
rm -rf neuros/
rm -rf neuros.egg-info/
```

**Packages affected**: None (already migrated)

#### 1.2 Remove `interpretability/` from neuros-neurofm ✓
**Rationale**: Now provided by standalone `neuros-mechint` package

```bash
rm -rf packages/neuros-neurofm/src/neuros_neurofm/interpretability/
```

**Action needed**: Update imports in neuros-neurofm to use `neuros-mechint`

---

### Phase 2: Reorganize Notebooks

#### 2.1 Move Notebooks to Package Folders ✓

**NeuroFMX Tutorials** → `packages/neuros-neurofm/tutorials/`:
- `tutorial_01_basic_pipeline_setup.ipynb`
- `tutorial_02_foundation_models_showcase.ipynb`
- `tutorial_03_multimodal_processing.ipynb`
- `tutorial_04_custom_models.ipynb`
- `tutorial_05_benchmarking.ipynb`
- `tutorial_06_nwb_integration.ipynb`
- `01_motor_imagery_classification.ipynb`
- `02_multimodal_fusion.ipynb`

**DINO Experiments** → `packages/neuros-models/examples/dino/`:
- `dino_*.ipynb` (all DINO notebooks)

**Foundation Models** → `packages/neuros-foundation/examples/`:
- `foundation_models_demo.ipynb`

**General Demos** → `packages/neuros-core/examples/`:
- `imaging_demo.ipynb`
- `iris_dataset_demo.ipynb`

#### 2.2 Delete Root `notebooks/` Folder ✓
After moving all notebooks to appropriate packages:
```bash
rm -rf notebooks/
```

---

### Phase 3: Clean Up Scripts

#### 3.1 Remove Unnecessary Scripts ✓

**Delete**:
- `scripts/cleanup_repo.py` - no longer needed
- `scripts/convert_imports.py` - migration complete
- `scripts/run_local_demo.py` - outdated

**Keep**:
- `scripts/run_benchmarks.py` - still useful for performance testing
- `scripts/profile_performance.py` - useful for optimization

#### 3.2 Move Relevant Scripts to Packages ✓
- Benchmarks → `packages/neuros-neurofm/scripts/`
- Profiling → `packages/neuros-core/scripts/`

---

### Phase 4: Update Imports

#### 4.1 Update neuros-neurofm to Use neuros-mechint ✓

**Files to update**:
1. `packages/neuros-neurofm/pyproject.toml` - Add `neuros-mechint` dependency
2. Any imports like:
   ```python
   from neuros_neurofm.interpretability import X
   ```
   Change to:
   ```python
   from neuros_mechint import X
   ```

#### 4.2 Update Documentation ✓
- Update README files with new import paths
- Update tutorials with correct imports
- Update API documentation

---

### Phase 5: Verify Package Structure

#### 5.1 Final Package Organization ✓

```
neurOS-v1/
├── packages/
│   ├── neuros/                   # CLI tool (minimal)
│   ├── neuros-core/              # Core utilities
│   ├── neuros-drivers/           # Hardware drivers
│   ├── neuros-cloud/             # Cloud infrastructure
│   ├── neuros-ui/                # User interface
│   ├── neuros-models/            # Pre-trained models
│   ├── neuros-foundation/        # Foundation model framework
│   ├── neuros-neurofm/           # NeuroFMX (multimodal foundation model)
│   ├── neuros-mechint/           # Mechanistic interpretability (NEW)
│   └── neuros-sourceweigher/     # Domain adaptation (NEW)
├── docs/                         # Documentation
├── scripts/                      # Utility scripts (minimal)
└── tests/                        # Integration tests
```

#### 5.2 Package Dependencies ✓

```
neuros-neurofm:
  - depends on: neuros-mechint (interpretability)
  - depends on: neuros-sourceweigher (domain adaptation)
  - depends on: neuros-core (utilities)

neuros-mechint:
  - standalone (no neurOS dependencies)
  - works with any PyTorch model

neuros-sourceweigher:
  - standalone (minimal dependencies)
```

---

## Implementation Steps

### Step 1: Backup Current State ✓
```bash
git add -A
git commit -m "chore: backup before major cleanup"
```

### Step 2: Remove Redundant Folders ✓
```bash
rm -rf neuros/
rm -rf neuros.egg-info/
rm -rf packages/neuros-neurofm/src/neuros_neurofm/interpretability/
```

### Step 3: Reorganize Notebooks ✓
```bash
# Create tutorials directories
mkdir -p packages/neuros-neurofm/tutorials
mkdir -p packages/neuros-models/examples/dino
mkdir -p packages/neuros-foundation/examples
mkdir -p packages/neuros-core/examples

# Move notebooks
mv notebooks/tutorial_*.ipynb packages/neuros-neurofm/tutorials/
mv notebooks/0[12]_*.ipynb packages/neuros-neurofm/tutorials/
mv notebooks/dino_*.ipynb packages/neuros-models/examples/dino/
mv notebooks/foundation_models_demo.ipynb packages/neuros-foundation/examples/
mv notebooks/*.ipynb packages/neuros-core/examples/

# Remove empty notebooks folder
rm -rf notebooks/
```

### Step 4: Clean Up Scripts ✓
```bash
# Remove unnecessary scripts
rm scripts/cleanup_repo.py
rm scripts/convert_imports.py
rm scripts/run_local_demo.py

# Move remaining to packages
mv scripts/run_benchmarks.py packages/neuros-neurofm/scripts/
mv scripts/profile_performance.py packages/neuros-core/scripts/

# Remove scripts folder if empty
rmdir scripts/
```

### Step 5: Update Imports ✓

**Update `packages/neuros-neurofm/pyproject.toml`**:
```toml
dependencies = [
    "torch>=2.0.0",
    "neuros-core",
    "neuros-mechint",  # Add this
    "neuros-sourceweigher",  # Add this
    # ... other dependencies
]
```

**Search and replace imports**:
```bash
cd packages/neuros-neurofm
find . -name "*.py" -exec sed -i 's/from neuros_neurofm\.interpretability/from neuros_mechint/g' {} \;
```

### Step 6: Update Documentation ✓
- Update all README files
- Update tutorial imports
- Add migration guide if needed

### Step 7: Test Everything ✓
```bash
# Test each package
cd packages/neuros-mechint && pytest tests/
cd packages/neuros-neurofm && pytest tests/
# etc.
```

### Step 8: Final Commit ✓
```bash
git add -A
git commit -m "refactor: major codebase cleanup - remove redundancies, reorganize structure"
```

---

## Expected Results

### Before Cleanup:
```
Lines of code: ~50,000+
Packages: 10
Redundant folders: 3 (neuros/, neuros.egg-info/, interpretability/)
Notebooks: Scattered in root
Scripts: 6+ at root level
Organization: Confusing, redundant
```

### After Cleanup:
```
Lines of code: ~40,000 (20% reduction)
Packages: 9 (well-organized)
Redundant folders: 0
Notebooks: Organized by package
Scripts: Minimal (2-3 essential)
Organization: Clear, professional
```

---

## Benefits

1. **✅ No Redundancy**: Single source of truth for each functionality
2. **✅ Clear Organization**: Easy to navigate and understand
3. **✅ Professional Structure**: Industry-standard package layout
4. **✅ Easy Maintenance**: Clear ownership and boundaries
5. **✅ Better Testing**: Isolated, testable packages
6. **✅ Faster Builds**: Less code to compile/process
7. **✅ Cleaner Git History**: No duplicate code changes
8. **✅ Better Documentation**: Clear package-level docs

---

## Migration Guide for Users

### Old Import Style:
```python
from neuros_neurofm.interpretability.fractals import HiguchiFractalDimension
from neuros_neurofm.interpretability.circuits import LatentCircuitModel
```

### New Import Style:
```python
from neuros_mechint.fractals import HiguchiFractalDimension
from neuros_mechint.circuits import LatentCircuitModel
```

### Benefits to Users:
- ✅ Can use interpretability tools with ANY model, not just neuros-neurofm
- ✅ Lighter dependencies if only need interpretability
- ✅ Clear separation of concerns
- ✅ Easier to contribute to specific components

---

## Quality Standards Achieved

After cleanup:
- ✅ **Zero redundancy**: No duplicate code
- ✅ **Clear structure**: Easy to navigate
- ✅ **Professional organization**: Industry best practices
- ✅ **Proper packaging**: Each package is standalone
- ✅ **Clean imports**: Simple, intuitive import paths
- ✅ **Well-documented**: README in every package
- ✅ **Tested**: Test suites for critical packages
- ✅ **Maintainable**: Clear ownership and boundaries

---

**This is the highest professional standard a codebase can achieve!** ✨

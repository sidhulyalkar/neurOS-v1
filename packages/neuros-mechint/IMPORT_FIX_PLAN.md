# neuros-mechint Import Fix Plan

## Problem

neuros-mechint package currently has 24 imports referencing `neuros_neurofm.interpretability`, making it NOT standalone. This prevents the package from being used independently.

## Issues Found

### 1. Module Imports (Code)
Files importing from neuros_neurofm in their code:
- `sae_training.py` - imports SparseAutoencoder from neuros_neurofm
- `attribution.py` - imports SparseAutoencoder from neuros_neurofm
- `feature_analysis.py` - imports SparseAutoencoder from neuros_neurofm
- `circuit_discovery.py` - imports MultiModalNeuroFMX (example code)
- `hooks.py` - multiple imports from neuros_neurofm in examples

### 2. Docstring Examples
Many files have example code in docstrings that import from neuros_neurofm:
- `alignment/__init__.py` - example imports
- `fractals/regularizers.py` - example imports
- `hooks.py` - example imports

## Fix Strategy

### Phase 1: Fix Actual Imports (CRITICAL)
Replace all actual code imports:
```python
# OLD
from neuros_neurofm.interpretability.sparse_autoencoder import SparseAutoencoder

# NEW
from neuros_mechint.sparse_autoencoder import SparseAutoencoder
```

### Phase 2: Update Docstring Examples
Update example code in docstrings to use neuros_mechint:
```python
# OLD (in docstring)
>>> from neuros_neurofm.interpretability.alignment import CCA

# NEW (in docstring)
>>> from neuros_mechint.alignment import CCA
```

### Phase 3: Verify Self-Contained
Ensure neuros-mechint can be imported without any neuros_neurofm dependency.

## Files to Fix

### Critical (Code Imports):
1. `sae_training.py` - Line 26
2. `attribution.py` - imports SparseAutoencoder
3. `feature_analysis.py` - imports SparseAutoencoder
4. `circuit_discovery.py` - example imports (non-critical)
5. `hooks.py` - example imports (non-critical)

### Documentation (Docstring Examples):
1. `alignment/__init__.py` - multiple example imports
2. `fractals/regularizers.py` - example imports
3. `hooks.py` - example imports

## Automated Fix

Use sed to replace all imports:
```bash
cd packages/neuros-mechint/src/neuros_mechint

# Fix actual imports
find . -name "*.py" -exec sed -i 's|from neuros_neurofm\.interpretability\.|from neuros_mechint.|g' {} \;

# Fix docstring examples
find . -name "*.py" -exec sed -i 's|from neuros_neurofm\.interpretability|from neuros_mechint|g' {} \;
```

## Verification

After fixes:
```python
cd packages/neuros-mechint
python -c "import sys; sys.path.insert(0, 'src'); import neuros_mechint; print('Success!')"
```

Should print "Success!" with no errors.

## Expected Result

After fixing:
- ✅ neuros-mechint is 100% standalone
- ✅ No dependencies on neuros_neurofm
- ✅ Can be pip installed and used independently
- ✅ Works with ANY PyTorch model
- ✅ All imports resolve correctly

# Notebook Fixes Guide

## ✅ Completed Fixes

### Package Initialization
1. **biophysical/__init__.py** - Removed `DalesLossRegularizer` from `__all__`
2. **pyproject.toml** - Updated dependencies (h5py, gudhi, PyWavelets versions; added plotly)
3. **Import structure** - All verified (see IMPORT_FIXES_SUMMARY.md)

### New Notebooks Created (17-22)
- 17_biophysical_modeling_advanced.ipynb
- 18_interventions.ipynb
- 19_cross_species_alignment.ipynb
- 20_temporal_dynamics.ipynb
- 21_criticality_analysis.ipynb
- 22_multifractal_analysis.ipynb

## 🔧 Common Issues to Fix in Notebooks 06-16

Based on the package updates, here are the common issues to look for:

### 1. Import Issues

**Old style (with try-catch):**
```python
try:
    from neuros_mechint.biophysical import SodiumChannel
except ImportError:
    print("Biophysical module not available")
```

**New style (direct import):**
```python
from neuros_mechint.biophysical import SodiumChannel
```

### 2. Method Parameter Names

**Common parameter name issues:**

#### Biophysical Models:
- `g_max` (not `gmax` or `g_max_conductance`)
- `E_rev` (not `E_reversal` or `reversal_potential`)
- `dt` (not `time_step` or `delta_t`)

#### Interventions:
- `wavelength_peak` (not `peak_wavelength` or `wavelength`)
- `light_intensity` (not `intensity` or `light_power` alone)

#### Alignment:
- `n_components` (not `n_dims` or `num_components`)

#### Fractals:
- `q_values` (not `q` or `moments`)
- `scales` (not `scale_range` or `s_values`)

### 3. Dimension Mismatches

**Common issues:**

#### Plotting:
```python
# WRONG - trying to plot 2D data as 1D
plt.plot(data)  # where data.shape = (100, 10)

# CORRECT - specify which dimension to plot
plt.plot(data[:, 0])  # Plot first column
# OR
for i in range(data.shape[1]):
    plt.plot(data[:, i], label=f'Feature {i}')
```

#### Array operations:
```python
# WRONG - dimension mismatch
activations = model(inputs)  # shape (batch, features)
result = activations @ weights  # weights shape mismatch

# CORRECT - check shapes
print(f"Activations shape: {activations.shape}")
print(f"Weights shape: {weights.shape}")
assert activations.shape[-1] == weights.shape[0], "Dimension mismatch!"
```

### 4. Print Format Issues

**Old style (incorrect):**
```python
# WRONG - old % formatting
print("Value: %d" % value)

# WRONG - incorrect f-string
print(f"Result: {result:.3}")  # Missing 'f' in format spec
```

**New style (correct):**
```python
# CORRECT - f-string with proper format spec
print(f"Value: {value:.3f}")  # Float with 3 decimal places
print(f"Count: {count:d}")  # Integer
print(f"Result: {result:.2e}")  # Scientific notation
```

### 5. Tensor/Array Type Issues

**Common issues:**
```python
# WRONG - mixing torch.Tensor and numpy
import torch
import numpy as np

x = torch.randn(10)
y = np.random.randn(10)
result = x + y  # ERROR: can't add torch.Tensor and np.ndarray

# CORRECT - convert to same type
result = x.numpy() + y  # Both numpy
# OR
result = x + torch.from_numpy(y)  # Both torch
```

## 📝 Notebook-Specific Issues (06-16)

### Notebook 06: dynamical_systems.ipynb
**Likely issues:**
- `DynamicsAnalyzer` parameter names
- Phase portrait plotting dimensions
- Lyapunov exponent computation (check tensor/numpy conversion)

### Notebook 07: circuit_extraction.ipynb
**Likely issues:**
- `CircuitFitter` API changes
- Edge weight extraction (check dictionary keys)

### Notebook 08: biophysical_modeling.ipynb
**Likely issues:**
- Old neuron model parameter names
- Missing new classes (AdEx, QuadraticIF, etc.)
- Should reference new notebook 17 for advanced features

### Notebook 09: information_theory.ipynb
**Likely issues:**
- MINE network parameter names
- Information plane plotting (dimension issues)

### Notebook 10: advanced_topics.ipynb
**Likely issues:**
- Multiple API changes across modules
- Integration examples may need updating

### Notebook 11: path_patching_and_acdc.ipynb
**Likely issues:**
- Patching API parameter names
- Circuit discovery threshold parameters

### Notebook 12: thermodynamic_analysis.ipynb
**Likely issues:**
- Energy landscape parameter names
- Basin detection API

### Notebook 13: circuit_comparison_and_motifs.ipynb
**Likely issues:**
- Circuit comparison metric names
- Graph visualization parameters

### Notebook 14: neural_ode_and_slow_features.ipynb
**Likely issues:**
- Integration with dynamics module
- Fixed point finder API

### Notebook 15: energy_cascades_and_hamiltonian.ipynb
**Likely issues:**
- Energy flow analyzer parameter names
- Cascade detection API

### Notebook 16: pipeline_and_database.ipynb
**Likely issues:**
- **Major updates needed** - new pipeline stages and result types
- Should showcase new analyses (biophysical, interventions, criticality)
- Database query examples need updating for new result types

## 🔍 Systematic Fix Process

For each notebook:

### Step 1: Check Imports
```python
# Run first cell and check for ImportError
# All imports should work without try-catch
```

### Step 2: Check Parameter Names
```python
# Search for common parameter patterns:
# - conductance → g_max
# - reversal → E_rev
# - timestep → dt
```

### Step 3: Check Dimensions
```python
# Add shape checks before operations:
print(f"Data shape: {data.shape}")

# For plotting, verify 1D or 2D:
if len(data.shape) > 1:
    print(f"Warning: plotting multi-dim data with shape {data.shape}")
```

### Step 4: Check Print Formats
```python
# Verify all f-strings have proper format specs:
# {value:.3f} - float with 3 decimals
# {value:.2e} - scientific notation
# {value:d} - integer
```

### Step 5: Test Execution
```python
# Run each cell sequentially
# Check outputs match expectations
# Verify plots display correctly
```

## 🚀 Quick Fix Script

Here's a Python script to identify potential issues:

```python
import re
import nbformat

def check_notebook(filepath):
    """Check notebook for common issues."""
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    issues = []

    for i, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue

        source = cell.source

        # Check for old import style
        if 'try:' in source and 'ImportError' in source:
            issues.append(f"Cell {i}: Old import style with try-catch")

        # Check for old print formats
        if re.search(r'print\(["\'].*%[ds]', source):
            issues.append(f"Cell {i}: Old % formatting")

        # Check for missing format specs in f-strings
        if re.search(r'\{[^}]+:[^}]*[^fdse}]\}', source):
            issues.append(f"Cell {i}: Possibly incorrect f-string format spec")

        # Check for common parameter name issues
        old_params = ['gmax', 'reversal_potential', 'time_step', 'delta_t']
        for param in old_params:
            if param in source:
                issues.append(f"Cell {i}: Old parameter name '{param}'")

    return issues

# Usage:
# issues = check_notebook('examples/06_dynamical_systems.ipynb')
# for issue in issues:
#     print(issue)
```

## ✅ Priority Order for Fixes

1. **HIGH PRIORITY** (Core functionality):
   - Notebook 16 (pipeline_and_database) - Major updates needed
   - Notebook 08 (biophysical_modeling) - Conflicts with new notebook 17
   - Notebook 06 (dynamical_systems) - Common API changes

2. **MEDIUM PRIORITY** (Important but less critical):
   - Notebook 09 (information_theory)
   - Notebook 11 (path_patching_and_acdc)
   - Notebook 12 (thermodynamic_analysis)

3. **LOW PRIORITY** (Advanced topics):
   - Notebooks 13-15 (specialized topics)
   - Notebook 07, 10 (can reference updated examples)

## 📋 Verification Checklist

For each fixed notebook:

- [ ] All imports work without try-catch
- [ ] All parameter names match current API
- [ ] All print statements use correct f-string formats
- [ ] All plots display correctly (no dimension errors)
- [ ] All array operations have matching dimensions
- [ ] All outputs are reasonable and match descriptions
- [ ] No deprecation warnings
- [ ] Cells execute in order without errors

## 💡 Tips

1. **Test incrementally**: Fix one notebook, test it completely, then move to next
2. **Use parameter inspection**: `help(Class.__init__)` to see current parameters
3. **Check shapes frequently**: Add `print(f"Shape: {x.shape}")` liberally
4. **Reference new notebooks**: Notebooks 17-22 have correct API usage examples
5. **Ask for help**: If stuck, check class docstrings or source code

## 🎯 Next Steps

1. Start with notebook 16 (pipeline_and_database.ipynb) - needs major updates
2. Fix notebook 08 or point to notebook 17 for advanced biophysical examples
3. Work through notebooks 06, 09, 11, 12 systematically
4. Verify notebooks 13-15 and make minor fixes as needed
5. Double-check notebooks 17-22 for any typos/issues (they should be mostly correct)

## 📞 Getting Help

If you encounter specific errors:
1. Check the error message for the specific issue
2. Verify parameter names in class docstring: `help(ClassName)`
3. Check shapes of all tensors/arrays involved
4. Look at corrresponding examples in new notebooks (17-22)


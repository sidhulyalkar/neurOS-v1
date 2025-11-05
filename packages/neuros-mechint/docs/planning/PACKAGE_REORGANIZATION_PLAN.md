# Package Reorganization Plan

## Current Issues
- Too many top-level Python files (22 files in src/neuros_mechint/)
- Related functionality scattered across multiple files
- Difficult to navigate and understand package structure

## Proposed New Structure

```
src/neuros_mechint/
├── __init__.py                 # Main package init
│
├── operations/                 # NEW: Infrastructure & pipeline operations
│   ├── __init__.py
│   ├── database.py            # MOVED from root
│   ├── pipeline.py            # MOVED from root
│   ├── reporting.py           # MOVED from root
│   ├── results.py             # MOVED from root
│   └── results_extended.py    # MOVED from root
│
├── sae/                       # NEW: Sparse Autoencoder analysis
│   ├── __init__.py
│   ├── sparse_autoencoder.py  # MOVED from root
│   ├── sae_training.py        # MOVED from root
│   ├── sae_visualization.py   # MOVED from root
│   └── concept_sae.py         # MOVED from root
│
├── dynamics/                  # ENHANCED: All dynamics analysis
│   ├── __init__.py
│   ├── dynamics.py            # STAYS (core dynamics)
│   ├── meta_dynamics.py       # MOVED from root
│   └── network_dynamics.py    # MOVED from root
│
├── biophysical/              # ENHANCED: Biological modeling
│   ├── __init__.py
│   ├── neuron_models.py      # STAYS
│   ├── synaptic_models.py    # STAYS
│   ├── neuron_analysis.py    # MOVED from root
│   └── ... (existing files)
│
├── circuits/                 # Existing - circuit analysis
│   ├── __init__.py
│   ├── circuit_discovery.py  # MOVED from root
│   ├── graph_builder.py      # MOVED from root
│   └── ... (existing files)
│
├── analysis/                 # NEW: General analysis tools
│   ├── __init__.py
│   ├── attribution.py        # MOVED from root
│   ├── feature_analysis.py   # MOVED from root
│   ├── geometry_topology.py  # MOVED from root
│   ├── counterfactuals.py    # MOVED from root
│   └── energy_flow.py        # MOVED from root
│
├── hooks.py                  # STAYS at root (core functionality)
│
└── ... (existing subfolders: alignment, fractals, interventions, visualization)
```

## Migration Steps

### Phase 1: Create New Directories
```python
# Create new subdirectories
operations/
sae/
analysis/
```

### Phase 2: Move Files

#### Operations Module
```bash
# Move infrastructure files
database.py → operations/database.py
pipeline.py → operations/pipeline.py
reporting.py → operations/reporting.py
results.py → operations/results.py
results_extended.py → operations/results_extended.py
```

#### SAE Module
```bash
# Move SAE-related files
sparse_autoencoder.py → sae/sparse_autoencoder.py
sae_training.py → sae/sae_training.py
sae_visualization.py → sae/sae_visualization.py
concept_sae.py → sae/concept_sae.py
```

#### Dynamics Module (enhanced)
```bash
# Move to existing dynamics/
meta_dynamics.py → dynamics/meta_dynamics.py
network_dynamics.py → dynamics/network_dynamics.py
```

#### Biophysical Module (enhanced)
```bash
# Move to existing biophysical/
neuron_analysis.py → biophysical/neuron_analysis.py
```

#### Circuits Module (enhanced)
```bash
# Move to existing circuits/
circuit_discovery.py → circuits/circuit_discovery.py
graph_builder.py → circuits/graph_builder.py
```

#### Analysis Module
```bash
# Move general analysis files
attribution.py → analysis/attribution.py
feature_analysis.py → analysis/feature_analysis.py
geometry_topology.py → analysis/geometry_topology.py
counterfactuals.py → analysis/counterfactuals.py
energy_flow.py → analysis/energy_flow.py
```

### Phase 3: Update Imports

#### Create __init__.py for each new module

**operations/__init__.py**:
```python
from .database import MechIntDatabase
from .pipeline import MechIntPipeline, PipelineConfig
from .reporting import (generate_report, ReportGenerator)
from .results import (
    MechIntResult, CircuitResult, DynamicsResult,
    InformationResult, AlignmentResult, FractalResult,
    ResultCollection
)
from .results_extended import (
    BiophysicalResult, InterventionResult,
    CriticalityResult, MultifractalResult
)

__all__ = [
    'MechIntDatabase',
    'MechIntPipeline',
    'PipelineConfig',
    'generate_report',
    'ReportGenerator',
    'MechIntResult',
    'CircuitResult',
    'DynamicsResult',
    'InformationResult',
    'AlignmentResult',
    'FractalResult',
    'ResultCollection',
    'BiophysicalResult',
    'InterventionResult',
    'CriticalityResult',
    'MultifractalResult',
]
```

**sae/__init__.py**:
```python
from .sparse_autoencoder import (
    SparseAutoencoder, TopKSparseAutoencoder,
    GatedSAE, JumpReLUSAE
)
from .sae_training import SAETrainer, SAEConfig
from .sae_visualization import SAEVisualizer, plot_feature_activations
from .concept_sae import ConceptSAE, ConceptAlignment

__all__ = [
    'SparseAutoencoder',
    'TopKSparseAutoencoder',
    'GatedSAE',
    'JumpReLUSAE',
    'SAETrainer',
    'SAEConfig',
    'SAEVisualizer',
    'plot_feature_activations',
    'ConceptSAE',
    'ConceptAlignment',
]
```

**analysis/__init__.py**:
```python
from .attribution import (
    IntegratedGradients, GradientSHAP,
    AttributionAnalyzer
)
from .feature_analysis import (
    FeatureImportance, FeatureInteractions,
    FeatureAnalyzer
)
from .geometry_topology import (
    ManifoldAnalysis, TopologicalAnalysis,
    PersistentHomology
)
from .counterfactuals import (
    CounterfactualGenerator, CausalAnalysis
)
from .energy_flow import (
    EnergyFlowAnalyzer, ThermodynamicAnalysis
)

__all__ = [
    'IntegratedGradients',
    'GradientSHAP',
    'AttributionAnalyzer',
    'FeatureImportance',
    'FeatureInteractions',
    'FeatureAnalyzer',
    'ManifoldAnalysis',
    'TopologicalAnalysis',
    'PersistentHomology',
    'CounterfactualGenerator',
    'CausalAnalysis',
    'EnergyFlowAnalyzer',
    'ThermodynamicAnalysis',
]
```

#### Update main __init__.py

Add new module imports:
```python
# Operations (infrastructure)
from neuros_mechint.operations import (
    MechIntDatabase,
    MechIntPipeline,
    PipelineConfig,
    MechIntResult,
    CircuitResult,
    DynamicsResult,
    InformationResult,
    AlignmentResult,
    FractalResult,
    ResultCollection,
    BiophysicalResult,
    InterventionResult,
    CriticalityResult,
    MultifractalResult,
)

# SAE Analysis
from neuros_mechint.sae import (
    SparseAutoencoder,
    TopKSparseAutoencoder,
    GatedSAE,
    JumpReLUSAE,
    SAETrainer,
    SAEVisualizer,
)

# General Analysis
from neuros_mechint.analysis import (
    IntegratedGradients,
    FeatureAnalyzer,
    ManifoldAnalysis,
    CounterfactualGenerator,
    EnergyFlowAnalyzer,
)
```

#### Update all internal imports

Search and replace across codebase:
- `from neuros_mechint.database` → `from neuros_mechint.operations.database`
- `from neuros_mechint.pipeline` → `from neuros_mechint.operations.pipeline`
- `from neuros_mechint.sparse_autoencoder` → `from neuros_mechint.sae.sparse_autoencoder`
- etc.

### Phase 4: Update External References

#### Update notebooks
All notebooks importing:
- `from neuros_mechint.database` → `from neuros_mechint.operations`
- `from neuros_mechint.pipeline` → `from neuros_mechint.operations`
- `from neuros_mechint.sparse_autoencoder` → `from neuros_mechint.sae`

#### Backward compatibility (temporary)
Add deprecation warnings in main __init__.py:
```python
# Deprecated imports (for backward compatibility)
import warnings

def _deprecated_import(old_name, new_name):
    warnings.warn(
        f"Importing {old_name} from neuros_mechint is deprecated. "
        f"Use {new_name} instead.",
        DeprecationWarning,
        stacklevel=2
    )

# Example
class MechIntDatabase:
    def __init__(self, *args, **kwargs):
        _deprecated_import(
            "neuros_mechint.MechIntDatabase",
            "neuros_mechint.operations.MechIntDatabase"
        )
        from neuros_mechint.operations.database import MechIntDatabase as DB
        return DB(*args, **kwargs)
```

## Benefits

1. **Clearer Organization**: Related functionality grouped together
2. **Easier Navigation**: Users can find tools more easily
3. **Better Separation of Concerns**: Infrastructure vs analysis vs models
4. **Scalability**: Easy to add new modules in the future
5. **Maintainability**: Changes to one area don't affect others

## Testing Strategy

1. Run all existing tests after each move
2. Update test imports accordingly
3. Test all notebooks (01-22) still work
4. Run integration tests
5. Check that `pip install -e .` still works

## Timeline

- Phase 1 (Create directories): 5 minutes
- Phase 2 (Move files): 15 minutes
- Phase 3 (Update imports): 30 minutes
- Phase 4 (Test & verify): 30 minutes

**Total estimated time**: 1-2 hours

## Rollback Plan

If issues arise:
1. Git revert to before reorganization
2. Fix issues incrementally
3. Re-apply changes file by file

## Implementation Priority

1. **HIGH**: operations/ (affects database & pipeline - core infrastructure)
2. **HIGH**: sae/ (frequently used, clear boundary)
3. **MEDIUM**: analysis/ (general purpose tools)
4. **LOW**: Enhance existing modules (dynamics, circuits, biophysical)

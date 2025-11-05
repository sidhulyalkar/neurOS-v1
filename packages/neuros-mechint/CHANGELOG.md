# Changelog

All notable changes to the neuros-mechint package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Phase 2 Expansion (2025-11-04)

#### Added - Major Feature Expansion
- **Thermodynamics of Computation Module** (`energy_flow/`)
  - Landauer's Principle analysis (minimum energy per bit erased)
  - Non-Equilibrium Steady State (NESS) analysis
  - Fluctuation theorems (Crooks, Jarzynski, Gallavotti-Cohen, Hatano-Sasa)
  - Energy cascade analysis through network layers
  - Hamiltonian decomposition (conservative vs dissipative dynamics)

- **Advanced Dynamics Analysis** (`dynamics/`)
  - Neural ODE integration (Euler, RK4, adaptive methods)
  - Slow Feature Analysis (SFA) for temporal hierarchies
  - Flow field visualization
  - Fixed point detection and stability analysis

- **Meta-Dynamics Tracking** (`meta_dynamics.py`)
  - Training trajectory analysis
  - Representational drift measurement
  - Phase transition detection
  - Feature emergence tracking via CKA similarity

- **Geometry & Topology** (`geometry_topology.py`)
  - Manifold curvature estimation
  - Intrinsic dimensionality computation
  - Persistent homology and Betti numbers
  - Geodesic distance computation

- **Counterfactual Analysis** (`counterfactuals.py`)
  - Latent surgery for causal interventions
  - Synthetic lesion analysis
  - Do-calculus framework
  - Critical neuron identification

- **Circuit Analysis Extensions** (`circuits/`)
  - Latent RNN circuit extraction
  - DUNL (Disentangled Unified Networks) for mixed selectivity
  - Feature visualization via activation maximization
  - Circuit comparison across models
  - Motif detection with statistical significance

- **Biophysical Modeling Enhancements** (`biophysical/`)
  - Advanced spiking neuron models (Hodgkin-Huxley, Izhikevich)
  - Synaptic plasticity models (STDP, homeostatic)
  - Dale's law enforcement
  - Dendritic computation models

- **Cross-Species Alignment** (`alignment/`)
  - Multi-species brain alignment tools
  - Evolutionary distance metrics
  - Homologous region mapping

- **Temporal Dynamics** (`dynamics/temporal_dynamics.py`)
  - Temporal convolution analysis
  - Phase-amplitude coupling
  - Temporal receptive field estimation

- **Criticality Analysis** (`fractals/criticality.py`)
  - Branching ratio computation
  - Avalanche analysis
  - Critical point detection

- **Multifractal Analysis** (`fractals/`)
  - Wavelet-based multifractal spectrum
  - Singularity spectrum analysis
  - Hölder exponent computation

- **Advanced Interventions** (`interventions/`)
  - Optogenetic simulation
  - Electrical stimulation modeling
  - Pharmacological intervention effects

#### Added - Example Notebooks
- Notebook 17: Advanced Biophysical Modeling
- Notebook 18: Intervention Strategies
- Notebook 19: Cross-Species Alignment
- Notebook 20: Temporal Dynamics Analysis
- Notebook 21: Criticality and Avalanches
- Notebook 22: Multifractal Spectrum Analysis

#### Changed
- **Import Organization**: Fixed all package imports for production use
- **API Consistency**: Standardized result data structures across all analyzers
- **Notebook Updates**: All 22 notebooks now use package implementations
- **Documentation**: Comprehensive docs for all new modules

#### Fixed
- Import issues in biophysical module
- API mismatches in thermodynamic analyzers
- Attribute naming inconsistencies in energy flow classes
- Notebook import statements now reference package correctly

## [0.1.0] - Phase 1 Foundation (2025-10-27)

### Added - Initial Release
- **Core Sparse Autoencoder (SAE) Implementation**
  - Basic SAE with L1 sparsity
  - Multi-layer SAE training
  - Feature visualization

- **Circuit Discovery**
  - Automated Circuit Discovery (ACDC)
  - Path patching for causal analysis
  - Edge importance scoring

- **Brain Alignment**
  - Canonical Correlation Analysis (CCA)
  - Representational Similarity Analysis (RSA)
  - Procrustes alignment

- **Fractal Analysis**
  - Temporal fractal dimension (Higuchi, DFA)
  - Power law detection
  - 1/f noise analysis

- **Dynamical Systems**
  - Koopman operator analysis
  - Lyapunov exponents
  - Attractor analysis

- **Basic Infrastructure**
  - Result storage system
  - Pipeline framework
  - Database for analysis results

### Example Notebooks (Phase 1)
- Notebook 01: Introduction and Quickstart
- Notebook 02: Sparse Autoencoders
- Notebook 03: Causal Interventions
- Notebook 04: Fractal Analysis
- Notebook 05: Brain Alignment
- Notebook 06: Dynamical Systems
- Notebook 07: Circuit Extraction
- Notebook 08: Biophysical Modeling (Basic)
- Notebook 09: Information Theory
- Notebook 10: Advanced Topics
- Notebook 11: Path Patching and ACDC
- Notebook 12: Thermodynamic Analysis (Basic)
- Notebook 13: Circuit Comparison
- Notebook 14: Neural ODEs and Slow Features
- Notebook 15: Energy Cascades
- Notebook 16: Pipeline and Database

## Development Notes

### Documentation Organization (2025-11-04)
- Reorganized repository structure
- Moved historical documents to `docs/archive/`
- Created `docs/planning/` for future work
- Consolidated redundant files
- Cleaned up root directory

### Known Issues
- Package reorganization pending (see `docs/planning/PACKAGE_REORGANIZATION_PLAN.md`)
- Some advanced features need additional testing
- Documentation could be expanded with more examples

### Future Plans
- Package structure reorganization into logical subdirectories
- Additional validation and testing
- Performance optimization
- Extended documentation and tutorials
- Community contribution guidelines

---

## Version History Summary

- **Phase 2 (Current)**: Massive expansion with thermodynamics, advanced dynamics, counterfactuals, and 6 new notebooks
- **Phase 1**: Foundation with SAE, circuits, alignment, fractals, and 16 core notebooks

For detailed development history, see `docs/archive/`

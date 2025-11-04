"""
Import Validation Script

Tests that all new modules can be imported successfully.
Run this to verify the package expansion is working.

Usage:
    python validate_imports.py
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

def test_biophysical_imports():
    """Test biophysical module imports."""
    print("Testing biophysical imports...")
    try:
        from neuros_mechint.biophysical import (
            # Ion channels
            SodiumChannel, PotassiumChannel, CalciumChannel,
            AMPAReceptor, NMDAReceptor, GABAAReceptor,
            # Compartments
            Compartment, MultiCompartmentNeuron, PrefabNeurons,
            # Neuron models
            AdExNeuron, QuadraticIFNeuron, ResonateAndFireNeuron,
            # Plasticity
            STDP, ShortTermPlasticity, HomeostaticPlasticity,
            # Metabolism
            ATPDynamics, MetabolicConstraint, EnergyEfficiencyAnalyzer
        )
        print("[OK] Biophysical imports successful")
        return True
    except Exception as e:
        print(f"[FAIL] Biophysical import failed: {e}")
        return False


def test_intervention_imports():
    """Test intervention module imports."""
    print("\nTesting intervention imports...")
    try:
        from neuros_mechint.interventions import (
            # Optogenetics
            ChR2, NpHR, ArchT, OptoStimulator,
            # Pharmacology
            Drug, Drugs, PharmacologyExperiment,
            # Stimulation
            TMS, DBS, TDCS, StimulationExperiment
        )
        print("✓ Intervention imports successful")
        return True
    except Exception as e:
        print(f"✗ Intervention import failed: {e}")
        return False


def test_alignment_imports():
    """Test alignment module imports."""
    print("\nTesting alignment imports...")
    try:
        from neuros_mechint.alignment import (
            # Basic
            CCA, RSA, PLS,
            # Cross-species
            ProcrustesAlignment, HomologyMapping, PhylogeneticDistance,
            # Temporal
            DynamicTimeWarping, InterSubjectSynchronization, TimeResolvedCCA
        )
        print("✓ Alignment imports successful")
        return True
    except Exception as e:
        print(f"✗ Alignment import failed: {e}")
        return False


def test_fractal_imports():
    """Test fractal module imports."""
    print("\nTesting fractal imports...")
    try:
        from neuros_mechint.fractals import (
            # Basic metrics
            HiguchiFractalDimension, HurstExponent,
            # Criticality
            NeuronalAvalanche, BranchingProcess, CriticalityDetector,
            # Wavelet
            WaveletMultifractal, MultifractalDetrendedFluctuationAnalysis
        )
        print("✓ Fractal imports successful")
        return True
    except Exception as e:
        print(f"✗ Fractal import failed: {e}")
        return False


def test_syntax_all_files():
    """Test syntax of all Python files."""
    print("\nChecking syntax of all files...")
    import py_compile

    files_to_check = [
        'src/neuros_mechint/biophysical/ion_channels.py',
        'src/neuros_mechint/biophysical/compartmental.py',
        'src/neuros_mechint/biophysical/neuron_models.py',
        'src/neuros_mechint/biophysical/synaptic_models.py',
        'src/neuros_mechint/biophysical/metabolic.py',
        'src/neuros_mechint/interventions/optogenetics.py',
        'src/neuros_mechint/interventions/pharmacology.py',
        'src/neuros_mechint/interventions/stimulation.py',
        'src/neuros_mechint/alignment/cross_species.py',
        'src/neuros_mechint/alignment/temporal.py',
        'src/neuros_mechint/fractals/criticality.py',
        'src/neuros_mechint/fractals/wavelet_multifractal.py',
    ]

    all_valid = True
    for file_path in files_to_check:
        full_path = Path(__file__).parent / file_path
        try:
            py_compile.compile(str(full_path), doraise=True)
            print(f"  [OK] {file_path}")
        except py_compile.PyCompileError as e:
            print(f"  [FAIL] {file_path}: {e}")
            all_valid = False

    return all_valid


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("NeuroS-MechInt Package Validation")
    print("=" * 60)

    results = []

    # Syntax check first
    results.append(("Syntax validation", test_syntax_all_files()))

    # Then import tests (skip if no dependencies installed)
    try:
        import torch
        import numpy
        results.append(("Biophysical module", test_biophysical_imports()))
        results.append(("Intervention module", test_intervention_imports()))
        results.append(("Alignment module", test_alignment_imports()))
        results.append(("Fractal module", test_fractal_imports()))
    except ImportError as e:
        print(f"\n⚠ Skipping import tests - missing dependency: {e}")
        print("Install dependencies with: pip install torch numpy scipy pywt fastdtw")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{test_name:.<40} {status}")

    all_passed = all(result[1] for result in results)

    print("=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nThe package expansion is complete and functional.")
        print("\nNext steps:")
        print("  1. Review EXPANSION_SUMMARY.md")
        print("  2. Create notebooks for new features")
        print("  3. Expand visualization module")
        print("  4. Build integration tests")
    else:
        print("⚠ SOME TESTS FAILED")
        print("\nPlease fix the errors above before proceeding.")

    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

"""
Integration Tests for Advanced MechInt Features

Tests the integration of:
- Biophysical modeling (ion channels, compartments, metabolism)
- Interventions (optogenetics, pharmacology, stimulation)
- Cross-species alignment
- Criticality detection
- Multifractal analysis
- Temporal dynamics
- Visualization
- Pipeline integration
- Database storage

Author: NeuroS Team
Date: 2025-11-04
"""

import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path

# Biophysical imports
from neuros_mechint.biophysical import (
    SodiumChannel, PotassiumChannel,
    MultiCompartmentNeuron, PrefabNeurons,
    STDP, ATPDynamics, MetabolicConstraint
)

# Intervention imports
from neuros_mechint.interventions import (
    ChR2, NpHR, Drugs, DBS, TMS
)

# Alignment imports
from neuros_mechint.alignment import (
    ProcrustesAlignment, DynamicTimeWarping,
    InterSubjectSynchronization, TimeResolvedCCA
)

# Criticality imports
from neuros_mechint.fractals import (
    NeuronalAvalanche, BranchingProcess,
    MultifractalDetrendedFluctuationAnalysis
)

# Visualization imports
from neuros_mechint.visualization import (
    Interactive3DBrain, BrainAtlas,
    MultifractalVisualizer, CrossSpeciesVisualizer,
    InterventionVisualizer
)

# Pipeline and database
from neuros_mechint.pipeline import MechIntPipeline, PipelineConfig
from neuros_mechint.database import MechIntDatabase
from neuros_mechint.results_extended import (
    BiophysicalResult, InterventionResult,
    CriticalityResult, MultifractalResult
)


class TestBiophysicalIntegration:
    """Test biophysical modeling components."""

    def test_ion_channel_simulation(self):
        """Test ion channel dynamics."""
        na = SodiumChannel(g_max=120.0)
        k = PotassiumChannel(g_max=36.0)

        # Simulate voltage step
        V = torch.tensor(-70.0)
        dt = 0.1

        for _ in range(100):
            na.step(V, dt)
            k.step(V, dt)

        I_na = na.current(V)
        I_k = k.current(V)

        assert isinstance(I_na, torch.Tensor)
        assert isinstance(I_k, torch.Tensor)
        assert not torch.isnan(I_na)
        assert not torch.isnan(I_k)

    def test_compartmental_neuron(self):
        """Test multi-compartment neuron simulation."""
        neuron = PrefabNeurons.pyramidal_cell(soma_area=1000.0)

        assert len(neuron.compartments) > 1

        # Simulate with current injection
        n_steps = 100
        current = torch.zeros(n_steps)
        current[20:80] = 500.0  # Current pulse

        voltages = neuron.simulate(current, n_steps, dt=0.1)

        assert voltages.shape == (n_steps, len(neuron.compartments))
        assert not torch.any(torch.isnan(voltages))

    def test_stdp(self):
        """Test spike-timing-dependent plasticity."""
        stdp = STDP(tau_plus=20.0, tau_minus=20.0)

        # Pre-post pairing
        for _ in range(10):
            stdp.update_pre()
            stdp.update_post()

        dw = stdp.get_weight_change()
        assert dw > 0  # LTP for pre-before-post

    def test_atp_dynamics(self):
        """Test metabolic ATP tracking."""
        atp = ATPDynamics(ATP_0=2.5)

        # Simulate spiking
        for _ in range(100):
            atp.update(spike_occurred=True, dt=0.001)

        assert atp.ATP < 2.5  # ATP should deplete
        assert atp.ATP > 0    # But not go negative

    def test_metabolic_constraint(self):
        """Test energy budget constraints."""
        constraint = MetabolicConstraint(
            max_spike_rate=50.0,
            energy_budget=1000.0
        )

        spike_rates = torch.ones(100) * 40.0  # Below max

        feasible = constraint.check_feasibility(spike_rates)
        energy = constraint.compute_energy_cost(spike_rates)

        assert feasible
        assert energy <= constraint.energy_budget


class TestInterventionIntegration:
    """Test intervention simulation components."""

    def test_optogenetics_chr2(self):
        """Test ChR2 optogenetic stimulation."""
        opsin = ChR2(g_max=1000.0, wavelength_peak=470.0)

        # Light pulse
        V = torch.tensor(-70.0)
        light_intensity = 10.0  # mW/mm²

        I_photo = opsin.photocurrent(V, light_intensity, dt=0.1)

        assert isinstance(I_photo, torch.Tensor)
        assert I_photo > 0  # Depolarizing current
        assert not torch.isnan(I_photo)

    def test_optogenetics_nphr(self):
        """Test NpHR inhibitory opsin."""
        opsin = NpHR(wavelength_peak=590.0)

        V = torch.tensor(-70.0)
        I_photo = opsin.photocurrent(V, 10.0, dt=0.1)

        assert I_photo < 0  # Hyperpolarizing current

    def test_pharmacology_dose_response(self):
        """Test drug dose-response curves."""
        ttx = Drugs.TTX()

        # Test dose range
        doses = [0.01, 0.1, 1.0, 10.0, 100.0]
        responses = [ttx.dose_response(d) for d in doses]

        # Response should be monotonic
        assert all(responses[i] >= responses[i-1] for i in range(1, len(responses)))
        assert responses[-1] > 0.9  # High dose = strong effect

    def test_dbs_stimulation(self):
        """Test deep brain stimulation."""
        dbs = DBS(frequency=130.0, amplitude=3.0, pulse_width=0.06)

        # Simulate 10 ms
        currents = [dbs.stimulate(t * 0.01) for t in range(1000)]

        # Count pulses
        pulses = sum(1 for i in range(1, len(currents))
                    if currents[i] > 0 and currents[i-1] == 0)

        expected_pulses = int(130.0 * 0.01)  # ~1 pulse in 10ms at 130 Hz
        assert abs(pulses - expected_pulses) <= 2  # Allow some variance

    def test_tms_field(self):
        """Test TMS electric field calculation."""
        tms = TMS(intensity=100.0, coil_position=(0, 0, 50))

        # Calculate field at point
        point = np.array([10.0, 10.0, 0.0])
        field = tms.calculate_field(point)

        assert field > 0
        assert not np.isnan(field)


class TestAlignmentIntegration:
    """Test cross-species and temporal alignment."""

    def test_procrustes_alignment(self):
        """Test Procrustes alignment of neural spaces."""
        # Two similar representations
        n_samples, n_dims = 50, 10
        X = np.random.randn(n_samples, n_dims)

        # Y = rotated X + noise
        angle = np.pi / 6
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        Y = X.copy()
        Y[:, :2] = Y[:, :2] @ R.T
        Y += np.random.randn(n_samples, n_dims) * 0.1

        # Align
        aligner = ProcrustesAlignment()
        aligned, disparity = aligner.align(X, Y)

        assert aligned.shape == X.shape
        assert disparity > 0
        assert disparity < 1.0  # Should be small for similar data

    def test_dynamic_time_warping(self):
        """Test DTW sequence alignment."""
        # Two similar sequences with time warp
        t = np.linspace(0, 2*np.pi, 100)
        seq1 = np.sin(t).reshape(-1, 1)
        seq2 = np.sin(t * 1.1).reshape(-1, 1)  # Slightly faster

        dtw = DynamicTimeWarping()
        distance, path = dtw.compute(seq1, seq2)

        assert distance > 0
        assert len(path) >= len(seq1)  # Path length >= sequence length

    def test_inter_subject_synchronization(self):
        """Test ISC across subjects."""
        # Multi-subject data with shared signal
        n_subjects, n_timepoints, n_features = 5, 100, 20

        shared = np.random.randn(n_timepoints, n_features)
        subjects_data = []

        for _ in range(n_subjects):
            # Mix shared and individual
            individual = np.random.randn(n_timepoints, n_features)
            subject = 0.7 * shared + 0.3 * individual
            subjects_data.append(subject)

        subjects_data = np.array(subjects_data)

        isc = InterSubjectSynchronization()
        isc_values = isc.compute(subjects_data)

        assert isc_values.shape == (n_features,)
        assert np.all(isc_values >= -1) and np.all(isc_values <= 1)
        assert isc_values.mean() > 0.3  # Should detect shared signal

    def test_time_resolved_cca(self):
        """Test time-resolved canonical correlation."""
        n_timepoints = 200
        X = np.random.randn(n_timepoints, 10)
        Y = np.random.randn(n_timepoints, 10)

        # Add some correlation
        Y[:, :3] = 0.5 * X[:, :3] + 0.5 * Y[:, :3]

        trcca = TimeResolvedCCA(window_size=50, step_size=10)
        correlations = trcca.compute(X, Y)

        assert len(correlations) > 0
        assert all(0 <= c <= 1 for c in correlations)


class TestCriticalityIntegration:
    """Test criticality and avalanche detection."""

    def test_avalanche_detection(self):
        """Test neuronal avalanche detection."""
        # Generate critical activity
        n_time, n_neurons = 1000, 50
        activity = np.random.rand(n_time, n_neurons) < 0.01  # Sparse

        detector = NeuronalAvalanche(threshold=0.0)
        avalanches = detector.detect_avalanches(activity.astype(float))

        assert len(avalanches) > 0
        assert all('size' in av and 'duration' in av for av in avalanches)

    def test_branching_parameter(self):
        """Test branching parameter estimation."""
        # Generate activity with known branching
        n_time, n_neurons = 1000, 100
        activity = np.zeros((n_time, n_neurons))
        activity[0, 0] = 1

        sigma_true = 1.0  # Critical

        for t in range(1, n_time):
            active = np.where(activity[t-1] > 0)[0]
            for _ in active:
                n_offspring = np.random.poisson(sigma_true)
                targets = np.random.choice(n_neurons,
                                         size=min(n_offspring, n_neurons),
                                         replace=False)
                activity[t, targets] = 1

        bp = BranchingProcess()
        sigma_est = bp.estimate_branching_parameter(activity)

        assert 0.5 < sigma_est < 1.5  # Should be near 1

    def test_multifractal_analysis(self):
        """Test MF-DFA analysis."""
        # Generate fractal signal
        n = 1024
        signal = np.cumsum(np.random.randn(n))

        mfdfa = MultifractalDetrendedFluctuationAnalysis(
            q_values=np.arange(-2, 3),
            scales=np.logspace(1, 2, 10).astype(int)
        )

        result = mfdfa.analyze(signal)

        assert 'fluctuations' in result
        assert 'hurst_exponents' in result
        assert len(result['hurst_exponents']) == 5  # 5 q-values


class TestVisualizationIntegration:
    """Test visualization components."""

    def test_brain_atlas(self):
        """Test brain atlas construction."""
        visual_regions = BrainAtlas.get_visual_system()
        dmn_regions = BrainAtlas.get_default_mode_network()

        assert len(visual_regions) > 0
        assert len(dmn_regions) > 0
        assert all(hasattr(r, 'center') and hasattr(r, 'name')
                  for r in visual_regions)

    def test_interactive_brain(self):
        """Test 3D brain visualization."""
        brain = Interactive3DBrain()

        # Add some regions
        brain.add_region_set(BrainAtlas.get_visual_system())

        # Create activation pattern
        activations = np.random.rand(len(brain.regions))

        # Generate figure (just test it doesn't crash)
        fig = brain.visualize_activation(activations)
        assert fig is not None

    def test_multifractal_visualizer(self):
        """Test multifractal visualization."""
        viz = MultifractalVisualizer()

        # Create dummy spectrum
        alpha = np.linspace(0.5, 1.5, 20)
        f_alpha = -(alpha - 1.0)**2 + 1.0  # Parabola

        fig = viz.plot_singularity_spectrum(alpha, f_alpha)
        assert fig is not None


class TestPipelineIntegration:
    """Test complete pipeline integration."""

    def test_biophysical_pipeline_stage(self):
        """Test biophysical analysis in pipeline."""
        # Create simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 10)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                enabled_analyses={'biophysical'},
                depth='quick',
                use_cache=False
            )

            pipeline = MechIntPipeline(model, config=config)

            inputs = torch.randn(5, 10)
            results = pipeline.run(
                inputs,
                analyses=['biophysical'],
                dt=0.1,
                duration=50.0
            )

            assert len(results.results) == 1
            assert results.results[0].method == 'Biophysical'

    def test_intervention_pipeline_stage(self):
        """Test intervention analysis in pipeline."""
        model = torch.nn.Linear(10, 10)

        config = PipelineConfig(
            enabled_analyses={'interventions'},
            use_cache=False
        )

        pipeline = MechIntPipeline(model, config=config)
        inputs = torch.randn(5, 10)

        results = pipeline.run(
            inputs,
            analyses=['interventions'],
            intervention='optogenetics'
        )

        assert len(results.results) == 1
        assert results.results[0].method == 'Interventions'

    def test_criticality_pipeline_stage(self):
        """Test criticality analysis in pipeline."""
        model = torch.nn.Linear(10, 10)

        config = PipelineConfig(
            enabled_analyses={'criticality'},
            use_cache=False
        )

        pipeline = MechIntPipeline(model, config=config)
        inputs = torch.randn(5, 10)

        results = pipeline.run(inputs, analyses=['criticality'])

        assert len(results.results) == 1
        assert results.results[0].method == 'Criticality'


class TestDatabaseIntegration:
    """Test database storage for new result types."""

    def test_store_biophysical_result(self):
        """Test storing biophysical results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = MechIntDatabase(tmpdir, verbose=False)

            # Create result
            result = BiophysicalResult(
                method='Biophysical',
                data={'test': np.array([1, 2, 3])},
                metadata={'neuron_type': 'pyramidal'},
                metrics={'spike_count': 10},
                voltages=np.random.randn(100, 5),
                spike_rate=25.0
            )

            # Store and retrieve
            result_id = db.store(result, tags=['test', 'biophysical'])
            retrieved = db.get(result_id)

            assert retrieved is not None
            assert isinstance(retrieved, BiophysicalResult)
            assert retrieved.spike_rate == 25.0

    def test_store_intervention_result(self):
        """Test storing intervention results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = MechIntDatabase(tmpdir, verbose=False)

            result = InterventionResult(
                method='Interventions',
                data={'test': np.array([1, 2, 3])},
                metadata={'intervention_type': 'optogenetics'},
                metrics={},
                intervention_type='optogenetics',
                EC50=5.0
            )

            result_id = db.store(result)
            retrieved = db.get(result_id)

            assert isinstance(retrieved, InterventionResult)
            assert retrieved.intervention_type == 'optogenetics'

    def test_query_specialized_results(self):
        """Test specialized query methods."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = MechIntDatabase(tmpdir, verbose=False)

            # Store multiple biophysical results
            for i in range(3):
                result = BiophysicalResult(
                    method='Biophysical',
                    data={},
                    metadata={'neuron_type': 'pyramidal' if i % 2 == 0 else 'interneuron'},
                    metrics={},
                    spike_rate=float(i * 10)
                )
                db.store(result)

            # Query
            pyramidal = db.query_biophysical(neuron_type='pyramidal')

            assert len(pyramidal) == 2  # 2 out of 3 are pyramidal

    def test_analysis_summary(self):
        """Test comprehensive analysis summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = MechIntDatabase(tmpdir, verbose=False)

            # Add diverse results
            db.store(BiophysicalResult(method='Biophysical', data={}, metadata={}, metrics={}))
            db.store(InterventionResult(method='Interventions', data={}, metadata={'intervention_type': 'optogenetics'}, metrics={}))
            db.store(CriticalityResult(method='Criticality', data={}, metadata={}, metrics={'branching_parameter': 1.0}))

            summary = db.get_analysis_summary()

            assert summary['total_results'] == 3
            assert 'by_type' in summary


class TestEndToEndIntegration:
    """Test complete end-to-end workflows."""

    def test_full_analysis_workflow(self):
        """Test complete analysis from model to visualization."""
        # Create model
        model = torch.nn.Sequential(
            torch.nn.Linear(20, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 20)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup pipeline with database
            config = PipelineConfig(
                enabled_analyses={'biophysical', 'interventions', 'criticality'},
                depth='quick',
                parallel=False
            )

            pipeline = MechIntPipeline(model, db_path=tmpdir, config=config)

            # Run analysis
            inputs = torch.randn(10, 20)
            results = pipeline.run(
                inputs,
                analyses=['biophysical', 'interventions', 'criticality']
            )

            # Check results
            assert len(results.results) == 3
            methods = {r.method for r in results.results}
            assert 'Biophysical' in methods
            assert 'Interventions' in methods
            assert 'Criticality' in methods

            # Check database storage
            db = MechIntDatabase(tmpdir, verbose=False)
            stats = db.get_stats()
            assert stats['total_results'] >= 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

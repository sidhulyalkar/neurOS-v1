"""
Comprehensive Integration Tests for Mechanistic Interpretability Suite
Tests end-to-end workflows across all mech-int modules
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def sample_model():
    """Create a simple model for testing"""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
            ])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    return SimpleModel()


@pytest.fixture
def sample_data():
    """Create sample neural data"""
    batch_size = 16
    time_steps = 100
    channels = 64

    # Simulate multi-modal data
    data = {
        'eeg': torch.randn(batch_size, time_steps, channels),
        'spikes': torch.randn(batch_size, time_steps, 100),
        'video': torch.randn(batch_size, time_steps, 512),
    }
    return data


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestSAEIntegration:
    """Test Sparse Autoencoder integration"""

    def test_hierarchical_sae_training(self, sample_model, sample_data):
        """Test hierarchical SAE training on model activations"""
        from neuros_neurofm.interpretability.concept_sae import HierarchicalSAE

        # Create hierarchical SAE
        hsae = HierarchicalSAE(
            layer_sizes=[64, 512, 2048],
            sparsity_coefficients=[0.01, 0.005, 0.001]
        )

        # Get activations
        activations = sample_data['eeg'].reshape(-1, 64)

        # Forward pass
        features_all = hsae(activations)

        # Check all levels present
        assert len(features_all) == 2
        assert features_all[0].shape == (activations.shape[0], 512)
        assert features_all[1].shape == (activations.shape[0], 2048)

        # Compute loss
        losses = hsae.compute_loss(activations, features_all)
        assert 'total' in losses
        assert losses['total'].item() > 0

    def test_concept_dictionary_building(self, sample_model):
        """Test building concept dictionary with labels"""
        from neuros_neurofm.interpretability.concept_sae import (
            HierarchicalSAE, ConceptDictionary
        )

        # Create SAE
        hsae = HierarchicalSAE(
            layer_sizes=[64, 256, 1024],
            sparsity_coefficients=[0.01, 0.005, 0.001]
        )

        # Sample data
        activations = torch.randn(100, 64)
        probe_labels = {
            'region': torch.randint(0, 5, (100,)),
            'behavior': torch.rand(100)
        }

        # Build dictionary
        dictionary = ConceptDictionary(hsae)
        dictionary.build_dictionary(activations, probe_labels)

        # Check labels created
        assert len(dictionary.labels) > 0

        # Check label structure
        for (level, feat_id), label in dictionary.labels.items():
            assert label.feature_id == feat_id
            assert label.confidence >= 0
            assert 'region' in label.evidence or 'behavior' in label.evidence


class TestAlignmentIntegration:
    """Test brain alignment integration"""

    def test_cca_alignment(self):
        """Test CCA alignment between representations"""
        from neuros_neurofm.interpretability.alignment import CCA

        # Create two representations
        X = torch.randn(100, 50)
        Y = torch.randn(100, 60)

        # Fit CCA
        cca = CCA(n_components=10)
        cca.fit(X, Y)

        # Transform
        X_c, Y_c = cca.transform(X, Y)

        # Check shapes
        assert X_c.shape == (100, 10)
        assert Y_c.shape == (100, 10)

        # Check correlations
        correlations = cca.canonical_correlations_
        assert len(correlations) == 10
        assert all(0 <= c <= 1 for c in correlations)

    def test_rsa_comparison(self):
        """Test RSA comparison between layers"""
        from neuros_neurofm.interpretability.alignment import RSA

        # Create representations
        X1 = torch.randn(50, 100)
        X2 = torch.randn(50, 100)

        # Compute RSA
        rsa = RSA(metric='correlation')
        similarity = rsa.compare(X1, X2)

        # Check similarity in valid range
        assert -1 <= similarity <= 1


class TestDynamicsIntegration:
    """Test dynamical systems analysis integration"""

    def test_koopman_analysis(self):
        """Test Koopman operator estimation"""
        from neuros_neurofm.interpretability.dynamics import DynamicsAnalyzer

        # Create trajectory (Lorenz system)
        T = 500
        D = 3
        n_trials = 5

        # Simple oscillatory dynamics
        t = np.linspace(0, 10, T)
        trajectories = np.zeros((n_trials, T, D))
        for trial in range(n_trials):
            trajectories[trial, :, 0] = np.sin(t + trial * 0.1)
            trajectories[trial, :, 1] = np.cos(t + trial * 0.1)
            trajectories[trial, :, 2] = np.sin(2*t + trial * 0.1)

        trajectories = torch.from_numpy(trajectories).float()

        # Analyze
        analyzer = DynamicsAnalyzer()
        K, eigenvalues, eigenvectors = analyzer.estimate_koopman_operator(trajectories)

        # Check outputs
        assert K.shape == (D, D)
        assert len(eigenvalues) == D

    def test_lyapunov_exponents(self):
        """Test Lyapunov exponent estimation"""
        from neuros_neurofm.interpretability.dynamics import DynamicsAnalyzer

        # Stable trajectory
        T = 300
        D = 4
        trajectories = torch.randn(3, T, D).cumsum(dim=1) * 0.1

        # Analyze
        analyzer = DynamicsAnalyzer()
        lyapunov = analyzer.compute_lyapunov_exponents(trajectories)

        # Check output
        assert isinstance(lyapunov, (float, torch.Tensor))


class TestCounterfactualIntegration:
    """Test counterfactual analysis integration"""

    def test_latent_surgery(self, sample_model, sample_data):
        """Test latent surgery and interventions"""
        from neuros_neurofm.interpretability.counterfactuals import LatentSurgery

        # Create surgery tool
        surgery = LatentSurgery(sample_model)

        # Test input
        x = sample_data['eeg'][:4, 0, :]  # (4, 64)

        # Define edit function
        def zero_first_dim(latent):
            edited = latent.clone()
            edited[:, 0] = 0
            return edited

        # Apply edit at layer 0
        output_edited = surgery.edit_latent(x, "layers.0", zero_first_dim)

        # Check output shape
        assert output_edited.shape == (4, 64)

        # Output should be different from original
        with torch.no_grad():
            output_original = sample_model(x)

        assert not torch.allclose(output_edited, output_original)

    def test_do_calculus_intervention(self, sample_model, sample_data):
        """Test do-calculus interventions"""
        from neuros_neurofm.interpretability.counterfactuals import DoCalculusInterventions

        # Create intervention tool
        do_calc = DoCalculusInterventions(sample_model)

        # Test input
        x = sample_data['eeg'][:4, 0, :]

        # Define outcome function
        def outcome_fn(output):
            return output.mean().item()

        # Estimate ATE
        ate = do_calc.estimate_ate(
            x,
            layer="layers.0",
            dim=0,
            treatment_value=1.0,
            control_value=0.0,
            outcome_fn=outcome_fn
        )

        # Check ATE is a number
        assert isinstance(ate, float)


class TestMetaDynamicsIntegration:
    """Test meta-dynamics and training trajectory analysis"""

    def test_training_phase_detection(self):
        """Test training phase detection"""
        from neuros_neurofm.interpretability.meta_dynamics import TrainingPhaseDetection

        # Simulate loss curve
        steps = np.arange(1000)
        loss_curve = 2.0 * np.exp(-steps / 100) + 0.1 + 0.01 * np.random.randn(1000)

        # Detect phases
        detector = TrainingPhaseDetection()
        phases = detector.detect_phases(loss_curve)

        # Should detect multiple phases
        assert len(phases) >= 2

        # Check phase names
        phase_names = {p.name for p in phases}
        assert 'warmup' in phase_names or 'fitting' in phase_names

    def test_representational_drift(self):
        """Test drift measurement between representations"""
        from neuros_neurofm.interpretability.meta_dynamics import RepresentationalTrajectory

        # Create trajectory (representations over time)
        trajectory = [
            torch.randn(100, 64) for _ in range(10)
        ]

        # Mock trajectory object
        class MockTrajectory:
            def measure_drift(self, traj, metric='cca'):
                if metric == 'cca':
                    return RepresentationalTrajectory._cca_similarity(self, traj[0], traj[1])
                return 0.5

        mock = MockTrajectory()

        # Test CCA similarity
        similarity = RepresentationalTrajectory._cca_similarity(
            mock, trajectory[0], trajectory[1]
        )

        assert -1 <= similarity <= 1


class TestReportingIntegration:
    """Test unified reporting integration"""

    def test_report_creation(self, temp_output_dir):
        """Test creating a complete mech-int report"""
        from neuros_neurofm.interpretability.reporting import MechIntReport
        import matplotlib.pyplot as plt
        import pandas as pd

        # Create report
        report = MechIntReport(temp_output_dir, title="Test Report")

        # Add content
        report.add_section("Introduction", "This is a test report.")

        # Add metric
        report.add_metric("Accuracy", 95.5, unit="%", description="Model accuracy")

        # Add figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        report.add_figure(fig, "Test Figure")
        plt.close(fig)

        # Add table
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        report.add_table(df, "Test Table")

        # Generate HTML
        html_path = report.generate_html("test_report.html")

        # Check file exists
        assert Path(html_path).exists()

        # Check content
        with open(html_path, 'r') as f:
            content = f.read()
            assert "Test Report" in content
            assert "95.5" in content
            assert "Introduction" in content


class TestHooksIntegration:
    """Test training hooks integration"""

    def test_activation_sampler(self, sample_model, sample_data, temp_output_dir):
        """Test activation sampling during training"""
        from neuros_neurofm.interpretability.hooks import ActivationSampler

        # Create sampler
        sampler = ActivationSampler(
            layers=['layers.0', 'layers.2'],
            save_dir=temp_output_dir
        )

        # Register hooks
        sampler.register_hooks(sample_model)

        # Forward pass
        x = sample_data['eeg'][:4, 0, :]
        with torch.no_grad():
            sample_model(x)

        # Check activations captured
        assert len(sampler.activations) == 2
        assert 'layers.0' in sampler.activations
        assert 'layers.2' in sampler.activations

        # Save activations
        sampler.save_activations(global_step=0)

        # Check files created
        saved_files = list(Path(temp_output_dir).glob("*.pt"))
        assert len(saved_files) > 0


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows"""

    def test_full_analysis_pipeline(self, sample_model, sample_data, temp_output_dir):
        """Test complete analysis pipeline from data to report"""

        # 1. Extract activations
        x = sample_data['eeg'][:8, 0, :]

        activations = []
        def hook_fn(module, input, output):
            activations.append(output.detach().cpu())

        handle = sample_model.layers[2].register_forward_hook(hook_fn)

        with torch.no_grad():
            sample_model(x)

        handle.remove()

        activation = activations[0]

        # 2. Train SAE on activations
        from neuros_neurofm.interpretability.concept_sae import HierarchicalSAE

        hsae = HierarchicalSAE(
            layer_sizes=[activation.shape[1], 512, 2048],
            sparsity_coefficients=[0.01, 0.005, 0.001]
        )

        features = hsae(activation)
        losses = hsae.compute_loss(activation, features)

        assert losses['total'].item() > 0

        # 3. Analyze dynamics
        from neuros_neurofm.interpretability.dynamics import DynamicsAnalyzer

        # Create trajectory
        trajectory = torch.randn(5, 100, activation.shape[1])
        analyzer = DynamicsAnalyzer()

        slow_manifold, info = analyzer.identify_slow_manifold(trajectory, n_components=3)
        assert slow_manifold.shape[2] == 3

        # 4. Generate report
        from neuros_neurofm.interpretability.reporting import MechIntReport

        report = MechIntReport(temp_output_dir, title="Full Analysis")
        report.add_section("SAE Analysis", f"Trained SAE with loss: {losses['total'].item():.4f}")
        report.add_section("Dynamics", f"Slow manifold: {info['variance_explained']:.2%} variance")
        report.add_metric("Total Loss", losses['total'].item())

        html_path = report.generate_html("full_analysis.html")
        assert Path(html_path).exists()

    def test_training_to_evaluation_workflow(self, sample_model, temp_output_dir):
        """Test workflow from training hooks to evaluation"""
        from neuros_neurofm.interpretability.hooks import MechIntConfig, ActivationSampler

        # 1. Setup config
        config = MechIntConfig(
            sample_layers=['layers.2'],
            save_hidden_every_n_steps=1,
            output_dir=temp_output_dir,
            storage_backend='local'
        )

        # 2. Create sampler
        sampler = ActivationSampler(
            layers=config.sample_layers,
            save_dir=temp_output_dir
        )

        # 3. Simulate training steps
        sampler.register_hooks(sample_model)

        for step in range(5):
            x = torch.randn(4, 64)
            with torch.no_grad():
                sample_model(x)

            if step % config.save_hidden_every_n_steps == 0:
                sampler.save_activations(global_step=step)

            sampler.clear_cache()

        # 4. Check saved files
        saved_files = list(Path(temp_output_dir).glob("*.pt"))
        assert len(saved_files) >= 5


class TestCrossModuleIntegration:
    """Test integration between multiple modules"""

    def test_sae_to_counterfactual(self, sample_model, sample_data):
        """Test using SAE features for counterfactual analysis"""
        from neuros_neurofm.interpretability.concept_sae import (
            HierarchicalSAE, CausalSAEProbe
        )

        # Train SAE
        x = sample_data['eeg'][:4, 0, :]

        activations = []
        def hook_fn(module, input, output):
            activations.append(output.detach().cpu())

        handle = sample_model.layers[2].register_forward_hook(hook_fn)
        with torch.no_grad():
            sample_model(x)
        handle.remove()

        activation = activations[0]

        hsae = HierarchicalSAE(
            layer_sizes=[activation.shape[1], 256],
            sparsity_coefficients=[0.01]
        )

        features = hsae(activation)

        # Use for causal intervention
        probe = CausalSAEProbe(hsae, sample_model)

        output_intervened = probe.reinsert_feature(
            x, 'layers.2', level=0, feature_id=0, magnitude=2.0
        )

        assert output_intervened.shape == (4, 64)

    def test_alignment_to_reporting(self, temp_output_dir):
        """Test alignment analysis to report generation"""
        from neuros_neurofm.interpretability.alignment import CCA
        from neuros_neurofm.interpretability.reporting import MechIntReport
        import matplotlib.pyplot as plt

        # Create representations
        X = torch.randn(100, 50)
        Y = torch.randn(100, 60)

        # Compute CCA
        cca = CCA(n_components=10)
        cca.fit(X, Y)

        # Create report
        report = MechIntReport(temp_output_dir, title="Alignment Analysis")

        # Add results
        report.add_section("CCA Results", "Canonical correlation analysis completed.")

        correlations = cca.canonical_correlations_
        for i, corr in enumerate(correlations):
            report.add_metric(f"CC{i+1}", corr, description=f"Canonical correlation {i+1}")

        # Add figure
        fig, ax = plt.subplots()
        ax.bar(range(len(correlations)), correlations)
        ax.set_xlabel("Component")
        ax.set_ylabel("Correlation")
        report.add_figure(fig, "Canonical Correlations")
        plt.close(fig)

        # Generate
        html_path = report.generate_html("alignment_report.html")
        assert Path(html_path).exists()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])

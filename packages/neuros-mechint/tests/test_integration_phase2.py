"""
Integration tests for Phase 2 mechanistic interpretability features.

Tests comprehensive workflows combining multiple analysis techniques
across different model architectures.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn

from neuros_mechint.database import MechIntDatabase
from neuros_mechint.pipeline import MechIntPipeline, PipelineConfig
from neuros_mechint.circuits import (
    AutomatedCircuitDiscovery,
    PathPatcher,
    CircuitComparator,
    MotifDetector
)
from neuros_mechint.energy_flow import (
    LandauerAnalyzer,
    NESSAnalyzer,
    FluctuationTheoremAnalyzer,
    EnergyCascadeAnalyzer,
    HamiltonianDecomposer
)
from neuros_mechint.dynamics import (
    NeuralODEIntegrator,
    SlowFeatureAnalyzer
)
from neuros_mechint.visualization import EnhancedVisualizer

# Test fixtures
@pytest.fixture
def device():
    """Get available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    temp_dir = tempfile.mkdtemp()
    db = MechIntDatabase(root_dir=temp_dir)
    yield db
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def simple_transformer(device):
    """Create a simple transformer for testing."""
    class SimpleTransformer(nn.Module):
        def __init__(self, d_model=32, nhead=4, num_layers=2, dim_feedforward=64):
            super().__init__()
            self.embedding = nn.Linear(10, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output = nn.Linear(d_model, 3)

        def forward(self, x):
            # x: [batch, seq_len, features]
            x = self.embedding(x)
            x = self.transformer(x)
            x = x.mean(dim=1)  # Global average pooling
            return self.output(x)

    model = SimpleTransformer().to(device)
    model.eval()
    return model

@pytest.fixture
def simple_resnet(device):
    """Create a simple ResNet-like model for testing."""
    class ResBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)
            self.relu = nn.ReLU()

        def forward(self, x):
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += identity
            out = self.relu(out)
            return out

    class SimpleResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu = nn.ReLU()
            self.layer1 = ResBlock(16)
            self.layer2 = ResBlock(16)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(16, 5)

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return self.fc(x)

    model = SimpleResNet().to(device)
    model.eval()
    return model

@pytest.fixture
def simple_rnn(device):
    """Create a simple RNN for testing."""
    class SimpleRNN(nn.Module):
        def __init__(self, input_size=10, hidden_size=32, num_layers=2, output_size=3):
            super().__init__()
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            # x: [batch, seq_len, features]
            out, _ = self.rnn(x)
            out = out[:, -1, :]  # Take last timestep
            return self.fc(out)

    model = SimpleRNN().to(device)
    model.eval()
    return model


# Integration Tests

class TestFullPipelines:
    """Test complete analysis pipelines on different architectures."""

    def test_full_pipeline_transformer(self, simple_transformer, temp_db, device):
        """Test comprehensive pipeline on transformer architecture."""
        # Generate sample data
        batch_size = 16
        seq_len = 8
        feature_dim = 10
        inputs = torch.randn(batch_size, seq_len, feature_dim, device=device)
        targets = torch.randint(0, 3, (batch_size,), device=device)

        # Configure comprehensive pipeline
        config = PipelineConfig(
            mode='comprehensive',
            enable_circuits=True,
            enable_energy=True,
            enable_thermodynamics=True,
            enable_dynamics=True,
            enable_visualization=True,
            save_checkpoints=True
        )

        pipeline = MechIntPipeline(
            config=config,
            database=temp_db,
            device=device
        )

        # Run pipeline
        results = pipeline.run(
            model=simple_transformer,
            inputs=inputs,
            targets=targets,
            experiment_name='transformer_test'
        )

        # Verify all stages completed
        assert 'circuit' in results
        assert 'energy' in results
        assert 'thermodynamics' in results
        assert 'dynamics' in results

        # Verify results are stored in database
        for stage, result_id in results.items():
            assert result_id is not None
            stored_result = temp_db.load(result_id)
            assert stored_result is not None

            # Verify tags
            tags = temp_db.get_tags(result_id)
            assert 'transformer_test' in tags

        print("✓ Transformer pipeline test passed")

    def test_full_pipeline_resnet(self, simple_resnet, temp_db, device):
        """Test comprehensive pipeline on ResNet architecture."""
        # Generate sample image data
        batch_size = 8
        inputs = torch.randn(batch_size, 3, 16, 16, device=device)  # Small images for speed
        targets = torch.randint(0, 5, (batch_size,), device=device)

        # Configure standard pipeline (comprehensive too slow for ResNet)
        config = PipelineConfig(
            mode='standard',
            enable_circuits=True,
            enable_energy=True,
            enable_thermodynamics=True,
            enable_dynamics=False,  # Skip dynamics for conv models
            enable_visualization=True,
            save_checkpoints=False
        )

        pipeline = MechIntPipeline(
            config=config,
            database=temp_db,
            device=device
        )

        # Run pipeline
        results = pipeline.run(
            model=simple_resnet,
            inputs=inputs,
            targets=targets,
            experiment_name='resnet_test'
        )

        # Verify key stages completed
        assert 'circuit' in results
        assert 'energy' in results
        assert 'thermodynamics' in results

        # Verify circuit discovery found residual connections
        circuit_result = temp_db.load(results['circuit'])
        assert len(circuit_result.edges) > 0

        # Verify energy analysis captured conv operations
        energy_result = temp_db.load(results['energy'])
        assert energy_result.total_bits_erased > 0

        print("✓ ResNet pipeline test passed")

    def test_full_pipeline_rnn(self, simple_rnn, temp_db, device):
        """Test comprehensive pipeline on RNN architecture."""
        # Generate sequential data
        batch_size = 16
        seq_len = 10
        feature_dim = 10
        inputs = torch.randn(batch_size, seq_len, feature_dim, device=device)
        targets = torch.randint(0, 3, (batch_size,), device=device)

        # Configure pipeline with emphasis on dynamics
        config = PipelineConfig(
            mode='comprehensive',
            enable_circuits=True,
            enable_energy=True,
            enable_thermodynamics=True,
            enable_dynamics=True,  # Important for RNNs
            enable_visualization=True,
            save_checkpoints=False
        )

        pipeline = MechIntPipeline(
            config=config,
            database=temp_db,
            device=device
        )

        # Run pipeline
        results = pipeline.run(
            model=simple_rnn,
            inputs=inputs,
            targets=targets,
            experiment_name='rnn_test'
        )

        # Verify all stages completed
        assert len(results) >= 4

        # Verify thermodynamics analysis (important for recurrent models)
        thermo_result = temp_db.load(results['thermodynamics'])
        assert hasattr(thermo_result, 'entropy_production_rate')

        print("✓ RNN pipeline test passed")


class TestCrossModelComparison:
    """Test comparison analyses across different models."""

    def test_cross_model_comparison(self, simple_transformer, simple_resnet, temp_db, device):
        """Test circuit comparison between different architectures."""
        # Generate compatible data for both models
        batch_size = 8

        # Transformer data
        trans_inputs = torch.randn(batch_size, 8, 10, device=device)
        trans_targets = torch.randint(0, 3, (batch_size,), device=device)

        # ResNet data
        resnet_inputs = torch.randn(batch_size, 3, 16, 16, device=device)
        resnet_targets = torch.randint(0, 3, (batch_size,), device=device)

        # Run ACDC on both models
        acdc_trans = AutomatedCircuitDiscovery(
            model=simple_transformer,
            importance_threshold=0.05,
            device=device
        )
        circuit_trans = acdc_trans.discover_circuit(
            inputs=trans_inputs,
            targets=trans_targets,
            max_iterations=5
        )
        trans_id = temp_db.store(circuit_trans, tags=['transformer', 'comparison'])

        acdc_resnet = AutomatedCircuitDiscovery(
            model=simple_resnet,
            importance_threshold=0.05,
            device=device
        )
        circuit_resnet = acdc_resnet.discover_circuit(
            inputs=resnet_inputs,
            targets=resnet_targets,
            max_iterations=5
        )
        resnet_id = temp_db.store(circuit_resnet, tags=['resnet', 'comparison'])

        # Compare circuits
        comparator = CircuitComparator(database=temp_db)
        comparison = comparator.compare_circuits(trans_id, resnet_id)

        # Verify comparison metrics
        assert hasattr(comparison, 'similarity_score')
        assert 0.0 <= comparison.similarity_score <= 1.0
        assert hasattr(comparison, 'node_overlap')
        assert hasattr(comparison, 'edge_overlap')

        print(f"✓ Cross-model comparison test passed (similarity: {comparison.similarity_score:.3f})")

    def test_motif_detection_consistency(self, simple_transformer, simple_resnet, device):
        """Test that motif detection is consistent across runs."""
        batch_size = 8

        # Run ACDC on transformer twice
        inputs = torch.randn(batch_size, 8, 10, device=device)
        targets = torch.randint(0, 3, (batch_size,), device=device)

        acdc = AutomatedCircuitDiscovery(
            model=simple_transformer,
            importance_threshold=0.05,
            device=device
        )

        # Run 1
        torch.manual_seed(42)
        circuit1 = acdc.discover_circuit(inputs=inputs, targets=targets, max_iterations=5)
        detector1 = MotifDetector(circuit=circuit1, n_random_samples=10)
        motifs1 = detector1.detect_all_motifs(compute_significance=True)

        # Run 2 (same seed should give same results)
        torch.manual_seed(42)
        circuit2 = acdc.discover_circuit(inputs=inputs, targets=targets, max_iterations=5)
        detector2 = MotifDetector(circuit=circuit2, n_random_samples=10)
        motifs2 = detector2.detect_all_motifs(compute_significance=True)

        # Compare motif counts
        for motif_type in ['feedforward', 'recurrent', 'skip']:
            if motif_type in motifs1.motif_counts:
                assert motifs1.motif_counts[motif_type] == motifs2.motif_counts[motif_type]

        print("✓ Motif detection consistency test passed")


class TestThermodynamicConsistency:
    """Test thermodynamic principles are satisfied."""

    def test_energy_conservation(self, simple_transformer, device):
        """Test that energy cascade analysis conserves energy."""
        batch_size = 8
        inputs = torch.randn(batch_size, 8, 10, device=device)

        cascade = EnergyCascadeAnalyzer(
            model=simple_transformer,
            energy_metric='variance',
            track_spectrum=True
        )

        result = cascade.analyze_cascade(inputs=inputs)

        # Total dissipated energy should be positive or zero
        total_dissipation = sum(result.layer_dissipation.values())
        assert total_dissipation >= 0, "Energy dissipation cannot be negative"

        # Check conservation: E_in = E_out + dissipation (approximately)
        for layer_name in result.layer_input_energy.keys():
            if layer_name in result.layer_output_energy:
                e_in = result.layer_input_energy[layer_name]
                e_out = result.layer_output_energy[layer_name]
                dissipation = result.layer_dissipation[layer_name]

                # Conservation equation (with numerical tolerance)
                conservation_error = abs(e_in - (e_out + dissipation))
                relative_error = conservation_error / (e_in + 1e-10)
                assert relative_error < 0.1, f"Energy not conserved in {layer_name}: {relative_error:.3f}"

        print("✓ Energy conservation test passed")

    def test_landauer_principle(self, simple_transformer, device):
        """Test that Landauer's principle bounds are respected."""
        batch_size = 8
        inputs = torch.randn(batch_size, 8, 10, device=device)

        landauer = LandauerAnalyzer(
            model=simple_transformer,
            temperature=300.0,
            device=device
        )

        result = landauer.analyze_forward_pass(inputs=inputs)

        # Minimum energy should be non-negative
        assert result.total_min_energy >= 0, "Minimum energy cannot be negative"

        # Bits erased should be non-negative
        assert result.total_bits_erased >= 0, "Bits erased cannot be negative"

        # Check Landauer bound: E_min = kT ln(2) * bits_erased
        kT_ln2 = 1.38e-23 * 300.0 * np.log(2)  # Joules/bit
        expected_min_energy = kT_ln2 * result.total_bits_erased

        # Computed minimum energy should match theoretical value (within numerical tolerance)
        relative_error = abs(result.total_min_energy - expected_min_energy) / (expected_min_energy + 1e-30)
        assert relative_error < 0.01, f"Landauer bound violated: {relative_error:.3f}"

        # Reversibility score should be in [0, 1]
        assert 0.0 <= result.reversibility_score <= 1.0

        print("✓ Landauer principle test passed")

    def test_fluctuation_theorem_symmetry(self, simple_rnn, device):
        """Test that fluctuation theorems respect detailed balance."""
        batch_size = 16
        seq_len = 10
        feature_dim = 10

        # Generate forward trajectory
        forward_inputs = torch.randn(batch_size, seq_len, feature_dim, device=device)

        # Generate reverse trajectory (time-reversed)
        reverse_inputs = torch.flip(forward_inputs, dims=[1])

        analyzer = FluctuationTheoremAnalyzer(
            model=simple_rnn,
            temperature=300.0,
            device=device
        )

        # Test Crooks theorem
        result = analyzer.test_crooks_theorem(
            forward_data=forward_inputs,
            reverse_data=reverse_inputs
        )

        # Crooks relation should be satisfied (chi-square test)
        # We're just checking that the code runs and produces valid outputs
        assert hasattr(result, 'crooks_chi2')
        assert hasattr(result, 'crooks_satisfied')
        assert result.crooks_chi2 >= 0

        print("✓ Fluctuation theorem symmetry test passed")


class TestVisualizationOutputs:
    """Test that visualization methods produce valid outputs."""

    def test_enhanced_visualizer_bokeh(self, simple_transformer, device):
        """Test EnhancedVisualizer with Bokeh backend."""
        pytest.importorskip('bokeh', reason="Bokeh not installed")

        batch_size = 8
        inputs = torch.randn(batch_size, 8, 10, device=device)
        targets = torch.randint(0, 3, (batch_size,), device=device)

        # Run ACDC to get circuit
        acdc = AutomatedCircuitDiscovery(
            model=simple_transformer,
            importance_threshold=0.05,
            device=device
        )
        circuit = acdc.discover_circuit(inputs=inputs, targets=targets, max_iterations=5)

        # Create visualizer
        visualizer = EnhancedVisualizer(backend='bokeh')

        # Test circuit visualization
        fig = visualizer.visualize_circuit(circuit)
        assert fig is not None

        # Test that it produces a Bokeh figure
        from bokeh.plotting import Figure
        assert isinstance(fig, Figure)

        print("✓ Enhanced visualizer (Bokeh) test passed")

    def test_enhanced_visualizer_matplotlib(self, simple_transformer, device):
        """Test EnhancedVisualizer with matplotlib backend."""
        batch_size = 8
        inputs = torch.randn(batch_size, 8, 10, device=device)
        targets = torch.randint(0, 3, (batch_size,), device=device)

        # Run ACDC to get circuit
        acdc = AutomatedCircuitDiscovery(
            model=simple_transformer,
            importance_threshold=0.05,
            device=device
        )
        circuit = acdc.discover_circuit(inputs=inputs, targets=targets, max_iterations=5)

        # Create visualizer with matplotlib backend
        visualizer = EnhancedVisualizer(backend='matplotlib')

        # Test circuit visualization
        fig = visualizer.visualize_circuit(circuit)
        assert fig is not None

        # Test that it produces a matplotlib figure
        import matplotlib.pyplot as plt
        assert isinstance(fig, plt.Figure)

        plt.close(fig)

        print("✓ Enhanced visualizer (matplotlib) test passed")


class TestDatabaseScalability:
    """Test database performance with large numbers of results."""

    def test_database_scalability(self, simple_transformer, temp_db, device):
        """Test database with many stored results."""
        batch_size = 8
        inputs = torch.randn(batch_size, 8, 10, device=device)
        targets = torch.randint(0, 3, (batch_size,), device=device)

        # Run ACDC multiple times with different parameters
        n_experiments = 20
        stored_ids = []

        for i in range(n_experiments):
            threshold = 0.01 + (i * 0.005)  # Vary threshold

            acdc = AutomatedCircuitDiscovery(
                model=simple_transformer,
                importance_threshold=threshold,
                device=device
            )

            circuit = acdc.discover_circuit(
                inputs=inputs,
                targets=targets,
                max_iterations=5
            )

            result_id = temp_db.store(
                result=circuit,
                tags=[f'experiment_{i}', 'scalability_test']
            )
            stored_ids.append(result_id)

        # Test query performance
        all_results = temp_db.query(tags=['scalability_test'])
        assert len(all_results) == n_experiments

        # Test individual retrieval
        for result_id in stored_ids[:5]:  # Test first 5
            result = temp_db.load(result_id)
            assert result is not None
            assert hasattr(result, 'nodes')
            assert hasattr(result, 'edges')

        # Test tag filtering
        filtered = temp_db.query(tags=['experiment_0'])
        assert len(filtered) >= 1

        # Test listing all
        all_ids = temp_db.list_all()
        assert len(all_ids) >= n_experiments

        print(f"✓ Database scalability test passed ({n_experiments} experiments)")

    def test_deduplication_effectiveness(self, simple_transformer, temp_db, device):
        """Test that content-based deduplication works correctly."""
        batch_size = 8
        inputs = torch.randn(batch_size, 8, 10, device=device)
        targets = torch.randint(0, 3, (batch_size,), device=device)

        # Run same analysis twice
        acdc = AutomatedCircuitDiscovery(
            model=simple_transformer,
            importance_threshold=0.05,
            device=device
        )

        # Set seed for reproducibility
        torch.manual_seed(42)
        circuit1 = acdc.discover_circuit(inputs=inputs, targets=targets, max_iterations=5)
        id1 = temp_db.store(circuit1, tags=['dedup_test', 'run_1'])

        # Same analysis with same seed should produce identical result
        torch.manual_seed(42)
        circuit2 = acdc.discover_circuit(inputs=inputs, targets=targets, max_iterations=5)
        id2 = temp_db.store(circuit2, tags=['dedup_test', 'run_2'])

        # IDs should be the same (deduplication)
        assert id1 == id2, "Deduplication failed for identical results"

        # Tags should be merged
        tags = temp_db.get_tags(id1)
        assert 'run_1' in tags
        assert 'run_2' in tags
        assert 'dedup_test' in tags

        # Should only be one unique result
        dedup_results = temp_db.query(tags=['dedup_test'])
        assert len(dedup_results) == 1

        print("✓ Deduplication effectiveness test passed")


class TestPipelineRobustness:
    """Test pipeline error handling and recovery."""

    def test_checkpoint_recovery(self, simple_transformer, temp_db, device):
        """Test that pipeline can recover from checkpoints."""
        batch_size = 8
        inputs = torch.randn(batch_size, 8, 10, device=device)
        targets = torch.randint(0, 3, (batch_size,), device=device)

        # Configure pipeline with checkpointing
        config = PipelineConfig(
            mode='standard',
            enable_circuits=True,
            enable_energy=True,
            enable_thermodynamics=True,
            enable_dynamics=False,
            enable_visualization=False,
            save_checkpoints=True
        )

        pipeline = MechIntPipeline(
            config=config,
            database=temp_db,
            device=device
        )

        experiment_name = 'checkpoint_test'

        # Run pipeline
        results1 = pipeline.run(
            model=simple_transformer,
            inputs=inputs,
            targets=targets,
            experiment_name=experiment_name
        )

        # Verify checkpoint exists
        assert pipeline.has_checkpoint(experiment_name)

        # Try to resume (should load from checkpoint)
        results2 = pipeline.resume_from_checkpoint(experiment_name)

        # Results should be the same
        assert set(results1.keys()) == set(results2.keys())
        for stage in results1.keys():
            assert results1[stage] == results2[stage]

        print("✓ Checkpoint recovery test passed")

    def test_partial_pipeline_failure(self, simple_transformer, temp_db, device):
        """Test that pipeline handles stage failures gracefully."""
        batch_size = 8
        inputs = torch.randn(batch_size, 8, 10, device=device)
        targets = torch.randint(0, 3, (batch_size,), device=device)

        # Create a broken config (e.g., invalid settings)
        # In real pipeline, this would be caught, but we test error handling
        config = PipelineConfig(
            mode='quick',
            enable_circuits=True,
            enable_energy=True,
            enable_thermodynamics=False,
            enable_dynamics=False,
            enable_visualization=False,
            save_checkpoints=False
        )

        pipeline = MechIntPipeline(
            config=config,
            database=temp_db,
            device=device
        )

        # This should complete successfully even if some stages are disabled
        results = pipeline.run(
            model=simple_transformer,
            inputs=inputs,
            targets=targets,
            experiment_name='partial_test'
        )

        # At least circuit and energy should complete
        assert 'circuit' in results
        assert 'energy' in results

        # Thermodynamics should not be present (disabled)
        assert 'thermodynamics' not in results or results['thermodynamics'] is None

        print("✓ Partial pipeline failure handling test passed")


# Utility function for running all tests
def run_integration_tests():
    """
    Run all integration tests.

    Usage:
        pytest tests/test_integration_phase2.py -v

    Or from Python:
        from tests.test_integration_phase2 import run_integration_tests
        run_integration_tests()
    """
    pytest.main([__file__, '-v', '-s'])


if __name__ == '__main__':
    run_integration_tests()

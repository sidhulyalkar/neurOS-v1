"""
Comprehensive tests for energy_flow module.

Tests all components of the Information Flow and Energy Landscape analysis:
- InformationFlowAnalyzer (MINE, k-NN, histogram MI estimation)
- EnergyLandscape (density, score-based, quadratic methods)
- EntropyProduction (dissipation rate, nonequilibrium detection)
- MINENetwork (neural network for MI estimation)
"""

import pytest
import torch
import numpy as np
from typing import Dict, List

from neuros_neurofm.interpretability.energy_flow import (
    InformationFlowAnalyzer,
    EnergyLandscape,
    EntropyProduction,
    MINENetwork,
    MutualInformationEstimate,
    InformationPlane,
    EnergyFunction,
    Basin,
    EntropyProductionEstimate,
)


# ==================== Fixtures ====================

@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    torch.manual_seed(42)
    np.random.seed(42)

    return {
        'X': torch.randn(500, 10),  # Input
        'Z': torch.randn(500, 8),   # Latent
        'Y': torch.randn(500, 5),   # Output
        'activations': {
            'layer_0': torch.randn(500, 64),
            'layer_1': torch.randn(500, 32),
            'layer_2': torch.randn(500, 16),
        },
        'trajectories': torch.cumsum(
            torch.randn(30, 50, 3) * 0.1, dim=1
        ),  # 30 trials, 50 timesteps, 3 dims
        'multimodal_latents': torch.cat([
            torch.randn(200, 2) + torch.tensor([2.0, 2.0]),
            torch.randn(200, 2) + torch.tensor([-2.0, -2.0]),
            torch.randn(100, 2),
        ], dim=0),
    }


@pytest.fixture
def info_analyzer():
    """Create InformationFlowAnalyzer instance"""
    return InformationFlowAnalyzer(verbose=False)


@pytest.fixture
def landscape_analyzer():
    """Create EnergyLandscape instance"""
    return EnergyLandscape(verbose=False)


@pytest.fixture
def entropy_analyzer():
    """Create EntropyProduction instance"""
    return EntropyProduction(verbose=False)


# ==================== InformationFlowAnalyzer Tests ====================

class TestInformationFlowAnalyzer:
    """Test InformationFlowAnalyzer class"""

    def test_initialization(self, info_analyzer):
        """Test analyzer initialization"""
        assert info_analyzer is not None
        assert info_analyzer.verbose == False

    def test_knn_mi_estimation(self, info_analyzer, sample_data):
        """Test k-NN mutual information estimation"""
        X, Z, Y = sample_data['X'], [sample_data['Z']], sample_data['Y']

        mi_results = info_analyzer.estimate_mutual_information(
            X, Z, Y, method='knn'
        )

        assert len(mi_results) == 1
        assert 'layer_0' in mi_results

        result = mi_results['layer_0']
        assert isinstance(result, MutualInformationEstimate)
        assert result.I_XZ >= 0  # MI is non-negative
        assert result.I_ZY >= 0
        assert hasattr(result, 'method')
        assert result.method == 'knn'

    def test_histogram_mi_estimation(self, info_analyzer, sample_data):
        """Test histogram-based MI estimation"""
        X, Z, Y = sample_data['X'], [sample_data['Z']], sample_data['Y']

        mi_results = info_analyzer.estimate_mutual_information(
            X, Z, Y, method='histogram'
        )

        assert len(mi_results) == 1
        result = mi_results['layer_0']
        assert isinstance(result, MutualInformationEstimate)
        assert result.I_XZ >= 0
        assert result.I_ZY >= 0
        assert result.method == 'histogram'

    def test_mine_mi_estimation(self, info_analyzer, sample_data):
        """Test MINE neural network MI estimation"""
        X = sample_data['X'][:100]  # Use smaller dataset for speed
        Z = [sample_data['Z'][:100]]
        Y = sample_data['Y'][:100]

        mi_results = info_analyzer.estimate_mutual_information(
            X, Z, Y,
            method='mine',
            mine_iterations=100,  # Fewer iterations for testing
        )

        assert len(mi_results) == 1
        result = mi_results['layer_0']
        assert isinstance(result, MutualInformationEstimate)
        assert result.method == 'mine'
        # MINE can be noisy with few iterations, just check it runs

    def test_information_plane(self, info_analyzer, sample_data):
        """Test information plane computation"""
        activations = sample_data['activations']
        X = sample_data['X']
        Y = sample_data['Y']

        info_plane = info_analyzer.information_plane(
            activations, X, Y, method='knn'
        )

        assert isinstance(info_plane, InformationPlane)
        assert len(info_plane.layers) == 3
        assert len(info_plane.I_XZ_per_layer) == 3
        assert len(info_plane.I_ZY_per_layer) == 3

        # All MI values should be non-negative
        for i_xz, i_zy in zip(info_plane.I_XZ_per_layer, info_plane.I_ZY_per_layer):
            assert i_xz >= 0
            assert i_zy >= 0

    def test_information_bottleneck_curve(self, info_analyzer, sample_data):
        """Test information bottleneck curve computation"""
        activations = sample_data['activations']
        X = sample_data['X']
        Y = sample_data['Y']

        # Add noise to activations to create IB curve
        beta_values = [0.1, 0.5, 1.0, 2.0]
        noisy_activations = []

        for beta in beta_values:
            noisy_acts = {}
            for layer_name, acts in activations.items():
                noise = torch.randn_like(acts) * (1.0 / beta)
                noisy_acts[layer_name] = acts + noise
            noisy_activations.append(noisy_acts)

        # Compute IB curve
        ib_curve = info_analyzer.information_bottleneck_curve(
            noisy_activations, X, Y, method='knn'
        )

        assert len(ib_curve) == len(beta_values)
        for plane in ib_curve:
            assert isinstance(plane, InformationPlane)


# ==================== EnergyLandscape Tests ====================

class TestEnergyLandscape:
    """Test EnergyLandscape class"""

    def test_initialization(self, landscape_analyzer):
        """Test landscape analyzer initialization"""
        assert landscape_analyzer is not None

    def test_density_based_landscape(self, landscape_analyzer, sample_data):
        """Test density-based energy landscape estimation"""
        latents = sample_data['multimodal_latents']

        landscape = landscape_analyzer.estimate_landscape(
            latents, method='density', grid_size=20
        )

        assert isinstance(landscape, EnergyFunction)
        assert landscape.energy.shape == (20, 20)
        assert landscape.grid_x.shape == (20,)
        assert landscape.grid_y.shape == (20,)
        assert torch.isfinite(landscape.energy).all()

    def test_score_based_landscape(self, landscape_analyzer, sample_data):
        """Test score-based energy landscape estimation"""
        latents = sample_data['multimodal_latents']

        landscape = landscape_analyzer.estimate_landscape(
            latents, method='score', grid_size=20
        )

        assert isinstance(landscape, EnergyFunction)
        assert landscape.energy.shape == (20, 20)
        assert torch.isfinite(landscape.energy).all()

    def test_quadratic_landscape(self, landscape_analyzer, sample_data):
        """Test quadratic energy landscape estimation"""
        latents = sample_data['multimodal_latents']

        landscape = landscape_analyzer.estimate_landscape(
            latents, method='quadratic', grid_size=20
        )

        assert isinstance(landscape, EnergyFunction)
        assert landscape.energy.shape == (20, 20)
        # Quadratic landscape should be smooth
        assert torch.isfinite(landscape.energy).all()

    def test_find_basins(self, landscape_analyzer, sample_data):
        """Test energy basin detection"""
        latents = sample_data['multimodal_latents']

        landscape = landscape_analyzer.estimate_landscape(
            latents, method='density', grid_size=30
        )

        basins = landscape_analyzer.find_basins(landscape, num_basins=3)

        assert len(basins) == 3
        for basin in basins:
            assert isinstance(basin, Basin)
            assert len(basin.centroid) == 2  # 2D landscape
            assert basin.energy > 0  # Energy should be positive
            assert basin.volume > 0  # Volume should be positive
            assert torch.isfinite(torch.tensor(basin.centroid)).all()

    def test_compute_barriers(self, landscape_analyzer, sample_data):
        """Test energy barrier computation between basins"""
        latents = sample_data['multimodal_latents']

        landscape = landscape_analyzer.estimate_landscape(
            latents, method='density', grid_size=30
        )

        basins = landscape_analyzer.find_basins(landscape, num_basins=3)
        barriers = landscape_analyzer.compute_barriers(landscape, basins)

        assert barriers.shape == (3, 3)
        # Diagonal should be zero (barrier to self)
        assert torch.allclose(torch.diag(barriers), torch.zeros(3), atol=1e-6)
        # Barriers should be symmetric
        assert torch.allclose(barriers, barriers.T, atol=1e-6)
        # All barriers should be non-negative
        assert (barriers >= 0).all()

    def test_visualize_landscape_2d(self, landscape_analyzer, sample_data, tmp_path):
        """Test 2D landscape visualization"""
        latents = sample_data['multimodal_latents']

        landscape = landscape_analyzer.estimate_landscape(
            latents, method='density'
        )

        basins = landscape_analyzer.find_basins(landscape, num_basins=2)

        # Test that visualization runs without error
        output_path = tmp_path / 'landscape.png'
        landscape_analyzer.visualize_landscape_2d(
            landscape,
            basins=basins,
            latents=latents,
            save_path=str(output_path)
        )

        assert output_path.exists()


# ==================== EntropyProduction Tests ====================

class TestEntropyProduction:
    """Test EntropyProduction class"""

    def test_initialization(self, entropy_analyzer):
        """Test entropy analyzer initialization"""
        assert entropy_analyzer is not None

    def test_entropy_production_estimation(self, entropy_analyzer, sample_data):
        """Test entropy production rate estimation"""
        trajectories = sample_data['trajectories']

        entropy_prod = entropy_analyzer.estimate_entropy_production(
            trajectories, dt=0.01
        )

        assert isinstance(entropy_prod, EntropyProductionEstimate)
        assert entropy_prod.dissipation_rate >= 0
        assert 0 <= entropy_prod.nonequilibrium_score <= 1
        assert entropy_prod.entropy_production_rate.shape[0] == trajectories.shape[1]

    def test_dissipation_rate(self, entropy_analyzer, sample_data):
        """Test dissipation rate calculation"""
        trajectories = sample_data['trajectories']

        entropy_prod = entropy_analyzer.estimate_entropy_production(
            trajectories, dt=0.01
        )

        # Check dissipation rate separately
        dissipation = entropy_prod.dissipation_rate
        assert isinstance(dissipation, float)
        assert dissipation >= 0

    def test_nonequilibrium_score(self, entropy_analyzer, sample_data):
        """Test nonequilibrium score calculation"""
        trajectories = sample_data['trajectories']

        entropy_prod = entropy_analyzer.estimate_entropy_production(
            trajectories, dt=0.01
        )

        score = entropy_prod.nonequilibrium_score
        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_equilibrium_vs_nonequilibrium(self, entropy_analyzer):
        """Test that equilibrium system has low entropy production"""
        # Create equilibrium-like trajectories (random walk)
        equilibrium_traj = torch.cumsum(torch.randn(20, 100, 3) * 0.05, dim=1)

        # Create nonequilibrium trajectories (directed drift)
        drift = torch.linspace(0, 5, 100).unsqueeze(0).unsqueeze(-1).expand(20, 100, 3)
        nonequilibrium_traj = equilibrium_traj + drift

        eq_entropy = entropy_analyzer.estimate_entropy_production(equilibrium_traj, dt=0.01)
        noneq_entropy = entropy_analyzer.estimate_entropy_production(nonequilibrium_traj, dt=0.01)

        # Nonequilibrium should have higher dissipation
        assert noneq_entropy.dissipation_rate >= eq_entropy.dissipation_rate


# ==================== MINENetwork Tests ====================

class TestMINENetwork:
    """Test MINE neural network"""

    def test_initialization(self):
        """Test MINE network initialization"""
        mine = MINENetwork(x_dim=10, z_dim=8, hidden_dim=64, n_layers=3)

        assert mine is not None
        assert mine.x_dim == 10
        assert mine.z_dim == 8

        # Check network has parameters
        num_params = sum(p.numel() for p in mine.parameters())
        assert num_params > 0

    def test_forward_pass(self):
        """Test forward pass through MINE network"""
        mine = MINENetwork(x_dim=10, z_dim=8, hidden_dim=64)

        x = torch.randn(32, 10)
        z = torch.randn(32, 8)

        output = mine(x, z)

        assert output.shape == (32,)
        assert torch.isfinite(output).all()

    def test_training_step(self):
        """Test that MINE can be trained"""
        mine = MINENetwork(x_dim=5, z_dim=5, hidden_dim=32)
        optimizer = torch.optim.Adam(mine.parameters(), lr=1e-3)

        # Create correlated data
        x = torch.randn(100, 5)
        z = x + torch.randn(100, 5) * 0.1  # Highly correlated

        # Training step
        mine.train()
        for _ in range(10):
            # Joint samples
            t_joint = mine(x, z).mean()

            # Marginal samples (shuffle z)
            z_shuffled = z[torch.randperm(100)]
            t_marginal = torch.logsumexp(mine(x, z_shuffled), dim=0) - np.log(100)

            # MINE lower bound
            loss = -(t_joint - t_marginal)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Just check training runs without error
        assert True


# ==================== Integration Tests ====================

class TestEnergyFlowIntegration:
    """Integration tests for complete workflows"""

    def test_full_information_analysis_pipeline(self, sample_data):
        """Test complete information flow analysis pipeline"""
        analyzer = InformationFlowAnalyzer(verbose=False)

        X = sample_data['X']
        activations = sample_data['activations']
        Y = sample_data['Y']

        # 1. Compute information plane
        info_plane = analyzer.information_plane(activations, X, Y, method='knn')

        # 2. Verify results
        assert len(info_plane.layers) == 3
        assert all(i >= 0 for i in info_plane.I_XZ_per_layer)
        assert all(i >= 0 for i in info_plane.I_ZY_per_layer)

        # 3. Check information bottleneck principle
        # Later layers should compress information about X
        # (I(X;Z) should decrease or stay similar)
        # while maintaining information about Y

    def test_full_energy_landscape_pipeline(self, sample_data):
        """Test complete energy landscape analysis pipeline"""
        analyzer = EnergyLandscape(verbose=False)

        latents = sample_data['multimodal_latents']

        # 1. Estimate landscape
        landscape = analyzer.estimate_landscape(latents, method='density')

        # 2. Find basins
        basins = analyzer.find_basins(landscape, num_basins=3)

        # 3. Compute barriers
        barriers = analyzer.compute_barriers(landscape, basins)

        # 4. Verify results
        assert len(basins) == 3
        assert barriers.shape == (3, 3)
        assert (barriers >= 0).all()
        assert torch.allclose(torch.diag(barriers), torch.zeros(3), atol=1e-6)

    def test_full_entropy_production_pipeline(self, sample_data):
        """Test complete entropy production analysis pipeline"""
        analyzer = EntropyProduction(verbose=False)

        trajectories = sample_data['trajectories']

        # 1. Estimate entropy production
        entropy_prod = analyzer.estimate_entropy_production(trajectories, dt=0.01)

        # 2. Verify results
        assert entropy_prod.dissipation_rate >= 0
        assert 0 <= entropy_prod.nonequilibrium_score <= 1
        assert len(entropy_prod.entropy_production_rate) == trajectories.shape[1]

    def test_combined_analysis(self, sample_data):
        """Test combining information flow and energy landscape analysis"""
        # This tests that both analyzers can work together

        info_analyzer = InformationFlowAnalyzer(verbose=False)
        landscape_analyzer = EnergyLandscape(verbose=False)

        activations = sample_data['activations']
        X = sample_data['X']
        Y = sample_data['Y']

        # 1. Information analysis
        info_plane = info_analyzer.information_plane(activations, X, Y, method='knn')

        # 2. Energy landscape on latent layer
        latent_layer = activations['layer_1']  # Middle layer
        landscape = landscape_analyzer.estimate_landscape(
            latent_layer, method='density'
        )

        # 3. Verify both work
        assert info_plane is not None
        assert landscape is not None

        # Could analyze relationship between information compression
        # and energy landscape structure


# ==================== Edge Cases and Robustness ====================

class TestEdgeCases:
    """Test edge cases and robustness"""

    def test_small_dataset(self, info_analyzer):
        """Test with very small dataset"""
        X = torch.randn(10, 5)
        Z = [torch.randn(10, 3)]
        Y = torch.randn(10, 2)

        # Should still run (though estimates may be poor)
        mi_results = info_analyzer.estimate_mutual_information(
            X, Z, Y, method='knn', k=3  # Small k for small dataset
        )

        assert len(mi_results) == 1

    def test_high_dimensional_data(self, info_analyzer):
        """Test with high-dimensional data"""
        X = torch.randn(200, 100)  # 100-dimensional input
        Z = [torch.randn(200, 50)]
        Y = torch.randn(200, 10)

        # Should handle high dimensions
        mi_results = info_analyzer.estimate_mutual_information(
            X, Z, Y, method='knn'
        )

        assert len(mi_results) == 1

    def test_degenerate_landscape(self, landscape_analyzer):
        """Test landscape with all points at same location"""
        # All points at origin
        latents = torch.zeros(100, 2)

        landscape = landscape_analyzer.estimate_landscape(
            latents, method='density'
        )

        # Should still produce valid landscape
        assert isinstance(landscape, EnergyFunction)
        assert torch.isfinite(landscape.energy).all()

    def test_constant_trajectories(self, entropy_analyzer):
        """Test entropy production with constant (no movement) trajectories"""
        # All trajectories are constant
        trajectories = torch.ones(10, 50, 3)

        entropy_prod = entropy_analyzer.estimate_entropy_production(
            trajectories, dt=0.01
        )

        # Should have zero or very low entropy production
        assert entropy_prod.dissipation_rate < 0.1
        assert entropy_prod.nonequilibrium_score < 0.1


# ==================== Performance Tests ====================

class TestPerformance:
    """Performance and scaling tests"""

    @pytest.mark.slow
    def test_large_scale_information_plane(self):
        """Test information plane with large dataset"""
        torch.manual_seed(42)

        X = torch.randn(5000, 20)
        Y = torch.randn(5000, 10)
        activations = {
            f'layer_{i}': torch.randn(5000, 64) for i in range(5)
        }

        analyzer = InformationFlowAnalyzer(verbose=True)

        # Should complete in reasonable time
        info_plane = analyzer.information_plane(
            activations, X, Y, method='knn'
        )

        assert len(info_plane.layers) == 5

    @pytest.mark.slow
    def test_fine_grained_landscape(self):
        """Test energy landscape with fine grid"""
        torch.manual_seed(42)

        latents = torch.randn(1000, 2)

        analyzer = EnergyLandscape(verbose=True)

        # Fine grid (100x100)
        landscape = analyzer.estimate_landscape(
            latents, method='density', grid_size=100
        )

        assert landscape.energy.shape == (100, 100)


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])

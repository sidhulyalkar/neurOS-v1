"""
Comprehensive Benchmarking Suite for NeuroFM-X Foundation Model

Evaluates:
1. Neural reconstruction quality
2. Behavioral decoding performance
3. Population geometry and dynamics
4. Computational efficiency
5. Transfer learning capability
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import json
import time
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from neuros_neurofm.models.neurofmx_multitask import NeuroFMXMultiTask


class NeuroFMXBenchmark:
    """Comprehensive benchmarking for NeuroFM-X."""

    def __init__(
        self,
        model: NeuroFMXMultiTask,
        test_loader,
        device: str = 'cuda',
        output_dir: Path = Path('./benchmark_results')
    ):
        self.model = model.to(device).eval()
        self.test_loader = test_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for results
        self.results = {}

    def run_all_benchmarks(self) -> Dict:
        """Run complete benchmark suite."""
        print(f"\n{'='*80}")
        print("NeuroFM-X Comprehensive Benchmark Suite")
        print(f"{'='*80}\n")

        with torch.no_grad():
            # 1. Neural Reconstruction
            print("1️⃣  Benchmarking Neural Reconstruction...")
            recon_metrics = self.benchmark_reconstruction()
            self.results['reconstruction'] = recon_metrics
            self._print_metrics("Reconstruction", recon_metrics)

            # 2. Behavioral Decoding
            print("\n2️⃣  Benchmarking Behavioral Decoding...")
            behavior_metrics = self.benchmark_behavior_decoding()
            self.results['behavior_decoding'] = behavior_metrics
            self._print_metrics("Behavior Decoding", behavior_metrics)

            # 3. Population Geometry
            print("\n3️⃣  Analyzing Population Geometry...")
            geometry_metrics = self.benchmark_population_geometry()
            self.results['population_geometry'] = geometry_metrics
            self._print_metrics("Population Geometry", geometry_metrics)

            # 4. Computational Efficiency
            print("\n4️⃣  Measuring Computational Efficiency...")
            efficiency_metrics = self.benchmark_computational_efficiency()
            self.results['computational_efficiency'] = efficiency_metrics
            self._print_metrics("Computational Efficiency", efficiency_metrics)

            # 5. Latent Space Quality
            print("\n5️⃣  Evaluating Latent Space Quality...")
            latent_metrics = self.benchmark_latent_quality()
            self.results['latent_quality'] = latent_metrics
            self._print_metrics("Latent Space Quality", latent_metrics)

        # Save results
        self._save_results()

        print(f"\n{'='*80}")
        print("Benchmark Complete!")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*80}\n")

        return self.results

    def benchmark_reconstruction(self) -> Dict:
        """Evaluate neural reconstruction quality."""
        all_predictions = []
        all_targets = []
        all_masks = []

        for batch in tqdm(self.test_loader, desc="Reconstruction", leave=False):
            tokens_raw = batch['tokens_raw'].to(self.device)
            unit_mask = batch['unit_mask'].to(self.device)
            unit_indices = batch['unit_indices'].to(self.device)

            # Get reconstruction from encoder head
            model_output = self.model(
                tokens_raw=tokens_raw,
                unit_mask=unit_mask,
                unit_indices=unit_indices,
                task='multi-task'
            )

            predictions = model_output['encoder']  # (B, S, N)
            targets = tokens_raw  # (B, S, N)

            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            all_masks.append((~unit_mask).cpu())  # True = valid unit

        # Concatenate all batches
        predictions = torch.cat(all_predictions, dim=0)  # (Total_B, S, N)
        targets = torch.cat(all_targets, dim=0)
        masks = torch.cat(all_masks, dim=0)  # (Total_B, N)

        # Compute metrics
        metrics = {}

        # 1. MSE (masked)
        mse_per_sample = []
        for i in range(len(predictions)):
            mask = masks[i].unsqueeze(0).expand(predictions.shape[1], -1)  # (S, N)
            pred_masked = predictions[i][mask]
            target_masked = targets[i][mask]
            mse = F.mse_loss(pred_masked, target_masked).item()
            mse_per_sample.append(mse)

        metrics['mse'] = float(np.mean(mse_per_sample))
        metrics['mse_std'] = float(np.std(mse_per_sample))

        # 2. R² Score (per unit, averaged)
        r2_scores = []
        for unit_idx in range(predictions.shape[2]):
            # Get all predictions/targets for this unit across all samples
            pred_unit = predictions[:, :, unit_idx].flatten().numpy()
            target_unit = targets[:, :, unit_idx].flatten().numpy()

            # Only compute if unit has variance
            if target_unit.std() > 1e-6:
                r2 = r2_score(target_unit, pred_unit)
                r2_scores.append(r2)

        metrics['r2_mean'] = float(np.mean(r2_scores)) if r2_scores else 0.0
        metrics['r2_std'] = float(np.std(r2_scores)) if r2_scores else 0.0
        metrics['r2_median'] = float(np.median(r2_scores)) if r2_scores else 0.0

        # 3. Correlation (per-sample)
        correlations = []
        for i in range(len(predictions)):
            mask = masks[i]
            pred = predictions[i, :, mask].flatten()
            target = targets[i, :, mask].flatten()

            if len(pred) > 1:
                corr = np.corrcoef(pred.numpy(), target.numpy())[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        metrics['correlation_mean'] = float(np.mean(correlations)) if correlations else 0.0
        metrics['correlation_std'] = float(np.std(correlations)) if correlations else 0.0

        # 4. Explained Variance
        explained_var = []
        for i in range(len(predictions)):
            mask = masks[i].unsqueeze(0).expand(predictions.shape[1], -1)
            pred = predictions[i][mask].numpy()
            target = targets[i][mask].numpy()

            var_pred = np.var(target - pred)
            var_target = np.var(target)

            if var_target > 1e-6:
                explained_var.append(1 - var_pred / var_target)

        metrics['explained_variance'] = float(np.mean(explained_var)) if explained_var else 0.0

        return metrics

    def benchmark_behavior_decoding(self) -> Dict:
        """Evaluate behavioral decoding performance."""
        all_predictions = []
        all_targets = []

        for batch in tqdm(self.test_loader, desc="Behavior Decoding", leave=False):
            tokens_raw = batch['tokens_raw'].to(self.device)
            unit_mask = batch['unit_mask'].to(self.device)
            unit_indices = batch['unit_indices'].to(self.device)

            # Get decoder output
            model_output = self.model(
                tokens_raw=tokens_raw,
                unit_mask=unit_mask,
                unit_indices=unit_indices,
                task='multi-task'
            )

            predictions = model_output['decoder']  # (B, D)
            targets = batch['behavior_target'][:, -1, :].to(self.device)  # (B, D)

            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())

        predictions = torch.cat(all_predictions, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()

        metrics = {}

        # MSE per dimension
        mse_per_dim = np.mean((predictions - targets) ** 2, axis=0)
        metrics['mse'] = float(np.mean(mse_per_dim))
        metrics['mse_per_dimension'] = mse_per_dim.tolist()

        # R² per dimension
        r2_per_dim = []
        for dim in range(predictions.shape[1]):
            r2 = r2_score(targets[:, dim], predictions[:, dim])
            r2_per_dim.append(r2)

        metrics['r2_mean'] = float(np.mean(r2_per_dim))
        metrics['r2_per_dimension'] = r2_per_dim

        # Correlation
        corr_per_dim = []
        for dim in range(predictions.shape[1]):
            corr = np.corrcoef(predictions[:, dim], targets[:, dim])[0, 1]
            corr_per_dim.append(corr if not np.isnan(corr) else 0.0)

        metrics['correlation_mean'] = float(np.mean(corr_per_dim))
        metrics['correlation_per_dimension'] = corr_per_dim

        return metrics

    def benchmark_population_geometry(self) -> Dict:
        """Analyze latent space geometry and population structure."""
        all_latents = []
        all_behaviors = []

        for batch in tqdm(self.test_loader, desc="Population Geometry", leave=False):
            tokens_raw = batch['tokens_raw'].to(self.device)
            unit_mask = batch['unit_mask'].to(self.device)
            unit_indices = batch['unit_indices'].to(self.device)

            # Extract latents (call forward without task to get raw latents)
            # Based on NeuroFMXMultiTask, task=None returns pooled latents
            latents = self.model(
                tokens_raw=tokens_raw,
                unit_mask=unit_mask,
                unit_indices=unit_indices,
                task=None
            )  # Returns (B, latent_dim)

            all_latents.append(latents.cpu())
            all_behaviors.append(batch['behavior_target'][:, -1, :].cpu())

        latents = torch.cat(all_latents, dim=0).numpy()  # (N_samples, latent_dim)
        behaviors = torch.cat(all_behaviors, dim=0).numpy()

        metrics = {}

        # 1. Effective dimensionality (PCA)
        pca = PCA()
        pca.fit(latents)

        explained_var = pca.explained_variance_ratio_
        cumsum_var = np.cumsum(explained_var)

        # Find number of dimensions to explain 90%, 95%, 99% variance
        metrics['dim_90pct'] = int(np.argmax(cumsum_var >= 0.90) + 1)
        metrics['dim_95pct'] = int(np.argmax(cumsum_var >= 0.95) + 1)
        metrics['dim_99pct'] = int(np.argmax(cumsum_var >= 0.99) + 1)

        # 2. Participation ratio (measure of dimensionality)
        eigenvalues = pca.explained_variance_
        participation_ratio = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
        metrics['participation_ratio'] = float(participation_ratio)

        # 3. Latent space statistics
        metrics['latent_mean_norm'] = float(np.linalg.norm(latents.mean(axis=0)))
        metrics['latent_std_norm'] = float(np.linalg.norm(latents.std(axis=0)))

        # 4. Behavior-latent correlation
        # Compute canonical correlation between latents and behaviors
        from scipy.stats import pearsonr

        corr_matrix = np.zeros((latents.shape[1], behaviors.shape[1]))
        for i in range(latents.shape[1]):
            for j in range(behaviors.shape[1]):
                corr, _ = pearsonr(latents[:, i], behaviors[:, j])
                corr_matrix[i, j] = corr if not np.isnan(corr) else 0.0

        metrics['max_behavior_correlation'] = float(np.abs(corr_matrix).max())
        metrics['mean_behavior_correlation'] = float(np.abs(corr_matrix).mean())

        return metrics

    def benchmark_computational_efficiency(self) -> Dict:
        """Measure inference speed and memory usage."""
        metrics = {}

        # Create dummy input
        dummy_input = {
            'tokens_raw': torch.randn(1, 100, 384).to(self.device),
            'unit_mask': torch.zeros(1, 384, dtype=torch.bool).to(self.device),
            'unit_indices': torch.arange(384).unsqueeze(0).to(self.device)
        }

        # Warmup
        for _ in range(10):
            _ = self.model(**dummy_input, task='multi-task')

        # Benchmark inference time
        torch.cuda.synchronize()
        start_time = time.time()

        n_runs = 100
        for _ in range(n_runs):
            _ = self.model(**dummy_input, task='multi-task')

        torch.cuda.synchronize()
        end_time = time.time()

        avg_time_ms = (end_time - start_time) / n_runs * 1000
        metrics['inference_time_ms'] = float(avg_time_ms)
        metrics['throughput_samples_per_sec'] = float(1000 / avg_time_ms)

        # Memory usage
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            _ = self.model(**dummy_input, task='multi-task')
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
            metrics['peak_memory_mb'] = float(peak_memory_mb)

        # Model size
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        metrics['total_parameters'] = int(param_count)
        metrics['trainable_parameters'] = int(trainable_count)
        metrics['model_size_mb'] = float(param_count * 4 / 1024**2)  # Assuming float32

        return metrics

    def benchmark_latent_quality(self) -> Dict:
        """Evaluate latent representation quality."""
        all_latents = []

        for batch in tqdm(self.test_loader, desc="Latent Quality", leave=False):
            tokens_raw = batch['tokens_raw'].to(self.device)
            unit_mask = batch['unit_mask'].to(self.device)
            unit_indices = batch['unit_indices'].to(self.device)

            latents = self.model(
                tokens_raw=tokens_raw,
                unit_mask=unit_mask,
                unit_indices=unit_indices,
                task=None
            )

            all_latents.append(latents.cpu())

        latents = torch.cat(all_latents, dim=0).numpy()

        metrics = {}

        # 1. Latent space coverage (measure of diversity)
        from scipy.spatial.distance import pdist

        # Sample for efficiency
        n_samples = min(1000, len(latents))
        sampled_latents = latents[np.random.choice(len(latents), n_samples, replace=False)]

        distances = pdist(sampled_latents, metric='euclidean')
        metrics['mean_pairwise_distance'] = float(distances.mean())
        metrics['std_pairwise_distance'] = float(distances.std())

        # 2. Latent smoothness (temporal if available)
        # Skip for now as we need sequential samples

        # 3. Latent sparsity
        sparsity = (np.abs(latents) < 0.01).mean()
        metrics['sparsity'] = float(sparsity)

        # 4. Latent statistics
        metrics['mean_activation'] = float(latents.mean())
        metrics['std_activation'] = float(latents.std())
        metrics['max_activation'] = float(latents.max())
        metrics['min_activation'] = float(latents.min())

        return metrics

    def _print_metrics(self, category: str, metrics: Dict):
        """Pretty print metrics."""
        print(f"\n{category} Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"   {key:30s}: {value:.6f}")
            elif isinstance(value, (int, str)):
                print(f"   {key:30s}: {value}")
            # Skip lists for clean output

    def _save_results(self):
        """Save benchmark results to JSON."""
        output_file = self.output_dir / "benchmark_results.json"

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✓ Results saved to: {output_file}")

    def generate_report(self):
        """Generate a comprehensive benchmark report."""
        report_path = self.output_dir / "benchmark_report.md"

        with open(report_path, 'w') as f:
            f.write("# NeuroFM-X Benchmark Report\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for category, metrics in self.results.items():
                f.write(f"## {category.replace('_', ' ').title()}\n\n")

                for key, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"- **{key}:** {value:.6f}\n")
                    elif isinstance(value, (int, str)):
                        f.write(f"- **{key}:** {value}\n")

                f.write("\n")

        print(f"✓ Report saved to: {report_path}")


def main():
    """Run benchmarking suite."""
    import argparse
    from full_train_streaming import Config, load_allen_dataset, StreamingNeuropixelsDataset, collate_fn
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="Benchmark NeuroFM-X")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='./benchmark_results',
                        help='Output directory for results')

    args = parser.parse_args()

    print("Loading model and data...")

    # Load config
    config = Config()

    # Load test data
    cache, session_ids = load_allen_dataset(config)
    test_session_ids = session_ids[:4]  # Use first 4 sessions for testing

    test_dataset = StreamingNeuropixelsDataset(
        processed_dir=config.data_dir / "processed_sequences_full",
        session_ids=test_session_ids
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location='cuda')

    model = NeuroFMXMultiTask(
        d_model=config.d_model,
        n_mamba_blocks=config.n_mamba_blocks,
        n_latents=config.n_latents,
        latent_dim=config.latent_dim,
        n_perceiver_layers=config.n_perceiver_layers,
        n_popt_layers=config.n_popt_layers,
        use_popt=config.use_popt,
        use_multi_rate=config.use_multi_rate,
        downsample_rates=config.downsample_rates,
        enable_decoder=config.enable_decoder,
        enable_encoder=config.enable_encoder,
        enable_contrastive=config.enable_contrastive,
        enable_forecast=config.enable_forecast,
        decoder_output_dim=config.decoder_output_dim,
        encoder_output_dim=config.max_units,
        dropout=config.dropout,
    )

    model.load_state_dict(checkpoint['model_state_dict'])

    # Run benchmark
    benchmark = NeuroFMXBenchmark(
        model=model,
        test_loader=test_loader,
        output_dir=Path(args.output_dir)
    )

    benchmark.run_all_benchmarks()
    benchmark.generate_report()


if __name__ == '__main__':
    main()

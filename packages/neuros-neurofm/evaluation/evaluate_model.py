"""
NeuroFMx Model Evaluation and Benchmarking

Comprehensive evaluation suite for trained models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt


class NeuroFMXBenchmark:
    """
    Comprehensive benchmark suite for NeuroFMx.

    Evaluates:
    - Neural reconstruction (encoder task)
    - Behavior decoding (decoder task)
    - Latent space quality
    - Cross-domain transfer
    - Few-shot adaptation
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        output_dir: str = './evaluation_results'
    ):
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {}

    def evaluate_reconstruction(
        self,
        dataloader: torch.utils.data.DataLoader,
        modality: str = 'spike'
    ) -> Dict[str, float]:
        """
        Evaluate neural reconstruction quality.

        Metrics:
        - MSE (Mean Squared Error)
        - MAE (Mean Absolute Error)
        - Pearson correlation
        - R² score
        """
        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                modality_dict = batch['modality_dict']
                target = modality_dict[modality].to(self.device)

                # Forward pass
                outputs = self.model(modality_dict, task='encoder')

                # Get reconstruction
                if 'encoder' in outputs:
                    pred = outputs['encoder']
                else:
                    continue

                all_predictions.append(pred.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        # Concatenate
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # Flatten for metrics
        pred_flat = predictions.reshape(-1)
        target_flat = targets.reshape(-1)

        # Compute metrics
        mse = mean_squared_error(target_flat, pred_flat)
        mae = mean_absolute_error(target_flat, pred_flat)
        r2 = r2_score(target_flat, pred_flat)

        # Pearson correlation
        pearson_r, pearson_p = pearsonr(pred_flat, target_flat)

        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2),
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p)
        }

        self.results[f'reconstruction_{modality}'] = metrics

        print(f"\n{modality.upper()} Reconstruction Metrics:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Pearson r: {pearson_r:.4f} (p={pearson_p:.2e})")

        return metrics

    def evaluate_decoding(
        self,
        dataloader: torch.utils.data.DataLoader,
        behavior_target: str = 'velocity'
    ) -> Dict[str, float]:
        """
        Evaluate behavior decoding from neural data.

        Metrics:
        - R² score
        - Correlation
        - MSE
        """
        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                modality_dict = batch['modality_dict']
                behavior = batch.get('behavior', {}).get(behavior_target)

                if behavior is None:
                    continue

                behavior = behavior.to(self.device)

                # Forward pass
                outputs = self.model(modality_dict, task='decoder')

                if 'decoder' in outputs:
                    pred = outputs['decoder']
                    all_predictions.append(pred.cpu().numpy())
                    all_targets.append(behavior.cpu().numpy())

        if len(all_predictions) == 0:
            print(f"No {behavior_target} data found for decoding evaluation")
            return {}

        # Concatenate
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # Flatten
        pred_flat = predictions.reshape(-1)
        target_flat = targets.reshape(-1)

        # Metrics
        r2 = r2_score(target_flat, pred_flat)
        mse = mean_squared_error(target_flat, pred_flat)
        pearson_r, pearson_p = pearsonr(pred_flat, target_flat)

        metrics = {
            'r2': float(r2),
            'mse': float(mse),
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p)
        }

        self.results[f'decoding_{behavior_target}'] = metrics

        print(f"\n{behavior_target.upper()} Decoding Metrics:")
        print(f"  R²: {r2:.4f}")
        print(f"  MSE: {mse:.6f}")
        print(f"  Pearson r: {pearson_r:.4f}")

        return metrics

    def evaluate_latent_space(
        self,
        dataloader: torch.utils.data.DataLoader,
        labels: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate latent space quality.

        Metrics:
        - Dimensionality (participation ratio)
        - Clustering quality (silhouette score if labels provided)
        - Smoothness (neighbor consistency)
        """
        self.model.eval()
        all_latents = []
        all_labels = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                modality_dict = batch['modality_dict']

                # Get latent representations
                outputs = self.model(modality_dict, task='contrastive')

                if 'contrastive' in outputs:
                    latents = outputs['contrastive']
                    all_latents.append(latents.cpu().numpy())

                    if 'labels' in batch:
                        all_labels.append(batch['labels'].cpu().numpy())

        # Concatenate
        latents = np.concatenate(all_latents, axis=0)

        if len(all_labels) > 0:
            labels = np.concatenate(all_labels, axis=0)
        else:
            labels = None

        # PCA for dimensionality analysis
        pca = PCA()
        pca.fit(latents)
        explained_var = pca.explained_variance_ratio_

        # Participation ratio
        participation_ratio = (explained_var.sum() ** 2) / (explained_var ** 2).sum()

        # Effective dimensionality (95% variance)
        cumsum = np.cumsum(explained_var)
        n_effective = np.argmax(cumsum >= 0.95) + 1

        metrics = {
            'participation_ratio': float(participation_ratio),
            'n_effective_dims': int(n_effective),
            'explained_variance_ratio': explained_var[:10].tolist()  # Top 10 PCs
        }

        # Clustering quality if labels provided
        if labels is not None:
            from sklearn.metrics import silhouette_score
            try:
                silhouette = silhouette_score(latents, labels)
                metrics['silhouette_score'] = float(silhouette)
            except:
                pass

        self.results['latent_space'] = metrics

        print(f"\nLatent Space Metrics:")
        print(f"  Participation Ratio: {participation_ratio:.2f}")
        print(f"  Effective Dimensions: {n_effective}")
        if 'silhouette_score' in metrics:
            print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")

        return metrics

    def evaluate_cross_domain(
        self,
        source_dataloader: torch.utils.data.DataLoader,
        target_dataloader: torch.utils.data.DataLoader,
        target_labels: np.ndarray,
        n_train: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate cross-domain transfer learning.

        Freeze model, extract features, train linear classifier on small target set.

        Args:
            source_dataloader: Source domain data
            target_dataloader: Target domain data
            target_labels: Labels for target data
            n_train: Number of target examples for training

        Returns:
            metrics: Transfer learning metrics
        """
        self.model.eval()

        # Extract features from target domain
        print("\nExtracting target domain features...")
        all_features = []
        all_labels = []

        with torch.no_grad():
            for batch in target_dataloader:
                modality_dict = batch['modality_dict']

                outputs = self.model(modality_dict, task='contrastive')

                if 'contrastive' in outputs:
                    features = outputs['contrastive']
                    all_features.append(features.cpu().numpy())

                if 'labels' in batch:
                    all_labels.append(batch['labels'].cpu().numpy())

        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0) if all_labels else target_labels

        # Split into train/test
        indices = np.random.permutation(len(features))
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        X_train = features[train_indices]
        y_train = labels[train_indices]
        X_test = features[test_indices]
        y_test = labels[test_indices]

        # Train linear classifier
        print(f"Training linear classifier on {n_train} examples...")
        classifier = Ridge(alpha=1.0)
        classifier.fit(X_train, y_train)

        # Evaluate
        y_pred = classifier.predict(X_test)

        # Metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        pearson_r, _ = pearsonr(y_test, y_pred)

        metrics = {
            'transfer_r2': float(r2),
            'transfer_mse': float(mse),
            'transfer_correlation': float(pearson_r),
            'n_train_samples': n_train
        }

        self.results['cross_domain_transfer'] = metrics

        print(f"\nCross-Domain Transfer (n={n_train}):")
        print(f"  R²: {r2:.4f}")
        print(f"  Correlation: {pearson_r:.4f}")

        return metrics

    def evaluate_few_shot(
        self,
        dataloader: torch.utils.data.DataLoader,
        k_shot: int = 5,
        n_classes: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate few-shot learning capability.

        Args:
            dataloader: Test data
            k_shot: Number of examples per class
            n_classes: Number of classes

        Returns:
            metrics: Few-shot learning metrics
        """
        from neuros_neurofm.meta_learning.few_shot import PrototypicalNetwork

        # Create prototypical network
        proto_net = PrototypicalNetwork(self.model, distance_metric='euclidean')
        proto_net.to(self.device)
        proto_net.eval()

        # Sample k-shot episodes and evaluate
        # This is a simplified version - full implementation in few_shot.py

        print(f"\nFew-Shot Learning ({k_shot}-shot, {n_classes}-way):")
        print("  See meta_learning/few_shot.py for full evaluation")

        metrics = {
            'k_shot': k_shot,
            'n_way': n_classes,
            'status': 'requires_episode_sampling'
        }

        self.results['few_shot'] = metrics

        return metrics

    def run_full_benchmark(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        modalities: List[str] = ['spike', 'lfp', 'calcium']
    ) -> Dict:
        """
        Run complete benchmark suite.

        Args:
            train_loader: Training data
            test_loader: Test data
            modalities: List of modalities to evaluate

        Returns:
            results: Complete evaluation results
        """
        print("="*80)
        print("NeuroFMx Benchmark Suite")
        print("="*80)

        # 1. Reconstruction
        print("\n[1/5] Evaluating Neural Reconstruction...")
        for modality in modalities:
            try:
                self.evaluate_reconstruction(test_loader, modality=modality)
            except Exception as e:
                print(f"  Error evaluating {modality}: {e}")

        # 2. Decoding
        print("\n[2/5] Evaluating Behavior Decoding...")
        try:
            self.evaluate_decoding(test_loader, behavior_target='velocity')
        except Exception as e:
            print(f"  Error: {e}")

        # 3. Latent Space
        print("\n[3/5] Evaluating Latent Space...")
        try:
            self.evaluate_latent_space(test_loader)
        except Exception as e:
            print(f"  Error: {e}")

        # 4. Cross-domain
        print("\n[4/5] Evaluating Cross-Domain Transfer...")
        try:
            # Use subset of test as "target domain"
            self.evaluate_cross_domain(
                train_loader,
                test_loader,
                target_labels=None,
                n_train=100
            )
        except Exception as e:
            print(f"  Error: {e}")

        # 5. Few-shot
        print("\n[5/5] Evaluating Few-Shot Learning...")
        try:
            self.evaluate_few_shot(test_loader, k_shot=5, n_classes=5)
        except Exception as e:
            print(f"  Error: {e}")

        # Save results
        self.save_results()

        print("\n" + "="*80)
        print("Benchmark Complete!")
        print(f"Results saved to: {self.output_dir}")
        print("="*80)

        return self.results

    def save_results(self):
        """Save evaluation results to JSON."""
        output_file = self.output_dir / 'benchmark_results.json'

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved to: {output_file}")

    def visualize_results(self):
        """Create visualization plots for results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Reconstruction metrics
        ax = axes[0, 0]
        recon_metrics = {k: v for k, v in self.results.items() if 'reconstruction' in k}
        if recon_metrics:
            r2_scores = [v['r2'] for v in recon_metrics.values()]
            modalities = [k.split('_')[1] for k in recon_metrics.keys()]
            ax.bar(modalities, r2_scores)
            ax.set_title('Reconstruction Quality (R²)')
            ax.set_ylabel('R² Score')
            ax.set_ylim([0, 1])

        # 2. Decoding metrics
        ax = axes[0, 1]
        decoding_metrics = {k: v for k, v in self.results.items() if 'decoding' in k}
        if decoding_metrics:
            r2_scores = [v['r2'] for v in decoding_metrics.values()]
            behaviors = [k.split('_')[1] for k in decoding_metrics.keys()]
            ax.bar(behaviors, r2_scores)
            ax.set_title('Decoding Performance (R²)')
            ax.set_ylabel('R² Score')
            ax.set_ylim([0, 1])

        # 3. Latent space dimensionality
        ax = axes[1, 0]
        if 'latent_space' in self.results:
            explained_var = self.results['latent_space']['explained_variance_ratio']
            ax.plot(range(1, len(explained_var)+1), np.cumsum(explained_var), 'o-')
            ax.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
            ax.set_title('Latent Space Dimensionality')
            ax.set_xlabel('Number of Components')
            ax.set_ylabel('Cumulative Explained Variance')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 4. Transfer learning
        ax = axes[1, 1]
        if 'cross_domain_transfer' in self.results:
            transfer = self.results['cross_domain_transfer']
            ax.bar(['Cross-Domain\nTransfer'], [transfer['transfer_r2']])
            ax.set_title('Transfer Learning Performance')
            ax.set_ylabel('R² Score')
            ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(self.output_dir / 'benchmark_visualization.png', dpi=150)
        print(f"Visualization saved to: {self.output_dir / 'benchmark_visualization.png'}")
        plt.close()


def main():
    """Example usage of benchmark suite."""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate NeuroFMx model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to evaluation data')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on')

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint)

    # Initialize model (architecture-specific)
    from neuros_neurofm.models.multimodal_neurofmx import MultiModalNeuroFMX
    model = MultiModalNeuroFMX(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load data
    # TODO: Implement data loading based on your dataset format

    # Run benchmark
    benchmark = NeuroFMXBenchmark(model, device=args.device, output_dir=args.output_dir)

    # results = benchmark.run_full_benchmark(train_loader, test_loader)

    # Visualize
    # benchmark.visualize_results()


if __name__ == '__main__':
    main()

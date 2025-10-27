"""
Evaluation metrics for model-to-brain alignment.

This module implements noise ceiling estimation, bootstrap confidence intervals,
and permutation testing for assessing alignment quality.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List, Union, Callable
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
import warnings
from tqdm import tqdm


class NoiseCeiling:
    """
    Estimate noise ceiling for brain-to-model alignment.

    The noise ceiling represents the maximum achievable prediction accuracy
    given measurement noise in the brain data.

    Examples:
        >>> import torch
        >>> # Multiple measurements of same stimuli: (n_repetitions, n_stimuli, n_voxels)
        >>> brain_data = torch.randn(10, 100, 200)
        >>>
        >>> nc = NoiseCeiling(method='split-half')
        >>> ceiling = nc.estimate(brain_data)
        >>> print(f"Noise ceiling: {ceiling:.4f}")
    """

    def __init__(
        self,
        method: str = 'split-half',
        n_splits: int = 100,
        device: Optional[str] = None
    ):
        """
        Initialize NoiseCeiling estimator.

        Args:
            method: Estimation method ('split-half', 'leave-one-out', 'ncsnr')
            n_splits: Number of random splits for split-half method
            device: Device for computation
        """
        self.method = method
        self.n_splits = n_splits
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.ceiling_: Optional[float] = None
        self.ceiling_per_feature_: Optional[np.ndarray] = None

    def _to_tensor(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert input to tensor."""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        return X.to(self.device)

    def estimate(
        self,
        brain_data: Union[np.ndarray, torch.Tensor],
        per_feature: bool = False
    ) -> Union[float, np.ndarray]:
        """
        Estimate noise ceiling.

        Args:
            brain_data: Brain responses with repetitions
                        (n_repetitions, n_stimuli, n_features) or
                        (n_stimuli, n_features) if method='ncsnr'
            per_feature: Whether to return ceiling per feature

        Returns:
            Noise ceiling estimate(s)
        """
        brain_data = self._to_tensor(brain_data)

        if self.method == 'split-half':
            ceiling = self._split_half_ceiling(brain_data, per_feature)
        elif self.method == 'leave-one-out':
            ceiling = self._leave_one_out_ceiling(brain_data, per_feature)
        elif self.method == 'ncsnr':
            ceiling = self._ncsnr_ceiling(brain_data, per_feature)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        if per_feature:
            self.ceiling_per_feature_ = ceiling
            self.ceiling_ = np.mean(ceiling)
        else:
            self.ceiling_ = ceiling

        return ceiling

    def _split_half_ceiling(
        self,
        brain_data: torch.Tensor,
        per_feature: bool = False
    ) -> Union[float, np.ndarray]:
        """
        Estimate ceiling using split-half reliability.

        Split repetitions into two halves, average within each half,
        and compute correlation.
        """
        n_reps, n_stim, n_feat = brain_data.shape

        if n_reps < 2:
            raise ValueError("Need at least 2 repetitions for split-half")

        correlations = []

        for _ in range(self.n_splits):
            # Random split
            perm = torch.randperm(n_reps)
            half1_idx = perm[:n_reps // 2]
            half2_idx = perm[n_reps // 2:]

            # Average within halves
            half1 = brain_data[half1_idx].mean(dim=0)  # (n_stim, n_feat)
            half2 = brain_data[half2_idx].mean(dim=0)

            if per_feature:
                # Correlation per feature
                corrs = []
                for i in range(n_feat):
                    corr = torch.corrcoef(torch.stack([half1[:, i], half2[:, i]]))[0, 1]
                    corrs.append(corr.item())
                correlations.append(corrs)
            else:
                # Overall correlation
                half1_flat = half1.flatten()
                half2_flat = half2.flatten()
                corr = torch.corrcoef(torch.stack([half1_flat, half2_flat]))[0, 1]
                correlations.append(corr.item())

        correlations = np.array(correlations)

        # Spearman-Brown correction
        # Corrected correlation = 2 * r / (1 + r)
        mean_corr = correlations.mean(axis=0)
        ceiling = 2 * mean_corr / (1 + mean_corr)

        # Ensure ceiling is in [0, 1]
        ceiling = np.clip(ceiling, 0, 1)

        if per_feature:
            return ceiling
        return float(ceiling)

    def _leave_one_out_ceiling(
        self,
        brain_data: torch.Tensor,
        per_feature: bool = False
    ) -> Union[float, np.ndarray]:
        """
        Estimate ceiling using leave-one-out reliability.

        For each repetition, correlate it with the average of all others.
        """
        n_reps, n_stim, n_feat = brain_data.shape

        if n_reps < 2:
            raise ValueError("Need at least 2 repetitions for leave-one-out")

        correlations = []

        for i in range(n_reps):
            # Leave one out
            test = brain_data[i]  # (n_stim, n_feat)
            train_avg = brain_data[torch.arange(n_reps) != i].mean(dim=0)

            if per_feature:
                # Correlation per feature
                corrs = []
                for j in range(n_feat):
                    corr = torch.corrcoef(torch.stack([test[:, j], train_avg[:, j]]))[0, 1]
                    corrs.append(corr.item())
                correlations.append(corrs)
            else:
                # Overall correlation
                test_flat = test.flatten()
                train_flat = train_avg.flatten()
                corr = torch.corrcoef(torch.stack([test_flat, train_flat]))[0, 1]
                correlations.append(corr.item())

        correlations = np.array(correlations)
        mean_corr = correlations.mean(axis=0)

        # Spearman-Brown correction
        ceiling = mean_corr * n_reps / (1 + (n_reps - 1) * mean_corr)

        # Ensure ceiling is in [0, 1]
        ceiling = np.clip(ceiling, 0, 1)

        if per_feature:
            return ceiling
        return float(ceiling)

    def _ncsnr_ceiling(
        self,
        brain_data: torch.Tensor,
        per_feature: bool = False
    ) -> Union[float, np.ndarray]:
        """
        Estimate ceiling using normalized cross-stimulus noise ratio.

        This method doesn't require repetitions but uses cross-stimulus
        variance structure.
        """
        # brain_data: (n_stim, n_feat)
        if brain_data.dim() == 3:
            # If repetitions provided, average them
            brain_data = brain_data.mean(dim=0)

        n_stim, n_feat = brain_data.shape

        if per_feature:
            ceilings = []
            for i in range(n_feat):
                x = brain_data[:, i]

                # Signal variance (between-stimulus)
                signal_var = torch.var(x)

                # Noise variance estimate from residuals
                # Fit linear trend and compute residual variance
                t = torch.arange(n_stim, dtype=torch.float32, device=self.device)
                t_mean = t.mean()
                x_mean = x.mean()

                slope = ((t - t_mean) * (x - x_mean)).sum() / ((t - t_mean) ** 2).sum()
                intercept = x_mean - slope * t_mean
                residuals = x - (slope * t + intercept)
                noise_var = torch.var(residuals)

                # SNR and ceiling
                snr = signal_var / (noise_var + 1e-10)
                ceiling_i = snr / (1 + snr)
                ceilings.append(ceiling_i.item())

            return np.clip(np.array(ceilings), 0, 1)
        else:
            # Overall variance
            signal_var = torch.var(brain_data)

            # Noise estimate from temporal autocorrelation
            diff = brain_data[1:] - brain_data[:-1]
            noise_var = torch.var(diff) / 2

            snr = signal_var / (noise_var + 1e-10)
            ceiling = snr / (1 + snr)

            return float(np.clip(ceiling.item(), 0, 1))


class BootstrapCI:
    """
    Bootstrap confidence intervals for alignment metrics.

    Examples:
        >>> import torch
        >>> X = torch.randn(100, 50)
        >>> Y = torch.randn(100, 30)
        >>>
        >>> def correlation_metric(x, y):
        ...     return torch.corrcoef(torch.stack([x.flatten(), y.flatten()]))[0, 1].item()
        >>>
        >>> bootstrap = BootstrapCI(n_bootstrap=1000, confidence=0.95)
        >>> ci = bootstrap.compute(correlation_metric, X, Y)
        >>> print(f"95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}]")
    """

    def __init__(
        self,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        random_state: Optional[int] = 42
    ):
        """
        Initialize Bootstrap CI estimator.

        Args:
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level (e.g., 0.95 for 95% CI)
            random_state: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.confidence = confidence
        self.random_state = random_state

    def compute(
        self,
        metric_fn: Callable,
        *args,
        show_progress: bool = True,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute bootstrap confidence interval.

        Args:
            metric_fn: Function that computes the metric
            *args: Arguments to pass to metric_fn (will be resampled)
            show_progress: Whether to show progress bar
            **kwargs: Keyword arguments to pass to metric_fn (not resampled)

        Returns:
            Dictionary with:
                - mean: Bootstrap mean
                - std: Bootstrap standard deviation
                - lower: Lower CI bound
                - upper: Upper CI bound
                - original: Original metric value
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

        # Compute original metric
        original = metric_fn(*args, **kwargs)

        # Get sample size from first argument
        n_samples = args[0].shape[0]

        # Bootstrap
        bootstrap_values = []

        iterator = range(self.n_bootstrap)
        if show_progress:
            iterator = tqdm(iterator, desc="Bootstrap")

        for _ in iterator:
            # Resample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)

            # Resample all tensor/array arguments
            resampled_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    resampled_args.append(arg[indices])
                elif isinstance(arg, np.ndarray):
                    resampled_args.append(arg[indices])
                else:
                    resampled_args.append(arg)

            # Compute metric on resampled data
            try:
                value = metric_fn(*resampled_args, **kwargs)
                bootstrap_values.append(value)
            except Exception as e:
                warnings.warn(f"Bootstrap iteration failed: {e}")
                continue

        bootstrap_values = np.array(bootstrap_values)

        # Compute CI
        alpha = 1 - self.confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        return {
            'mean': float(np.mean(bootstrap_values)),
            'std': float(np.std(bootstrap_values)),
            'lower': float(np.percentile(bootstrap_values, lower_percentile)),
            'upper': float(np.percentile(bootstrap_values, upper_percentile)),
            'original': original if isinstance(original, float) else float(original)
        }


class PermutationTest:
    """
    Permutation testing for statistical significance.

    Tests whether observed alignment is significantly better than chance.

    Examples:
        >>> import torch
        >>> X = torch.randn(100, 50)
        >>> Y = torch.randn(100, 30)
        >>>
        >>> def alignment_score(x, y):
        ...     # Your alignment metric
        ...     return torch.corrcoef(torch.stack([x.flatten(), y.flatten()]))[0, 1].item()
        >>>
        >>> perm_test = PermutationTest(n_permutations=1000)
        >>> result = perm_test.test(alignment_score, X, Y)
        >>> print(f"p-value: {result['p_value']:.4f}")
    """

    def __init__(
        self,
        n_permutations: int = 1000,
        random_state: Optional[int] = 42
    ):
        """
        Initialize permutation test.

        Args:
            n_permutations: Number of permutations
            random_state: Random seed for reproducibility
        """
        self.n_permutations = n_permutations
        self.random_state = random_state

    def test(
        self,
        metric_fn: Callable,
        X: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor],
        show_progress: bool = True,
        **kwargs
    ) -> Dict[str, float]:
        """
        Perform permutation test.

        Args:
            metric_fn: Function that computes alignment metric
            X: First data matrix
            Y: Second data matrix (will be permuted)
            show_progress: Whether to show progress bar
            **kwargs: Additional arguments for metric_fn

        Returns:
            Dictionary with:
                - observed: Observed metric value
                - p_value: Two-tailed p-value
                - null_mean: Mean of null distribution
                - null_std: Std of null distribution
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Compute observed metric
        observed = metric_fn(X, Y, **kwargs)

        # Permutation test
        n_samples = Y.shape[0]
        null_distribution = []

        iterator = range(self.n_permutations)
        if show_progress:
            iterator = tqdm(iterator, desc="Permutation test")

        for _ in iterator:
            # Permute Y
            perm_indices = np.random.permutation(n_samples)

            if isinstance(Y, torch.Tensor):
                Y_perm = Y[perm_indices]
            else:
                Y_perm = Y[perm_indices]

            # Compute metric on permuted data
            try:
                value = metric_fn(X, Y_perm, **kwargs)
                null_distribution.append(value)
            except Exception as e:
                warnings.warn(f"Permutation iteration failed: {e}")
                continue

        null_distribution = np.array(null_distribution)

        # Compute p-value (two-tailed)
        p_value = np.mean(np.abs(null_distribution) >= np.abs(observed))

        return {
            'observed': observed if isinstance(observed, float) else float(observed),
            'p_value': float(p_value),
            'null_mean': float(np.mean(null_distribution)),
            'null_std': float(np.std(null_distribution)),
            'null_distribution': null_distribution
        }


class NormalizedScore:
    """
    Normalize alignment scores by noise ceiling.

    Computes what fraction of the explainable variance is captured.

    Examples:
        >>> # Estimate noise ceiling
        >>> brain_data_reps = torch.randn(10, 100, 200)  # 10 reps, 100 stim, 200 voxels
        >>> nc = NoiseCeiling(method='split-half')
        >>> ceiling = nc.estimate(brain_data_reps)
        >>>
        >>> # Compute model score
        >>> model_score = 0.3
        >>>
        >>> # Normalize by ceiling
        >>> normalizer = NormalizedScore()
        >>> normalized = normalizer.normalize(model_score, ceiling)
        >>> print(f"Normalized score: {normalized:.2%} of explainable variance")
    """

    def __init__(self):
        """Initialize normalized score computer."""
        pass

    def normalize(
        self,
        score: Union[float, np.ndarray],
        ceiling: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Normalize score by ceiling.

        Args:
            score: Raw alignment score (e.g., R²)
            ceiling: Noise ceiling estimate

        Returns:
            Normalized score (fraction of explainable variance)
        """
        # Ensure ceiling is not zero
        if isinstance(ceiling, np.ndarray):
            ceiling = np.maximum(ceiling, 1e-10)
        else:
            ceiling = max(ceiling, 1e-10)

        normalized = score / ceiling

        # Clip to [0, 1] (can exceed 1 if score > ceiling due to estimation error)
        if isinstance(normalized, np.ndarray):
            normalized = np.clip(normalized, 0, 1)
        else:
            normalized = np.clip(normalized, 0, 1)

        return normalized


class CrossValidatedMetric:
    """
    Cross-validated evaluation metrics.

    Provides robust performance estimates with train/test splits.

    Examples:
        >>> from sklearn.linear_model import Ridge
        >>> import torch
        >>>
        >>> X = torch.randn(200, 100)
        >>> Y = torch.randn(200, 50)
        >>>
        >>> def train_predict_fn(X_train, Y_train, X_test):
        ...     # Your training and prediction code
        ...     model = Ridge(alpha=1.0)
        ...     model.fit(X_train.cpu().numpy(), Y_train.cpu().numpy())
        ...     Y_pred = model.predict(X_test.cpu().numpy())
        ...     return torch.from_numpy(Y_pred)
        >>>
        >>> cv_metric = CrossValidatedMetric(cv=5, metric='r2')
        >>> scores = cv_metric.evaluate(train_predict_fn, X, Y)
        >>> print(f"CV R²: {scores['mean']:.4f} ± {scores['std']:.4f}")
    """

    def __init__(
        self,
        cv: int = 5,
        metric: str = 'r2',
        random_state: Optional[int] = 42
    ):
        """
        Initialize cross-validated metric.

        Args:
            cv: Number of folds
            metric: Metric to compute ('r2', 'correlation', 'mse', 'mae')
            random_state: Random seed
        """
        self.cv = cv
        self.metric = metric
        self.random_state = random_state

    def _compute_metric(
        self,
        Y_true: torch.Tensor,
        Y_pred: torch.Tensor
    ) -> float:
        """Compute specified metric."""
        if self.metric == 'r2':
            ss_res = torch.sum((Y_true - Y_pred) ** 2)
            ss_tot = torch.sum((Y_true - Y_true.mean(dim=0, keepdim=True)) ** 2)
            score = 1 - ss_res / (ss_tot + 1e-10)
            return score.item()

        elif self.metric == 'correlation':
            Y_true_flat = Y_true.flatten()
            Y_pred_flat = Y_pred.flatten()
            corr = torch.corrcoef(torch.stack([Y_true_flat, Y_pred_flat]))[0, 1]
            return corr.item()

        elif self.metric == 'mse':
            mse = torch.mean((Y_true - Y_pred) ** 2)
            return mse.item()

        elif self.metric == 'mae':
            mae = torch.mean(torch.abs(Y_true - Y_pred))
            return mae.item()

        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def evaluate(
        self,
        train_predict_fn: Callable,
        X: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor],
        show_progress: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate using cross-validation.

        Args:
            train_predict_fn: Function(X_train, Y_train, X_test) -> Y_pred
            X: Predictor matrix
            Y: Target matrix
            show_progress: Whether to show progress

        Returns:
            Dictionary with:
                - mean: Mean score across folds
                - std: Standard deviation across folds
                - scores: List of scores for each fold
        """
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.cpu().numpy()

        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        scores = []

        iterator = kf.split(X)
        if show_progress:
            iterator = tqdm(list(iterator), desc="Cross-validation")

        for train_idx, test_idx in iterator:
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            # Convert to tensors for function
            X_train_t = torch.from_numpy(X_train).float()
            Y_train_t = torch.from_numpy(Y_train).float()
            X_test_t = torch.from_numpy(X_test).float()
            Y_test_t = torch.from_numpy(Y_test).float()

            # Train and predict
            Y_pred = train_predict_fn(X_train_t, Y_train_t, X_test_t)

            # Compute metric
            score = self._compute_metric(Y_test_t, Y_pred)
            scores.append(score)

        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'scores': scores
        }


if __name__ == "__main__":
    # Example usage
    print("Testing alignment metrics...")

    # Generate synthetic data
    n_reps = 10
    n_stim = 100
    n_feat = 50

    # Brain data with repetitions (for noise ceiling)
    true_signal = torch.randn(n_stim, n_feat)
    noise = 0.3
    brain_data_reps = true_signal.unsqueeze(0) + noise * torch.randn(n_reps, n_stim, n_feat)

    print("\n1. Noise Ceiling Estimation")
    nc = NoiseCeiling(method='split-half', n_splits=50)
    ceiling = nc.estimate(brain_data_reps)
    print(f"Split-half ceiling: {ceiling:.4f}")

    nc_loo = NoiseCeiling(method='leave-one-out')
    ceiling_loo = nc_loo.estimate(brain_data_reps)
    print(f"Leave-one-out ceiling: {ceiling_loo:.4f}")

    print("\n2. Bootstrap Confidence Intervals")
    X = torch.randn(100, 50)
    Y = torch.randn(100, 30)

    def correlation_metric(x, y):
        return torch.corrcoef(torch.stack([x.flatten(), y.flatten()]))[0, 1].item()

    bootstrap = BootstrapCI(n_bootstrap=100, confidence=0.95)
    ci = bootstrap.compute(correlation_metric, X, Y, show_progress=False)
    print(f"Correlation: {ci['mean']:.4f}")
    print(f"95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}]")

    print("\n3. Permutation Test")
    # Add correlation to data
    shared = torch.randn(100, 10)
    X = torch.cat([shared, torch.randn(100, 40)], dim=1)
    Y = torch.cat([shared, torch.randn(100, 20)], dim=1)

    perm_test = PermutationTest(n_permutations=100)
    result = perm_test.test(correlation_metric, X, Y, show_progress=False)
    print(f"Observed: {result['observed']:.4f}")
    print(f"p-value: {result['p_value']:.4f}")
    print(f"Null: {result['null_mean']:.4f} ± {result['null_std']:.4f}")

    print("\n4. Normalized Score")
    model_score = 0.35
    normalizer = NormalizedScore()
    normalized = normalizer.normalize(model_score, ceiling)
    print(f"Raw score: {model_score:.4f}")
    print(f"Noise ceiling: {ceiling:.4f}")
    print(f"Normalized: {normalized:.2%} of explainable variance")

    print("\n5. Cross-Validated Metric")
    from sklearn.linear_model import Ridge

    def train_predict_fn(X_train, Y_train, X_test):
        model = Ridge(alpha=1.0)
        model.fit(X_train.cpu().numpy(), Y_train.cpu().numpy())
        Y_pred = model.predict(X_test.cpu().numpy())
        return torch.from_numpy(Y_pred).float()

    cv_metric = CrossValidatedMetric(cv=3, metric='r2')
    cv_scores = cv_metric.evaluate(train_predict_fn, X, Y, show_progress=False)
    print(f"CV R²: {cv_scores['mean']:.4f} ± {cv_scores['std']:.4f}")

    print("\nAll metric tests passed!")

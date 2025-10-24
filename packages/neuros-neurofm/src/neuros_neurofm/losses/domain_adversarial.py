"""
Domain Adversarial Loss for Cross-Species Alignment

Implements domain adversarial training to learn species/domain-invariant features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainAdversarialLoss(nn.Module):
    """
    Domain adversarial loss for learning domain-invariant features.

    The feature extractor (backbone) tries to fool the domain discriminator,
    while the discriminator tries to correctly classify the domain/species.

    This encourages the backbone to learn features that are useful across domains.

    Args:
        reduction: Loss reduction method
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def forward(
        self,
        domain_logits: torch.Tensor,
        domain_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute domain adversarial loss.

        Args:
            domain_logits: (batch, n_domains) predicted domain logits
            domain_labels: (batch,) true domain labels

        Returns:
            loss: Domain classification loss
        """
        return self.criterion(domain_logits, domain_labels)


class DomainConfusionLoss(nn.Module):
    """
    Alternative: Domain confusion loss.

    Maximizes entropy of domain predictions to make them uniform.
    This encourages domain-invariant features without explicit labels.
    """

    def __init__(self):
        super().__init__()

    def forward(self, domain_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            domain_logits: (batch, n_domains) domain predictions

        Returns:
            loss: Negative entropy (to maximize entropy via minimization)
        """
        # Softmax probabilities
        probs = F.softmax(domain_logits, dim=-1)

        # Entropy: -sum(p * log(p))
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

        # Return negative entropy (we minimize, so this maximizes entropy)
        return -entropy


class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) loss for domain adaptation.

    Measures distance between feature distributions of different domains.
    Minimizing MMD aligns the domains in feature space.

    Args:
        kernel: Kernel type ('rbf', 'linear')
        bandwidth: Bandwidth for RBF kernel
    """

    def __init__(self, kernel: str = 'rbf', bandwidth: float = 1.0):
        super().__init__()
        self.kernel = kernel
        self.bandwidth = bandwidth

    def _rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute RBF (Gaussian) kernel between x and y.

        Args:
            x: (n, dim)
            y: (m, dim)

        Returns:
            K: (n, m) kernel matrix
        """
        # Compute pairwise squared distances
        x_norm = (x ** 2).sum(dim=1, keepdim=True)  # (n, 1)
        y_norm = (y ** 2).sum(dim=1, keepdim=True)  # (m, 1)

        dist_sq = x_norm + y_norm.T - 2.0 * torch.matmul(x, y.T)  # (n, m)

        # RBF kernel
        K = torch.exp(-dist_sq / (2 * self.bandwidth ** 2))

        return K

    def _linear_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Linear kernel (dot product)."""
        return torch.matmul(x, y.T)

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MMD between source and target domain features.

        Args:
            source_features: (n_source, dim)
            target_features: (n_target, dim)

        Returns:
            mmd: MMD loss value
        """
        if self.kernel == 'rbf':
            kernel_fn = self._rbf_kernel
        else:
            kernel_fn = self._linear_kernel

        # Compute kernel matrices
        K_ss = kernel_fn(source_features, source_features)
        K_tt = kernel_fn(target_features, target_features)
        K_st = kernel_fn(source_features, target_features)

        # MMD^2 = E[K(s,s)] + E[K(t,t)] - 2*E[K(s,t)]
        n_source = source_features.shape[0]
        n_target = target_features.shape[0]

        mmd_sq = (
            K_ss.sum() / (n_source * n_source) +
            K_tt.sum() / (n_target * n_target) -
            2 * K_st.sum() / (n_source * n_target)
        )

        return mmd_sq


# Example usage
if __name__ == '__main__':
    batch_size = 16
    n_domains = 3  # mouse, monkey, human
    feature_dim = 128

    # Domain adversarial loss
    domain_logits = torch.randn(batch_size, n_domains)
    domain_labels = torch.randint(0, n_domains, (batch_size,))

    da_loss = DomainAdversarialLoss()
    loss = da_loss(domain_logits, domain_labels)
    print(f"Domain adversarial loss: {loss.item():.4f}")

    # Domain confusion loss
    dc_loss = DomainConfusionLoss()
    confusion_loss = dc_loss(domain_logits)
    print(f"Domain confusion loss: {confusion_loss.item():.4f}")

    # MMD loss
    source_features = torch.randn(32, feature_dim)
    target_features = torch.randn(24, feature_dim)

    mmd_loss = MMDLoss(kernel='rbf', bandwidth=1.0)
    mmd = mmd_loss(source_features, target_features)
    print(f"MMD loss: {mmd.item():.4f}")

"""
Multi-Task Learning Loss Functions

Implements:
- Uncertainty-weighted multi-task loss
- Dynamic loss balancing
- Adaptive task weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class UncertaintyWeightedLoss(nn.Module):
    """
    Multi-task learning with learned uncertainty weights.

    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    (Kendall et al., 2018)

    Each task's loss is weighted by learned log-variance parameters.
    This automatically balances tasks based on their inherent uncertainty.

    Loss = sum_i [ (1 / (2*sigma_i^2)) * L_i + log(sigma_i) ]

    Args:
        n_tasks: Number of tasks
        init_log_vars: Initial log variance values (optional)
    """

    def __init__(self, n_tasks: int, init_log_vars: Optional[List[float]] = None):
        super().__init__()

        if init_log_vars is None:
            init_log_vars = [0.0] * n_tasks  # sigma = 1

        self.log_vars = nn.Parameter(
            torch.tensor(init_log_vars, dtype=torch.float32)
        )

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute uncertainty-weighted loss.

        Args:
            losses: Dict mapping task names to loss values

        Returns:
            weighted_loss: Combined weighted loss
        """
        task_names = sorted(losses.keys())  # Consistent ordering

        weighted_loss = 0.0

        for i, task_name in enumerate(task_names):
            if i >= len(self.log_vars):
                # Fallback: unit weight
                weighted_loss += losses[task_name]
            else:
                # Uncertainty weighting
                precision = torch.exp(-self.log_vars[i])
                weighted_loss += precision * losses[task_name] + self.log_vars[i]

        return weighted_loss

    def get_weights(self) -> Dict[str, float]:
        """Get current task weights (1 / sigma^2)."""
        weights = {}
        for i in range(len(self.log_vars)):
            weights[f"task_{i}"] = torch.exp(-self.log_vars[i]).item()
        return weights


class GradNormLoss(nn.Module):
    """
    GradNorm: Gradient Normalization for Adaptive Loss Balancing.

    Dynamically adjusts task weights to balance gradient magnitudes across tasks.

    From "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks"
    (Chen et al., 2018)

    Args:
        n_tasks: Number of tasks
        alpha: Restoring force strength (higher = more aggressive balancing)
    """

    def __init__(self, n_tasks: int, alpha: float = 1.5):
        super().__init__()

        self.n_tasks = n_tasks
        self.alpha = alpha

        # Task weights (learnable)
        self.weights = nn.Parameter(torch.ones(n_tasks))

        # Initial task losses (for computing relative inverse training rate)
        self.register_buffer('initial_losses', torch.zeros(n_tasks))
        self.initialized = False

    def forward(
        self,
        losses: List[torch.Tensor],
        shared_params: nn.Parameter
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GradNorm loss.

        Args:
            losses: List of task losses
            shared_params: A shared parameter (e.g., last layer of shared encoder)
                          Used to compute gradient norms

        Returns:
            weighted_loss: Task-weighted loss
            gradnorm_loss: GradNorm balancing loss
        """
        # Initialize on first call
        if not self.initialized:
            with torch.no_grad():
                self.initial_losses.copy_(torch.tensor([l.item() for l in losses]))
            self.initialized = True

        # Compute weighted task loss
        weighted_losses = [w * l for w, l in zip(self.weights, losses)]
        total_loss = sum(weighted_losses)

        # Compute gradient norms for each task
        grad_norms = []
        for i, loss in enumerate(losses):
            # Compute gradient of task loss w.r.t shared params
            grad = torch.autograd.grad(
                loss, shared_params,
                retain_graph=True, create_graph=True
            )[0]

            grad_norm = torch.norm(grad)
            grad_norms.append(grad_norm)

        grad_norms = torch.stack(grad_norms)

        # Average gradient norm
        mean_grad = grad_norms.mean()

        # Relative inverse training rate
        current_losses = torch.tensor([l.item() for l in losses], device=self.weights.device)
        loss_ratios = current_losses / (self.initial_losses + 1e-8)
        inverse_train_rates = loss_ratios / (loss_ratios.mean() + 1e-8)

        # Target gradient norms
        target_grads = mean_grad * (inverse_train_rates ** self.alpha)

        # GradNorm loss: L1 distance between actual and target grad norms
        gradnorm_loss = F.l1_loss(grad_norms, target_grads.detach())

        return total_loss, gradnorm_loss


class MultiTaskLossManager:
    """
    Unified manager for multi-task loss computation.

    Handles:
    - Multiple task losses
    - Automatic balancing (uncertainty weighting, GradNorm, or manual)
    - Loss logging and monitoring

    Args:
        task_names: List of task names
        balancing_method: 'uncertainty', 'gradnorm', 'manual', 'equal'
        manual_weights: Dict of manual weights (if using 'manual')
        **kwargs: Additional arguments for balancing methods
    """

    def __init__(
        self,
        task_names: List[str],
        balancing_method: str = 'uncertainty',
        manual_weights: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        self.task_names = task_names
        self.balancing_method = balancing_method

        if balancing_method == 'uncertainty':
            self.balancer = UncertaintyWeightedLoss(len(task_names))

        elif balancing_method == 'gradnorm':
            self.balancer = GradNormLoss(len(task_names), alpha=kwargs.get('alpha', 1.5))

        elif balancing_method == 'manual':
            if manual_weights is None:
                manual_weights = {name: 1.0 for name in task_names}
            self.manual_weights = manual_weights

        elif balancing_method == 'equal':
            self.manual_weights = {name: 1.0 for name in task_names}

        else:
            raise ValueError(f"Unknown balancing method: {balancing_method}")

    def compute_loss(
        self,
        task_losses: Dict[str, torch.Tensor],
        shared_params: Optional[nn.Parameter] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total multi-task loss.

        Args:
            task_losses: Dict mapping task names to loss tensors
            shared_params: Shared parameter for GradNorm (if using)

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual and weighted losses
        """
        if self.balancing_method == 'uncertainty':
            total_loss = self.balancer(task_losses)

            # Log individual losses
            loss_dict = {f"{k}_unweighted": v.item() for k, v in task_losses.items()}
            loss_dict['total'] = total_loss.item()

            # Log weights
            weights = self.balancer.get_weights()
            loss_dict.update({f"weight_{k}": v for k, v in weights.items()})

        elif self.balancing_method == 'gradnorm':
            # Convert dict to list (consistent ordering)
            losses_list = [task_losses[name] for name in self.task_names]

            total_loss, gradnorm_loss = self.balancer(losses_list, shared_params)

            loss_dict = {f"{k}_unweighted": v.item() for k, v in task_losses.items()}
            loss_dict['total'] = total_loss.item()
            loss_dict['gradnorm_loss'] = gradnorm_loss.item()

            # Add GradNorm loss to total
            total_loss = total_loss + gradnorm_loss

        else:  # manual or equal
            total_loss = 0.0
            loss_dict = {}

            for name, loss in task_losses.items():
                weight = self.manual_weights.get(name, 1.0)
                weighted_loss = weight * loss

                total_loss += weighted_loss

                loss_dict[f"{name}_unweighted"] = loss.item()
                loss_dict[f"{name}_weighted"] = weighted_loss.item()

            loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict

    def state_dict(self):
        """Get state dict for saving."""
        if hasattr(self.balancer, 'state_dict'):
            return self.balancer.state_dict()
        return {}

    def load_state_dict(self, state_dict):
        """Load state dict."""
        if hasattr(self.balancer, 'load_state_dict'):
            self.balancer.load_state_dict(state_dict)


# Example usage
if __name__ == '__main__':
    # Test uncertainty weighting
    task_losses = {
        'decoder': torch.tensor(0.5),
        'encoder': torch.tensor(0.3),
        'contrastive': torch.tensor(0.8)
    }

    uw_loss = UncertaintyWeightedLoss(n_tasks=3)
    total = uw_loss(task_losses)
    print(f"Uncertainty-weighted loss: {total.item():.4f}")
    print(f"Weights: {uw_loss.get_weights()}")

    # Test multi-task manager
    manager = MultiTaskLossManager(
        task_names=['decoder', 'encoder', 'contrastive'],
        balancing_method='uncertainty'
    )

    total_loss, loss_dict = manager.compute_loss(task_losses)
    print(f"\nTotal loss: {total_loss.item():.4f}")
    print("Loss dict:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")

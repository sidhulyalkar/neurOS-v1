"""
Few-Shot Learning and Meta-Learning for NeuroFMx

Rapid adaptation to new tasks/datasets with minimal data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import OrderedDict


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Networks for few-shot learning.

    Learn to classify based on distance to class prototypes in embedding space.
    """

    def __init__(
        self,
        encoder: nn.Module,
        distance_metric: str = 'euclidean'
    ):
        """
        Args:
            encoder: Backbone encoder (frozen NeuroFMx)
            distance_metric: 'euclidean' or 'cosine'
        """
        super().__init__()

        self.encoder = encoder
        self.distance_metric = distance_metric

    def compute_prototypes(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute class prototypes as mean of support examples.

        Args:
            support_embeddings: (n_support, d_model)
            support_labels: (n_support,)

        Returns:
            prototypes: (n_classes, d_model)
        """
        n_classes = len(support_labels.unique())
        d_model = support_embeddings.shape[1]

        prototypes = torch.zeros(n_classes, d_model, device=support_embeddings.device)

        for c in range(n_classes):
            mask = support_labels == c
            prototypes[c] = support_embeddings[mask].mean(dim=0)

        return prototypes

    def compute_distances(
        self,
        query_embeddings: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distances between queries and prototypes.

        Args:
            query_embeddings: (n_query, d_model)
            prototypes: (n_classes, d_model)

        Returns:
            distances: (n_query, n_classes)
        """
        if self.distance_metric == 'euclidean':
            # Squared Euclidean distance
            distances = torch.cdist(query_embeddings, prototypes, p=2) ** 2

        elif self.distance_metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            query_norm = F.normalize(query_embeddings, dim=1)
            proto_norm = F.normalize(prototypes, dim=1)
            similarities = query_norm @ proto_norm.t()
            distances = 1 - similarities

        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        return distances

    def forward(
        self,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        query_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Few-shot classification.

        Args:
            support_data: Support set inputs
            support_labels: Support set labels
            query_data: Query set inputs

        Returns:
            logits: (n_query, n_classes) classification logits
            prototypes: (n_classes, d_model) class prototypes
        """
        # Encode support and query sets
        with torch.no_grad():
            support_embeddings = self.encoder(support_data)
            query_embeddings = self.encoder(query_data)

        # Compute prototypes
        prototypes = self.compute_prototypes(support_embeddings, support_labels)

        # Compute distances (negative for logits)
        distances = self.compute_distances(query_embeddings, prototypes)
        logits = -distances

        return logits, prototypes


class MAML(nn.Module):
    """
    Model-Agnostic Meta-Learning (MAML).

    Learn initial parameters that can be quickly adapted to new tasks.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
        first_order: bool = False
    ):
        """
        Args:
            model: Model to meta-learn
            inner_lr: Learning rate for inner loop (task adaptation)
            inner_steps: Number of inner loop gradient steps
            first_order: Use first-order approximation (faster, less accurate)
        """
        super().__init__()

        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order

    def inner_loop(
        self,
        task_data: Dict[str, torch.Tensor],
        params: Optional[OrderedDict] = None
    ) -> OrderedDict:
        """
        Adapt to a single task using gradient descent.

        Args:
            task_data: Dict with 'support_x', 'support_y'
            params: Current model parameters (None = use model.parameters())

        Returns:
            adapted_params: Parameters after adaptation
        """
        if params is None:
            params = OrderedDict(self.model.named_parameters())

        support_x = task_data['support_x']
        support_y = task_data['support_y']

        # Inner loop: adapt to support set
        for step in range(self.inner_steps):
            # Forward pass with current params
            outputs = self._forward_with_params(support_x, params)
            loss = F.cross_entropy(outputs, support_y)

            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                params.values(),
                create_graph=not self.first_order
            )

            # Update params
            params = OrderedDict([
                (name, param - self.inner_lr * grad)
                for (name, param), grad in zip(params.items(), grads)
            ])

        return params

    def _forward_with_params(
        self,
        x: torch.Tensor,
        params: OrderedDict
    ) -> torch.Tensor:
        """
        Forward pass using custom parameters.

        This is simplified - in practice you'd need to handle
        the full model architecture with functional API.
        """
        # This is a placeholder - actual implementation depends on model structure
        return self.model(x)

    def forward(
        self,
        task_batch: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Meta-training step across multiple tasks.

        Args:
            task_batch: List of tasks, each with 'support_x', 'support_y', 'query_x', 'query_y'

        Returns:
            meta_loss: Loss on query sets after adaptation
            accuracy: Meta-test accuracy
        """
        meta_losses = []
        meta_accs = []

        for task in task_batch:
            # Adapt to support set
            adapted_params = self.inner_loop(task)

            # Evaluate on query set
            query_x = task['query_x']
            query_y = task['query_y']

            outputs = self._forward_with_params(query_x, adapted_params)
            loss = F.cross_entropy(outputs, query_y)

            meta_losses.append(loss)

            # Compute accuracy
            preds = outputs.argmax(dim=1)
            acc = (preds == query_y).float().mean()
            meta_accs.append(acc)

        # Average across tasks
        meta_loss = torch.stack(meta_losses).mean()
        meta_acc = torch.stack(meta_accs).mean()

        return meta_loss, meta_acc


class TransferAdapter(nn.Module):
    """
    Lightweight adapter for transfer learning.

    Freeze the backbone and only train a small adapter network.
    """

    def __init__(
        self,
        backbone: nn.Module,
        d_model: int,
        n_classes: int,
        adapter_dim: int = 64,
        freeze_backbone: bool = True
    ):
        """
        Args:
            backbone: Pretrained NeuroFMx encoder
            d_model: Backbone output dimension
            n_classes: Number of classes for new task
            adapter_dim: Adapter hidden dimension
            freeze_backbone: Freeze backbone weights
        """
        super().__init__()

        self.backbone = backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Adapter network
        self.adapter = nn.Sequential(
            nn.Linear(d_model, adapter_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(adapter_dim, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input data

        Returns:
            logits: (batch, n_classes) classification logits
        """
        # Extract features with frozen backbone
        with torch.no_grad():
            features = self.backbone(x)

        # Pool over sequence if needed
        if features.ndim == 3:
            features = features.mean(dim=1)

        # Adapter classification
        logits = self.adapter(features)

        return logits


class FewShotDataset:
    """
    Dataset generator for few-shot learning episodes.

    Creates N-way K-shot tasks.
    """

    def __init__(
        self,
        data: Dict[int, List[torch.Tensor]],
        n_way: int = 5,
        k_shot: int = 5,
        q_query: int = 15
    ):
        """
        Args:
            data: Dict mapping class_id -> list of examples
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            q_query: Number of query examples per class
        """
        self.data = data
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

        self.classes = list(data.keys())

    def sample_episode(self) -> Dict[str, torch.Tensor]:
        """
        Sample a single N-way K-shot episode.

        Returns:
            episode: Dict with 'support_x', 'support_y', 'query_x', 'query_y'
        """
        # Sample N classes
        selected_classes = np.random.choice(self.classes, size=self.n_way, replace=False)

        support_x = []
        support_y = []
        query_x = []
        query_y = []

        for i, class_id in enumerate(selected_classes):
            # Sample K+Q examples from this class
            class_examples = self.data[class_id]
            selected_indices = np.random.choice(
                len(class_examples),
                size=self.k_shot + self.q_query,
                replace=False
            )

            # Split into support and query
            support_indices = selected_indices[:self.k_shot]
            query_indices = selected_indices[self.k_shot:]

            for idx in support_indices:
                support_x.append(class_examples[idx])
                support_y.append(i)  # Use episode-specific label

            for idx in query_indices:
                query_x.append(class_examples[idx])
                query_y.append(i)

        # Stack into tensors
        episode = {
            'support_x': torch.stack(support_x),
            'support_y': torch.tensor(support_y),
            'query_x': torch.stack(query_x),
            'query_y': torch.tensor(query_y)
        }

        return episode

    def generate_episodes(self, n_episodes: int) -> List[Dict[str, torch.Tensor]]:
        """Generate multiple episodes."""
        return [self.sample_episode() for _ in range(n_episodes)]


def evaluate_few_shot(
    model: nn.Module,
    test_dataset: FewShotDataset,
    n_episodes: int = 100,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Evaluate few-shot learning performance.

    Args:
        model: Prototypical network or MAML
        test_dataset: Few-shot dataset
        n_episodes: Number of test episodes
        device: Device to run on

    Returns:
        metrics: Dict with accuracy and confidence intervals
    """
    model.eval()
    accuracies = []

    with torch.no_grad():
        for _ in range(n_episodes):
            episode = test_dataset.sample_episode()

            # Move to device
            support_x = episode['support_x'].to(device)
            support_y = episode['support_y'].to(device)
            query_x = episode['query_x'].to(device)
            query_y = episode['query_y'].to(device)

            if isinstance(model, PrototypicalNetwork):
                logits, _ = model(support_x, support_y, query_x)
                preds = logits.argmax(dim=1)

            elif isinstance(model, MAML):
                task = {
                    'support_x': support_x,
                    'support_y': support_y,
                    'query_x': query_x,
                    'query_y': query_y
                }
                adapted_params = model.inner_loop(task)
                outputs = model._forward_with_params(query_x, adapted_params)
                preds = outputs.argmax(dim=1)

            else:
                raise ValueError(f"Unknown model type: {type(model)}")

            acc = (preds == query_y).float().mean().item()
            accuracies.append(acc)

    # Compute statistics
    accuracies = np.array(accuracies)
    mean_acc = accuracies.mean()
    std_acc = accuracies.std()
    ci_95 = 1.96 * std_acc / np.sqrt(n_episodes)

    return {
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'ci_95': ci_95,
        'accuracies': accuracies
    }


class MetaLearningTrainer:
    """
    Trainer for meta-learning algorithms.
    """

    def __init__(
        self,
        model: nn.Module,
        meta_lr: float = 0.001,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)

    def train_epoch(
        self,
        train_dataset: FewShotDataset,
        n_tasks: int = 100
    ) -> Dict[str, float]:
        """
        Train for one epoch (multiple meta-training tasks).

        Args:
            train_dataset: Few-shot dataset
            n_tasks: Number of tasks per epoch

        Returns:
            metrics: Training metrics
        """
        self.model.train()
        total_loss = 0
        total_acc = 0

        for i in range(n_tasks):
            # Sample task
            episode = train_dataset.sample_episode()

            # Move to device
            support_x = episode['support_x'].to(self.device)
            support_y = episode['support_y'].to(self.device)
            query_x = episode['query_x'].to(self.device)
            query_y = episode['query_y'].to(self.device)

            # Forward pass
            if isinstance(self.model, PrototypicalNetwork):
                logits, _ = self.model(support_x, support_y, query_x)
                loss = F.cross_entropy(logits, query_y)
                preds = logits.argmax(dim=1)

            elif isinstance(self.model, MAML):
                task = {
                    'support_x': support_x,
                    'support_y': support_y,
                    'query_x': query_x,
                    'query_y': query_y
                }
                loss, acc = self.model([task])

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if isinstance(self.model, PrototypicalNetwork):
                acc = (preds == query_y).float().mean().item()

            total_acc += acc

        return {
            'loss': total_loss / n_tasks,
            'accuracy': total_acc / n_tasks
        }

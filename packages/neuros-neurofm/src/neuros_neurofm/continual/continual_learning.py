"""
Continual Learning Scaffold for NeuroFMx

Online learning with catastrophic forgetting prevention.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from collections import deque
import numpy as np


class ExperienceReplayBuffer:
    """
    Store past examples for replay to prevent catastrophic forgetting.
    """

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, example: Dict):
        """Add example to buffer."""
        self.buffer.append(example)

    def sample(self, batch_size: int) -> List[Dict]:
        """Sample random batch."""
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


class ContinualLearner:
    """
    Scaffold for continual learning.

    Supports:
    - Experience replay
    - Elastic Weight Consolidation (EWC)
    - Progressive neural networks
    - Adapter-based task isolation
    """

    def __init__(
        self,
        model: nn.Module,
        replay_buffer_size: int = 10000,
        replay_ratio: float = 0.5,
        use_ewc: bool = False,
        ewc_lambda: float = 1000.0
    ):
        self.model = model
        self.replay_buffer = ExperienceReplayBuffer(capacity=replay_buffer_size)
        self.replay_ratio = replay_ratio
        self.use_ewc = use_ewc
        self.ewc_lambda = ewc_lambda

        # EWC: Fisher information matrix
        self.fisher_dict = {}
        self.optimal_params = {}

    def train_step(
        self,
        new_batch: Dict,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Single training step with continual learning.

        Args:
            new_batch: New data batch
            optimizer: Optimizer

        Returns:
            losses: Dict of loss components
        """
        # Add new examples to replay buffer
        for i in range(len(new_batch['modality_dict'])):
            example = {k: v[i] for k, v in new_batch['modality_dict'].items()}
            self.replay_buffer.add(example)

        # Sample replay examples
        if len(self.replay_buffer) > 0:
            replay_batch_size = int(len(new_batch) * self.replay_ratio)
            replay_examples = self.replay_buffer.sample(replay_batch_size)

            # Combine new and replay batches
            # (Implementation depends on data format)
            pass

        # Compute loss on combined batch
        outputs = self.model(new_batch['modality_dict'], task='multi-task')

        # Standard task loss
        task_loss = self._compute_task_loss(outputs, new_batch)

        # EWC regularization loss
        ewc_loss = 0
        if self.use_ewc and len(self.fisher_dict) > 0:
            ewc_loss = self._compute_ewc_loss()

        total_loss = task_loss + self.ewc_lambda * ewc_loss

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return {
            'task_loss': task_loss.item(),
            'ewc_loss': ewc_loss.item() if isinstance(ewc_loss, torch.Tensor) else ewc_loss,
            'total_loss': total_loss.item()
        }

    def _compute_task_loss(self, outputs: Dict, batch: Dict) -> torch.Tensor:
        """Compute task-specific loss."""
        # Placeholder - implement based on your tasks
        return torch.tensor(0.0)

    def _compute_ewc_loss(self) -> torch.Tensor:
        """
        Compute EWC regularization loss.

        L_EWC = sum_i F_i * (theta_i - theta*_i)^2
        """
        ewc_loss = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher_dict:
                fisher = self.fisher_dict[name]
                optimal = self.optimal_params[name]
                ewc_loss += (fisher * (param - optimal).pow(2)).sum()

        return ewc_loss

    def consolidate_task(self, dataloader: torch.utils.data.DataLoader):
        """
        Consolidate current task by computing Fisher information.

        Call this after finishing training on a task.
        """
        if not self.use_ewc:
            return

        print("Computing Fisher information matrix...")

        # Store current optimal parameters
        for name, param in self.model.named_parameters():
            self.optimal_params[name] = param.data.clone()

        # Compute Fisher information
        self.model.eval()
        fisher_dict = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}

        for batch in dataloader:
            self.model.zero_grad()
            outputs = self.model(batch['modality_dict'], task='multi-task')

            # Use log likelihood for Fisher
            loss = self._compute_task_loss(outputs, batch)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher_dict[name] += param.grad.pow(2)

        # Average over dataset
        n_samples = len(dataloader.dataset)
        for name in fisher_dict:
            fisher_dict[name] /= n_samples

        self.fisher_dict = fisher_dict

        print("Fisher information computed and stored")


def continual_training_loop():
    """
    Placeholder for continual learning training loop.

    This demonstrates the structure - actual implementation
    depends on your specific use case.
    """
    # Initialize
    model = None  # Your model
    learner = ContinualLearner(model, use_ewc=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Task sequence
    tasks = [
        # Task 1: Initial training
        {'name': 'task1', 'dataloader': None},
        # Task 2: New data arrives
        {'name': 'task2', 'dataloader': None},
        # Task 3: Another new task
        {'name': 'task3', 'dataloader': None},
    ]

    for task in tasks:
        print(f"\nTraining on {task['name']}...")

        dataloader = task['dataloader']

        # Train on this task
        for epoch in range(10):
            for batch in dataloader:
                losses = learner.train_step(batch, optimizer)

        # Consolidate task (compute Fisher)
        learner.consolidate_task(dataloader)

        # Evaluate on all previous tasks (check forgetting)
        for prev_task in tasks[:tasks.index(task)]:
            print(f"  Evaluating on {prev_task['name']}...")
            # Compute metrics
            pass


if __name__ == '__main__':
    continual_training_loop()

"""Extended training loop with mixture‑model domain adaptation.

This module defines the :class:`NeuroFMXXTrainer`, which extends the
base :class:`NeuroFMXTrainer` with a three‑phase curriculum and
mixture‑model domain adaptation.  The adaptation strategy is
implemented via a **SourceWeigher** microservice that computes
continuous mixture weights over a set of source domains.  During the
domain‑weighted phase the trainer periodically computes simple
performance moments (mean squared error and pseudo‑accuracy) for each
source on the target validation set, sends them to the microservice
and receives a set of weights.  These weights are then used to
probabilistically sample batches from the source datasets, biasing
training towards sources that are more similar to the target.

For brevity and clarity this example omits many production details
(e.g. distributed training, advanced data augmentation, additional
moment metrics or regularisation).  The code serves as a template to
illustrate how one could integrate adaptive domain weighting into
neurOS training loops.  In practice you may wish to extend
``compute_moments`` to include other metrics (e.g. calibration error,
representational similarity), add entropy regularisation to the
weighting, or update the weights online between epochs.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import requests
import torch
from torch.utils.data import DataLoader, Dataset

from .curriculum import Curriculum, TrainingPhase
from .neurofmx import NeuroFMX
from .trainer import NeuroFMXTrainer, TrainConfig


class NeuroFMXXTrainer(NeuroFMXTrainer):
    """Trainer that implements mixture‑model domain adaptation via SourceWeigher.

    Args:
        model: A :class:`NeuroFMX` instance.
        source_datasets: Mapping from source identifiers to their datasets.
        val_dataset: Dataset used to compute target moments (validation set for the target subject).
        target_dataset: Full dataset for the target subject used in the final phase.
        source_weigher_url: URL of the SourceWeigher service (e.g. ``http://localhost:8000/weigh``).
        config: Base training configuration shared across all phases.
    """

    def __init__(
        self,
        model: NeuroFMX,
        source_datasets: Dict[str, Dataset],
        val_dataset: Dataset,
        target_dataset: Dataset,
        source_weigher_url: str,
        config: Optional[TrainConfig] = None,
    ) -> None:
        # Use an empty dataset for the base trainer; we override training below
        dummy_dataset = list(source_datasets.values())[0]
        super().__init__(model, dummy_dataset, val_dataset, config)
        self.source_datasets = source_datasets
        self.val_dataset = val_dataset
        self.target_dataset = target_dataset
        self.source_weigher_url = source_weigher_url

    # ------------------------------------------------------------------
    # Moment computation
    # ------------------------------------------------------------------
    def _compute_loss_and_accuracy(self, dataset: Dataset) -> Tuple[float, float]:
        """Compute simple regression loss (MSE) and a pseudo‑accuracy metric.

        For demonstration purposes, the accuracy is defined as the
        percentage of predictions within one standard deviation of the
        target.  For classification tasks you should replace this with
        actual accuracy.
        """
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        self.model.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                preds = self.model(x)
                mse = torch.mean((preds - y) ** 2, dim=list(range(1, preds.ndim)))  # mean over feature dims
                total_loss += mse.sum().item()
                total_samples += x.size(0)
                # pseudo‑accuracy: count predictions close to target
                closeness = torch.abs(preds - y) <= y.std()
                total_correct += closeness.all(dim=list(range(1, preds.ndim))).sum().item()
        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)
        return avg_loss, accuracy

    def compute_moments(self) -> Tuple[List[List[float]], List[float], List[str]]:
        """Compute moment vectors for each source and the target.

        Returns a tuple (source_moments, target_moments, source_ids)
        where ``source_moments`` is a list of lists of floats, one per source,
        ``target_moments`` is a list of floats for the target validation set,
        and ``source_ids`` preserves the ordering of the sources.
        """
        source_ids = list(self.source_datasets.keys())
        source_moments: List[List[float]] = []
        for sid in source_ids:
            loss, acc = self._compute_loss_and_accuracy(self.source_datasets[sid])
            source_moments.append([loss, acc])
        tgt_loss, tgt_acc = self._compute_loss_and_accuracy(self.val_dataset)
        target_moments = [tgt_loss, tgt_acc]
        return source_moments, target_moments, source_ids

    # ------------------------------------------------------------------
    # Weight estimation
    # ------------------------------------------------------------------
    def estimate_weights(self, source_moments: List[List[float]], target_moments: List[float]) -> List[float]:
        """Call the SourceWeigher service to estimate mixture weights.

        The JSON payload is of the form:

        ``{"source_moments": [[m11, m12, ...], [m21, m22, ...], ...], "target_moments": [t1, t2, ...]}``

        The service returns a JSON object ``{"weights": [w1, w2, ...]}``.
        """
        payload = {
            "source_moments": source_moments,
            "target_moments": target_moments,
        }
        try:
            resp = requests.post(self.source_weigher_url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data.get("weights", [])
        except Exception as e:
            # In case of failure, fall back to uniform weights
            n = len(source_moments)
            print(f"Warning: failed to obtain weights from SourceWeigher ({e}). Using uniform weights.")
            return [1.0 / n] * n

    # ------------------------------------------------------------------
    # Weighted sampling
    # ------------------------------------------------------------------
    def _weighted_dataloader(self, weights: List[float]) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        """Yield batches sampled from source datasets according to weights.

        This generator repeatedly samples a source according to the
        provided distribution, then draws a random batch from that
        source's DataLoader.  It yields tuples (x, y).
        """
        assert len(weights) == len(self.source_datasets)
        source_ids = list(self.source_datasets.keys())
        loaders = {
            sid: DataLoader(
                self.source_datasets[sid],
                batch_size=self.config.batch_size,
                shuffle=True,
                drop_last=True,
            )
            for sid in source_ids
        }
        # Create infinite iterators per source
        iters = {sid: iter(loader) for sid, loader in loaders.items()}
        while True:
            sid = random.choices(source_ids, weights=weights, k=1)[0]
            try:
                batch = next(iters[sid])
            except StopIteration:
                # restart iterator when exhausted
                iters[sid] = iter(loaders[sid])
                batch = next(iters[sid])
            yield batch

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(
        self,
        num_pretrain_epochs: int,
        num_weighted_epochs: int,
        num_target_epochs: int,
    ) -> None:
        """Run the three‑phase training procedure.

        Args:
            num_pretrain_epochs: Number of epochs for uniform pretraining.
            num_weighted_epochs: Number of epochs for domain‑weighted fine‑tuning.
            num_target_epochs: Number of epochs for target fine‑tuning.
        """
        curriculum = Curriculum(num_pretrain_epochs, num_weighted_epochs, num_target_epochs)
        for phase, n_epochs in curriculum.phases():
            if phase is TrainingPhase.PRETRAIN:
                print(f"Starting pretraining for {n_epochs} epochs...")
                # Combine all source datasets into one
                combined_dataset = torch.utils.data.ConcatDataset(list(self.source_datasets.values()))
                # Replace the train_dataset and run the base trainer
                self.train_dataset = combined_dataset
                self.config.num_epochs = n_epochs
                super().train()
            elif phase is TrainingPhase.DOMAIN_WEIGHTED:
                print(f"Starting domain‑weighted training for {n_epochs} epochs...")
                for epoch in range(n_epochs):
                    # Compute moments and weights at the start of each epoch
                    src_moments, tgt_moments, src_ids = self.compute_moments()
                    weights = self.estimate_weights(src_moments, tgt_moments)
                    print(f"Epoch {epoch + 1}/{n_epochs} weights: {dict(zip(src_ids, weights))}")
                    # Training with weighted sampling
                    self.model.train()
                    total_loss = 0.0
                    total_samples = 0
                    # Determine number of steps from combined dataset size
                    steps_per_epoch = sum(len(ds) for ds in self.source_datasets.values()) // self.config.batch_size
                    batch_iter = self._weighted_dataloader(weights)
                    for _ in range(steps_per_epoch):
                        x, y = next(batch_iter)
                        x = x.to(self.device)
                        y = y.to(self.device)
                        preds = self.model(x)
                        loss = self.criterion(preds, y)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        total_loss += loss.item() * x.size(0)
                        total_samples += x.size(0)
                    avg_loss = total_loss / max(total_samples, 1)
                    print(f"Epoch {epoch + 1}/{n_epochs} - weighted_loss: {avg_loss:.4f}")
            elif phase is TrainingPhase.TARGET_FINE_TUNE:
                print(f"Starting target fine‑tuning for {n_epochs} epochs...")
                self.train_dataset = self.target_dataset
                self.config.num_epochs = n_epochs
                super().train()
            else:
                raise ValueError(f"Unknown training phase: {phase}")
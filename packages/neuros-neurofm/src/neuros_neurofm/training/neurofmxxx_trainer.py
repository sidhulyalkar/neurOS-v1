"""Advanced training loop with class‑conditional mixture‑model domain adaptation.

This module implements a more sophisticated version of the neuroFMx
training loop where mixture weights are estimated **per class** and
used to sample sources conditionally on the class label.  The
approach follows the general method described in the mixture‑model
domain adaptation paper: the target distribution is modelled as a
convex combination of source distributions, but here we refine the
mixture for each class to better handle class‑specific shifts.  The
trainer will compute moment vectors (loss and accuracy) separately
for each class on each source and the target validation set, call
the SourceWeigher microservice for each class, and then sample
training batches by first selecting a class and then sampling a
source according to the class‑specific weights.

This example assumes a classification task with integer labels.  For
regression tasks or more complex labels you could generalise the
moment computation and sampling logic accordingly.

Note: This implementation is intentionally simple: it groups all
samples by their class label in memory, computes two basic moments
(mean squared error of the model's predictions and classification
accuracy), and uses those moments to estimate weights.  It does not
implement online weight updates between batches or additional
regularisation.  You can extend this template to include richer
metrics (e.g., calibration errors, representation similarity), add
entropy regularisation, or perform moment smoothing over time.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import requests
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from .curriculum import Curriculum, TrainingPhase
from .neurofmx import NeuroFMX
from .trainer import NeuroFMXTrainer, TrainConfig


class NeuroFMXXXTrainer(NeuroFMXTrainer):
    """Trainer that performs class‑conditional mixture‑model adaptation.

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
        # Use an arbitrary dataset for the base trainer; training is overridden below
        dummy_dataset = list(source_datasets.values())[0]
        super().__init__(model, dummy_dataset, val_dataset, config)
        self.source_datasets = source_datasets
        self.val_dataset = val_dataset
        self.target_dataset = target_dataset
        self.source_weigher_url = source_weigher_url
        # Precompute class groupings for each source and the target
        self.source_class_groups = self._group_by_class(self.source_datasets)
        self.val_class_groups = self._group_by_class({"val": val_dataset})["val"]
        self.target_class_groups = self._group_by_class({"tgt": target_dataset})["tgt"]

    # ------------------------------------------------------------------
    # Utility: group datasets by class
    # ------------------------------------------------------------------
    def _group_by_class(self, datasets: Dict[str, Dataset]) -> Dict[str, Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """Group each dataset's samples by their integer class label.

        Returns a nested dictionary ``{source_id: {class_id: [(x, y), ...]}}``.
        This operation loads the entire dataset into memory; for very
        large datasets you should implement a streaming approach.
        """
        groups: Dict[str, Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]] = {}
        for sid, ds in datasets.items():
            class_map: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
            loader = DataLoader(ds, batch_size=1, shuffle=False)
            for x, y in loader:
                # Assume y is a scalar class label (convert tensor -> int)
                if isinstance(y, torch.Tensor):
                    cls = int(y.item())
                else:
                    cls = int(y)
                class_map.setdefault(cls, []).append((x.squeeze(0), y.squeeze(0)))
            groups[sid] = class_map
        return groups

    # ------------------------------------------------------------------
    # Moment computation per class
    # ------------------------------------------------------------------
    def _compute_class_moments(
        self,
        class_id: int,
    ) -> Tuple[List[List[float]], List[float], List[str]]:
        """Compute moment vectors for a specific class across all sources and the target.

        Returns a tuple (source_moments, target_moments, source_ids) where
        ``source_moments`` is a list of lists of floats, one per source,
        ``target_moments`` is a list of floats for the target validation set,
        and ``source_ids`` preserves the ordering of the sources.
        """
        source_ids = list(self.source_datasets.keys())
        source_moments: List[List[float]] = []
        # compute moments for each source for this class
        for sid in source_ids:
            samples = self.source_class_groups[sid].get(class_id, [])
            if not samples:
                # if no examples of this class in this source, use inf loss and zero acc
                source_moments.append([float("inf"), 0.0])
                continue
            # build a batch
            xs = torch.stack([x for x, _ in samples]).to(self.device)
            ys = torch.stack([y for _, y in samples]).to(self.device)
            self.model.eval()
            with torch.no_grad():
                preds = self.model(xs)
                mse = torch.mean((preds - ys) ** 2, dim=list(range(1, preds.ndim)))
                total_loss = mse.mean().item()
                # classification accuracy
                pred_labels = preds.argmax(dim=1)
                acc = (pred_labels == ys).float().mean().item()
            source_moments.append([total_loss, acc])
        # compute moments for target validation set
        tgt_samples = self.val_class_groups.get(class_id, [])
        if not tgt_samples:
            tgt_moments = [float("inf"), 0.0]
        else:
            xs = torch.stack([x for x, _ in tgt_samples]).to(self.device)
            ys = torch.stack([y for _, y in tgt_samples]).to(self.device)
            self.model.eval()
            with torch.no_grad():
                preds = self.model(xs)
                mse = torch.mean((preds - ys) ** 2, dim=list(range(1, preds.ndim)))
                total_loss = mse.mean().item()
                pred_labels = preds.argmax(dim=1)
                acc = (pred_labels == ys).float().mean().item()
            tgt_moments = [total_loss, acc]
        return source_moments, tgt_moments, source_ids

    # ------------------------------------------------------------------
    # Weight estimation per class
    # ------------------------------------------------------------------
    def _estimate_weights_per_class(self, class_id: int) -> List[float]:
        """Estimate mixture weights for a specific class.

        This calls the SourceWeigher microservice with moment vectors
        computed by :func:`_compute_class_moments` and returns a list of
        weights (one per source).  If the service call fails it
        returns uniform weights.
        """
        src_moments, tgt_moments, src_ids = self._compute_class_moments(class_id)
        payload = {
            "source_moments": src_moments,
            "target_moments": tgt_moments,
        }
        try:
            resp = requests.post(self.source_weigher_url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            weights = data.get("weights", [])
        except Exception as e:
            n = len(src_moments)
            print(f"Warning: failed to obtain weights for class {class_id} ({e}). Using uniform weights.")
            weights = [1.0 / n] * n
        # normalise weights to sum to 1
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(src_moments)] * len(src_moments)
        return weights

    # ------------------------------------------------------------------
    # Weighted sampling per class
    # ------------------------------------------------------------------
    def _weighted_dataloader_per_class(
        self,
        weights_by_class: Dict[int, List[float]],
    ) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        """Yield batches sampled from class‑specific source groups according to weights.

        This generator repeatedly chooses a class uniformly from those
        present in the target validation set, selects a source with
        probability given by ``weights_by_class[class_id]``, and then
        yields a random batch from that source's data for that class.
        """
        # Determine available classes
        class_ids = list(self.val_class_groups.keys())
        # Create data loaders for each (source, class)
        loaders: Dict[Tuple[str, int], Iterable[Tuple[torch.Tensor, torch.Tensor]]] = {}
        iters: Dict[Tuple[str, int], Iterable[Tuple[torch.Tensor, torch.Tensor]]] = {}
        for sid in self.source_datasets.keys():
            for c in class_ids:
                samples = self.source_class_groups[sid].get(c, [])
                if not samples:
                    continue
                # build a dataset from the samples
                xs = torch.stack([x for x, _ in samples])
                ys = torch.stack([y for _, y in samples])
                dataset = torch.utils.data.TensorDataset(xs, ys)
                loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True)
                loaders[(sid, c)] = loader
                iters[(sid, c)] = iter(loader)
        while True:
            # sample a class uniformly among available classes
            c = random.choice(class_ids)
            # sample a source according to weights for this class
            weights = weights_by_class.get(c)
            source_ids = list(self.source_datasets.keys())
            # handle missing weights by uniform
            if weights is None or len(weights) != len(source_ids):
                weights = [1.0 / len(source_ids)] * len(source_ids)
            sid = random.choices(source_ids, weights=weights, k=1)[0]
            # if no loader for this (sid, c), skip (rare)
            if (sid, c) not in loaders:
                continue
            try:
                batch = next(iters[(sid, c)])
            except StopIteration:
                iters[(sid, c)] = iter(loaders[(sid, c)])
                batch = next(iters[(sid, c)])
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
        """Run the three‑phase training procedure with class‑conditional weighting.

        Args:
            num_pretrain_epochs: Number of epochs for uniform pretraining.
            num_weighted_epochs: Number of epochs for class‑conditional weighted fine‑tuning.
            num_target_epochs: Number of epochs for target fine‑tuning.
        """
        curriculum = Curriculum(num_pretrain_epochs, num_weighted_epochs, num_target_epochs)
        for phase, n_epochs in curriculum.phases():
            if phase is TrainingPhase.PRETRAIN:
                print(f"Starting pretraining for {n_epochs} epochs...")
                combined_dataset = ConcatDataset(list(self.source_datasets.values()))
                self.train_dataset = combined_dataset
                self.config.num_epochs = n_epochs
                super().train()
            elif phase is TrainingPhase.DOMAIN_WEIGHTED:
                print(f"Starting class‑conditional domain‑weighted training for {n_epochs} epochs...")
                for epoch in range(n_epochs):
                    # Estimate weights for each class present in the target validation set
                    weights_by_class: Dict[int, List[float]] = {}
                    for c in self.val_class_groups.keys():
                        weights_by_class[c] = self._estimate_weights_per_class(c)
                    print(f"Epoch {epoch + 1}/{n_epochs} weights by class:")
                    for c, ws in weights_by_class.items():
                        print(f"  class {c}: {{ {', '.join(f'{sid}={w:.3f}' for sid, w in zip(self.source_datasets.keys(), ws))} }}")
                    # Create an infinite generator for weighted sampling
                    batch_iter = self._weighted_dataloader_per_class(weights_by_class)
                    self.model.train()
                    total_loss = 0.0
                    total_samples = 0
                    # compute total number of samples across all source datasets
                    total_samples_sources = sum(len(ds) for ds in self.source_datasets.values())
                    steps_per_epoch = total_samples_sources // self.config.batch_size
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
                    print(f"Epoch {epoch + 1}/{n_epochs} - class‑weighted loss: {avg_loss:.4f}")
            elif phase is TrainingPhase.TARGET_FINE_TUNE:
                print(f"Starting target fine‑tuning for {n_epochs} epochs...")
                self.train_dataset = self.target_dataset
                self.config.num_epochs = n_epochs
                super().train()
            else:
                raise ValueError(f"Unknown training phase: {phase}")
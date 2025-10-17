"""
Composite model for multi‑modal fusion.

This class encapsulates a set of sub‑models, one for each modality or
feature subset, and combines their predictions via a simple voting
scheme.  During training, each sub‑model is trained on its
corresponding slice of the feature matrix.  At prediction time, the
composite model queries each sub‑model and returns the majority vote
of their predictions.  If the number of sub‑models is even and
predictions are tied, the prediction of the first sub‑model is used
as the tie breaker.

The composite model can be extended to incorporate more advanced
fusion strategies such as weighted voting, stacking (training a
meta‑model on sub‑model outputs) or deep architectures that learn to
combine modality embeddings.  For the sake of simplicity, this
implementation demonstrates the basic mechanics of multi‑modal fusion
within the neurOS framework.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from neuros.models.base_model import BaseModel


class CompositeModel(BaseModel):
    """Ensemble of sub‑models, one per modality.

    Parameters
    ----------
    sub_models : sequence of BaseModel
        Models used to make predictions on disjoint feature slices of
        the input.  The length of ``sub_models`` must match the length
        of ``feature_slices``.
    feature_slices : sequence of slice
        Slices specifying the columns of the feature matrix assigned
        to each sub‑model.  Each slice should be disjoint and
        collectively cover the full feature vector.
    """

    def __init__(self, sub_models: Sequence[BaseModel], feature_slices: Sequence[slice]) -> None:
        if len(sub_models) != len(feature_slices):
            raise ValueError("The number of sub_models must equal the number of feature_slices.")
        super().__init__()
        self.sub_models: List[BaseModel] = list(sub_models)
        self.feature_slices: List[slice] = list(feature_slices)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        # train each sub‑model on its slice of features
        for model, sl in zip(self.sub_models, self.feature_slices):
            model.train(X[:, sl], y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("CompositeModel has not been trained.  Call train() first.")
        # gather predictions from each sub‑model
        preds = []
        for model, sl in zip(self.sub_models, self.feature_slices):
            preds.append(model.predict(X[:, sl]))
        # stack predictions: shape (n_models, n_samples)
        pred_array = np.stack(preds, axis=0)
        # majority vote along models axis
        # we assume discrete class labels (integers)
        # compute mode for each sample; ties broken by first sub‑model
        n_models, n_samples = pred_array.shape
        final_preds = np.empty(n_samples, dtype=pred_array.dtype)
        for j in range(n_samples):
            # get j‑th predictions across models
            votes = pred_array[:, j]
            # count occurrences
            values, counts = np.unique(votes, return_counts=True)
            max_count = np.max(counts)
            # find all labels with max count
            tied_labels = values[counts == max_count]
            # if multiple labels tie, choose the vote of the first model
            if len(tied_labels) == 1:
                final_preds[j] = tied_labels[0]
            else:
                final_preds[j] = votes[0]
        return final_preds

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # delegate partial fitting to sub‑models if supported; else fallback to full training
        for model, sl in zip(self.sub_models, self.feature_slices):
            if hasattr(model, "partial_fit"):
                model.partial_fit(X[:, sl], y)
            else:
                model.train(X[:, sl], y)

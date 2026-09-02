"""Train-only sequence-safe predictive representation controls."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .autoencoder import _MLPAutoencoder
from .contracts import FitRegime, RepresentationEmbedding, SequenceBatch
from .pca import _positive_int


def build_lagged_training_pairs(
    batch: SequenceBatch,
    *,
    lag: int = 1,
    shuffle_targets: bool = False,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Build within-sequence x_t -> x_(t+lag) pairs without crossing boundaries.

    When ``shuffle_targets`` is true, successor targets are permuted independently
    inside each sequence. The input/target marginal distributions are preserved,
    but temporal correspondence is destroyed without exchanging samples between
    trajectories.
    """
    lag = _positive_int(lag, name="lag")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    seed = int(seed)

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    pair_sequence_ids: list[str] = []
    rng = np.random.default_rng(seed)

    for sequence_id, sequence in zip(batch.sequence_ids, batch.sequences, strict=True):
        source = np.asarray(sequence)
        if source.shape[0] <= lag:
            raise ValueError(
                f"sequence {sequence_id!r} has {source.shape[0]} timepoints, "
                f"which is not enough for lag={lag}"
            )
        sequence_inputs = np.array(source[:-lag], copy=True)
        sequence_targets = np.array(source[lag:], copy=True)

        if shuffle_targets and sequence_targets.shape[0] > 1:
            indices = rng.permutation(sequence_targets.shape[0])
            if np.array_equal(indices, np.arange(sequence_targets.shape[0])):
                indices = np.roll(indices, 1)
            sequence_targets = sequence_targets[indices]

        inputs.append(sequence_inputs)
        targets.append(sequence_targets)
        pair_sequence_ids.extend([sequence_id] * sequence_inputs.shape[0])

    return (
        np.concatenate(inputs, axis=0),
        np.concatenate(targets, axis=0),
        tuple(pair_sequence_ids),
    )


class LagPredictiveAutoencoderRepresentation:
    """Matched-capacity nonlinear encoder trained on a lagged prediction objective.

    The network architecture, optimizer family, standardization, batching and
    deterministic seed handling intentionally mirror ``AutoencoderRepresentation``.
    Only the training objective changes from x_t -> x_t reconstruction to
    x_t -> x_(t+lag) prediction. Evaluation samples are encoded independently and
    never contribute successor targets to fitting.
    """

    fit_regime = FitRegime.TRAIN_ONLY_INDUCTIVE

    def __init__(
        self,
        n_components: int = 2,
        *,
        hidden_dim: int | None = None,
        epochs: int = 50,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        lag: int = 1,
        shuffle_targets: bool = False,
        seed: int = 0,
        method_id: str = "predictive_autoencoder",
    ) -> None:
        self.n_components = _positive_int(n_components, name="n_components")
        self.hidden_dim = (
            None if hidden_dim is None else _positive_int(hidden_dim, name="hidden_dim")
        )
        self.epochs = _positive_int(epochs, name="epochs")
        self.batch_size = _positive_int(batch_size, name="batch_size")
        self.lag = _positive_int(lag, name="lag")
        if isinstance(learning_rate, bool) or not isinstance(
            learning_rate, (int, float, np.integer, np.floating)
        ):
            raise TypeError("learning_rate must be a finite positive real")
        self.learning_rate = float(learning_rate)
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be a finite positive real")
        if not isinstance(shuffle_targets, (bool, np.bool_)):
            raise TypeError("shuffle_targets must be a boolean")
        self.shuffle_targets = bool(shuffle_targets)
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer")
        self.seed = int(seed)
        if not isinstance(method_id, str) or not method_id.strip():
            raise ValueError("method_id must be a nonblank string")
        self.method_id = method_id

        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.training_loss_: float | None = None
        self.training_pair_count_: int | None = None
        self.model_: _MLPAutoencoder | None = None

    def embed(
        self,
        train: SequenceBatch,
        evaluation: SequenceBatch,
    ) -> RepresentationEmbedding:
        if train.feature_count != evaluation.feature_count:
            raise ValueError("train and evaluation feature dimensions must match")

        train_x = np.asarray(train.concatenate(), dtype=np.float32)
        mean = np.mean(train_x, axis=0, dtype=np.float64).astype(np.float32)
        scale = np.std(train_x, axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1e-6] = 1.0

        lag_inputs, lag_targets, _ = build_lagged_training_pairs(
            train,
            lag=self.lag,
            shuffle_targets=self.shuffle_targets,
            seed=self.seed,
        )
        standardized_inputs = (
            np.asarray(lag_inputs, dtype=np.float32) - mean
        ) / scale
        standardized_targets = (
            np.asarray(lag_targets, dtype=np.float32) - mean
        ) / scale

        hidden_dim = self.hidden_dim
        if hidden_dim is None:
            hidden_dim = max(8, min(128, max(self.n_components * 2, train.feature_count)))

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            model = _MLPAutoencoder(train.feature_count, hidden_dim, self.n_components)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            criterion = torch.nn.MSELoss()
            input_tensor = torch.from_numpy(np.array(standardized_inputs, copy=True))
            target_tensor = torch.from_numpy(np.array(standardized_targets, copy=True))
            dataset = TensorDataset(input_tensor, target_tensor)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed)
            loader = DataLoader(
                dataset,
                batch_size=min(self.batch_size, len(dataset)),
                shuffle=True,
                generator=generator,
                num_workers=0,
                drop_last=False,
            )

            model.train()
            last_loss = 0.0
            for _ in range(self.epochs):
                loss_total = 0.0
                sample_total = 0
                for batch_inputs, batch_targets in loader:
                    optimizer.zero_grad(set_to_none=True)
                    predictions = model(batch_inputs)
                    loss = criterion(predictions, batch_targets)
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            "predictive autoencoder training produced a non-finite loss"
                        )
                    loss.backward()
                    optimizer.step()
                    count = int(batch_inputs.shape[0])
                    loss_total += float(loss.detach()) * count
                    sample_total += count
                last_loss = loss_total / max(sample_total, 1)

            model.eval()
            embedded: list[np.ndarray] = []
            with torch.no_grad():
                for sequence in evaluation.sequences:
                    x = np.asarray(sequence, dtype=np.float32)
                    x = (x - mean) / scale
                    z = model.encoder(torch.from_numpy(np.array(x, copy=True)))
                    embedded.append(z.cpu().numpy())

        mean_copy = np.array(mean, copy=True)
        scale_copy = np.array(scale, copy=True)
        mean_copy.setflags(write=False)
        scale_copy.setflags(write=False)
        self.mean_ = mean_copy
        self.scale_ = scale_copy
        self.training_loss_ = float(last_loss)
        self.training_pair_count_ = int(lag_inputs.shape[0])
        self.model_ = model

        return RepresentationEmbedding(
            method_id=self.method_id,
            sequences=tuple(embedded),
            sequence_ids=evaluation.sequence_ids,
            fit_regime=self.fit_regime,
            metadata={
                "n_components": self.n_components,
                "hidden_dim": hidden_dim,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "lag": self.lag,
                "shuffle_targets": self.shuffle_targets,
                "target_mode": (
                    "within_sequence_shuffled_successor"
                    if self.shuffle_targets
                    else "within_sequence_successor"
                ),
                "seed": self.seed,
                "fit_sample_count": train.sample_count,
                "training_pair_count": self.training_pair_count_,
                "training_sequence_count": len(train.sequences),
                "sequence_boundary_policy": "never_cross",
                "target_specific_fit_observations": 0,
                "training_loss": self.training_loss_,
                "coordinate_frame": "shared_train_fitted_encoder",
                "implementation": "torch_mlp_lag_predictive_autoencoder",
            },
        )

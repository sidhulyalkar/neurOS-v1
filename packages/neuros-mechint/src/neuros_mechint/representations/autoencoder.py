"""Native train-only autoencoder representation baseline."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .contracts import FitRegime, RepresentationEmbedding, SequenceBatch
from .pca import _positive_int


class _MLPAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AutoencoderRepresentation:
    """Small deterministic CPU autoencoder fit only on declared train samples."""

    fit_regime = FitRegime.TRAIN_ONLY_INDUCTIVE

    def __init__(
        self,
        n_components: int = 2,
        *,
        hidden_dim: int | None = None,
        epochs: int = 50,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        seed: int = 0,
        method_id: str = "autoencoder",
    ) -> None:
        self.n_components = _positive_int(n_components, name="n_components")
        self.hidden_dim = (
            None if hidden_dim is None else _positive_int(hidden_dim, name="hidden_dim")
        )
        self.epochs = _positive_int(epochs, name="epochs")
        self.batch_size = _positive_int(batch_size, name="batch_size")
        if isinstance(learning_rate, bool) or not isinstance(
            learning_rate, (int, float, np.integer, np.floating)
        ):
            raise TypeError("learning_rate must be a finite positive real")
        self.learning_rate = float(learning_rate)
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be a finite positive real")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer")
        self.seed = int(seed)
        if not isinstance(method_id, str) or not method_id.strip():
            raise ValueError("method_id must be a nonblank string")
        self.method_id = method_id

        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.training_loss_: float | None = None
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
        standardized = (train_x - mean) / scale

        hidden_dim = self.hidden_dim
        if hidden_dim is None:
            hidden_dim = max(8, min(128, max(self.n_components * 2, train.feature_count)))

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            model = _MLPAutoencoder(train.feature_count, hidden_dim, self.n_components)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()
            tensor = torch.from_numpy(np.array(standardized, copy=True))
            dataset = TensorDataset(tensor)
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
                for (batch,) in loader:
                    optimizer.zero_grad(set_to_none=True)
                    reconstructed = model(batch)
                    loss = criterion(reconstructed, batch)
                    if not torch.isfinite(loss):
                        raise RuntimeError("autoencoder training produced a non-finite loss")
                    loss.backward()
                    optimizer.step()
                    count = int(batch.shape[0])
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
                "seed": self.seed,
                "fit_sample_count": train.sample_count,
                "target_specific_fit_observations": 0,
                "training_loss": self.training_loss_,
                "coordinate_frame": "shared_train_fitted_encoder",
                "implementation": "torch_mlp_autoencoder",
            },
        )

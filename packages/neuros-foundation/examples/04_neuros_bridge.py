"""Attach foundation embeddings to the neurOS BaseModel contract."""

import numpy as np

from neuros.foundation_models import FoundationEmbeddingDecoder


def encoder(x: np.ndarray) -> np.ndarray:
    """Replace with a verified adapter/model encoder."""
    x = np.asarray(x, dtype=float)
    return np.concatenate([x, x**2], axis=1)


rng = np.random.default_rng(0)
x_train = rng.normal(size=(128, 8))
y_train = (x_train[:, 0] + 0.5 * x_train[:, 1] > 0).astype(int)
x_test = rng.normal(size=(16, 8))

model = FoundationEmbeddingDecoder(
    encoder=encoder,
    task="classification",
    model_id="example-foundation-encoder",
)
model.train(x_train, y_train)

# Inherits BaseModel.infer(), so it can serve as a neurOS decoder.
print(model.infer(x_test[:1]))

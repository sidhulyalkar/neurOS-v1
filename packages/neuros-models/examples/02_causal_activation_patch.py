"""Train a tiny decoder and test whether its final representation is causally used."""

import numpy as np
import torch

from neuros.models import EEGConformerModel

rng = np.random.default_rng(3)
X = rng.normal(size=(48, 4, 96)).astype("float32")
y = (X[:, 0, 24:48].mean(axis=1) > 0).astype(int)

model = EEGConformerModel(
    n_channels=4,
    n_classes=2,
    embedding_dim=16,
    n_heads=4,
    n_layers=2,
    pool_length=12,
    pool_stride=6,
    n_epochs=2,
    batch_size=16,
    random_state=3,
)
model.train(X, y)
adapter = model.mechint_adapter()
backend = model.analysis_model()

inputs = torch.as_tensor(X[:8], dtype=torch.float32, device=next(backend.parameters()).device)
clean = adapter.forward(inputs)
cache = adapter.capture_outputs(inputs, ["embedding_norm"])
ablated = adapter.forward_with_replacements(
    inputs,
    {"embedding_norm": torch.zeros_like(cache["embedding_norm"])},
)

print("mean |logit causal effect|:", float((clean - ablated).abs().mean().cpu()))
print("Interpretation: this is an intervention effect, not a semantic label for the representation.")

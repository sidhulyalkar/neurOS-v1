"""Connect task-model representations to the neurOS source-reliability layer."""

import numpy as np

from neuros.models import CNNModel
from neuros_sourceweigher import RepresentationSourceWeigher

rng = np.random.default_rng(11)
X_train = rng.normal(size=(48, 4, 64)).astype("float32")
y_train = (X_train[:, 0].mean(axis=1) > 0).astype(int)
model = CNNModel(
    n_channels=4,
    n_classes=2,
    hidden_channels=16,
    n_blocks=2,
    n_epochs=2,
    batch_size=16,
    random_state=11,
)
model.train(X_train, y_train)

sources = {
    "subject-a": rng.normal(0.0, 1.0, size=(24, 4, 64)).astype("float32"),
    "subject-b": rng.normal(0.4, 1.0, size=(24, 4, 64)).astype("float32"),
    "subject-c": rng.normal(-0.6, 1.0, size=(24, 4, 64)).astype("float32"),
}
target = rng.normal(0.35, 1.0, size=(20, 4, 64)).astype("float32")

source_embeddings = {name: model.encode(values) for name, values in sources.items()}
target_embeddings = model.encode(target)
result = RepresentationSourceWeigher().estimate(source_embeddings, target_embeddings)
print(result.by_source())

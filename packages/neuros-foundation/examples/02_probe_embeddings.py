"""Compare two embedding spaces under one explicit downstream protocol."""

import numpy as np

from neuros.foundation_models import EvaluationProtocol, benchmark_embeddings

rng = np.random.default_rng(7)
n_train, n_test = 240, 120
y_train = rng.integers(0, 2, n_train)
y_test = rng.integers(0, 2, n_test)

model_a_train = rng.normal(size=(n_train, 32))
model_a_test = rng.normal(size=(n_test, 32))
model_a_train[:, 0] += 1.8 * y_train
model_a_test[:, 0] += 1.8 * y_test

model_b_train = rng.normal(size=(n_train, 48))
model_b_test = rng.normal(size=(n_test, 48))
model_b_train[:, 0] += 0.6 * y_train
model_b_test[:, 0] += 0.6 * y_test

protocol = EvaluationProtocol(
    name="example-cross-subject-linear-probe",
    split_unit="subject",
    transfer_regime="linear_probe",
    notes=("Synthetic example only; replace with held-out subjects in real work.",),
)

report = benchmark_embeddings(
    train_embeddings={"model-a": model_a_train, "model-b": model_b_train},
    test_embeddings={"model-a": model_a_test, "model-b": model_b_test},
    train_targets=y_train,
    test_targets=y_test,
    protocol=protocol,
)
print(report.to_json())

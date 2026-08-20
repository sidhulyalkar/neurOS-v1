"""Dependency-light probes for comparing neural foundation-model representations.

These probes operate on embeddings rather than model internals, so EEG, spikes,
calcium, fMRI, and multimodal models can be compared with the same measurement
vocabulary after each upstream adapter performs modality-specific preprocessing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np

ProbeTask = Literal["auto", "classification", "regression"]


def _matrix(values: Any, *, name: str = "embeddings") -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix; got shape {matrix.shape}")
    if matrix.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two samples")
    if matrix.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one feature")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return matrix


def effective_rank(embeddings: Any, *, eps: float = 1e-12) -> float:
    """Entropy-based effective dimensionality of a representation matrix."""
    x = _matrix(embeddings)
    x = x - x.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    energy = singular_values**2
    total = float(energy.sum())
    if total <= eps:
        return 0.0
    probabilities = energy / total
    entropy = -float(np.sum(probabilities * np.log(probabilities + eps)))
    return float(np.exp(entropy))


def mean_pairwise_cosine(embeddings: Any, *, eps: float = 1e-12) -> float:
    """Mean cosine similarity across all distinct sample pairs in O(ND)."""
    x = _matrix(embeddings)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    valid = norms[:, 0] > eps
    if valid.sum() < 2:
        return 0.0
    unit = x[valid] / norms[valid]
    n = unit.shape[0]
    numerator = float(np.dot(unit.sum(axis=0), unit.sum(axis=0)) - n)
    return numerator / float(n * (n - 1))


def linear_cka(x: Any, y: Any, *, eps: float = 1e-12) -> float:
    """Linear centered-kernel alignment between two representation spaces."""
    a = _matrix(x, name="x")
    b = _matrix(y, name="y")
    if a.shape[0] != b.shape[0]:
        raise ValueError("x and y must contain the same number of aligned samples")
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    cross = a.T @ b
    aa = a.T @ a
    bb = b.T @ b
    numerator = float(np.sum(cross * cross))
    denominator = float(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb)))
    if denominator <= eps:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def invariance_score(reference: Any, shifted: Any, *, eps: float = 1e-12) -> float:
    """Mean row-aligned cosine similarity after a domain/session perturbation."""
    a = _matrix(reference, name="reference")
    b = _matrix(shifted, name="shifted")
    if a.shape != b.shape:
        raise ValueError(f"reference and shifted embeddings must match; got {a.shape} vs {b.shape}")
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    valid = denominator > eps
    if not np.any(valid):
        return 0.0
    values = numerator[valid] / denominator[valid]
    return float(np.mean(np.clip(values, -1.0, 1.0)))


def representation_report(embeddings: Any) -> dict[str, float | int | bool]:
    """Return a compact geometry/health report for one embedding matrix."""
    x = _matrix(embeddings)
    norms = np.linalg.norm(x, axis=1)
    centered = x - x.mean(axis=0, keepdims=True)
    return {
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "finite": True,
        "mean_norm": float(norms.mean()),
        "feature_variance": float(np.mean(np.var(centered, axis=0))),
        "effective_rank": effective_rank(x),
        "mean_pairwise_cosine": mean_pairwise_cosine(x),
    }


def _infer_task(y: np.ndarray) -> Literal["classification", "regression"]:
    if y.dtype.kind in {"b", "i", "u", "O", "U", "S"}:
        return "classification"
    unique = np.unique(y)
    threshold = max(20, int(np.sqrt(max(len(y), 1))))
    if y.ndim == 1 and len(unique) <= threshold and np.allclose(unique, np.round(unique)):
        return "classification"
    return "regression"


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    design = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    regularizer = np.eye(design.shape[1], dtype=x.dtype) * alpha
    regularizer[-1, -1] = 0.0
    return np.linalg.pinv(design.T @ design + regularizer) @ design.T @ y


def _ridge_predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    design = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    return design @ weights


def linear_probe(
    train_embeddings: Any,
    train_targets: Any,
    test_embeddings: Any,
    test_targets: Any,
    *,
    task: ProbeTask = "auto",
    alpha: float = 1e-3,
) -> dict[str, Any]:
    """Run a deterministic ridge linear probe without scikit-learn."""
    x_train = _matrix(train_embeddings, name="train_embeddings")
    x_test = _matrix(test_embeddings, name="test_embeddings")
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("train/test embedding dimensions must match")

    y_train = np.asarray(train_targets)
    y_test = np.asarray(test_targets)
    if y_train.ndim == 0 or y_test.ndim == 0:
        raise ValueError("targets must have a sample dimension")
    if len(y_train) != len(x_train) or len(y_test) != len(x_test):
        raise ValueError("target lengths must match their embedding matrices")

    resolved_task = _infer_task(y_train) if task == "auto" else task
    if resolved_task not in {"classification", "regression"}:
        raise ValueError("task must be 'auto', 'classification', or 'regression'")

    if resolved_task == "classification":
        y_train_flat = y_train.reshape(-1)
        y_test_flat = y_test.reshape(-1)
        classes = np.unique(y_train_flat)
        if classes.size < 2:
            raise ValueError("classification probe needs at least two training classes")
        class_to_index = {value: index for index, value in enumerate(classes.tolist())}
        encoded = np.zeros((len(y_train_flat), len(classes)), dtype=np.float64)
        for row, value in enumerate(y_train_flat.tolist()):
            encoded[row, class_to_index[value]] = 1.0
        weights = _ridge_fit(x_train, encoded, alpha)
        scores = _ridge_predict(x_test, weights)
        predictions = classes[np.argmax(scores, axis=1)]
        accuracy = float(np.mean(predictions == y_test_flat))
        return {
            "task": "classification",
            "metric": "accuracy",
            "score": accuracy,
            "n_train": int(len(x_train)),
            "n_test": int(len(x_test)),
            "n_features": int(x_train.shape[1]),
            "n_classes": int(len(classes)),
            "alpha": float(alpha),
        }

    y_train_float = np.asarray(y_train, dtype=np.float64)
    y_test_float = np.asarray(y_test, dtype=np.float64)
    if y_train_float.ndim == 1:
        y_train_float = y_train_float[:, None]
    if y_test_float.ndim == 1:
        y_test_float = y_test_float[:, None]
    weights = _ridge_fit(x_train, y_train_float, alpha)
    predictions = _ridge_predict(x_test, weights)
    residual = float(np.sum((y_test_float - predictions) ** 2))
    baseline = float(np.sum((y_test_float - y_test_float.mean(axis=0, keepdims=True)) ** 2))
    r2 = 0.0 if baseline <= 1e-12 else 1.0 - residual / baseline
    mse = float(np.mean((y_test_float - predictions) ** 2))
    return {
        "task": "regression",
        "metric": "r2",
        "score": float(r2),
        "mse": mse,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "n_features": int(x_train.shape[1]),
        "alpha": float(alpha),
    }


def domain_leakage_probe(
    train_embeddings: Any,
    train_domains: Any,
    test_embeddings: Any,
    test_domains: Any,
    *,
    alpha: float = 1e-3,
) -> dict[str, Any]:
    """Measure how linearly decodable subject/site/device identity remains."""
    result = linear_probe(
        train_embeddings,
        train_domains,
        test_embeddings,
        test_domains,
        task="classification",
        alpha=alpha,
    )
    result["probe"] = "domain_leakage"
    return result


def pairwise_cka(embeddings: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compute a tidy pairwise CKA table for multiple model representations."""
    names = list(embeddings)
    rows: list[dict[str, Any]] = []
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            rows.append(
                {"left": left, "right": right, "linear_cka": linear_cka(embeddings[left], embeddings[right])}
            )
    return rows

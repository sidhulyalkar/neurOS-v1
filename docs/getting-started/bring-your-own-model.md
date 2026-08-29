# Bring Your Own Model

You do **not** need neurOS to train your model.

That is the point of Neural System Qualification (NSQ): your code owns the architecture, optimizer, features, representation, and fitting procedure; neurOS owns the evidence boundary around the comparison.

The smallest useful mental model is:

```text
your model factory
      |
      v
fresh decoder instance
      |
      +-- fit(authorized X, y)
      |
      +-- predict(untouched X_final)
      |
      +-- optional predict_proba(X_final)
      |
      v
neurOS validates observation roles, outputs,
metrics, failure states, provenance, and identity
```

## The contract is intentionally small

A normal supervised decoder needs:

1. an `ExternalDecoderMethodSpec` describing the algorithm/configuration identity;
2. a factory with a `method_spec` property and `create()` method;
3. a fresh decoder exposing `fit(X, y)` and `predict(X)`;
4. `learned_state()` describing how strongly the fitted state can be identified.

If the method declares probability output, also provide:

- `predict_proba(X)`;
- `probability_class_labels()` after fitting, so neurOS does not guess which probability column belongs to which task class.

You do not subclass a neurOS model class.

## Minimal sklearn example

The example below deliberately keeps the external implementation ordinary. Flattening is declared as part of the method identity so it is not hidden benchmark behavior.

```python
from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from neuros.foundation_models.qualification import (
    ExternalDecoderMethodSpec,
    ExternalLearnedState,
)


class MyDecoder:
    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=1000)

    @staticmethod
    def _features(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError("expected X=(sample, channel, time)")
        return X.reshape(len(X), -1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(self._features(X), np.asarray(y))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(self._features(X)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba(self._features(X)), dtype=np.float64)

    def probability_class_labels(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.model.classes_)

    def learned_state(self) -> ExternalLearnedState:
        # Participation does not require pretending arbitrary sklearn state has a
        # qualified tensor/checkpoint serializer. Be explicit when state identity
        # is weaker.
        return ExternalLearnedState(
            state_identity_kind="opaque_unverified",
            metadata={"reason": "example_sklearn_state_serializer_not_qualified"},
        )


@dataclass(frozen=True)
class MyFactory:
    @property
    def method_spec(self) -> ExternalDecoderMethodSpec:
        return ExternalDecoderMethodSpec(
            method_id="example-logistic-regression",
            implementation="sklearn.linear_model.LogisticRegression",
            implementation_version=(
                f"scikit-learn={importlib.metadata.version('scikit-learn')}"
            ),
            input_axes=("sample", "channel", "time"),
            probability_semantics="uncalibrated_probability",
            target_adaptation_mode="none",
            source_reference="your repository, paper, or method URL",
            metadata={
                "feature_transform": "flatten-channel-time",
                "solver": "lbfgs",
                "max_iter": 1000,
            },
        )

    def create(self) -> MyDecoder:
        return MyDecoder()
```

The factory is the key boundary. NSQ calls `create()` again for each authorized calibration point rather than warm-starting the previous fitted state.

## What neurOS controls

Under a frozen qualification study, the runner controls and content-addresses:

- the upstream dataset/revision identity;
- the exact processed data authority;
- source-history observations;
- labeled target calibration observations;
- untouched final-assessment observations;
- the calibration budget;
- preprocessing/calibration authorities declared by the study;
- metric scorecard semantics;
- the method specification SHA;
- the run contract and fitted-state binding;
- success, failure, unavailable, OOM, skipped, and nonconverged attempts.

This is what lets a custom lab model, an sklearn baseline, Braindecode, a foundation model, and eventually ORION enter the **same scientific referee** without sharing training code.

## What your adapter controls

Your adapter owns:

- architecture and feature computation;
- optimizer/training loop;
- deterministic transforms internal to the method;
- fitted preprocessing that is learned only from the authorized fit observations;
- prediction behavior;
- probability behavior, if declared;
- learned-state serialization/hash semantics, when you can provide them honestly.

Anything that affects the method result should be represented in `method_spec` metadata or a separately governed preprocessing/model-lineage authority. Do not hide favorable preprocessing inside an unnamed wrapper.

## Probability output is optional

If your method only emits labels, set:

```python
probability_semantics="unavailable"
```

Then omit `predict_proba()` and probability class order. Balanced accuracy and accuracy can still participate; probability-dependent metrics remain explicitly unavailable rather than being fabricated.

If your method emits probabilities, neurOS validates shape, finite values, `[0, 1]` bounds, row sums, and the fitted class-column order. It will not renormalize malformed output or guess class ordering for you.

## Pretrained and foundation models

A pretrained model adds one hard question: **what data has already influenced the representation?**

Provide model/pretraining lineage where it is known. If overlap with the evaluation corpus cannot be established, the correct verdict is unknown, not deployment-disjoint.

A foundation representation can still participate, but its claim language must respect that lineage uncertainty.

## Unlabeled target adaptation

Do not route target-session observations through an `adapt()` method just because they lack labels.

The study must first freeze a scientifically distinct unlabeled-target observation role. The current v1 longitudinal executor intentionally refuses to manufacture an unlabeled pool from “whatever calibration rows are left.”

This keeps an observation from changing roles as calibration budget changes.

## Before opening a model-submission issue

Be ready to state:

- exact implementation/version;
- input axes, units, channel and sampling assumptions;
- fixed and fitted preprocessing;
- validation/early-stopping/checkpoint-selection behavior;
- probability semantics;
- learned-state identity strength;
- pretraining lineage, if applicable;
- the frozen benchmark/study the method should enter.

Use the **External model / method submission** GitHub issue form. The goal is not to make your implementation neurOS-native. The goal is to make its evidence comparable and auditable.

## Reference implementations

The repository already proves this boundary with maintained upstream methods rather than reimplementations:

- MNE CSP + scikit-learn LDA;
- pyRiemann covariance + tangent-space logistic regression;
- direct upstream Braindecode models.

See [NSQ Runner v1](../NSQ_RUNNER_V1.md) for the complete authority chain and failure semantics.

## Claim boundary

Passing the adapter and NSQ contracts can establish that an external method was evaluated under a specific frozen data, calibration, scoring, and provenance authority.

It does **not** by itself establish physiological mechanism, general superiority, physical hardware validity, closed-loop safety, or clinical benefit.

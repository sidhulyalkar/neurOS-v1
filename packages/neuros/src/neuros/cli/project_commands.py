"""Project scaffolding for the supported neurOS developer on-ramp."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from neuros.errors import ConfigurationError


SUPPORTED_PROJECT_TEMPLATES = ("mock-bci", "nsq-method")

_MOCK_BCI_CONFIG = """schema_version: 1

metadata:
  name: neurOS-starter
  purpose: deterministic local runtime and qualification starter

streams:
  - id: eeg
    source:
      plugin: mock
      options:
        sampling_rate: 250.0
        channels: 8
    transforms:
      - plugin: smoothing
        options:
          window_size: 3

decoder:
  plugin: threshold
  options:
    threshold: 0.0

runtime:
  queue_capacity: 16
  overflow_policy: drop_oldest

sinks: []
monitors: []
"""

_MOCK_PROJECT_README = """# neurOS starter project

This project is deliberately small. It exercises the maintained neurOS runtime,
record/replay path, and software-qualification boundary without downloading a
dataset, training a model, or implying a biological or hardware claim.

## 1. Inspect the environment

```bash
neuros doctor
neuros compatibility
```

## 2. Validate and run

```bash
neuros validate neuros.yaml
neuros run neuros.yaml --duration 2
```

## 3. Produce a reproducible software-evidence bundle

```bash
neuros qualify neuros.yaml --output evidence/qualification --duration 1
neuros reproduce evidence/qualification
```

The qualification bundle is software evidence. It does not establish neural
model efficacy, hardware validity, closed-loop performance, safety, or clinical
benefit.

## Next steps

- Replace the mock source through a maintained BrainFlow, LSL, replay, or dataset
  integration rather than editing the neurOS kernel.
- Add an external decoder through the plugin/qualification interfaces.
- Use NSQ for leakage-controlled comparative neural-system evidence.
- Use ORION only when you need governed neural representations, tokenization, or
  adaptation on top of the same evidence authority.

Project documentation: https://github.com/sidhulyalkar/neurOS-v1
"""

_NSQ_METHOD_PYPROJECT = """[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "neuros-nsq-method-starter"
version = "0.1.0"
description = "External decoder starter for neurOS Neural System Qualification"
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.24",
  "scikit-learn>=1.4",
  "neuros-foundation>=2.1.0",
]

[project.optional-dependencies]
test = ["pytest>=8"]

[tool.setuptools]
py-modules = ["method", "demo"]

[tool.pytest.ini_options]
pythonpath = ["."]
"""

_NSQ_METHOD = '''"""A normal external sklearn decoder participating in neurOS NSQ."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from neuros.foundation_models.qualification import (
    ExternalDecoderMethodSpec,
    ExternalLearnedState,
)


class MyDecoder:
    """Example external model. neurOS does not own this training implementation."""

    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=1000, solver="lbfgs")

    @staticmethod
    def _features(X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 3:
            raise ValueError("expected X=(sample, channel, time)")
        return values.reshape(len(values), -1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(self._features(X), np.asarray(y).astype(str))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(self._features(X))).astype(str)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.model.predict_proba(self._features(X)), dtype=np.float64
        )

    def probability_class_labels(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.model.classes_)

    def learned_state(self) -> ExternalLearnedState:
        """Content-address the fitted sklearn state without pickle serialization."""

        digest = hashlib.sha256()
        for name, raw in (
            ("coef", self.model.coef_),
            ("intercept", self.model.intercept_),
        ):
            array = np.ascontiguousarray(raw)
            descriptor = {
                "name": name,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
            }
            digest.update(
                json.dumps(
                    descriptor,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            digest.update(array.tobytes(order="C"))
        digest.update(
            json.dumps(
                [str(value) for value in self.model.classes_],
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return ExternalLearnedState(
            state_identity_kind="tensor_sha256",
            state_sha256=digest.hexdigest(),
            metadata={
                "serializer": "explicit-numpy-coef-intercept-classes-v1",
                "unsafe_pickle_used": False,
            },
        )


@dataclass(frozen=True)
class MyFactory:
    """NSQ creates a fresh instance for every authorized calibration budget."""

    @property
    def method_spec(self) -> ExternalDecoderMethodSpec:
        return ExternalDecoderMethodSpec(
            method_id="starter-logistic-regression",
            implementation="sklearn.linear_model.LogisticRegression",
            implementation_version=(
                f"scikit-learn={importlib.metadata.version('scikit-learn')}"
            ),
            input_axes=("sample", "channel", "time"),
            probability_semantics="uncalibrated_probability",
            target_adaptation_mode="none",
            source_reference="generated by neurOS init --template nsq-method",
            metadata={
                "feature_transform": "flatten-channel-time",
                "solver": "lbfgs",
                "max_iter": 1000,
            },
        )

    def create(self) -> MyDecoder:
        return MyDecoder()
'''

_NSQ_DEMO = '''"""Synthetic NSQ referee demo for an external model.

This proves adapter and authority plumbing only. Numerical scores from this
synthetic fixture are not evidence of real neural decoding efficacy.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from method import MyFactory
from neuros.foundation_models.longitudinal import (
    chronological_partition,
    make_nested_calibration_split,
)
from neuros.foundation_models.longitudinal_authority import LongitudinalCaseAuthority
from neuros.foundation_models.qualification import QualificationProtocolSpec
from neuros.foundation_models.qualification_runner import (
    DEFAULT_CLASSIFICATION_SCORECARD,
    QualificationExecutionContext,
    run_external_qualification_case,
)
from neuros.foundation_models.real_world import GroupedEvaluationData


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def synthetic_data() -> GroupedEvaluationData:
    n_samples = 48
    labels = np.asarray(["left", "right"] * (n_samples // 2), dtype=str)
    sign = np.where(labels == "right", 1.0, -1.0).astype(np.float32)
    base = np.linspace(-0.15, 0.15, 16, dtype=np.float32).reshape(2, 8)
    X = sign[:, None, None] + base[None, :, :]
    sessions = np.repeat(np.asarray(["s1", "s2", "s3"], dtype=str), 16)
    return GroupedEvaluationData(
        dataset_id="synthetic-nsq-method-starter",
        X=X,
        y=labels,
        groups={
            "subject": np.asarray(["p1"] * n_samples, dtype=str),
            "session": sessions,
            "trial": np.asarray([f"t{index:03d}" for index in range(n_samples)], dtype=str),
        },
    )


def build_authority(data: GroupedEvaluationData) -> LongitudinalCaseAuthority:
    partition = chronological_partition(
        data,
        split_unit="session",
        held_out_value="s3",
        order=("s1", "s2", "s3"),
    )
    split = make_nested_calibration_split(
        partition,
        evaluation_fraction=0.5,
        seed=11,
    )
    return LongitudinalCaseAuthority.from_split(
        split,
        case_id="p1:s3",
        history_policy="prior",
        observed_group_order=("s1", "s2", "s3"),
        case_metadata={"participant": "p1", "fixture": True},
    )


def run_demo() -> dict[str, Any]:
    data = synthetic_data()
    authority = build_authority(data)
    split = authority.restore(data)
    lineage_sha = _sha("synthetic-nsq-method-starter:dataset-lineage:v1")
    protocol = QualificationProtocolSpec(
        protocol_id="synthetic-external-method-referee-v1",
        dataset_id=data.dataset_id,
        dataset_lineage_sha256=lineage_sha,
        task_id="synthetic-left-vs-right",
        independent_unit="participant",
        grouping_hierarchy=("participant", "session", "trial"),
        calibration_budgets_per_class=tuple(
            range(split.max_budget_per_class + 1)
        ),
        metric_scorecard_sha256=DEFAULT_CLASSIFICATION_SCORECARD.sha256,
        protocol_status="frozen",
        metadata={
            "synthetic_fixture": True,
            "numerical_result_interpretable": False,
        },
    )
    factory = MyFactory()
    result = run_external_qualification_case(
        data,
        authority,
        protocol,
        factory,
        execution_context=QualificationExecutionContext(
            observed_dataset_lineage_sha256=lineage_sha,
            preprocessing_authority_sha256s=(
                _sha("synthetic-nsq-method-starter:fixed-preprocessing:v1"),
            ),
            calibration_authority_sha256s=(
                _sha("synthetic-nsq-method-starter:calibration-policy:v1"),
            ),
        ),
    )
    rows = [row.to_dict() for row in result.rows]
    return {
        "schema_version": 1,
        "artifact_kind": "synthetic_external_method_nsq_referee_demo",
        "method_spec": factory.method_spec.to_dict(),
        "method_spec_sha256": factory.method_spec.sha256,
        "protocol_sha256": protocol.sha256,
        "case_authority_sha256": authority.authority_sha256,
        "case_result_sha256": result.sha256,
        "rows": rows,
        "claim_boundary": {
            "synthetic_fixture": True,
            "adapter_contract_exercised": True,
            "observation_authority_exercised": True,
            "calibration_frontier_exercised": True,
            "learned_state_identity_exercised": True,
            "numerical_result_interpretable": False,
            "real_neural_efficacy_claim_permitted": False,
        },
    }


def main() -> None:
    print(json.dumps(run_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
'''

_NSQ_TEST = '''from __future__ import annotations

from demo import run_demo


def test_external_method_runs_through_complete_synthetic_nsq_referee():
    payload = run_demo()
    assert payload["artifact_kind"] == "synthetic_external_method_nsq_referee_demo"
    assert len(payload["method_spec_sha256"]) == 64
    assert len(payload["protocol_sha256"]) == 64
    assert len(payload["case_authority_sha256"]) == 64
    assert len(payload["case_result_sha256"]) == 64

    rows = payload["rows"]
    assert len(rows) >= 2
    assert rows[0]["calibration_per_class"] == 0
    assert all(row["status"] == "success" for row in rows)
    assert all(row["learned_state_addressable"] is True for row in rows)
    assert len({row["run_contract_sha256"] for row in rows}) == len(rows)
    assert len({row["evaluation_indices_sha256"] for row in rows}) == 1

    boundary = payload["claim_boundary"]
    assert boundary["adapter_contract_exercised"] is True
    assert boundary["numerical_result_interpretable"] is False
    assert boundary["real_neural_efficacy_claim_permitted"] is False
'''

_NSQ_README = """# External neurOS NSQ method starter

This project demonstrates the intended ecosystem boundary: **your method owns its
training implementation; neurOS owns the qualification authority around it.**

The generated method is ordinary scikit-learn logistic regression. The synthetic
referee demo freezes source history, target calibration, untouched final
assessment, score semantics, and fitted-state identity before producing one row
per calibration budget.

## Current repository developer preview

From the neurOS repository root, install the research authority stack once:

```bash
python scripts/bootstrap.py --profile research --test-tools
python -m pip install scikit-learn
```

Then generate and enter this project:

```bash
neuros init my-method --template nsq-method
cd my-method
python demo.py
pytest -q
```

When coordinated package publishing is enabled, the intended standalone setup is
`pip install 'neuros[evidence]' scikit-learn pytest` followed by the same demo and
test commands.

## Files

- `method.py` contains only the external decoder and factory contract.
- `demo.py` builds a deterministic synthetic longitudinal case and sends the
  method through the production NSQ referee.
- `test_method.py` checks fresh calibration-frontier rows, stable final-assessment
  identity, content-addressed learned state, and the explicit claim boundary.
- `pyproject.toml` makes the starter a normal Python project rather than a hidden
  neurOS-internal example.

## Replace the example with your model

Keep the boundary small:

1. describe the method in `ExternalDecoderMethodSpec`;
2. return a fresh decoder from `create()`;
3. implement `fit(X, y)` and `predict(X)`;
4. add `predict_proba(X)` plus `probability_class_labels()` only when real
   probability-shaped output exists;
5. make `learned_state()` content-addressable when you can serialize the fitted
   state deterministically and safely, otherwise declare it opaque.

Do not move validation, early stopping, calibration selection, target adaptation,
or favorable preprocessing outside the declared method/authority boundary.

## Claim boundary

The included data are synthetic and deliberately easy. Their numerical scores
are **not** evidence of EEG performance, generalization, calibration reduction,
hardware validity, closed-loop utility, safety, or clinical benefit.

The demo proves only that an external implementation can enter the same
leakage-controlled, failure-preserving NSQ referee used by maintained neurOS
studies without rewriting the model inside neurOS.
"""

_PROJECT_GITIGNORE = """.venv/
__pycache__/
*.py[cod]
.pytest_cache/
evidence/
sessions/
artifacts/
"""


def _template_files(template: str) -> dict[str, str]:
    if template == "mock-bci":
        return {
            "neuros.yaml": _MOCK_BCI_CONFIG,
            "README.md": _MOCK_PROJECT_README,
            ".gitignore": _PROJECT_GITIGNORE,
        }
    if template == "nsq-method":
        return {
            "pyproject.toml": _NSQ_METHOD_PYPROJECT,
            "method.py": _NSQ_METHOD,
            "demo.py": _NSQ_DEMO,
            "test_method.py": _NSQ_TEST,
            "README.md": _NSQ_README,
            ".gitignore": _PROJECT_GITIGNORE,
        }
    raise ConfigurationError(
        f"Unsupported project template {template!r}; "
        f"choose one of {', '.join(SUPPORTED_PROJECT_TEMPLATES)}"
    )


def _next_commands(template: str) -> list[str]:
    if template == "mock-bci":
        return [
            "neuros doctor",
            "neuros validate neuros.yaml",
            "neuros run neuros.yaml --duration 2",
            "neuros qualify neuros.yaml --output evidence/qualification --duration 1",
            "neuros reproduce evidence/qualification",
        ]
    return [
        "python demo.py",
        "pytest -q",
    ]


def _evidence_boundary(template: str) -> str:
    if template == "mock-bci":
        return (
            "starter workflow produces software/runtime evidence only; it does not "
            "qualify neural efficacy, hardware, closed-loop behavior, safety, or clinical benefit"
        )
    return (
        "synthetic NSQ starter exercises adapter, observation-role, calibration-frontier, "
        "metric, failure, and learned-state identity contracts only; its numerical scores "
        "cannot support real neural efficacy or deployment claims"
    )


def init_project(
    destination: str | Path,
    *,
    template: str = "mock-bci",
    force: bool = False,
) -> dict[str, Any]:
    """Create a minimal, runnable neurOS project without deleting user files.

    ``force`` permits replacement of neurOS-managed starter files only. Unrelated
    files in an existing directory are preserved.
    """

    root = Path(destination).expanduser().resolve()
    files = _template_files(template)

    if root.exists() and not root.is_dir():
        raise ConfigurationError(f"Project destination is not a directory: {root}")

    existing_managed = [name for name in files if (root / name).exists()]
    if existing_managed and not force:
        joined = ", ".join(sorted(existing_managed))
        raise ConfigurationError(
            f"Project destination already contains neurOS starter files: {joined}. "
            "Use --force to replace only those managed files."
        )

    root.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    replaced: list[str] = []
    for relative, content in files.items():
        path = root / relative
        existed = path.exists()
        path.write_text(content, encoding="utf-8")
        (replaced if existed else created).append(relative)

    config_path = root / "neuros.yaml"
    return {
        "schema_version": 1,
        "project_root": str(root),
        "template": template,
        "config": str(config_path) if config_path.is_file() else None,
        "created": sorted(created),
        "replaced": sorted(replaced),
        "next_commands": _next_commands(template),
        "evidence_boundary": _evidence_boundary(template),
    }


__all__ = ["SUPPORTED_PROJECT_TEMPLATES", "init_project"]

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one replacement, found {count}")
    target.write_text(text.replace(old, new, 1))


def insert_after_once(path: str, marker: str, addition: str) -> None:
    replace_once(path, marker, marker + addition)


contracts = "packages/neuros-mechint/src/neuros_mechint/representations/contracts.py"

replace_once(
    contracts,
    '''class FitRegime(str, Enum):
    TRAIN_ONLY_INDUCTIVE = "train_only_inductive"
    TRANSDUCTIVE_TARGET_OBSERVED = "transductive_target_observed"
    EXTERNAL_PRETRAINED = "external_pretrained"


class MethodStatus(str, Enum):
''',
    '''class FitRegime(str, Enum):
    TRAIN_ONLY_INDUCTIVE = "train_only_inductive"
    TRANSDUCTIVE_TARGET_OBSERVED = "transductive_target_observed"
    EXTERNAL_PRETRAINED = "external_pretrained"


class EvaluationScope(str, Enum):
    """How one method consumes a declared evaluation batch."""

    BATCH_TRANSFORM = "batch_transform"
    SEQUENCE_LOCAL = "sequence_local"


class MethodStatus(str, Enum):
''',
)

insert_after_once(
    contracts,
    '''class RepresentationUnavailableError(RepresentationError):
    """Optional external representation capability is unavailable."""


''',
    '''def _strict_metric_value(value: Any, *, name: str) -> float:
    """Accept only explicit finite real scientific metric scalars."""

    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a finite real number")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be a finite real number")
    return numeric


''',
)

replace_once(
    contracts,
    '''                else:
                    numeric = float(value)
                    if not np.isfinite(numeric):
                        raise ValueError("metric values must be finite or None")
                    metric_values[key] = numeric
''',
    '''                else:
                    metric_values[key] = _strict_metric_value(
                        value,
                        name=f"metric {key!r}",
                    )
''',
)

replace_once(
    contracts,
    '''class RepresentationMethod(Protocol):
    method_id: str
    fit_regime: FitRegime

    def embed(self, train: SequenceBatch, evaluation: SequenceBatch) -> RepresentationEmbedding:
        ...
''',
    '''class RepresentationMethod(Protocol):
    method_id: str
    fit_regime: FitRegime
    evaluation_scope: EvaluationScope

    def embed(self, train: SequenceBatch, evaluation: SequenceBatch) -> RepresentationEmbedding:
        ...
''',
)

replace_once(
    "packages/neuros-mechint/src/neuros_mechint/representations/pca.py",
    "from .contracts import FitRegime, RepresentationEmbedding, SequenceBatch\n",
    "from .contracts import EvaluationScope, FitRegime, RepresentationEmbedding, SequenceBatch\n",
)
insert_after_once(
    "packages/neuros-mechint/src/neuros_mechint/representations/pca.py",
    "    fit_regime = FitRegime.TRAIN_ONLY_INDUCTIVE\n",
    "    evaluation_scope = EvaluationScope.BATCH_TRANSFORM\n",
)

replace_once(
    "packages/neuros-mechint/src/neuros_mechint/representations/autoencoder.py",
    "from .contracts import FitRegime, RepresentationEmbedding, SequenceBatch\n",
    "from .contracts import EvaluationScope, FitRegime, RepresentationEmbedding, SequenceBatch\n",
)
insert_after_once(
    "packages/neuros-mechint/src/neuros_mechint/representations/autoencoder.py",
    "    fit_regime = FitRegime.TRAIN_ONLY_INDUCTIVE\n",
    "    evaluation_scope = EvaluationScope.BATCH_TRANSFORM\n",
)

replace_once(
    "packages/neuros-mechint/src/neuros_mechint/representations/tphate.py",
    '''from .contracts import (
    FitRegime,
''',
    '''from .contracts import (
    EvaluationScope,
    FitRegime,
''',
)
insert_after_once(
    "packages/neuros-mechint/src/neuros_mechint/representations/tphate.py",
    "    fit_regime = FitRegime.TRANSDUCTIVE_TARGET_OBSERVED\n",
    "    evaluation_scope = EvaluationScope.SEQUENCE_LOCAL\n",
)

replace_once(
    "packages/neuros-mechint/src/neuros_mechint/representations/external.py",
    "from .contracts import FitRegime, RepresentationEmbedding, SequenceBatch, _freeze_metadata\n",
    "from .contracts import (\n    EvaluationScope,\n    FitRegime,\n    RepresentationEmbedding,\n    SequenceBatch,\n    _freeze_metadata,\n)\n",
)
insert_after_once(
    "packages/neuros-mechint/src/neuros_mechint/representations/external.py",
    "    fit_regime = FitRegime.EXTERNAL_PRETRAINED\n",
    "    evaluation_scope = EvaluationScope.BATCH_TRANSFORM\n",
)

replace_once(
    "packages/neuros-mechint/src/neuros_mechint/representations/__init__.py",
    '''from .contracts import (
    FitRegime,
''',
    '''from .contracts import (
    EvaluationScope,
    FitRegime,
''',
)
insert_after_once(
    "packages/neuros-mechint/src/neuros_mechint/representations/__init__.py",
    '    "ControlledTemporalManifold",\n',
    '    "EvaluationScope",\n',
)

print("contract/scope patch applied")

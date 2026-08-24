"""Dataset-specific MOABB semantics for longitudinal neurOS evidence studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class MOABBLongitudinalDatasetSpec:
    key: str
    class_name: str
    source_id: str
    description: str
    paradigm: Literal["left_right", "motor_imagery"]
    events: tuple[str, ...] | None = None
    expected_session_order: tuple[str, ...] | None = None

    def case_metadata(self, subject: int) -> dict[str, Any]:
        metadata: dict[str, Any] = {"subject": int(subject)}
        if self.key == "kumar2024":
            if not 1 <= int(subject) <= 18:
                raise ValueError("Kumar2024 subject must lie in 1..18")
            metadata["original_protocol"] = "GR" if int(subject) <= 9 else "PAR"
        return metadata


MOABB_LONGITUDINAL_DATASETS: dict[str, MOABBLongitudinalDatasetSpec] = {
    "kumar2024": MOABBLongitudinalDatasetSpec(
        key="kumar2024",
        class_name="Kumar2024",
        source_id="moabb-kumar2024",
        description="18 participants x 6 separate-day MI sessions",
        paradigm="left_right",
        expected_session_order=("0", "1", "2", "3", "4", "5"),
    ),
    "ma2020": MOABBLongitudinalDatasetSpec(
        key="ma2020",
        class_name="Ma2020",
        source_id="moabb-ma2020",
        description="25 participants x 15 right-hand/right-elbow MI sessions",
        paradigm="motor_imagery",
        events=("right_hand", "right_elbow"),
    ),
    "lee2019-mi": MOABBLongitudinalDatasetSpec(
        key="lee2019-mi",
        class_name="Lee2019_MI",
        source_id="moabb-lee2019-family",
        description="OpenBMI MI member of the shared 54-person cohort",
        paradigm="left_right",
    ),
    "wang2026": MOABBLongitudinalDatasetSpec(
        key="wang2026",
        class_name="Wang2026",
        source_id="moabb-wang2026",
        description="39 participants x 5 sessions with online cursor-control study",
        paradigm="left_right",
    ),
}


def get_moabb_longitudinal_spec(key: str) -> MOABBLongitudinalDatasetSpec:
    try:
        return MOABB_LONGITUDINAL_DATASETS[key]
    except KeyError as exc:
        raise KeyError(
            f"unknown longitudinal MOABB dataset {key!r}; "
            f"available={sorted(MOABB_LONGITUDINAL_DATASETS)}"
        ) from exc


def build_moabb_longitudinal_dataset(
    key: str,
    *,
    fmin: float = 8.0,
    fmax: float = 30.0,
    resample: float | None = None,
):
    """Instantiate the declared upstream dataset and paradigm.

    The function fails closed when the installed MOABB version does not expose a
    declared dataset or considers the dataset/paradigm combination invalid.
    """
    try:
        import moabb.datasets as datasets
        from moabb.paradigms import LeftRightImagery, MotorImagery
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "MOABB longitudinal data requires `neuros-foundation[evidence]`."
        ) from exc

    spec = get_moabb_longitudinal_spec(key)
    dataset_cls = getattr(datasets, spec.class_name, None)
    if dataset_cls is None:
        raise RuntimeError(
            f"installed MOABB does not expose {spec.class_name}; choose another "
            "dataset or pin an upstream version that contains it"
        )
    dataset = dataset_cls()
    if spec.paradigm == "left_right":
        paradigm = LeftRightImagery(
            fmin=float(fmin),
            fmax=float(fmax),
            resample=resample,
        )
    else:
        assert spec.events is not None
        paradigm = MotorImagery(
            n_classes=len(spec.events),
            events=list(spec.events),
            fmin=float(fmin),
            fmax=float(fmax),
            resample=resample,
        )
    if not paradigm.is_valid(dataset):
        raise RuntimeError(
            f"MOABB rejected {spec.class_name} for declared paradigm {spec.paradigm}"
        )
    return spec, dataset, paradigm


def validate_observed_sessions(
    spec: MOABBLongitudinalDatasetSpec,
    observed: tuple[str, ...],
) -> tuple[str, ...]:
    """Apply dataset-specific chronology checks when upstream semantics are pinned."""
    observed = tuple(str(value) for value in observed)
    if spec.expected_session_order is not None and observed != spec.expected_session_order:
        raise ValueError(
            f"{spec.key} session order differs from pinned upstream chronology; "
            f"expected={spec.expected_session_order}, observed={observed}"
        )
    return observed

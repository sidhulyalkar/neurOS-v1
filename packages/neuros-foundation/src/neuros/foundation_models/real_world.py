"""Real-world evidence sources and leakage-resistant grouped evaluation contracts.

This module deliberately stops before model fitting. Its job is to make the
*evidence boundary* explicit: what public source is being used, which deployment
unit is held out, and which examples are allowed to influence fitting or
calibration. Model/representation benchmarks can then consume the resulting
partition under :class:`~neuros.foundation_models.benchmark.EvaluationProtocol`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np

from .benchmark import EvaluationProtocol, SplitUnit, TransferRegime

EvidenceRole = Literal[
    "canonical_benchmark",
    "longitudinal_bci",
    "online_bci",
    "cross_subject_transfer",
    "cross_session_transfer",
    "cross_paradigm_transfer",
    "cross_site_transfer",
    "cross_day_adaptation",
    "few_shot_adaptation",
    "foundation_pretraining",
    "population_modeling",
    "mechanism_replication",
    "representation_stress_test",
]


@dataclass(frozen=True, slots=True)
class EvidenceSource:
    """A public source selected for one or more neurOS evidence questions.

    Counts are descriptive metadata, not benchmark results. ``sessions`` may be
    ``None`` when the natural unit is a day/set rather than a uniform per-subject
    session count.
    """

    id: str
    title: str
    ecosystem: str
    modality: str
    task: str
    access_url: str
    citation: str
    license: str | None = None
    subjects: int | None = None
    sessions: int | None = None
    roles: tuple[EvidenceRole, ...] = ()
    external_id: str | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("evidence source id must be non-empty")
        if not self.title.strip():
            raise ValueError("evidence source title must be non-empty")
        if not self.access_url.startswith("https://"):
            raise ValueError("evidence source access_url must use https")
        if self.subjects is not None and self.subjects <= 0:
            raise ValueError("subjects must be positive when provided")
        if self.sessions is not None and self.sessions <= 0:
            raise ValueError("sessions must be positive when provided")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Stable public sources selected because they expose deployment-relevant shift,
# not because neurOS claims state-of-the-art performance on them. Each source
# should answer a distinct falsifiable question.
REAL_WORLD_EVIDENCE_SOURCES: tuple[EvidenceSource, ...] = (
    EvidenceSource(
        id="moabb-wang2026",
        title="Wang2026 sensory-guided motor-imagery BCI",
        ecosystem="MOABB",
        modality="EEG",
        task="4-class motor imagery with online 1D/2D cursor control",
        access_url=(
            "https://moabb.neurotechx.com/docs/generated/"
            "moabb.datasets.Wang2026.html"
        ),
        citation="doi:10.1038/s41467-026-75435-5; data doi:10.1184/R1/32293995.v1",
        license="CC BY-NC 4.0",
        subjects=39,
        sessions=5,
        roles=(
            "longitudinal_bci",
            "online_bci",
            "cross_subject_transfer",
            "cross_session_transfer",
        ),
        external_id="Wang2026",
        notes=(
            "Primary non-invasive flagship: repeated sessions plus real online cursor feedback.",
            "Use cohort/group metadata explicitly; do not collapse experimental conditions silently.",
        ),
    ),
    EvidenceSource(
        id="moabb-kumar2024",
        title="Kumar2024 longitudinal motor-imagery training",
        ecosystem="MOABB",
        modality="EEG",
        task="2-class motor imagery with repeated online visual feedback",
        access_url=(
            "https://moabb.neurotechx.com/docs/generated/"
            "moabb.datasets.Kumar2024.html"
        ),
        citation="doi:10.1093/pnasnexus/pgae076; data doi:10.5281/zenodo.10694880",
        license="CC BY 4.0",
        subjects=18,
        sessions=6,
        roles=(
            "longitudinal_bci",
            "online_bci",
            "cross_session_transfer",
            "few_shot_adaptation",
        ),
        external_id="Kumar2024",
        notes=(
            "Useful for calibration-cost/adaptation studies because session 1 is offline and later sessions include online feedback.",
            "MOABB includes the bar-feedback runs and excludes the car-racing runs.",
        ),
    ),
    EvidenceSource(
        id="moabb-ma2020",
        title="Ma2020 same-limb motor-imagery longitudinal EEG",
        ecosystem="MOABB",
        modality="EEG",
        task="2-class right-hand versus right-elbow motor imagery",
        access_url=(
            "https://moabb.neurotechx.com/docs/generated/"
            "moabb.datasets.Ma2020.html"
        ),
        citation="doi:10.1038/s41597-020-0535-2; data doi:10.7910/DVN/RBN3XG",
        license="CC BY 4.0",
        subjects=25,
        sessions=15,
        roles=(
            "longitudinal_bci",
            "cross_session_transfer",
            "representation_stress_test",
        ),
        external_id="Ma2020",
        notes=(
            "Fifteen motor-imagery sessions per subject make this a strong drift/stability stress test.",
        ),
    ),
    EvidenceSource(
        id="moabb-lee2019-family",
        title="Lee2019 OpenBMI MI / ERP / SSVEP family",
        ecosystem="MOABB",
        modality="EEG",
        task="three BCI paradigms in a shared 54-participant cohort",
        access_url=(
            "https://moabb.neurotechx.com/docs/generated/"
            "moabb.datasets.Lee2019_MI.html"
        ),
        citation="doi:10.1093/gigascience/giz002; data doi:10.5524/100542",
        license="GPL 3.0 dataset/toolbox distribution",
        subjects=54,
        sessions=2,
        roles=(
            "online_bci",
            "cross_subject_transfer",
            "cross_session_transfer",
            "cross_paradigm_transfer",
        ),
        external_id="Lee2019",
        notes=(
            "The same OpenBMI study includes MI, ERP/P300, and SSVEP variants.",
            "Use this family to test cross-paradigm representation reuse without mixing paradigm-specific labels or metrics.",
        ),
    ),
    EvidenceSource(
        id="falcon-h1",
        title="FALCON H1 human reach-and-grasp iBCI",
        ecosystem="FALCON / DANDI",
        modality="intracortical",
        task="reach and grasp kinematic decoding across days",
        access_url="https://snel-repo.github.io/falcon/datasets.html",
        citation="FALCON benchmark; DANDI 000954",
        license=None,
        subjects=1,
        sessions=None,
        roles=("cross_day_adaptation", "few_shot_adaptation"),
        external_id="DANDI:000954",
        notes=(
            "Human Utah-array data with explicit early-day held-in and later-day held-out benchmark sets.",
            "Primary invasive ORION adaptation target after verifying the released neural representation contract.",
        ),
    ),
    EvidenceSource(
        id="falcon-h2",
        title="FALCON H2 human handwriting iBCI",
        ecosystem="FALCON / DANDI",
        modality="intracortical",
        task="handwriting / brain-to-text decoding across days",
        access_url="https://snel-repo.github.io/falcon/datasets.html",
        citation="FALCON benchmark; DANDI 000950",
        license=None,
        subjects=1,
        sessions=None,
        roles=("cross_day_adaptation", "few_shot_adaptation"),
        external_id="DANDI:000950",
        notes=(
            "Communication iBCI target with a long chronological held-in period followed by later held-out days.",
        ),
    ),
    EvidenceSource(
        id="nlb-mc-maze",
        title="Neural Latents Benchmark MC_Maze",
        ecosystem="NLB / DANDI",
        modality="macaque motor-cortex spikes",
        task="delayed center-out reaching with straight and curved trajectories",
        access_url="https://neurallatents.github.io/datasets.html",
        citation="Neural Latents Benchmark (NLB'21)",
        license=None,
        subjects=None,
        sessions=None,
        roles=(
            "canonical_benchmark",
            "population_modeling",
            "representation_stress_test",
        ),
        external_id="NLB:MC_Maze",
        notes=(
            "Canonical population-modeling benchmark with motor and premotor cortex spiking plus kinematics.",
            "The former EvalAI test data became public for local evaluation in January 2026; report multiple seeds and avoid tuning directly on the released test split.",
        ),
    ),
    EvidenceSource(
        id="ibl-repeated-site",
        title="IBL Reproducible Ephys repeated-site release",
        ecosystem="IBL / ONE",
        modality="mouse Neuropixels spikes/LFP",
        task="standardized decision-making task at the same repeated brain site across laboratories",
        access_url=(
            "https://docs.internationalbrainlab.org/notebooks_external/"
            "2024_data_release_repro_ephys.html"
        ),
        citation="International Brain Laboratory Reproducible Ephys data release",
        license=None,
        subjects=None,
        sessions=91,
        roles=(
            "cross_site_transfer",
            "population_modeling",
            "mechanism_replication",
            "representation_stress_test",
        ),
        external_id="IBL:RepeatedSite",
        notes=(
            "Ninety-one released Neuropixels sessions across 12 laboratories target the same repeated site.",
            "High-value SourceWeigher and mechanistic-stability benchmark because lab/site effects can be tested without deliberately changing the task and target location.",
        ),
    ),
    EvidenceSource(
        id="ibl-brain-wide-map",
        title="IBL Brain Wide Map 2025",
        ecosystem="IBL / ONE",
        modality="mouse Neuropixels spikes/LFP + behavior/video",
        task="standardized decision-making task sampled across the brain and laboratories",
        access_url=(
            "https://docs.internationalbrainlab.org/notebooks_external/"
            "2025_data_release_brainwidemap.html"
        ),
        citation="International Brain Laboratory Brain Wide Map; doi:10.1038/s41586-025-09235-0",
        license=None,
        subjects=139,
        sessions=459,
        roles=(
            "cross_site_transfer",
            "foundation_pretraining",
            "population_modeling",
            "mechanism_replication",
        ),
        external_id="IBL:Brainwidemap",
        notes=(
            "The current release spans 459 sessions, 699 probe insertions, 139 subjects, and 12 laboratories.",
            "Use for cross-lab/brain-region representation and mechanism replication, not as a direct human BCI efficacy claim.",
        ),
    ),
    EvidenceSource(
        id="eegdash-corpus",
        title="EEGDash BIDS-first electrophysiology corpus",
        ecosystem="EEGDash",
        modality="EEG/MEG/iEEG/fNIRS/EMG",
        task="large-scale corpus discovery and representation pretraining",
        access_url="https://eegdash.org/",
        citation="EEGDash project",
        license=None,
        subjects=None,
        sessions=None,
        roles=("foundation_pretraining", "cross_subject_transfer"),
        external_id="EEGDash",
        notes=(
            "Use as a scale/discovery corpus, not as a single leaderboard score.",
            "Preserve BIDS subject/session/task/run entities in every downstream split.",
        ),
    ),
)

_SOURCE_BY_ID = {source.id: source for source in REAL_WORLD_EVIDENCE_SOURCES}
if len(_SOURCE_BY_ID) != len(REAL_WORLD_EVIDENCE_SOURCES):  # pragma: no cover
    raise RuntimeError("duplicate real-world evidence source id")

_EXTERNAL_TO_SOURCE_ID = {
    source.external_id: source.id
    for source in REAL_WORLD_EVIDENCE_SOURCES
    if source.external_id is not None
}


def get_evidence_source(source_id: str) -> EvidenceSource:
    """Return one curated evidence source by stable neurOS id."""
    try:
        return _SOURCE_BY_ID[source_id]
    except KeyError as exc:
        raise KeyError(
            f"unknown evidence source {source_id!r}; available={sorted(_SOURCE_BY_ID)}"
        ) from exc


def find_evidence_sources(
    *,
    modality: str | None = None,
    ecosystem: str | None = None,
    role: EvidenceRole | None = None,
) -> tuple[EvidenceSource, ...]:
    """Filter the curated evidence catalog without downloading any data."""
    values = REAL_WORLD_EVIDENCE_SOURCES
    if modality is not None:
        needle = modality.strip().lower()
        values = tuple(source for source in values if needle in source.modality.lower())
    if ecosystem is not None:
        needle = ecosystem.strip().lower()
        values = tuple(source for source in values if needle in source.ecosystem.lower())
    if role is not None:
        values = tuple(source for source in values if role in source.roles)
    return values


def _metadata_records(metadata: Any) -> tuple[dict[str, Any], ...]:
    """Normalize pandas-style or sequence metadata into plain dictionaries."""
    if hasattr(metadata, "to_dict"):
        try:
            records = metadata.to_dict(orient="records")
        except TypeError:
            records = metadata.to_dict("records")
    else:
        records = metadata
    if isinstance(records, Mapping) or isinstance(records, (str, bytes)):
        raise TypeError("metadata must be a table/sequence of row mappings")
    try:
        normalized = tuple(dict(row) for row in records)
    except (TypeError, ValueError) as exc:
        raise TypeError("metadata rows must be mapping-like") from exc
    return normalized


def _string_group(
    records: Sequence[Mapping[str, Any]],
    key: str,
) -> np.ndarray | None:
    if not records or any(key not in row for row in records):
        return None
    return np.asarray([str(row[key]) for row in records], dtype=str)


def _recording_group(groups: Mapping[str, np.ndarray]) -> np.ndarray | None:
    """Derive MOABB's contiguous-recording identity as subject/session/run."""
    if "subject" not in groups or "session" not in groups:
        return None
    subject = groups["subject"]
    session = groups["session"]
    run = groups.get("run")
    if run is None:
        return np.asarray(
            [
                f"{sub}/{ses}"
                for sub, ses in zip(subject, session, strict=True)
            ],
            dtype=str,
        )
    return np.asarray(
        [
            f"{sub}/{ses}/{run_id}"
            for sub, ses, run_id in zip(subject, session, run, strict=True)
        ],
        dtype=str,
    )


@dataclass(frozen=True, slots=True)
class GroupedEvaluationData:
    """Array data plus explicit deployment-unit identities.

    ``groups`` is the critical field. A benchmark that cannot state which
    subject/session/site/device/recording each sample belongs to cannot claim a
    deployment-unit-disjoint result through this contract.
    """

    dataset_id: str
    X: np.ndarray
    y: np.ndarray
    groups: Mapping[str, np.ndarray]
    metadata: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        x = np.asarray(self.X)
        y = np.asarray(self.y)
        if x.ndim == 0:
            raise ValueError("X must have a sample dimension")
        if y.ndim == 0:
            raise ValueError("y must have a sample dimension")
        if len(x) != len(y):
            raise ValueError("X and y must contain the same number of samples")
        if len(x) < 2:
            raise ValueError("real-world evaluation requires at least two samples")
        normalized_groups: dict[str, np.ndarray] = {}
        for key, values in self.groups.items():
            name = str(key).strip()
            if not name:
                raise ValueError("group names must be non-empty")
            array = np.asarray(values).reshape(-1).astype(str)
            if len(array) != len(x):
                raise ValueError(f"group {name!r} length must match X")
            normalized_groups[name] = array
        if not normalized_groups:
            raise ValueError("at least one deployment-unit group is required")
        if self.metadata and len(self.metadata) != len(x):
            raise ValueError("metadata row count must match X")
        object.__setattr__(self, "X", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "groups", normalized_groups)

    @classmethod
    def from_moabb_result(
        cls,
        result: tuple[Any, Any, Any],
        *,
        dataset_id: str,
    ) -> "GroupedEvaluationData":
        """Build from MOABB's conventional ``(X, labels, metadata)`` result.

        No MOABB import is required here, which keeps the core foundation
        package light and makes the metadata/split contract independently
        testable. Install ``neuros-foundation[evidence]`` to obtain MOABB 1.5+
        for actual dataset loading.
        """
        if not isinstance(result, tuple) or len(result) != 3:
            raise TypeError("MOABB result must be a 3-tuple: (X, labels, metadata)")
        X, y, metadata = result
        records = _metadata_records(metadata)
        if len(records) != len(np.asarray(X)):
            raise ValueError("MOABB metadata row count must match X")

        groups: dict[str, np.ndarray] = {}
        for key in ("subject", "session", "run"):
            values = _string_group(records, key)
            if values is not None:
                groups[key] = values
        recording = _recording_group(groups)
        if recording is not None:
            groups["recording"] = recording
        if not groups:
            raise ValueError(
                "MOABB metadata must contain at least one of subject/session/run"
            )
        return cls(
            dataset_id=dataset_id,
            X=np.asarray(X),
            y=np.asarray(y),
            groups=groups,
            metadata=records,
        )


def collect_moabb(
    dataset: Any,
    paradigm: Any,
    *,
    subjects: Sequence[int] | None = None,
    dataset_id: str | None = None,
    **get_data_kwargs: Any,
) -> GroupedEvaluationData:
    """Collect one MOABB paradigm result through a tiny, testable adapter.

    ``dataset`` and ``paradigm`` are intentionally duck-typed. With MOABB 1.5+
    this corresponds to ``paradigm.get_data(dataset=dataset, subjects=...)``.
    The function performs no implicit preprocessing beyond whatever paradigm is
    supplied by the caller; preprocessing identity belongs in the benchmark
    manifest/protocol.
    """
    getter = getattr(paradigm, "get_data", None)
    if not callable(getter):
        raise TypeError("paradigm must provide a callable get_data method")
    result = getter(dataset=dataset, subjects=subjects, **get_data_kwargs)
    upstream_id = str(
        getattr(dataset, "code", None) or dataset.__class__.__name__
    )
    resolved_id = dataset_id or _EXTERNAL_TO_SOURCE_ID.get(upstream_id, upstream_id)
    return GroupedEvaluationData.from_moabb_result(
        result,
        dataset_id=str(resolved_id),
    )


@dataclass(frozen=True, slots=True)
class EvaluationPartition:
    """Train/test indices for one explicit held-out deployment unit."""

    data: GroupedEvaluationData
    split_unit: SplitUnit
    train_indices: np.ndarray
    test_indices: np.ndarray
    held_out_values: tuple[str, ...]

    def __post_init__(self) -> None:
        train = np.asarray(self.train_indices, dtype=np.int64).reshape(-1)
        test = np.asarray(self.test_indices, dtype=np.int64).reshape(-1)
        if len(train) == 0 or len(test) == 0:
            raise ValueError("both train and test partitions must be non-empty")
        if np.intersect1d(train, test).size:
            raise ValueError("train and test indices overlap")
        if np.any(train < 0) or np.any(test < 0):
            raise ValueError("partition indices must be non-negative")
        if np.any(train >= len(self.data.X)) or np.any(test >= len(self.data.X)):
            raise ValueError("partition indices exceed dataset length")
        object.__setattr__(self, "train_indices", train)
        object.__setattr__(self, "test_indices", test)

    @property
    def train(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.X[self.train_indices], self.data.y[self.train_indices]

    @property
    def test(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.X[self.test_indices], self.data.y[self.test_indices]

    def protocol(
        self,
        *,
        name: str,
        transfer_regime: TransferRegime = "linear_probe",
        preprocessing: str = "fit preprocessing on train partition only",
        notes: Iterable[str] = (),
        seed: int = 0,
    ) -> EvaluationProtocol:
        """Create the shared foundation-model protocol for this partition."""
        held = ",".join(self.held_out_values)
        return EvaluationProtocol(
            name=name,
            split_unit=self.split_unit,
            transfer_regime=transfer_regime,
            preprocessing=preprocessing,
            seed=seed,
            leakage_controls=(
                f"no {self.split_unit} overlap between train/test",
                "fit normalization/preprocessing on train partition only",
                "adaptation/calibration examples are excluded from final evaluation examples",
            ),
            notes=(
                f"dataset={self.data.dataset_id}",
                f"held_out_{self.split_unit}={held}",
                *tuple(notes),
            ),
        )

    @property
    def fingerprint(self) -> str:
        """Fingerprint split assignment, labels, and deployment identities.

        This is not a raw-data checksum. A promoted benchmark artifact must also
        preserve the upstream dataset/version checksum supplied by its data
        ecosystem. The partition fingerprint prevents accidental split drift.
        """
        payload = {
            "dataset_id": self.data.dataset_id,
            "split_unit": self.split_unit,
            "held_out_values": self.held_out_values,
            "train_indices": self.train_indices.tolist(),
            "test_indices": self.test_indices.tolist(),
            "labels": np.asarray(self.data.y).astype(str).tolist(),
            "groups": {
                key: np.asarray(values).astype(str).tolist()
                for key, values in sorted(self.data.groups.items())
            },
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def manifest(self, *, protocol: EvaluationProtocol) -> dict[str, Any]:
        """Emit a machine-readable pre-model evidence manifest."""
        if protocol.split_unit != self.split_unit:
            raise ValueError(
                "protocol split unit does not match partition: "
                f"{protocol.split_unit!r} != {self.split_unit!r}"
            )
        group = self.data.groups[self.split_unit]
        train_values = sorted(set(group[self.train_indices].tolist()))
        test_values = sorted(set(group[self.test_indices].tolist()))
        overlap = sorted(set(train_values).intersection(test_values))
        if overlap:
            raise ValueError(f"deployment-unit leakage detected: {overlap}")
        source = _SOURCE_BY_ID.get(self.data.dataset_id)
        return {
            "schema_version": 1,
            "evidence_tier": "real_dataset",
            "dataset_id": self.data.dataset_id,
            "source": source.to_dict() if source is not None else None,
            "n_samples": int(len(self.data.X)),
            "train_samples": int(len(self.train_indices)),
            "test_samples": int(len(self.test_indices)),
            "input_shape": list(self.data.X.shape),
            "target_shape": list(self.data.y.shape),
            "split_unit": self.split_unit,
            "held_out_values": list(self.held_out_values),
            "train_group_values": train_values,
            "test_group_values": test_values,
            "partition_fingerprint": self.fingerprint,
            "protocol": protocol.to_dict(),
            "limitations": [
                "partition fingerprint is not a raw-dataset checksum",
                "this manifest establishes split/provenance semantics, not model superiority",
            ],
        }


def hold_out_groups(
    data: GroupedEvaluationData,
    *,
    split_unit: SplitUnit,
    held_out_values: Iterable[Any],
) -> EvaluationPartition:
    """Create a strict held-out subject/session/site/device/recording split."""
    if split_unit == "sample":
        raise ValueError(
            "hold_out_groups is for deployment-unit evaluation; "
            "use an explicit sample split elsewhere"
        )
    if split_unit not in data.groups:
        raise ValueError(
            f"dataset has no {split_unit!r} group; available={sorted(data.groups)}"
        )
    held = tuple(sorted({str(value) for value in held_out_values}))
    if not held:
        raise ValueError("held_out_values must be non-empty")
    group = np.asarray(data.groups[split_unit]).astype(str)
    known = set(group.tolist())
    missing = sorted(set(held) - known)
    if missing:
        raise ValueError(f"unknown held-out {split_unit} values: {missing}")
    test_mask = np.isin(group, np.asarray(held, dtype=str))
    train = np.flatnonzero(~test_mask)
    test = np.flatnonzero(test_mask)
    if len(train) == 0:
        raise ValueError("held-out values consume the entire dataset")
    return EvaluationPartition(
        data=data,
        split_unit=split_unit,
        train_indices=train,
        test_indices=test,
        held_out_values=held,
    )

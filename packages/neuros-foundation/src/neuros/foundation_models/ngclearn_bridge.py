"""Optional ngc-learn interoperability for biologically plausible neural dynamics.

The bridge is intentionally outside ``neuros-core``. ngc-learn and JAX remain
optional research dependencies, while neurOS owns explicit input geometry,
upstream identity, output geometry, and evidence hashes.

The initial qualified surface is deliberately narrow: ngc-learn 3.2 RateCell
execution. Predictive-coding circuits, spiking networks, Hebbian/STDP learning,
and larger ngc-learn systems can be layered on this boundary after their exact
execution contracts are independently exercised.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

NGCLEARN_SUPPORTED_SERIES = (3, 2)
NGCLEARN_REFERENCE_REPOSITORY = "https://github.com/NACLab/ngc-learn"
NgcLearnOutput = Literal["linear", "nonlinear"]


class NgcLearnUnavailableError(ImportError):
    """Raised when the optional ngc-learn/JAX integration cannot be used."""


class NgcLearnVersionError(RuntimeError):
    """Raised when an unqualified ngc-learn version is selected."""


def _version() -> str:
    try:
        return importlib.metadata.version("ngclearn")
    except importlib.metadata.PackageNotFoundError as exc:
        raise NgcLearnUnavailableError(
            'ngc-learn is not installed; use `pip install "neuros-foundation[ngclearn]"`'
        ) from exc


def _major_minor(version: str) -> tuple[int, int]:
    core = version.split("+", 1)[0].split("-", 1)[0]
    pieces = core.split(".")
    if len(pieces) < 2:
        raise NgcLearnVersionError(f"cannot parse ngc-learn version {version!r}")
    try:
        return int(pieces[0]), int(pieces[1])
    except ValueError as exc:
        raise NgcLearnVersionError(f"cannot parse ngc-learn version {version!r}") from exc


def _load_upstream() -> tuple[Any, Any, Any, Any, Any, str]:
    version = _version()
    if _major_minor(version) != NGCLEARN_SUPPORTED_SERIES:
        raise NgcLearnVersionError(
            "neurOS currently qualifies only the ngc-learn 3.2 line; "
            f"found {version}. Install ngclearn>=3.2,<3.3 or add a new qualification lane."
        )
    try:
        ngclearn = importlib.import_module("ngclearn")
        components = importlib.import_module("ngclearn.components")
        jax = importlib.import_module("jax")
        jnp = importlib.import_module("jax.numpy")
        random = importlib.import_module("jax.random")
    except ImportError as exc:
        raise NgcLearnUnavailableError(
            "ngc-learn is installed but its JAX/runtime dependencies could not be imported"
        ) from exc

    for symbol in ("RateCell", "LIFCell", "HebbianSynapse", "STDPSynapse"):
        if not hasattr(components, symbol):
            raise NgcLearnVersionError(
                f"qualified ngc-learn 3.2 surface is missing expected component {symbol}"
            )
    if not hasattr(ngclearn, "Context") or not hasattr(ngclearn, "MethodProcess"):
        raise NgcLearnVersionError("qualified ngc-learn surface is missing Context/MethodProcess")
    return ngclearn, components, jax, jnp, random, version


def ngclearn_runtime_identity() -> dict[str, Any]:
    """Return the exact installed upstream surface used by the integration."""

    ngclearn, components, jax, _, _, version = _load_upstream()
    try:
        x64_enabled: bool | None = bool(jax.config.read("jax_enable_x64"))
    except Exception:
        x64_enabled = None
    return {
        "integration": "ngc-learn",
        "reference_repository": NGCLEARN_REFERENCE_REPOSITORY,
        "ngclearn_version": version,
        "jax_version": str(getattr(jax, "__version__", "unknown")),
        "jax_backend": str(jax.default_backend()),
        "jax_enable_x64": x64_enabled,
        "qualified_series": "3.2.x",
        "qualified_symbols": [
            f"{ngclearn.__name__}.Context",
            f"{ngclearn.__name__}.MethodProcess",
            f"{components.RateCell.__module__}.{components.RateCell.__name__}",
            f"{components.LIFCell.__module__}.{components.LIFCell.__name__}",
            f"{components.HebbianSynapse.__module__}.{components.HebbianSynapse.__name__}",
            f"{components.STDPSynapse.__module__}.{components.STDPSynapse.__name__}",
        ],
    }


def _matrix(values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"samples must be a 2D time x channel matrix; got shape {matrix.shape}")
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("samples must contain at least one time step and one channel")
    if not np.isfinite(matrix).all():
        raise ValueError("samples contain NaN or infinite values")
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class NgcLearnRateCellEvidence:
    """Provenance for one ngc-learn RateCell transform."""

    ngclearn_version: str
    jax_version: str
    jax_backend: str
    jax_enable_x64: bool | None
    component: str
    output_compartment: str
    tau_m_ms: float
    gamma: float
    activation: str
    integration_type: str
    seed: int
    sample_rate_hz: float
    dt_ms: float
    input_shape: tuple[int, int]
    output_shape: tuple[int, int]
    input_sha256: str
    output_sha256: str
    evidence_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "integration": "ngc-learn",
            "reference_repository": NGCLEARN_REFERENCE_REPOSITORY,
            "ngclearn_version": self.ngclearn_version,
            "jax_version": self.jax_version,
            "jax_backend": self.jax_backend,
            "jax_enable_x64": self.jax_enable_x64,
            "component": self.component,
            "output_compartment": self.output_compartment,
            "tau_m_ms": self.tau_m_ms,
            "gamma": self.gamma,
            "activation": self.activation,
            "integration_type": self.integration_type,
            "seed": self.seed,
            "sample_rate_hz": self.sample_rate_hz,
            "dt_ms": self.dt_ms,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "evidence_sha256": self.evidence_sha256,
            "claim_boundary": {
                "upstream_package_executed": True,
                "rate_cell_contract_exercised": True,
                "predictive_coding_circuit_qualified": False,
                "spiking_network_qualified": False,
                "online_learning_qualified": False,
                "real_dataset_qualified": False,
                "hardware_qualified": False,
                "closed_loop_qualified": False,
            },
        }


@dataclass(slots=True)
class NgcLearnRateCellResult:
    """RateCell representation values plus immutable execution evidence."""

    values: np.ndarray
    evidence: NgcLearnRateCellEvidence


class NgcLearnRateCellTransform:
    """Execute an ngc-learn 3.2 RateCell over a time x channel matrix.

    The transform performs no resampling, filtering, normalization, padding,
    channel reordering, or fitting. One RateCell unit is created per input
    channel and the upstream dynamical state is reset before each call.
    """

    def __init__(
        self,
        *,
        tau_m_ms: float = 10.0,
        gamma: float = 1.0,
        activation: str = "identity",
        integration_type: str = "euler",
        output: NgcLearnOutput = "linear",
        seed: int = 0,
    ) -> None:
        if isinstance(tau_m_ms, bool) or not isinstance(tau_m_ms, (int, float)):
            raise ValueError("tau_m_ms must be numeric")
        if float(tau_m_ms) <= 0 or not math.isfinite(float(tau_m_ms)):
            raise ValueError("tau_m_ms must be positive and finite")
        if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
            raise ValueError("gamma must be numeric")
        if not math.isfinite(float(gamma)):
            raise ValueError("gamma must be finite")
        if not isinstance(activation, str) or not activation.strip():
            raise ValueError("activation must be a non-empty string")
        if not isinstance(integration_type, str) or not integration_type.strip():
            raise ValueError("integration_type must be a non-empty string")
        if output not in {"linear", "nonlinear"}:
            raise ValueError("output must be 'linear' or 'nonlinear'")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.tau_m_ms = float(tau_m_ms)
        self.gamma = float(gamma)
        self.activation = activation.strip()
        self.integration_type = integration_type.strip()
        self.output = output
        self.seed = seed

    def transform(self, samples: Any, *, sample_rate_hz: float) -> NgcLearnRateCellResult:
        if isinstance(sample_rate_hz, bool) or not isinstance(sample_rate_hz, (int, float)):
            raise ValueError("sample_rate_hz must be numeric")
        sample_rate_hz = float(sample_rate_hz)
        if sample_rate_hz <= 0 or not math.isfinite(sample_rate_hz):
            raise ValueError("sample_rate_hz must be positive and finite")

        matrix = _matrix(samples)
        ngclearn, components, jax, jnp, random, version = _load_upstream()
        dt_ms = 1000.0 / sample_rate_hz
        n_channels = matrix.shape[1]

        with ngclearn.Context("neuros_ngclearn_ratecell"):
            cell = components.RateCell(
                "z0",
                n_units=n_channels,
                tau_m=self.tau_m_ms,
                act_fx=self.activation,
                prior=("gaussian", self.gamma),
                integration_type=self.integration_type,
                key=random.PRNGKey(self.seed),
            )
            advance_process = ngclearn.MethodProcess("advance") >> cell.advance_state
            reset_process = ngclearn.MethodProcess("reset") >> cell.reset

        reset_process.run()
        outputs: list[np.ndarray] = []
        for index, row in enumerate(matrix):
            cell.j.set(jnp.asarray(row[None, :]))
            advance_process.run(t=float(index) * dt_ms, dt=dt_ms)
            compartment = cell.z if self.output == "linear" else cell.zF
            value = np.asarray(compartment.get(), dtype=np.float64).reshape(-1)
            if value.size != n_channels:
                raise RuntimeError(
                    "ngc-learn RateCell output geometry changed unexpectedly: "
                    f"expected {n_channels} values, received {value.size}"
                )
            if not np.isfinite(value).all():
                raise RuntimeError("ngc-learn RateCell produced non-finite output")
            outputs.append(value)

        representation = np.ascontiguousarray(np.stack(outputs, axis=0), dtype=np.float64)
        try:
            x64_enabled: bool | None = bool(jax.config.read("jax_enable_x64"))
        except Exception:
            x64_enabled = None
        payload: dict[str, Any] = {
            "ngclearn_version": version,
            "jax_version": str(getattr(jax, "__version__", "unknown")),
            "jax_backend": str(jax.default_backend()),
            "jax_enable_x64": x64_enabled,
            "component": f"{components.RateCell.__module__}.{components.RateCell.__name__}",
            "output_compartment": "z" if self.output == "linear" else "zF",
            "tau_m_ms": self.tau_m_ms,
            "gamma": self.gamma,
            "activation": self.activation,
            "integration_type": self.integration_type,
            "seed": self.seed,
            "sample_rate_hz": sample_rate_hz,
            "dt_ms": dt_ms,
            "input_shape": [int(value) for value in matrix.shape],
            "output_shape": [int(value) for value in representation.shape],
            "input_sha256": _array_sha256(matrix),
            "output_sha256": _array_sha256(representation),
        }
        payload["evidence_sha256"] = _canonical_sha256(payload)
        evidence = NgcLearnRateCellEvidence(
            ngclearn_version=str(payload["ngclearn_version"]),
            jax_version=str(payload["jax_version"]),
            jax_backend=str(payload["jax_backend"]),
            jax_enable_x64=payload["jax_enable_x64"],
            component=str(payload["component"]),
            output_compartment=str(payload["output_compartment"]),
            tau_m_ms=float(payload["tau_m_ms"]),
            gamma=float(payload["gamma"]),
            activation=str(payload["activation"]),
            integration_type=str(payload["integration_type"]),
            seed=int(payload["seed"]),
            sample_rate_hz=float(payload["sample_rate_hz"]),
            dt_ms=float(payload["dt_ms"]),
            input_shape=tuple(payload["input_shape"]),
            output_shape=tuple(payload["output_shape"]),
            input_sha256=str(payload["input_sha256"]),
            output_sha256=str(payload["output_sha256"]),
            evidence_sha256=str(payload["evidence_sha256"]),
        )
        return NgcLearnRateCellResult(values=representation, evidence=evidence)

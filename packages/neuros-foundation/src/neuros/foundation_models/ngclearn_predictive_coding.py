"""Qualified ngc-learn predictive-coding representation dynamics.

This module intentionally qualifies a small, inference-only predictive-coding
surface. It uses real ngc-learn 3.2 components to infer a latent representation
by iteratively reducing reconstruction mismatch. Synaptic learning is not part
of this contract; fixed generative weights make state mutation, replay, and
claim boundaries explicit.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from neuros.foundation_models.ngclearn_bridge import (
    NGCLEARN_REFERENCE_REPOSITORY,
    NgcLearnVersionError,
    _array_sha256,
    _canonical_sha256,
    _load_upstream,
    _matrix,
)

NgcLearnPredictiveOutput = Literal["linear", "nonlinear"]
_PC_CONTEXT_IDS = itertools.count()
PC_METHOD_ID = "neuros-ngclearn-predictive-reconstruction-v1"


def _positive_finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if result <= 0 or not math.isfinite(result):
        raise ValueError(f"{name} must be positive and finite")
    return result


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _default_weights(*, latent_dim: int, input_dim: int, seed: int, scale: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((latent_dim, input_dim)).astype(np.float64)
    singular_values = np.linalg.svd(raw, compute_uv=False)
    spectral_norm = float(singular_values[0]) if singular_values.size else 0.0
    if spectral_norm <= np.finfo(np.float64).eps:
        raise RuntimeError("could not construct a non-degenerate predictive-coding weight matrix")
    return np.ascontiguousarray(raw * (scale / spectral_norm), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class NgcLearnPredictiveCodingEvidence:
    """Immutable provenance for one predictive-coding transform execution."""

    method_id: str
    ngclearn_version: str
    jax_version: str
    jax_backend: str
    jax_enable_x64: bool | None
    latent_component: str
    error_component: str
    generative_synapse_component: str
    feedback_synapse_component: str
    activation: str
    integration_type: str
    output_compartment: str
    latent_dim: int
    settling_steps: int
    settling_dt_ms: float
    tau_m_ms: float
    prior_gamma: float
    weight_scale: float
    seed: int
    sample_rate_hz: float
    reset_per_sample: bool
    tied_transpose_feedback: bool
    learning_enabled: bool
    input_shape: tuple[int, int]
    latent_shape: tuple[int, int]
    reconstruction_shape: tuple[int, int]
    input_sha256: str
    weights_sha256: str
    latent_sha256: str
    reconstruction_sha256: str
    error_trajectory_sha256: str
    initial_mse: float
    final_mse: float
    error_reduction_fraction: float | None
    samples_improved_fraction: float
    evidence_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "integration": "ngc-learn",
            "reference_repository": NGCLEARN_REFERENCE_REPOSITORY,
            "ngclearn_version": self.ngclearn_version,
            "jax_version": self.jax_version,
            "jax_backend": self.jax_backend,
            "jax_enable_x64": self.jax_enable_x64,
            "components": {
                "latent": self.latent_component,
                "error": self.error_component,
                "generative_synapse": self.generative_synapse_component,
                "feedback_synapse": self.feedback_synapse_component,
            },
            "activation": self.activation,
            "integration_type": self.integration_type,
            "output_compartment": self.output_compartment,
            "latent_dim": self.latent_dim,
            "settling_steps": self.settling_steps,
            "settling_dt_ms": self.settling_dt_ms,
            "tau_m_ms": self.tau_m_ms,
            "prior_gamma": self.prior_gamma,
            "weight_scale": self.weight_scale,
            "seed": self.seed,
            "sample_rate_hz": self.sample_rate_hz,
            "reset_per_sample": self.reset_per_sample,
            "tied_transpose_feedback": self.tied_transpose_feedback,
            "learning_enabled": self.learning_enabled,
            "input_shape": list(self.input_shape),
            "latent_shape": list(self.latent_shape),
            "reconstruction_shape": list(self.reconstruction_shape),
            "input_sha256": self.input_sha256,
            "weights_sha256": self.weights_sha256,
            "latent_sha256": self.latent_sha256,
            "reconstruction_sha256": self.reconstruction_sha256,
            "error_trajectory_sha256": self.error_trajectory_sha256,
            "initial_mse": self.initial_mse,
            "final_mse": self.final_mse,
            "error_reduction_fraction": self.error_reduction_fraction,
            "samples_improved_fraction": self.samples_improved_fraction,
            "evidence_sha256": self.evidence_sha256,
            "claim_boundary": {
                "upstream_package_executed": True,
                "predictive_coding_circuit_qualified": True,
                "iterative_error_feedback_exercised": True,
                "fixed_weight_inference_only": True,
                "hebbian_learning_qualified": False,
                "stdp_learning_qualified": False,
                "online_learning_qualified": False,
                "real_dataset_qualified": False,
                "hardware_qualified": False,
                "closed_loop_qualified": False,
                "clinical_qualified": False,
            },
        }


@dataclass(slots=True)
class NgcLearnPredictiveCodingResult:
    """Latent values, reconstructions, and settling evidence."""

    values: np.ndarray
    reconstruction: np.ndarray
    mean_squared_error_by_step: np.ndarray
    evidence: NgcLearnPredictiveCodingEvidence


class NgcLearnPredictiveCodingTransform:
    """Infer fixed-weight predictive-coding latents for time x channel samples.

    Each row is treated as an independent observation. The circuit is reset,
    the observation is clamped to a Gaussian prediction-error target, and a
    latent ``RateCell`` settles under residual feedback from the reconstruction.
    The generative synapse is fixed and the feedback synapse is tied to its
    transpose. No fitting, filtering, resampling, padding, or channel reordering
    occurs inside this transform.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 8,
        settling_steps: int = 20,
        settling_dt_ms: float = 1.0,
        tau_m_ms: float = 20.0,
        prior_gamma: float = 0.0,
        activation: str = "identity",
        integration_type: str = "euler",
        output: NgcLearnPredictiveOutput = "nonlinear",
        weight_scale: float = 0.75,
        seed: int = 0,
        weights: Any | None = None,
    ) -> None:
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim < 1:
            raise ValueError("latent_dim must be a positive integer")
        if isinstance(settling_steps, bool) or not isinstance(settling_steps, int) or settling_steps < 1:
            raise ValueError("settling_steps must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(activation, str) or not activation.strip():
            raise ValueError("activation must be a non-empty string")
        if not isinstance(integration_type, str) or not integration_type.strip():
            raise ValueError("integration_type must be a non-empty string")
        if output not in {"linear", "nonlinear"}:
            raise ValueError("output must be 'linear' or 'nonlinear'")
        self.latent_dim = latent_dim
        self.settling_steps = settling_steps
        self.settling_dt_ms = _positive_finite("settling_dt_ms", settling_dt_ms)
        self.tau_m_ms = _positive_finite("tau_m_ms", tau_m_ms)
        self.prior_gamma = _finite("prior_gamma", prior_gamma)
        if self.prior_gamma < 0:
            raise ValueError("prior_gamma must be non-negative")
        self.weight_scale = _positive_finite("weight_scale", weight_scale)
        self.activation = activation.strip()
        self.integration_type = integration_type.strip()
        self.output = output
        self.seed = seed
        self._provided_weights = None if weights is None else np.asarray(weights, dtype=np.float64)
        if self._provided_weights is not None:
            if self._provided_weights.ndim != 2:
                raise ValueError("weights must be a 2D latent x input matrix")
            if self._provided_weights.shape[0] != latent_dim:
                raise ValueError(
                    "weights first dimension must equal latent_dim; "
                    f"expected {latent_dim}, received {self._provided_weights.shape[0]}"
                )
            if not np.isfinite(self._provided_weights).all():
                raise ValueError("weights contain NaN or infinite values")
            self._provided_weights = np.ascontiguousarray(self._provided_weights, dtype=np.float64)
        self._context_id = next(_PC_CONTEXT_IDS)
        self._runtime: tuple[Any, ...] | None = None
        self._input_dim: int | None = None
        self._weights: np.ndarray | None = None

    def _ensure_runtime(self, input_dim: int) -> tuple[Any, ...]:
        if self._runtime is not None:
            if self._input_dim != input_dim:
                raise ValueError(
                    "NgcLearnPredictiveCodingTransform input channel count is fixed after first execution; "
                    f"expected {self._input_dim}, received {input_dim}"
                )
            return self._runtime

        ngclearn, components, jax, jnp, random, version = _load_upstream()
        for symbol in ("GaussianErrorCell", "StaticSynapse", "RateCell"):
            if not hasattr(components, symbol):
                raise NgcLearnVersionError(
                    f"qualified ngc-learn 3.2 predictive-coding surface is missing {symbol}"
                )

        if self._provided_weights is not None:
            if self._provided_weights.shape[1] != input_dim:
                raise ValueError(
                    "weights second dimension must equal input channel count; "
                    f"expected {input_dim}, received {self._provided_weights.shape[1]}"
                )
            weights = self._provided_weights.copy()
        else:
            weights = _default_weights(
                latent_dim=self.latent_dim,
                input_dim=input_dim,
                seed=self.seed,
                scale=self.weight_scale,
            )

        context_name = f"neuros_ngclearn_pc_{self._context_id}"
        with ngclearn.Context(context_name):
            latent = components.RateCell(
                "z",
                n_units=self.latent_dim,
                tau_m=self.tau_m_ms,
                act_fx=self.activation,
                prior=("gaussian", self.prior_gamma),
                integration_type=self.integration_type,
                key=random.PRNGKey(self.seed),
            )
            error = components.GaussianErrorCell("e0", n_units=input_dim)
            generative = components.StaticSynapse(
                "W",
                shape=(self.latent_dim, input_dim),
                key=random.PRNGKey(self.seed + 1),
            )
            feedback = components.StaticSynapse(
                "E",
                shape=(input_dim, self.latent_dim),
                key=random.PRNGKey(self.seed + 2),
            )

            latent.zF >> generative.inputs
            generative.outputs >> error.mu
            error.dmu >> feedback.inputs
            feedback.outputs >> latent.j

            # Follow the upstream Rao/Ballard-style discrete message-passing
            # order: previous residual -> latent update -> reconstruction -> new residual.
            advance = (
                ngclearn.MethodProcess("pc_advance")
                >> feedback.advance_state
                >> latent.advance_state
                >> generative.advance_state
                >> error.advance_state
            )
            reset = (
                ngclearn.MethodProcess("pc_reset")
                >> latent.reset
                >> error.reset
                >> generative.reset
                >> feedback.reset
            )

        generative.weights.set(jnp.asarray(weights))
        feedback.weights.set(jnp.asarray(weights.T))
        self._weights = np.ascontiguousarray(weights, dtype=np.float64)
        self._input_dim = input_dim
        self._runtime = (
            latent,
            error,
            generative,
            feedback,
            advance,
            reset,
            jax,
            jnp,
            version,
        )
        return self._runtime

    def transform(self, samples: Any, *, sample_rate_hz: float) -> NgcLearnPredictiveCodingResult:
        sample_rate_hz = _positive_finite("sample_rate_hz", sample_rate_hz)
        matrix = _matrix(samples)
        input_dim = matrix.shape[1]
        (
            latent,
            error,
            generative,
            feedback,
            advance,
            reset,
            jax,
            jnp,
            version,
        ) = self._ensure_runtime(input_dim)
        assert self._weights is not None

        latents: list[np.ndarray] = []
        reconstructions: list[np.ndarray] = []
        trajectories: list[np.ndarray] = []
        sample_improved: list[bool] = []

        for row in matrix:
            reset.run()
            target = jnp.asarray(row[None, :])
            error.target.set(target)
            initial_mse = float(np.mean(np.square(row)))
            mse_steps = [initial_mse]

            for step in range(self.settling_steps):
                advance.run(t=float(step) * self.settling_dt_ms, dt=self.settling_dt_ms)
                reconstruction = np.asarray(error.mu.get(), dtype=np.float64).reshape(-1)
                if reconstruction.size != input_dim or not np.isfinite(reconstruction).all():
                    raise RuntimeError("ngc-learn predictive-coding reconstruction is invalid")
                mse_steps.append(float(np.mean(np.square(row - reconstruction))))

            compartment = latent.z if self.output == "linear" else latent.zF
            latent_value = np.asarray(compartment.get(), dtype=np.float64).reshape(-1)
            reconstruction = np.asarray(error.mu.get(), dtype=np.float64).reshape(-1)
            if latent_value.size != self.latent_dim or not np.isfinite(latent_value).all():
                raise RuntimeError("ngc-learn predictive-coding latent geometry is invalid")
            latents.append(latent_value)
            reconstructions.append(reconstruction)
            trajectory = np.asarray(mse_steps, dtype=np.float64)
            trajectories.append(trajectory)
            sample_improved.append(bool(trajectory[-1] <= trajectory[0] + 1e-12))

        representation = np.ascontiguousarray(np.stack(latents), dtype=np.float64)
        reconstruction_matrix = np.ascontiguousarray(np.stack(reconstructions), dtype=np.float64)
        trajectory_matrix = np.ascontiguousarray(np.stack(trajectories), dtype=np.float64)
        mean_trajectory = np.ascontiguousarray(trajectory_matrix.mean(axis=0), dtype=np.float64)
        initial_mse = float(mean_trajectory[0])
        final_mse = float(mean_trajectory[-1])
        reduction = None if initial_mse <= np.finfo(np.float64).eps else float(1.0 - final_mse / initial_mse)
        try:
            x64_enabled: bool | None = bool(jax.config.read("jax_enable_x64"))
        except Exception:
            x64_enabled = None

        payload: dict[str, Any] = {
            "method_id": PC_METHOD_ID,
            "ngclearn_version": version,
            "jax_version": str(getattr(jax, "__version__", "unknown")),
            "jax_backend": str(jax.default_backend()),
            "jax_enable_x64": x64_enabled,
            "latent_component": f"{latent.__class__.__module__}.{latent.__class__.__name__}",
            "error_component": f"{error.__class__.__module__}.{error.__class__.__name__}",
            "generative_synapse_component": f"{generative.__class__.__module__}.{generative.__class__.__name__}",
            "feedback_synapse_component": f"{feedback.__class__.__module__}.{feedback.__class__.__name__}",
            "activation": self.activation,
            "integration_type": self.integration_type,
            "output_compartment": "z" if self.output == "linear" else "zF",
            "latent_dim": self.latent_dim,
            "settling_steps": self.settling_steps,
            "settling_dt_ms": self.settling_dt_ms,
            "tau_m_ms": self.tau_m_ms,
            "prior_gamma": self.prior_gamma,
            "weight_scale": self.weight_scale,
            "seed": self.seed,
            "sample_rate_hz": sample_rate_hz,
            "reset_per_sample": True,
            "tied_transpose_feedback": True,
            "learning_enabled": False,
            "input_shape": [int(v) for v in matrix.shape],
            "latent_shape": [int(v) for v in representation.shape],
            "reconstruction_shape": [int(v) for v in reconstruction_matrix.shape],
            "input_sha256": _array_sha256(matrix),
            "weights_sha256": _array_sha256(self._weights),
            "latent_sha256": _array_sha256(representation),
            "reconstruction_sha256": _array_sha256(reconstruction_matrix),
            "error_trajectory_sha256": _array_sha256(mean_trajectory),
            "initial_mse": initial_mse,
            "final_mse": final_mse,
            "error_reduction_fraction": reduction,
            "samples_improved_fraction": float(np.mean(sample_improved)),
        }
        payload["evidence_sha256"] = _canonical_sha256(payload)
        evidence = NgcLearnPredictiveCodingEvidence(
            method_id=PC_METHOD_ID,
            ngclearn_version=str(payload["ngclearn_version"]),
            jax_version=str(payload["jax_version"]),
            jax_backend=str(payload["jax_backend"]),
            jax_enable_x64=payload["jax_enable_x64"],
            latent_component=str(payload["latent_component"]),
            error_component=str(payload["error_component"]),
            generative_synapse_component=str(payload["generative_synapse_component"]),
            feedback_synapse_component=str(payload["feedback_synapse_component"]),
            activation=str(payload["activation"]),
            integration_type=str(payload["integration_type"]),
            output_compartment=str(payload["output_compartment"]),
            latent_dim=int(payload["latent_dim"]),
            settling_steps=int(payload["settling_steps"]),
            settling_dt_ms=float(payload["settling_dt_ms"]),
            tau_m_ms=float(payload["tau_m_ms"]),
            prior_gamma=float(payload["prior_gamma"]),
            weight_scale=float(payload["weight_scale"]),
            seed=int(payload["seed"]),
            sample_rate_hz=float(payload["sample_rate_hz"]),
            reset_per_sample=True,
            tied_transpose_feedback=True,
            learning_enabled=False,
            input_shape=tuple(payload["input_shape"]),
            latent_shape=tuple(payload["latent_shape"]),
            reconstruction_shape=tuple(payload["reconstruction_shape"]),
            input_sha256=str(payload["input_sha256"]),
            weights_sha256=str(payload["weights_sha256"]),
            latent_sha256=str(payload["latent_sha256"]),
            reconstruction_sha256=str(payload["reconstruction_sha256"]),
            error_trajectory_sha256=str(payload["error_trajectory_sha256"]),
            initial_mse=float(payload["initial_mse"]),
            final_mse=float(payload["final_mse"]),
            error_reduction_fraction=payload["error_reduction_fraction"],
            samples_improved_fraction=float(payload["samples_improved_fraction"]),
            evidence_sha256=str(payload["evidence_sha256"]),
        )
        return NgcLearnPredictiveCodingResult(
            values=representation,
            reconstruction=reconstruction_matrix,
            mean_squared_error_by_step=mean_trajectory,
            evidence=evidence,
        )

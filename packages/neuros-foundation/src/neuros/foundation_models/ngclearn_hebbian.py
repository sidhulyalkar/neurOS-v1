"""Auditable ngc-learn Hebbian adaptation for predictive reconstruction.

This module qualifies one deliberately narrow state-mutating surface: a
one-layer predictive reconstruction circuit whose generative synapse is a real
ngc-learn 3.2 ``HebbianSynapse``. Inference settles before each M-step, feedback
is tied to the current generative transpose, and evaluation can be run without
mutating learned state.

ORION adaptation authority is intentionally not imported here. The foundation
integration owns the external learning mechanism; evidence scripts compose it
with ORION's authority without reversing package dependencies.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

from neuros.foundation_models.ngclearn_bridge import (
    NGCLEARN_REFERENCE_REPOSITORY,
    NgcLearnVersionError,
    _array_sha256,
    _canonical_sha256,
    _load_upstream,
    _matrix,
)
from neuros.foundation_models.ngclearn_predictive_coding import _default_weights

HEBBIAN_PC_METHOD_ID = "neuros-ngclearn-hebbian-predictive-reconstruction-v1"


def _positive_finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if result <= 0 or not math.isfinite(result):
        raise ValueError(f"{name} must be positive and finite")
    return result


def _nonnegative_finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if result < 0 or not math.isfinite(result):
        raise ValueError(f"{name} must be non-negative and finite")
    return result


def _tree_sha256(jax: Any, tree: Any) -> str:
    """Hash a JAX pytree by deterministic leaf order, dtype, shape, and bytes."""

    digest = hashlib.sha256()
    digest.update(b"neuros.ngclearn.optimizer-state.v1\0")
    leaves = jax.tree_util.tree_leaves(tree)
    digest.update(str(len(leaves)).encode("ascii"))
    digest.update(b"\0")
    for index, leaf in enumerate(leaves):
        array = np.ascontiguousarray(np.asarray(leaf))
        if array.dtype.hasobject:
            raise TypeError("optimizer state contains object-dtype leaf")
        digest.update(str(index).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(int(v) for v in array.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def _state_sha256(weights_sha256: str, optimizer_sha256: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"neuros.ngclearn.hebbian-state.v1\0")
    digest.update(weights_sha256.encode("ascii"))
    digest.update(b"\0")
    digest.update(optimizer_sha256.encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class NgcLearnHebbianState:
    """Exact mutable learning state required for identity and rollback."""

    weights: np.ndarray
    optimizer_state: Any
    weights_sha256: str
    optimizer_sha256: str
    state_sha256: str


@dataclass(frozen=True, slots=True)
class NgcLearnHebbianAdaptationEvidence:
    method_id: str
    ngclearn_version: str
    jax_version: str
    jax_backend: str
    jax_enable_x64: bool | None
    latent_component: str
    error_component: str
    generative_synapse_component: str
    feedback_synapse_component: str
    latent_dim: int
    settling_steps: int
    settling_dt_ms: float
    tau_m_ms: float
    prior_gamma: float
    activation: str
    learning_rate: float
    optimizer: str
    sign_value: float
    weight_bound: float
    row_normalization_after_update: bool
    epochs: int
    n_observations: int
    update_count: int
    sample_rate_hz: float
    adaptation_input_sha256: str
    state_before_sha256: str
    state_after_sha256: str
    weights_before_sha256: str
    weights_after_sha256: str
    optimizer_before_sha256: str
    optimizer_after_sha256: str
    weight_delta_l2: float
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
            "latent_dim": self.latent_dim,
            "settling_steps": self.settling_steps,
            "settling_dt_ms": self.settling_dt_ms,
            "tau_m_ms": self.tau_m_ms,
            "prior_gamma": self.prior_gamma,
            "activation": self.activation,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "sign_value": self.sign_value,
            "weight_bound": self.weight_bound,
            "row_normalization_after_update": self.row_normalization_after_update,
            "epochs": self.epochs,
            "n_observations": self.n_observations,
            "update_count": self.update_count,
            "sample_rate_hz": self.sample_rate_hz,
            "adaptation_input_sha256": self.adaptation_input_sha256,
            "state_before_sha256": self.state_before_sha256,
            "state_after_sha256": self.state_after_sha256,
            "weights_before_sha256": self.weights_before_sha256,
            "weights_after_sha256": self.weights_after_sha256,
            "optimizer_before_sha256": self.optimizer_before_sha256,
            "optimizer_after_sha256": self.optimizer_after_sha256,
            "weight_delta_l2": self.weight_delta_l2,
            "evidence_sha256": self.evidence_sha256,
            "claim_boundary": {
                "upstream_package_executed": True,
                "hebbian_synapse_executed": True,
                "predictive_coding_inference_before_update": True,
                "state_identity_includes_optimizer": True,
                "rollback_state_supported": True,
                "transactional_checkpoint_validation": True,
                "row_normalization_after_update": False,
                "orion_authority_applied_here": False,
                "real_dataset_qualified": False,
                "calibration_reduction_qualified": False,
                "stdp_learning_qualified": False,
                "hardware_qualified": False,
                "closed_loop_qualified": False,
                "clinical_qualified": False,
            },
        }


@dataclass(slots=True)
class NgcLearnHebbianAdaptationResult:
    state_before: NgcLearnHebbianState
    state_after: NgcLearnHebbianState
    evidence: NgcLearnHebbianAdaptationEvidence


@dataclass(slots=True)
class NgcLearnHebbianInferenceResult:
    values: np.ndarray
    reconstruction: np.ndarray
    mean_squared_error: float
    state_sha256: str


class NgcLearnHebbianPredictiveCoding:
    """One-layer predictive coding with real ngc-learn Hebbian M-steps.

    The object keeps learned synaptic/optimizer state across ``adapt`` calls.
    ``infer`` never calls ``HebbianSynapse.evolve`` and must preserve the full
    state SHA-256. No filtering, resampling, normalization, padding, channel
    reordering, or hidden row normalization is performed.
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
        learning_rate: float = 1e-3,
        optimizer: str = "sgd",
        sign_value: float = -1.0,
        weight_bound: float = 0.0,
        weight_scale: float = 0.25,
        seed: int = 0,
        weights: Any | None = None,
    ) -> None:
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim < 1:
            raise ValueError("latent_dim must be a positive integer")
        if isinstance(settling_steps, bool) or not isinstance(settling_steps, int) or settling_steps < 1:
            raise ValueError("settling_steps must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if optimizer not in {"sgd", "adam"}:
            raise ValueError("optimizer must be 'sgd' or 'adam'")
        if not isinstance(activation, str) or not activation.strip():
            raise ValueError("activation must be a non-empty string")
        self.latent_dim = latent_dim
        self.settling_steps = settling_steps
        self.settling_dt_ms = _positive_finite("settling_dt_ms", settling_dt_ms)
        self.tau_m_ms = _positive_finite("tau_m_ms", tau_m_ms)
        self.prior_gamma = _nonnegative_finite("prior_gamma", prior_gamma)
        self.learning_rate = _positive_finite("learning_rate", learning_rate)
        self.sign_value = float(sign_value)
        if not math.isfinite(self.sign_value) or self.sign_value == 0.0:
            raise ValueError("sign_value must be finite and non-zero")
        self.weight_bound = _nonnegative_finite("weight_bound", weight_bound)
        self.weight_scale = _positive_finite("weight_scale", weight_scale)
        self.activation = activation.strip()
        self.optimizer = optimizer
        self.seed = seed
        self._provided_weights = None if weights is None else np.asarray(weights)
        if self._provided_weights is not None:
            if self._provided_weights.ndim != 2 or self._provided_weights.shape[0] != latent_dim:
                raise ValueError("weights must be a 2D latent_dim x input_dim matrix")
            if not np.issubdtype(self._provided_weights.dtype, np.number):
                raise ValueError("weights must be numeric")
            if not np.isfinite(self._provided_weights).all():
                raise ValueError("weights contain NaN or infinite values")
            self._provided_weights = np.ascontiguousarray(self._provided_weights)
        self._runtime: tuple[Any, ...] | None = None
        self._input_dim: int | None = None

    def _ensure_runtime(self, input_dim: int) -> tuple[Any, ...]:
        if self._runtime is not None:
            if self._input_dim != input_dim:
                raise ValueError(
                    "input channel count is fixed after first execution; "
                    f"expected {self._input_dim}, received {input_dim}"
                )
            return self._runtime

        ngclearn, components, jax, jnp, random, version = _load_upstream()
        for symbol in ("RateCell", "GaussianErrorCell", "HebbianSynapse", "StaticSynapse"):
            if not hasattr(components, symbol):
                raise NgcLearnVersionError(
                    f"qualified ngc-learn 3.2 Hebbian surface is missing {symbol}"
                )

        if self._provided_weights is not None:
            if self._provided_weights.shape[1] != input_dim:
                raise ValueError(
                    "weights second dimension must equal input channel count; "
                    f"expected {input_dim}, received {self._provided_weights.shape[1]}"
                )
            initial_weights = self._provided_weights.copy()
        else:
            initial_weights = _default_weights(
                latent_dim=self.latent_dim,
                input_dim=input_dim,
                seed=self.seed,
                scale=self.weight_scale,
            )

        # ngcsimlib keeps a process-global Context registry. UUID naming avoids
        # collisions even after notebook/module reloads that reset module counters.
        context_name = f"neuros_ngclearn_hebbian_pc_{uuid.uuid4().hex}"
        with ngclearn.Context(context_name):
            latent = components.RateCell(
                "z",
                n_units=self.latent_dim,
                tau_m=self.tau_m_ms,
                act_fx=self.activation,
                prior=("gaussian", self.prior_gamma),
                integration_type="euler",
                key=random.PRNGKey(self.seed),
            )
            error = components.GaussianErrorCell("e0", n_units=input_dim)
            generative = components.HebbianSynapse(
                "W",
                shape=(self.latent_dim, input_dim),
                eta=self.learning_rate,
                w_bound=self.weight_bound,
                is_nonnegative=False,
                prior=("constant", 0.0),
                sign_value=self.sign_value,
                optim_type=self.optimizer,
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
            latent.zF >> generative.pre
            error.dmu >> generative.post

            advance = (
                ngclearn.MethodProcess("hebbian_pc_advance")
                >> feedback.advance_state
                >> latent.advance_state
                >> generative.advance_state
                >> error.advance_state
            )
            reset = (
                ngclearn.MethodProcess("hebbian_pc_reset")
                >> latent.reset
                >> error.reset
                >> generative.reset
                >> feedback.reset
            )
            evolve = ngclearn.MethodProcess("hebbian_pc_evolve") >> generative.evolve

        generative.weights.set(jnp.asarray(initial_weights))
        feedback.weights.set(jnp.asarray(initial_weights.T))
        self._input_dim = input_dim
        self._runtime = (
            latent,
            error,
            generative,
            feedback,
            advance,
            reset,
            evolve,
            jax,
            jnp,
            version,
        )
        return self._runtime

    def snapshot_state(self) -> NgcLearnHebbianState:
        if self._runtime is None:
            raise RuntimeError("state is unavailable before the first infer/adapt call")
        _, _, generative, _, _, _, _, jax, _, _ = self._runtime
        # Preserve the upstream dtype exactly. State identity is defined on the
        # actual JAX weight array, not a float64-normalized convenience copy.
        weights = np.ascontiguousarray(np.asarray(generative.weights.get())).copy()
        weights.setflags(write=False)
        weights_sha = _array_sha256(weights)
        # JAX leaves are immutable; tree_map clones all list/tuple/dict container
        # structure so the checkpoint never aliases the live optimizer container.
        optimizer_state = jax.tree_util.tree_map(
            lambda leaf: leaf,
            generative.opt_params.get(),
        )
        optimizer_sha = _tree_sha256(jax, optimizer_state)
        return NgcLearnHebbianState(
            weights=weights,
            optimizer_state=optimizer_state,
            weights_sha256=weights_sha,
            optimizer_sha256=optimizer_sha,
            state_sha256=_state_sha256(weights_sha, optimizer_sha),
        )

    def _validate_checkpoint(self, state: NgcLearnHebbianState) -> None:
        if self._runtime is None:
            raise RuntimeError("runtime must be initialized before state validation")
        _, _, generative, _, _, _, _, jax, _, _ = self._runtime
        expected_shape = tuple(int(v) for v in generative.weights.get().shape)
        if tuple(state.weights.shape) != expected_shape:
            raise ValueError(
                f"rollback weight shape mismatch: expected {expected_shape}, received {state.weights.shape}"
            )
        actual_weights_sha = _array_sha256(np.ascontiguousarray(state.weights))
        if actual_weights_sha != state.weights_sha256:
            raise ValueError("rollback checkpoint weight SHA-256 does not match checkpoint contents")
        actual_optimizer_sha = _tree_sha256(jax, state.optimizer_state)
        if actual_optimizer_sha != state.optimizer_sha256:
            raise ValueError("rollback checkpoint optimizer SHA-256 does not match checkpoint contents")
        actual_state_sha = _state_sha256(actual_weights_sha, actual_optimizer_sha)
        if actual_state_sha != state.state_sha256:
            raise ValueError("rollback checkpoint state SHA-256 does not match checkpoint contents")

    def restore_state(self, state: NgcLearnHebbianState) -> None:
        if self._runtime is None:
            raise RuntimeError("runtime must be initialized before state restoration")
        # Validate the complete checkpoint before mutating live state. A corrupt
        # rollback object must fail transactionally and leave the learner intact.
        self._validate_checkpoint(state)
        _, _, generative, feedback, _, reset, _, _, jnp, _ = self._runtime
        generative.weights.set(jnp.asarray(state.weights))
        generative.opt_params.set(state.optimizer_state)
        feedback.weights.set(jnp.asarray(state.weights.T))
        reset.run()
        restored = self.snapshot_state()
        if restored.state_sha256 != state.state_sha256:
            raise RuntimeError("restored ngc-learn state does not match requested rollback identity")

    def _settle_row(self, row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self._runtime is not None
        latent, error, generative, feedback, advance, reset, _, _, jnp, _ = self._runtime
        reset.run()
        # Feedback is tied to the current generative state before each inference.
        feedback.weights.set(jnp.transpose(generative.weights.get()))
        error.target.set(jnp.asarray(row[None, :]))
        for step in range(self.settling_steps):
            advance.run(t=float(step) * self.settling_dt_ms, dt=self.settling_dt_ms)
        latent_value = np.asarray(latent.zF.get()).reshape(-1)
        reconstruction = np.asarray(error.mu.get()).reshape(-1)
        if latent_value.size != self.latent_dim or not np.isfinite(latent_value).all():
            raise RuntimeError("ngc-learn Hebbian predictive latent is invalid")
        if reconstruction.size != row.size or not np.isfinite(reconstruction).all():
            raise RuntimeError("ngc-learn Hebbian predictive reconstruction is invalid")
        return latent_value, reconstruction

    def infer(self, samples: Any, *, sample_rate_hz: float) -> NgcLearnHebbianInferenceResult:
        sample_rate_hz = _positive_finite("sample_rate_hz", sample_rate_hz)
        matrix = _matrix(samples)
        self._ensure_runtime(matrix.shape[1])
        state_before = self.snapshot_state()
        latents: list[np.ndarray] = []
        reconstructions: list[np.ndarray] = []
        for row in matrix:
            latent, reconstruction = self._settle_row(row)
            latents.append(latent)
            reconstructions.append(reconstruction)
        state_after = self.snapshot_state()
        if state_after.state_sha256 != state_before.state_sha256:
            raise RuntimeError("inference mutated ngc-learn Hebbian learning state")
        latent_matrix = np.ascontiguousarray(np.stack(latents))
        reconstruction_matrix = np.ascontiguousarray(np.stack(reconstructions))
        mse = float(np.mean(np.square(matrix - reconstruction_matrix)))
        return NgcLearnHebbianInferenceResult(
            values=latent_matrix,
            reconstruction=reconstruction_matrix,
            mean_squared_error=mse,
            state_sha256=state_after.state_sha256,
        )

    def adapt(
        self,
        samples: Any,
        *,
        sample_rate_hz: float,
        epochs: int = 1,
    ) -> NgcLearnHebbianAdaptationResult:
        sample_rate_hz = _positive_finite("sample_rate_hz", sample_rate_hz)
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        matrix = _matrix(samples)
        runtime = self._ensure_runtime(matrix.shape[1])
        latent, error, generative, feedback, _, _, evolve, jax, jnp, version = runtime
        before = self.snapshot_state()
        update_count = 0

        for _ in range(epochs):
            for row in matrix:
                # E-step: reset activities, tie feedback, and settle on one observation.
                self._settle_row(row)
                # M-step: real upstream two-factor Hebbian update using the final
                # latent pre statistic and Gaussian residual post statistic.
                evolve.run(
                    t=float(self.settling_steps) * self.settling_dt_ms,
                    dt=self.settling_dt_ms,
                )
                feedback.weights.set(jnp.transpose(generative.weights.get()))
                update_count += 1

        after = self.snapshot_state()
        if after.state_sha256 == before.state_sha256:
            raise RuntimeError("Hebbian adaptation produced no mutable-state change")
        weight_delta_l2 = float(
            np.linalg.norm(after.weights.astype(np.float64) - before.weights.astype(np.float64))
        )
        if not math.isfinite(weight_delta_l2) or weight_delta_l2 <= 0:
            raise RuntimeError("Hebbian adaptation produced invalid weight delta")
        try:
            x64_enabled: bool | None = bool(jax.config.read("jax_enable_x64"))
        except Exception:
            x64_enabled = None

        payload: dict[str, Any] = {
            "method_id": HEBBIAN_PC_METHOD_ID,
            "ngclearn_version": version,
            "jax_version": str(getattr(jax, "__version__", "unknown")),
            "jax_backend": str(jax.default_backend()),
            "jax_enable_x64": x64_enabled,
            "latent_component": f"{latent.__class__.__module__}.{latent.__class__.__name__}",
            "error_component": f"{error.__class__.__module__}.{error.__class__.__name__}",
            "generative_synapse_component": f"{generative.__class__.__module__}.{generative.__class__.__name__}",
            "feedback_synapse_component": f"{feedback.__class__.__module__}.{feedback.__class__.__name__}",
            "latent_dim": self.latent_dim,
            "settling_steps": self.settling_steps,
            "settling_dt_ms": self.settling_dt_ms,
            "tau_m_ms": self.tau_m_ms,
            "prior_gamma": self.prior_gamma,
            "activation": self.activation,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "sign_value": self.sign_value,
            "weight_bound": self.weight_bound,
            "row_normalization_after_update": False,
            "epochs": epochs,
            "n_observations": int(matrix.shape[0]),
            "update_count": update_count,
            "sample_rate_hz": sample_rate_hz,
            "adaptation_input_sha256": _array_sha256(matrix),
            "state_before_sha256": before.state_sha256,
            "state_after_sha256": after.state_sha256,
            "weights_before_sha256": before.weights_sha256,
            "weights_after_sha256": after.weights_sha256,
            "optimizer_before_sha256": before.optimizer_sha256,
            "optimizer_after_sha256": after.optimizer_sha256,
            "weight_delta_l2": weight_delta_l2,
        }
        payload["evidence_sha256"] = _canonical_sha256(payload)
        evidence = NgcLearnHebbianAdaptationEvidence(**payload)
        return NgcLearnHebbianAdaptationResult(
            state_before=before,
            state_after=after,
            evidence=evidence,
        )

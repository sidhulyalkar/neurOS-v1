"""Multidimensional observable EEG audits for synthetic/recorded comparison.

The audit intentionally returns separate interpretable measurements instead of a
single 'realism score'. No finite set of sensor-space features establishes
physiological equivalence, but diverse features make obvious simulator/generator
shortcuts easier to detect.
"""
from __future__ import annotations

import numpy as np

AUDIT_SCHEMA = "neuros.synthetic_bci_arena.eeg_observable_audit.v1"


def _validate(data_uv: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
    data = np.asarray(data_uv, dtype=float)
    if data.ndim != 2 or data.shape[0] < 1 or data.shape[1] < 64:
        raise ValueError("expected channels x samples EEG with at least 64 samples")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    if not np.all(np.isfinite(data)):
        raise ValueError("observable EEG audit requires finite data")
    return data


def _median_channel_autocorrelation(centered: np.ndarray, lag: int) -> float:
    if lag <= 0 or lag >= centered.shape[1] - 2:
        return 0.0
    values = []
    for channel in centered:
        a = channel[:-lag]
        b = channel[lag:]
        if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
            continue
        values.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.median(values)) if values else 0.0


def _band_fraction(power: np.ndarray, frequencies: np.ndarray, low: float, high: float) -> float:
    valid = (frequencies >= 1.0) & (frequencies <= min(80.0, frequencies[-1]))
    band = (frequencies >= low) & (frequencies < high)
    denominator = float(np.sum(power[:, valid]))
    return float(np.sum(power[:, band]) / denominator) if denominator > 0 else 0.0


def _effective_rank(covariance: np.ndarray) -> tuple[float, float]:
    values = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    total = float(values.sum())
    if total <= 1e-15:
        return 0.0, 0.0
    probabilities = values / total
    positive = probabilities[probabilities > 0]
    entropy = -float(np.sum(positive * np.log(positive)))
    effective = float(np.exp(entropy))
    largest = float(values[-1] / total)
    return effective, largest


def _aperiodic_loglog_slope(power_1d: np.ndarray, frequencies: np.ndarray) -> float:
    # A deliberately simple observable summary, not a FOOOF/specparam replacement.
    # Exclude DC, mains-adjacent bins, and canonical alpha to reduce domination by
    # obvious oscillatory peaks. The value is reported as a regression slope,
    # not interpreted as a biological exponent.
    valid = (frequencies >= 2.0) & (frequencies <= min(40.0, frequencies[-1]))
    valid &= ~((frequencies >= 7.5) & (frequencies <= 13.5))
    valid &= ~((frequencies >= 49.0) & (frequencies <= 61.0))
    x = frequencies[valid]
    y = power_1d[valid]
    positive = (x > 0) & (y > 0)
    if np.count_nonzero(positive) < 8:
        return 0.0
    slope, _ = np.polyfit(np.log10(x[positive]), np.log10(y[positive]), 1)
    return float(slope)


def eeg_observable_audit(data_uv: np.ndarray, sampling_rate_hz: float) -> dict:
    """Return a versioned sensor-space audit with no aggregate realism score."""
    data = _validate(data_uv, sampling_rate_hz)
    centered = data - np.mean(data, axis=1, keepdims=True)
    abs_data = np.abs(centered)
    channel_std = np.std(centered, axis=1)
    channel_rms = np.sqrt(np.mean(centered**2, axis=1))
    channel_mad = np.median(np.abs(centered - np.median(centered, axis=1, keepdims=True)), axis=1)

    window = np.hanning(centered.shape[1])
    fft = np.fft.rfft(centered * window[None, :], axis=1)
    power = np.abs(fft) ** 2
    frequencies = np.fft.rfftfreq(centered.shape[1], d=1.0 / sampling_rate_hz)
    median_power = np.median(power, axis=0)
    usable = (frequencies >= 1.0) & (frequencies <= min(80.0, frequencies[-1]))
    normalized = median_power[usable]
    normalized = normalized / max(float(normalized.sum()), 1e-15)
    positive = normalized[normalized > 0]
    spectral_entropy = -float(np.sum(positive * np.log(positive)))
    if positive.size > 1:
        spectral_entropy /= float(np.log(positive.size))

    alpha = (frequencies >= 8.0) & (frequencies <= 13.0)
    alpha_peak = 0.0
    if np.any(alpha):
        alpha_freqs = frequencies[alpha]
        alpha_peak = float(alpha_freqs[int(np.argmax(median_power[alpha]))])

    covariance = np.cov(centered, bias=True)
    covariance = np.atleast_2d(covariance)
    trace_scale = float(np.trace(covariance) / covariance.shape[0]) if covariance.size else 1.0
    covariance = covariance + max(trace_scale, 1e-9) * 1e-6 * np.eye(covariance.shape[0])
    effective_rank, first_component_fraction = _effective_rank(covariance)
    correlation = np.corrcoef(centered)
    if correlation.ndim == 2 and correlation.shape[0] > 1:
        upper = np.abs(correlation[np.triu_indices(correlation.shape[0], 1)])
        mean_abs_corr = float(np.mean(upper))
        median_abs_corr = float(np.median(upper))
    else:
        mean_abs_corr = 0.0
        median_abs_corr = 0.0

    def lag_samples(seconds: float) -> int:
        return max(1, int(round(seconds * sampling_rate_hz)))

    zero_crossings = np.mean(np.diff(np.signbit(centered), axis=1) != 0, axis=1)
    return {
        "schema": AUDIT_SCHEMA,
        "sampling_rate_hz": float(sampling_rate_hz),
        "channels": int(data.shape[0]),
        "samples": int(data.shape[1]),
        "amplitude": {
            "median_channel_rms_uv": float(np.median(channel_rms)),
            "median_channel_mad_uv": float(np.median(channel_mad)),
            "p95_abs_uv": float(np.percentile(abs_data, 95)),
            "p99_abs_uv": float(np.percentile(abs_data, 99)),
            "max_abs_uv": float(np.max(abs_data)),
        },
        "spectrum": {
            "delta_1_4_fraction": _band_fraction(power, frequencies, 1.0, 4.0),
            "theta_4_8_fraction": _band_fraction(power, frequencies, 4.0, 8.0),
            "alpha_8_13_fraction": _band_fraction(power, frequencies, 8.0, 13.0),
            "beta_13_30_fraction": _band_fraction(power, frequencies, 13.0, 30.0),
            "gamma_30_45_fraction": _band_fraction(power, frequencies, 30.0, 45.0),
            "high_45_80_fraction": _band_fraction(power, frequencies, 45.0, 80.0),
            "alpha_peak_hz": alpha_peak,
            "normalized_spectral_entropy": float(spectral_entropy),
            "aperiodic_loglog_slope_2_40": _aperiodic_loglog_slope(median_power, frequencies),
        },
        "temporal": {
            "median_autocorrelation_20ms": _median_channel_autocorrelation(centered, lag_samples(0.020)),
            "median_autocorrelation_100ms": _median_channel_autocorrelation(centered, lag_samples(0.100)),
            "median_autocorrelation_500ms": _median_channel_autocorrelation(centered, lag_samples(0.500)),
            "median_zero_crossing_fraction": float(np.median(zero_crossings)),
        },
        "spatial": {
            "mean_abs_channel_correlation": mean_abs_corr,
            "median_abs_channel_correlation": median_abs_corr,
            "covariance_effective_rank": effective_rank,
            "largest_covariance_component_fraction": first_component_fraction,
        },
        "quality": {
            "flat_channel_fraction_std_lt_0_2uv": float(np.mean(channel_std < 0.2)),
            "extreme_sample_fraction_abs_gt_300uv": float(np.mean(abs_data > 300.0)),
        },
        "evidence_boundary": (
            "Observable sensor-space audit only. Matching these features does not establish physiological equivalence or human performance."
        ),
    }


def flatten_audit_metrics(audit: dict) -> dict[str, float]:
    """Flatten numeric audit leaves for declared distance/weighting methods."""
    if audit.get("schema") != AUDIT_SCHEMA:
        raise ValueError(f"expected audit schema {AUDIT_SCHEMA!r}")
    flattened: dict[str, float] = {}
    for section in ("amplitude", "spectrum", "temporal", "spatial", "quality"):
        values = audit.get(section, {})
        for key, value in values.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                flattened[f"{section}.{key}"] = float(value)
    return flattened

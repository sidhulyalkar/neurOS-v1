"""Dependency-light feature signatures for calibrating synthetic data against recordings."""
from __future__ import annotations

import numpy as np


def _band_fraction(power: np.ndarray, frequencies: np.ndarray, low: float, high: float) -> float:
    band = (frequencies >= low) & (frequencies < high)
    total = (frequencies >= 1.0) & (frequencies <= 80.0)
    denominator = float(np.sum(power[:, total]))
    return float(np.sum(power[:, band]) / denominator) if denominator > 0 else 0.0


def feature_signature(data_uv: np.ndarray, sampling_rate_hz: float) -> dict[str, float]:
    """Return coarse observable features; this is not a claim of physiological realism."""
    data = np.asarray(data_uv, dtype=float)
    if data.ndim != 2 or data.shape[1] < 16 or sampling_rate_hz <= 0:
        raise ValueError("expected channels x samples EEG and a positive sample rate")
    centered = data - np.mean(data, axis=1, keepdims=True)
    rms = np.sqrt(np.mean(centered**2, axis=1))
    std = np.std(centered)
    fourth = float(np.mean(centered**4) / max(std**4, 1e-12))
    window = np.hanning(data.shape[1])
    power = np.abs(np.fft.rfft(centered * window[None, :], axis=1)) ** 2
    frequencies = np.fft.rfftfreq(data.shape[1], d=1.0 / sampling_rate_hz)
    corr = np.corrcoef(centered)
    upper = np.abs(corr[np.triu_indices(corr.shape[0], 1)]) if corr.shape[0] > 1 else np.asarray([0.0])
    return {
        "median_channel_rms_uv": float(np.median(rms)),
        "alpha_8_13_fraction": _band_fraction(power, frequencies, 8.0, 13.0),
        "beta_13_30_fraction": _band_fraction(power, frequencies, 13.0, 30.0),
        "high_30_80_fraction": _band_fraction(power, frequencies, 30.0, 80.0),
        "mean_abs_channel_correlation": float(np.mean(upper)),
        "fourth_moment_ratio": fourth,
    }


def compare_feature_signatures(reference: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    """Compare observable feature scales without calling the result a realism score."""
    common = sorted(set(reference) & set(candidate))
    if not common:
        raise ValueError("signatures share no features")
    log_ratios = []
    absolute = []
    for key in common:
        ref = float(reference[key])
        cand = float(candidate[key])
        if ref > 0 and cand > 0:
            log_ratios.append(abs(float(np.log(cand / ref))))
        else:
            absolute.append(abs(cand - ref))
    return {
        "features_compared": float(len(common)),
        "mean_abs_log_ratio": float(np.mean(log_ratios)) if log_ratios else 0.0,
        "max_abs_log_ratio": float(np.max(log_ratios)) if log_ratios else 0.0,
        "mean_absolute_difference_nonpositive": float(np.mean(absolute)) if absolute else 0.0,
    }

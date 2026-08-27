"""Causal, stateful filtering utilities for live neural streams.

The live neurOS processing path treats transport chunking as an implementation
detail, not part of the scientific signal. Therefore filters in this module
carry their causal state across ``apply()`` calls and expose explicit ``reset()``
methods for acquisition/session boundaries.

Samples are always interpreted along the last array axis. Leading dimensions
identify independent streams/channels and are bound to filter state after the
first non-empty block. A geometry change requires an explicit reset so state is
never silently reassigned to different channels.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from scipy.signal import butter, sosfilt


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _stream_array(data: np.ndarray | object) -> np.ndarray:
    array = np.asarray(data)
    if array.ndim < 1:
        raise ValueError("filter input must have a sample axis")
    if not (np.issubdtype(array.dtype, np.number) and not np.issubdtype(array.dtype, np.complexfloating)):
        raise ValueError("filter input must contain real numeric samples")
    return np.asarray(array, dtype=np.result_type(array.dtype, np.float64))


class BandpassFilter:
    """Causal Butterworth bandpass filter with persistent streaming state.

    Parameters
    ----------
    lowcut : float
        Low cutoff frequency in Hz, strictly greater than zero.
    highcut : float
        High cutoff frequency in Hz, strictly below Nyquist.
    fs : float
        Sampling frequency in Hz.
    order : int, optional
        Butterworth prototype order. Defaults to 4.

    Notes
    -----
    The implementation uses second-order sections for numerical robustness and
    zero initial state. The same ordered sample stream therefore produces the
    same output, within floating-point tolerance, whether delivered as one block
    or many blocks. ``reset()`` returns the filter to that deterministic initial
    state.
    """

    def __init__(self, lowcut: float, highcut: float, fs: float, order: int = 4) -> None:
        self.lowcut = _finite_real(lowcut, name="lowcut")
        self.highcut = _finite_real(highcut, name="highcut")
        self.fs = _finite_real(fs, name="fs")
        self.order = _positive_integer(order, name="order")
        if self.fs <= 0:
            raise ValueError("fs must be positive")
        nyquist = 0.5 * self.fs
        if not 0.0 < self.lowcut < self.highcut < nyquist:
            raise ValueError(
                "bandpass cutoffs must satisfy 0 < lowcut < highcut < Nyquist "
                f"({nyquist:g} Hz)"
            )
        self.sos = butter(
            self.order,
            [self.lowcut, self.highcut],
            btype="bandpass",
            fs=self.fs,
            output="sos",
        )
        self._state: np.ndarray | None = None
        self._leading_shape: tuple[int, ...] | None = None

    @property
    def nyquist_hz(self) -> float:
        return 0.5 * self.fs

    @property
    def initialized(self) -> bool:
        return self._state is not None

    def reset(self) -> None:
        """Forget stream history and return to deterministic zero initial state."""
        self._state = None
        self._leading_shape = None

    def _ensure_state(self, array: np.ndarray) -> None:
        leading_shape = tuple(array.shape[:-1])
        if self._leading_shape is not None and leading_shape != self._leading_shape:
            raise ValueError(
                "bandpass stream geometry changed from "
                f"{self._leading_shape} to {leading_shape}; call reset() before reusing "
                "the filter for a different channel/stream geometry"
            )
        if self._state is None:
            self._leading_shape = leading_shape
            state_shape = (self.sos.shape[0], *leading_shape, 2)
            self._state = np.zeros(
                state_shape,
                dtype=np.result_type(array.dtype, self.sos.dtype, np.float64),
            )

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Filter a live block along the last axis while preserving causal state."""
        array = _stream_array(data)
        if array.shape[-1] == 0:
            return array.copy()
        self._ensure_state(array)
        assert self._state is not None
        filtered, final_state = sosfilt(self.sos, array, axis=-1, zi=self._state)
        self._state = final_state
        return np.asarray(filtered)


class SmoothingFilter:
    """Causal moving-average filter with persistent trailing history.

    Unlike centered ``mode='same'`` convolution, each output sample depends only
    on the current and preceding samples. Startup history is deterministic zero
    state. This makes online output independent of transport chunk boundaries
    and avoids hidden within-block look-ahead.
    """

    def __init__(self, window_size: int = 5) -> None:
        self.window_size = _positive_integer(window_size, name="window_size")
        self.kernel = np.ones(self.window_size, dtype=float) / float(self.window_size)
        self._history: np.ndarray | None = None
        self._leading_shape: tuple[int, ...] | None = None

    @property
    def initialized(self) -> bool:
        return self._leading_shape is not None

    def reset(self) -> None:
        """Forget trailing samples and restore deterministic zero startup history."""
        self._history = None
        self._leading_shape = None

    def _ensure_history(self, array: np.ndarray) -> None:
        leading_shape = tuple(array.shape[:-1])
        if self._leading_shape is not None and leading_shape != self._leading_shape:
            raise ValueError(
                "smoothing stream geometry changed from "
                f"{self._leading_shape} to {leading_shape}; call reset() before reusing "
                "the filter for a different channel/stream geometry"
            )
        if self._leading_shape is None:
            self._leading_shape = leading_shape
            self._history = np.zeros(
                (*leading_shape, self.window_size - 1),
                dtype=np.result_type(array.dtype, np.float64),
            )

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply a causal moving average along the last axis."""
        array = _stream_array(data)
        if array.shape[-1] == 0:
            return array.copy()
        self._ensure_history(array)
        if self.window_size == 1:
            return array.copy()
        assert self._history is not None
        combined = np.concatenate((self._history, array), axis=-1)
        result = np.apply_along_axis(
            lambda values: np.convolve(values, self.kernel, mode="valid"),
            axis=-1,
            arr=combined,
        )
        self._history = combined[..., -(self.window_size - 1) :].copy()
        return np.asarray(result)

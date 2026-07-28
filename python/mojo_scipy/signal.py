from __future__ import annotations

import numpy as np

from . import fft as _fft
from ._lib import addr, f64, lib, require_1d


def _apply_mode(full: np.ndarray, n: int, m: int, mode: str) -> np.ndarray:
    if mode == "full":
        return full
    if mode == "same":
        start = (m - 1) // 2
        return full[start : start + n]
    if mode == "valid":
        start = min(n, m) - 1
        return full[start : start + abs(n - m) + 1]
    raise ValueError("mode must be 'full', 'same', or 'valid'")


def convolve(in1, in2, mode="full", method="auto"):
    a = require_1d(in1, "convolve")
    b = require_1d(in2, "convolve")
    if not len(a) or not len(b):
        raise ValueError("a cannot be empty")
    if method not in {"auto", "direct", "fft"}:
        raise ValueError("Acceptable method flags are 'auto', 'direct', or 'fft'")
    if method == "fft":
        return fftconvolve(a, b, mode=mode)
    full = np.empty(len(a) + len(b) - 1, dtype=np.float64)
    lib().msc_convolve(addr(a), addr(b), addr(full), len(a), len(b))
    return _apply_mode(full, len(a), len(b), mode)


def correlate(in1, in2, mode="full", method="auto"):
    a = require_1d(in1, "correlate")
    b = require_1d(in2, "correlate")
    return convolve(a, np.ascontiguousarray(b[::-1]), mode=mode, method=method)


def fftconvolve(in1, in2, mode="full", axes=None):
    if axes not in (None, -1, (0,), [0]):
        raise NotImplementedError("fftconvolve currently supports one-dimensional arrays")
    a = require_1d(in1, "fftconvolve")
    b = require_1d(in2, "fftconvolve")
    if not len(a) or not len(b):
        raise ValueError("a cannot be empty")
    size = len(a) + len(b) - 1
    fast = 1 << (size - 1).bit_length()
    fa = _fft.rfft(a, n=fast)
    fb = _fft.rfft(b, n=fast)
    full = _fft.irfft(fa * fb, n=fast)[:size]
    return _apply_mode(full, len(a), len(b), mode)


def lfilter(b, a, x, axis=-1, zi=None):
    if axis not in (-1, 0):
        raise NotImplementedError("lfilter currently supports one-dimensional arrays")
    if zi is not None:
        raise NotImplementedError("lfilter state zi is not yet supported")
    b = require_1d(b, "lfilter")
    a = require_1d(a, "lfilter")
    x = require_1d(x, "lfilter")
    if not len(b):
        raise ValueError("b must be non-empty")
    if not len(a) or a[0] == 0:
        raise ValueError("a[0] must be nonzero")
    if not len(x):
        raise ValueError("x must be non-empty")
    result = np.empty_like(x)
    lib().msc_lfilter(addr(b), addr(a), addr(x), addr(result), len(b), len(a), len(x))
    return result

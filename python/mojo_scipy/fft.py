from __future__ import annotations

import math

import numpy as np

from ._lib import addr, c128, lib


def _prepare(x, n, axis):
    value = c128(x)
    if value.ndim != 1 or axis not in (-1, 0):
        raise NotImplementedError("FFT kernels currently support one-dimensional arrays")
    if n is None:
        n = len(value)
    if n < 1:
        raise ValueError(f"invalid number of data points ({n}) specified")
    if n == len(value):
        return value
    prepared = np.zeros(n, dtype=np.complex128)
    prepared[: min(n, len(value))] = value[:n]
    return prepared


def _scale(result, n, inverse, norm):
    if norm in (None, "backward"):
        return result
    if norm == "forward":
        return result * (n if inverse else 1.0 / n)
    if norm == "ortho":
        return result * (math.sqrt(n) if inverse else 1.0 / math.sqrt(n))
    raise ValueError(f'Invalid norm value {norm!r}')


def fft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None):
    if plan is not None:
        raise NotImplementedError("precomputed FFT plans are not supported")
    source = _prepare(x, n, axis)
    result = np.empty_like(source)
    lib().msc_fft(addr(source), addr(result), len(source), 0)
    return _scale(result, len(source), False, norm)


def ifft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None):
    if plan is not None:
        raise NotImplementedError("precomputed FFT plans are not supported")
    source = _prepare(x, n, axis)
    result = np.empty_like(source)
    lib().msc_fft(addr(source), addr(result), len(source), 1)
    return _scale(result, len(source), True, norm)


def rfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None):
    value = np.asarray(x)
    if np.iscomplexobj(value):
        raise TypeError("x must be a real sequence")
    size = len(value) if n is None else n
    return fft(value, n=n, axis=axis, norm=norm, workers=workers, plan=plan)[: size // 2 + 1]


def irfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None):
    half = c128(x)
    if half.ndim != 1 or axis not in (-1, 0):
        raise NotImplementedError("irfft currently supports one-dimensional arrays")
    if n is None:
        n = 2 * (len(half) - 1)
    if n < 1:
        raise ValueError(f"invalid number of data points ({n}) specified")
    spectrum = np.zeros(n, dtype=np.complex128)
    take = min(len(half), n // 2 + 1)
    spectrum[:take] = half[:take]
    upper = (n + 1) // 2
    for k in range(1, upper):
        if k < take:
            spectrum[n - k] = np.conjugate(spectrum[k])
    return ifft(spectrum, n=n, axis=axis, norm=norm, workers=workers, plan=plan).real


def fftfreq(n, d=1.0, *, xp=None, device=None):
    if xp not in (None, np) or device is not None:
        raise NotImplementedError("array API device output is not supported")
    return np.fft.fftfreq(n, d)


def rfftfreq(n, d=1.0, *, xp=None, device=None):
    if xp not in (None, np) or device is not None:
        raise NotImplementedError("array API device output is not supported")
    return np.fft.rfftfreq(n, d)


def fftshift(x, axes=None):
    return np.fft.fftshift(x, axes=axes)


def ifftshift(x, axes=None):
    return np.fft.ifftshift(x, axes=axes)

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB_PATH = os.path.join(ROOT, "dist", "libmojo-scipy.so")
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "msc_convolve": ([I, I, I, I, I], None),
    "msc_lfilter": ([I, I, I, I, I, I, I], None),
    "msc_fft": ([I, I, I, I], None),
    "msc_lu_solve": ([I, I, I, I], I),
    "msc_cholesky": ([I, I], I),
    "msc_triangular_solve": ([I, I, I, I, I, I, I], None),
    "msc_linear_assignment": ([I] * 10, None),
    "msc_rosen": ([I, I], F),
    "msc_rosen_der": ([I, I, I], None),
    "msc_interp_linear": ([I, I, I, I, I, I], None),
    "msc_interp_nearest": ([I, I, I, I, I, I], None),
    "msc_pchip_eval": ([I, I, I, I, I, I, I], None),
    "msc_stats_summary": ([I, I, I, I], I),
    "msc_zscore": ([I, I, I, F, F], None),
    "msc_raw_moment": ([I, I, I, F, I], F),
    "msc_entropy": ([I, I, I, I], F),
    "msc_rankdata": ([I, I, I, I, I], None),
}

_library: ctypes.CDLL | None = None


def build() -> str:
    source = os.path.join(ROOT, "src", "kernels.mojo")
    stale = (
        not os.path.exists(LIB_PATH)
        or os.path.getmtime(LIB_PATH) < os.path.getmtime(source)
    )
    if stale:
        subprocess.run(
            ["bash", os.path.join(ROOT, "build", "build.sh")],
            cwd=ROOT,
            check=True,
        )
    return LIB_PATH


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (args, result) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = args
            fn.restype = result
    return _library


def f64(value, *, copy: bool = False) -> np.ndarray:
    source = np.asarray(value)
    if np.iscomplexobj(source):
        raise TypeError("complex inputs are not supported by float64 kernels")
    if source.dtype.kind == "f" and source.dtype.itemsize > 8:
        raise TypeError("floating-point inputs wider than float64 are not supported")
    if copy:
        return np.array(value, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(value, dtype=np.float64)


def c128(value) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind == "c" and source.dtype.itemsize > 16:
        raise TypeError("complex inputs wider than complex128 are not supported")
    if source.dtype.kind == "f" and source.dtype.itemsize > 8:
        raise TypeError("floating-point inputs wider than float64 are not supported")
    return np.ascontiguousarray(value, dtype=np.complex128)


def addr(value: np.ndarray) -> int:
    if not isinstance(value, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if not value.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous")
    if value.dtype not in (np.dtype(np.float64), np.dtype(np.complex128), np.dtype(np.int64)):
        raise TypeError(f"unsupported FFI buffer dtype: {value.dtype}")
    if value.size == 0 or value.ctypes.data == 0:
        raise ValueError("empty buffers must not cross the FFI boundary")
    return int(value.ctypes.data)


def require_1d(value, name: str, *, copy: bool = False) -> np.ndarray:
    result = f64(value, copy=copy)
    if result.ndim != 1:
        raise NotImplementedError(f"{name} currently supports one-dimensional arrays")
    return result

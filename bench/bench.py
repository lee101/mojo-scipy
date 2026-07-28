from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from mojo_scipy import fft, interpolate, linalg, optimize, signal, stats  # noqa: E402
from scipy import fft as scipy_fft  # noqa: E402
from scipy import interpolate as scipy_interpolate  # noqa: E402
from scipy import linalg as scipy_linalg  # noqa: E402
from scipy import optimize as scipy_optimize  # noqa: E402
from scipy import signal as scipy_signal  # noqa: E402
from scipy import stats as scipy_stats  # noqa: E402


def best_time(fn, repeats=5):
    best = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def cases():
    rng = np.random.default_rng(123)

    x_filter = np.ascontiguousarray(rng.normal(size=2_000_000))
    b = np.array([0.02, 0.06, 0.12, 0.2, 0.2, 0.2, 0.12, 0.06, 0.02])
    yield (
        "signal.lfilter (2M, 9-tap FIR)",
        lambda: signal.lfilter(b, [1.0], x_filter),
        lambda: scipy_signal.lfilter(b, [1.0], x_filter),
    )

    x_fft = np.ascontiguousarray(
        rng.normal(size=262_144) + 1j * rng.normal(size=262_144)
    )
    yield (
        "fft.fft (262144 complex128)",
        lambda: fft.fft(x_fft),
        lambda: scipy_fft.fft(x_fft),
    )

    raw = rng.normal(size=(192, 192))
    matrix = np.ascontiguousarray(raw @ raw.T + np.eye(192))
    rhs = np.ascontiguousarray(rng.normal(size=(192, 8)))
    yield (
        "linalg.solve (192x192, 8 RHS)",
        lambda: linalg.solve(matrix, rhs),
        lambda: scipy_linalg.solve(matrix, rhs),
    )

    cost = np.ascontiguousarray(rng.normal(size=(600, 600)))
    yield (
        "optimize.linear_sum_assignment (600x600)",
        lambda: optimize.linear_sum_assignment(cost),
        lambda: scipy_optimize.linear_sum_assignment(cost),
    )

    grid = np.linspace(0, 100, 20_000)
    values = np.sin(grid) + 0.1 * np.cos(7 * grid)
    query = np.ascontiguousarray(rng.uniform(0, 100, size=1_000_000))
    ours_interp = interpolate.PchipInterpolator(grid, values)
    scipy_interp = scipy_interpolate.PchipInterpolator(grid, values)
    yield (
        "PchipInterpolator eval (1M queries)",
        lambda: ours_interp(query),
        lambda: scipy_interp(query),
    )

    sample = np.ascontiguousarray(rng.normal(size=5_000_000))
    yield (
        "stats.zscore (5M float64)",
        lambda: stats.zscore(sample),
        lambda: scipy_stats.zscore(sample),
    )


def main():
    print(f"Machine: {cpu_name()}; {platform.system()} {platform.machine()}")
    print()
    print("| Kernel | Mojo (ms) | SciPy (ms) | SciPy / Mojo | Result |")
    print("|---|---:|---:|---:|---|")
    for name, ours, upstream in cases():
        ours()
        upstream()
        mojo_time = best_time(ours)
        scipy_time = best_time(upstream)
        ratio = scipy_time / mojo_time
        result = "faster" if ratio > 1 else "slower"
        print(
            f"| {name} | {mojo_time * 1000:.2f} | {scipy_time * 1000:.2f} "
            f"| {ratio:.2f}x | {result} |"
        )


if __name__ == "__main__":
    main()

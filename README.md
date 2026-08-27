# mojo-scipy

`mojo-scipy` is a standalone Mojo port of compute-heavy kernels from selected
parts of SciPy. It presents familiar Python modules and function signatures
through `mojo_scipy`, while the numerical loops execute in a single compiled
Mojo shared library.

This is a useful subset, not a replacement for SciPy. The current port focuses
on one-dimensional real `float64` data, one-dimensional `complex128` FFT data,
and dense real `float64` matrices. Python sequences, integer arrays, and
narrower real arrays are converted to those types. Wider floating-point and
complex inputs to real-only kernels are rejected rather than silently narrowed.

## Coverage

| Module | Covered API |
|---|---|
| `signal` | `convolve`, `correlate`, `fftconvolve`, `lfilter` |
| `fft` | `fft`, `ifft`, `rfft`, `irfft`, `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift` |
| `linalg` | `solve`, `cholesky`, `solve_triangular`, `cho_factor`, `cho_solve` |
| `optimize` | `linear_sum_assignment`, `rosen`, `rosen_der` |
| `interpolate` | `interp1d` with linear/nearest interpolation, `PchipInterpolator` |
| `stats` | `moment`, `skew`, `kurtosis`, `variation`, `sem`, `zscore`, `describe`, `entropy`, `rankdata` |

The FFT uses an iterative radix-2 transform and a correct direct DFT fallback
for other lengths. Linear solves use LU decomposition with partial pivoting,
Cholesky validates positive definiteness, assignment uses the rectangular
Hungarian algorithm, and PCHIP uses the same shape-preserving slope rules as
SciPy.

Not covered are multidimensional kernels, sparse matrices, complex linear
algebra, general interpolation axes, callback-driven optimizers, probability
distributions, signal filter state (`zi`), PCHIP derivatives, FFT plans,
non-NumPy array backends, and the rest of SciPy. The package does not promise
drop-in compatibility outside the table above. Explicitly unsupported modes
raise `NotImplementedError`; invalid shapes, domains, and dtypes raise an
appropriate Python exception before entering Mojo.

## Install

The repository pins the tested Mojo nightly and includes SciPy for parity
testing:

```bash
pixi install
pixi run build
pixi run test
```

The build produces `dist/libmojo-scipy.so`. Pixi configures `PYTHONPATH` for
the package in `python/`.

## Usage

```python
import numpy as np
from mojo_scipy import interpolate, linalg, signal, stats

x = np.linspace(0.0, 4.0, 200)
smoothed = signal.lfilter(np.ones(7) / 7, [1.0], np.sin(x))

a = np.array([[4.0, 1.0], [1.0, 3.0]])
solution = linalg.solve(a, np.array([1.0, 2.0]))

curve = interpolate.PchipInterpolator(x, smoothed)
values = curve(np.array([0.25, 1.5, 3.75]))
standardized = stats.zscore(values)

print(solution)
print(standardized)
```

Run it inside the environment with `pixi run python example.py`.

## Benchmarks

Measured by the final `pixi run bench` run on this machine. The script reports
the best of five warm runs and holds a machine-wide file lock. The machine was
an Intel Xeon E5-2697 v4 at 2.30 GHz running Linux x86_64. `SciPy / Mojo` above
one means Mojo was faster in this run.

| Kernel | Mojo (ms) | SciPy (ms) | SciPy / Mojo | Result |
|---|---:|---:|---:|---|
| signal.lfilter (2M, 9-tap FIR) | 16.61 | 20.78 | 1.25x | faster |
| fft.fft (262144 complex128) | 10.11 | 7.59 | 0.75x | slower |
| linalg.solve (192x192, 8 RHS) | 1.14 | 0.73 | 0.64x | slower |
| optimize.linear_sum_assignment (600x600) | 14.63 | 13.46 | 0.92x | slower |
| PchipInterpolator eval (1M queries) | 95.27 | 144.04 | 1.51x | faster |
| stats.zscore (5M float64) | 49.75 | 194.48 | 3.91x | faster |

SciPy remains faster where its mature PocketFFT and LAPACK/BLAS implementations
dominate. SIMD potential updates and compact augmenting-path bookkeeping bring
the assignment kernel close to parity. Mojo does best here on fused,
memory-oriented kernels that avoid temporary arrays and Python-level passes.

No GPU path is included. The benchmarked LU problem is too small to amortize
device transfer and launch costs, while radix-2 FFT stages require global
synchronization. The direct DFT fallback has high nominal arithmetic intensity,
but offloading its O(n²) algorithm would not be competitive with an optimized
O(n log n) CPU FFT. CPU remains the default and only execution path.

## How it works

`src/kernels.mojo` is one compilation unit exported as a C ABI shared library.
Python loads it with `ctypes`. NumPy buffer addresses cross the C ABI as
integers and are reconstructed inside each export as mutable `UnsafePointer`s.

Arrays are caller-owned: Python converts inputs to C-contiguous row-major
`float64` or interleaved `complex128`, allocates outputs and scratch space, and
validates non-empty buffers, lengths, layout, and dtype before a call. Local
references keep every buffer alive until the synchronous call returns. Mojo
never owns or frees Python memory. FFT complex values use NumPy's adjacent
real/imaginary layout.

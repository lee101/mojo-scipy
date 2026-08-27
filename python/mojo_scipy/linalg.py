from __future__ import annotations

import numpy as np

from ._lib import addr, f64, lib


def _matrix(a, check_finite, *, copy=True):
    value = f64(a, copy=copy)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("expected a square matrix")
    if check_finite and not np.isfinite(value).all():
        raise ValueError("array must not contain infs or NaNs")
    return value


def _rhs(b, n, check_finite, *, copy=True):
    value = f64(b, copy=copy)
    was_vector = value.ndim == 1
    if was_vector:
        value = np.ascontiguousarray(value.reshape(n, 1))
    if value.ndim != 2 or value.shape[0] != n:
        raise ValueError("incompatible dimensions")
    if check_finite and not np.isfinite(value).all():
        raise ValueError("array must not contain infs or NaNs")
    return value, was_vector


def solve(
    a,
    b,
    lower=False,
    overwrite_a=False,
    overwrite_b=False,
    check_finite=True,
    assume_a=None,
    transposed=False,
):
    matrix = _matrix(a, check_finite, copy=not overwrite_a)
    if transposed:
        matrix = np.ascontiguousarray(matrix.T)
    rhs, was_vector = _rhs(b, len(matrix), check_finite, copy=not overwrite_b)
    if matrix.size == 0 or rhs.size == 0:
        return rhs[:, 0] if was_vector else rhs
    ok = lib().msc_lu_solve(addr(matrix), addr(rhs), len(matrix), rhs.shape[1])
    if not ok:
        raise np.linalg.LinAlgError("Matrix is singular.")
    return rhs[:, 0] if was_vector else rhs


def cholesky(a, lower=False, overwrite_a=False, check_finite=True):
    matrix = _matrix(a, check_finite, copy=not overwrite_a)
    if matrix.size == 0:
        return matrix
    if not lib().msc_cholesky(addr(matrix), len(matrix)):
        raise np.linalg.LinAlgError("leading minor is not positive definite")
    return matrix if lower else np.ascontiguousarray(matrix.T)


def solve_triangular(
    a,
    b,
    trans=0,
    lower=False,
    unit_diagonal=False,
    overwrite_b=False,
    check_finite=True,
):
    matrix = _matrix(a, check_finite, copy=False)
    rhs, was_vector = _rhs(b, len(matrix), check_finite, copy=not overwrite_b)
    if trans in (0, "N", "n"):
        transpose = 0
    elif trans in (1, 2, "T", "t", "C", "c"):
        transpose = 1
    else:
        raise ValueError("invalid trans")
    if matrix.size == 0 or rhs.size == 0:
        return rhs[:, 0] if was_vector else rhs
    if not unit_diagonal and np.any(np.diag(matrix) == 0):
        raise np.linalg.LinAlgError("singular matrix: zero diagonal")
    lib().msc_triangular_solve(
        addr(matrix), addr(rhs), len(matrix), rhs.shape[1],
        int(lower), transpose, int(unit_diagonal),
    )
    return rhs[:, 0] if was_vector else rhs


def cho_factor(a, lower=False, overwrite_a=False, check_finite=True):
    return cholesky(a, lower=lower, check_finite=check_finite), lower


def cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True):
    factor, lower = c_and_lower
    factor = _matrix(factor, check_finite)
    if lower:
        first = solve_triangular(
            factor, b, lower=True, check_finite=check_finite
        )
        return solve_triangular(
            factor, first, lower=True, trans=1, check_finite=check_finite
        )
    first = solve_triangular(
        factor, b, lower=False, trans=1, check_finite=check_finite
    )
    return solve_triangular(
        factor, first, lower=False, check_finite=check_finite
    )

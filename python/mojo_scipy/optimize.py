from __future__ import annotations

import numpy as np

from ._lib import addr, f64, lib, require_1d


def linear_sum_assignment(cost_matrix, maximize=False):
    cost = f64(cost_matrix)
    if cost.ndim != 2:
        raise ValueError("expected a matrix (2-D array)")
    if not np.isfinite(cost).all():
        raise ValueError("matrix contains invalid numeric entries")
    transposed = cost.shape[0] > cost.shape[1]
    if transposed:
        cost = np.ascontiguousarray(cost.T)
    n, m = cost.shape
    if n == 0 or m == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty.copy(), empty
    if maximize:
        cost = np.ascontiguousarray(cost.max() - cost)
    cols = np.empty(n, dtype=np.int64)
    u = np.empty(n + 1, dtype=np.float64)
    v = np.empty(m + 1, dtype=np.float64)
    p = np.empty(m + 1, dtype=np.int64)
    way = np.empty(m + 1, dtype=np.int64)
    minv = np.empty(m + 1, dtype=np.float64)
    used = np.empty(m + 1, dtype=np.int64)
    lib().msc_linear_assignment(
        addr(cost), addr(cols), addr(u), addr(v), addr(p), addr(way),
        addr(minv), addr(used), n, m,
    )
    rows = np.arange(n, dtype=np.int64)
    if transposed:
        order = np.argsort(cols)
        return cols[order], rows[order]
    return rows, cols


def rosen(x):
    value = require_1d(x, "rosen")
    if not len(value):
        return 0.0
    return lib().msc_rosen(addr(value), len(value))


def rosen_der(x):
    value = require_1d(x, "rosen_der")
    if not len(value):
        raise IndexError("rosen_der requires at least one value")
    result = np.empty_like(value)
    lib().msc_rosen_der(addr(value), addr(result), len(value))
    return result

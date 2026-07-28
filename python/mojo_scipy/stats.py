from __future__ import annotations

import math
from collections import namedtuple

import numpy as np

from ._lib import addr, f64, lib, require_1d

DescribeResult = namedtuple(
    "DescribeResult", "nobs minmax mean variance skewness kurtosis"
)


def _policy(nan_policy):
    if nan_policy == "raise":
        return "raise"
    if nan_policy in ("propagate", "omit"):
        return nan_policy
    raise ValueError("nan_policy must be 'propagate', 'raise', or 'omit'")


def _summary(a, axis, nan_policy):
    if axis not in (None, 0, -1):
        raise NotImplementedError("statistics currently support one-dimensional arrays")
    value = require_1d(a, "statistics")
    policy = _policy(nan_policy)
    if policy == "raise" and np.isnan(value).any():
        raise ValueError("The input contains nan values")
    result = np.empty(4, dtype=np.float64)
    if not len(value):
        result.fill(np.nan)
        return value, result, 0
    count = lib().msc_stats_summary(
        addr(value), addr(result), len(value), int(policy == "omit")
    )
    return value, result, count


def moment(a, moment=1, axis=0, nan_policy="propagate", *, center=None):
    value = require_1d(a, "moment")
    policy = _policy(nan_policy)
    if policy == "raise" and np.isnan(value).any():
        raise ValueError("The input contains nan values")
    orders = np.asarray(moment)
    scalar = orders.ndim == 0
    orders = np.atleast_1d(orders)
    if not np.issubdtype(orders.dtype, np.integer) or np.any(orders < 0):
        raise ValueError("moment must contain non-negative integers")
    if center is None:
        _, summary, _ = _summary(value, axis, policy)
        center = summary[0]
    results = np.array([
        np.nan if not len(value) else lib().msc_raw_moment(
            addr(value), len(value), int(order), float(center), int(policy == "omit")
        )
        for order in orders
    ])
    return float(results[0]) if scalar else results


def skew(a, axis=0, bias=True, nan_policy="propagate"):
    _, result, n = _summary(a, axis, nan_policy)
    m2, m3 = result[1], result[2]
    if m2 == 0:
        return np.nan
    value = m3 / m2 ** 1.5
    if not bias and n > 2:
        value *= math.sqrt(n * (n - 1)) / (n - 2)
    return value


def kurtosis(a, axis=0, fisher=True, bias=True, nan_policy="propagate"):
    _, result, n = _summary(a, axis, nan_policy)
    m2, m4 = result[1], result[3]
    if m2 == 0:
        return np.nan
    excess = m4 / (m2 * m2) - 3.0
    if not bias and n > 3:
        excess = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * excess + 6)
    return excess if fisher else excess + 3.0


def variation(a, axis=0, nan_policy="propagate", ddof=0, *, keepdims=False):
    _, result, n = _summary(a, axis, nan_policy)
    variance = result[1] * n / (n - ddof)
    return math.sqrt(variance) / result[0]


def sem(a, axis=0, ddof=1, nan_policy="propagate"):
    _, result, n = _summary(a, axis, nan_policy)
    return math.sqrt(result[1] * n / (n - ddof)) / math.sqrt(n)


def zscore(a, axis=0, ddof=0, nan_policy="propagate"):
    value, result, n = _summary(a, axis, nan_policy)
    scale = math.sqrt(result[1] * n / (n - ddof))
    output = np.empty_like(value)
    lib().msc_zscore(addr(value), addr(output), len(value), result[0], scale)
    return output


def describe(a, axis=0, ddof=1, bias=True, nan_policy="propagate"):
    value, result, n = _summary(a, axis, nan_policy)
    selected = value[~np.isnan(value)] if nan_policy == "omit" else value
    variance = result[1] * n / (n - ddof)
    return DescribeResult(
        n,
        (selected.min(), selected.max()),
        result[0],
        variance,
        skew(selected, axis=0, bias=bias),
        kurtosis(selected, axis=0, fisher=True, bias=bias),
    )


def entropy(pk, qk=None, base=None, axis=0):
    if axis not in (None, 0, -1):
        raise NotImplementedError("entropy currently supports one-dimensional arrays")
    p = require_1d(pk, "entropy")
    q = p if qk is None else require_1d(qk, "entropy")
    if len(q) != len(p):
        raise ValueError("Array shapes are incompatible for broadcasting.")
    if not len(p):
        return 0.0
    if base is not None and base <= 0:
        raise ValueError("base must be a positive number or None")
    if np.isnan(p).any() or np.isposinf(p).any():
        return np.nan
    if np.any(p < 0):
        return -np.inf
    if p.sum() == 0:
        return np.nan
    if qk is not None and (
        np.isnan(q).any() or np.any(q < 0) or not np.isfinite(q.sum()) or q.sum() == 0
    ):
        return np.nan
    value = lib().msc_entropy(addr(p), addr(q), len(p), int(qk is not None))
    if base is None:
        return value
    if base == 1:
        raise ValueError("base must not be 1")
    return value / math.log(base)


def rankdata(a, method="average", *, axis=None, nan_policy="propagate"):
    if axis is not None:
        raise NotImplementedError("rankdata currently supports axis=None")
    value = np.asarray(a, dtype=np.float64)
    flat = np.ascontiguousarray(value.ravel())
    policy = _policy(nan_policy)
    nan_mask = np.isnan(flat)
    if policy == "raise" and nan_mask.any():
        raise ValueError("The input contains nan values")
    if policy == "propagate" and nan_mask.any():
        return np.full(value.size, np.nan)
    valid = np.ascontiguousarray(flat[~nan_mask], dtype=np.float64)
    work = valid.copy()
    indices = np.empty(len(valid), dtype=np.int64)
    ranked = np.empty(len(valid), dtype=np.float64)
    methods = {"average": 0, "min": 1, "max": 2, "dense": 3, "ordinal": 4}
    if method not in methods:
        raise ValueError(f"unknown method {method!r}")
    if len(valid):
        lib().msc_rankdata(
            addr(work), addr(indices), addr(ranked), len(valid), methods[method]
        )
    result = np.full(len(flat), np.nan)
    result[~nan_mask] = ranked
    return result

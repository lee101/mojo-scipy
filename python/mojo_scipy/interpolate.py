from __future__ import annotations

import numpy as np

from ._lib import addr, lib, require_1d


def _xy(x, y, assume_sorted=False):
    x = require_1d(x, "interpolation")
    y = require_1d(y, "interpolation")
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("x and y arrays must be equal in length and contain at least 2 points")
    if not assume_sorted:
        order = np.argsort(x)
        x = np.ascontiguousarray(x[order])
        y = np.ascontiguousarray(y[order])
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")
    return x, y


class interp1d:
    def __init__(
        self,
        x,
        y,
        kind="linear",
        axis=-1,
        copy=True,
        bounds_error=None,
        fill_value=np.nan,
        assume_sorted=False,
    ):
        if axis not in (-1, 0):
            raise NotImplementedError("interp1d currently supports one-dimensional y")
        if kind not in ("linear", "nearest"):
            raise NotImplementedError("supported interp1d kinds are 'linear' and 'nearest'")
        self.x, self.y = _xy(x, y, assume_sorted)
        self.kind = kind
        self.bounds_error = fill_value != "extrapolate" if bounds_error is None else bounds_error
        self.fill_value = fill_value

    def __call__(self, x_new):
        shape = np.shape(x_new)
        query = np.ascontiguousarray(np.asarray(x_new, dtype=np.float64).ravel())
        if self.bounds_error and (
            np.any(query < self.x[0]) or np.any(query > self.x[-1])
        ):
            raise ValueError("A value in x_new is outside the interpolation range.")
        result = np.empty_like(query)
        if not len(query):
            return result.reshape(shape)
        fn = lib().msc_interp_linear if self.kind == "linear" else lib().msc_interp_nearest
        fn(addr(self.x), addr(self.y), addr(query), addr(result), len(self.x), len(query))
        if self.fill_value != "extrapolate":
            below = query < self.x[0]
            above = query > self.x[-1]
            if isinstance(self.fill_value, tuple):
                result[below] = self.fill_value[0]
                result[above] = self.fill_value[1]
            else:
                result[below | above] = self.fill_value
        result = result.reshape(shape)
        return result.item() if result.ndim == 0 else result


def _edge_slope(h0, h1, m0, m1):
    value = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    if np.sign(value) != np.sign(m0):
        return 0.0
    if np.sign(m0) != np.sign(m1) and abs(value) > 3 * abs(m0):
        return 3 * m0
    return value


class PchipInterpolator:
    def __init__(self, x, y, axis=0, extrapolate=None):
        if axis not in (0, -1):
            raise NotImplementedError("PchipInterpolator currently supports one-dimensional y")
        self.x, self.y = _xy(x, y, True)
        self.extrapolate = True if extrapolate is None else extrapolate
        h = np.diff(self.x)
        slopes = np.diff(self.y) / h
        self._d = np.empty_like(self.y)
        if len(self.x) == 2:
            self._d[:] = slopes[0]
        else:
            self._d[0] = _edge_slope(h[0], h[1], slopes[0], slopes[1])
            self._d[-1] = _edge_slope(h[-1], h[-2], slopes[-1], slopes[-2])
            for i in range(1, len(self.x) - 1):
                left, right = slopes[i - 1], slopes[i]
                if left == 0 or right == 0 or np.sign(left) != np.sign(right):
                    self._d[i] = 0.0
                else:
                    w1 = 2 * h[i] + h[i - 1]
                    w2 = h[i] + 2 * h[i - 1]
                    self._d[i] = (w1 + w2) / (w1 / left + w2 / right)

    def __call__(self, x, nu=0, extrapolate=None):
        if nu != 0:
            raise NotImplementedError("PchipInterpolator derivatives are not yet supported")
        use_extrapolate = self.extrapolate if extrapolate is None else extrapolate
        shape = np.shape(x)
        query = np.ascontiguousarray(np.asarray(x, dtype=np.float64).ravel())
        result = np.empty_like(query)
        if not len(query):
            return result.reshape(shape)
        lib().msc_pchip_eval(
            addr(self.x), addr(self.y), addr(self._d), addr(query), addr(result),
            len(self.x), len(query),
        )
        if not use_extrapolate:
            result[(query < self.x[0]) | (query > self.x[-1])] = np.nan
        result = result.reshape(shape)
        return result.item() if result.ndim == 0 else result

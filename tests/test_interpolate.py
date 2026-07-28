import numpy as np
import pytest
from scipy import interpolate as scipy_interpolate

from mojo_scipy import interpolate


@pytest.mark.parametrize("kind", ["linear", "nearest"])
def test_interp1d_inside_domain(kind):
    rng = np.random.default_rng(7)
    x = np.sort(rng.uniform(-2, 3, size=30))
    y = np.sin(x) + x * 0.1
    q = np.linspace(x[0], x[-1], 301)
    ours = interpolate.interp1d(x, y, kind=kind)
    theirs = scipy_interpolate.interp1d(x, y, kind=kind)
    assert np.allclose(ours(q), theirs(q))


def test_interp1d_fill_and_extrapolation():
    x = np.linspace(0, 2, 8)
    y = np.cos(x)
    q = np.array([-0.2, 0.5, 2.3])
    for fill in ["extrapolate", (-10.0, 20.0)]:
        ours = interpolate.interp1d(x, y, bounds_error=False, fill_value=fill)
        theirs = scipy_interpolate.interp1d(x, y, bounds_error=False, fill_value=fill)
        assert np.allclose(ours(q), theirs(q))


def test_interp1d_default_bounds_error():
    fn = interpolate.interp1d([0, 1], [0, 1])
    with pytest.raises(ValueError):
        fn([-1])


def test_interp1d_nearest_tie_rounds_down():
    ours = interpolate.interp1d([0.0, 2.0], [10.0, 20.0], kind="nearest")
    theirs = scipy_interpolate.interp1d([0.0, 2.0], [10.0, 20.0], kind="nearest")
    assert np.array_equal(ours([1.0]), theirs([1.0]))


def test_pchip_parity_random_and_extrapolation():
    rng = np.random.default_rng(8)
    x = np.sort(rng.uniform(0, 5, size=24))
    y = rng.normal(size=24)
    q = np.linspace(-0.2, 5.2, 401)
    ours = interpolate.PchipInterpolator(x, y)
    theirs = scipy_interpolate.PchipInterpolator(x, y)
    assert np.allclose(ours(q), theirs(q), atol=2e-12)


def test_pchip_no_extrapolation():
    x = np.arange(5.0)
    fn = interpolate.PchipInterpolator(x, x**2, extrapolate=False)
    result = fn([-1, 2, 6])
    assert np.isnan(result[[0, 2]]).all()
    assert result[1] == pytest.approx(4)

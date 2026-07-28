import numpy as np
import pytest
from scipy import signal as scipy_signal

from mojo_scipy import signal


@pytest.mark.parametrize("mode", ["full", "same", "valid"])
def test_convolve_parity(mode):
    rng = np.random.default_rng(1)
    a = rng.normal(size=37)
    b = rng.normal(size=12)
    assert np.allclose(
        signal.convolve(a, b, mode=mode),
        scipy_signal.convolve(a, b, mode=mode, method="direct"),
    )


@pytest.mark.parametrize("mode", ["full", "same", "valid"])
def test_correlate_parity(mode):
    rng = np.random.default_rng(2)
    a = rng.normal(size=12)
    b = rng.normal(size=19)
    assert np.allclose(
        signal.correlate(a, b, mode=mode),
        scipy_signal.correlate(a, b, mode=mode, method="direct"),
    )


@pytest.mark.parametrize("mode", ["full", "same", "valid"])
def test_fftconvolve_parity(mode):
    rng = np.random.default_rng(3)
    a = rng.normal(size=143)
    b = rng.normal(size=61)
    assert np.allclose(
        signal.fftconvolve(a, b, mode=mode),
        scipy_signal.fftconvolve(a, b, mode=mode),
        atol=2e-12,
    )


def test_lfilter_fir_and_iir_parity():
    rng = np.random.default_rng(4)
    x = rng.normal(size=500)
    for b, a in [([0.2, 0.5, 0.2], [1.0]), ([1.0, -0.1], [2.0, -0.7, 0.15])]:
        assert np.allclose(signal.lfilter(b, a, x), scipy_signal.lfilter(b, a, x))


def test_signal_rejects_unsupported_dimensions_and_state():
    with pytest.raises(NotImplementedError):
        signal.convolve(np.ones((2, 2)), np.ones(2))
    with pytest.raises(NotImplementedError):
        signal.lfilter([1], [1], [1, 2], zi=[0])
    with pytest.raises(ValueError):
        signal.fftconvolve([], [1])
    with pytest.raises(ValueError):
        signal.lfilter([1], [1], [])

import numpy as np
import pytest
from scipy import fft as scipy_fft

from mojo_scipy import fft


@pytest.mark.parametrize("n", [1, 2, 8, 32, 7, 15])
@pytest.mark.parametrize("norm", [None, "forward", "ortho"])
def test_fft_ifft_parity(n, norm):
    rng = np.random.default_rng(n)
    x = rng.normal(size=n) + 1j * rng.normal(size=n)
    assert np.allclose(fft.fft(x, norm=norm), scipy_fft.fft(x, norm=norm), atol=2e-12)
    assert np.allclose(fft.ifft(x, norm=norm), scipy_fft.ifft(x, norm=norm), atol=2e-12)


@pytest.mark.parametrize("n", [8, 9, 32])
def test_real_fft_roundtrip_and_parity(n):
    rng = np.random.default_rng(n)
    x = rng.normal(size=n)
    transformed = fft.rfft(x)
    assert np.allclose(transformed, scipy_fft.rfft(x), atol=2e-12)
    assert np.allclose(fft.irfft(transformed, n=n), x, atol=2e-12)


def test_fft_padding_truncation_and_frequencies():
    x = np.arange(9.0)
    assert np.allclose(fft.fft(x, n=16), scipy_fft.fft(x, n=16))
    assert np.allclose(fft.fft(x, n=5), scipy_fft.fft(x, n=5))
    assert np.array_equal(fft.fftfreq(7, 0.25), scipy_fft.fftfreq(7, 0.25))
    assert np.array_equal(fft.rfftfreq(8, 0.25), scipy_fft.rfftfreq(8, 0.25))


def test_fft_shifts_and_empty_errors():
    x = np.arange(7)
    assert np.array_equal(fft.fftshift(x), scipy_fft.fftshift(x))
    assert np.array_equal(fft.ifftshift(x), scipy_fft.ifftshift(x))
    with pytest.raises(ValueError):
        fft.fft([])
    with pytest.raises(ValueError):
        fft.irfft([1.0])

import numpy as np
import pytest
from scipy import stats as scipy_stats

from mojo_scipy import stats


@pytest.fixture
def sample():
    return np.random.default_rng(9).normal(loc=2.0, scale=3.0, size=1001)


@pytest.mark.parametrize("order", [0, 1, 2, 3, 4, 6])
def test_moment_parity(sample, order):
    assert stats.moment(sample, moment=order) == pytest.approx(
        scipy_stats.moment(sample, moment=order), rel=1e-12
    )


@pytest.mark.parametrize("bias", [False, True])
def test_skew_kurtosis_parity(sample, bias):
    assert stats.skew(sample, bias=bias) == pytest.approx(
        scipy_stats.skew(sample, bias=bias)
    )
    assert stats.kurtosis(sample, bias=bias) == pytest.approx(
        scipy_stats.kurtosis(sample, bias=bias)
    )


def test_zscore_sem_variation_parity(sample):
    assert np.allclose(stats.zscore(sample, ddof=1), scipy_stats.zscore(sample, ddof=1))
    assert stats.sem(sample) == pytest.approx(scipy_stats.sem(sample))
    assert stats.variation(sample) == pytest.approx(scipy_stats.variation(sample))


def test_describe_parity(sample):
    ours = stats.describe(sample)
    theirs = scipy_stats.describe(sample)
    assert ours.nobs == theirs.nobs
    assert np.allclose(ours.minmax, theirs.minmax)
    assert ours.mean == pytest.approx(theirs.mean)
    assert ours.variance == pytest.approx(theirs.variance)
    assert ours.skewness == pytest.approx(theirs.skewness)
    assert ours.kurtosis == pytest.approx(theirs.kurtosis)


def test_nan_policies(sample):
    value = sample.copy()
    value[7] = np.nan
    assert stats.skew(value) != stats.skew(value)
    assert stats.skew(value, nan_policy="omit") == pytest.approx(
        scipy_stats.skew(value, nan_policy="omit")
    )
    with pytest.raises(ValueError):
        stats.kurtosis(value, nan_policy="raise")


@pytest.mark.parametrize("method", ["average", "min", "max", "dense", "ordinal"])
def test_rankdata_parity(method):
    value = np.array([[3, 1, 3, 8], [2, 2, -1, 8]], dtype=float)
    assert np.array_equal(
        stats.rankdata(value, method=method),
        scipy_stats.rankdata(value, method=method),
    )


def test_rankdata_nan_omit_and_propagate():
    value = [3.0, np.nan, 1.0, 3.0]
    assert np.allclose(
        stats.rankdata(value, nan_policy="omit"),
        scipy_stats.rankdata(value, nan_policy="omit"),
        equal_nan=True,
    )
    assert np.isnan(stats.rankdata(value)).all()


def test_entropy_parity():
    p = np.array([0.2, 0.3, 0.5])
    q = np.array([0.1, 0.7, 0.2])
    assert stats.entropy(p) == pytest.approx(scipy_stats.entropy(p))
    assert stats.entropy(p, q, base=2) == pytest.approx(scipy_stats.entropy(p, q, base=2))
    for p_bad, q_bad in [([-1, 2], None), ([0, 0], None), ([1, 2], [1, -1])]:
        with np.errstate(all="ignore"):
            expected = scipy_stats.entropy(p_bad, q_bad)
        assert np.allclose(stats.entropy(p_bad, q_bad), expected, equal_nan=True)


def test_empty_statistics_do_not_cross_ffi():
    assert np.isnan(stats.moment([]))
    assert stats.entropy([]) == 0.0
    assert stats.rankdata([]).size == 0


def test_complex_input_is_not_silently_narrowed(sample):
    with pytest.raises(TypeError):
        stats.zscore(sample.astype(np.complex128) + 1j)

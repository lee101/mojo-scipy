import numpy as np
import pytest
from scipy import optimize as scipy_optimize

from mojo_scipy import optimize


@pytest.mark.parametrize("shape", [(8, 8), (5, 11), (12, 6)])
@pytest.mark.parametrize("maximize", [False, True])
def test_linear_sum_assignment_parity(shape, maximize):
    rng = np.random.default_rng(sum(shape))
    cost = rng.normal(size=shape)
    rows, cols = optimize.linear_sum_assignment(cost, maximize=maximize)
    ref_rows, ref_cols = scipy_optimize.linear_sum_assignment(cost, maximize=maximize)
    assert np.array_equal(rows, ref_rows)
    assert cost[rows, cols].sum() == pytest.approx(cost[ref_rows, ref_cols].sum())


def test_linear_sum_assignment_integer_ties():
    cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    rows, cols = optimize.linear_sum_assignment(cost)
    assert np.array_equal(rows, [0, 1, 2])
    assert cost[rows, cols].sum() == 5


def test_rosen_and_derivative_parity():
    x = np.array([1.2, 0.9, -0.3, 2.0, 1.1])
    assert optimize.rosen(x) == pytest.approx(scipy_optimize.rosen(x))
    assert np.allclose(optimize.rosen_der(x), scipy_optimize.rosen_der(x))


def test_empty_assignment_and_rosen_do_not_cross_ffi():
    for shape in ((0, 3), (3, 0)):
        rows, cols = optimize.linear_sum_assignment(np.empty(shape))
        assert rows.size == cols.size == 0
    assert optimize.rosen([]) == 0.0
    with pytest.raises(IndexError):
        optimize.rosen_der([])

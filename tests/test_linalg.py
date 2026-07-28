import numpy as np
import pytest
from scipy import linalg as scipy_linalg

from mojo_scipy import linalg


@pytest.fixture
def matrices():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(9, 9))
    spd = a @ a.T + np.eye(9)
    b = rng.normal(size=(9, 3))
    return a, spd, b


def test_solve_vector_and_multiple_rhs(matrices):
    a, _, b = matrices
    a += 2 * np.eye(9)
    assert np.allclose(linalg.solve(a, b), scipy_linalg.solve(a, b))
    assert np.allclose(linalg.solve(a, b[:, 0]), scipy_linalg.solve(a, b[:, 0]))


def test_solve_simd_tail():
    rng = np.random.default_rng(105)
    a = rng.normal(size=(10, 10)) + 4 * np.eye(10)
    b = rng.normal(size=(10, 5))
    assert np.allclose(linalg.solve(a, b), scipy_linalg.solve(a, b))


def test_solve_transposed(matrices):
    a, _, b = matrices
    a += 2 * np.eye(9)
    assert np.allclose(
        linalg.solve(a, b, transposed=True),
        scipy_linalg.solve(a, b, transposed=True),
    )


@pytest.mark.parametrize("lower", [False, True])
def test_cholesky_parity(matrices, lower):
    _, a, _ = matrices
    assert np.allclose(
        linalg.cholesky(a, lower=lower),
        scipy_linalg.cholesky(a, lower=lower),
    )


@pytest.mark.parametrize("lower,trans", [(True, 0), (False, 0), (True, 1), (False, 1)])
def test_solve_triangular_parity(matrices, lower, trans):
    _, a, b = matrices
    triangular = np.tril(a) if lower else np.triu(a)
    assert np.allclose(
        linalg.solve_triangular(triangular, b, lower=lower, trans=trans),
        scipy_linalg.solve_triangular(triangular, b, lower=lower, trans=trans),
    )


@pytest.mark.parametrize("lower", [False, True])
def test_cho_solve_parity(matrices, lower):
    _, a, b = matrices
    factor = linalg.cho_factor(a, lower=lower)
    assert np.allclose(
        linalg.cho_solve(factor, b),
        scipy_linalg.cho_solve(scipy_linalg.cho_factor(a, lower=lower), b),
    )


def test_singular_and_non_positive_definite_errors():
    with pytest.raises(np.linalg.LinAlgError):
        linalg.solve(np.ones((3, 3)), np.ones(3))
    with pytest.raises(np.linalg.LinAlgError):
        linalg.cholesky([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(np.linalg.LinAlgError):
        linalg.solve_triangular([[0.0, 1.0], [0.0, 2.0]], [1.0, 2.0])


def test_empty_linalg_inputs_do_not_cross_ffi():
    assert linalg.solve(np.empty((0, 0)), np.empty(0)).shape == (0,)
    assert linalg.cholesky(np.empty((0, 0))).shape == (0, 0)

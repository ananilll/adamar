"""
Tests for Hadamard matrix construction and orthogonality.
"""

import numpy as np
import pytest

from wht_forecast.hadamard import build_hadamard_matrix, build_normalized_hadamard


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64, 128])
def test_hadamard_orthogonality(n: int) -> None:
    """Test that A @ A.T == I for normalized Hadamard matrix."""
    A = build_normalized_hadamard(n)
    product = A @ A.T
    identity = np.eye(n)
    np.testing.assert_array_almost_equal(product, identity)


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32])
def test_hadamard_structure(n: int) -> None:
    """Test recursive structure: H_1 = [1], H_{2k} = [[H_k, H_k], [H_k, -H_k]]."""
    H = build_hadamard_matrix(n)
    assert H.shape == (n, n)
    assert np.all(np.isin(H, [-1.0, 1.0]))


def test_hadamard_n_not_power_of_two() -> None:
    """Test that non-power-of-two n raises ValueError."""
    with pytest.raises(ValueError, match="power of 2"):
        build_hadamard_matrix(3)
    with pytest.raises(ValueError, match="power of 2"):
        build_hadamard_matrix(10)


def test_normalized_hadamard_scale() -> None:
    """Test that normalized matrix has correct scaling."""
    n = 8
    H = build_hadamard_matrix(n)
    A = build_normalized_hadamard(n)
    expected = H / np.sqrt(n)
    np.testing.assert_array_almost_equal(A, expected)

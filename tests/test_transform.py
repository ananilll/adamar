"""
Tests for forward and inverse Walsh-Hadamard transform.
"""

import numpy as np
import pytest

from wht_forecast.hadamard import build_normalized_hadamard
from wht_forecast.transform import wht_forward, wht_inverse


@pytest.mark.parametrize("block_size", [8, 16, 32, 64])
def test_transform_inverse_roundtrip(block_size: int) -> None:
    """Test that inverse(forward(x)) == x."""
    A = build_normalized_hadamard(block_size)
    x = np.random.randn(block_size)
    coeffs = wht_forward(x, A)
    x_reconstructed = wht_inverse(coeffs, A)
    np.testing.assert_array_almost_equal(x, x_reconstructed)


@pytest.mark.parametrize("block_size", [8, 16, 32])
def test_transform_preserves_energy(block_size: int) -> None:
    """Test Parseval: ||x||^2 == ||C||^2 for normalized A."""
    A = build_normalized_hadamard(block_size)
    x = np.random.randn(block_size)
    coeffs = wht_forward(x, A)
    energy_x = np.dot(x, x)
    energy_c = np.dot(coeffs, coeffs)
    np.testing.assert_almost_equal(energy_x, energy_c)

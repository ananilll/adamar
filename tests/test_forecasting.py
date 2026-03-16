"""
Tests for forecasting algorithm.
"""

import numpy as np
import pytest

from wht_forecast.forecasting import forecast_next_block
from wht_forecast.synthetic_data import generate_synthetic_series


def test_forecast_output_shape() -> None:
    """Test that forecast has correct shape (block_size)."""
    block_size = 32
    series = generate_synthetic_series(n=256, seed=42)
    forecast, info = forecast_next_block(series, block_size=block_size, top_k=8)
    assert forecast.shape == (block_size,)
    assert len(info["blocks"]) >= 2


def test_forecast_series_too_short() -> None:
    """Test that too short series raises ValueError."""
    series = np.random.randn(16)  # Only 1 block of size 16
    with pytest.raises(ValueError, match="too short"):
        forecast_next_block(series, block_size=16)


@pytest.mark.parametrize("block_size", [8, 16, 32, 64, 128])
def test_forecast_block_sizes(block_size: int) -> None:
    """Test forecasting works for various block sizes."""
    n = max(512, block_size * 4)
    series = generate_synthetic_series(n=n, seed=42)
    forecast, _ = forecast_next_block(series, block_size=block_size, top_k=min(4, block_size))
    assert forecast.shape == (block_size,)


def test_forecast_large_series() -> None:
    """Test algorithm works for series length up to 100k."""
    n = 100_000
    series = generate_synthetic_series(n=n, seed=42)
    forecast, info = forecast_next_block(series, block_size=128, top_k=16)
    assert forecast.shape == (128,)
    assert len(info["blocks"]) == n // 128

"""
Tests for CSV data loading and normalization.
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from wht_forecast.data_loader import (
    load_csv_series,
    load_time_series_from_csv,
    normalize_series,
    validate_series_for_forecasting,
)


@pytest.fixture
def data_dir() -> Path:
    return Path(__file__).parent.parent / "data"


def test_load_csv_timestamp_value(data_dir: Path) -> None:
    """Test loading CSV with timestamp,value format."""
    path = data_dir / "example_timestamp_value.csv"
    if not path.exists():
        pytest.skip("Example data not found")
    series = load_time_series_from_csv(path, verbose=False)
    assert series.ndim == 1
    assert len(series) >= 64
    assert np.issubdtype(series.dtype, np.floating)


def test_load_csv_single_column(data_dir: Path) -> None:
    """Test loading CSV with single value column."""
    path = data_dir / "example_single_column.csv"
    if not path.exists():
        pytest.skip("Example data not found")
    series = load_time_series_from_csv(path, verbose=False)
    assert series.ndim == 1
    assert len(series) >= 64


def test_load_csv_value_column_specified(tmp_path: Path) -> None:
    """Test loading with explicit value column."""
    csv = tmp_path / "test.csv"
    base = datetime(2023, 1, 1)
    lines = ["date,value"] + [
        f"{(base + timedelta(days=i)).strftime('%Y-%m-%d')},{10 + (i % 3)}"
        for i in range(70)
    ]
    csv.write_text("\n".join(lines))
    series = load_time_series_from_csv(str(csv), value_column="value", verbose=False)
    assert len(series) >= 64


def test_load_csv_file_not_found() -> None:
    """Test FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_time_series_from_csv("/nonexistent/path.csv", verbose=False)


def test_load_csv_no_numeric_column(tmp_path: Path) -> None:
    """Test ValueError when no numeric columns."""
    csv = tmp_path / "test.csv"
    csv.write_text("a,b\nx,y\nz,w\n")
    with pytest.raises(ValueError, match="numeric"):
        load_time_series_from_csv(str(csv), verbose=False)


def test_load_csv_empty_after_nan_drop(tmp_path: Path) -> None:
    """Test ValueError when all values are NaN or invalid."""
    csv = tmp_path / "test.csv"
    csv.write_text("value\n,\n,\n")
    with pytest.raises(ValueError, match="empty|No numeric|valid"):
        load_time_series_from_csv(str(csv), verbose=False)


def test_normalize_zscore() -> None:
    """Test z-score normalization."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = normalize_series(x, method="zscore")
    np.testing.assert_almost_equal(np.mean(out), 0.0)
    np.testing.assert_almost_equal(np.std(out), 1.0)


def test_normalize_minmax() -> None:
    """Test min-max normalization."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = normalize_series(x, method="minmax")
    assert out.min() == 0.0
    assert out.max() == 1.0


def test_normalize_none() -> None:
    """Test no normalization returns copy."""
    x = np.array([1.0, 2.0, 3.0])
    out = normalize_series(x, method="none")
    np.testing.assert_array_equal(out, x)
    assert out is not x


def test_validate_series_too_short() -> None:
    """Test validation fails for short series."""
    series = np.random.randn(60)  # < 3*32=96
    with pytest.raises(ValueError, match="too short"):
        validate_series_for_forecasting(series, block_size=32)


def test_validate_series_ok() -> None:
    """Test validation passes for long enough series."""
    series = np.random.randn(100)  # >= 3*32
    validate_series_for_forecasting(series, block_size=32)


def test_load_european_format(tmp_path: Path) -> None:
    """Test European number format: 19403,9 and 19 403,9."""
    csv = tmp_path / "test.csv"
    base = datetime(2023, 1, 1)
    lines = [
        "Date,Close",
        "2023-01-01,\"19403,9\"",
        "2023-01-02,\"19 403,9\"",
        "2023-01-03,19280.79",
    ] + [
        f"{(base + timedelta(days=i)).strftime('%Y-%m-%d')},{19000 + i * 10}"
        for i in range(4, 70)
    ]
    csv.write_text("\n".join(lines))
    series = load_time_series_from_csv(str(csv), verbose=False)
    assert len(series) >= 64
    np.testing.assert_almost_equal(series[0], 19403.9)
    np.testing.assert_almost_equal(series[1], 19403.9)


def test_load_min_64_validation(tmp_path: Path) -> None:
    """Test that fewer than 64 values raises clear error."""
    csv = tmp_path / "test.csv"
    lines = ["Close"] + [str(100 + i) for i in range(50)]
    csv.write_text("\n".join(lines))
    with pytest.raises(ValueError, match="64"):
        load_time_series_from_csv(str(csv), verbose=False)


def test_load_price_column_priority(tmp_path: Path) -> None:
    """Test Close is preferred over Volume."""
    csv = tmp_path / "test.csv"
    base = datetime(2023, 1, 1)
    lines = ["Date,Open,High,Low,Close,Volume"]
    lines += [
        f"{(base + timedelta(days=i)).strftime('%Y-%m-%d')},{100+i},{101+i},{99+i},{100+i},0"
        for i in range(70)
    ]
    csv.write_text("\n".join(lines))
    series = load_time_series_from_csv(str(csv), verbose=False)
    assert len(series) >= 64
    assert np.any(series != 0)  # Not Volume (all zeros)

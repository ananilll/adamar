"""
WHT-based forecasting algorithm.

Pipeline:
1. Split series into blocks
2. Apply WHT to each block
3. Select top-k coefficients
4. Compute deltas between consecutive blocks
5. Smooth deltas
6. Forecast next block coefficients
7. Apply inverse WHT
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from wht_forecast.blocks import split_into_blocks
from wht_forecast.filtering import select_top_coefficients
from wht_forecast.hadamard import build_normalized_hadamard
from wht_forecast.transform import wht_forward, wht_inverse


def compute_deltas(coeffs_history: List[np.ndarray]) -> List[np.ndarray]:
    """
    Compute coefficient deltas between consecutive blocks.

    delta_C^{(k)} = C_filtered^{(k)} - C_filtered^{(k-1)}

    Parameters
    ----------
    coeffs_history : List[np.ndarray]
        History of filtered coefficients per block.

    Returns
    -------
    List[np.ndarray]
        List of delta vectors.
    """
    deltas: List[np.ndarray] = []
    for i in range(1, len(coeffs_history)):
        deltas.append(coeffs_history[i] - coeffs_history[i - 1])
    return deltas


def smooth_deltas(deltas: List[np.ndarray], m: int) -> np.ndarray:
    """
    Smooth deltas over the last m blocks.

    delta_avg = mean(delta^{(k)}, delta^{(k-1)}, ..., delta^{(k-m+1)})

    Parameters
    ----------
    deltas : List[np.ndarray]
        List of coefficient deltas.
    m : int
        Smoothing window size.

    Returns
    -------
    np.ndarray
        Averaged delta vector.

    Raises
    ------
    ValueError
        If deltas is empty (requires at least 2 blocks).
    """
    if len(deltas) == 0:
        raise ValueError("Need at least 2 blocks to compute deltas")

    window = deltas[-m:] if len(deltas) >= m else deltas
    return np.mean(window, axis=0)


def forecast_next_block(
    series: np.ndarray,
    A: Optional[np.ndarray] = None,
    block_size: int = 32,
    top_k: int = 8,
    smooth_window: int = 3,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Forecast the next block of a time series using WHT.

    Algorithm:
    1. Split series into blocks of block_size.
    2. Apply WHT to each block.
    3. Select top_k coefficients by energy.
    4. Compute deltas between consecutive blocks.
    5. Smooth deltas over last smooth_window blocks.
    6. Forecast: C_hat^{(K+1)} = C_filtered^{(K)} + delta_avg
    7. Apply inverse WHT to obtain forecast block.

    Parameters
    ----------
    series : np.ndarray
        Input time series.
    A : np.ndarray
        Normalized Walsh-Hadamard matrix.
    block_size : int
        Block length (default: 32).
    top_k : int
        Number of coefficients to retain (default: 8).
    smooth_window : int
        Smoothing window for deltas (default: 3).

    Returns
    -------
    Tuple[np.ndarray, Dict[str, object]]
        - forecast: Predicted next block in original domain.
        - info: Dict with blocks, coeffs, deltas, etc. for analysis.

    Raises
    ------
    ValueError
        If series is too short for forecasting (need at least 2 blocks).
    """
    if A is None:
        A = build_normalized_hadamard(block_size)

    blocks = split_into_blocks(series, block_size)

    if len(blocks) < 2:
        raise ValueError("Series too short for forecasting (need at least 2 blocks)")

    all_raw_coeffs: List[np.ndarray] = []
    all_filtered_coeffs: List[np.ndarray] = []

    for block in blocks:
        raw = wht_forward(block, A)
        filtered, _ = select_top_coefficients(raw, top_k)
        all_raw_coeffs.append(raw)
        all_filtered_coeffs.append(filtered)

    deltas = compute_deltas(all_filtered_coeffs)
    avg_delta = smooth_deltas(deltas, smooth_window)

    last_filtered = all_filtered_coeffs[-1]
    forecast_coeffs = last_filtered + avg_delta
    forecast_block = wht_inverse(forecast_coeffs, A)

    info: Dict[str, object] = {
        "blocks": blocks,
        "all_raw_coeffs": all_raw_coeffs,
        "all_filtered_coeffs": all_filtered_coeffs,
        "deltas": deltas,
        "avg_delta": avg_delta,
        "forecast_coeffs": forecast_coeffs,
        "last_filtered": last_filtered,
    }

    return forecast_block, info

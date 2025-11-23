"""Utility helpers for scaling and smoothing time-series data.

The helpers here are intentionally lightweight and only use NumPy so the
project remains easy to understand for beginners.
"""
from __future__ import annotations

import numpy as np


def standardize(data: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Standardize an array to zero mean and unit variance.

    Args:
        data: Input array.

    Returns:
        Tuple containing the standardized data, mean, and standard deviation.
    """
    mean = np.mean(data)
    std = np.std(data) + 1e-8
    standardized = (data - mean) / std
    return standardized, mean, std


def destandardize(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Reverse the standardization transform."""
    return data * std + mean


def moving_average(sequence: np.ndarray, window: int = 3) -> np.ndarray:
    """Compute a simple moving average for smoothing curves.

    Args:
        sequence: 1D data to smooth.
        window: Number of points used in the rolling window.

    Returns:
        Smoothed sequence with the same length as input.
    """
    if window <= 1:
        return sequence
    cumsum = np.cumsum(np.insert(sequence, 0, 0))
    smoothed = (cumsum[window:] - cumsum[:-window]) / float(window)
    # Pad the beginning so the output length matches input length
    pad = np.full(window - 1, smoothed[0])
    return np.concatenate([pad, smoothed])

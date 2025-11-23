"""Dataset helpers for simple time-series prediction tasks.

This module creates a noisy sine wave and prepares supervised learning
pairs for next-step prediction. The goal is to keep the data small and
friendly so the quantum and classical reservoirs can run quickly.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split as skl_train_test_split


def generate_sine_sequence(n_points: int = 300, noise: float = 0.05, freq: float = 1.0, seed: int | None = 0) -> np.ndarray:
    """Generate a noisy sine wave.

    Args:
        n_points: Number of time steps to simulate.
        noise: Standard deviation of Gaussian noise added to the sine wave.
        freq: Frequency multiplier for the sine wave.
        seed: Optional random seed for reproducibility.

    Returns:
        A NumPy array of shape ``(n_points,)`` representing the signal.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 10, n_points)
    clean = np.sin(2 * np.pi * freq * t)
    noisy = clean + rng.normal(scale=noise, size=n_points)
    return noisy


def build_supervised_pairs(sequence: np.ndarray, window_size: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 1D sequence into supervised input/target pairs.

    Each input contains a sliding window of consecutive points, and the target
    is the next point. For example, with ``window_size=3`` the first input is
    ``[x0, x1, x2]`` and the target is ``x3``.

    Args:
        sequence: 1D array of values.
        window_size: Number of past points to use for predicting the next one.

    Returns:
        ``(X, y)`` where ``X`` has shape ``(T, window_size)`` and ``y`` has
        shape ``(T,)``.
    """
    if sequence.ndim != 1:
        raise ValueError("Sequence must be 1D for this simple example.")

    X = []
    y = []
    for idx in range(len(sequence) - window_size):
        window = sequence[idx : idx + window_size]
        target = sequence[idx + window_size]
        X.append(window)
        y.append(target)

    return np.array(X), np.array(y)


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int | None = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Wrapper around ``sklearn.model_selection.train_test_split`` with defaults.

    Args:
        X: Input features shaped ``(n_samples, window_size)``.
        y: Targets shaped ``(n_samples,)``.
        test_size: Fraction reserved for evaluation.
        seed: Random seed for reproducibility.

    Returns:
        ``X_train, X_test, y_train, y_test`` split arrays.
    """
    return skl_train_test_split(X, y, test_size=test_size, random_state=seed, shuffle=True)

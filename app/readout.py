"""Classical linear readout for reservoir states.

Reservoir computing keeps the reservoir weights fixed and trains only this
readout layer. We use ridge regression to keep the solution stable.
"""
from __future__ import annotations

import numpy as np
from numpy.linalg import inv


def train_readout(reservoir_states: np.ndarray, targets: np.ndarray, alpha: float = 1e-3) -> np.ndarray:
    """Train a ridge-regression readout in closed form.

    Args:
        reservoir_states: Matrix of shape ``(T, d)``.
        targets: Vector of shape ``(T,)``.
        alpha: Ridge penalty to stabilize the solution.

    Returns:
        Weight vector ``w`` of shape ``(d,)``.
    """
    R = np.asarray(reservoir_states)
    y = np.asarray(targets)
    # Closed-form ridge regression: w = (R^T R + alpha I)^-1 R^T y
    RtR = R.T @ R
    d = RtR.shape[0]
    regularized = RtR + alpha * np.eye(d)
    pseudo_inv = inv(regularized) @ R.T
    w = pseudo_inv @ y
    return w


def predict_readout(reservoir_states: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Make predictions using trained readout weights."""
    return np.asarray(reservoir_states) @ np.asarray(weights)

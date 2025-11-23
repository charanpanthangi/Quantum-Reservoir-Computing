"""Classical reservoir computing baseline using a simple echo state network.

The reservoir uses fixed random weights and a tanh nonlinearity to transform
inputs into richer feature vectors. Only the final readout layer is trained.
"""
from __future__ import annotations

import numpy as np


def _spectral_radius(matrix: np.ndarray) -> float:
    """Compute the spectral radius (max absolute eigenvalue)."""
    eigvals = np.linalg.eigvals(matrix)
    return float(np.max(np.abs(eigvals)))


class ClassicalReservoir:
    """Simple echo state network style reservoir.

    Args:
        n_inputs: Size of input vector.
        n_reservoir: Number of reservoir units.
        spectral_radius: Desired spectral radius to keep dynamics stable.
        input_scale: Scaling factor for input weights.
        seed: Optional random seed.
    """

    def __init__(
        self,
        n_inputs: int,
        n_reservoir: int = 50,
        spectral_radius: float = 0.9,
        input_scale: float = 0.5,
        seed: int | None = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.W_in = rng.uniform(-input_scale, input_scale, size=(n_reservoir, n_inputs))
        raw_W = rng.uniform(-1, 1, size=(n_reservoir, n_reservoir))
        radius = _spectral_radius(raw_W)
        self.W_res = raw_W * (spectral_radius / (radius + 1e-8))
        self.state = np.zeros(n_reservoir)

    def reset_state(self) -> None:
        """Reset the internal reservoir state to zeros."""
        self.state = np.zeros_like(self.state)

    def step(self, u_t: np.ndarray) -> np.ndarray:
        """Update the reservoir given a single input vector."""
        pre_activation = self.W_in @ u_t + self.W_res @ self.state
        self.state = np.tanh(pre_activation)
        return self.state

    def encode_sequence(self, inputs: np.ndarray) -> np.ndarray:
        """Encode a sequence of inputs into reservoir states.

        Args:
            inputs: Array shaped ``(T, n_inputs)`` or ``(T,)``.

        Returns:
            Reservoir states shaped ``(T, n_reservoir)``.
        """
        inputs = np.asarray(inputs)
        if inputs.ndim == 1:
            inputs = inputs[:, None]

        self.reset_state()
        states = []
        for row in inputs:
            state = self.step(row)
            states.append(state.copy())
        return np.stack(states, axis=0)

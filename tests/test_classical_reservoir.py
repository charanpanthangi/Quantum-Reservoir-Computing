"""Tests for the classical reservoir baseline."""
import numpy as np

from app.classical_reservoir import ClassicalReservoir


def test_classical_reservoir_shapes():
    reservoir = ClassicalReservoir(n_inputs=1, n_reservoir=10, seed=2)
    inputs = np.linspace(0, 1, 6)
    states = reservoir.encode_sequence(inputs)
    assert states.shape == (6, 10)
    assert np.all(np.isfinite(states))

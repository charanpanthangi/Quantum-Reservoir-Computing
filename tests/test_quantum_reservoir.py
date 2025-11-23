"""Tests for the quantum reservoir implementation."""
import numpy as np

from app.quantum_reservoir import QuantumReservoir


def test_encode_sequence_shape():
    reservoir = QuantumReservoir(n_qubits=2, depth=1, seed=1)
    inputs = np.linspace(0, 1, 5)
    states = reservoir.encode_sequence(inputs)
    assert states.shape == (5, 2)
    assert np.all(np.isfinite(states))

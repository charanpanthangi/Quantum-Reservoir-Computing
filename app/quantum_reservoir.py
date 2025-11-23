"""Quantum reservoir implementation using PennyLane.

The quantum reservoir is a fixed random parameterized quantum circuit. Inputs
are encoded into rotations, and we read out expectation values of Pauli-Z on
all qubits as the reservoir state. Only the classical readout is trained.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml


def create_qrc_device(n_qubits: int) -> qml.Device:
    """Create a PennyLane device for the reservoir.

    Args:
        n_qubits: Number of qubits in the device.

    Returns:
        PennyLane device ready for simulation.
    """
    return qml.device("default.qubit", wires=n_qubits)


def build_random_reservoir_circuit(n_qubits: int, depth: int, seed: int | None = 0) -> np.ndarray:
    """Generate fixed random parameters for the quantum reservoir circuit.

    Args:
        n_qubits: Number of qubits to use.
        depth: Number of repeated random layers.
        seed: Random seed for reproducibility.

    Returns:
        Array of random rotation angles with shape ``(depth, n_qubits, 3)``
        representing RZ, RY, RZ rotations per qubit per layer.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(low=0.0, high=2 * np.pi, size=(depth, n_qubits, 3))


def _layer_random_rotations(params: np.ndarray, wires: list[int]) -> None:
    """Apply a single layer of random rotations and simple entanglement."""
    for wire, (phi, theta, lam) in zip(wires, params):
        qml.RZ(phi, wires=wire)
        qml.RY(theta, wires=wire)
        qml.RZ(lam, wires=wire)
    # Add nearest-neighbor CNOTs for entanglement
    for i in range(len(wires) - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])


def _encode_input(u_t: float, wires: list[int]) -> None:
    """Encode a scalar input using simple rotation gates."""
    angle = float(u_t) * np.pi  # scale input into a rotation angle
    for wire in wires:
        qml.RY(angle, wires=wire)


def build_qnode(device: qml.Device, params: np.ndarray, wires: list[int]):
    """Construct a QNode that processes a single time step.

    The circuit resets to ``|0...0>`` each call, encodes the input, applies
    the random reservoir layers, and measures Pauli-Z expectations on all
    qubits.
    """

    @qml.qnode(device)
    def reservoir_step(u_t: float) -> np.ndarray:
        _encode_input(u_t, wires)
        for layer_params in params:
            _layer_random_rotations(layer_params, wires)
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    return reservoir_step


class QuantumReservoir:
    """Quantum reservoir wrapper for encoding sequences.

    Attributes:
        n_qubits: Number of qubits used in the reservoir.
        depth: Number of random layers in the circuit.
        reservoir_step: QNode that maps a scalar input to reservoir features.
    """

    def __init__(self, n_qubits: int = 2, depth: int = 2, seed: int | None = 0):
        self.n_qubits = n_qubits
        self.depth = depth
        self.params = build_random_reservoir_circuit(n_qubits, depth, seed)
        self.device = create_qrc_device(n_qubits)
        self.wires = list(range(n_qubits))
        self.reservoir_step = build_qnode(self.device, self.params, self.wires)

    def encode_sequence(self, inputs: np.ndarray) -> np.ndarray:
        """Encode a sequence of scalars into quantum reservoir states.

        Args:
            inputs: 1D or 2D array. If 2D, each row is a window; we average
                the window to a single scalar for simplicity.

        Returns:
            Array of shape ``(T, n_qubits)`` representing reservoir states.
        """
        inputs = np.asarray(inputs)
        if inputs.ndim == 2:
            scalars = inputs.mean(axis=1)
        else:
            scalars = inputs

        states = []
        for value in scalars:
            state = np.array(self.reservoir_step(float(value)), dtype=float)
            states.append(state)
        return np.stack(states, axis=0)

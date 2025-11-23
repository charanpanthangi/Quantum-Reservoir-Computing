"""Training and evaluation helpers for QRC and classical reservoirs.

The logic is intentionally simple: encode the sequence with a reservoir,
fit a linear readout, and compute mean squared error on a held-out set.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error

from .classical_reservoir import ClassicalReservoir
from .quantum_reservoir import QuantumReservoir
from .readout import predict_readout, train_readout


def train_qrc_and_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_qubits: int = 2,
    depth: int = 2,
    classical_dim: int = 50,
    seed: int | None = 0,
):
    """Train quantum and classical reservoirs on the same data.

    Args:
        X_train: Training windows.
        y_train: Training targets.
        X_test: Test windows.
        y_test: Test targets.
        n_qubits: Number of qubits for QRC.
        depth: Depth of the random quantum circuit.
        classical_dim: Number of units in the classical reservoir.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with metrics, predictions, and reservoir states.
    """
    # Quantum reservoir pipeline
    q_reservoir = QuantumReservoir(n_qubits=n_qubits, depth=depth, seed=seed)
    R_q_train = q_reservoir.encode_sequence(X_train)
    w_q = train_readout(R_q_train, y_train)
    R_q_test = q_reservoir.encode_sequence(X_test)
    y_q_pred = predict_readout(R_q_test, w_q)
    mse_q = mean_squared_error(y_test, y_q_pred)

    # Classical reservoir pipeline
    c_reservoir = ClassicalReservoir(n_inputs=X_train.shape[1], n_reservoir=classical_dim, seed=seed)
    R_c_train = c_reservoir.encode_sequence(X_train)
    w_c = train_readout(R_c_train, y_train)
    R_c_test = c_reservoir.encode_sequence(X_test)
    y_c_pred = predict_readout(R_c_test, w_c)
    mse_c = mean_squared_error(y_test, y_c_pred)

    return {
        "mse_qrc": mse_q,
        "mse_classical": mse_c,
        "pred_qrc": y_q_pred,
        "pred_classical": y_c_pred,
        "reservoir_states_qrc": R_q_train,
        "reservoir_states_classical": R_c_train,
    }

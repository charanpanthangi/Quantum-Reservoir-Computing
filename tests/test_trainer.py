"""Integration-style test for the training pipeline."""
import numpy as np

from app import dataset
from app.trainer import train_qrc_and_baseline


def test_training_pipeline_runs():
    seq = dataset.generate_sine_sequence(n_points=40, noise=0.01, seed=1)
    X, y = dataset.build_supervised_pairs(seq, window_size=3)
    X_train, X_test, y_train, y_test = dataset.train_test_split(X, y, test_size=0.25, seed=1)
    results = train_qrc_and_baseline(
        X_train,
        y_train,
        X_test,
        y_test,
        n_qubits=1,
        depth=1,
        classical_dim=5,
        seed=1,
    )
    assert "pred_qrc" in results
    assert results["pred_qrc"].shape == y_test.shape
    assert results["pred_classical"].shape == y_test.shape

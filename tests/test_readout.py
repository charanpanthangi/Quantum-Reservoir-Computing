"""Tests for readout training and prediction."""
import numpy as np

from app.readout import predict_readout, train_readout


def test_readout_training_and_prediction():
    rng = np.random.default_rng(0)
    R = rng.normal(size=(20, 5))
    true_w = np.array([1, -0.5, 0.25, 0.1, 0.0])
    y = R @ true_w
    w = train_readout(R, y, alpha=1e-4)
    preds = predict_readout(R, w)
    assert preds.shape == y.shape
    assert np.mean(np.abs(preds - y)) < 1e-6

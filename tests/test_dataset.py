"""Tests for dataset generation utilities."""
import numpy as np

from app import dataset


def test_generate_sine_sequence_shape():
    seq = dataset.generate_sine_sequence(n_points=50)
    assert seq.shape == (50,)


def test_build_supervised_pairs_shapes():
    seq = np.arange(10, dtype=float)
    X, y = dataset.build_supervised_pairs(seq, window_size=3)
    assert X.shape == (7, 3)
    assert y.shape == (7,)

"""Tests for the PixelModel class."""

import numpy as np
import pytest
import tempfile
from pathlib import Path
from aipixel.model import PixelModel


def test_train_linearly_separable():
    """Model should learn a linearly separable dataset to high accuracy."""
    np.random.seed(99)
    X = np.vstack([
        np.random.uniform(0.6, 1.0, (30, 2)),
        np.random.uniform(0.0, 0.4, (30, 2)),
    ])
    y = np.array([1] * 30 + [0] * 30)

    model = PixelModel(n_inputs=2)
    model.train(X, y, epochs=500, lr=0.5)

    acc = model.accuracy(X, y)
    assert acc >= 0.9, f"Expected accuracy >= 0.9, got {acc}"


def test_bounded_weights():
    """Weights should always stay within [-4, 4]."""
    np.random.seed(100)
    X = np.random.rand(100, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(float)

    model = PixelModel(n_inputs=2)
    model.train(X, y, epochs=1000, lr=1.0)

    assert np.all(np.abs(model.weights) <= 4.0), f"Weights out of bounds: {model.weights}"
    assert abs(model.bias) <= 4.0, f"Bias out of bounds: {model.bias}"


def test_training_history():
    """Training history should be populated and loss should decrease."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 1], dtype=float)  # OR gate (linearly separable)

    model = PixelModel(n_inputs=2)
    model.train(X, y, epochs=200)

    assert len(model.training_history) == 200
    assert model.training_history[-1] < model.training_history[0]


def test_predict_proba_range():
    """Probabilities should be in [0, 1]."""
    model = PixelModel(n_inputs=2)
    X = np.random.rand(50, 2)
    y = (X[:, 0] > 0.5).astype(float)
    model.train(X, y, epochs=100)

    probs = model.predict_proba(X)
    assert np.all(probs >= 0) and np.all(probs <= 1)


def test_1_input_model():
    """1-input model should train and predict."""
    X = np.array([[0.1], [0.2], [0.3], [0.7], [0.8], [0.9]])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = PixelModel(n_inputs=1)
    model.train(X, y, epochs=500)
    assert model.accuracy(X, y) >= 0.8


def test_3_input_model():
    """3-input model should train and predict."""
    np.random.seed(101)
    X = np.random.rand(60, 3)
    y = (X[:, 0] + X[:, 1] + X[:, 2] > 1.5).astype(float)

    model = PixelModel(n_inputs=3)
    model.train(X, y, epochs=500)
    assert model.accuracy(X, y) >= 0.8


def test_invalid_n_inputs():
    """Should reject n_inputs outside 1-3."""
    with pytest.raises(ValueError):
        PixelModel(n_inputs=0)
    with pytest.raises(ValueError):
        PixelModel(n_inputs=4)

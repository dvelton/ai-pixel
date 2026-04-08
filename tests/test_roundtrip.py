"""End-to-end tests: train -> save pixel -> load pixel -> predict."""

import numpy as np
import tempfile
import pytest
from pathlib import Path
from aipixel.model import PixelModel
from aipixel.datasets import umbrella, sunscreen, escalate


def test_roundtrip_save_load():
    """Save model as PNG, load it back, predictions should match."""
    np.random.seed(200)
    X = np.random.rand(50, 2)
    y = (X[:, 0] * 0.7 + X[:, 1] * 0.3 > 0.5).astype(float)

    model = PixelModel(n_inputs=2)
    model.train(X, y, epochs=500)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = Path(f.name)

    model.to_image(path)
    loaded = PixelModel.from_image(path)

    original_preds = model.predict(X)
    loaded_preds = loaded.predict(X)

    match_rate = np.mean(original_preds == loaded_preds)
    assert match_rate >= 0.95, f"Prediction match rate {match_rate:.1%} after round-trip"

    path.unlink()


def test_pixel_values_roundtrip():
    """from_pixel(to_pixel()) should produce similar weights."""
    model = PixelModel(n_inputs=2)
    X = np.array([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])
    y = np.array([1, 0, 1, 0])
    model.train(X, y, epochs=300)

    pixel = model.to_pixel()
    restored = PixelModel.from_pixel(*pixel)

    np.testing.assert_allclose(model.weights, restored.weights, atol=0.035)
    assert abs(model.bias - restored.bias) < 0.035


def test_quantization_report():
    """Quantization report should have all expected fields."""
    model = PixelModel(n_inputs=2)
    X = np.random.rand(30, 2)
    y = (X[:, 0] > 0.5).astype(float)
    model.train(X, y, epochs=100)

    report = model.quantization_report()
    assert "float_weights" in report
    assert "pixel_rgb" in report
    assert "pixel_hex" in report
    assert "max_param_error" in report
    assert report["max_param_error"] < 0.016


@pytest.mark.parametrize("dataset_fn", [umbrella, sunscreen, escalate])
def test_dataset_trainable(dataset_fn):
    """Each built-in dataset should be trainable to reasonable accuracy."""
    X, y, meta = dataset_fn()
    model = PixelModel(n_inputs=2)
    model.train(X, y, epochs=500)

    acc = model.accuracy(X, y)
    assert acc >= 0.85, f"{meta['name']} accuracy too low: {acc:.1%}"

    # Save and reload
    pixel = model.to_pixel()
    restored = PixelModel.from_pixel(*pixel)
    q_acc = restored.accuracy(X, y)
    assert q_acc >= 0.80, f"{meta['name']} pixel accuracy too low: {q_acc:.1%}"

"""Tests for the codec module."""

import numpy as np
import pytest
from aipixel.codec import weight_to_byte, byte_to_weight, encode_weights, decode_weights, quantization_error


def test_byte_roundtrip_boundaries():
    """Boundary values should round-trip correctly."""
    assert weight_to_byte(-4.0) == 0
    assert weight_to_byte(4.0) == 255
    assert weight_to_byte(0.0) == 128  # close to center

    assert byte_to_weight(0) == pytest.approx(-4.0)
    assert byte_to_weight(255) == pytest.approx(4.0)


def test_byte_roundtrip_precision():
    """Round-trip error should be small (< 0.016 per param)."""
    for w in np.linspace(-4.0, 4.0, 100):
        b = weight_to_byte(w)
        w_back = byte_to_weight(b)
        assert abs(w - w_back) < 0.016, f"Round-trip error too large for {w}: got {w_back}"


def test_clamping():
    """Values outside [-4, 4] should be clamped."""
    assert weight_to_byte(-10.0) == 0
    assert weight_to_byte(10.0) == 255


def test_encode_decode_2input():
    """2-input model should round-trip through encode/decode."""
    weights = np.array([1.5, -2.3])
    bias = 0.7
    pixel = encode_weights(weights, bias)
    assert len(pixel) == 3

    dec_weights, dec_bias, n_inputs = decode_weights(pixel)
    assert n_inputs == 2
    assert len(dec_weights) == 2
    np.testing.assert_allclose(dec_weights, weights, atol=0.032)
    assert abs(dec_bias - bias) < 0.032


def test_encode_decode_1input():
    """1-input model uses n_inputs parameter for decoding."""
    weights = np.array([2.0])
    bias = -1.0
    pixel = encode_weights(weights, bias)
    assert len(pixel) == 3
    assert pixel[2] == 128  # unused channel

    dec_weights, dec_bias, n_inputs = decode_weights(pixel, n_inputs=1)
    assert n_inputs == 1
    assert len(dec_weights) == 1
    np.testing.assert_allclose(dec_weights, weights, atol=0.032)


def test_encode_decode_3input():
    """3-input model uses RGBA."""
    weights = np.array([1.0, -1.0, 3.0])
    bias = -0.5
    pixel = encode_weights(weights, bias)
    assert len(pixel) == 4

    dec_weights, dec_bias, n_inputs = decode_weights(pixel)
    assert n_inputs == 3
    assert len(dec_weights) == 3
    np.testing.assert_allclose(dec_weights, weights, atol=0.032)


def test_quantization_error_report():
    """Error report should have expected keys and reasonable values."""
    weights = np.array([1.234, -3.456])
    bias = 0.789
    report = quantization_error(weights, bias)
    assert "max_param_error" in report
    assert "max_logit_error" in report
    assert report["max_param_error"] < 0.016

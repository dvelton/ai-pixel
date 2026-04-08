"""
Codec for encoding/decoding neural network weights as pixel RGB values.

Format (v1):
    - Each byte maps linearly from 0-255 to [-4.0, +4.0]
    - 2-input model: R=W1, G=W2, B=Bias
    - 1-input model: R=W1, G=Bias, B=0x80 (unused)
    - 3-input model: R=W1, G=W2, B=W3, A=Bias (RGBA)
    - PNG only (lossless required)
"""

import numpy as np

WEIGHT_MIN = -4.0
WEIGHT_MAX = 4.0
WEIGHT_RANGE = WEIGHT_MAX - WEIGHT_MIN

def weight_to_byte(w: float) -> int:
    """Encode a single float weight as an 8-bit integer (0-255)."""
    clamped = np.clip(w, WEIGHT_MIN, WEIGHT_MAX)
    normalized = (clamped - WEIGHT_MIN) / WEIGHT_RANGE
    return int(np.round(normalized * 255))


def byte_to_weight(b: int) -> float:
    """Decode an 8-bit integer (0-255) back to a float weight."""
    return (b / 255.0) * WEIGHT_RANGE + WEIGHT_MIN


def encode_weights(weights: np.ndarray, bias: float) -> tuple:
    """
    Encode model weights and bias as pixel color values.

    Args:
        weights: Array of 1-3 weights.
        bias: Scalar bias value.

    Returns:
        Tuple of (R, G, B) or (R, G, B, A) integers.
    """
    n = len(weights)
    if n == 1:
        return (weight_to_byte(weights[0]), weight_to_byte(bias), 128)
    elif n == 2:
        return (
            weight_to_byte(weights[0]),
            weight_to_byte(weights[1]),
            weight_to_byte(bias),
        )
    elif n == 3:
        return (
            weight_to_byte(weights[0]),
            weight_to_byte(weights[1]),
            weight_to_byte(weights[2]),
            weight_to_byte(bias),
        )
    else:
        raise ValueError(f"Supports 1-3 weights, got {n}")


def decode_weights(pixel: tuple, n_inputs=None) -> tuple:
    """
    Decode pixel color values back to weights and bias.

    Args:
        pixel: Tuple of (R, G, B) or (R, G, B, A).
        n_inputs: Number of inputs (1, 2, or 3). If None, inferred from pixel length.

    Returns:
        (weights_array, bias_float, n_inputs)
    """
    if n_inputs == 3 or (n_inputs is None and len(pixel) == 4):
        weights = np.array([
            byte_to_weight(pixel[0]),
            byte_to_weight(pixel[1]),
            byte_to_weight(pixel[2]),
        ])
        bias = byte_to_weight(pixel[3]) if len(pixel) == 4 else byte_to_weight(pixel[2])
        return weights, bias, 3

    if n_inputs == 1:
        weights = np.array([byte_to_weight(pixel[0])])
        bias = byte_to_weight(pixel[1])
        return weights, bias, 1

    r, g, b = pixel[0], pixel[1], pixel[2]

    # Default: 2-input model (most common case)
    weights = np.array([byte_to_weight(r), byte_to_weight(g)])
    bias = byte_to_weight(b)
    return weights, bias, n_inputs if n_inputs else 2


def quantization_error(original_weights: np.ndarray, original_bias: float) -> dict:
    """
    Report the error introduced by quantization.

    Returns dict with per-parameter errors and max logit error estimate.
    """
    all_params = np.append(original_weights, original_bias)
    encoded = [weight_to_byte(p) for p in all_params]
    decoded = np.array([byte_to_weight(b) for b in encoded])
    errors = np.abs(all_params - decoded)

    return {
        "per_param_error": errors,
        "max_param_error": float(np.max(errors)),
        "mean_param_error": float(np.mean(errors)),
        "max_logit_error": float(np.sum(errors)),  # worst case: all errors additive
    }

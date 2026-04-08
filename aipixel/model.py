"""
PixelModel: A single-neuron classifier that fits in one pixel.

Trains via gradient descent with sigmoid activation and binary cross-entropy loss.
Weights are bounded to [-4, 4] using projected gradient descent (clip after each step).
The trained model encodes into the RGB values of a 1x1 PNG.
"""

import numpy as np
from pathlib import Path
from aipixel.codec import (
    encode_weights, decode_weights, weight_to_byte, byte_to_weight,
    quantization_error, WEIGHT_MAX,
)


def _sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class PixelModel:
    """A single-neuron binary classifier that fits in one pixel."""

    def __init__(self, n_inputs=2):
        """
        Create a new model.

        Args:
            n_inputs: Number of input features (1, 2, or 3).
        """
        if n_inputs not in (1, 2, 3):
            raise ValueError("n_inputs must be 1, 2, or 3")
        self.n_inputs = n_inputs
        self._weights = np.zeros(n_inputs)
        self._bias = 0.0
        self._trained = False
        self._train_history = []

    @property
    def weights(self) -> np.ndarray:
        """Weights in [-4, 4]."""
        return self._weights.copy()

    @property
    def bias(self) -> float:
        """Bias in [-4, 4]."""
        return float(self._bias)

    def _forward(self, X):
        """Compute sigmoid output."""
        z = X @ self._weights + self._bias
        return _sigmoid(z)

    def train(self, X, y, epochs=500, lr=0.2, verbose=False):
        """
        Train the model on labeled data.

        Args:
            X: Input features, shape (n_samples, n_inputs). Should be normalized to [0, 1].
            y: Binary labels, shape (n_samples,). Values should be 0 or 1.
            epochs: Number of training iterations.
            lr: Learning rate.
            verbose: Print loss every 50 epochs.

        Returns:
            self (for chaining).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != self.n_inputs:
            raise ValueError(f"Expected {self.n_inputs} features, got {X.shape[1]}")

        n = len(y)
        self._train_history = []

        for epoch in range(epochs):
            # Forward pass
            y_hat = self._forward(X)
            y_hat_clipped = np.clip(y_hat, 1e-7, 1 - 1e-7)

            # Binary cross-entropy loss
            loss = -np.mean(y * np.log(y_hat_clipped) + (1 - y) * np.log(1 - y_hat_clipped))
            self._train_history.append(loss)

            if verbose and epoch % 50 == 0:
                acc = np.mean((y_hat >= 0.5) == y)
                print(f"Epoch {epoch:4d}  loss={loss:.4f}  acc={acc:.3f}")

            # Gradient step
            error = y_hat - y
            self._weights -= lr * (X.T @ error) / n
            self._bias -= lr * np.mean(error)

            # Project back to feasible region [-4, 4]
            self._weights = np.clip(self._weights, -WEIGHT_MAX, WEIGHT_MAX)
            self._bias = np.clip(self._bias, -WEIGHT_MAX, WEIGHT_MAX)

        self._trained = True
        return self

    def predict_proba(self, X):
        """Return probability of class 1."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1) if self.n_inputs == 1 else X.reshape(1, -1)
        return self._forward(X)

    def predict(self, X):
        """Return binary predictions (0 or 1)."""
        return (self.predict_proba(X) >= 0.5).astype(int)

    def accuracy(self, X, y):
        """Compute accuracy on labeled data."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.size == 0 or y.size == 0:
            return 0.0
        preds = self.predict(X)
        return float(np.mean(preds == y))

    def to_pixel(self) -> tuple:
        """Encode the trained model as RGB (or RGBA) values."""
        return encode_weights(self.weights, self.bias)

    def to_image(self, path):
        """Save the model as a 1x1 PNG file with n_inputs metadata."""
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo
        pixel = self.to_pixel()
        mode = "RGBA" if len(pixel) == 4 else "RGB"
        img = Image.new(mode, (1, 1))
        img.putpixel((0, 0), pixel)
        meta = PngInfo()
        meta.add_text("ai-pixel-inputs", str(self.n_inputs))
        img.save(str(path), format="PNG", pnginfo=meta)
        return path

    @classmethod
    def from_pixel(cls, *pixel_values, n_inputs=None):
        """
        Reconstruct a model from pixel RGB (or RGBA) values.

        Args:
            pixel_values: R, G, B (or R, G, B, A) values.
            n_inputs: Number of inputs. If None, defaults to 2 for RGB, 3 for RGBA.

        Usage:
            model = PixelModel.from_pixel(142, 87, 201)
            model = PixelModel.from_pixel(142, 87, n_inputs=1)
        """
        weights, bias, detected_n = decode_weights(pixel_values, n_inputs=n_inputs)
        model = cls(n_inputs=detected_n)
        model._weights = weights.copy()
        model._bias = float(bias)
        model._trained = True
        return model

    @classmethod
    def from_image(cls, path):
        """Load a model from a 1x1 PNG file."""
        from PIL import Image
        img = Image.open(str(path))
        if img.size != (1, 1):
            raise ValueError(f"Expected 1x1 image, got {img.size}")
        pixel = img.getpixel((0, 0))
        if isinstance(pixel, int):
            raise ValueError("Image must be RGB or RGBA, not grayscale")
        n_inputs = None
        if hasattr(img, "text") and "ai-pixel-inputs" in img.text:
            n_inputs = int(img.text["ai-pixel-inputs"])
        return cls.from_pixel(*pixel, n_inputs=n_inputs)

    def quantization_report(self):
        """Compare float-precision model with its quantized pixel version."""
        pixel = self.to_pixel()
        q_weights, q_bias, _ = decode_weights(pixel, n_inputs=self.n_inputs)
        error = quantization_error(self.weights, self.bias)

        return {
            "float_weights": self.weights.tolist(),
            "float_bias": float(self.bias),
            "pixel_rgb": pixel,
            "pixel_hex": "#{:02x}{:02x}{:02x}".format(*pixel[:3]),
            "quantized_weights": q_weights.tolist(),
            "quantized_bias": float(q_bias),
            "max_param_error": error["max_param_error"],
            "max_logit_error": error["max_logit_error"],
        }

    def summary(self):
        """Print a summary of the model."""
        pixel = self.to_pixel()
        report = self.quantization_report()
        hex_color = report["pixel_hex"]

        lines = [
            f"PixelModel ({self.n_inputs} input{'s' if self.n_inputs > 1 else ''})",
            f"  Weights: {self.weights}",
            f"  Bias:    {self.bias:.4f}",
            f"  Pixel:   RGB({pixel[0]}, {pixel[1]}, {pixel[2]})  {hex_color}",
            f"  Max quantization error: {report['max_param_error']:.4f}",
        ]
        if self._train_history:
            lines.append(f"  Final loss: {self._train_history[-1]:.4f}")
        print("\n".join(lines))

    @property
    def training_history(self):
        """Loss values from training."""
        return list(self._train_history)

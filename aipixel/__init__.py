"""
ai-pixel: Train a real AI model that fits in a single pixel.
"""

from aipixel.model import PixelModel
from aipixel.codec import encode_weights, decode_weights

__version__ = "0.1.0"
__all__ = ["PixelModel", "encode_weights", "decode_weights"]

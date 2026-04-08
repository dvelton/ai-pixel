# ai-pixel

Train a real AI model that fits in a single pixel.

ai-pixel takes your data, runs actual gradient descent to learn a classifier, then encodes the entire trained model into the RGB values of a 1x1 PNG image. Three parameters (two weights and a bias), three color channels, one pixel.

Load that pixel later to run predictions. The pixel is the model.

## Try it

Open the [interactive demo](https://dvelton.github.io/ai-pixel/) in your browser. Click to place data points, hit Train, and watch the model collapse into a single colored pixel you can download.

## Install

```
pip install git+https://github.com/dvelton/ai-pixel.git
```

Requires Python 3.9+. For visualization support, also install matplotlib:

```
pip install matplotlib
```

## Quick start

```python
from aipixel import PixelModel
from aipixel.datasets import umbrella

# Load example data
X, y, meta = umbrella()

# Train
model = PixelModel(n_inputs=2)
model.train(X, y, epochs=500)

# The model is a pixel
pixel = model.to_pixel()
print(f"Your AI: RGB({pixel[0]}, {pixel[1]}, {pixel[2]})")

# Save as a 1x1 PNG
model.to_image("model.png")

# Load it back
loaded = PixelModel.from_image("model.png")
loaded.predict([[0.8, 0.7]])  # => array([1])  "Bring umbrella"
```

## How it works

A single-neuron binary classifier has three learnable parameters for two inputs:

- Weight 1: how much input 1 matters
- Weight 2: how much input 2 matters
- Bias: the threshold offset

ai-pixel trains this classifier using gradient descent with sigmoid activation and binary cross-entropy loss. Weights are bounded to [-4.0, 4.0] during training so they always fit in one byte each. After training, each parameter is quantized to 8 bits and mapped to a color channel:

| Channel | Parameter | Range |
|---------|-----------|-------|
| R | Weight 1 | [-4.0, 4.0] in 256 steps |
| G | Weight 2 | [-4.0, 4.0] in 256 steps |
| B | Bias | [-4.0, 4.0] in 256 steps |

The resulting color IS the model. Different training data produces different colors.

## Built-in datasets

```python
from aipixel.datasets import umbrella, sunscreen, escalate, xor

# Each returns (X, y, metadata)
X, y, meta = umbrella()     # Rain chance + wind -> bring umbrella?
X, y, meta = sunscreen()    # UV index + hours outside -> wear sunscreen?
X, y, meta = escalate()     # Sentiment + severity -> escalate ticket?
X, y, meta = xor()          # XOR pattern (unsolvable — demonstrates limits)
```

## Visualization

```python
from aipixel.viz import plot_decision_boundary, plot_pixel

# Decision boundary with confidence heatmap
plot_decision_boundary(model, X, y, meta)

# Display the pixel as a color swatch
plot_pixel(model)
```

## CLI

```bash
# Train from CSV (last column = label, all others = features)
ai-pixel train data.csv --output model.png --epochs 500

# Inspect a pixel model
ai-pixel inspect model.png

# Run a prediction
ai-pixel predict model.png --input "0.8,0.6"
```

## Quantization transparency

ai-pixel reports how much accuracy is lost when the model is compressed to a pixel:

```python
report = model.quantization_report()
print(report["max_param_error"])   # Max error per parameter (typically < 0.016)
print(report["pixel_hex"])         # The model's color
```

## Limitations

A single neuron draws a straight line through your data. It works well for linearly separable problems but cannot learn patterns like XOR, circles, or anything requiring a curved decision boundary.

Try the XOR example to see it fail:

```python
from aipixel.datasets import xor

X, y, meta = xor()
model = PixelModel(n_inputs=2)
model.train(X, y, epochs=500)
print(f"XOR accuracy: {model.accuracy(X, y):.1%}")  # ~50% (random chance)
```

This is why deeper networks exist. ai-pixel is an educational tool and a compression experiment, not a production ML framework.

## The format

ai-pixel uses PNG exclusively. The format is lossless, which matters because the model's weights are encoded in exact byte values. Saving as JPEG, taking a screenshot, or running through image optimization will corrupt the model.

## License

MIT

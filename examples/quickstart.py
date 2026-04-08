"""Quickstart: Train an AI model that fits in one pixel."""

from aipixel import PixelModel
from aipixel.datasets import umbrella

# Load data
X, y, meta = umbrella()
print(f"Dataset: {meta['name']} — {meta['description']}")
print(f"Features: {meta['feature_names']}")
print(f"Classes: {meta['class_names']}")
print(f"Samples: {len(y)}")

# Train
model = PixelModel(n_inputs=2)
model.train(X, y, epochs=500, verbose=True)

# The model IS a pixel
pixel = model.to_pixel()
print(f"\nYour AI model: RGB({pixel[0]}, {pixel[1]}, {pixel[2]})")
print(f"That's it. That pixel IS the model.\n")

# Save it
model.to_image("examples/pixels/umbrella.png")
print("Saved to examples/pixels/umbrella.png (1x1 PNG)")

# Load it back and prove it works
from aipixel import PixelModel
loaded = PixelModel.from_image("examples/pixels/umbrella.png")
acc = loaded.accuracy(X, y)
print(f"Loaded pixel model accuracy: {acc:.1%}")

# Show quantization transparency
report = model.quantization_report()
print(f"Max quantization error: {report['max_param_error']:.4f}")

model.summary()

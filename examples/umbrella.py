"""Should you bring an umbrella? Train a pixel to decide."""

from aipixel import PixelModel
from aipixel.datasets import umbrella

X, y, meta = umbrella()

model = PixelModel(n_inputs=2)
model.train(X, y, epochs=500)
model.to_image("examples/pixels/umbrella.png")
model.summary()

# Try some predictions
print("\nPredictions:")
test_cases = [
    ([0.8, 0.7], "High rain chance, high wind"),
    ([0.2, 0.3], "Low rain chance, low wind"),
    ([0.6, 0.5], "Moderate rain, moderate wind"),
]
for inputs, desc in test_cases:
    prob = model.predict_proba([inputs])[0]
    decision = "Bring umbrella" if prob >= 0.5 else "Leave it"
    print(f"  {desc}: {decision} ({prob:.1%} confidence)")

# Visualize (requires matplotlib)
try:
    from aipixel.viz import plot_decision_boundary
    plot_decision_boundary(model, X, y, meta)
except ImportError:
    print("\nInstall matplotlib for visualization: pip install matplotlib")

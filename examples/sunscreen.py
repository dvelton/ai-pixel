"""Should you wear sunscreen? Train a pixel to decide."""

from aipixel import PixelModel
from aipixel.datasets import sunscreen

X, y, meta = sunscreen()

model = PixelModel(n_inputs=2)
model.train(X, y, epochs=500)
model.to_image("examples/pixels/sunscreen.png")
model.summary()

print("\nPredictions:")
test_cases = [
    ([0.9, 0.8], "High UV, long time outside"),
    ([0.1, 0.2], "Low UV, short time"),
    ([0.5, 0.5], "Moderate UV, moderate time"),
]
for inputs, desc in test_cases:
    prob = model.predict_proba([inputs])[0]
    decision = "Wear sunscreen" if prob >= 0.5 else "Skip it"
    print(f"  {desc}: {decision} ({prob:.1%} confidence)")

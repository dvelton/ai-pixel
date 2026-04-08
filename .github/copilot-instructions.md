# ai-pixel

When working on this project:

- This is a single-neuron classifier (sigmoid + gradient descent + BCE loss), not a perceptron. Use the term "single-neuron classifier" in code and docs.
- Bounded training: weights are parameterized as 4*tanh(u) so they always fit [-4.0, 4.0].
- PNG only. The format is lossless. JPEG or screenshots will corrupt the model.
- The web interface uses pure vanilla HTML/CSS/JS. No frameworks, no build step.
- Keep the Python library minimal: numpy and Pillow are the only required deps. Matplotlib is optional (viz extra).
- Tests should verify codec round-trip accuracy and float-vs-pixel accuracy delta.
- README is written for non-technical users. Lead with what it does, not how it works.

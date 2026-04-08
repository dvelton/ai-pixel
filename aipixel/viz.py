"""
Visualization tools for ai-pixel models.

Requires matplotlib (install with: pip install ai-pixel[viz])
"""

import numpy as np


def plot_decision_boundary(model, X=None, y=None, meta=None, ax=None, show=True):
    """
    Plot the decision boundary of a 2-input PixelModel.

    Args:
        model: Trained PixelModel with n_inputs=2.
        X: Optional data points to overlay, shape (n, 2).
        y: Optional labels for data points.
        meta: Optional dataset metadata dict (feature_names, class_names, name).
        ax: Optional matplotlib Axes. If None, creates a new figure.
        show: Whether to call plt.show().
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    if model.n_inputs != 2:
        raise ValueError("Decision boundary plot requires a 2-input model")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    # Create mesh grid
    xx, yy = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    probs = model.predict_proba(grid).reshape(xx.shape)

    # Confidence heatmap
    cmap = mcolors.LinearSegmentedColormap.from_list("aipixel", ["#ff6b6b", "#f8f8f8", "#4ecdc4"])
    ax.contourf(xx, yy, probs, levels=50, cmap=cmap, alpha=0.7)
    ax.contour(xx, yy, probs, levels=[0.5], colors=["#2d3436"], linewidths=2)

    # Data points
    if X is not None and y is not None:
        X = np.asarray(X)
        y = np.asarray(y)
        colors = ["#ff6b6b" if label == 0 else "#4ecdc4" for label in y]
        ax.scatter(X[:, 0], X[:, 1], c=colors, edgecolors="#2d3436", linewidth=0.5, s=40, zorder=5)

    # Labels
    feature_names = meta.get("feature_names", ["Feature 1", "Feature 2"]) if meta else ["Feature 1", "Feature 2"]
    title = meta.get("name", "Decision Boundary") if meta else "Decision Boundary"

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Show pixel color in corner
    pixel = model.to_pixel()
    hex_color = "#{:02x}{:02x}{:02x}".format(*pixel[:3])
    ax.text(0.98, 0.02, f"Pixel: {hex_color}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="#636e72",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=hex_color, alpha=0.3))

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_training_loss(model, ax=None, show=True):
    """Plot the training loss curve."""
    import matplotlib.pyplot as plt

    if not model.training_history:
        raise ValueError("No training history. Train the model first.")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    ax.plot(model.training_history, color="#6c5ce7", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_pixel(model, ax=None, show=True):
    """Display the model's pixel color as a large swatch."""
    import matplotlib.pyplot as plt

    pixel = model.to_pixel()
    rgb_norm = tuple(c / 255.0 for c in pixel[:3])
    hex_color = "#{:02x}{:02x}{:02x}".format(*pixel[:3])

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))

    ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=rgb_norm, edgecolor="#2d3436", linewidth=2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"AI Pixel: RGB({pixel[0]}, {pixel[1]}, {pixel[2]})\n{hex_color}", fontsize=10)

    if show:
        plt.tight_layout()
        plt.show()

    return ax

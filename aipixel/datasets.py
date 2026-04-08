"""
Built-in example datasets for ai-pixel demos.

All features are pre-normalized to [0, 1].
All labels are binary (0 or 1).
"""

import numpy as np


def umbrella():
    """
    Should you bring an umbrella?

    Features: rain_chance (0-1), wind_speed (0-1 normalized from 0-50mph)
    Label: 1 = bring umbrella, 0 = leave it

    Linearly separable: high rain + high wind = umbrella.
    """
    rng = np.random.default_rng(42)
    n = 60

    # Umbrella cases: high rain chance and/or high wind
    X_yes = np.column_stack([
        rng.uniform(0.5, 1.0, n // 2),
        rng.uniform(0.3, 1.0, n // 2),
    ])
    # No umbrella: low rain, low-moderate wind
    X_no = np.column_stack([
        rng.uniform(0.0, 0.45, n // 2),
        rng.uniform(0.0, 0.6, n // 2),
    ])

    X = np.vstack([X_yes, X_no])
    y = np.array([1] * (n // 2) + [0] * (n // 2))

    idx = rng.permutation(n)
    return X[idx], y[idx], {
        "name": "Umbrella",
        "description": "Should you bring an umbrella?",
        "feature_names": ["Rain chance", "Wind speed"],
        "class_names": ["Leave it", "Bring umbrella"],
    }


def sunscreen():
    """
    Should you wear sunscreen?

    Features: uv_index (0-1 normalized from 0-11+), hours_outside (0-1 normalized from 0-8h)
    Label: 1 = wear sunscreen, 0 = skip it
    """
    rng = np.random.default_rng(43)
    n = 60

    X_yes = np.column_stack([
        rng.uniform(0.45, 1.0, n // 2),
        rng.uniform(0.3, 1.0, n // 2),
    ])
    X_no = np.column_stack([
        rng.uniform(0.0, 0.4, n // 2),
        rng.uniform(0.0, 0.5, n // 2),
    ])

    X = np.vstack([X_yes, X_no])
    y = np.array([1] * (n // 2) + [0] * (n // 2))

    idx = rng.permutation(n)
    return X[idx], y[idx], {
        "name": "Sunscreen",
        "description": "Should you wear sunscreen?",
        "feature_names": ["UV index", "Hours outside"],
        "class_names": ["Skip it", "Wear sunscreen"],
    }


def escalate():
    """
    Should a support ticket be escalated?

    Features: sentiment_score (0-1, lower=angrier), severity (0-1, higher=worse)
    Label: 1 = escalate, 0 = handle normally
    """
    rng = np.random.default_rng(44)
    n = 60

    # Escalate: low sentiment (angry) and/or high severity
    X_yes = np.column_stack([
        rng.uniform(0.0, 0.45, n // 2),
        rng.uniform(0.5, 1.0, n // 2),
    ])
    # Normal: high sentiment (happy), low severity
    X_no = np.column_stack([
        rng.uniform(0.5, 1.0, n // 2),
        rng.uniform(0.0, 0.5, n // 2),
    ])

    X = np.vstack([X_yes, X_no])
    y = np.array([1] * (n // 2) + [0] * (n // 2))

    idx = rng.permutation(n)
    return X[idx], y[idx], {
        "name": "Escalate Ticket",
        "description": "Should this support ticket be escalated?",
        "feature_names": ["Sentiment (low=angry)", "Severity"],
        "class_names": ["Handle normally", "Escalate"],
    }


def xor():
    """
    XOR dataset — intentionally NOT linearly separable.

    Use this to demonstrate the limits of a single-neuron classifier.
    The model will fail to achieve high accuracy on this data.
    """
    rng = np.random.default_rng(45)
    n_per = 20

    # Class 1: top-left and bottom-right
    X_1 = np.vstack([
        np.column_stack([rng.uniform(0.0, 0.4, n_per), rng.uniform(0.6, 1.0, n_per)]),
        np.column_stack([rng.uniform(0.6, 1.0, n_per), rng.uniform(0.0, 0.4, n_per)]),
    ])
    # Class 0: top-right and bottom-left
    X_0 = np.vstack([
        np.column_stack([rng.uniform(0.6, 1.0, n_per), rng.uniform(0.6, 1.0, n_per)]),
        np.column_stack([rng.uniform(0.0, 0.4, n_per), rng.uniform(0.0, 0.4, n_per)]),
    ])

    X = np.vstack([X_1, X_0])
    y = np.array([1] * (2 * n_per) + [0] * (2 * n_per))

    idx = rng.permutation(len(y))
    return X[idx], y[idx], {
        "name": "XOR (Unsolvable)",
        "description": "A pattern a single neuron cannot learn. Demonstrates the limits of linear classification.",
        "feature_names": ["X", "Y"],
        "class_names": ["Class 0", "Class 1"],
    }

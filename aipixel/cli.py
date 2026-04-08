"""
CLI interface for ai-pixel.

Usage:
    ai-pixel train data.csv --output model.png
    ai-pixel inspect model.png
    ai-pixel predict model.png --input "0.8,0.6"
"""

import argparse
import sys
import csv
import numpy as np
from pathlib import Path


def cmd_train(args):
    """Train a model from a CSV file and save as a pixel PNG."""
    from aipixel.model import PixelModel

    # Read CSV — auto-detect header by checking if first row is numeric
    X_list, y_list = [], []
    with open(args.data) as f:
        reader = csv.reader(f)
        first_row = next(reader, None)
        if first_row:
            try:
                values = [float(v) for v in first_row]
                X_list.append(values[:-1])
                y_list.append(values[-1])
            except ValueError:
                pass  # non-numeric first row = header, skip it
        for row in reader:
            if not row:
                continue
            values = [float(v) for v in row]
            X_list.append(values[:-1])
            y_list.append(values[-1])

    X = np.array(X_list)
    y = np.array(y_list)
    n_inputs = X.shape[1]

    if n_inputs > 3:
        print(f"Error: ai-pixel supports 1-3 features, got {n_inputs}", file=sys.stderr)
        sys.exit(1)

    model = PixelModel(n_inputs=n_inputs)
    model.train(X, y, epochs=args.epochs, lr=args.lr, verbose=True)

    output = Path(args.output)
    model.to_image(output)

    pixel = model.to_pixel()
    acc = model.accuracy(X, y)
    report = model.quantization_report()

    print(f"\nTrained model saved to {output}")
    print(f"  Pixel: RGB({pixel[0]}, {pixel[1]}, {pixel[2]})  {report['pixel_hex']}")
    print(f"  Accuracy (float):  {acc:.1%}")

    # Quantized accuracy
    q_model = PixelModel.from_pixel(*pixel)
    q_acc = q_model.accuracy(X, y)
    print(f"  Accuracy (pixel):  {q_acc:.1%}")
    print(f"  Max quant error:   {report['max_param_error']:.4f}")


def cmd_inspect(args):
    """Inspect a pixel PNG to see the encoded model."""
    from aipixel.model import PixelModel

    model = PixelModel.from_image(args.image)
    model.summary()


def cmd_predict(args):
    """Load a pixel model and run prediction on input."""
    from aipixel.model import PixelModel

    model = PixelModel.from_image(args.image)
    values = [float(v) for v in args.input.split(",")]
    if len(values) != model.n_inputs:
        print(f"Error: model expects {model.n_inputs} input(s), got {len(values)}", file=sys.stderr)
        sys.exit(1)
    X = np.array([values])
    prob = model.predict_proba(X)[0]
    pred = int(prob >= 0.5)
    print(f"Input:       {values}")
    print(f"Probability: {prob:.4f}")
    print(f"Prediction:  {pred}")


def main():
    parser = argparse.ArgumentParser(
        prog="ai-pixel",
        description="Train AI models that fit in a single pixel.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Train
    p_train = subparsers.add_parser("train", help="Train a model from CSV data")
    p_train.add_argument("data", help="CSV file (features + label in last column)")
    p_train.add_argument("--output", "-o", default="model.png", help="Output PNG path (default: model.png)")
    p_train.add_argument("--epochs", type=int, default=500, help="Training epochs (default: 500)")
    p_train.add_argument("--lr", type=float, default=0.5, help="Learning rate (default: 0.5)")

    # Inspect
    p_inspect = subparsers.add_parser("inspect", help="Inspect a pixel model PNG")
    p_inspect.add_argument("image", help="Path to 1x1 PNG")

    # Predict
    p_predict = subparsers.add_parser("predict", help="Run prediction with a pixel model")
    p_predict.add_argument("image", help="Path to 1x1 PNG")
    p_predict.add_argument("--input", "-i", required=True, help="Comma-separated input values (e.g., '0.8,0.6')")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "predict":
        cmd_predict(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

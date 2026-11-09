"""Train and run inference with the included synthetic sensor sample.

Run from data/notebook:
    python train_demo.py
    python train_demo.py --data data/raw/sensor_data.csv --target failure
"""
import argparse
from pathlib import Path

from src.configs.inference import run_inference
from src.configs.models.baseline_model import BaselineModel
from src.data.data_loader import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline model and run inference on a sensor CSV.")
    parser.add_argument("--data", default="data/raw/sensor_data.csv", help="CSV path, relative to data/notebook or absolute")
    parser.add_argument("--target", default="failure", help="Target column to train against")
    parser.add_argument("--model-dir", default="models", help="Directory where the trained model is saved")
    args = parser.parse_args()

    data_path = str(Path(args.data))
    frame = DataLoader(data_path).load()
    if args.target not in frame.columns:
        raise ValueError(f"Target column {args.target!r} not found. Available columns: {list(frame.columns)}")

    # Train on raw numeric sensor columns. Keep inference preprocessing disabled
    # here so the demo applies the same feature representation at both stages.
    model = BaselineModel(target_col=args.target, models_dir=args.model_dir)
    model.train(frame)
    metrics = model.evaluate()
    model_path = model.save_model(filename=f"baseline_{args.target}.pkl")

    result = run_inference(
        data_path,
        model_path=model_path,
        target_col=args.target,
        apply_preprocessing=False,
        apply_feature_engineering=False,
        display=False,
    )

    print("Baseline model saved to:", model_path)
    print("Inference rows:", len(result))
    print("Evaluation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()

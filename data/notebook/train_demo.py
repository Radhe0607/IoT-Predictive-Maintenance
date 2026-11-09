"""Train and run inference with the included synthetic sensor sample.

Run from data/notebook:
    python train_demo.py
"""
from src.configs.inference import run_inference
from src.configs.models.baseline_model import BaselineModel
from src.data.data_loader import DataLoader


def main() -> None:
    sample_path = "data/raw/sensor_data.csv"
    frame = DataLoader(sample_path).load()

    # Train on raw numeric sensor columns. Keep inference preprocessing disabled
    # here so the demo applies the same feature representation at both stages.
    model = BaselineModel(target_col="failure")
    model.train(frame)
    metrics = model.evaluate()
    model_path = model.save_model()

    result = run_inference(
        sample_path,
        model_path=model_path,
        target_col="failure",
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

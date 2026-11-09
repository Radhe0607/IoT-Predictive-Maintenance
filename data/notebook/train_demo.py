"""Train a reproducible baseline model from the included sample sensor data.

Run from data/notebook:
    python train_demo.py
"""
from src.configs.models.baseline_model import BaselineModel
from src.data.data_loader import DataLoader


def main() -> None:
    frame = DataLoader("data/raw/sensor_data.csv").load()
    model = BaselineModel(target_col="failure")
    model.train(frame)
    metrics = model.evaluate()
    model_path = model.save_model()

    print("Baseline model saved to:", model_path)
    print("Evaluation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()

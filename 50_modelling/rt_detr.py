# train_rtdetr_gpu.py
# Usage example:
# python train_rtdetr_gpu.py --data_dir path/to/data.yaml --project_dir path/to/runs --run_name my_run --epochs 100 --img_size 720

import argparse
import os
from ultralytics import RTDETR
import torch

def train_rtdetr(data_yaml, project_dir, run_name, epochs=100, img_size=720, device=None):
    # Автоматически выбрать GPU, если есть
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(os.path.join(project_dir, run_name), exist_ok=True)

    model = RTDETR("rtdetr-l.pt")

    print(f"Starting RTDETR training with dataset {data_yaml} on device: {device}")
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        project=project_dir,
        name=run_name,
        exist_ok=True,
        device=device
    )
    print(f"Training completed. Logs and checkpoints saved under {os.path.join(project_dir, run_name)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RTDETR model with GPU if available.")

    parser.add_argument("--data_dir", required=True, help="Path to dataset YAML file")
    parser.add_argument("--project_dir", required=True, help="Base directory for logs and checkpoints")
    parser.add_argument("--run_name", required=True, help="Name of this training run (subfolder)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--img_size", type=int, default=720, help="Input image size")
    parser.add_argument("--device", type=str, default=None, help="Device to train on (cpu, cuda, or None for auto)")

    args = parser.parse_args()

    train_rtdetr(
        data_yaml=args.data_dir,
        project_dir=args.project_dir,
        run_name=args.run_name,
        epochs=args.epochs,
        img_size=args.img_size,
        device=args.device
    )

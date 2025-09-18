# scripts/train_tuned_v8s.py
from ultralytics import YOLO
from pathlib import Path
import torch

def train_tuned_v8s():
    # ---------- Dataset ----------
    dataset_yaml = Path(__file__).parent.parent / "dataset" / "data.yaml"

    # ---------- Model ----------
    model = YOLO("yolov8s.pt")  # pretrained YOLOv8s

    # ---------- Hyperparameters ----------
    epochs = 50
    batch_size = 16        # increase if GPU allows
    img_size = 640
    lr0 = 0.003            # base learning rate
    momentum = 0.937
    weight_decay = 0.0005

    # ---------- Data Augmentation ----------
    augment = True  # rotations, flips, brightness/contrast etc.

    # ---------- Training ----------
    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        lr0=lr0,
        momentum=momentum,
        weight_decay=weight_decay,
        augment=augment,
        project="runs",
        name="tuned_v8s",
        exist_ok=True
    )

    # ---------- Confirmation ----------
    print("Training started on GPU:", torch.cuda.is_available())

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    train_tuned_v8s()

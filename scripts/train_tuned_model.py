from ultralytics import YOLO
from pathlib import Path
import torch

def train_tuned_model():

    dataset_yaml = Path(__file__).parent.parent / "dataset" / "data.yaml"


    model = YOLO("yolov8n.pt")  # pretrained YOLOv8n

    epochs = 50
    batch_size = 16
    img_size = 640
    lr0 = 0.003   
    momentum = 0.937
    weight_decay = 0.0005

    augment = True  


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
        name="tuned_v8n",
        exist_ok=True
    )


    print("Training started on GPU:", torch.cuda.is_available())

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    train_tuned_model()

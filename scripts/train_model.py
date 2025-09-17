from ultralytics import YOLO
from pathlib import Path

def train_model():
    # Load pretrained YOLOv8n
    model = YOLO("yolov8n.pt")

    # Train tuned model
    model.train(
        data = Path(__file__).parent.parent / "dataset" / "data.yaml", 
        epochs=50,
        batch=8,
        imgsz=640,
        lr0=0.005,
        momentum=0.9,
        weight_decay=0.0005,
        augment=True,
        project="runs",         # relative to script folder
        name="phase1_v8n",
        exist_ok=True
    )

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    train_model()

import argparse
import os
from ultralytics import YOLO
import torch

def main(args):
    # check cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    if device == 'cpu':
        print("WARNING: Training on CPU will be very slow. Prefer GPU.")

    # model
    model = YOLO(args.model)  
    out_project = args.project or 'runs/quick_sanity'
    name = args.name or f'sanity_{os.path.basename(args.model).split(".")[0]}'
    print(f"Starting quick training: model={args.model}, data={args.data}, epochs={args.epochs}, imgsz={args.img}, batch={args.batch}")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img,
        batch=args.batch,
        device=0 if device=='cuda' else 'cpu',
        project=out_project,
        name=name,
        optimizer=args.optimizer,
        lr0=args.lr,
        augment=True,
        workers=4,
        patience=10
    )
    print("Training finished. Best weights (if saved) will be in the runs/ directory.")
    print("Example best path:", os.path.join(out_project, name, 'weights', 'best.pt'))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='dataset/data.yaml', help='path to data.yaml')
    p.add_argument('--model', default='yolov8n.pt', help='yolo model checkpoint to start from')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--img', type=int, default=640)
    p.add_argument('--project', default=None)
    p.add_argument('--name', default=None)
    p.add_argument('--optimizer', default='SGD')
    p.add_argument('--lr', type=float, default=0.01)
    args = p.parse_args()
    main(args)

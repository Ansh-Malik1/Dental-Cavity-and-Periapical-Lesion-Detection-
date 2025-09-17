import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_training_curves(run_dir):
    results_path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(results_path):
        print(f"results.csv not found in {run_dir}")
        return

    print(f"Found results.csv at: {results_path}")

    # Load CSV
    df = pd.read_csv(results_path)

    # Plot Loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train/box_loss"], label="Train Box Loss")
    plt.plot(df["epoch"], df["train/cls_loss"], label="Train Cls Loss")
    plt.plot(df["epoch"], df["train/dfl_loss"], label="Train DFL Loss")
    plt.plot(df["epoch"], df["val/box_loss"], label="Val Box Loss")
    plt.plot(df["epoch"], df["val/cls_loss"], label="Val Cls Loss")
    plt.plot(df["epoch"], df["val/dfl_loss"], label="Val DFL Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    loss_path = os.path.join(run_dir, "loss_curves.png")
    plt.savefig(loss_path)
    plt.close()
    print(f"Loss curve saved at: {loss_path}")

    # Plot Metrics curves
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["metrics/precision(B)"], label="Precision")
    plt.plot(df["epoch"], df["metrics/recall(B)"], label="Recall")
    plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@50")
    plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@50-95")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title("Evaluation Metrics")
    plt.legend()
    plt.grid(True)
    metrics_path = os.path.join(run_dir, "metrics_curves.png")
    plt.savefig(metrics_path)
    plt.close()
    print(f"Metrics curve saved at: {metrics_path}")


if __name__ == "__main__":
    # Scripts folder se ek level upar jaa kar runs folder target kar raha hai
    base_dir = os.path.dirname(os.path.dirname(__file__))
    run_dir = os.path.join(base_dir, "runs", "phase1_v8n")
    plot_training_curves(run_dir)

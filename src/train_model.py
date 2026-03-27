import csv
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from config import settings
from model.gesture_net import GestureNet
from utils.logger import logger


def load_label_names() -> list[str]:
    label_path = "model/keypoint_classifier/keypoint_classifier_label.csv"
    try:
        with open(label_path, encoding="utf-8-sig") as f:
            return [row[0] for row in csv.reader(f)]
    except Exception:
        return []


def main() -> None:
    dataset_path = "model/keypoint_classifier/keypoint.csv"
    model_save_path = "model/keypoint_classifier/keypoint_classifier.pt"
    input_size = 42  # 21 landmarks * 2 (x, y)

    if not os.path.exists(dataset_path):
        logger.error(f"Error: Dataset not found at {dataset_path}")
        return

    logger.info("Loading dataset...")
    df = pd.read_csv(dataset_path, header=None)

    X = df.iloc[:, 1:].values.astype("float32")
    y = df.iloc[:, 0].values.astype("int64")

    label_names = load_label_names()
    num_classes = len(np.unique(y))
    logger.info(f"Found {num_classes} classes: {np.unique(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_test = torch.from_numpy(X_test).to(device)
    y_test = torch.from_numpy(y_test).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(
        train_dataset, batch_size=settings.BATCH_SIZE, shuffle=True
    )

    model = GestureNet(
        input_size=input_size, hidden_size=96, num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    logger.info("Starting training...")

    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            # Gaussian noise augmentation — makes model robust to landmark jitter
            X_batch = X_batch + torch.randn_like(X_batch) * 0.01

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs, 1)
        accuracy = (predicted == y_test).sum().item() / len(y_test)
        logger.info(f"Overall accuracy: {accuracy * 100:.2f}%")

        predicted_np = predicted.cpu().numpy()
        y_test_np = y_test.cpu().numpy()
        logger.info("Per-class accuracy:")
        for class_id in sorted(np.unique(y_test_np)):
            mask = y_test_np == class_id
            class_acc = (predicted_np[mask] == class_id).sum() / mask.sum()
            name = (
                label_names[class_id] if class_id < len(label_names) else str(class_id)
            )
            logger.info(f"  [{class_id}] {name:<14} {class_acc * 100:.1f}%")

    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Model saved to {model_save_path}")

    onnx_save_path = model_save_path.replace(".pt", ".onnx")
    dummy_input = torch.randn(1, input_size, requires_grad=True).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_save_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    logger.info(f"ONNX model saved to {onnx_save_path}")


if __name__ == "__main__":
    main()
